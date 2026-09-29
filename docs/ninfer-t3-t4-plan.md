# NInfer follow-ups T3 (INT8 KV + decode attention) and T4 (launch fusion) - analysis and plan

Status: 2026-09-29 (TASKS #153). Companion to `docs/mmsmt-implementation.md` (T1, F4/T2) and
`research/ninfer-v100-survey-2026-09-27.md` (sections 1.4-1.5, 4). Both items are decode-SPEED
items; neither changes output quality except T3's KV quantisation, which lands with a KL gate.

## 1. T3: what NInfer does and what the fork has

NInfer-v100 (survey 1.4): KV stored as per-token, per-64-group absmax int8 with an fp16 scale
(528 B per token per kv-head at head dim 256 vs 1024 B bf16), a normalised Sylvester-Hadamard
rotation of K at append and of Q at query time (V unrotated) to flatten outliers before the
quantisation, and (huangserva fork) a decode-attention kernel that feeds the int8 K/V to the
Volta `m8n8k4` fp16 MMA after in-register dequant, 8 warps / Bc = 64 (1.29 ms per 393 MB call,
~305 GB/s). Reported: 186K context 36 -> 56 tok/s single GPU; no int8-vs-bf16 perplexity numbers
were published (survey open question).

The fork (ggml-cuda/fattn.cu, `ggml_cuda_get_best_fattn_kernel`, Volta branch): `q8_0` K/V is a
supported cache type (8.5 bits per value, group 32, fp16 scale - within 3 % of NInfer's 8.25 bits),
no rotation. Kernel choice on Volta for Qwen3.8-27B (head 256, GQA 24/4 -> gqa_ratio_eff 2):

| step | Q rows x gqa_eff | kernel | tensor cores |
|---|---|---|---|
| plain decode (1 token) | 2 | VEC (fattn-vec, SIMT, dequantises q8_0 per element) | no |
| MTP verify (4-5 rows) | 8-10 | TILE ("tensor cores only pay for large matrices") | no |
| prefill | >16 | MMA_F16 | yes |

So at long context BOTH the decode step and the verify step run SIMT attention on Volta, and
q8_0 halves the bytes but not the per-element dequant work. That is exactly NInfer's T3 claim:
their int8 KV attention is an MMA kernel. The Qwen3.8-27B KV is small (16 of 65 layers keep KV:
16 x 2 x 4 x 256 x 2 B = 64 KB per token f16, 34 KB q8_0): at 64K tokens the step reads 4.2 GB
(f16) or 2.2 GB (q8_0) of KV next to 16.3 GB of weights - attention is 12-20 % of the decode
step's bytes, more if the SIMT kernels run below the weight-stream rate.

## 2. T3 phase 0 (measure before building)

`153-p0/t3-phase0.sh` (queued after the T2 gate): llama-bench on the UD-Q4_K_XL, FA on, `-ctk/-ctv`
f16 vs q8_0, `-n 32 -d 0,16384,65536,131072` (decode t/s at KV depth), then
`153-p0/t3-phase0b.sh`: `-p 5 -n 0` at the same depths (the verify-width step). Read:

- t/s(depth 0) - t/s(depth d) per KV type = the attention cost at depth d; if q8_0 at 64K costs
  the same ms as f16 despite half the bytes, the SIMT kernel is compute-bound and an MMA kernel is
  the lever (T3 kernel); if q8_0 already tracks its byte count, the lever is bytes only (KV type).
- The verify-width cost at depth vs width 1: the TILE kernel's share of an MTP round at long ctx.
- Decision rule: build the T3 kernel only if attention exceeds ~25 % of the decode step at the
  user's serving contexts (64K-128K) AND the q8_0 arm shows the compute-bound signature.

## 3. T3: why the Volta decode attention is 4x off the byte rate, and the design

Facts from the fork's kernels (mapped 2026-09-29 02:00; anchors in ggml/src/ggml-cuda):

- Selection (fattn.cu Volta branch): head 256, GQA 6 -> `gqa_ratio_eff` 2; width 1 -> VEC; widths
  4-5 (MTP verify) -> TILE; MMA_F16 only above 16 rows, and its Volta path (mma.cuh m8n8k4)
  refuses `ncols1*ncols2 < 32` (fattn-mma-f16.cuh ~1752): the tensor-core kernel cannot serve
  decode or verify widths today. (q8_0, q8_0, 256) VEC instances exist.
- VEC (fattn-vec.cuh) at width 1: 128 threads, `__launch_bounds__(128, 1)`, ONE Q head per block
  (`ncols2 = 1`, no GQA packing): the 6 Q heads of a KV head run as 6 block sets that each re-read
  and re-convert the same K/V (6x the KV bytes through L2/L1: ~25 GB at 64k). The half2 dot path is
  gated on `V_DOT2_F32_F16_AVAILABLE` (AMD only), so on CUDA every element goes f16->f32 (F2F,
  quarter rate on sm70) before an fp32 FMA; 8 threads share a K row and reduce with shuffles row
  by row; split-KV = `parallel_blocks` sized to one wave (~13 at 64k) with ~16 warps per SM.
  q8_0 adds 2-byte loads (34-byte blocks), a 32-lane reduction per K row and I2F per V element -
  the reason q8_0 does not beat f16 here (docs/perf-tuning-v100.md "FA compute-bound").
- TILE (widths 4-5) needs f16 K/V: with a q8_0 cache `launch_fattn` converts the WHOLE cache to a
  temporary f16 buffer on every call (1.5x the bytes of one read), then runs ncols2 = 2 half2
  tiles with 3x redundancy per KV head and a half-empty second tile at width 5.

Design (the NInfer route, lever c): **a Volta m8n8k4 decode/verify attention kernel with
in-register KV dequant** - `fattn-mma-volta-small.cuh`, selected on cc 7.0 for
`Q rows x gqa_eff <= 32` and K/V in {f16, q8_0}:

1. Tile mapping: M = the KV head's Q heads x query rows (6 x 1..5 = 6..30, padded to a multiple
   of 8 = one, two or four m8n8k4 A tiles), N = KV positions in steps of 64 (8 quadpair tiles),
   K = head dim 256 in 64 slices of 4. QK^T: A = Q (fp16, staged once in shared), B = K rows
   dequantised in registers - q8_0 through the T1 machinery (`mmsmt_bytes_lo/hi` + `HSUB2` with
   `MMSMT_BIAS_1152`, block scale folded into the fp32 partial per 32-wide block), f16 through
   plain 16-byte loads. Every K element is loaded and converted ONCE per KV head (GQA packing in
   M), which alone removes the 6x redundancy.
2. Online softmax per M row over the 64-position step (fp32 running max/sum in registers, the
   fattn-mma-f16 pattern), P kept as fp16 A fragments for the PV mma: B = V rows dequantised
   the same way, accumulate O[M x 256] in fp32 (8 accumulator tiles of 8x8 per warp pair).
3. Split-KV across CTAs as today (`parallel_blocks` + `flash_attn_combine_results`), one CTA
   = 4 warps that split the 64-position step's K slices; 2-4 CTAs per SM (registers: 8 O tiles
   x 8 floats + fragments ~110 per thread, launch bounds 128/2).
4. Mask handling, ALiBi and logit softcap as in fattn-vec (the mask row per Q row; the verify
   batch's causal mask is what makes widths 4-5 differ from width 1).
5. Gates: test-backend-ops FLASH_ATTN_EXT cases on the V100 (all D/ncols/type combinations the
   selector routes), KL ub 8/4/3 and the 27B exact-answer probes at 64k prompts (the long-context
   correctness gate), the depth sweeps of 5.2/5.3 as the speed gate, MTP serve at 64k depth.

Expected: near the byte rate for f16 (4.8 ms at 64k vs 20.5 today) and q8_0 (~2.5 ms): the step
at 64k from 48.5 to ~33 ms (+47 %), at 128k from 70 to ~40 (+75 %); the verify step gains the same
absolute ms and stops paying the whole-cache conversion with q8_0 KV. Effort 1-2 weeks of kernel
work with the T1 harness pattern (oracle -> comparator -> bench -> KL -> serve).

Cheaper first steps, measured before the kernel (phase 0b, `t3-phase0b.sh`): (a) zero-code:
`GGML_CUDA_FA_NO_MMA=1` routes f16 width 1 to TILE (ncols2 = 2, half2 math: 3x instead of 6x
redundancy) - a Volta selection rule if it wins at width 1 without hurting prefill; (b) a VEC
variant with GQA packing + a CUDA half2 dot path (2-3x, ~1 week) if the mma kernel slips.

The Hadamard-rotated int8 g64 KV type (NInfer's format) stays excluded: it is a quality item for
int8 KV, not a speed item, and llama.cpp's q8_0 already carries the bytes.

## 4. T4: launch fusion

NInfer's claim (survey 1.5, PR #11): 1255 -> 726 launches per step through fused projections
([Q|K|gate|V], GDN [q|k|v|z] + conv + SiLU), fused norm+rope, and one CUDA graph per round;
launch count alone was worth 66.1 -> 38.9 ms together with the fusions on a graph-less path.
llama.cpp already captures CUDA graphs for decode on Volta (`USE_GRAPHS = 1` in the fork's
system_info; `GGML_CUDA_DISABLE_GRAPHS=1` opts out), so launch overhead is mostly hidden; what
remains is the memory-traffic side of fusion (fewer passes over activations) and the fixed cost
per kernel inside the graph.

Census method (`scratchpad t4-census.sh`, CPU build, `GGML_SCHED_DEBUG=2`, one decode step of
Qwen3.8-27B): histogram of graph ops per step = the upper bound of kernel launches; the CUDA
backend fuses some at execution (this fork: RMS_NORM+MUL, ADD chains where the July ggml did).
Result and the fusion candidates: section 5 (filled from the census). Upstream master has gained
graph-level fusions since the fork's last merge (52be8d1d6, 2026-07-30); the #67 upstream merge
is the cheapest way to take them, so T4's rule is: census now, list what upstream fuses that we do
not, re-census after the merge, and only then hand-fuse what is left and hot (GDN kernels are the
likely residue - the sequential GDN scan is already one kernel here).

## 5. Results

### 5.1 T4 census (2026-09-29 01:45, CPU build of 39fe2889a + tree, `GGML_SCHED_DEBUG=2 -v`, one decode step, Qwen3.8-27B UD-Q4_K_XL)

2293 compute nodes per decode step (views excluded; the reserve graph reports 3655 nodes incl.
views). Op histogram: MUL_MAT 497, MUL 321, RMS_NORM 209, GET_ROWS 194, CPY 192, ADD 176,
SCALE 96, SILU 96, L2_NORM 96, SIGMOID 64, SWIGLU 64, CONCAT 48, SSM_CONV 48, SOFTPLUS 48,
GATED_DELTA_NET 48, ROPE 32, SET_ROWS 32, FLASH_ATTN_EXT 16, CONT 16. Per layer: ~38 nodes
for each of the 48 GDN layers, ~25 for each of the 16 attention layers; SWIGLU is already the
fused gate+up form.

Where the launches can go (each item = nodes per step it removes, before the CUDA backend's own
fusions, which in this July-2026 ggml cover RMS_NORM+MUL and some ADD chains):

| candidate | nodes/step | who has it |
|---|---|---|
| RMS_NORM + MUL (weight) [+ ADD residual] | ~209 (+~100) | upstream fuses norm+mul(+add) |
| GDN elementwise chain per layer (SCALE, SILU, SOFTPLUS, SIGMOID, MUL, L2_NORM around the delta rule: ~10 nodes) | ~350-400 | hand fusion (fork-local) or upstream's newer GDN graph |
| GDN state GET_ROWS x4 + CPY x4 per layer (recurrent state read/write) | ~380 | upstream's SSM state handling landed several reductions after 07-30 |
| ROPE + SET_ROWS (K/V cache write) | 32 | upstream fuses rope+set_rows |
| fused Q|K|V(|gate) projections (one MUL_MAT per layer instead of 3-4) | ~130 | NInfer's T4; llama.cpp only where the GGUF ships a fused tensor (not qwen35) |

Bound on the prize: a T=1 step is 30.6 ms of which ~26 ms is the weight stream (16.3 GiB at
~620 GiB/s); the rest, ~4-5 ms, is attention at 8k plus ~2000 small kernels' fixed cost inside the
CUDA graph (Volta ~2-3 us each). Removing ~1000 nodes recovers at most ~2-3 ms per step (~7-9 %
at width 1, the same absolute ms per verify step). Decision (section 4 rule): take upstream's
fusions through the #67 merge first (norm+mul+add, rope+set_rows, the SSM/GDN state path),
re-census, then hand-fuse the GDN elementwise chain if it is still ~350 nodes. Not built here.

### 5.2 T3 phase 0: decode t/s vs KV depth (2026-09-29 01:43-01:52, V5 bins devbins-t5, `-b/-ub 2048`, FA on, tg32 at depth)

| KV | d 0 | d 16384 | d 65536 | d 131072 |
|---|---|---|---|---|
| f16 | 35.72 (28.0 ms) | 30.32 (33.0 ms) | 20.61 (48.5 ms) | 14.26 (70.1 ms) |
| q8_0 (VEC) | 34.06 (29.4 ms) | 27.59 (36.2 ms) | 21.65 (46.2 ms) | 15.43 (64.8 ms) |
| f16, `GGML_CUDA_FA_NO_MMA=1` (TILE at width 1) | 33.97 (29.4 ms) | - | **24.43 (40.9 ms)** | - |

f16 reading: the attention cost per token is +5.0 / +20.5 / +42.1 ms at 16k / 64k / 128k. The KV
bytes read per token (16 layers x 2 x 4 heads x 256 x 2 B = 64 KB per position) are 1.07 / 4.3 /
8.6 GB, i.e. 1.2 / 4.8 / 9.5 ms at the 900 GB/s HBM rate: the Volta VEC decode-attention kernel
(head 256, gqa_eff 2, width 1) runs at ~205-215 GB/s effective, four times below the byte
roofline, and attention is 42 % of the step at 64k and 60 % at 128k. The FLOP side is not it
either (4.2 GFLOP at 64k = 0.3 ms at SIMT fp32 rate). This is NInfer's T3 claim reproduced on
our side: at the user's serving contexts (64k-262k) the attention kernel, not the weights, bounds
the step. Prize if a kernel reaches ~80 % of the byte rate: 64k 48.5 -> ~34 ms (+43 %), 128k
70 -> ~40 ms (+75 %). The q8_0 row tells whether halving the bytes helps (byte-bound) or not
(compute/occupancy-bound), which decides between "kernel rewrite" and "kernel + KV type".

### 5.3 T3 phase 0b: verify-width step (5 rows, `-p 5`) vs KV depth (2026-09-29 02:04-02:22, same bins)

| KV | d 0 | d 16384 | d 65536 | d 131072 |
|---|---|---|---|---|
| f16 (TILE kernel) | 97.7 t/s (51.2 ms/step) | 81.0 (61.7) | 63.9 (78.2) | 47.6 (105.0) |
| q8_0 (TILE after a whole-cache f16 conversion) | 95.9 (52.1) | 68.2 (73.3) | 57.8 (86.5) | 42.2 (118.5) |

f16 reading: the width-5 verify step pays +10.5 / +27.0 / +53.8 ms of attention at 16k / 64k /
128k, 1.3x the width-1 VEC cost at the same depth (the TILE kernel's 3x GQA redundancy vs VEC's
6x is offset by 5 query rows and the half-empty second tile). In an MTP round at 64k the verify
step alone grows from ~56 ms to ~83 ms; with the T3 kernel at the byte rate the attention share
would be ~5 ms, i.e. the round returns to within ~10 % of its depth-0 cost.

q8_0 reading (width 5): +21 / +34 / +66 ms at 16k / 64k / 128k, i.e. SLOWER than f16 with half the
bytes - the TILE path converts the entire q8_0 cache to a temporary f16 buffer on every call (1.5x
the bytes of one pass plus the conversion) and then runs the same 3x-redundant tiles. This is the
compute-bound signature of section 2's decision rule: on Volta the KV type cannot buy long-context
speed until the kernel reads quantised K/V directly - the T3 kernel (section 3) is the item, and
q8_0 KV should be treated as a memory-capacity option only until it lands.

### 5.4 Width-1 q8_0 row and the zero-code lever (2026-09-29 02:23-02:40)

q8_0 on the VEC kernel: attention +6.9 / +16.8 / +35.4 ms at 16k / 64k / 128k vs f16's +5.0 /
+20.5 / +42.1 - half the bytes buy 16-18 % at long depth (2.15 GB in 16.8 ms = 128 GB/s
effective, still ~7x off the byte rate), and cost 5 % at depth 0. `GGML_CUDA_FA_NO_MMA=1`, which
routes width 1 f16 to the TILE kernel (half2 math, ncols2 = 2), reads 24.4 t/s at 64k against
VEC's 20.6 (+19 %, attention +11.5 ms instead of +20.5) and 5 % below VEC at depth 0; the width-5
step is unchanged (TILE either way). **Lever a adopted as a Volta selection rule** (fattn.cu, the
`volta_mma_available` branch): f16 K/V, `gqa_opt_applies`, Q rows x gqa_eff <= 2 and
`K->ne[1] >= 16384` -> TILE; quantised K/V keeps VEC. Gate `153-p0/t3-gate.sh` (test-backend-ops
FLASH_ATTN_EXT, the f16 sweep + pp2048 on the new vs old bins, KL at 32k context TILE vs VEC).
For the user's 200k-262k q8_0-KV serves the rule does not apply: that is the T3 kernel's job.

### 5.5 Lever a gate (2026-09-29 02:45-02:59, `153-p0/t3-gate.sh`, rule build devbins-t3a vs VEC build devbins-t5)

- test-backend-ops FLASH_ATTN_EXT on the V100: 2884/2884 passed (the rule only reorders existing,
  tested kernels).
- KL at 32k context (wiki.test, 1 chunk of 32768, `-b/-ub 2048`), rule vs VEC: mean KLD 0.000000,
  99.9 % KLD 5e-5, top-1 100.0 %, PPL ratio 1.0003 - exact class.
- Speed (f16 KV, `-p 2048 -n 32`, two reps): rule build pp2048 981 / 706 / 356 t/s and tg32 35.5 /
  32.9 / 25.7 t/s at depth 0 / 16k / 64k; the VEC build, run second on the already-hot card, read
  800 / 573 / 351 and 30.6 / 25.9 / 19.5 (its depth-0 decode, where the rule is inactive, is 16 %
  below the first arm = the drift of the night). Drift-corrected: +9 % at 16k, +19..25 % at 64k
  (the phase-0b NO_MMA arm, measured in the same thermal regime as its VEC counterpart, read +19 %).
  Prefill is unchanged (the MMA kernel stays selected above 16 rows).

Verdict: adopted (fattn.cu, uncommitted); ships with the next coordinator image. It moves only f16
K/V serves at >= 16k depth; quantised-KV serves (the user's 200k-262k configs) wait for the T3
kernel (section 3), whose phase-0 prize stands at +47 % (64k) / +75 % (128k).
