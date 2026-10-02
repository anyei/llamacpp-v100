# NInfer follow-ups T3 (INT8 KV + decode attention) and T4 (launch fusion) - analysis and plan

Status: **T3 SHIPPED, default on** - lever a (the Volta TILE rule, 5.5) and the small-width kernel increments 1-2
(sections 6-7, `fattn-mma-volta-small.cuh`, `GGML_CUDA_FA_NO_VOLTA_SMALL=1` disables) are in every image since
`a9885783c-t3k3` (2026-09-29). Next T3 lever (not started): overlap the width-5 attention tiles with the memory stream
(7.3). T4: census + plan only (sections 4, 5.1). Originally 2026-09-29 (TASKS #153). Companion to `docs/mmsmt-implementation.md` (T1, F4/T2) and
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

## 6. T3 kernel: build log (2026-09-29, TASKS #153)

### 6.1 Increment 1 as built: `ggml/src/ggml-cuda/fattn-mma-volta-small.cuh` (all widths <= 8 rows in one kernel)

Fragment layouts were pinned first with a 6-test probe on the V100 (`llama.cpp-work/153-p0/t3probe/probe.cu`, run inside
the a9885783c image): for `mma.sync.m8n8k4` lane t = (lane&3) + 4*(lane>>4) holds A row t (a0 = k 0..1, a1 = k 2..3);
B `.col`: lane t holds B[k 0..3][n = t]; B `.row`: lane holds B[k = lane&3][n = 4*(lane>>4) + 0..3]; the fp32 D register i
sits at row 4*(lane>>4) + (lane&1) + (i&2), column (lane&2) + (i&5). These match the tested `mma.cuh` Volta tiles
(un-permuted) and `mmsmt.cu`; the design's "flip" is that the upstream Volta MMA kernel stacks 32 query columns in the
A tile (hence ncols >= 32) while this kernel mirrors the 8-row Q tile across the four quadpairs and gives each
quadpair its own K positions.

Geometry (one template `<D, NT, type_K, type_V, softcap>`, D in {64, 128, 256}, NT = 1..4 m8n8k4 M tiles, K/V in
{f16 (f32 converted), q8_0}): a block owns <= 8 query rows x `pack` heads of one KV head (`pack` = the largest head
tiling with rows x pack <= 32, derived in-kernel from gridDim.z; the 27B at width 1 packs its 6 heads into one tile,
width 5 x 6 = 30 rows in four tiles), 4 warps, CTA step = 128 KV positions:

- phase 1: warp w streams K rows 32w..32w+31, one row per lane (f16: 16-byte loads; q8_0: 17 words per block pair,
  `byte_perm` into the fp16 mantissa, minus 1152, times the block scale in fp16 - the T1 reader), the Q tile is fp16
  in shared memory (`scale` folded, 16-byte padded rows) and mirrored across quadpairs; S in fp32 per quadpair
  (8 regs per tile); softcap, mask, ALiBi on the D layout; row max over the warp with three xor shuffles, CTA-wide
  through shared memory (one `__syncthreads`); P = exp(S - max) to fp16, the D -> A remap costs two xor-2 shuffles,
  and P lands in shared memory as A fragments (row stride 144 halves: conflict-free 16-byte stores).
- phase 2 (after the second `__syncthreads`): warp w owns dims [w*D/4, (w+1)*D/4); quadpair = (M tile, position slice)
  with NQ_T x NQ_P = 4; each lane streams V[pos][its D/8 dims] as B `.row` fragments (the dims of the 8-wide n-tiles are
  permuted so a lane's fragments are one contiguous run: 64 bytes of f16 or exactly one q8_0 block per position at
  D = 256), O += P V in fp32 (64 accumulator registers per lane at every D/NT combination).
- epilogue: position slices summed with xor-4/8 shuffles, row sums reduced through shared memory, sinks as one extra
  logit per row, O staged in shared memory and written normalised (gridDim.y == 1) or raw + `dst_meta` for
  `launch_fattn`'s split-KV combine (`parallel_blocks`, unchanged).

Selection (fattn.cu, Volta branch, before the tile/vec rules): Q rows <= 8, head 64/128/256 with DV == DKQ,
KV length % 256 == 0, K/V both f16/f32 (16-byte row strides) or both q8_0, softcap only at head 128/256; the
kill switch `GGML_CUDA_FA_NO_VOLTA_SMALL=1` (docs/env-gates.md 8b, wizard catalog 134) restores the previous routing
in the same binary. Instances: `template-instances/fattn-mma-volta-small-instance-{f16-f16,q8_0-q8_0}-d{64,128,256}.cu`
(generator updated). Compiler report (sm_70): 254-255 registers, 0 stack, 0 spills for every D = 256 instance ->
`__launch_bounds__(128, 2)` = 2 CTAs / 8 warps per SM; dynamic shared memory 4.9 KB (NT 1, D 64) .. 34.4 KB (NT 4, D 256).

Oracle additions (tests/test-backend-ops.cpp): the upstream FLASH_ATTN_EXT matrix never reaches 3-4 M tiles or q8_0 at
head 256, so the 27B geometry (256, GQA 6, kv 1024, nb 1/2/3/4/5/8, f16 + q8_0) and a head-128 GQA-4 twin were added,
plus sinks / softcap / ALiBi variants at width 5.

First oracle (devbins-t3k1, before the new cases): 1492 passed / 156 failed - every failure had > 1 packed row
(nb 3, or nb 1 with 4 packed heads), every 1-row case passed: the output staging read the second D row's
accumulators with the wrong register offset (4h instead of 2h). Fixed in t3k1c together with a q8_0 V reader
over-read (one word past the last block of a row when the quants are word-aligned).

Second oracle (t3k1c): 1985+ passed, 0 failed, then a HANG (GPU at 100 % for 30 min) on the first NT = 3 shape
(head 128, GQA 12, 3 query rows -> 18 rows in three tiles): quadpair 3 has no tile there and skipped the phase-2 loop
while its warp-mates issued `mma.sync` - a warp-level `mma.sync.aligned` needs all 32 lanes in convergence, so the
warp deadlocked. Fixed in t3k1d: the spare quadpair computes a discarded copy of tile 0 (same instruction stream);
the oracle runs now use a pseudo-TTY so the log is line-buffered (the block-buffered log had hidden the real case).
Trap for the ledger: never put `mma.sync` or `__shfl_sync` under a lane-dependent branch on Volta.

### 6.2 Increment 1 gate (devbins-t3k1d = a9885783c tree + the kernel, `153-p0/t3k-gate.sh t3k1d`, 2026-09-29)

- Oracle: test-backend-ops FLASH_ATTN_EXT on the V100, kernel ON 2911/2911 passed (4.5 min), kernel OFF
  (`GGML_CUDA_FA_NO_VOLTA_SMALL=1`, same binary) 2911/2911 - the 27 added cases (27B geometry nb 1-8 f16/q8_0,
  head-128 GQA-4 twin, sinks/softcap/ALiBi at width 5) pass on both paths, so the fallback routing is intact.
- Speed (Qwen3.8-27B UD-Q4_K_XL, V100 at the 1380 MHz lock, `-b/-ub 2048 -fa on -r 2`, arms interleaved ON / OFF / ON in
  one run so the drift is visible; ON = the kernel, OFF = `GGML_CUDA_FA_NO_VOLTA_SMALL=1` in the same binary = the
  a9885783c routing: f16 -> tile rule at >= 16k, q8_0 -> VEC at width 1, TILE + whole-cache f16 conversion at width 5).

  Width 1 (tg32, t/s; ms per token in parentheses):

  | KV | arm | d 0 | d 16384 | d 65536 | d 131072 |
  |---|---|---|---|---|---|
  | f16 | ON | 35.39 (28.3) | 34.31 (29.1) | 30.59 (32.7) | 27.03 (37.0) |
  | f16 | OFF | 35.31 (28.3) | 32.17 (31.1) | 27.03 (37.0) | 22.36 (44.7) |
  | f16 | ON (repeat) | 35.18 | 33.27 | 30.41 | 27.09 |
  | q8_0 | ON | 34.63 (28.9) | 32.10 (31.2) | 29.40 (34.0) | 26.54 (37.7) |
  | q8_0 | OFF | 34.57 (28.9) | 29.59 (33.8) | 22.56 (44.3) | 16.62 (60.2) |
  | q8_0 | ON (repeat) | 34.57 | 31.53 | 29.55 | 26.07 |

  Reading: f16 +4..7 % at 16k, +13 % at 64k, +21 % at 128k over the tile rule (+48 % / +89 % over the VEC numbers of
  5.2); the attention cost per token at 128k is 8.7 +- 1.5 ms for 8.6 GB of KV (16 full-attention layers x 4 KV heads x
  256 x 2 x 2 B = 64 KB per position, GGUF metadata checked), i.e. the HBM rate (phase-0 floor 9.5 ms at 900 GB/s)
  within the noise of the depth-0 baseline. q8_0 +8 % at 16k, +30 % at 64k, +60 % at 128k over VEC and now within 3 %
  of the f16 path (it was slower than f16 at width 5 before); it does NOT beat f16: the in-register dequant is ~1.5 ALU
  ops per value (byte_perm + hsub2 + hmul2 per pair; the fused hfma2 form is inexact because 1152*d does not round to
  fp16) and the kernel runs issue-bound at ~60 % of the q8_0 byte rate (37.7 - 28.9 = 8.8 ms at 128k vs a 5.1 ms floor).
  On Volta q8_0 KV therefore buys capacity, not speed - the speed penalty is gone, the prize is not there.

  Width 5 (the MTP verify step, `-p 5 -n 0`, t/s; ms per step in parentheses):

  | KV | arm | d 0 | d 16384 | d 65536 | d 131072 |
  |---|---|---|---|---|---|
  | f16 | ON | 96.9 (51.6) | 85.4 (58.5) | 78.9 (63.3) | 70.3 (71.1) |
  | f16 | OFF | 97.4 (51.4) | 81.5 (61.4) | 64.1 (78.0) | 47.4 (105.5) |
  | q8_0 | ON | 94.6 (52.8) | 82.8 (60.4) | 76.3 (65.5) | 69.0 (72.5) |
  | q8_0 | OFF | 95.4 (52.4) | 77.4 (64.6) | 57.2 (87.5) | 42.2 (118.6) |

  Reading: the verify step at 64k / 128k costs 63 / 71 ms instead of 78 / 106 (f16, +23 % / +48 % t/s); attention at
  128k is ~20 ms of the step for 30 rows (four M tiles: each quadpair re-issues the V loads of its warp, L1-served, and
  P goes through shared memory) against ~9 ms at width 1 - room left, but the MTP round at 64k drops from ~83 to ~68 ms.
  q8_0 at width 5 tracks f16 within 2 % (the whole-cache f16 conversion is gone): 65 / 73 ms instead of 88 / 119
  at 64k / 128k (+33 % / +64 % t/s), and it stops being slower than f16.

- KL at 32k context (wiki.test, one 32768-token chunk, `-b/-ub 2048`), kernel ON vs the OFF base from the same binary:
  f16 mean KLD 0.000000, 99.9 % KLD 5.1e-5, top-1 100.000 %, PPL ratio 1.0003 (base PPL 6.0034); q8_0 mean KLD
  0.000000, 99.9 % KLD 4.9e-5, top-1 99.988 %, PPL ratio 1.0003 (base 6.0039) - the exact class on both KV types,
  the same bar the tile rule passed in 5.5 (byte identity across kernels is not expected: different summation order).

- MTP serve legs at depth (`153-p0/t3k-serve.sh`, the rolled image, llama-server `-c 131072 -b/-ub 2048 -fa on`, MTP head
  n-max 3, one 56568-token wiki prompt through the chat template, 160 greedy tokens, ON / OFF / ON per KV type):

  | KV | arm | prefill t/s | generation t/s (ms/tok) | acceptance | text |
  |---|---|---|---|---|---|
  | f16 | ON | 510 | 42.07 (23.8) | 100/176 = 57 % | correct summary (Boulter, Du Fu, ...) |
  | f16 | OFF | 509 | 35.41 (28.2) | 100/176 | identical |
  | f16 | ON (repeat) | 499 | 41.90 (23.9) | 100/176 | identical |
  | q8_0 | ON | 514 | 41.28 (24.2) | 100/176 | identical |
  | q8_0 | OFF | 511 | 33.51 (29.8) | 100/176 | identical |
  | q8_0 | ON (repeat) | 509 | 40.75 (24.5) | 100/176 | identical |

  Reading: the MTP round at 56k depth gains +19 % (f16) and +23 % (q8_0); all six legs produce the same 160 greedy tokens
  and the same acceptance counts, prefill is unchanged (it keeps the upstream MMA kernel). The first attempt used the raw
  `/completion` endpoint and got an immediate EOS from the instruct model - serve legs go through the chat template.

Verdict (2026-09-29 13:20): adopted; selected by default on Volta for the decode/verify shapes, kill switch kept.
Ships in `llamacpp-local-v100:a9885783c-t3k` (working tree on a9885783c, commit = user). Open after this increment:
the MTP serve legs at depth (d7-gates.sh pattern, `-c 131072`), the width-5 attention share (NT = 4 re-issues each
warp's V loads per quadpair and moves P through shared memory: ~20 ms at 128k vs ~9 ms at width 1), and the q8_0 issue
bound (~1.5 ALU ops per dequantised value; the fused hfma2 form is not exact).

## 7. T3 kernel increment 2: the verify-width (multi-tile) phase 2 (2026-09-29, user: "sure go ahead")

### 7.1 Where the width-5 time goes

Measured (6.2): width 1 pays ~8.7 ms of attention per token at 128k (the byte rate); width 5 (30 rows = four M tiles)
pays ~19.5 ms per step for the same K/V bytes. Phase 1 scales cleanly (K is streamed once per lane, the four tiles
only add mma issues). Phase 2 does not: with quadpair = (M tile, position slice) and NT = 4 every quadpair has to
cover all 16 chunks of the CTA step, so each warp re-issues its V loads four times (the LSU serves the same 16-byte
addresses to four lanes of one instruction, so HBM traffic is unchanged, but the LDG count, the register traffic and -
for q8_0 - the dequant ALU are all 4x), and at NT = 4 the S tiles (32 registers) push the kernel to the 255-register cap,
which shortens the load pipeline the compiler can build. Budget per warp and CTA step (128 positions, D = 256, f16):
memory floor ~31k clocks per SM step (8 warps x 32 KB at the SM's share of 900 GB/s); tensor cores 512 warp-mma x 8 clk
= 4k per warp (~26 % of the floor spread over 2 warps per sub-partition); the 4x LDG re-issue adds ~3k LSU clocks
per warp - none of these alone explains 2x, the register cap + the serialised phases (no load in flight across the two
barriers) do: at width 1 the same structure still reaches the byte rate because its phase-2 body is 4x shorter.

### 7.2 Design: dims across quadpairs, M tiles looped (V read once per warp)

Phase 2 becomes: warp w keeps its D/4 dims; the D/32 8-wide n-tiles of the warp are split across the quadpairs
(NQ_D = min(4, D/32): 4 at D >= 128, 2 at D = 64 where the remaining factor splits positions as today), each quadpair
loops over ALL M tiles for its n-tiles. A lane then streams V[pos][4*NJ dims] once per position (16 bytes of f16 at
D = 256: a contiguous 8-dim run under the n-tile permutation dim = Wd + 16q + 8h + 4j + e), pulls the P fragment of
every tile from shared memory (NT LDS.128 per chunk, broadcast across quadpairs) and issues 2*NJ mma per tile.
Accumulators: O[NT][NJ][8] = 16*NT registers at D = 256 (64 at NT = 4 as before, but 16 at NT = 1 instead of 64),
8*NT at D = 128/64. The V loads per step drop to the width-1 count for every NT, the q8_0 dequant with them; the
mma count is unchanged; the P reads grow from 4 to 16 LDS.128 per lane per step at NT = 1 (negligible). The
epilogue loses the position-slice reduction except at D = 64 (xor-8 sum over lane bit 3). Output staging: lane row
r0/r0+2 of tile m, columns {c0, c0+1} -> dims Wd + 16q + 4j + c0, {c0+4, c0+5} -> + 8. Nothing changes in phase 1,
the softmax, the P remap or the launch plumbing; the selector and the instances stay.

Expected: the width-5 step at 128k from ~71 ms toward ~62 (attention ~11 ms instead of ~20) and NT = 1 unchanged or
slightly better (lighter registers). Gate = the 6.2 protocol (oracle ON/OFF, width 1 + width 5 sweeps, KL 32k, the MTP
serve legs) against devbins-t3k1d and the a9885783c-t3k2 image.

### 7.3 Increment 2 gate (devbins-t3k2a vs devbins-t3k1d as the interleaved reference, `153-p0/t3k-chain.sh`, 2026-09-29)

- Compiler: D = 256 registers 242 (NT 4), 216 (NT 3), 204 (NT 2), 218 (NT 1), 0 spills (increment 1: 254-255 at every NT).
- Oracle: FLASH_ATTN_EXT 2911/2911 with the kernel ON and OFF, first build.
- Width 1 (tg32 t/s at 0 / 16k / 64k / 128k; arms new / reference / new-repeat in one run): f16 35.5 / 34.4 / 30.6 / 27.0,
  ref 35.2 / 33.0 / 30.3 / 26.9, repeat 35.2 / 33.3 / 30.0 / 26.8; q8_0 34.6 / 31.7 / 29.4 / 26.0, ref 34.6 / 30.1 / 28.9 / 26.1,
  repeat 34.6 / 31.6 / 28.7 / 26.5 - neutral, as designed (width 1 was already at the byte rate).
- Width 5 (`-p 5 -n 0`, t/s; ms per step in parentheses): f16 new 97.1 / 86.3 / 81.4 / 73.1 (51.5 / 57.9 / 61.4 / 68.4),
  ref 97.1 / 85.7 / 79.3 / 70.4 (51.5 / 58.3 / 63.1 / 71.0), repeat 97.2 / 86.6 / 81.6 / 73.4 -> **+2.6 % at 64k, +3.8 % at
  128k** (-1.7 / -2.6 ms per verify step); q8_0 new 95.2 / 83.3 / 79.0 / 71.5 (52.5 / 60.0 / 63.3 / 70.0), ref 94.5 / 82.0 / 77.0 / 69.0 (52.9 / 61.0 / 65.0 / 72.4), repeat 95.1 / 84.2 / 78.4 / 71.2 -> **+2.6 % at 64k,
  +3.5 % at 128k**.
- KL at 32k, kernel ON vs OFF (same binary): f16 mean 0.000000 / 99.9 % 5.1e-5 / top-1 100.000 % / PPL ratio 1.0003,
  q8_0 0.000000 / 4.9e-5 / 99.988 % / 1.0003 - the same figures as increment 1 (6.2). The MTP serve legs were skipped
  on the user's call (the width-5 sweep is the same shape; the 6.2 legs stand for the path).

Reading: the gain is real but a third of the 7.2 estimate. Attention at 128k / width 5 is now ~17 ms per step
(was ~19.5) against ~9 ms at width 1 and a 9.5 ms byte floor: the V re-issue was not the dominant cost. What is left is
the four-tile compute that does not overlap the memory stream inside a CTA - per warp step at NT = 4: 512 warp-mma
(phase 1 + 2, ~8k tensor-core clocks per SM step spread over the sub-partitions), ~190 LDS.128 of Q and P fragments
(~6k shared-memory clocks per SM step), the exp/softmax of 32 S values per lane, and two barriers per step - against a
~31k-clock memory floor per SM step; with two CTAs per SM the phases of the two CTAs overlap only partially. The next
lever would be a deeper restructure (software-pipelined K/V prefetch across the barriers, or a 256-position step to
halve the barrier count), not a mapping change; it is not part of this increment.

Verdict: adopted (strictly better or equal on every leg, lighter registers); ships in `a9885783c-t3k3`.
