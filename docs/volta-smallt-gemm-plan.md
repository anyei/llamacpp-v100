# Volta small-T tensor-core GEMM for quantized weights (TASKS #153 T1)

Status: BUILDING (2026-09-28). Phase 0 (`research/ninfer-phase0-2026-09-28.md`) measured the
problem: on a V100 a forward of T=2..8 tokens costs 1.3-3.1x a single-token step because the
multi-column MMVQ kernel (dp4a, SIMT) does not use Volta's tensor cores, so a speculative
verify of width 4-5 costs two target steps instead of ~1.1. NInfer's kernel origin
(dnv2003/v100-skinny, MIT) and its V100 port show the fix: `mma.sync.m8n8k4` in the
"quadpair-split-N" mapping.

## 1. Kernel shape

- One CTA owns 32 output rows (the mma N axis); its NWARPS warps (4/8/16, chosen by N) split K
  in 256-element super-blocks; a lane owns one weight row and streams that row's blocks
  global -> registers. No shared memory in the main loop; one cross-warp reduce at the end.
- m8n8k4 is issued per quadpair (lanes {0-3,16-19}, {4-7,20-23}, {8-11,24-27}, {12-15,28-31}).
  Quadpair q owns output rows q*8..q*8+7 and all four share one 8-token activation tile, so the
  warp tile is 8 tokens x 32 rows; up to 4 tiles (T <= 32) reuse every weight fragment.
- Activations are converted ONCE per call from f32 to f16 (T x K halves) plus 16-wide partial
  sums (T x K/16 floats) for the K-quant min/offset terms. Lane loads its token row's 8 k as
  one 16-byte load (L1/L2 resident; quadpair siblings share it).
- Numerics: weights enter the MMA as exact small integers in fp16 (nibbles / int8 / table
  values), products accumulate in fp32 per sub-block, then `c += alpha * c_sub + beta * S`
  with alpha = d*scale and beta = -dmin*min (or -offset*alpha) applied in fp32. This keeps
  the int-times-scale exact (MMVQ-class numerics with fp16 instead of q8_1 activations)
  and costs 8 FMAs per 8 MMAs.
- Dispatch (`ggml_cuda_mul_mat`, before MMVQ): compiled arch 70 AND type in the supported
  list AND 2 <= ne11 <= 32 AND ne00 % 256 == 0 AND 2-D operands AND f32 in/out. Env
  `GGML_CUDA_SMT=0` disables (value-parsed). T=1 stays on MMVQ (already at the bandwidth floor).

## 2. Formats (readers), in build order

| reader | files that need it | block | notes |
|---|---|---|---|
| Q8_0 | GSQ-RCO attention, drafters, XL mixes (110 tensors in the 27B) | 34 B, 2-aligned | int8 -> fp16 via byte-perm + 1152 bias; loads funnel-shifted to 4-byte alignment |
| Q4_K | UD-Q4_K_XL (69 tensors in the 27B), production | 144 B, 16-aligned | 6-bit sub-scales via `get_scale_min_k4`, min term via S32 |
| Q5_K | UD-Q4_K_XL majority (191 tensors in the 27B) | 176 B | Q4_K + high-bit plane |
| Q6_K | UD-Q4_K_XL (56) | 210 B | 4+2 bit planes, signed via -32 offset |
| IQ4_XS | UD-Q4_K_XL (70) | 136 B | nonlinear int8 table `kvalues_iq4nl`, 6-bit scales |
| Q4_0 | drafters | 18 B | offset -8 |
| Q3_K | GSQ-RCO experts (Flash-Next 72, GLM-5.3 105) | 110 B | 16-wide sub-scales, hmask plane |
| Q2_K | GSQ-RCO experts | 84 B | 16-wide scales+mins |
| Q2_0 | GSQ-RCO Flash-Next experts (62) | 10 B | 2-bit codes |

Then the `mul_mat_id` sibling (experts see 1-8 rows each at top-k routing; composes with the
#151 cache chain) using the same reader + MMA core with an expert/row indirection.

## 3. Gates

1. `test-backend-ops -b CUDA0 -o MUL_MAT` on the X99 V100 (dev bins inside the widefix image)
   for every reader, all its ne11 cases.
2. Phase-0 width bench (`llama.cpp-work/153-p0/bench-dev.sh`): pp2..pp32 vs the baseline
   curve (target: <= ~1.3 steps at width 5).
3. KL at `-ub 4` and `-ub 8` vs the image base (band <= the MMQ-vs-MMVQ class ~2e-3).
4. Serve-level MTP ladder on Qwen3.8-27B (phase-0 probes) and the #151 cache battery once
   `mul_mat_id` lands.

## 4. Log

- 2026-09-28: plan written; skeleton with Q8_0 + Q4_K readers in `ggml/src/ggml-cuda/mmsmt.{cu,cuh}`.
- 2026-09-28 (cont.): skeleton + all nine readers correct on test-backend-ops (1179/1179 on the
  V100) after two fixes: (1) the m8n8k4 accumulator registers of a lane belong to four different
  output columns, so per-column scales cannot be applied to a lane's own accumulators - v1 gathered
  them with quadpair shuffles per sub-block; (2) `__byte_perm` selectors are nibbles, not bytes
  (IQ4_XS table lookup). v1 was slow (Q4_K_S width 8: 109 t/s vs 140 MMVQ) because it used
  190-255 registers (two accumulator sets, per-sub-block shuffle rescale, whole-super-block
  register buffers) = 8 warps/SM. v2 folds sub-scales and mins into the fp16 B fragments
  (fp16-weight numerics, the cuBLAS-route class), rescales once per super-block for K-quants and
  never for Q8_0/Q4_0/Q2_0, loads per 32-68-byte chunk, caps tiles at 2 (T <= 16; wider stays
  on MMQ) and forces 64 registers via launch bounds.
- 2026-09-28 (late): kernel iterations v2-v6, all oracle-green (1179/1179 each) on the X99 V100:
  v2 (scales folded into fp16 B, 64 regs, per-lane row streaming) and v3 (shared A tile, K split
  across CTAs) were slower than MMVQ (Q4_K_S width 8: 99 / 74 t/s vs 140) - per-lane row streaming
  gives HBM ~80k short scattered streams. v4/v5 stage each warp's 32-row super-block through
  shared memory with coalesced loads (v5 adds register prefetch of the next super-block):
  **v5 = parity with MMQ at widths 5-8 (Q4_K_S 137 vs 140 t/s at width 8; Q8_0 92 vs 113), below
  MMVQ at widths 2-4 (35 vs 55 at width 2).** v6 (4 accumulator chains + incremental copy
  addressing with a divergent per-lane loop) regressed to 98. Diagnosis: v5 spends ~250 registers
  per thread on the whole-super-block prefetch (37-69 words per lane) so only 8 warps fit per SM
  and the SM issues ~0.35 instructions per clock; NInfer's kernel runs 32 warps/SM at 64 registers.
  Next: stage half/quarter super-blocks (header + chunks per format) so the prefetch is ~20
  registers and ~16 warps fit; NACC=2 chains; division-based copy addressing. Box facts that cap
  every kernel here: the X99 V100 power limit is set to 200 W (default 250) and thermal throttling
  held the SM clock at 1000-1300 MHz during the benches (throttle reason 0x20 in 141 of 329
  samples). llama-bench captures CUDA graphs at these widths, so event timing must run with
  GGML_CUDA_DISABLE_GRAPHS=1. The route is compiled in but OFF by default (`GGML_CUDA_SMT=1`
  enables) until it beats the incumbents.
- 2026-09-28 (evening): v7-v9, all oracle-green (1179/1179). **v7** (chunked staging: header +
  64-byte chunk segments per format, lane-linear copy, `__launch_bounds__(128,4)` = 16 warps/SM)
  was 2x SLOWER than the incumbents (Q4_K_S width 2/8: 17.6/69 t/s vs 55.7/140): the SASS census
  showed ~1250 integer instructions per chunk for 64 HMMA steps (a division and a 64-bit multiply
  per copied word, recomputed every chunk at the 128-register cap) plus 200 bytes of local memory
  in the Q4_K one-tile kernel. **v8** (4 lanes per row with hoisted row pointers, immediate
  offsets, 16-byte shared reads, register-resident scale decoding, chunk loop unrolled) = 28.0/104.7;
  the event timer reads 219-241 GB/s of weights on both Q4_K_S and Q8_0, flat across widths 2-8.
  **v9** (16-byte global loads for 16-byte-aligned formats Q4_K/Q5_K, 8-byte for IQ4_XS/Q4_0/Q2_0,
  exact segments for aligned formats) = 32.4/121.3 t/s, 262-278 GB/s; Q8_0 (still 4-byte loads)
  17.5/64.6, 235-253 GB/s. Per-step time is flat from width 2 to 8 in every version, so the kernel
  is purely weight-stream bound and the stream runs at 25-30% of HBM peak; scheduler IPC implied by
  the timer is ~0.18 with no local memory and 4x fewer instructions than v7. More warps (v8, 16/SM)
  did WORSE than v5 (8/SM, whole 144-byte super-blocks per row). Reading: DRAM access granularity
  per row (64-byte chunks x ~41k concurrent row streams) is the ceiling, the same ceiling dp4a MMQ
  sits on (277 GB/s, 128 B per row per step) while MMVQ sweeps whole rows (440-590 GB/s).
  **Analysis and the redesign (D6: CTA = 32 rows, 256-576 contiguous bytes per row per step, warps
  split the staged K range, cross-warp reduce) are in `docs/mmsmt-implementation.md`**, written
  before any further kernel code per the new "implementation document first" rule. Section 5 of
  it defines two ablations (copy-only / compute-only, env `GGML_CUDA_SMT_ABLATE`, dev builds only)
  to confirm the memory-pattern hypothesis before D6 is built.
- 2026-09-28 (evening, cont.): ablations on the v9 geometry (timer GB/s of weights, width 2 / 8).
  Q4_K_S: real 272 / 258, **copy-only 316 / 287**, compute-only 439 / 431. Q8_0: real 321 / 293,
  copy-only 338 / 305, compute-only 802 / 788. The memory pipeline of the 128-rows-per-CTA
  64-byte-chunk geometry tops out at ~316 GB/s with no compute at all = the access-pattern
  hypothesis holds; for Q4_K the decode + HMMA phase alone would also cap near 440, so both sides
  need work, memory first. Probe: 4 accumulator chains instead of 2 made it worse (real 252,
  compute-only 394; register pressure), reverted. **v10 = the D6 redesign of
  `docs/mmsmt-implementation.md`** (CTA = 32 rows, per-step tile of NSB whole super-blocks per row
  staged into 16-byte slots with 8 lanes x 16-byte pieces = 128 contiguous bytes per row per
  instruction and 272-352 per row per step, warps split the chunk-units, chunk index as a template
  parameter, cross-warp reduce through the stage buffer, direct dst write when ksplit == 1) built
  and sent to the X99 gates. Box: X99 V100 power limit 200 W (default 250), sudo needs a password =
  user item; the raise/lock command is in the session notes.
- 2026-09-28 (night): **v10 (D6 redesign) + v11 (shared activation slices) = the first versions that
  beat the incumbents.** v10: Q4_K_S timer 343 GB/s (copy-only 653, compute-only 414), Q8_0 485;
  bench wins at widths 4-8 (Q4_K) / 6-8 (Q8_0). The activation-fragment loads (L2 round trips right
  before each mma) were the next stall (no-load ablation: 537 real / 630 compute-only) -> v11 stages
  each warp's activation slice through a private shared region: **Q4_K_S 439 GB/s, Q8_0 610**; bench
  route on vs off at widths 2/3/4/5/8: Q4_K_S 0.85/1.06/1.28/1.45/1.31, Q8_0 0.77/0.81/0.87/0.93/1.27.
  Width-5 verify step = 1.6 T=1 steps (was 2.35). Route cap lowered to 8 (two-tile kernels spill and
  lose at 12-16). **Gate finding: the oracle script never enabled the opt-in route, so v6-v10 oracle
  passes tested the stock path; fixed, v11 re-validated with the route on (73/73 routed cases).**
  Details, tables and the route-table proposal: `docs/mmsmt-implementation.md` 4.5-4.6. KL gate
  (ub 8/4/3/2 vs route-off base, UD-Q4_K_XL) queued; serve-level MTP A/B and the default flip are
  user decisions.
- 2026-09-28 (night, cont.): the first KL run with the route on was NaN: the K-quant min fold divides
  by the super-block scale and real models have all-zero super-blocks (guarded, exact). With the
  guard (devbins-smt24): comparator vs cuBLAS on the real model = NMSE 5e-8 / 3e-14, no bad call;
  **KL gate PASS** at ubatch 8/4/3 (mean KLD 0.00165, top-1 98.2%, PPL ratio 1.000; control off-ub4 =
  0.00178 / 98.0%). Serve-level MTP A/B running. Oracle blind spots recorded in the implementation
  doc (k = 256 only, random non-zero weights): the comparator + KL gate are mandatory per version.
- 2026-09-28 (late night): **serve-level MTP A/B (dev server, Phase 0 prompts, two passes)**: hot-vs-hot
  tokens/s off/on = target-only 30.6/31.4 (never routes; cold-to-hot drift ~10% on the 200 W card),
  MTP n-max 2 47.7/45.3, n-max 3 46.5/50.5, n-max 4 44.1/52.3. Best point +6..+9% (n-max 4 on vs
  n-max 2 off); the ladder now climbs with n-max. Gains are below the bench's because the Q4_0 MTP
  head and the Q8_0 tensors stay on MMVQ and the round's draft/host costs (T2) are untouched; the
  n-max-2 loss is the extra prep/reduce launches per routed matmul (fold the f32->f16 conversion
  into the kernel = next tuning item, documented first). Decision points for the user: default flip
  (quality-neutral; loses 5% at n-max 2, wins 9-19% at 3-4), commit, image roll.
- 2026-09-28 (late night, cont.): user decisions: **route ON by default** (`GGML_CUDA_SMT=0` opts out) and
  **image roll** (`llamacpp-local-v100:1c63c03a1-smt`, rollback e117ee884-widefix). env-gates.md rows
  added. Dev binaries for the gates: devbins-smt24 on the X99.
- 2026-09-28 17:05: **ROLLED**. `llamacpp-local-v100:1c63c03a1-smt` = `:latest` (local + registry), X99
  launcher recreated on it; shipped-image checks: bench pp4/pp8 89.8/177.2 default vs 75.9/135.4 with
  `GGML_CUDA_SMT=0`, 1-chunk PPL 4.3517 at -ub 4. Rollback e117ee884-widefix, or `GGML_CUDA_SMT=0`.
- 2026-09-28 17:25: X99 power cap lifted by the user (250 W, SM clock locked 1380 MHz, no throttle
  reasons). Shipped-image bench at full power: pp4/pp8 90.7/178.6 default vs 76.6/136.3 with
  `GGML_CUDA_SMT=0` (short benches never throttled; the cap matters for sustained serves). Cleanup:
  27 X99 dev-binary dirs (2.5 GB) and the 42 GB tmpfs model copies removed; scripts and logs stay
  in `/home/anyei/153-p0`.
- 2026-09-28 (evening): follow-up program F1-F4 (details in `docs/mmsmt-implementation.md` 4.8-4.9):
  F1 rejected (in-kernel f32 conversion = net loss), F2 closed (V5 = 64-row CTAs with shared
  activation slices WINS +5-7% over v11; two-tile and deeper prefetch lose), F3 not built (Volta
  mul_mat_id already MMVQ-id at T <= 8; ~1 row per expert), F4 measured (verify = 83% of the round;
  fused chain worth 0; GPU-resident round <= ~12%). Per-type route table measured on requantised
  copies: Q3_K/Q2_K from 3, Q4_K/Q5_K from 4, Q6_K/IQ4_XS/Q8_0 from 6. Roll candidate V5 + table =
  devbins-smt33, gates queued.
- 2026-09-28 18:50: **ROLL 2** = V5 + per-type table, `llamacpp-local-v100:1c63c03a1-smt2` (= :latest,
  rollback 1c63c03a1-smt). Mix bench route on/off widths 1-8 = 1.00/1.00/1.00/1.17/1.24/1.34/1.34/1.33;
  serve n-max 2/3/4 = 50.8/56.1/56.9 vs 52.8/47.7/44.7; KL + comparator + oracle green.
