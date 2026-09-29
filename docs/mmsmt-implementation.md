# mmsmt: small-T tensor-core GEMM for Volta — implementation document

Status: 2026-09-28 (TASKS #153 T1), D6 + activation staging built as v11 (section 4.6): wins at widths 3-8 on Q4_K, 6-8 on Q8_0; KL gate + MTP A/B pending. Written
after kernel versions v1-v9 to freeze the analysis before the next design step, per the
"implementation document before code" rule. Companion: the dated
log in `docs/volta-smallt-gemm-plan.md`; Phase 0 evidence in `research/ninfer-phase0-2026-09-28.md`.

## 1. Problem

Decode-time matmuls with 2..8 activation rows (MTP verify width, dual-drafter verify, `-np` small
batches) run on V100 through `mul_mat_vec_q` (MMVQ, multi-column, T <= 5) or the dp4a MMQ tile
(T >= 6). Neither streams the weights at HBM speed for small T:

| path | T | step time, 27B Q4_K_S (15.8 GB) | effective weight bandwidth |
|---|---|---|---|
| MMVQ single column | 1 | 26.6 ms | 594 GB/s |
| MMVQ multi-column | 2 | 35.9 ms | 440 GB/s |
| MMVQ multi-column | 5 | 62.5 ms | 253 GB/s |
| MMQ dp4a | 8 | 57.0 ms | 277 GB/s |
| MMQ dp4a | 16 | 72.4 ms | 218 GB/s |

The MTP round cost gap vs NInfer (Phase 0) is entirely this: a width-3..5 verify step costs 1.5-2.0
T=1 steps here, 1.22 there. Target: T = 2..8 at >= 500 GB/s, i.e. a width-8 step at ~32 ms (250 t/s
at pp8, vs 140 today) and width-2 at ~33 ms (60 t/s, vs 55.7).

V100 (sm_70) facts that shape the design: fp16 tensor cores via `mma.sync.m8n8k4` only (no int8 MMA,
no `cp.async`, no `ldmatrix`); 64K registers and 96 KB shared per SM; 4 schedulers; HBM2 ~900 GB/s
peak, ~250-300 GB/s for scattered 32-64 B accesses.

## 2. What exists (v9 at the time of writing; `ggml/src/ggml-cuda/mmsmt.cu`; the route is ON by default since v11, `GGML_CUDA_SMT=0` disables)

- Route: `ggml_cuda_should_use_mmsmt` — Volta, 2 <= ne11 <= 16, ne00 % 256 == 0, 2-D contiguous
  f32 in/out, supported quant type. Hooked in `ggml_cuda_mul_mat` before MMVQ.
- Host: `mmsmt_prep_act` (f32 -> f16 activation tile, padded to 8/16 rows), the kernel writing
  partials `part[ksplit][tpad][N]`, `mmsmt_reduce` summing them in fixed order. Comparator
  (`GGML_CUDA_SMT_CHECK`) and event timer (`GGML_CUDA_SMT_TIME`) skip while a CUDA graph captures.
- Kernel geometry: CTA = 4 warps x 32 output rows each (128 rows), grid = (N/128) x ksplit. Each
  warp stages its 32 rows' super-block pieces into a private shared region (row stride = odd
  multiple of 16 B), prefetching the next piece into registers, and each lane decodes its own row.
- Formats (9): Q8_0, Q4_0, Q2_0, Q4_K, Q5_K, Q6_K, IQ4_XS, Q3_K, Q2_K. That covers Unsloth UD-Q4 mixes
  and the GSQ-RCO recipes (Q3_K/Q2_K/Q2_0 experts, Q8_0 attention).
- Correctness: `test-backend-ops -o MUL_MAT` 1179/1179 on every version since v4.

### 2.1 Invariants the readers rely on (do not break)

- m8n8k4 fragment maps. Quadpair `qp = (lane>>2)&3`; `r = (lane&3) + ((lane&16)?4:0)` is both the
  activation row of the A fragment and the weight column within the quadpair for B. C register i of
  lane L holds activation row `(i&2)|((L&16)?4:0)|(L&1)` and quadpair-local column
  `(i&1)|(((L>>1)&1)<<1)|((i>>2)<<2)`. Consequence: a lane's accumulators belong to four OTHER
  columns, so per-column scales are folded into the fp16 B fragments (block scales) or gathered from
  the owner lane `src = (lane&0xC)+(n&3)+((n&4)<<2)` by shuffle (`mmsmt_scale_acc`, super-block d).
- Bias trick: `__byte_perm(w, 0x64006400, sel)` builds half2 {1024+b0, 1024+b1}; subtracting the
  bias constant (1024/1025/1028/1032/1056/1152) yields the signed value exactly in fp16.
- Two accumulator chains per tile (`MMSMT_NACC = 2`) so consecutive mma do not serialize.
- Staged rows keep their 2-byte parity for formats whose super-block size is not a multiple of 4
  (Q6_K 210 B, Q3_K 110 B): the copy starts at the 4-byte floor and readers funnel-shift.
  `ALIGNED` formats never carry parity and their segments are exact (no slack words).
- Shared reads are 16-byte (`mmsmt_load_words`), which with the odd-multiple-of-16 row stride are
  bank-conflict-free per quarter warp. 4-byte shared reads at that stride are 4-way conflicted.
- Scale decoding happens on register words with constant indices (chunk loop fully unrolled);
  any runtime index into a register array silently becomes local memory.
- The K-quant min fold divides by the super-block scale (`mrat = dmin/d`). Real models contain
  all-zero super-blocks (d == dmin == 0: unused vocabulary rows, zero-padded tensors); the fold must
  guard d == 0 (the -dmin*m term is then 0 exactly), otherwise NaN reaches the logits. Found by the
  KL gate on 2026-09-28 (v11 route on: PPL NaN) after the oracle passed.
- The oracle (`test-backend-ops -o MUL_MAT`) covers k = 256 (one super-block: a single, partial
  step), random non-zero weights and m = 16 only. It cannot see multi-step pipeline bugs, zero
  blocks, or real tensor shapes; every version also needs the comparator (`GGML_CUDA_SMT_CHECK=1`)
  and the KL gate on a real model before it counts as correct.

## 3. Measurements that drive the redesign

| version | staging | warps/SM | pp2 / pp8 t/s (Q4_K_S) | weight BW |
|---|---|---|---|---|
| v1-v4 | none, per-lane row streaming | 16-32 | - | ~200 GB/s |
| v5 | whole super-block per warp, 250 regs | 8 | 60 / 137 | ~270 GB/s |
| v7 | 64 B chunks, lane-linear copy, 128 regs | 16 | 17.6 / 69 | ~110 GB/s (address math + spills) |
| v8 | 64 B chunks, 4 lanes/row copy | 16 | 28.0 / 104.7 | 219-241 GB/s (measured by timer) |
| reference | MMVQ / MMQ | - | 55.7 / 140.3 | 440 / 277 GB/s |

v8 instruction census (Q4_K, one tile, per super-block): 1480 instructions, 256 HMMA steps, 108 LDG,
48 STS, 11 LDS, no local memory. Scheduler IPC implied by the timer: ~0.18. The kernel is stalled
82% of the time on something that more warps (v8 vs v5) made WORSE, not better.

Interpretation (H1, primary): DRAM access granularity. Every design so far has each warp own 32
rows and advance them 64-144 B at a time; with 16 warps/SM that is ~41k concurrent row streams
each touching 1-2 sectors per step. HBM row-buffer locality collapses and the achieved bandwidth
matches the known random-sector ceiling of V100 (~250 GB/s). The reference kernels rank the same
way: contiguous bytes per stream episode 64 B -> ~210 GB/s (v8), 128-144 B -> 270-300 (MMQ, v5),
2.9 KB (whole row, MMVQ) -> 440-590.

Secondary hypotheses, to be excluded by ablation (section 5): H2 load latency not covered by the
one-piece register prefetch; H3 HMMA dependency latency (4 dependent steps per mma on Volta).

## 4. Design D6: CTA-wide row-contiguous tile

Principle: every warp load instruction reads >= 128 B contiguous of ONE row, every CTA step reads
>= 256 B contiguous per row, and the number of concurrent row streams drops 4x.

### 4.1 Work assignment

- CTA = 4 warps, **32 output rows** (was 128). Grid = (N/32) x ksplit, ksplit = ceil(4*nsm /
  (N/32)) capped at nsb/NSB; for N = 10240 that is ksplit 1 (no reduce kernel needed), N = 5120 -> 2,
  N = 1024 -> 10.
- A step stages `NSB` consecutive super-blocks of all 32 rows: the tile is 32 rows x `NSB*SB_BYTES`
  contiguous bytes per row. NSB per format targets 256-576 B per row:

| format | SB bytes | NCH | NSB | bytes/row/step | chunk-units/step | units per warp |
|---|---|---|---|---|---|---|
| Q8_0 | 272 | 4 | 1 | 272 | 4 | 1 |
| Q4_0 | 144 | 2 | 2 | 288 | 4 | 1 |
| Q2_0 | 72 | 1 | 4 | 288 | 4 | 1 |
| Q4_K | 144 | 2 | 2 | 288 | 4 | 1 |
| Q5_K | 176 | 2 | 2 | 352 | 4 | 1 |
| Q6_K | 210 | 2 | 2 | 420 | 4 | 1 |
| IQ4_XS | 136 | 2 | 2 | 272 | 4 | 1 |
| Q3_K | 110 | 2 | 4 | 440 | 8 | 2 |
| Q2_K | 84 | 2 | 4 | 336 | 8 | 2 |

- The 4 warps split the staged K range by chunk-unit: warp w decodes units `[w*U, (w+1)*U)`, unit u
  = super-block `u / NCH`, chunk `u % NCH` — the existing `reader::chunk` / `reader::finish` are
  reused unchanged (finish runs per super-block per warp: with U = 1 a warp holds half a super-block
  of K-quants, so finish applies d to that half; correct because d is per super-block and cpart holds
  only that super-block's partial).
- Each warp accumulates its own K-subset in `crun`; after the last step the 4 warps' `crun` are
  summed through shared memory (4 x 32 lanes x 8 floats = 4 KB, reusing the stage buffer after a
  `__syncthreads`), warp 0 writes the CTA's 32 x tpad outputs to `part[blockIdx.y]` (or straight to
  dst when ksplit == 1).

### 4.2 Staging: one 16-byte-aligned slot per super-block, raw block layout inside

- The tile row holds `NSB` slots of `SLOT = pad16(SB_BYTES + parity slack)` bytes; super-block i of
  the step is copied from global `row*nb01 + (sb+i)*SB_BYTES` (4-byte floor for parity formats)
  into slot i. Readers see the raw ggml block layout at `hp = slot + par` and use the block's own
  byte offsets (Q4_K: d/dmin at 0, scales at 4, qs at 16 + 64c; Q6_K: ql at 64c, qh at 128 + 32c,
  scales at 192, d at 208; Q3_K: hmask 0, qs 32 + 32c, scales 96, d 108; Q2_K: scales 0, qs 16 +
  32c, d/dmin 80; Q5_K: header 0, qh 16, qs 48 + 64c; IQ4_XS: header 0, qs 8 + 64c; Q8_0/Q4_0/Q2_0:
  blocks of 34/18/18 bytes). The segmented `*_AT` slots of v7-v9 disappear.
- Per-format widths. `GW` = global load width (alignment of a segment start in the weight buffer),
  `RW` = shared read/store width (alignment of every in-block offset the reader touches, relative to
  the 16-byte slot base). The row stride in WORDS is chosen so the reader's `RW`-wide loads are
  bank-conflict-free for the lanes that read simultaneously: RW 16 (quarter warp) -> stride = 4 mod 8;
  RW 8 (half warp) -> stride = 2 mod 4; RW 4 (full warp) -> stride odd. The same rule makes the
  copy's stores conflict-free (8 lanes per row, 4 rows per instruction).

| format | SB | slot | NSB | NCH | GW | RW | stride words | smem/CTA | prefetch words |
|---|---|---|---|---|---|---|---|---|---|
| Q8_0 | 272 | 272 | 1 | 4 | 16 | 4 | 69 | 8.8 KB | 24 |
| Q4_0 | 144 | 144 | 2 | 2 | 16 | 8 | 74 | 9.5 KB | 32 |
| Q2_0 | 72 | 80 | 4 | 1 | 8 | 8 | 82 | 10.5 KB | 32 |
| Q4_K | 144 | 144 | 2 | 2 | 16 | 16 | 76 | 9.7 KB | 32 |
| Q5_K | 176 | 176 | 2 | 2 | 16 | 16 | 92 | 11.8 KB | 32 |
| Q6_K | 210 | 224 | 2 | 2 | 4 | 16+parity | 116 | 14.8 KB | 32 |
| IQ4_XS | 136 | 136 | 2 | 2 | 8 | 8 | 70 | 9.0 KB | 32 |
| Q3_K | 110 | 112 | 2 | 2 | 4 | 16+parity | 60 | 7.7 KB | 16 |
| Q2_K | 84 | 96 | 2 | 2 | 4 | 16 | 52 | 6.7 KB | 16 |

  (Q3_K/Q2_K start at NSB 2 = 220/168 bytes per row per step to keep 16 prefetch words; NSB 4 is
  the first tuning knob if they lag.) All formats keep 4 CTAs per SM (<= 24 KB shared each).
- Copy geometry: 8 lanes per row, 4 rows per warp instruction, each lane owns one 16-byte piece
  (`GW` = 16: one LDG.128; 8: two LDG.64; 4: four LDG.32) at `16*lane8 + 128*j` of the segment; a
  segment needs ceil(SLOT/128) instructions per row group; warp w copies rows 8w..8w+7 (2 groups).
  Row pointers: 2 per lane, hoisted; loads are pointer + immediate. Rows past N clamp to row 0.
  Stores use `RW`-wide STS from the same 4 registers. Prefetch words per lane = 4 x pieces (table).
- Pipeline per step: `__syncthreads`; store the prefetched tile; `__syncthreads`; issue the next
  tile's loads; decode. Partial last tile: `nsb_step = min(NSB, sb1 - sb)`; segments and units
  beyond it are skipped (never read past the tensor).

### 4.3 Decode assignment and reduction

- Chunk-units per step = NSB x NCH; warp w decodes units `[w*U, (w+1)*U)`, U = NSB*NCH/4 (1 or 2).
  Unit u -> super-block `u / NCH`, chunk `u % NCH`. The chunk index must be a compile-time constant
  inside the readers (register-array scale decoding), so `reader::chunk<NTILES, C>` takes it as a
  template parameter and the warp-uniform runtime value is dispatched through a `switch` (folds
  when U is a multiple of NCH). `finish` runs after every unit (cpart then holds one super-block's
  partial; d is per super-block, so this is exact for any U).
- Every warp's lane owns the same output row set (row `colw + qp*8 + r`), so the 4 warps' `crun`
  are elementwise partials over disjoint K: after the step loop, `__syncthreads`, warps 1..3 write
  `crun` into the (now free) stage buffer (3 x 32 x 8 x NTILES floats <= 6 KB), `__syncthreads`,
  warp 0 sums in fixed order and writes. ksplit == 1 writes `dst` directly (no partial buffer, no
  reduce kernel); otherwise `part[blockIdx.y]` + `mmsmt_reduce` as today.
- Grid: (ceil(N/32), ksplit); ksplit = ceil(4*nsm / ceil(N/32)) capped at the step count, and
  `sb_per_split` rounded up to a multiple of NSB.

### 4.3b Register budget (128 cap, 4 CTAs/SM = 16 warps)

Q4_K one tile: crun 8 + cpart 16 + bq 16 + decode words 16 + header 4 + prefetch 32 + pointers 4 +
loop/addresses ~15 = ~111. If a one-tile kernel spills, the first lever is 4 lanes per row (64-byte
pieces, prefetch 24 for Q4_K), the second NSB. Two-tile variants (T 9..16) will spill as today;
acceptable (MTP verify widths are <= 8), listed as follow-up.

### 4.4 Expected outcome and decision rule

- If H1 holds, D6 with NSB = 2 should reach the MMQ/v5 band or better at once (>= 300 GB/s) and the
  copy-only ablation (E1 on D6) should show >= 500 GB/s headroom; then tune NSB / LPR / prefetch.
- If D6 does not move the number (still ~230 GB/s), H1 is wrong: stop, run E1/E2 (section 5) and
  re-diagnose before any further kernel change.

### 4.5 D6 result and the next step: activation fragments through shared memory (v11)

D6 (v10) measured on the X99 V100, timer GB/s of weights at width 2 / 8: Q4_K_S real 343 / 341,
copy-only 653 / 608, compute-only 414 / 414; Q8_0 real 485 / 481, copy-only 743 / 696, compute-only
639 / 635. Bench: Q4_K_S width 4/5/6/7/8 = +6/+20/+9/+9/+9% over the incumbents, width 2/3 = 0.70/0.88,
widths 12-16 = 0.75 (two-tile spills); Q8_0 6-8 = +8%, 2-5 = 0.63-0.78. The memory side is fixed;
the decode phase is now the limiter.

Cause found by ablation: the A-fragment loads. Each lane loads 8 x 16 B of the fp16 activation tile
from global memory per chunk, right before the mma that consumes it; the tile (tpad x K x 2 B = 80 KB)
does not stay in L1 next to the streaming weights, so every load is an L2 round trip (~250 cycles)
that the compiler cannot hoist at the 128-register cap. Replacing the loads by register constants
lifts the real kernel to 537 / 512 GB/s and the compute-only ceiling to 630 / 627 (Q4_K_S).

Design (v11):
- Each warp stages the activation slice of its chunk-unit for the step into a private shared region:
  NTILES x 8 rows x K_PER_CH x 2 B (Q4_K: 2 KB), row stride K_PER_CH*2 + 16 B so the 8 rows an mma
  slice-pair reads sit on distinct 4-bank groups (conflict-free; the 4 lanes sharing a row broadcast).
- Load: lane-linear 16-byte pieces (NTILES x 8 x K_PER_CH/8 pieces, 4 x NTILES per lane, 512
  contiguous bytes per instruction), issued at the end of the warp's decode of the previous step (the
  region is private, so only the warp's own reads must be done), stored right after the first CTA
  barrier next to the weight stores. The 4 x NTILES uint4 live only across the barrier, outside the
  decode-phase register peak.
- `mmsmt_sub` reads the fragment from shared memory with a unit-local k; readers pass chunk-local
  offsets (k0 = 0 per unit).
- Shared per CTA: weights + NWARPS x U x NTILES x 8 x (K_PER_CH*2 + 16): Q4_K 9.7 + 8.7 = 18.4 KB (4
  CTAs/SM); Q8_0 (K_PER_CH 64) + 4.6 KB; Q2_0 (K_PER_CH 256) + 16.9 KB = 27 KB -> 3 CTAs/SM (follow-up:
  NCH 2 for Q2_0). Two-tile kernels: + double -> 3 CTAs/SM.
- Route cap: `MMSMT_MAX_BATCH_SIZE` 16 -> 8 for now; widths 9-16 stay on MMQ until the two-tile
  register budget is solved (they lose 10-25% today).

Expected: Q4_K real ~500-540 GB/s = width-2 step ~30 ms (pp2 ~65 t/s vs MMVQ 55.6), width-8 ~250
t/s (vs MMQ 140). Q8_0 will still trail MMVQ at widths 2-3 (MMVQ Q8_0 reaches ~715 GB/s there);
the route becomes a per-type, per-width table gated by measurement (section 6).

### 4.6 v11 result (2026-09-28 evening) and the route table

v11 = D6 + shared activation slices, route cap 8. Timer GB/s of weights at width 2 / 8: Q4_K_S
**439 / 436** (v10 343), copy-only (now including the activation staging) 494 / 493, compute-only
569 / 569; Q8_0 **610 / 592** (v10 485). Bench, route on vs off (t/s):

| width | Q4_K_S off | on | ratio | Q8_0 off | on | ratio |
|---|---|---|---|---|---|---|
| 2 | 55.5 | 47.3 | 0.85 | 49.1 | 37.7 | 0.77 |
| 3 | 66.9 | 71.2 | 1.06 | 69.4 | 56.1 | 0.81 |
| 4 | 73.5 | 93.9 | 1.28 | 84.3 | 73.6 | 0.87 |
| 5 | 80.2 | 116.1 | 1.45 | 98.4 | 91.5 | 0.93 |
| 8 | 140.2 | 184.1 | 1.31 | 112.7 | 142.6 | 1.27 |

The width-5 verify step now costs 1.6 single-token steps (was 2.35 on MMVQ; NInfer 1.22). MMVQ on
Q8_0 is unusually strong at widths 2-5 (~715 GB/s at width 2), so the route needs a per-type minimum
width: K-quants / IQ4_XS / Q3_K / Q2_K -> from width 3 (measured on Q4_K; the other K-quant decodes
cost the same or more per byte, so the same crossover is assumed until measured); Q8_0 / Q4_0 / Q2_0
(cheap decodes, strong MMVQ) -> from width 6. Env `GGML_CUDA_SMT_MIN` overrides the minimum for
experiments. The route is ON by default since 2026-09-28 (user decision after the KL gate and the serve A/B, 4.7); `GGML_CUDA_SMT=0` disables it.

**Gate finding**: the oracle script never set `GGML_CUDA_SMT=1`, so from the moment the route became
opt-in (v6) every "1179/1179" ran the stock kernels; the no-activation ablation build passing the
oracle exposed it. Fixed in `oracle-smt.sh` (route forced on). v11 re-validated with the route on:
all 73 routed cases (nine types, widths 2-8) pass; the one failure in the run is q5_0 at width 1
(not a routed type or width) at the error threshold, re-run queued. v6-v10 were never validated
individually; v11 supersedes them.

Remaining levers, in order: (1) overlap (real 439 vs min(copy 494, compute 569)); (2) the
activation staging cost (copy-only 653 -> 494 when it was added: 8 KB of L2 reads per CTA step);
(3) two-tile register budget for widths 9-16; (4) `mul_mat_id` sibling for MoE experts.

### 4.7 Gates on the real model (2026-09-28 night, v11 + zero-block guard = devbins-smt24)

- First KL run with the route on returned PPL NaN at ubatch 8: the K-quant min fold `dmin/d` on
  all-zero super-blocks (d == dmin == 0). Guarded (`mrat = d != 0 ? dmin/d : 0`, exact since the
  -dmin*m term is 0 there). The comparator on the real UD-Q4_K_XL model then reads NMSE 5e-8 (q5_K)
  and 3e-14 (q8_0) vs cuBLAS with no bad call at ubatch 8 or 3, PPL identical to 4 digits.
- KL gate, 16 x 512 wikitext, base = route off at ubatch 8, same dev binary:

| leg | mean KLD | 99.9% KLD | top-1 | PPL(Q)/PPL(base) |
|---|---|---|---|---|
| control: off, ubatch 4 | 0.001775 | 0.078 | 97.99 % | 1.0008 |
| on, ubatch 8 | 0.001655 | 0.053 | 98.16 % | 0.9999 |
| on, ubatch 4 | 0.001647 | 0.053 | 98.16 % | 1.0000 |
| on, ubatch 3 | 0.001649 | 0.053 | 98.19 % | 0.9999 |
| on, ubatch 2 (= control: width 2 is not routed) | 0.001775 | 0.078 | 97.99 % | 1.0008 |

  Verdict: quality-neutral; the route sits slightly closer to the base than the stock MMVQ/MMQ
  switch does (fp16-weight numerics vs dp4a re-quantised activations).
- Serve-level MTP A/B (`serve-smt.sh` / `serve-smtr.sh`: the Phase 0 prompts, dev server, route off
  vs on, two passes in opposite order because the 200 W card drifts ~10% cold-to-hot; the target-only
  leg never routes and shows the drift: off 34.0 cold / 30.6 hot, on 32.0 / 31.4). Route-on numbers
  reproduced within 0.3 t/s across passes. Hot-vs-hot, tokens/s:

| config | off | on | on/off |
|---|---|---|---|
| target-only (no routed width) | 30.6 | 31.4 | drift |
| MTP n-max 2 (verify width <= 3) | 47.7 | 45.3 | 0.95 |
| MTP n-max 3 | 46.5 | 50.5 | 1.09 |
| MTP n-max 4 | 44.1 | 52.3 | 1.19 |

  Best ladder point: off = n-max 2 at 47.7-49.4, on = n-max 4 at 52.3 -> +6..+9%. The ladder now
  climbs with n-max instead of peaking at 2, as the kernel intends; the absolute gain is smaller than
  the bench gain (+28..45% at widths 4-5) because the MTP head (Q4_0) draft steps and the Q8_0
  tensors stay on MMVQ, and the round's host/draft costs (T2) are untouched. The 5% loss at n-max 2
  is the extra launches per routed matmul (f32->f16 prep and, for narrow N, the K-split reduce) on
  steps that are only ~1.06x faster in the bench; folding the conversion into the kernel's activation
  staging removes the prep launch and the x16 buffer (follow-up, needs its section here first).

## 4.8 Follow-up program (user go 2026-09-28: items F1-F4, "analysis and development unless a real problem")

### F1. Fold the f32 -> f16 activation conversion into the kernel

Analysis. Every routed matmul today launches `mmsmt_prep_act` (f32 rows -> f16 tile `x16[tpad][K]`,
a pool allocation) before the main kernel, and for narrow N a `mmsmt_reduce` after it. The serve A/B
showed a 5% loss at MTP n-max 2 (verify width <= 3, where the bench gain is only 1.06x) that these
extra launches explain: ~450 routed matmuls per step on the 27B, each with 1-2 extra graph nodes
(~2-3 us each in replay) = 1-3 ms on a ~43 ms step. The conversion itself is trivial work.

Design. `load_act` reads the f32 activation rows directly: per 16-byte f16 piece (8 values) it loads
two `float4` from `src1 + row*nb11 + k` and converts with `__floats2half2_rn` into the same `apf`
uint4 register, so the register budget is unchanged; rows >= ne11 (tile padding) are zeroed in
registers (no memory read). The kernel gets `src1`, `nb11` (floats) instead of `x16`; the pool
buffer and the prep launch disappear; the timer's "prep" phase becomes 0. L2 traffic for
activations doubles (f32), still far below the weight stream (8 KB -> 16 KB per CTA step vs 9 KB of
weights) and it is L2-resident. The K-split reduce stays (needed for determinism; N = 1024
projections only). Gates: oracle route-on, comparator, KL at ub 3/4/8, width bench, serve A/B n-max
2 (the target: >= parity with route off).

### F2. Overlap, width 2, two-tile budget (experiment matrix, one build each, timer + bench)

- E-ov1: `__launch_bounds__(128, 3)` (170 registers, 12 warps/SM) with a two-step register
  prefetch (NPF x 2) - tests whether deeper prefetch beats occupancy.
- E-ov2: NSB 4 for Q4_K-class with 4 lanes per row (prefetch 48 words) - longer per-row streams.
- E-ov3: activation slice loaded for step s+1 during the decode of step s (16 more live registers)
  vs the current end-of-decode load.
- Width 2: re-bench after F1; if the route still trails MMVQ at width 2 the per-type minimum stays 3
  (no code).
- Two-tile (T 9-16): NACC 1 for NTILES 2 (cpart 16 instead of 32) + half-chunk decode words (w[8])
  to fit 128 registers; bench widths 12/16 vs MMQ (221 t/s at 16 today). If it does not beat MMQ the
  cap stays at 8.

F2 results (2026-09-28 18:13, real UD-Q4_K_XL, route forced from width 2, t/s at widths 3/4/8/12/16;
off reference 69.4/78.1/118.4/151.5/199.1):

| variant | 3 | 4 | 8 | 12 | 16 | timer w4 |
|---|---|---|---|---|---|---|
| F1 as built (4 CTAs/SM, one tile in flight; V1 bound was a no-op at 128 regs) | 60.1 | 80.0 | 142.7 | unrouted | unrouted | 369 GB/s |
| V4: two-tile kernels with one chain, cap 16 | 57.6 | 76.8 | 137.6 | 134.6 | 168.8 | 351 |
| V3: V4 + two weight tiles in flight, 160 regs, 3 CTAs/SM | 56.9 | 75.7 | 137.4 | 146.1 | 183.6 | 342 |

Verdict: occupancy beats prefetch depth on this kernel (V3 loses 7%); the two-tile path still loses
11-15% to MMQ at 12/16 even without spills (3 CTAs/SM from the doubled activation slice) - the cap
stays at 8. Width 3 on the MIXED model is a loss (0.87) even for F1: not an overlap problem but a
per-type one (section 4.9). E-ov2 (NSB 4) is not worth a build after V3: it needs the same
register/occupancy trade.

### F1 verdict: rejected (2026-09-28 18:17)

Back-to-back on the real UD-Q4_K_XL (route forced from width 2, widths 3/4/8): v11 58.7 / 78.7 /
150.8 t/s, kernel timer 396 GB/s (prep 9 us per call); F1 57.7 / 76.7 / 137.8, timer 352. The
in-kernel conversion doubles the activation bytes each CTA pulls from L2 (f32 instead of f16), and
with 32-row CTAs the activation traffic is already ~90% of the weight stream (320 CTAs x 80 KB tile
= 26 MB per 29 MB matmul), so the extra L2/LSU work costs more than the 9 us prep launch it saved.
The comparator/KL/oracle gates passed (numerically identical), so F1 is correct but slower: not
shipped; source kept as `mmsmt.cu.f1-final`. The activation-traffic observation is the lead for V5.

### V5: 64-row CTAs (halve the activation re-read), same occupancy

- CTA = 8 warps, 64 output rows: warps 0-3 decode rows 0-31, warps 4-7 rows 32-63; within a half the
  4 warps split the step's chunk-units as today (`kw = warp % 4`). Each warp still copies 8 rows
  (8 x 8 = 64), so the copy code is unchanged with `row base = warp*8`.
- Activation slices are per K-unit, shared by the two warps with the same `kw`: loaded and stored by
  the `half == 0` warp only, read by both after the second barrier (the first barrier already
  orders the previous step's reads before the overwrite). Activation L2 traffic per row halves
  (13 MB per matmul instead of 26).
- Shared per CTA: weights 64 x stride (Q4_K 19.5 KB) + 4 activation slices (8.7 KB) = 28 KB -> 2
  CTAs/SM by registers (128 x 256 threads x 2 = the file), i.e. the same 16 warps/SM as today, 56 KB
  of shared. Cross-warp reduce per half (3 x 32 lanes x 8 floats each, fits in the stage buffer).
- Grid (ceil(N/64), ksplit) with ksplit aimed at 2 CTAs/SM (N 5120 -> 80 groups x 2).
- Expected: the activation staging cost the copy side 24% when it was added (653 -> 494 GB/s in the
  v10 ablation); halving that traffic should return roughly half of it (~+8-12%), at no occupancy
  cost. Gates: oracle route-on, comparator, bench vs v11 back to back, KL if it wins.
- Result (2026-09-28 18:32, real UD-Q4_K_XL, route forced from width 2, back to back): **V5 61.7 /
  82.7 / 157.8 t/s at widths 3/4/8, timer 423 GB/s; v11 58.6 / 78.7 / 151.2, timer 396** -> +5% bench,
  +7% kernel. Oracle route-on 1179/1179. The two-tile instantiation was dropped in the same change
  (its doubled activation slices put Q2_0 over the 48 KB static shared limit at 64 rows, and it had
  lost to MMQ anyway). V5 + the 4.9 table = the roll candidate (devbins-smt33); comparator, KL, bench
  and serve legs queued.

### 4.9 Per-type route table from measurement (the width-3 finding)

On the real UD-Q4_K_XL mix, F1 route on vs off = 0.83 / 0.98 / 1.16 at widths 3/4/8, far below the
pure-Q4_K copy (1.06 / 1.28 / 1.31), and the serve loss at MTP n-max 2 (-12%) survived F1. The mix is
191 Q5_K, 110 Q8_0, 70 IQ4_XS, 69 Q4_K, 56 Q6_K tensors; only Q4_K and Q8_0 were ever measured, and
the table assumed the other K-quants cross over where Q4_K does. Their decodes are heavier (Q5_K:
high-bit plane merge; Q6_K: 6-bit unpack from two planes; IQ4_XS: table lookup, ~3x Q4_K's
per-word work), so their compute ceilings sit lower and they can lose to MMVQ at widths 3-4 while
Q4_K wins. Measurement: requantised pure copies (Q4_K_S, Q5_K_S, Q6_K, IQ4_XS, Q3_K_S, Q2_K) benched
route off vs on (min width forced to 2) at widths 2-8; the per-type minimum width is the first width
where on >= 1.03 x off, or "never" (type excluded from the route).

Results (2026-09-28 18:22-18:30, v11 build, X99 at 250 W, `-r 8`; route on / route off):

| type | w2 | w3 | w4 | w5 | w6 | w8 | min width |
|---|---|---|---|---|---|---|---|
| Q3_K | 0.98 | 1.20 | 1.33 | 1.36 | 1.50 | 1.50 | 3 |
| Q2_K | 0.89 | 1.10 | 1.38 | 1.51 | 1.54 | 1.53 | 3 |
| Q4_K | 0.81 | 1.00 | 1.19 | 1.37 | 1.31 | 1.30 | 4 |
| Q5_K | 0.86 | 0.97 | 1.19 | 1.28 | 1.41 | 1.41 | 4 |
| Q6_K | 0.72 | 0.82 | 0.95 | 1.03 | 1.19 | 1.19 | 6 |
| IQ4_XS | 0.54 | 0.63 | 0.71 | 0.75 | 1.13 | 1.13 | 6 |
| Q8_0 (4.6) | 0.77 | 0.81 | 0.87 | 0.93 | 1.08 | 1.08 | 6 |

Reading: the kernel is flat in width (weight-stream bound) so the crossover is set by how strong
the incumbent is for that type. MMVQ's IQ4_XS path (table via a fast dp4a lookup) and Q8_0 are the
strongest, Q3_K/Q2_K the weakest (their dp4a unpack is costly), K-quants in between. On the GSQ-RCO
fleet's DENSE tensors this means: Q8_0 attention from width 6 only, but any Q3_K/Q2_K dense tensor
(shared experts, MTP heads) from width 3. Applied to `ggml_cuda_should_use_mmsmt`; Q4_0/Q2_0 are
grouped with Q8_0 until measured. The width-3 loss on the UD-Q4_K_XL mix is now explained by
Q5_K (0.97), Q6_K (0.82) and IQ4_XS (0.63) all routing at width 3; with the table the mix routes
Q4_K/Q5_K from 4 and Q6_K/IQ4_XS/Q8_0 from 6, so MTP n-max 2 (verify width 3) falls back to the
stock kernels entirely (no loss, no gain) and n-max 3-4 keep their gains.

### F3. `mul_mat_id` sibling - analysis says: not for the models in this fleet

The CUDA expert matmul already has expert-aware fused kernels: MMVQ-id up to
`get_mmvq_mmid_max_batch` rows and MMQ-id above it (both on Volta), plus the generic sorted path
(the #151 bug site). What a tensor-core sibling would change is the per-EXPERT width regime: the
m8n8k4 tile is 8 activation rows, and the kernel's time is flat in the width (weight-stream bound),
so it only pays when an expert sees >= 3 rows in one call. With fine-grained MoE routing (E experts,
top-k) at decode/verify width T the expected rows per used expert is T*k / (E*(1-(1-k/E)^T)):
Flash-Next / DeepSeek-V4-class (E 256, k 8) at T 4 -> 1.05 rows per expert; at T 8 -> 1.1. Those
experts run at width 1-2 where MMVQ-id (~590 GB/s single column) beats this kernel (~440 GB/s at
any width) by ~35%. Per-expert widths >= 3 only appear at T >= ~24 (prefill chunks, where MMQ-id is
already the right tool) or for coarse MoE (E <= 16, Mixtral-style) which the fleet does not run.
The MoE models' dense parts (attention projections, shared experts, the MTP head) are plain
`mul_mat` and already take the route. Decision: measure instead of build - a route-on/off bench on
a MoE model (dense parts) and a rows-per-expert histogram from a MoE decode; build the sibling only
if a model with wide experts appears. This is the "real problem" clause: the sibling as designed
would not pay for the current fleet. Confirmed in code: on Volta `get_mmvq_mmid_max_batch` returns
`MMVQ_MAX_BATCH_SIZE` (8), i.e. every expert matmul at T <= 8 already runs the fused expert-aware
MMVQ, the right kernel for width-1..2 experts.

Second finding for the GSQ-RCO fleet (Flash-Next, GLM-5.3, MiMo-2.6 on the X99: Q3_K/Q2_K/Q2_0
experts, Q8_0 + BF16 attention): at MTP verify widths 3-5 the dense route contributes ~nothing
there either - Q8_0 routes only from width 6 (MMVQ wins below), BF16 is not a routed type, and the
experts are width-1 MMVQ-id. T1 is a dense-model win (Qwen3.8-27B UD-Q4_K_XL and any K-quant dense
model); for the MoE fleet the levers are the round overhead (F4) and expert bytes (#151), not this
kernel. A route-on/off bench on Flash-Next would show ~0 by construction; run it only if the user
wants the number on record.

### F4. GPU-resident speculative round (T2) - what exists, what remains

What exists in the fork: #140 fused in-graph draft chain (`LLAMA_SPEC_MTP_FUSED=1`): a single-seq
MTP round runs all n draft steps in ONE decode with in-graph argmax (the drafted ids come back as one
packed row), i.e. the per-draft-step decode round trips (~11.6 ms each on a V100 per its note) are
already gone when it is on. It is off unless the env is set, disabled for row/tensor split and
shared-memory drafters; none of today's Phase 0 or A/B legs set it, so every MTP number in this
document has the per-step host loop in it.

What remains per round with the chain on: (1) the verify decode; (2) the target logits crossing to
the host for `common_sampler_sample_and_accept_n` on n+1 rows (n+1 x vocab x 4 B = up to 3 MB at
n 4) and the host sampler chain per row; (3) the KV rollback of rejected positions (host-side memory
metadata); (4) the accepted token's hidden state (`pending_h`, n_embd floats) copied D2H then H2D as
the drafter's input; (5) the fused draft decode and its id row D2H. NInfer does 1-5 in one graph.

Plan: measure first, then build the contained pieces. Step 1 (no code): MTP n-max 2/3/4 with the
fused chain on x route on/off, with `LLAMA_SPEC_TIMING` (draft / ckpt / decode / accept ms per
round).

Step 1 result (2026-09-28 17:46-17:54, shipped image, 250 W, Phase 0 prompts, t/s): n-max 2: route on
46.6 fused / 46.5 unfused, off 53.0 / 52.0; n-max 3: on 51.3 / 51.7, off 48.7 / 48.6; n-max 4: on
52.7 / 53.2, off 44.7 / 45.6. **The fused chain changes nothing (within 1 t/s) at any n-max, with or
without the route**, confirming #140's own note: the draft phase is not the round's cost. The host
timer (per 64 rounds) reads draft 290-790 ms, decode 45-355 ms, accept 76-130 ms, but these are host
clocks around asynchronous launches and cannot place the GPU time; a per-round GPU timeline needs
CUDA events inside the decode/sample path (F4 instrument, to build before step 2). What the matrix
does show: the route's n-max-2 loss (-12%) is real and width-3-specific, consistent with the
prep/reduce launch overhead (~6 ms of a 45 ms step by the kernel timer) that F1 removes. Step 1b result, GPU-inclusive timeline (`LLAMA_SPEC_TIMING_SYNC=1`, 2026-09-28 18:18, per round =
per-64-iteration sums / 64): n-max 2: route on verify decode 50.4 ms, draft 4.6, accept 1.5; route
off verify 46, draft 5.0, accept 1.5. n-max 4: route on verify 56.5, draft 9.2 (2.3 ms per MTP step),
accept 2.0; route off verify 64.5, draft 10.3, accept 1.9. **The verify decode is 83% of the round;
draft + accept are ~11 ms at n-max 4**, so a fully GPU-resident round (NInfer-style) can return at
most ~12% here - real but second-order next to the verify kernel on the mixed-type model (route on
is still slower than off at width 3 on the mix: 50.4 vs 46 ms). F4 continues after the kernel work
(V5, per-type table); the contained piece stays as designed:

Step 2 (contained): backend sampling for the target rows when the sampler chain is
greedy/top-k (the fork already offloads the drafter's sampling with `llama_set_sampler`), so (2)
shrinks to n+1 token ids. Step 3 (engine): device-resident handoff of the accepted hidden state to
the drafter (removes (4)); fusing verify + draft into one graph needs the acceptance count on the
device to select the draft input, i.e. a device-side batch - the deep end, to be sized after steps
1-2 show what is left.

### 4.10 Roll 2 (2026-09-28 18:50): V5 + per-type table = `llamacpp-local-v100:1c63c03a1-smt2`

Gates on devbins-smt33 (same source as the image): comparator NMSE 5.4e-8 / 3.7e-14 (identical to
v11); KL vs the route-off base at ubatch 8 / 4 / 3 = 0.001655 / 0.001718 / 0.001844 mean KLD with
top-1 98.16 / 98.24 / 98.06 % (control 0.001775 / 97.99 %); oracle route-on 1179/1179; bench on the
real UD-Q4_K_XL mix, route on / off at widths 1..8 = **1.00 / 1.00 / 1.00 / 1.17 / 1.24 / 1.34 / 1.34
/ 1.33** (34.0/54.6/70.7/92.6/111.9/121.2/141.2/160.2 vs 34.1/54.6/70.9/79.4/90.5/90.5/105.5/120.1
t/s); kernel timer at width 5 on the routed tensors 493 GB/s; serve MTP n-max 2 / 3 / 4 route on
50.8 / 56.1 / 56.9 vs off 52.8 / 47.7 / 44.7 (the n-max 2 pair is inside the drift band; best point
56.9 vs the stock 52.8, +8%). Image verified to carry V5 (8 warps per CTA, 28 KB shared for Q4_K)
and the sync instrument; `:latest` re-pointed, X99 launcher recreated (no serve was loaded),
rollback `1c63c03a1-smt`, `e117ee884-widefix` pruned locally and on the X99.

## 5. Diagnostic ablations (small changes, run before/alongside D6)

Compile-time `MMSMT_ABLATE` (default 0; dev builds only, never in an image):
- E1 = 1: copy only — the staged pipeline runs, readers are skipped (accumulators stay 0, outputs
  wrong). Measures the memory pipeline ceiling of the current geometry.
- E2 = 2: compute only — the copy is skipped after the first piece, readers decode stale shared
  data. Measures the decode + HMMA ceiling.
Both are read from the timer line (`GGML_CUDA_SMT_TIME=1 GGML_CUDA_DISABLE_GRAPHS=1`, width 2 and 8,
Q4_K_S and Q8_0). Interpretation: E1 ~ 230 GB/s confirms H1; E1 >> E2 means compute-bound (H3);
E1 high and E2 high but the real kernel low means overlap failure (H2).

Result 2026-09-28 (v9 geometry, Q4_K_S): real 272 GB/s (w2) / 258 (w8); **copy-only 316 / 287**, compute-only 439 / 431; Q8_0 real 321 / 293, copy-only 338 / 305, compute-only 802 / 788.
The memory pipeline of the 128-rows-per-CTA, 64-byte-chunk geometry tops out at ~316 GB/s with no
compute at all: H1 confirmed, D6 goes ahead. (Remaining legs recorded in the plan-doc log.)

## 6. Gates (every version)

1. `test-backend-ops test -b CUDA0 -o MUL_MAT` 1179/1179 (oracle-smt.sh).
2. Width bench route off vs on, Q4_K_S and Q8_0 copies, `-p 1..32 -n 16 -b 512 -ub 512 -r 10`
   (bench-dev.sh); the timer leg for GB/s.
3. Before flipping the route default: KL gates at `-ub 4` and `-ub 8` vs the route-off base (band
   ~1.6e-3), a serve-level MTP ladder A/B, then the image roll (user call).

## 7. Out of scope for this step

`mul_mat_id` (MoE experts) sibling; T 9..16 register budget; NInfer-style GPU-resident spec round
(T2). Each gets its own section here before code.
