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

## 2. What exists (v9, `ggml/src/ggml-cuda/mmsmt.cu`, route opt-in via `GGML_CUDA_SMT=1`)

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
experiments. The route stays opt-in until the KL gate and a serve-level MTP A/B pass.

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
