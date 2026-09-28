# NInfer vs our fork on one V100: phase 0 measurements (TASKS #153)

Date: 2026-09-28. Box: X99, one Tesla V100 (PG500-216, 32 GB), driver 470-era host, CUDA 12.8
images. Our engine = `llamacpp-local-v100:e117ee884-widefix` (fork HEAD e117ee884 + the #151
wide-node fix). Their engine = `ninfer-local-v100:latest` (geoffwatts/ninfer-v100 b37d0dd3, the
user's build). Model class = Qwen3.8-27B dense: ours `Qwen3.8-27B-UD-Q4_K_XL.gguf` (16.34 GiB
on device, Q4_K medium mix) + `mtp-Qwen3.8-27B-Q4_0.gguf`; theirs `qwen3_8_27b_nvfp4.ninfer`
(NVFP4 MLP 0-55 + FP8 rest, ~19.7 GiB on device, native MTP layer). Raw logs:
`llama.cpp-work/153-p0/x99/`. Scripts: `bench-width.sh`, `serve-legs.sh`, `probe.py` (same dir).

The survey (`research/ninfer-v100-survey-2026-09-27.md`) predicted that plain decode is a tie
and the whole gap is the cost of a speculative round, with the small-T GEMM as the mechanism.
Phase 0 tests that on our own silicon.

## 1. Engine-level step cost vs token width (ours)

`llama bench -m Qwen3.8-27B-UD-Q4_K_XL -ngl 99 -fa on -b 512 -ub 512 -r 5`, one ubatch per
row, so `pp<T>` is the cost of ONE forward of T tokens (= a verify step of width T) and `tg16`
is the T=1 decode step. ms/step = T / (t/s) x 1000. "steps" = ms/step divided by the tg16 step.

| T | t/s | ms/step | steps | kernel route (Volta, Q4_K) |
|---:|---:|---:|---:|---|
| tg16 (T=1 decode) | 33.70 | 29.7 | 1.00 | MMVQ |
| pp1 | 32.20 | 31.1 | 1.05 | MMVQ |
| 2 | 51.72 | 38.7 | 1.30 | MMVQ |
| 3 | 66.60 | 45.0 | 1.52 | MMVQ |
| 4 | 74.22 | 53.9 | 1.82 | MMVQ |
| 5 | 84.43 | 59.2 | 2.00 | MMVQ |
| 6 | 88.57 | 67.7 | 2.28 | MMVQ |
| 8 | 87.99 | 90.9 | 3.06 | MMVQ (worst point of the curve) |
| 12 | 149.84 | 80.1 | 2.70 | MMQ dp4a |
| 16 | 198.05 | 80.8 | 2.72 | MMQ dp4a |
| 24 | 250.57 | 95.8 | 3.23 | MMQ dp4a |
| 32 | 277.10 | 115.5 | 3.89 | MMQ dp4a |
| 48 | 325.73 | 147.4 | 4.96 | MMQ dp4a |
| 63 | 351.27 | 179.3 | 6.04 | MMQ dp4a (last MMQ width on Volta) |
| 64 | 178.31 | 358.9 | 12.1 | dequant to fp16 + cuBLAS HMMA |
| 96 | 242.09 | 396.5 | 13.4 | cuBLAS |
| 128 | 316.20 | 404.8 | 13.6 | cuBLAS |
| 256 | 536.13 | 477.5 | 16.1 | cuBLAS |
| 512 | 710.25 | 720.9 | 24.3 | cuBLAS |

Readings:
- The T=1 step reads ~16.3 GiB in 29.7 ms = ~590 GB/s, 65% of the 900 GB/s HBM2 floor. That
  matches NInfer's own no-spec efficiency (survey 2.7): plain decode is a tie, as predicted.
- Verify width is NOT nearly free for us. NInfer's kernel origin quotes "+0.38 ms per extra
  draft row"; we pay +9 ms per row from 1 to 2, and width 5 (MTP n-max 4) costs 2.0 steps,
  width 8 costs 3.1 steps. The survey's kill rule ("if width 5 already costs <= 1.3 steps, T1
  is not the lever") fails: T1 (small-T tensor-core GEMM) IS the lever on this box.
- MMVQ collapses between 6 and 8 columns (pp8 has the same t/s as pp6), while the dp4a MMQ
  tile path at 12-16 columns is cheaper per step (80 ms) than MMVQ at 8 (91 ms). The MMVQ /
  MMQ hand-over at `MMVQ_MAX_BATCH_SIZE = 8` is mis-tuned for Volta at widths 6-8 (see 1.1).
- Above 63 columns Volta leaves MMQ for the dequant-to-fp16 + cuBLAS route (mmq.cu: fp16 MMA
  hardware without int8 MMA -> `ne11 < 64`), and the step cost DOUBLES at the boundary
  (179 -> 359 ms). The whole-weight dequant pass (16 GiB read, 32 GiB fp16 written and read
  back) is the fixed tax the survey attributed to NInfer's CUTLASS route; we pay it too.

### 1.1 Where the width cost comes from (code)

`ggml_cuda_mul_mat` (ggml-cuda.cu:1873) dispatch for a quantized weight on Volta: MMVQ while
`ne11 <= MMVQ_MAX_BATCH_SIZE` (8; `ggml_cuda_should_use_mmvq` has per-type limits only for
CDNA), then MMQ dp4a tiles while `ne11 < 64`, then dequant + cuBLAS. The MMVQ launch geometry
comes from `calc_nwarps` / `calc_rows_per_block` (mmvq.cu): the `MMVQ_PARAMETERS_VOLTA` table
was tuned for `ncols_dst == 1` only (comment: "multi-column paths keep the GENERIC values",
i.e. nwarps 4 for 2-4 columns, 2 for 5-8, rows_per_block 2) - the verify-width regime was never
tuned on sm_70. The fused gate/up path (`ggml_cuda_should_fuse_mul_mat_vec_q`, ggml-cuda.cu:1853)
uses the same 8-column limit.

## 2. Prefill ubatch (survey item T5) and clocks

Same bench, `-b 2048`, prompt 2048 and 8192 tokens:

| ubatch | pp2048 t/s | pp8192 t/s |
|---:|---:|---:|
| 512 (our production default) | 705.6 | 662.3 |
| 2048 | 870.5 (+23%) | 771.3 (+16%) |

Config-only win, consistent with the survey's reading (fewer dequant passes on the cuBLAS
route). NInfer's pp2048 on the same GPU class is ~1,040-1,170 t/s (survey 2.1-2.3), so
`-ub 2048` closes about half of the prefill gap; the rest is the dequant tax itself (their
smem-dequant Volta MMA GEMM) and fused projections.

SM clock during the bench: 1035-1380 MHz (sampled every 5 s; application clock 1260 MHz, max
1380). JimmyMax locked 1380 with `nvidia-smi -lgc 1380,1380` and measured -10% unlocked; the
lock needs root on the X99 (sudo asks for a password from this session) = user action.

## 3. Our MTP round on the server (same prompts the NInfer legs get)

`llama serve` from the widefix image, `-ngl 99 -fa on -c 8192 -np 1`, MTP head
`mtp-Qwen3.8-27B-Q4_0.gguf` on the GPU, greedy, 6 prompts x 200 tokens (code, prose, math, list,
code2, story), `timings` from the server. Rounds = draft_n / n-max (drafts are padded to n-max).
All outputs were read: coherent, and byte-identical across the M legs on the same prompt.

| leg | t/s (6-prompt mean) | best / worst prompt | acceptance | tok/round | ms/round | = T=1 steps | bench verify width n+1 | draft + host per round |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| T0 target-only | 32.7 | 33.5 / 31.1 | - | 1.00 | 30.6 | 1.03 | 29.7 (w1) | 0.9 |
| M1 n-max 1 | 44.4 | 47.2 code / 39.3 story | 66-96% | 1.87 | 42.1 | 1.42 | 38.7 (w2) | 3.4 |
| M2 n-max 2 | 48.0 | 53.0 / 38.6 | 53-89% | 2.54 | 52.8 | 1.78 | 45.0 (w3) | 7.8 |
| M3 n-max 3 | 45.1 | 54.8 / 31.6 | 36-84% | 2.97 | 65.9 | 2.22 | 53.9 (w4) | 12.0 |
| M4 n-max 4 | 42.6 | 54.7 / 29.3 | 31-78% | 3.20 | 75.2 | 2.53 | 59.2 (w5) | 16.0 |
| N2 ngram-mod + n-max 2 | 48.7 | 52.9 / 38.3 | same as M2 | - | - | - | - | (n-gram never fires on fresh text) |

Readings:
- The verify step IS the round: 38.7 of 42.1 ms at n-max 1, 59.2 of 75.2 ms at n-max 4. The
  bench width curve (section 1) predicts the server round to within the draft cost.
- Each draft token adds ~4 ms on top: one MTP-head forward (a 1-layer draft + the full LM
  head, ~1 ms of weight bytes) plus the host round trip (D2H of the logits, host argmax,
  batch rebuild) and the host-side acceptance over width+1 rows of a 248k vocab. That is the
  survey's T2 item (GPU-resident round); it is the smaller of the two terms.
- Because a width-5 verify costs 2.0 steps, n-max 4 needs >50% acceptance just to break
  even, and the story prompt (31%) drops BELOW target-only (29.3 vs 32.7). The ladder peaks
  at n-max 2 (48.0), exactly where the earlier X99 tuning landed. NInfer's ladder on real
  prompts peaks at K=3-4 because their width-5 verify costs ~1.1-1.3 steps.
- `LLAMA_SPEC_TIMING` / `LLAMA_DECODE_TIMING` are host-side clocks: CUDA launches return
  before the GPU finishes, so their "decode = 2 ms/iter" is launch time and the GPU wait
  lands in the next phase. They cannot decompose the round; the bench arithmetic above can.

## 4. Cheap knobs: MMVQ launch geometry and the MMVQ/MMQ hand-over on Volta

Dev builds of `build-cuda75` (one-line variants of mmvq.cu) run inside the widefix image on the
X99 (`LD_LIBRARY_PATH=/devbins`). Round 1 = 5 reps (12 s per run, +-10%); round 2 = 25 reps,
interleaved with two baselines to expose drift. pp<T> t/s, higher is better (ms/step = T/t/s).

| variant | change (Volta table only) | pp2 | pp3 | pp4 | pp5 | pp6 | pp8 | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| mb (baseline, opening) | none | 53.4 | 67.6 | 75.5 | 85.4 | 89.7 | 89.0 | |
| mb (baseline, closing, 3 min later) | none | 52.3 | 66.3 | 74.1 | 83.7 | 86.1 | 83.0 | -2..-7% = clock drift under load |
| v1 | 5-8 cols: nwarps 2 -> 4 | 52.3 | 66.3 | 74.0 | 79.4 | 80.7 | 77.9 | worse |
| v2 | 2-8 cols: nwarps 8 | 46.7 | 55.5 | 60.7 | 65.5 | 69.6 | 49.2 | much worse |
| v3 | 2-8 cols: rows_per_block 2 -> 1 | 48.9 | 59.6 | 65.4 | 71.7 | 74.0 | 78.1 | worse |
| v5 | 5-8 cols: nwarps 2 -> 1 | 52.4 | 66.3 | 74.1 | 82.4 | 87.7 | 90.1 | tie (inside drift) |
| v4 | dense MMVQ only up to 4 cols; 5-8 -> MMQ | 52.5 | 66.5 | 74.3 | 77.2 | 92.5 | **122.0** | MMQ wins from 6-7 up |

Readings:
- The existing Volta MMVQ geometry (nwarps 4 for 2-4 columns, 2 for 5-8, two rows per
  block) is already the best of the tried launch shapes. The multi-column MMVQ kernel itself
  is the limit at widths 2-6 (1.3-2.3 steps); no launch-geometry knob fixes that. That is the
  T1 (small-T tensor-core GEMM) territory.
- The MMVQ -> MMQ hand-over is mis-placed on Volta: MMQ at width 8 runs 122 t/s (65.6 ms per
  step = 2.2 steps) against MMVQ's 89 (89.9 ms = 3.0 steps), +37%; width 7 +22% (round 1);
  width 6 is a tie; width 5 loses 9%. So the right rule for sm_70 is MMVQ up to 6 columns,
  MMQ from 7 (variant v6, round 3 below). Widths 7-8 are exactly the DFlash2 / n-gram /
  MTP n-max 6-7 verify shapes, so this is a real serve-level gain for those stacks, for free.
- GPU clocks: the X99 V100 runs at its 1260 MHz application clock under sustained load (max
  1380) and the closing baseline reads 2-7% below the opening one after ~3 minutes; every
  variant comparison above is interleaved for that reason.

### 4.1 Round 3: placing the hand-over (25 reps, interleaved v6 / mb / v6 / v4)

| width | mb (MMVQ <= 8) | v6 (MMVQ <= 6, MMQ 7+) | v4 (MMVQ <= 4, MMQ 5+) |
|---:|---:|---:|---:|
| 4 | 74.2 | 74.6 / 74.2 | 74.2 |
| 5 | 84.2 | 84.5 / 84.0 | 77.2 (MMQ, -8%) |
| 6 | 88.2 | 88.5 / 88.0 | 92.3 (MMQ, +5%) |
| 7 | 86.0 | 107.0 / 106.8 (MMQ, +24%) | 106.5 |
| 8 | 86.9 | 122.1 / 121.8 (MMQ, +41%) | 120.1 |

Chosen rule (variant v8, applied to the working tree in `ggml_cuda_should_use_mmvq`,
mmvq.cu): on `cc == GGML_CUDA_CC_VOLTA` use MMVQ only up to 5 columns; 6 and wider go to the
dp4a MMQ tiles. Expected step costs at widths 5/6/7/8: 59/65/65/66 ms (2.0/2.2/2.2/2.2
steps) instead of 59/68/81/92 ms. Gate = KL at `-ub 8` against the image's `-ub 8` logits
(section 4.3) plus one confirmation bench of v8 (section 4.2).

### 4.2 Round 4: the cuBLAS cliff (MMQ everywhere on Volta, variant v7, ubatch 512, 5 reps)

| width | mb (MMQ < 64, then dequant + cuBLAS) | v7 (MMQ for every width) |
|---:|---:|---:|
| 48 | 323.6 | 323.7 (same kernel) |
| 64 | 176.2 | 356.9 (+103%) |
| 96 | 242.8 | 384.0 (+58%) |
| 128 | 316.2 | 384.2 (+22%) |
| 192 | 421.3 | 393.4 (-7%) |
| 256 | 531.4 | 403.1 (-24%) |
| 512 | 701.0 | 416.1 (-41%) |
| 2048 (ubatch 2048) | 870.5 (section 2) | ~406 |

Reading: the dp4a MMQ tile saturates at ~405-416 t/s on this 27B (Volta dp4a is
compute-bound there), while the dequant + cuBLAS HMMA route keeps climbing with width and only
overtakes MMQ at ~160 columns. So `MMQ_DP4A_MAX_BATCH_SIZE = 64` is mis-placed for sm_70 by
2.5x: widths 64-159 pay the whole-weight dequant tax for nothing. Variant v9 = v8 + `ne11 <
160` on Volta in `ggml_cuda_should_use_mmq` (mmq.cu); confirmation bench in 4.4. Where it
matters: any ubatch of 64-159 tokens (short-prompt prefill, the tail chunk of every prompt,
wide tree/n-gram verify), not the 512/2048 prefill steady state, which stays on cuBLAS. It
also says something about the prefill gap to NInfer: at 2048 our cuBLAS route reaches 870
t/s vs their ~1,100; the remaining 25% is their smem-dequant Volta MMA GEMM (no global fp16
scratch pass) plus fused projections - a kernel lane, not a knob.

### 4.3 KL gate for the MMVQ -> MMQ hand-over (dense 27B, wikitext 16 x 512, `-fa on`)

Base = image binary at `-ub 8` (MMVQ at width 8), PPL 7.3893.

| leg | binary | width | mean KLD | same-top | PPL ratio | meaning |
|---|---|---:|---:|---:|---:|---|
| ctrl16 | image | 16 (MMQ) | 0.001650 | 98.26% | 1.0004 | the stock MMQ-vs-MMVQ rounding band |
| img-ub4 | image | 4 (MMVQ) | 0.000427 | 99.12% | 1.0014 | the MMVQ column-count band |
| v6-ub8 | dev v6 | 8 (now MMQ) | 0.001825 | 98.21% | 1.0010 | = ctrl16 band: quality-neutral |
| v6-ub4 | dev v6 | 4 (MMVQ) | 0.000427 | 99.12% | 1.0014 | digit-identical to img-ub4: untouched path untouched |

PASS: the hand-over only swaps one validated kernel for another (the same MMQ that already
serves widths 9-63), at the fp-reordering level (~2e-3, threshold band per the quality
battery), with no change on the widths that keep MMVQ.

## 5. Head-to-head on one GPU: NInfer vs our fork, same six prompts, greedy, 200 tokens

`ninfer-serve` (user's build of geoffwatts b37d0dd3, nvfp4 artifact, `--kv-dtype int8
--max-context 8192 --prefill-chunk 2048 --lm-head-draft --model-id default`), `timings` from
its server (predicted_per_second = (n-1)/predicted_ms, i.e. the same definition as ours within
0.5%). Outputs read: coherent on both engines; wording differs (different quantisation), structure
identical (same headings, same code shape). Rounds = draft_n / K.

| engine / mode | t/s (6-prompt mean) | best / worst | acceptance | tok/round | ms/round | = own T=1 steps |
|---|---:|---|---:|---:|---:|---:|
| NInfer no-spec | 30.0 | 30.7 / 28.3 | - | 1.00 | 33.3 | 1.00 |
| ours target-only | 32.7 | 33.5 / 31.1 | - | 1.00 | 30.6 | 1.03 |
| NInfer MTP K=3 | 71.3 | 85.1 math / 55.4 story | 41-82% | 2.91 | 40.9 | 1.22 |
| ours MTP n-max 3 | 45.1 | 54.8 / 31.6 | 36-84% | 2.97 | 65.9 | 2.22 |
| NInfer MTP K=4 | 69.6 | 84.7 / 50.4 | 31-71% | 3.15 | 45.2 | 1.35 |
| ours MTP n-max 4 | 42.6 | 54.7 / 29.3 | 31-78% | 3.20 | 75.2 | 2.53 |
| ours best (n-max 2) | 48.0 | 53.0 / 38.6 | 53-89% | 2.54 | 52.8 | 1.78 |

Readings:
- Plain decode: we are 9% FASTER (30.6 vs 33.3 ms per token; our Q4_K_XL reads ~16.3 GiB per
  step, their NVFP4+FP8 artifact ~19.7 GiB, both at ~60-65% of HBM2). Their prefill on these
  short prompts is ~440 ms for the code prompt vs ours (not compared here; section 2 has the
  bench numbers).
- Same acceptance, same tokens per round: at K=3 both engines commit ~2.9-3.0 tokens per
  round. The entire 1.5-1.6x gap is the round cost: 41 ms vs 66 ms at K=3, 45 vs 75 at K=4.
- Where our extra 25-30 ms per round goes (from sections 1 and 3): +24 ms is the verify
  width (a 4-column step costs 53.9 ms against 29.7 for one column; NInfer's whole K=3 round,
  verify + 3 drafts + acceptance, adds only 7.6 ms over its single-token step) and +12 ms is
  three draft steps with host round trips (their in-graph drafting adds ~1-2 ms each).
- Consequence for the ladder: their marginal cost per extra draft row is 4.3 ms (K=3 -> 4),
  ours 9.3 ms, so their MTP stays profitable down to ~30% acceptance (story: 50.4 t/s = 1.7x
  their step) while ours breaks even near 50% and loses below it (story at n-max 4: 29.3 <
  32.7). That is why our production ladder peaks at n-max 2 and theirs at K=3-4.

## 6. Verdict and what to build (phase 0 answers the survey's decision points)

1. **T1 - small-T tensor-core GEMM on sm_70 is the lever, confirmed.** Kill rule failed by a
   wide margin: width 5 costs 2.0 steps, not <= 1.3. Physics: a 4-column verify reads the same
   16 GiB as a 1-column step; NInfer's m8n8k4 quadpair kernel makes it ~1.1 steps. Expected
   gain on our fork if the width-2..8 step drops to ~1.1-1.3 steps: the K=3 round goes
   66 -> ~40 ms, i.e. 45 -> ~70 t/s on these prompts at 1x V100, before touching T2. It also
   moves the MoE cache chain (#151) and every drafter (MTP, DFlash2, n-gram). Scope as the
   survey said: a new `mul_mat` route for `cc == 700 && 2 <= ne11 <= ~32`, Q4_0/Q8_0 first,
   then Q4_K, then the `mul_mat_id` sibling; oracle test-backend-ops + this doc's bench + KL.
2. **T2 - GPU-resident spec round is the second term (12 of 30 ms at K=3).** Draft steps
   plus host acceptance/sampling cost ~4 ms per draft token; NInfer's in-graph round ~1-2 ms.
   Worth doing after T1 (or together: the #140 fused chain already removes part of the draft
   host trips on 1-GPU shapes).
3. **Two cheap tunes, gated, in the working tree now (9 lines, mmvq.cu + mmq.cu):**
   (a) Volta MMVQ only up to 5 columns, MMQ from 6: widths 6/7/8 +4/+24/+40% (section 4);
   KL at width 8 inside the MMQ band, width 4 bit-identical. Pays on every 6-8-wide verify
   (DFlash2 k=7, n-gram drafts to 8, MTP n-max 5-7) and on the #151 chain at those widths.
   (b) Volta MMQ up to 160 columns instead of 64: widths 64/96/128 +103/+58/+22% (section
   4.2); confirmation bench + KL at width 128 in 4.4. Pays on short-prompt prefill and the
   tail ubatch of every prompt.
4. **T5 config items:** `-ub 2048` = +23% prefill at 2048 tokens (section 2), free; SM clock
   lock needs root on the X99 (user); the 64-column Volta FA tile was not measured (prefill
   attention is not the bottleneck at these lengths).
5. **T3 (INT8 Hadamard KV + int8 MMA attention) and T4 (launch fusion)** were not exercised
   here (short contexts); the survey's ranking stands, and both come after T1/T2.
6. **Not the reason:** the artifact format, NVFP4, the hand-written schedule, CUDA graphs for
   plain decode (we tie there), or their prefill attention (it is our kernel).

Artifacts: `llama.cpp-work/153-p0/` (scripts, `x99/` raw logs and probe JSONs with full
outputs, `mmvq.cu.orig` / `mmq.cu.orig` pristine copies, `variants.py` for the sweep).

### 4.4 Round 6: the MMQ limit at 160 (variant v9 = v8 + `ne11 < 160`, 5 reps, interleaved mb / v9 / mb / v9)

| width | mb (MMQ < 64) | v9 (MMQ < 160) |
|---:|---:|---:|
| 96 | 240.4 / 240.3 | 392.4 / 392.4 (+63%) |
| 128 | 312.6 / 312.3 | 396.0 / 395.9 (+27%) |
| 160 | 359.1 / 354.5 | 362.0 / 359.3 (cuBLAS on both) |
| 192 | 418.3 / 409.4 | 420.2 / 413.9 (same) |
| 256 | 526.2 / 509.7 | 522.9 / 507.6 (same) |

The exact crossover is ~170-185 columns (MMQ ~400 flat vs cuBLAS 360 at 160, 418 at 192);
160 is a safe cut. KL gate for the new MMQ region at width 128: see the kl9 log
(`llama.cpp-work/153-p0/x99/kl9/`), appended below when done.

KL gate for v9 (base = image at `-ub 128`, cuBLAS route, PPL 7.3879): control image `-ub 48`
(MMQ) = 0.001574 / 98.19%; **v9 `-ub 128` (now MMQ) = 0.001624 / 97.97%** = the same
MMQ-vs-cuBLAS band; v9 `-ub 512` (still cuBLAS) = 0.000002 / 100.00% = untouched path
untouched. PASS. Both tunes (v8 rule + the 160 limit = the working-tree diff) are gated.
