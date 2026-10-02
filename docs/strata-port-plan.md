# Strata port plan (TASKS #154) - MoE decode on the X99

Status: **DONE 2026-09-30 - PORTED, ROLLED, COMMITTED.** Item 1 (AVX2 Q2_0 CPU kernel, section 6), item 3 (the D2
MoE doorbell, sections 7 and 10) and the prefill stream ring (section 8.6) are in the official tree (committed by the
user as 063bdc2a0 + 1300d2bef), image `llamacpp-local-v100:063bdc2a0-db` = `:latest`, and the saved Flash-Next config
runs them with `-ub 4096 -b 4096` and CUDA graphs off. Result vs the 2026-09-29 production (section 9.1): decode +43 %
(94.6 -> 64.0 ms per MTP step), prefill 4.1x. Items 2, 3b, 4, 5, 6 (`-ub 8192`), 7a, 7b closed with numbers (8.1-8.6).
Filed follow-ups (not started): stable CUDA graphs (KV padding, 8.3), gather-based sparse attention for long contexts
(8.5), a CJK-aware draft vocabulary subset (8.2), a K80 gate before the Kepler image uses the doorbell.

Source of the ideas: Niko1221/Strata @ a790805 (`src/kernels/cpu/q2_avx2.cpp` for item 1). Background: TASKS #154
(two-round analysis) and #155 (the isolated experiment that measured the decode budget).

## 1. Where the decode time goes (verified 2026-09-30)

Serve: X99 (2x Xeon E5-2690 v4 = 28 cores / 56 threads, AVX2 + FMA + F16C, no AVX-512; one V100 32 GB), Flash-Next
GSQ-RCO 3.5bit, wizard config 55366197782 (`-ngl 99 -ncmoe 48`, MoE cache 15000 MiB, MTP head n-max 3, `-t 40`).

- **Budget of a plain token (#155 section 8.7, speed-of-light runs):** ~49.5 ms = GPU ~35.6 ms (~72 %) + per-layer
  GPU->CPU->GPU handoff ~9 ms (~18 %) + CPU work on missed experts ~5 ms (~10 %, 87 % cache hits).
- **Expert types (the GGUF header, parsed):**

| type | share of expert params | share of expert bytes | where |
|---|---|---|---|
| q3_K | 50.0 % | 59.9 % | gate + up of 36 layers |
| **q2_0** | **43.1 %** | 33.8 % | **every** `ffn_down_exps` ([640 -> 2560]: rows of 640 = 10 blocks) + gate/up of 7 layers |
| q2_K | 6.9 % | 6.4 % | gate + up of 5 layers |

  Dimensions: n_embd 2560, expert FFN 640, 512 experts, top-10, 48 layers; one expert = 3 x 2560 x 640 = 4.9 M weights.
- **Our x86 Q2_0 kernel is the scalar one:** the x86 block of `ggml/src/ggml-cpu/arch-fallback.h` maps
  `ggml_vec_dot_q2_0_q8_0_generic` to the public name (ARM has a NEON kernel in `arch/arm/quants.c`). Upstream master
  (fetched 2026-09-27) is the same. The type came from upstream PR #24448.
- **Kernel speed on the X99** (`llama.cpp-work/154/q2bench.c`: `ggml_get_type_traits_cpu(type)->vec_dot` over the
  Flash-Next expert shapes, rows replicated over 512 experts so the weights stream from DRAM; G weights/s, "4 tokens"
  = each row dotted with 4 activation rows, as in a 4-row MTP verify step):

| kernel | 1 thread, 1 token | 40 threads, 1 token | 40 threads, 4 tokens |
|---|---|---|---|
| q2_0 (scalar), down K=640 | 1.9 | **39.2** (11 GB/s: compute-bound) | 10.6 |
| q3_K (AVX2), gate K=2560 | 10.7-12.9 | 136.7 (59 GB/s: memory-bound) | 67.5 |
| q2_K (AVX2), gate K=2560 | 16.8-19.6 | 179.9 (59 GB/s) | 113.1 |
| q4_0 (AVX2), K=640 | 11.1 | 103.2 (58 GB/s) | 57.4 |

  Every AVX2 type reaches the same ~59 GB/s wall at 40 threads; the scalar Q2_0 reaches 11 GB/s. Per weight it is
  3.5x slower than q3_K and 4.6x slower than q2_K at 40 threads with 1 token, 6.4x / 10.7x at 4 tokens.
- **The budget model checks out:** a plain token misses ~62 experts (13 % of 10 x 48) = ~306 M weights. At the
  40-thread rates: q2_0 132 M / 39.2 G = 3.4 ms, q3_K 153 M / 136.7 G = 1.1 ms, q2_K 21 M / 180 G = 0.1 ms: **~4.6 ms
  predicted vs ~5 ms measured**. Q2_0 is ~73 % of the CPU miss time at 1 token and ~84 % at the 4-row verify width.

## 2. Expected gains, and the item order

- **Item 1 (AVX2 Q2_0):** at the ~59 GB/s wall Q2_0 streams ~210 G weights/s (5.4x the scalar). Plain decode: misses
  ~4.6 -> ~1.9 ms, **about -2.7 ms per token (~+5 %)**. MTP decode (the production config): a verify step runs 4 rows,
  so the miss work is up to ~4x a plain token's and Q2_0 is ~84 % of it; expected **-8 to -12 ms per verify step,
  +10-15 % decode**. Falsifier, decided up front: under +3 % on the MTP decode A/B means the model above is wrong -
  stop and profile before anything else.
- **Item 3 (doorbell handoff):** removes up to ~9 ms per step, plain and MTP alike (the handoff is paid per step, not
  per row). Similar size under MTP, but a scheduler/graph redesign; it gets its own design after item 1.
- **Item 2 (multi-token CPU kernels):** the 4-token column above is its target. Item 1 already lifts that column;
  whether a kernel that unpacks once for 4 rows is worth more is decided on item 1's numbers.
- Order: item 1 first (small, contained, upstreamable, largest expected MTP gain), then item 3, then item 2.

## 3. Item 1 design - AVX2 `ggml_vec_dot_q2_0_q8_0`

Files: `ggml/src/ggml-cpu/arch/x86/quants.c` (new function next to `ggml_vec_dot_q1_0_q8_0`) and
`ggml/src/ggml-cpu/arch-fallback.h` (drop the x86 `q2_0` define). Nothing else: `vec_dot_type` stays Q8_0 (no change
for other architectures or backends), `nrc` stays 1, and x86 builds without AVX2 keep the generic kernel (the
function calls it in its `#else` branch, like `ggml_vec_dot_tq2_0_q8_K`). No env gate: A/B is done by swapping
`libggml-cpu.so` (section 5).

Q2_0 block: fp16 `d` + 16 bytes; value i sits in byte i/4, bits 2(i%4); code c in 0..3 means c - 1 in {-1, 0, 1, 2}.
One Q2_0 block (64 values) pairs with two Q8_0 blocks (32 values each, own scale). Per block:

1. **Codes in value order (Strata's unpack, lanes split up front).** Broadcast the 16 code bytes to both 128-bit
   lanes and `vpshufb` so lane 0 holds bytes 0-3 and 8-11, lane 1 bytes 4-7 and 12-15. Planes
   `c_s = (v >> 2s) & 3` (16-bit shifts; the mask keeps only in-byte bits). `u = unpacklo_epi8(c0, c1)`,
   `w = unpacklo_epi8(c2, c3)`; then `unpacklo_epi16(u, w)` = values 0-31 (lane 0: bytes 0-3, lane 1: bytes 4-7)
   and `unpackhi_epi16(u, w)` = values 32-63. 5 shuffles per 64 values instead of the 10 of a 128-bit unpack.
2. **Offset:** `codes - 1` as int8 (`sub_epi8`), so the products are exact signed products.
3. **Dot:** `mul_sum_i8_pairs_float(codes - 1, y)` per 32 values - the q8_0 kernel's helper (sign trick + maddubs +
   madd, or `dpbssd` with AVX-VNNI-INT8) - then `acc = fmadd(d_x * d_y, sums, acc)` per Q8_0 block, `hsum_float_8`
   at the end. Bounds: |c - 1| <= 2, |y| <= 127, pair sums <= 508: no int16 saturation.

Numerics: the integer products are identical to the generic kernel's; only the float accumulation order differs
(per-lane partial sums), as with every SIMD kernel in the file: expect ~1e-7 relative difference.

## 4. Gates for item 1

- **G0 build:** `build-cpu` and `build-cuda75` compile clean.
- **G1 unit (local + X99):** `test-quantize-fns` passes (vec_dot error vs the float reference, existing
  thresholds); q2bench: the new kernel vs a double-precision dot of the dequantized operands, and vs a copy of the
  generic kernel, both <= 1e-6 relative on 256 rows x 4 activation rows.
- **G2 op (X99):** `test-backend-ops -o MUL_MAT` and `-o MUL_MAT_ID` for q2_0, CUDA0 vs the new CPU kernel: all pass.
- **G3 model (X99):** Flash-Next with every expert computed on the CPU (`--no-op-offload`, cache off), a few 512-token
  chunks: KLD of the new CPU library vs the base library ~1e-6 (float order only), same top token >= 99.5 %.
- **G4 speed (X99):** q2bench before/after (1 and 40 threads, 1 and 4 tokens); server decode A/B at the production
  shape, MTP on, three prompts (code / engine / history), arms interleaved in both orders, two runs each; plain
  decode (no MTP) A/B; one long prompt to confirm prefill is unchanged (its experts run on the GPU). Read the
  outputs, not only the t/s.

## 5. Harness

- `llama.cpp-work/154/q2bench.c` (= `/work/154` in `llama-devcuda`): built against `build-cpu`.
- X99: `/home/anyei/bench154/{base,new}` (q2bench + libs); server A/B bins in `/home/anyei/devbins154` (the
  `build-cuda75` bin dir) with `cpu-base/` and `cpu-new/` holding the two `libggml-cpu.so.0` builds, selected by
  `LD_LIBRARY_PATH`; run inside the rolled image (a9885783c-t3k3) read-only, only while no launcher serve is loaded.

## 6. Results (2026-09-30, X99)

Build: the kernel of section 3 in the official tree (uncommitted); `build-cuda75` bins shipped to
`X99:/home/anyei/devbins154`, A/B by swapping `libggml-cpu.so.0` only (base md5 52fc2cd6, new 030687de); rolled image
a9885783c-t3k3 as the runtime. Scripts: `X99:/home/anyei/bench154/gate154{,b,c}.sh`.

- **G0 build:** clean (build-cpu, build-cuda75).
- **G1 unit:** `test-quantize-fns` q2_0 dot product error 0.140698 (scalar kernel: 0.140697), 0 failures; q2bench
  error vs the double-precision reference 2.1e-8 relative (scalar 1.3e-8), vs the scalar kernel 2.2-2.8e-8.
- **G2 op:** `test-backend-ops` CUDA0 vs CPU, `type_a=q2_0`: MUL_MAT 45/45, MUL_MAT_ID 75/75, with both libraries.
- **G4 kernel speed (q2bench, G weights/s, weights in DRAM):**

| case | scalar | AVX2 | x |
|---|---|---|---|
| 1 thread, 1 token (down, K=640) | 1.90 | 11.46 | 6.0 |
| 1 thread, 4 tokens | 0.48 | 2.96 | 6.2 |
| 40 threads, 1 token | 39.2 | 194.1 (54.6 GB/s: at the memory wall, q3_K 58.6) | 5.0 |
| 40 threads, 4 tokens | 10.6 | 53.9 (15 GB/s: compute-bound) | 5.1 |

  The budget model of section 1 then predicts plain-token misses ~4.6 -> ~1.9 ms.
- **G4 server decode, MTP (production shape, `--moe-cache 15000`, n-max 3; two interleaved rounds, 4 runs per arm
  for base and fast, 2 for the exact-order variant).** The t/s of the two arms is NOT a clean kernel measure here: the
  fast kernel changes the greedy text (G3 below), and the new texts happened to accept more drafts (code 74.5 ->
  77.7 %, engine 53.6 -> 63.3 %, history 39.5 -> 43.9 %). Time per verify step (predicted ms / (tokens - accepted
  drafts); drafts per step 2.96-2.97 in every arm) is the kernel measure:

| prompt | base t/s | fast t/s | base ms/step | fast ms/step | exact-order ms/step |
|---|---|---|---|---|---|
| code | 33.6-34.7 | 37.5-39.4 | 92.4 | 85.2 (**-7.8 %**) | 85.6 (-7.3 %) |
| engine | 26.1-27.1 | 31.9-33.7 | 97.3 | 87.9 (**-9.7 %**) | 89.1 (-8.4 %) |
| history | 22.9-23.4 | 26.3-27.6 | 93.8 | 85.6 (**-8.8 %**) | 86.6 (-7.6 %) |

  **Kernel gain at equal acceptance: -8.3 ms per verify step = +8-11 % decode** (section 2 predicted -8 to -12 ms,
  +10-15 %: low end; the falsifier did not fire). The t/s gains of the fast arm (+13.5 / +22.6 / +14.9 %) include the
  acceptance luck of its texts. Plain decode (no MTP, 1 token per step, no acceptance effect): base 19.36 / 19.01 /
  20.08 and 20.28 / 19.85 / 20.42, fast 20.44 / 20.13 / 20.60 and 20.60 / 20.19 / 20.81 = **+3 to +6 %** (predicted
  ~+5 %). Prefill (3706-token prompt, experts on the GPU) unchanged: 79.9-81.8 t/s in every arm. Cache hit rate
  unchanged (85.3 % MTP, 88.0-88.4 % plain). This harness' base reads lower on the engine prompt than the #155
  baselines (26.4 vs 29.1-29.8 t/s; other build and flags); every comparison here is within one harness.
- **G3 model - the fast kernel is NOT bit-identical, and the divergence is rounding only:** KLD vs the scalar
  kernel, all experts on the CPU (`--no-op-offload`, cache off), 3 x 512 tokens:

| comparison | mean KLD | max KLD | same top token | PPL ratio |
|---|---|---|---|---|
| scalar vs scalar (determinism control) | 0.000000 | 0.000048 | 100.000 % | 1.000255 |
| exact-order AVX2 variant vs scalar | 0.000000 | 0.000048 | 100.000 % | 1.000255 |
| **fast AVX2 kernel vs scalar** (two runs, identical) | **0.017244** | 0.575993 | **96.471 %** | 1.0057 +- 0.0078 |

  The exact-order variant (test-only build: same unpack and integer products, then the scalar kernel's float math
  per block) reproduces the scalar results bit for bit, so the unpack, offsets and integer sums are right; the
  whole divergence of the fast kernel comes from summing float partials in a different order (per-lane FMAs with
  d_x*d_y, then a horizontal sum). Both kernels are equally close to the double-precision dot product (~2e-8);
  this model turns ~1e-7 rounding differences into different expert choices, and the texts then drift apart
  (greedy outputs share their first 71-1039 characters). Decode outputs of both arms read coherent.
- **Noise floor of this model:** the same run with the experts computed on the GPU (op offload on: the path of
  every prefill and every cache hit) vs the CPU scalar base: **mean KLD 0.018697, same top 95.686 %**, PPL ratio
  0.9877 +- 0.0088. The fast kernel's divergence (0.0172) is the size of the GPU/CPU difference the serve already
  mixes on every token.
- **The exact-order variant as a shippable option:** bit-identical to the scalar kernel (KLD 0.000000, greedy decode
  texts IDENTICAL on all three prompts), but slower than the fast kernel: q2bench 7.4 / 140 / 35.6 G weights/s
  (1 thread / 40 threads 1 token / 40 threads 4 tokens) vs 11.5 / 194 / 54; decode -7.4 ms per verify step vs -8.3.

**Choice (user pick):** (A) the fast kernel - the ggml convention (every x86 SIMD kernel differs from its scalar
version in float order), upstreamable as is, ~1 ms per step faster and 1.4x faster as a kernel (matters more for
item 2's compute-bound 4-token case); outputs drift from today's at the model's noise floor. (B) the exact-order
kernel - outputs stay byte-identical to today's, ~90 % of the decode gain. Recommendation: (A).

**PICKED 2026-09-30 (user: "alright let's go with A"): the fast kernel (A)** - the version in the tree; the
exact-order form stays a recorded option only.

### 6.1 Rolled (2026-09-30, user: "yes to all three, go ahead")

- **Image** `llamacpp-local-v100:99b3c21ee-q2avx2` (sha256:9020e764; working tree = HEAD 99b3c21ee + this kernel)
  built, pushed; `:latest` re-pointed locally and in the registry (rollback `a9885783c-t3k3`). X99 pulled it and its
  launcher was recreated on the pinned tag (compose `/home/anyei/server/services/llamacpp-v100/`, defaults): healthy,
  wizard served, 23 models, 27 saved configs, `/wizard/dirs` and `/wizard/hw` intact.
- **Saved config 55366197782** (Flash-Next GSQ-RCO): `GGML_CUDA_DISABLE_GRAPHS` removed from its `gateOff`, so the
  MTP-head template's `GGML_CUDA_DISABLE_GRAPHS=1` applies again (+4.5 % measured, section 7.1); `NO_PAD` stays off.
  Edited in place in the launcher volume (`/root/.cache/llama.cpp/wizard-configs.json`, backup `.bak-20260930`).
- **Roll check** (`X99:/home/anyei/bench154/rollcheck154.sh`: the new image's own `/app` binaries, production MTP shape,
  graphs off): 40.36 / 34.18 / 27.88 t/s = 81.3 / 84.2 / 82.7 ms per verify step, prefill 80.0 t/s (3706 tokens,
  `-ub 512`), texts coherent. Versus the old production path (scalar Q2_0, graphs on: ~94.5 ms/step, 34.0 / 26.4 /
  23.2 t/s): **-12.5 % per verify step** (kernel -8.3 ms + graphs off -3.7 ms); raw t/s +19 / +29 / +21 % on these
  prompts, part of it acceptance of the changed texts.

## 7. Item 3 - doorbell handoff: analysis, design, prototype (2026-09-30: D2 built in the experiment clone, gated)

### 7.1 Where the per-layer handoff goes (measured, X99, plain decode = 1 token per step)

Experiment bins (bins-e10: the #155 `LLAMA_EXP_MOE_SOL` switch, scalar Q2_0), `sol.json` 256 tokens, two runs per leg
(`X99:/home/anyei/exp-moe/exp-doorbell0.sh`):

| leg | CUDA graphs on | CUDA graphs off |
|---|---|---|
| full decode | 47.8-49.3 ms/token | 46.6-47.1 |
| SOL=1: CPU miss work skipped, CPU splits kept, `-t 40` | 43.4 | 41.2-42.2 |
| SOL=1, `-t 8` / `-t 1` | 41.3-41.6 / 39.8-40.0 | - |
| SOL=2: no CPU chain in the graph (GPU only) | 35.5 | **33.0-33.4** |

- **GPU floor 33.2 ms**: ~8,000 small GPU nodes per token (the graph trace: ~36 splits of 162 nodes + ~13 of 195 per
  token) at ~4 us each - the step is kernel-count bound, not bandwidth bound.
- **The CPU split per layer costs 8.5 ms per token** (41.7 - 33.2, ~177 us per layer): ~3.5 ms of it is the
  40-thread pool waking and joining for every CPU node even with nothing to compute (`-t 1` vs `-t 40`), the rest
  (~5 ms, ~100 us per layer) is the host sync, the D2H copy of the FFN input, the H2D copy of the CPU chain's output
  - all 10 expert slots per row, 410 KB per layer at the 4-row verify width, mostly zeros (cached lanes) - and the
  GPU queue draining while the host runs the CPU split.
- **Order inside a layer:** the graph runs router -> CPU chain (misses) -> GPU chain (hits) -> merge; the GPU and the
  CPU take turns, they never overlap.
- **CUDA graphs are a net loss on this serve** (trace: 40 % of the split computes need a graph update because node 0
  changes between steps): +1.3 ms per plain token, and at the production MTP shape with the item-1 kernel
  (`bench154/gate154d.sh`, arms on/off/off/on, texts IDENTICAL) 85.7 vs **82.0 ms per verify step = +4.5 % t/s with
  `GGML_CUDA_DISABLE_GRAPHS=1`**. The saved config 55366197782 cut this template env together with
  `LLAMA_SPEC_DRAFT_NO_PAD` (plan moe-cache 12.9: the pair cost 27-40 -> 20-25 t/s - NO_PAD is the MTP kill-switch);
  graphs were never measured alone. Zero-code, user's config: put `GGML_CUDA_DISABLE_GRAPHS=1` back.
- **Threads** (production MTP shape, item-1 kernel, graphs off, `gate154e.sh`): `-t 40` 84.9, 28 85.7, 20 86.2,
  14 87.3, 8 88.9 ms per step - 40 stays best; the pool overhead is real but fewer threads lose more miss compute.

### 7.2 The design: the MoE CPU work leaves the ggml graph (decode widths only)

Per MoE layer of a decode/verify graph (`n_tokens <= moe_cache->max_batch()`), everything stays on the GPU:

1. **`MOE_RING` (new CUDA op):** one small kernel writes the FFN input rows (T x 2560 f32), `ids_cpu` (T x 10, the
   cache's miss lanes) and the gating weights (T x 10) into this layer's slot of a pinned host mailbox (UVA: pinned
   memory is device-addressable), `__threadfence_system()`, then stores `ring[l] = step`. No host sync.
2. **GPU hits chain** as today (`up_g/gate_g/act/down_g` over the slot pools), then weighted and summed over the
   10 slots on the GPU -> `hits_sum` (T x 2560). This runs WHILE the CPU computes the misses.
3. **`MOE_JOIN` (new CUDA op):** one block spins on `done[l] == step` in the mailbox (volatile UVA loads, nanosleep
   backoff, a timeout that aborts with a clear error instead of hanging), then adds the CPU's weighted partial
   (T x 2560 f32, 40 KB at T=4 instead of 410 KB) to `hits_sum` -> `moe_out`.
4. **CPU executor (new, host side):** a team of spinning worker threads serves the layers in order: quantize the
   rows once (q8_K for q3_K/q2_K gate/up, q8_0 for q2_0), up/gate rows of every miss lane split across threads,
   one barrier, SwiGLU + q8_0 quantize, down rows split across threads, weighted accumulation into the mailbox
   partial, store-release `done[l] = step`. It calls ggml-cpu's own `from_float`/`vec_dot` through
   `ggml_get_type_traits_cpu`, so each dot product equals today's CPU chain's; only the weighted-sum order moves
   (misses summed on the CPU, then added to the GPU's sum). Workers sleep on a condition variable between steps.
5. The step number: the host bumps a mapped `step` word before launching each decode graph; kernels read it, so
   the graph topology never changes between steps.

Unchanged: the #151 cache (tables, step(), fills, eviction), prefill (T > max_batch keeps today's path and op
offload), the MTP head (GPU only). The decode graph becomes a single GPU split (plus the input-embedding lookup),
so the GPU queue never drains inside a step.

### 7.3 Expected gain and falsifiers

- Plain token (graphs off, item 1 in): ~44.7 ms today -> ~34-35 ms (handoff 8.5 -> ~0.5 ms, most of the ~2.5 ms
  of misses hidden under the hits chain): **~+28 %**. MTP verify step: ~82 -> ~70-73 ms: **~+12-17 %**.
- Falsifier 1 (first prototype step, before the executor does any math): ring + join with an executor that
  answers immediately with zeros (SOL-doorbell) must land within ~2 ms of the SOL=2 floor (33.2 ms plain). If the
  mailbox round trip itself costs more than ~40 us per layer, stop.
- Falsifier 2: the full executor must beat the ggml CPU chain on the same misses (SOL=0 vs doorbell, same cache
  counters); if the hand-rolled executor is slower than ggml's mul_mat_id for the miss work, reconsider (option D1).

### 7.4 Options (user pick)

- **D0 - zero code, now:** `GGML_CUDA_DISABLE_GRAPHS=1` back in the saved config (+4.5 %, measured).
- **D1 - doorbell, CPU work via small ggml CPU graphs:** the worker runs a prebuilt per-layer ggml CPU graph
  (mul_mat_id x3 on the mailbox inputs) instead of a hand-rolled executor. Less new code, but ggml's per-node
  thread barriers stay (part of the 3.5 ms pool cost).
- **D2 - full doorbell (recommended):** sections 7.2-7.3. New ggml ops (fork-only) + mailbox + executor: roughly
  800-1,200 lines across ggml-cuda (two ops), ggml.h/ggml.c (op enums), src/llama-moe-cache (mailbox, executor),
  build_moe_ffn (graph), llama-context (step word, lifecycle), env `LLAMA_EXP_MOE_DOORBELL` default off.
- Lane: per the #154 PLAN, prototyped first in the isolated experiment clone (`llama.cpp-exp-moe` + container
  `llama-exp-moe`), then ported to the official tree once gated.

### 7.5 Gates (D2)

G0 build; G1 falsifier 1 (SOL-doorbell vs SOL=2); G2 correctness: KLD doorbell vs today's CPU chain at `-ub 4`
with the cache on (expect the float-order noise floor of section 6, not more), greedy decode texts read coherent;
G3 stress: 2,000+ decode steps with MTP (no hang, no timeout), load/unload cycles; G4 speed: plain and MTP decode
A/B at the production shape, ms per step, arms interleaved.

### 7.6 Prototype (D2) - built 2026-09-30 in the experiment clone (user: "yes to all three, go ahead")

Code (clone `llama.cpp-exp-moe`, uncommitted, env default off): `ggml.h/ggml.c` two fork ops (`GGML_OP_MOE_RING`,
`GGML_OP_MOE_JOIN`, op count 101 -> 103, RPC patch version bumped), `ggml-cuda/moe-doorbell.{cu,cuh}` (ring: one block
copies the rows/ids/weights into the pinned slot, system fences, publishes the step; join: per-block device spin on the
slot's DONE word with a timeout that traps, then `a + partial` with `__ldcv` reads), CPU backend refuses both ops,
`src/llama-moe-doorbell.{h,cpp}` (mailbox in the CUDA device's pinned host buffer type, 8 rows x 2560 per slot,
7.7 MiB for 48 layers; executor team: leader builds the miss list, then quantize / gate+up rows / SwiGLU + q8_0 /
down rows with the weighted sum over each row's misses, spin barriers between phases; ggml-cpu `from_float` +
`vec_dot` through `ggml_get_type_traits_cpu`), `build_moe_ffn` doorbell branch (ring -> hits chain -> weighted sum
-> join, CPU chain not built; only when `n_expert_used` is the model's and the width fits), context hooks (create
before the reserve, `begin()` bumps the step word and wakes the executor before each graph that rings).
Envs: `LLAMA_EXP_MOE_DOORBELL=1` (=2 timing only: zeros), `_THREADS=N` (default `-t`), `_STATS=N`. Item 1's kernel
is applied in the clone too. Bins `X99:/home/anyei/exp-moe/bins-e11`.

**Phase 1 (plain decode, graphs off, `sol.json` 256 tokens, two runs, `X99:/home/anyei/exp-moe/exp-doorbell1.sh`):**

| leg | ms/token |
|---|---|
| today's path (CPU splits, item-1 kernel) | 45.3 / 46.5, repeat 49.6 / 50.2 (the box drifted slower) |
| SOL=2 floor (no CPU chain) | 33.1 / 33.2 |
| **doorbell, timing only (zeros): falsifier 1** | 34.9 / 35.4 = floor + 1.8-2.2 ms (~40 us per layer): **passes** |
| **doorbell, 40 executor threads** | **36.1 / 36.4 (+30 % t/s vs today's path)** |
| doorbell, 16 / 8 threads | 36.7-37.1 / 37.4-38.0 |

Executor counters (40 threads): 1.13-1.28 misses per layer, work 3.8-4.4 ms per token (overlapped with the GPU),
waiting for the GPU ~24 ms per token. Greedy texts diverge from today's after 71-417 characters (float order, as with
item 1); all read coherent. The design predicted 34-35 ms (section 7.3).

**Phase 2 - MTP at the production shape (graphs off, `--moe-cache 15000`, n-max 3, arms base / db / db / base,
`X99:/home/anyei/exp-moe/exp-doorbell2.sh`), ms per verify step:**

| prompt | base | doorbell | change | t/s base -> doorbell |
|---|---|---|---|---|
| code | 82.8 / 83.0 | 62.7 / 62.2 | -24.6 % | 38.6 -> 45.0 |
| engine | 87.2 / 87.0 | 64.5 / 63.9 | -26.3 % | 32.0 -> 42.4 |
| history | 84.0 / 84.1 | 64.1 / 63.5 | -24.1 % | 25.8 -> 33.4 |

**-21 ms per verify step (84.7 -> 63.5 ms) = +33 % decode at equal acceptance** - well past the +12-17 % of 7.3;
the doorbell texts accepted slightly FEWER drafts (code 74.9 -> 62.0 %, engine 60.5 -> 58.7 %, history 39.4 ->
38.6 %), so none of it is acceptance luck. The 4-row verify step paid more handoff than the plain token (the CPU chain
of 4 rows plus a 410 KB H2D copy of all slot outputs per layer). Executor: 5.77 misses per layer per step, work
12.0-13.6 ms per step (overlapped), idle ~36 ms per step.

**Phase 3 - correctness, KLD at the verify width with the cache on (`-b 4 -ub 4`, 3 x 512 tokens):** base vs base
0.000000 / 100 % (deterministic); **doorbell vs base: mean KLD 0.021970, same top 95.556 %, PPL ratio 1.000276 +-
0.009885** - the float-order class of section 6 (item-1 kernel 0.0172, GPU-vs-CPU experts 0.0187): the misses are
summed on the host in another order, with libm `expf` in SiLU. A layer-level unit test of the executor against
ggml's CPU chain on identical inputs belongs to the port (G2a).

**Phase 4 - stress:** 3,000 generated tokens, MTP, temperature 0.7: 36.54 t/s, no timeout, server alive, text
coherent to the end; executor 3.65 misses per layer per step, work 8.7 ms per step.

## 8. The remaining Strata items (2, 3b, 4, 5, 6, 7) - measured first, then built (2026-09-30)

User (after the D2 prototype): "yes, add them to the plan and measure first. Then go ahead and work on all of them
come back to me when you are done with a table with the baseline (prefill and decode) t/s vs after". Lane: the
experiment clone (as items 3-7 of the #154 PLAN), on top of D2; nothing rolled. New reference point for everything
below = **D2 on, graphs off, production MTP shape: ~63.5 ms per verify step**; plain token ~36.3 ms; GPU-only floor
33.2 ms per plain token.

Model facts that set the budgets (GGUF headers): the main model has 12 full-attention layers (QSA sparse attention,
indexer top-k 2048, 2 KV heads x 256) and 36 linear/SSM layers - KV ~24 KB per token (0.8 GB at 32k, 6.4 GB at 262k
f16); `output.weight` q8_0 [2560 x 248,320] = 675 MB. The MTP head (4.1 GB, all on the GPU) is one full layer with its
own 512 q8_0 experts plus its OWN 675 MB output head, run 3x per verify step (n-max 3).

| item | question the measurement answers | measurement (X99, D2 on) | stop rule |
|---|---|---|---|
| 3b CUDA graph per verify width | do graphs pay now that the decode step is one GPU split? | MTP ms/step graphs on vs off + graph trace | on not faster -> fix the update churn or close |
| 2 multi-token CPU kernels, 4 copy-engine share | how much executor work is still on the critical path? | MTP ms/step D2 vs D2 timing-only (zeros) | < 3 ms/step -> close both |
| 7b reduced-vocab MTP head | what does drafting cost per step, and the vocab head inside it? | `LLAMA_SPEC_TIMING(+_SYNC)` draft vs verify | draft < 5 ms/step -> close |
| 5 warm start + swap pacing | how much do the first requests after a load lose to a cold cache? | 5 requests in sequence after a load: t/s + hit rate; faster fill pacing | < 10 % on request 1 -> close |
| 6 prefill | stream ring with D2; what a full-cache `-ub 8192` needs | 16k-prompt prefill at `-ub 512 / 4096 / 4096+stream`, VRAM at 8192 | - (prefill wins already measured) |
| 7a KV streaming | how much decode does 1 GB of extra cache buy? | MTP ms/step at `--moe-cache` 8000 / 11500 / 15000 | slope x KV freed < 5 % -> close for 32k, size for 262k |

### 8.1 Measurements (X99, bins-e11 = item 1 + D2 + stream ring, `exp-m154.sh`, D2 on, graphs off unless noted)

| item | measurement | result | verdict |
|---|---|---|---|
| 3b graphs | MTP ms/step, off / on / on / off | off 62.5-64.5 (63.6), **on 67.0-68.5 (67.6)**; trace: 1081 graph computes, 971 replayed, 109 updates (~1.3 per step: `SET_ROWS cache_idx_k_l*`, `REPEAT hc_init`, a `MUL_MAT`) | graphs replay 90 % yet lose 4 ms/step: host checks over ~8,000 nodes + the per-step updates. **BUILD**: stable graphs - the largest lever left (the GPU side is launch-bound) |
| 2 / 4 executor on the critical path | MTP ms/step, D2 vs timing-only D2 (zeros) | zeros 59.4-60.9 vs real 63.6-64.5 on the two 256-token prompts: **~4 ms/step** still on the critical path (5.8 misses/layer/step) | 2: **BUILD** (group the misses by expert so each weight row serves every row of the step; a 4-row Q2_0 kernel). 4: **CLOSED** - shipping a missed expert over PCIe 3 (~1.7 MB, ~0.14 ms) costs ~2x computing it on the host (~0.07 ms per pair) |
| 7b draft | `LLAMA_SPEC_TIMING` + `_SYNC`, per verify step | draft **5.4 ms** (3 separate draft decodes: full 248k-row head + 1 MB logits copy + host sampling each), verify ~54 ms, accept ~2 ms | **BUILD**: the fused draft chain for qwen4exp (3 drafts in one graph, argmax in-graph) + a reduced-vocabulary head |
| 5 warm start | 5 prompts right after a load, then the first again | request 1 cold 74.2 ms/step vs 69.1 warm (same prompt, cache warm on other topics) = 7 %; faster fill pacing (8 inserts, 768 MiB/step) 73.6 / 64.8 / 64.4 ms = no gain | **CLOSED** (under the 10 % rule; one request per load) |
| 6 prefill | 16k prompt, D2 on | `-ub 512` 81.4, `-ub 4096` 253.4, `-ub 4096` + stream **327.5 t/s**; `-ub 8192` + stream: compute buffer 2280 MiB reserved, then CUDA OOM mid-prefill (runtime pool temps not counted) | **BUILD**: count the big-ubatch temps in the cache reserve (our form of "borrowed slots"); the after-config uses `-ub 4096` + the stream ring |
| 7a KV streaming | MTP ms/step vs cache size | 8.0 GB 74.0, 11.5 GB 68.5, 15.0 GB 63.4 ms/step = **~1.5 ms/step per GB** | 32k: KV 0.8 GB -> at most ~1.2 ms/step: **CLOSED for 32k**. 262k (KV ~6.4 GB): naive streaming of the indexer's top-2048 rows over PCIe 3 costs more than the cache it frees; needs a locality measurement first (next) |

Build order: 3b, 7b, 2, 6, then 7a's locality check.

### 8.2 Item 7b - fused draft chain + reduced-vocabulary head: built, CLOSED (no net win)

Built in the clone (bins-e12): qwen4exp `graph_mtp` gains the #140 fused chain (`LLAMA_SPEC_MTP_FUSED=1`: the 3-token draft
batch iterates the MTP block 3x in one graph, argmax in-graph, ids packed in the nextn channel at the bundle width)
and a reduced-vocabulary head for the chain (`LLAMA_EXP_MTP_VOCAB=N`: the first N rows of the head's output matrix as
a view - BPE ids are merge-ordered - with the declared output row zero-padded back to the vocabulary width). Greedy
MTP, D2 on, graphs off, per-step drafting run before and after (`exp-t154a.sh`), ms per verify step on code / engine
/ history / a Chinese prompt:

| arm | ms/step | acceptance (Chinese prompt) | text vs per-step |
|---|---|---|---|
| per-step drafting (reference, two runs) | 62.9 / 64.5 / 63.7 / 64.9 and 62.7 / 64.4 / 64.1 / 65.0 | 144/330 | - / SAME |
| fused chain, full vocabulary | 63.0 / 64.8 / 64.6 / 65.5 | 144/330 | SAME (4/4) |
| fused, first 64k ids | 60.8 / 63.9 / 61.8 / 61.6 | **78/526** | differs |
| fused, first 32k ids | 61.7 / 62.9 / 62.2 / 61.6 | **77/530** | differs |

The fused chain is exact (identical texts) but 0.1-0.9 ms slower: the qwen4exp head carries its own 512-expert MoE,
so the chain's placeholder rows cost what the two saved round trips give back. The reduced head saves ~2 ms per step,
but drafts can no longer name ids above N: the Chinese prompt's acceptance collapses 44 % -> 15 % (35.2 -> 23.3
t/s) and English is mixed (-7 % to +12 % t/s). Texts move because different drafts change the cache's admissions and
with them the host/GPU split of the experts (float order). A frequency-ranked subset that covers CJK is the only
version worth trying; it is capped at ~3 %.

### 8.3 Item 3b - CUDA graphs: tried, CLOSED

Graph-field trace (bins-e12, `GGML_CUDA_GRAPH_TRACE` extended with a per-field diff): the main model's graphs (4,398 and
4,151 nodes) alternated under ONE graph key - the key was the first node's address, which every graph llama builds in
the same context memory shares - and each switch changed thousands of nodes (shapes, data pointers): a full re-capture.
bins-e13 keys the CUDA graph by shape (first node + node count + three probe nodes' ne/op; `GGML_CUDA_GRAPH_SHAPE_KEY=0`
restores the old key). Result (`exp-t154b.sh`, `exp-t154c.sh`, D2 on):

| leg | graphs off | graphs on (shape key) |
|---|---|---|
| MTP, ms/step (two runs) | 64.6-66.1 | 66.1-70.0 (+3.3 ms) |
| plain decode, ms/token (two runs) | 36.4-37.2 | 38.7-38.9 (+2 ms) |

Even at a constant width, 89 re-captures over ~512 tokens (475 distinct shapes): the KV views grow with the context
(`cache_k_l*` ne) and some rebuilds reallocate. With 85 % of computes replayed and still a 2 ms loss, pure replay is
worth at most ~3 ms per token; getting there needs coarse KV padding plus allocation-stable rebuilds (graph-reuse
work in the KV cache), not a key change. Closed; `GGML_CUDA_DISABLE_GRAPHS=1` stays.

### 8.4 Item 2 - multi-row expert reuse in the executor: tried, CLOSED

bins-e13 grouped the executor's misses by expert, so each weight row was read once and dotted with every row of the
step that missed the same expert. Measured (`exp-t154b.sh`): 6.80 misses per layer per step come from 5.34 distinct
experts - only 21 % of the misses share an expert, so there is little to reuse - and the work per miss did not drop
(44 us before, 45-49 us after; ms/step 63.3 -> 65.1, with different texts because the summation order moved).
Reverted (bins-e14 keeps only the distinct-expert counter). A 4-row Q2_0 kernel has the same ceiling. The executor's
remaining ~4 ms/step on the critical path is memory-bound streaming of distinct experts.

### 8.5 Item 7a - KV streaming: analysed, CLOSED for this fork's attention

The fork's QSA sparse attention is a MASK over dense attention: the indexer's top-2048 cells are unmasked in a -inf
mask and `build_attn_mha` then reads every cached cell (qwen4exp.cpp `build_attn_qsa`). So nothing is "read only
partly": streaming the KV to host memory would only add PCIe traffic. At the production 32k context the dense masked
attention reads ~0.8 GB per step (~1 ms) and freeing the KV's 0.8 GB for the cache is worth <= 1.2 ms/step (8.1). The
lever Strata's number points at is a gather-based sparse attention (read only the selected rows): ~1 ms/step at 32k,
~8 ms/step once a 262k context is actually full - a kernel project of its own, filed as a follow-up, not built here.

### 8.6 Item 6 - prefill: the stream ring + `-ub 4096` in; `-ub 8192` / borrowed slots CLOSED

- In the combined build (bins-e14 = item 1 + D2 + the #155 stream ring), 16k-token prompt with D2 on and MTP loaded:
  `-ub 512` 81.4, `-ub 4096` 253.4, **`-ub 4096` + stream 327.5 t/s** with the full 15 GB cache and no decode cost.
- `-ub 8192` + stream reaches **441.9 t/s** (no MTP, cache 11 GB, peak 24.3 of 32.8 GB), but with the MTP head loaded
  it does not fit next to a 13.5-15 GB cache: at rest 30.7-32.2 GB, +1.8 GB during the prefill -> CUDA OOM
  (`exp-t154d/e.sh`). Raising `LLAMA_MOE_CACHE_RESERVE_MB` (3584 / 4096 / 4608) never clamped the cache: the clamp runs
  before the draft head loads and sees free VRAM that is gone later (robustness issue, noted; not fixed here). A cache
  of ~12-12.5 GB would fit, at ~1.5 ms/step per GB = ~+4 ms/step decode for +35 % long-prompt prefill: not the
  default. Real "borrowed slots" (lend cache memory to one big prefill, refill after) needs releasable pool memory
  (the pools are fixed buffers and the CUDA VMM pool never shrinks) - closed.

## 9. Final table - baseline vs after (2026-09-30, X99, one session, arms B C A A C B, `exp-final154.sh`)

Model: Qwen3.8-Flash-Next GSQ-RCO 3.5bit (+ Q4_0 PLE shard) with the Q8_0 MTP head (n-max 3, p-min 0.75); one V100
32 GB, `-ngl 99 --n-cpu-moe 48 --moe-cache 15000 -c 32768 -t 40 -fa on`. Decode = three 256-token greedy prompts after a
warm-up; prefill = a 3,706-token and a 15,855-token prompt. Two runs per arm (means; per-run values in the log).

- **BASELINE** = production before #154: image a9885783c-t3k3, CUDA graphs on (the saved config then), `-ub 512`.
- **CURRENT** = rolled today: image 99b3c21ee-q2avx2 (item 1 AVX2 Q2_0) + `GGML_CUDA_DISABLE_GRAPHS=1`, `-ub 512`.
- **AFTER** = experiment bins-e14: item 1 + D2 doorbell + stream ring, `-b 4096 -ub 4096`, graphs off.

| | BASELINE | CURRENT | AFTER | after vs baseline |
|---|---|---|---|---|
| decode, code prompt (t/s) | 33.81 | 40.19 | **44.77** | +32 % |
| decode, engine prompt (t/s) | 26.39 | 34.03 | **42.34** | +60 % |
| decode, history prompt (t/s) | 22.97 | 27.66 | **33.39** | +45 % |
| decode, mean of the three (t/s) | 27.72 | 33.96 | **40.17** | **+45 %** |
| time per verify step (ms, mean) | 95.3 | 83.2 | **63.7** | **-33 % (1.50x)** |
| prefill 3.7k prompt (t/s) | 80.5 | 80.3 | **325.9** | **4.05x** |
| prefill 15.9k prompt (t/s) | 81.9 | 82.4 | **330.2** | **4.03x** |

Per-step time is the clean measure (acceptance differs with the text: AFTER's code prompt accepted 62 % of drafts vs
74.5 % at BASELINE, the other two prompts within 1-2 points). Where it comes from: item 1 -8.3 ms/step, graphs off
-3.7 ms/step (both live now = CURRENT), the D2 doorbell -19.5 ms/step, the stream ring + `-ub 4096` the 4x prefill.
Closed with numbers: items 2, 3b, 4, 5, 6 (`-ub 8192` / borrowed slots), 7a, 7b (sections 8.1-8.6).
To get AFTER in production: port D2 + the stream ring into the official tree (user's go) and set `-ub 4096 -b 4096`
in the saved config.

### 9.1 The table again after the code-review fixes (2026-09-30, `exp-final154b.sh`, arms B C A14 A15 A15 A14 C B)

Same model, shape and prompts as section 9; AFTER pre-fix = bins-e14, AFTER fixed = bins-e15 (research/code-review-154
"Fixes"; default spin budget 1000 us). Means of two runs; "CPU" = the server container's CPU use while decoding.

| | BASELINE | CURRENT | AFTER pre-fix | **AFTER fixed** | fixed vs baseline |
|---|---|---|---|---|---|
| decode, code (t/s) | 34.21 | 40.62 | 45.00 | **44.80** | +31 % |
| decode, engine (t/s) | 26.53 | 34.48 | 42.29 | **42.15** | +59 % |
| decode, history (t/s) | 23.14 | 28.14 | 33.43 | **33.08** | +43 % |
| decode, mean (t/s) | 27.96 | 34.41 | 40.24 | **40.01** | **+43 %** |
| ms per verify step (mean) | 94.6 | 82.1 | 63.6 | **64.0** | **-32 % (1.48x)** |
| prefill 3.7k (t/s) | 79.5 | 80.5 | 326.8 | **326.2** | 4.1x |
| prefill 15.9k (t/s) | 81.1 | 82.0 | 331.0 | **330.3** | 4.1x |
| CPU while decoding (cores) | 27.0 | 26.8 | 22.7 | 23.2 | -14 % |

The fixes are speed-neutral: 64.0 vs 63.6 ms per step (+0.6 %, inside the run-to-run spread of ~1 ms), prefill and
texts identical. What they change is correctness and safety (V4 clamp order, no job cancellation or shared step, no
error-path hang, clean teardown, bounds checks, the stream-ring race) plus a test that proves the executor bit-exact
against ggml's CPU chain. The doorbell does not raise CPU use: the baseline's own CPU threadpool keeps ~27 cores busy
while decoding, the doorbell build ~23.

## 10. Port into the official tree (2026-09-30, user: "yes go ahead with the port") - PORTED + ROLLED (10.1-10.2)

What goes in (from the experiment clone, bins-e15, with the review fixes): the D2 doorbell, the prefill stream ring, the
executor unit test. What stays out: every closed experiment (lookahead, MTP-aware, global pool, SOL switch, probes,
the CUDA-graph shape key and field trace, the fused draft chain / reduced vocabulary). Nothing committed (commit =
user); everything default OFF behind env gates.

- ggml: `GGML_OP_MOE_RING` / `GGML_OP_MOE_JOIN` appended at the END of the op enum (no existing id moves) - `ggml.h`,
  `ggml.c`; CPU backend refuses them; CUDA implements them (`ggml-cuda/moe-doorbell.{cu,cuh}` + dispatch + supports_op).
- **RPC (decision flagged to the user before any roll)**: the RPC patch version is an op-set fingerprint checked at
  HELLO - bumping it would reject every deployed worker until the whole fleet (incl. the user-only .15 box) is rebuilt.
  The two ops never travel: the RPC client's `supports_op` now refuses them (the scheduler cannot place them on a remote
  device) and new workers reject out-of-range op ids instead of casting them. With ids unchanged and the ops local-only
  the fingerprint stays (patch 3; the static_assert moves to 103 with the reason). Bumping instead = one line.
- Scheduler: the prefill stream ring in `ggml-backend.cpp`, envs `GGML_SCHED_PREFILL_STREAM=<slots>` (1 = 3 slots) and
  `GGML_SCHED_PREFILL_STREAM_MIN` (tokens, default 1024).
- llama: `src/llama-moe-doorbell{,-compute}.{h,cpp}`, the `build_moe_ffn` branch + `llm_graph_input_moe_db`, context
  hooks (create before the reserve, job before the inputs, sync + abandon on a failed compute, sync in the destructor,
  training without it), `llama_moe_cache::layer_at`. Envs `LLAMA_MOE_DOORBELL` (=1 on, =2 timing only: wrong text),
  `LLAMA_MOE_DOORBELL_THREADS`, `LLAMA_MOE_DOORBELL_SPIN_US` (default 1000), `LLAMA_MOE_DOORBELL_STATS`.
- tests: `tests/test-moe-doorbell.cpp`. Docs: `docs/env-gates.md` rows + the wizard gates catalog.
- Gates: G0 build (build-cpu with tests, build-cuda75); G1 unit test; G2 OFF-gate identity on the X99 (the port build
  with the envs unset vs the rolled image: identical greedy texts, same ms/step and prefill); G3 ON-gate (the plan-9.1
  numbers, KLD vs the CPU chain, 3,000-token stress, back-to-back ubatches); G4 RPC: the loopback harness on the port
  build (reference sha) and a port-build client against a deployed (older) worker.

### 10.1 Port results (2026-09-30)

- Applied on HEAD 063bdc2a0 (the user's "strata port plan" commit), uncommitted: 16 files changed + 7 new
  (`ggml-cuda/moe-doorbell.{cu,cuh}`, `src/llama-moe-doorbell{,-compute}.{h,cpp}`, `tests/test-moe-doorbell.cpp`). The
  experiment clone's closed experiments were not carried; the clone's `ggml-cuda.cu` was not copied (only the dispatch +
  supports_op hunks).
- G0: build-cpu (with tests) and build-cuda75 clean; the only `-Wreorder` warning left is the pre-existing
  `moe_cache` / `expert_mask` one.
- G1: `test-moe-doorbell` - all 8 cases bit-identical to ggml's CPU chain, the V4-wrong-order control caught (0.33).
- G4 RPC (`llama.cpp-work/154/rpc-gate{,-old}.sh`, loopback trunc vehicle, the #71 off-leg stack): port-build client +
  port-build workers **sha c80261ff** (the reference), stable 6/6; port-build client + the DEPLOYED worker image
  (`llamacpp-cpu:rpc-worker-latest`, 2026-08-19) **sha c80261ff**, stable 6/6 - the kept op-set fingerprint works
  across the fleet with no worker rebuild.
- Docs: `docs/env-gates.md` section "MoE doorbell + prefill stream" (6 rows), wizard gates catalog 134 -> 140 (tick
  values: doorbell 1, stream 1, threads 40, spin 0, stats 128), `gen-wizard-flags.py --check` clean.
- G2 OFF (switches unset) vs the rolled image 99b3c21ee-q2avx2, arms image / port / port / image
  (`X99:/home/anyei/bench154/port154.sh`): greedy texts **IDENTICAL** on all three prompts in both port runs; ms/step
  image 82.6-85.5, port 82.6-86.4 (mean +0.7 %, inside the V100 drift - the off path is an early return); 16k
  prefill 81.9 / 80.9 vs 81.1 / 80.9 t/s.
- G3 ON (`LLAMA_MOE_DOORBELL=1 GGML_SCHED_PREFILL_STREAM=1`, `-ub 4096 -b 4096`): **63.7-65.0 ms per MTP step**
  (50.1 / 42.8 / 35.8 t/s), **16k prefill 328.3 / 330.7 t/s**; KLD at `-ub 4` with the cache vs the CPU chain
  0.020267 / same top 95.29 % / PPL ratio 0.996 +- 0.0096; 3,000-token MTP stress 36.02 t/s, alive, coherent;
  back-to-back `-ub 4` prompt ubatches clean.

### 10.2 Rolled (2026-09-30 evening, user: "Go ahead")

- The user's idle Flash-Next serve was unloaded for the gates (launch args + env captured first).
- `llamacpp-local-v100:063bdc2a0-db` (sha256:a47b290d; HEAD 063bdc2a0 + the uncommitted port) = `:latest` locally and in
  the registry; rollback 99b3c21ee-q2avx2; a9885783c-t3k2 / -t3k3 removed locally and on the X99 (still in the
  registry). X99 launcher recreated on the pinned tag: healthy, wizard (140 gates), 23 models, 27 configs, dirs, hw.
- Saved config 55366197782: `gateOn` `LLAMA_MOE_DOORBELL=1` + `GGML_SCHED_PREFILL_STREAM=1`, `flagSet`
  `--ubatch-size 4096 --batch-size 4096` (backup `wizard-configs.json.bak-20260930b` in the launcher volume).
- The serve relaunched through `/models/load` with the captured args + the new ones; through the router: **decode
  50.53 / 43.32 t/s at 64.1 / 64.9 ms per MTP step, 3.7k prefill 326.5 t/s**, texts coherent.
