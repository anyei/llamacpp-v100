# `--ssd-streaming` — Design & Investigation (Task 15, Phase 0)

Three-tier weight placement — resident GPU (`-ngl`), resident CPU, streamed
from SSD — so models can run when they don't fit in VRAM, or even in
VRAM + RAM. Phase 0 deliverable: measured baselines, prior-art survey, code
map, and the design the later phases implement. Task breakdown: `TASKS.md`
item 15.

## 1. Measured reality (this box, 2026-07-07)

Hardware: 2x V100-SXM2-32GB, 46 GB RAM, 20 cores, models on one NVMe.
Model: `Qwen3.6-27B-UD-Q4_K_XL-MTP.gguf` (16.4 GB dense hybrid).

| Baseline | Result | Method |
|---|---|---|
| NVMe sequential read (O_DIRECT) | **1.6 GB/s** | `dd iflag=direct`, 4 GiB |
| 27B CPU-only, fits in RAM (reference) | **1.51 t/s** tg, 3.5 t/s pp | `-ngl 0`, 10 threads, RSS 27.7 GB |
| 27B CPU-only, 14 GB cgroup cap (mmap thrash) | **0.21 t/s** tg (4.8 s/token), 0.58 t/s pp | same + `docker --memory 14g`, model page-cache evicted first |

Reading the numbers:

- The thrash case is what llama.cpp *already does* when a model doesn't fit:
  default mmap demand-pages weights from disk and the kernel evicts under
  pressure. **0.21 t/s is the baseline managed streaming must beat.**
- Thrash lands *above* the naive floor (16.4 GB / 1.6 GB/s ≈ 0.1 t/s)
  because weight access order is sequential per layer, the loader sets
  `POSIX_FADV_SEQUENTIAL` (kernel readahead works), and whatever fits in
  the remaining page-cache window stays hot. A managed tier wins by
  (a) prefetching layer n+1 *during* layer n's compute instead of faulting
  synchronously, (b) pinning the hottest tensors deliberately rather than
  letting LRU churn them, (c) not evicting the resident CPU tier's pages
  (O_DIRECT bypasses page cache).
- ~11.3 GB of the CPU run is anonymous memory (KV, compute, output buffers)
  — the *weights* are the only streamable part; placement budgeting must
  account for the fixed anonymous floor.
- Measurement trap for future benchmarks: cgroup caps only charge pages
  *faulted by that cgroup*. If the model is already in the host page cache,
  a capped container reads it for free and shows no thrash. Evict first:
  `os.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)`.

Honest expectations for the feature (dense 27B, this NVMe):

| Tier mix | Expected decode |
|---|---|
| Everything streamed | ~0.1-0.3 t/s — runnability, not speed |
| 46/48 layers resident, 2 streamed | ~ +0.4 GB/token ≈ resident speed − ~0.25 s/token |
| Large-batch prefill, streamed | one 16 GB stream amortized over the whole ubatch (~10 s per 4k batch) |
| MoE with hot-expert cache | the real win — see prior art below |

## 2. Prior art

- Upstream has **no managed disk streaming**; the maintainers' answer is
  "mmap demand-paging is the mechanism"
  ([discussion #19163](https://github.com/ggml-org/llama.cpp/discussions/19163)).
- [Issue #20757](https://github.com/ggml-org/llama.cpp/issues/20757)
  proposes almost exactly our phase 4: GPU VRAM expert slots + pinned RAM
  backing + mmap SSD tier, SLRU eviction, frequency-gated admission. Python
  PoC on an 8 GB GPU running GPT-OSS-120B: 12-14 t/s steady state (~98%
  cache hits) vs 0.5-1 t/s pure CPU offload. No maintainer response, no PR —
  the field is open, and it validates the MoE sweet spot quantitatively.
- [Issue #18766](https://github.com/ggml-org/llama.cpp/issues/18766):
  upstream/this fork already ships `--direct-io` in the loader (see §3).
- [Issue #9059](https://github.com/ggml-org/llama.cpp/issues/9059):
  `--no-mmap` stages GPU layers through RAM — long-standing pain point that
  a streamed buft also fixes as a side effect (SSD → staging → VRAM without
  a full RAM copy).
- [Issue #20697](https://github.com/ggml-org/llama.cpp/issues/20697):
  disk offload for context checkpoints — adjacent, same staging/IO
  machinery could serve it later.
- Literature: FlexGen (throughput-oriented tier scheduling), DeepSpeed
  ZeRO-Inference (layer-wise streaming with double buffering). Their core
  lesson transfers: overlap transfer with compute and batch enough work per
  streamed byte.

## 3. What the tree already has (code map)

Anchors from a full read of the loading/alloc/exec paths:

- **O_DIRECT aligned reads, CLI-wired**: `llama_file` takes
  `use_direct_io`; `src/llama-mmap.cpp:199` opens `O_RDONLY|O_DIRECT`,
  `read_aligned_chunk` (`:319-349`) does bounce-buffer aligned reads.
  CLI `--direct-io`/`-dio` (`common/arg.cpp:2401`). mmap and direct-io are
  mutually exclusive in the loader (`llama-model-loader.cpp:564-567`).
- **Per-tensor file offsets already exist**: `weights_map` /
  `llama_tensor_weight.offs` in the model loader — GGUF is
  offset-addressable per tensor, no format work needed.
- **Async pinned-staging upload pipeline**: the no-mmap GPU load path
  (`llama-model-loader.cpp:1584-1635`) already rotates 4x64 MB pinned
  buffers through `read_raw_unsafe` → `ggml_backend_tensor_set_async` +
  events. This is the transfer engine for streamed SSD→VRAM, reusable
  nearly as-is.
- **Non-resident buffer precedent**: the RPC buffer
  (`ggml/src/ggml-rpc/ggml-rpc.cpp:610-621`) implements
  `ggml_backend_buffer_i` where `get_base` returns an opaque handle and
  set/get materialize data on demand — the template for a streamed buft.
  Minimum iface: `get_base`, `set_tensor`, `get_tensor`, `clear`;
  `init_tensor` is the natural place to attach `(file, offset)` extras.
- **Pre-compute hook point**: `ggml_backend_sched_compute_splits`
  (`ggml/src/ggml-backend.cpp:1546+`) — the eval-callback path (`:1687+`)
  already demonstrates per-node pre-compute interception, but forces
  node-at-a-time execution; a dedicated prefetch hook belongs just before
  `graph_compute_async` (`:1683`), per split, where each split's node list
  is known in advance.
- **Per-graph prefetch schedules for free**: the fork's decode-graph cache
  (`src/llama-context.cpp:1339-1430`) keeps one scheduler per cached graph
  shape and replays identical topology — a per-`decode_cache_entry`
  streamed-weight schedule stays valid across reuses and is computed once.
- **Anti-features to disable for streamed tensors**: `llama_mlock`
  (pins pages), `MAP_POPULATE`/`POSIX_MADV_WILLNEED` prefetch-at-load.
- **Gap**: `-ot`/`--override-tensor` only offers *device default* bufts
  (`common/arg.cpp:248-280`), so the streamed buft must be registered
  through a device (or the parser extended) to be targetable per tensor.
- Nothing io_uring/cuFile/`cudaMemPrefetchAsync` in the tree — phase 5
  material, not needed for correctness.

## 4. Design

A new **streamed buffer type** (per backing device: CPU-streamed or
CUDA-streamed), modeled on the RPC buffer:

- Tensors carry `(file_id, offset, size)` instead of resident bytes
  (attached at `init_tensor`; offsets come from the loader's existing
  `weights_map`). "Loading" a streamed tensor is metadata-only — instant.
- A **pinned staging ring** (`--ssd-stream-budget`, default a few hundred
  MB) feeds reads; reads use the existing O_DIRECT `read_aligned_chunk`
  so streaming never evicts the resident CPU tier from page cache.
- **Prefetch**: per scheduler split, the ordered list of streamed tensors
  is precomputed (once per cached decode graph); a worker thread reads
  split n+1's tensors into the ring while split n computes; the compute
  path blocks on a per-tensor event only when prefetch hasn't finished.
  Dense access order is fully deterministic — prefetch hit rate should be
  ~100%; MoE expert selection is the only dynamic part (phase 4).
- **Placement/CLI**: `--ssd-streaming` streams everything not covered by
  `-ngl`; `-ngl N --ssd-streaming` keeps N layers resident on GPU and
  streams the tail (front layers resident first — they're reused by every
  token equally, so placement policy is just "as many resident as fit");
  interplay with `-ot` for per-tensor control once the buft is registered.
- **Correctness**: weights are immutable and read-only — no eviction
  bookkeeping, no writeback, crash-safe by construction. Dequant stays on
  GPU/CPU exactly as today; the streamed buft only changes *where bytes
  live*, never *what math runs*. Gate every phase on temp-0 byte-exactness
  vs a fully-resident run.

## 5. Phases

0. ✅ This document: baselines, prior art, code map, design.
1. ✅ **CPU-tier managed streaming — NEGATIVE RESULT (2026-07-07).**
   Implemented as an mmap residency director (`common/ssd-streaming.cpp`,
   `LLAMA_SSD_STREAMING=1` + `LLAMA_SSD_STREAM_BUDGET=<MiB>`): a
   scheduler-callback-driven loop that WILLNEEDs a budgeted window ahead of
   compute and DONTNEEDs behind it, with the weight order discovered on the
   first pass and treated as a ring. Byte-identical by construction
   (madvise is advisory). Measured, 27B @ 14 GB cap, cold cache:
   **thrash 0.266 t/s vs director 0.274 t/s — no measurable win.**
   Root cause of the null result: for DENSE models the access pattern is
   perfectly sequential, and the kernel's readahead (helped by the loader's
   `POSIX_FADV_SEQUENTIAL`) already overlaps IO with compute near-optimally —
   both configs run at ~3.7 s/token against a ~10 s/token serial-IO floor.
   The disk is bandwidth-saturated either way; steering cannot add bandwidth.
   **Consequence: skip further CPU-tier work for dense models.** The director
   code stays (env-gated, off by default) as scaffolding — its ring/cursor/
   callback machinery is exactly what phase 4 needs, where access is NOT
   sequential and the kernel CANNOT predict it (MoE expert selection).
2. GPU-tier streaming (ring → `tensor_set_async` on a side stream,
   overlapped with layer compute; reuses the loader's staging pattern).
   Unlike the CPU tier this bypasses the RAM staging bottleneck entirely,
   so the phase-1 null result does not apply.

   **Phase 2/4 feasibility spike — MEASURED, PASS (2026-07-08).** The whole
   design rests on one number: random small-block **O_DIRECT** read bandwidth
   (the cache-miss path). Measured on this box's NVMe against the real 81 GB
   `DeepSeek-V4-Flash-IQ2XXS` GGUF (`scripts/ssd-stream-bench-odirect.cpp`,
   O_DIRECT `pread` at random aligned offsets across the 78 GB expert region):

   | Read size | 1 thread | 2-8 threads |
   |---|---|---|
   | 2.16 MB (one expert slice) | 1672 MB/s | **~2700 MB/s** |
   | 7.08 MB (gate+up+down set) | 2448 MB/s | **~2800 MB/s** |

   Random O_DIRECT reads do **not** collapse — they *exceed* the 1.6 GB/s
   sequential `dd` figure and saturate at ~2 threads, and O_DIRECT never
   touches the page cache, so the `madvise(WILLNEED)` direct-reclaim trap
   (phase 4, −60x) cannot recur. DeepSeek-V4-Flash layout (via
   `scripts/ssd-stream-gguf-layout.py`): 43 layers × 256 experts, **6 used +
   1 shared** per token; expert tensors are **77.9 GB of the 81 GB** (gate/up
   IQ2_XXS ~2.16 MB/slice, down Q2_K ~2.75 MB/slice), non-expert weights only
   **8.8 GB** (fit RAM or one GPU). Cold per-token expert traffic = 258
   slice-sets × 7.08 MB = **1.83 GB**. Projected expert-IO-bound decode at
   2.7 GB/s:

   | Cache hit rate | miss/token | t/s (IO-bound) |
   |---|---|---|
   | 0% (pure stream, no cache) | 1.83 GB | **1.48** |
   | 44% (uniform, 34 GB RAM cache) | 1.02 GB | **2.64** |
   | 80% (moderate routing skew) | 0.37 GB | **7.39** |
   | 95% (strong skew, cf. #20757) | 0.09 GB | **29.6** → compute-bound |

   Even a zero-cache pure stream (1.48 t/s) beats the mmap-thrash floor
   (1.12 t/s) purely by replacing page-faults with O_DIRECT `pread`; a RAM
   cache and real routing skew multiply from there. **Conclusion: build it.**
   The 81 GB DeepSeek-V4-Flash (doesn't fit VRAM *or* RAM) is the reference
   test vehicle — it can only run via this feature. Note: t/s above is
   expert-IO only; at high hit rates the model becomes compute-bound (MLA
   attention + 6 experts/layer on whatever tier runs the matmul), which is the
   next thing to measure once a build exists.
3. Placement policy + CLI (`--ssd-streaming`, `--ssd-stream-budget`),
   docs, compose profile.
4. MoE-aware expert streaming with a hot-expert cache
   (SLRU per issue #20757's evidence; admission-filtered).
   **Feasibility measured (2026-07-07) — the prize is real, the madvise
   route is not:**
   - GLM-4.7-Flash (14 GB MoE, ~3B active) CPU-only: **10.87 t/s uncapped
     vs 1.12 t/s at a 10 GB cap** — a 10x collapse (semi-random expert
     access defeats page LRU + readahead), so expert-aware residency has
     an order of magnitude of headroom. This is the one tier where the
     kernel demonstrably fails.
   - BUT steering via `madvise(WILLNEED)` per selected expert made it
     **60x WORSE** (0.018 t/s): under a memory cgroup at its limit,
     WILLNEED blocks in direct reclaim per call (~1,100 calls/token) —
     page-cache hints are self-defeating under exactly the pressure they
     target. The per-selection steering code remains in the director
     (env-gated) as a record; do not enable it for capped MoE.
   - **Conclusion: phase 4 must be a userspace cache** — pread/O_DIRECT
     into an owned arena (pinned for the GPU tier), explicit SLRU over
     expert slices, no page-cache involvement. That is the #20757 design,
     and it is the same machinery the GPU tier needs anyway — phases 2
     and 4 merge into one build.
5. Stretch: io_uring read queues, cuFile/GDS direct SSD→VRAM, worker-side
   streaming for TP islands (models bigger than an island's combined VRAM),
   disk-backed context checkpoints (#20697) on the same machinery.

## 6. Using the experimental director (what exists today)

There is no `--ssd-streaming` CLI flag yet — the shipped artifact of phases
1/4 is the **mmap residency director**, enabled by environment variable on
any tool that goes through `common_init_from_params` (llama-server,
llama-cli, llama-perplexity, ...):

```bash
LLAMA_SSD_STREAMING=1 \
LLAMA_SSD_STREAM_BUDGET=4096 \   # resident weight window, MiB (default 4096)
llama-server -m model.gguf -ngl 0 ...
```

What it does: rides the scheduler's eval callback, discovers the weight-use
order on the first pass, then keeps a budgeted window resident
(`WILLNEED` ahead of the compute cursor, `DONTNEED` behind). Requires mmap
loading (the default; it forces `--mlock` off). Output is **byte-identical**
with the director on or off — madvise is advisory and computation is never
altered.

Honest guidance, from the measurements in §5:

| Situation | Should you enable it? |
|---|---|
| Model fits in RAM | **No** — pure overhead (per-node callback barriers), and a budget smaller than the weights forces needless re-streaming |
| Dense model larger than RAM | **No benefit** — kernel readahead already overlaps IO/compute optimally (0.274 vs 0.266 t/s); it just doesn't hurt |
| MoE model larger than RAM / under a memory cap | **NO — actively harmful.** The per-expert WILLNEED steering measures 60x slower under cgroup pressure (madvise blocks in direct reclaim). Run unmanaged mmap instead |
| Bounding RSS of an over-provisioned box (weights resident elsewhere) | The one plausible niche: `DONTNEED` keeps the page-cache footprint near the budget instead of letting it grow to the full model |

In short: the director exists to document what does not work and as
scaffolding for the real feature. The actual `--ssd-streaming` flag arrives
with the userspace expert cache (merged phases 2+4 above), which is the
design the measurements point to.

## 7. Verification (house rules)

- Temp-0 byte-exactness vs fully-resident at every phase.
- Enforced ceilings: RSS + VRAM measured under load with the budget knobs.
- A tg/pp table per tier mix in `docs/perf-tuning-v100.md` when phases land.
- Repeat all capped-memory benchmarks with the page-cache eviction step
  (§1) — results without it are invalid.

## 8. Build plan — phase 2/4 implementation (2026-07-08, in progress)

Decision: **land experts on the GPU (VRAM) as the end-state compute/cache tier,
with RAM as an L2 victim cache and SSD as the cold tier** (the V100 out-computes
the CPU by 10-50x, and at realistic MoE hit rates the model is compute-bound
where the GPU wins; a VRAM miss that hits RAM refills at PCIe ~12 GB/s instead
of re-reading SSD at 2.7 GB/s). But **build the CPU-landing tier first** — it
gets the model *running* with zero kernel changes and lets us measure the real
expert hit rate cheaply; the GPU tier is a later speed upgrade.

**Simulation harness:** to force a genuine "doesn't fit VRAM+RAM" case on this
box, constrain to **one V100 (32 GB) + 46 GB RAM = 78 GB < 81 GB** (`CUDA_VISIBLE_DEVICES=0`).
The DeepSeek-V4-Flash 81 GB then cannot be held resident by any `-ngl` split and
*must* stream. (On both GPUs it fits combined — that case is the stress vehicle,
not a can't-run case; the true can't-run target is >110 GB models.)

Increment staircase (each is a commit with its own gate):

1. **✅ DONE (2026-07-08, commit 5b3a7377c). Streamed host buffer type + LRU
   expert cache (CPU landing).** Result: Qwen-35B-A3B (23 GB, `-ngl 0 --no-mmap`)
   byte-IDENTICAL to fully-resident, peak RSS 8.2 GB. **Headline: DeepSeek-V4-Flash
   81 GB runs on ONE V100 (32 GB VRAM) + 46 GB RAM** — load 9.2 s, coherent output,
   decode 0.69 t/s (cold cache / CPU compute / node-at-a-time = the correctness-
   first floor). Implementation note vs the original sketch below: fill uses
   buffered `pread` (not yet O_DIRECT) and a node-at-a-time eval callback (the
   router that produces the expert-selection ids must compute before the fill
   reads them — batching them corrupted output; fixed). Original sketch:
   New `ggml_backend_buffer_type` reporting `is_host`,
   backed by a `mmap(MAP_ANON|MAP_NORESERVE)` arena sized to the streamed
   tensors at their natural offsets; loading is metadata-only (record file
   offset, no copy). A pre-`MUL_MAT_ID` hook `pread`s (O_DIRECT, via the
   loader's `read_aligned_chunk`) the *selected* experts' slices into the arena,
   computes, then `MADV_DONTNEED`s them — RSS stays at the in-flight working
   set. `ffn_*_exps.weight` routed to the buft (auto-detect or `-ot`). **Gate:**
   (a) byte-exact temp-0 vs fully-resident on a small MoE that fits (GLM-4.7-Flash
   14 GB or Qwen3.6-35B-A3B 23 GB); (b) DeepSeek-V4-Flash 81 GB *runs* on 1 GPU
   with RSS bounded to the budget. Expected speed = the ~1.5 t/s streaming floor.

   **Optimizations landed since increment 1 (DeepSeek-V4 81GB, 1×V100, 30GB budget):**
   batch non-expert nodes / node-at-a-time only as fallback (`0d07b9647`, 1.00→1.29
   t/s; `LLAMA_SSD_STREAM_SERIAL=1` for -ngl 0) · O_DIRECT bounce reads
   (`1bb6244c5`, bypass page cache → bounded cgroup charge + faster DMA, 1.29→1.66
   t/s). **Trajectory: 0.69 cold → 1.00 warm → 1.29 batched → 1.66 O_DIRECT t/s.**
   Next big lever = GPU landing (3) to drop the CPU expert-matmul.
2. **✅ DONE - SLRU measured, NO WIN; kept opt-in, LRU stays default.**
   Implemented a segmented LRU (probation + protected, 2nd hit promotes, eviction
   takes the probation tail first) behind `LLAMA_SSD_STREAM_SLRU=1`
   (`LLAMA_SSD_STREAM_PROTECTED_PCT`, default 80). Offline unit test confirms scan
   resistance; on-model it is **output-neutral** (SLRU == LRU == big-budget,
   byte-identical on the deterministic Qwen-35B-A3B `-ngl 0` path; identical
   hit/miss/evict stats on DeepSeek).
   **DeepSeek-V4 81GB, 1×V100, `-ngl 99`, decode hit rate (deterministic) + t/s:**

   | budget | LRU hit | SLRU hit | t/s |
   |-------:|--------:|---------:|----:|
   |  8 GB  |  54.0%  |  52.9%   | ~1.7 |
   | 20 GB  |  68.1%  |  67.8%   | ~2.5 |
   | 30 GB  |  73.4%  |  73.6%   | ~3.1 |

   SLRU ties LRU at 30 GB and is marginally *worse* at 8/20 GB - DeepSeek routed-
   expert reuse is not skewed enough for the protected segment to earn its
   complexity (a burst of one-time experts is rare; the working set is
   near-uniform over the 256 experts). **Keep plain LRU (simpler, >= SLRU
   everywhere measured); SLRU stays opt-in for more skewed future workloads.**
   The measured hit rates *exceed* the phase-2/4 feasibility projections (44% @
   34 GB) - real MoE locality is stronger than the uniform-random model assumed;
   30 GB (39% of the 77.9 GB expert set resident) already hits 73% and 3.1 t/s
   (vs the 1.66 t/s increment-1 O_DIRECT figure, which was a colder/earlier
   measurement). Harness: `llama cli -ngl 99 --no-mmap -st [--ignore-eos]`,
   `GGML_SSD_STREAM_DEBUG=1 -v` for the hit-rate line.

   **⚠ TWO pre-existing issues found while probing correctness (NEITHER caused by
   increment 2; increment 2's code is output-neutral by construction and verified
   so - SLRU == LRU == no-eviction big budget, byte-identical):**

   **(a) Generation: a small near-tie MoE divergence from fully-resident.** On the
   deterministic Qwen `-ngl 0` path, streamed greedy output differs from resident
   by ~1 token in 40 (coherent, ~97% identical: "The user says" vs "User says") -
   a MoE-router near-tie flip, same class as task 13/17 for GLM. A/B established:
   cache-policy-independent (SLRU/LRU/big-budget identical); **NOT the read method**
   (buffered `LLAMA_SSD_STREAM_NO_ODIRECT=1` == O_DIRECT), so intrinsic to the
   streaming compute path. Generation is coherent and top-1-stable at every config
   tested - single-stream decode and prefill up to ~80-token prompts, both `-ngl 0`
   and `-ngl 99` (Qwen `-ngl 99` streamed decode == `-ngl 0` streamed, coherent).
   At `-ngl 99` output is additionally run-to-run non-deterministic (GPU/MoE
   near-tie noise; two identical LRU runs differ with byte-identical cache stats).
   Logits are clearly *good* in generation (garbage logits would give garbage
   greedy text; the text is coherent), so this reads as benign near-tie noise -
   but see (b): it could NOT be PPL-validated.

   **(b) [FIXED 2026-07-09] `llama-perplexity` / all-position eval produced GARBAGE
   with streaming.** Streamed PPL over wikitext-2 (Qwen-35B-A3B, 16 chunks) = **193**
   vs resident **7.09**; garbage from chunk 1.
   **Root cause: the scheduler's "copy only the used experts" optimization**
   (`ggml-backend.cpp` compute_splits, ~1582-1665) materializes a `MUL_MAT_ID`'s
   expert weights straight from the source buffer (`input->data`) into a per-split
   copy during the **input-copy phase - which runs before any node in the split**,
   i.e. before the lazy eval-callback fill. So it copies *unfilled* (`MAP_NORESERVE`
   zero) arena pages; the matmul then computes from that copy and the fill lands too
   late / into a buffer the compute no longer reads. Fires whenever a streamed
   expert is a cross-split input (both `-ngl 0` via the distinct buft boundary and
   `-ngl 99` CPU->GPU). This ALSO explains issue (a)'s generation near-tie flip:
   first use of an expert reads the unfilled arena (1-2 wrong tokens), later decode
   steps read the now-filled arena -> coherent.
   **Fix:** new hook `ggml_ssd_stream_prefill_experts(w, used_ids_bitset, n_expert)`
   called from that block (using the `used_ids` the optimization already computes)
   to make the used experts resident *before* the copy reads them; no-op when
   streaming is off (so `-cmoe`/resident are untouched). **Verified: Qwen-35B-A3B
   streamed PPL 158/193 -> 6.2022, byte-identical to the `-cmoe` control (6.2022)
   and resident; and the flagship DeepSeek-V4-Flash 81 GB on ONE V100 (30 GB
   budget, `-ub 128`) now scores a sane PPL 4.0568 (was the 100+ garbage class).**
   The generation near-tie flip at `-ngl 0` remains (5541 vs 7e35) but is now
   **proven quality-neutral** - PPL is identical, so the distribution matches and
   only a near-tied argmax flips (benign, task-13 class). Requires the per-node
   expert working set to fit the budget (true for decode; large prefill ubatches
   need a budget >= one ubatch's expert union, or a smaller `-ub`).
   Original isolation trail (kept for the record):
   - Independent of read method, cache policy, and budget (24 GB = no eviction).
   - Independent of ubatch size (`-ub 512` 158.6, `-ub 64` 158.7 - identical) and
     of n_seq (`-b 512` = n_seq 1 still garbage).
   - **Independent of backend**: `-ngl 0` SERIAL streamed PPL = **100** vs ~5
     resident - so it is not the `-ngl 99` cross-backend split, and SERIAL
     (node-at-a-time, fully synchronized) does NOT fix it.
   - The ONLY axis that flips it: **perplexity mode itself** (all-position /
     `logits_all` evaluation) vs generation (last-position logits). Same model,
     same prompt, same budget: an ~80-token prompt *generated* coherently but the
     same tokens under perplexity score PPL 100+.
   The `-cmoe` control was the clue that cracked it: keeping experts on a *normal*
   CPU buffer (`llama-perplexity -ngl 99 -cmoe`, no streaming - identical partial
   offload) scores PPL 6.20 (correct), so only the streaming arena breaks it - which,
   combined with reading the scheduler (a study of antirez/ds4 pointed at the
   all-position case being special), located the copy-path materialization above.
   Standing test = `-cmoe` vs streamed PPL A/B (`scratchpad/verify_fix.sh`).
   With the fix, `llama-perplexity` can validate a streamed model and `-np > 1`
   concurrent serving is verified: a 2-slot streamed server (`-np 2`, `-ngl 99`)
   served two concurrent temp-0 requests with coherent output (same multi-output
   `n_outputs>1` path the fix covers).
   **Reference (ds4/DwarfStar)**: antirez's DeepSeek-V4 engine sidesteps this whole
   class - experts live in a *compact* VRAM/host slot cache and a *custom* MoE kernel
   does id->slot indirection, so streamed experts never go through ggml's
   buffer/scheduler copy paths at all. Its prefill is layer-major (batch all tokens
   through each layer, amortizing one expert load per layer), and it validates
   against official all-position logits. That is also the increment-3 GPU-landing
   design (compact slot cache + `MUL_MAT_ID` id->slot), which would replace the
   natural-stride arena.
3. **GPU landing + `MUL_MAT_ID` slot indirection** (hot experts in VRAM cache
   slots, GPU computes them), RAM demoted to L2 victim cache. **Gate:** byte-exact
   (reduction-order tolerance for GPU), t/s at GPU compute speed.
   **Recon (2026-07-09):** `MUL_MAT_ID` offloads to the GPU only when
   n_tokens >= `GGML_OP_OFFLOAD_MIN_BATCH` (default 32, `get_op_batch_size` = the
   op's ne[2]; `ggml-cuda.cu`). So **prefill already computes experts on the GPU**
   (that path hit the copy bug, now fixed), but **decode (1 token) computes experts
   on the CPU** - that is the ~2.5-3 t/s decode bottleneck. The scheduler's
   selective-copy already compacts the *used* experts onto the GPU when offloaded,
   but transiently (re-copied every token, no persistent VRAM cache).
   **Phase-0 experiment (`GGML_OP_OFFLOAD_MIN_BATCH=1`, force decode onto GPU, no
   code):** DeepSeek-81GB streamed decode **2.5 -> 1.2 t/s (WORSE)** - the
   1.83 GB/token of experts re-copied H2D over PCIe every token dominates; Qwen-35B
   (smaller expert volume) **2.5 -> 3.6 t/s (better)**. **Conclusion: a persistent
   VRAM slot cache is REQUIRED, not optional** - naive offload is a net loss on the
   flagship. Skip a "just force offload" phase. **Bonus:** the GPU-offloaded run is
   **byte-identical to resident** (Qwen sha == resident) - GPU expert compute matches
   the resident GPU reduction order, so the CPU-path near-tie flip disappears on GPU.
   **Design (phase 2, ds4-style):** a bounded CUDA buffer of K expert slots with a
   stable **id->slot table** (LRU eviction); hot experts persist in VRAM across
   tokens, only misses fill (RAM arena -> VRAM H2D, or SSD -> staging -> VRAM). Before
   the matmul, remap global expert ids -> slot ids and run the existing `MUL_MAT_ID`
   kernel against the compact K-slot tensor (no custom kernel needed for a first cut).
   RAM arena stays as the L2 victim tier. Composes with CPU-only mode (no GPU => just
   don't allocate the VRAM tier; CPU-resident + SSD-streamed still runs).
4. **Prefetch / overlap** (double-buffer reads behind compute; within-flight
   only — a token's experts are known only after its router runs).
5. Stretch (§5): io_uring, cuFile/GDS direct SSD→VRAM, worker-side streaming for
   TP islands, disk-backed checkpoints.

### 8.1 Implementation spec / integration anchors (from code recon 2026-07-08)

Key constraint: a ggml CPU kernel reads `src[i]->data` as a raw host pointer at
compute time — there is no set/get indirection to hook. So the streamed buffer
**must own real host backing** that we populate *before* the consuming node
runs. Mechanism: `is_host`→true buffer over a `mmap(MAP_ANON|MAP_NORESERVE)`
arena (virtual = sum of streamed tensors at gallocr-assigned offsets; physical =
only pread'd, not-yet-`MADV_DONTNEED`'d pages), + an eval-callback that returns
`true` on streamed-expert nodes to force a per-node pre-compute fill.

- **Buffer ifaces**: `ggml/src/ggml-backend-impl.h` — `ggml_backend_buffer_type_i`
  (17-29; required get_name/alloc_buffer/get_alignment; optional is_host,
  get_alloc_size), `ggml_backend_buffer_i` (41-64; required get_base/clear;
  provide set_tensor/get_tensor). Factory `ggml_backend_buffer_init`.
- **Template**: RPC buffer, `ggml/src/ggml-rpc/ggml-rpc.cpp` iface tables
  752-765 / 859-866, buft singleton 1094-1125. (RPC `is_host`=NULL → must
  override to true for in-place CPU matmul.)
- **CPU accepts host bufts**: `ggml_backend_buft_is_host` (ggml-backend.cpp
  74-80); `ggml_backend_cpu_device_supports_buft` (ggml-cpu.cpp 476-477).
- **Loader routing + skip-read**: `src/llama-model-loader.cpp` buft select
  1143-1205 (-ot match 1162-1188); bulk read host path `file->seek(offs);
  read_raw(cur->data, n_size)` at 1574-1576 — must be skipped for streamed
  tensors and (tensor→file,offs) registered instead. Offsets:
  `llama_tensor_weight.offs` (llama-model-loader.h 33-50), file idx `.idx`.
  Increment 1 routes `blk.\d+.ffn_(gate|down|up)_exps.weight` via a
  `--ssd-streaming` flag (own path), not `-ot` (the -ot buft-discovery gap).
- **MUL_MAT_ID srcs** (`ggml.c` 3290-3314): src[0]=experts (ne[2]=n_expert,
  slice stride nb[2]), src[1]=activations, src[2]=ids (I32, ne[0]=used/token,
  ne[1]=n_tokens); ids->data host-readable at eval.
- **Eval-callback hook**: `ggml-backend.cpp` compute_splits 1546-1730, ask=true
  pre-compute call at 1693; returning true forces the node to compute alone
  right after the fill (nodes with false return get batched, losing the seam).
- **O_DIRECT reads**: self-contained `pread` with 4 KB-aligned bounce buffer
  (proven at ~2.7 GB/s in `scripts/ssd-stream-bench-odirect.cpp`); or
  `llama_file(path,"rb",use_direct_io=true)` + `seek;read_aligned_chunk`
  (llama-mmap.h 16-41). Home: new `src/llama-ssd-stream.{cpp,h}` (llama-internal,
  reachable by the loader; the existing `common/ssd-streaming.cpp` director is a
  separate, advisory mechanism and stays as-is).

### 8.2 Phase 2 design — VRAM expert slot cache (GPU landing), scoped 2026-07-09

Goal: compute streamed experts on the GPU at decode without paying a per-token
H2D of the whole used-expert set (phase-0 proved naive offload loses on DeepSeek:
2.5 -> 1.2 t/s). Keep hot experts resident in a bounded VRAM cache across tokens;
only misses fill. This is ds4/DwarfStar's design (`cuda_stream_expert_cache`:
compact gate/up/down slot buffers + id->slot indirection).

Injection point: `llm_graph_context::build_moe_ffn` (`src/llama-graph.cpp:1746`),
which calls `ggml_mul_mat_id(w, cur, ids)` for gate/up/down over the streamed
`*_exps` tensors with the router's `selected_experts` ids. When
`ggml_ssd_stream_is_streamed(w)` and GPU-landing is enabled, swap `(w, ids)` ->
`(slot_buffer, slot_ids)`.

Components:
- **VRAM slot pool** per streamed expert tensor (or one pool keyed by
  (tensor,expert)): a persistent CUDA buffer of K slots x expert-slice-bytes,
  K << n_expert, allocated at load like a weight (outside gallocr). K derived from
  a VRAM budget knob (`LLAMA_SSD_STREAM_VRAM_BUDGET`). Quantized slice format
  preserved (IQ2/Q2_K) so the existing MMQ/MMVQ `MUL_MAT_ID` kernel consumes it.
- **Slot table** (host): (tensor,expert) -> slot, LRU. RAM arena stays as the L2
  victim tier (VRAM miss -> check RAM arena -> else SSD).
- **Remap+fill op** before each streamed `mul_mat_id` (a `ggml_map_custom` or new
  op): reads `selected_experts`; for each used expert ensure a VRAM slot (miss =>
  evict LRU slot, enqueue H2D from the RAM arena / pinned staging on the compute
  stream); output `slot_ids` (global id -> slot id). In-stream ordering makes the
  fills complete before the matmul; no extra sync.

Key facts that make it tractable:
- `MUL_MAT_ID` reads src0 expert `s` at `s*nb[2]` and picks experts via src2 ids,
  so a K-slot tensor (ne[2]=K) + remapped slot_ids reuses the existing kernel -
  NO custom matmul kernel needed for the first cut (unlike ds4's fully custom one).
- The scheduler's selective-copy already proves the used-expert -> GPU compaction
  works; phase 2 makes it persistent + compact instead of full-size + per-token.

Constraints / risks:
- Budget K must hold one layer-ubatch's used-expert union (same rule as the CPU
  arena) or slots thrash within a node.
- Eviction must not drop a slot the current node still needs (touch-then-pin, like
  the CPU tier).
- gate/up/down are separate tensors -> separate slot pools (ds4 keeps 3 buffers).
- H2D from the arena wants pinned staging for bandwidth (non-pinned is ~2x slower).

Sub-increments (each its own commit + gate):
- 3.2a Allocate VRAM slot pool + host slot table; budget accounting; no compute
  change. Gate: allocates, RSS/VRAM bounded.
- 3.2b Remap+fill custom op (fill from arena, emit slot_ids). Offline unit-test the
  remap/LRU like the SLRU test.
- 3.2c Rewire `build_moe_ffn` for streamed experts behind `LLAMA_SSD_STREAM_GPU=1`.
  Gate: byte-exact vs resident within GPU reduction tolerance on Qwen (recall the
  phase-0 GPU path was byte-identical to resident); coherent DeepSeek.
- 3.2d Measure t/s vs CPU baseline (2.5) and phase-0 offload (1.2); target > 2.5 and
  toward the projected 7+ at high VRAM hit rate.
- 3.2e Tune K/budget; RAM arena as L2 victim; optional pinned staging ring.
- (Phase 3) prefetch/overlap H2D behind compute on a side stream.

Composes with CPU-only mode: no GPU => don't allocate the VRAM tier; CPU-resident +
SSD-streamed still runs (just slower). GPU-landing is strictly additive.

#### 8.2.1 Prior-art validation (2026-07-09) — the design is proven, be optimistic

Two independent implementations confirm the phase-2 design and, importantly,
correct a framing mistake (I had under-emphasized *persistence*):

- **ds4/DwarfStar (ships it)**: `ds4_gpu_stream_expert_cache_begin_selected_load`
  builds `compact_ids` (unique used experts) + `slot_ids` (remapped) and loads via
  `cuda_stream_selected_cache_begin_compact_load`, which tracks
  cache_hits/misses/host_hits/direct_loads against a persistent host hash cache +
  a peer-GPU VRAM cache tier (`DS4_CUDA_PEER_EXPERT_CACHE_GB`, +12% decode). So the
  compact-per-token load is fed by persistent tiers — hits avoid re-read.
- **llama.cpp issue #20757 (spec'd + measured, NO PR yet — field is open)**:
  proposes exactly this — a GPU buffer of N slots (`--moe-expert-cache-size N`,
  N << n_expert) with a **persistent expert_id->slot_idx mapping carried across
  decode passes: HIT = zero copy**, miss = evict + copy from pinned RAM + update
  map. **Hook is the same block as our bug fix**: `compute_splits` selective
  expert-copy (`expert offset = first_id*expert_size`) — remap to slot indices
  there; "a small change to the `GGML_OP_MUL_MAT_ID` path" for slot-remapped ids.
  Eviction = **SLRU (probationary ~20% / protected ~80%) + a 2nd-miss admission
  filter**, +8-15pp hit rate over LRU. Measured on GPT-OSS-120B (57GB experts) on
  an **8GB** GPU: cold 1.9-2.5 t/s (48-56% hit) -> steady **12-14 t/s at ~98-100%
  hit**, where "the model is GPU-compute bound, the PCIe pipe mostly idle." CPU
  baseline 0.5-1 t/s. MoE routing is skewed (~15-20% of experts serve ~80% of
  tokens), which is why a small VRAM cache reaches ~98% hit.
- Adjacent field: ik_llama.cpp hybrid CPU/GPU MoE, the HF "MoE offload" guide,
  KTransformers - the broad approach (keep hot experts on GPU, stream the tail) is
  well-trodden, not speculative.

**What this changes in our plan:**
1. **Persistence is the whole game.** Phase-0's 1.2 t/s (re-copy every token) is
   exactly the antipattern #20757 identifies; with a persistent id->slot map that
   number becomes ~12 t/s. Design the mapping to survive across `llama_decode`
   calls from day one.
2. **Hook = the `compute_splits` selective-copy block we already own** (where
   `ggml_ssd_stream_prefill_experts` lives), NOT `build_moe_ffn`. Redirect the
   used-expert copy into a persistent K-slot VRAM buffer and remap the ids GPU copy
   to slot indices right there. We know this code intimately from the bug fix.
3. **SLRU is rescued and belongs here.** Increment 2's "no win" was the *RAM* tier
   at 30GB/73% hit; the *VRAM* tier is small + high-pressure + skewed, exactly where
   #20757 measures SLRU +8-15pp. Reuse the increment-2 SLRU and add the 2nd-miss
   admission filter.
4. **Realistic target: compute-bound 12+ t/s at steady state** on the skewed-MoE
   models, not the timid "> 2.5" I first wrote.

#### 8.2.2 3.2b implementation notes (in progress 2026-07-09)

- **3.2b-1 (landed in tree, gated off):** `gpu_slot_pool` policy engine (LRU, K
  slots, touch->{slot,hit/miss}, offline-tested) + lazy VRAM slot-buffer alloc
  (`ggml_ssd_stream_gpu_ensure_pool`, one pool per expert-slice shape, K = VRAM
  budget / slice) driven from the `compute_splits` expert-copy block via the
  generic backend API (no CUDA in ggml-backend.cpp). Config: `LLAMA_SSD_STREAM_GPU`,
  `LLAMA_SSD_STREAM_VRAM_BUDGET`. No compute change yet.
- **3.2b-2 subtlety to solve (the fill + remap + src swap):** the block detects the
  streamed offloaded MUL_MAT_ID via `node->src[0] == input_cpy`. If we permanently
  swap `src[0]` to the persistent K-slot buffer, the NEXT token's block no longer
  matches (src[0] is now the slot buffer) -> no fill -> stale. Re-entry must be
  handled. Plan: track the node by the stable **input-weight identity** (the arena
  tensor, which never changes), capture the **original ids tensor** once (so we can
  keep reading the router's fresh selection after swapping src[2]), and each token:
  touch the pool (hits skip fill), H2D-fill misses from the RAM arena
  (prefill_experts already made them resident -> RAM is the L2 victim tier), build
  slot_ids into a persistent GPU scratch, set `src[0]=slot_buffer` + `src[2]=slot_ids`,
  and skip the old full-size `copy_experts`. On Volta there are no CUDA graphs, so
  per-token src mutation is safe. Gate: byte-exact vs resident on Qwen (phase-0
  showed the GPU path is byte-identical), then measure decode t/s (needs 3.2b-3:
  force offload at batch-1 for streamed experts).

#### 8.2.3 3.2b-2 result (2026-07-09) — GPU landing WORKS: byte-exact + ~2-3x decode

The full path landed: fill used-expert slots (hit = zero copy) H2D from the RAM
arena, remap ids -> slot ids into a contiguous GPU scratch, alias `input_cpy` to
the persistent slot buffer (keeps `src[0]==input_cpy` so the block's detection and
re-entry stay valid), swap `src[2]` to slot ids, skip the full copy. Re-entry solved
by capturing the router-selection tensor per node. +1 pad slot for the MMQ over-read.

Gate (Qwen-35B-A3B, `-ngl 99`, forced offload `GGML_OP_OFFLOAD_MIN_BATCH=1`, 6 GB
VRAM cache):
- **Correctness: byte-exact.** cache ON == cache OFF == resident (sha 7e35bdc...).
- **Perf:** no-cache forced offload 3.7 t/s -> **cache 6.1 t/s (40 tok) -> 7.2 t/s
  (96 tok, warming)**; vs the ~2.5 t/s CPU-compute baseline = ~3x, still climbing.
  Matches the #20757 trajectory (warms toward compute-bound).

Remaining in 3.2b/3.2e: (3.2b-3) scope the offload to streamed expert nodes only so
decode uses GPU landing without the global `GGML_OP_OFFLOAD_MIN_BATCH` env; GPU-cache
hit-rate debug log; DeepSeek-81GB measurement; then SLRU + 2nd-miss admission (the
increment-2 policy, #20757 +8-15pp) and RAM-tier skip on VRAM hits.

#### 8.2.4 RAM-skip (L2 victim tier) + honest perf split (2026-07-09)

Made the RAM arena a true L2 victim tier: on GPU landing, the unconditional
prefill is skipped; `gpu_bind` fills RAM (ssd_touch, pread on a RAM miss) ONLY on a
VRAM miss, then H2Ds that slice to the slot. VRAM hits touch neither RAM nor SSD.
Added a GPU-cache hit-rate debug line. Qwen still byte-exact; Qwen decode 6.1 ->
**7.0 t/s** (the redundant RAM fill on hits is gone).

**Honest perf split:**
- **Cache covers the hot set (Qwen-35B-A3B): big win.** 2.5 (CPU) -> **7.0 t/s**,
  byte-exact vs resident.
- **Experts >> VRAM (DeepSeek-V4 81GB, 77.9GB experts / 8-16GB cache): not winning
  yet.** LRU ~30% hit, per-token H2D of the misses dominates: 1.5-1.8 t/s vs CPU 2.5.
  Large VRAM budgets also hit pressure from the wasted full-size `input_cpy` gallocr
  buffers (aliasing leaves them allocated) -> a transient exit=137 at 12-16GB.

Not the final word for DeepSeek: issue #20757 reached 98% hit / 12-14 t/s at a
*similar* ratio (57GB experts / 8GB GPU) via routing SKEW + SLRU + 2nd-miss
admission, warmed over ~160 tokens. Our runs are LRU, <=96 tokens, VRAM-pressured.

**3.2e (next, to settle DeepSeek):** (a) stop gallocr from allocating the full-size
`input_cpy` for GPU-landing experts (reclaim VRAM -> bigger cache, no OOM); (b) SLRU
+ 2nd-miss admission on the slot pool (reuse increment-2 SLRU); (c) longer warm +
per-pool budget by hot-set size. Also: 3.2b-3 (scope offload to streamed expert
nodes so decode uses GPU landing without the global GGML_OP_OFFLOAD_MIN_BATCH).
Committed state: GPU landing works, byte-exact, gated off by default (LLAMA_SSD_STREAM_GPU).

#### 8.2.5 Pool sizing + recycled-node crash fix (2026-07-09)

Two DeepSeek findings:
- **Pool sizing was silently starving the cache.** Per-pool budget = total /
  LLAMA_SSD_STREAM_VRAM_POOLS (default 6). DeepSeek has only 2 expert-slice classes
  (gate_up iq2_xxs, down q2_K), so a "12 GB" budget gave 12/6=2 GB/pool -> ~4 GB
  effective. Setting VRAM_POOLS=2 gives full utilization (2 x 6.3 GB) and hit rate
  jumped 30% -> **64.5%** - DeepSeek routing IS skewed and a right-sized cache
  captures the hot set. (Follow-up: auto-size pools by discovered class count.)
- **Recycled-node crash (exit 139).** g_gpu_nodes is keyed by node pointer; DeepSeek's
  longer/bigger decode churns the graph cache, and a rebuilt node can reuse a freed
  address -> stale entry with a dangling orig_ids -> garbage ids -> the MUL_MAT_ID
  id-range assert. Fixed: ggml_ssd_stream_gpu_orig_ids validates node->src[2] is
  either the captured orig_ids or our slot_ids; otherwise the node was recycled, so
  re-capture and drop the stale scratch. (Qwen never triggered it - stable decode
  graph, verified byte-exact + survives heavy eviction.)

**DeepSeek now: crash-free, 64.5% hit, 2.3 t/s decode (POOLS=2, 12 GB) vs CPU 2.5 -
break-even, up from 1.5 pre-fix.** Clear-win levers remain: SLRU + 2nd-miss admission
(reuse the inc-2 SLRU on gpu_slot_pool; #20757 +8-15pp), reclaim the wasted full-size
input_cpy VRAM for a bigger cache, and prefetch/overlap the miss H2D (3.3). Qwen-class
(cache covers the hot set) is already a clear win: 2.5 -> 7.0 t/s.

#### 8.2.6 SLRU on the GPU slot pool - NEUTRAL for DeepSeek (2026-07-09)

Added a segmented LRU (probation/protected, promote-on-2nd-hit, evict probation-
tail-first) to gpu_slot_pool, default on (LLAMA_SSD_STREAM_GPU_SLRU, PROTECTED_PCT
80). Offline-tested (scan resistance holds; zero-copy hit preserved). On-model it
is a NO-WIN for DeepSeek: LRU 64.5% hit / 2.3 t/s vs SLRU 64.7% / 2.5 t/s (noise) -
same result as the increment-2 RAM-tier SLRU. At a 12 GB cache the resident set
already covers what fits, so scan resistance isn't the binding constraint; the
misses are genuinely cold experts neither policy keeps. (The 2nd-miss admission
filter #20757 pairs with SLRU was NOT added; given pure SLRU is neutral here it is
unlikely to move the needle much for this workload.) Kept default-on as the
prior-art policy (harmless, may help more-skewed models); Qwen still byte-exact.

**Increment-3 verdict:** GPU landing is a clear win where the VRAM cache covers the
hot expert set (Qwen-35B-A3B **2.5 -> 7 t/s**, byte-exact). For the >>VRAM extreme
(DeepSeek-V4 81GB, 77.9GB experts / 12GB cache) it reaches **break-even (2.5 t/s)**
after the pool-sizing fix (30 -> 64.7% hit) and crash fix; SLRU is neutral. Beyond
break-even needs prefetch/overlap of the miss H2D (phase 3.3) and/or reclaiming the
wasted full-size input_cpy VRAM for a bigger cache - diminishing returns for a model
this far over VRAM. Recommend shipping GPU landing for the Qwen-class win and
treating DeepSeek-scale as runnable-at-parity.

#### 8.2.7 3.2e-a input_cpy VRAM reclaim - DONE (2026-07-09)

Reclaimed the dead-weight VRAM the graph allocator reserves for each streamed-expert
`MUL_MAT_ID` input copy. The scheduler creates `input_cpy` as a full copy of one
layer's expert weight (n_expert slices), but on GPU landing `gpu_bind` redirects
`input_cpy->data` to the persistent slot pool - so that full-size arena reservation
is never read. New `ggml_ssd_stream_gpu_shrink_copy()` shrinks the copy to a single
slice (`ne[2]=1`) at the copy-creation site in `ggml_backend_sched_split_graph`,
BEFORE gallocr sizes the arena. Compute-time behaviour is unchanged: `gpu_bind`
already overwrote `ne[2]`/`nb[2]`/`data` regardless of the copy's original size, and
the only residual (`nb[3]`) is unused since the weight is 3D (ne[3]=1).

- **Correctness (the gate): PPL identical.** Qwen-35B-A3B streamed + GPU landing,
  16 chunks: reclaim ON **7.0898 ± 0.27805** == OFF **7.0898 ± 0.27805**. (Greedy
  temp-0 text is NOT a usable gate here - `-ngl 99` GPU MoE is intermittently
  non-deterministic from near-tie routing noise: across repeat batches ON==OFF held
  in 2 of 3 and the lone mismatch had ON≠OFF *and* OFF≠OFF - independent of the flag.
  PPL averages over the noise and is the fork's standard streamed-correctness gate.)
- **Reclaim measured (CUDA0 compute buffer, tenant-independent):** Qwen **407.00 ->
  73.69 MiB (-82%, ~333 MiB)**; DeepSeek-81GB **1344.25 -> 517.77 MiB (-61%, ~826
  MiB)** (bigger slices -> bigger reclaim). Hit rate and t/s identical at equal
  budget (same cache); the freed VRAM is additive headroom for a larger cache/budget
  and removes the transient `exit=137` pressure at 12-16 GB.
- **Safety:** the shrink does NOT allocate the pool (it runs in the measure/reserve
  pass before weights load - allocating there mis-sized pools by class count: the
  registry was empty so auto-size saw 0 classes and gave each pool the whole budget,
  4x25 GB on Qwen - caught and fixed). The pool is allocated on the compute path with
  a complete registry; a shrunk copy that ever reached the CPU-compute fallback would
  overrun its one slice, so that path aborts loudly (only possible on catastrophic
  VRAM exhaustion). Kill-switch: `LLAMA_SSD_STREAM_GPU_NO_RECLAIM=1`.

#### 8.2.8 3.3 parallel miss-path reads - prefill win, decode-neutral (2026-07-09)

Profiled the GPU-landing miss path first (env-gated timers, `GGML_SSD_STREAM_DEBUG`):
on a VRAM miss `ssd_touch` either finds the slice resident in the RAM arena or preads
it from SSD, then we H2D it to the slot. DeepSeek-81GB (14 GB VRAM / 30 GB RAM,
66-68% VRAM hit) split as **read 26 s vs H2D-issue 16 s** over a 96-tok run, and **62%
of VRAM misses also miss RAM** -> a single-threaded O_DIRECT pread (~1.5 GB/s, the
1-thread baseline) dominated. So the reads, not the H2D, were the thing to attack.

Parallelized the preads (`LLAMA_SSD_STREAM_READ_THREADS`, default 1 = serial/unchanged):
`gpu_bind` now runs three phases - (1) touch the VRAM pool + reserve the RAM slot
serially (SLRU state is not thread-safe; `ssd_touch` gained a `defer_read` mode that
does bookkeeping/eviction but skips the pread), (2) pread the RAM-missed slices across
N workers, each with its OWN O_DIRECT bounce buffer (the shared `st.bounce` is not
concurrency-safe; distinct dst + bounce -> race-free), (3) H2D every VRAM miss.

Result (DeepSeek-81GB, 96 tok, sweep 1/2/4/8): read time **25.9 -> 20.1 s (-22%,
saturates at 2 threads)**; **prefill +31% (1.3 -> 1.7 t/s)** - prefill's per-node call
has a large batch of misses to parallelize; **decode neutral (2.5 -> 2.6 t/s, noise)** -
decode touches only ~6 experts/layer, so within-call parallelism has little to chew on,
and the token is now co-bound by the untouched pageable H2D (16 s ~ 20 s read).
**Correctness: DeepSeek PPL rt=1 4.5240 == rt=4 4.5240** (8 chunks, prefill-heavy so it
exercises the parallel path hard).

**Verdict:** parallel reads help the batch-amortized path (prefill / long prompts), not
single-stream decode of a model this far over VRAM. Kept env-gated (default 1). Pushing
decode further would need the H2D overlapped + pinned (a Volta side-stream, risky - see
task 17's negative overlap result) and coarser read batching across layers -
diminishing returns; not pursued. The profiling timers stay (debug-gated) as the tool
that sized this.

### 8.3 Multi-GPU behavior - benchmarked (2026-07-09)

First real multi-GPU measurements of the streaming feature (2x V100-32GB, cli
`-n 128` temp-0). Prompt-processing t/s omitted; decode (tg) shown.

| Model | Config | 1-GPU | 2-GPU layer (`-sm layer`) | 2-GPU tensor (`-sm tensor`) |
|---|---|---:|---:|---:|
| Qwen-35B-A3B (23 GB, **fits**) | resident | - | 100.6 | 107.9 |
| Qwen | streaming (CPU tier) | - | 10.7 | 9.6 |
| Qwen | streaming + GPU landing | **11.7** | 10.8 (no-op) | 9.6 (no-op) |
| DeepSeek-V4 81 GB (**can't fit 64 GB**) | streaming | ~2.5 | 1.5 | **load crash** |
| DeepSeek | streaming + GPU landing | ~1.5-2.5 | 1.5 (no-op) | - |

**Findings:**
1. **GPU landing is single-GPU-only.** In *both* multi-GPU modes the VRAM slot pool
   never allocates (0 pool-alloc lines): `-sm tensor` routes compute through the
   meta backend, which bypasses the CUDA-device offload hook; `-sm layer` splits
   layers across devices, and the class-keyed pool (one buffer, one GPU) doesn't
   fit that. So 2 GPUs buy **no streaming speedup** - the CPU-tier expert path is
   single-stream-bound regardless of GPU count, and 1-GPU GPU-landing (11.7) beats
   every 2-GPU streaming config.
2. **The only real 2-GPU win is resident** (Qwen 100-108 t/s) - for models that fit.
   Streaming is a single-GPU-focused runnability tool; when a model fits, resident
   tensor-split is ~10x faster than streaming it (108 vs 9.6).
3. **DeepSeek 81 GB loads + runs in `-sm layer`** (1.5 t/s, coherent) but **crashes
   at load in `-sm tensor`**: `llama_meta_device_get_split_state` ->
   `GGML_ASSERT(!suffix_fallback.empty())` (`llama-model.cpp:427`) - the meta
   backend's split-state resolver doesn't handle the streamed expert tensors for the
   `deepseek2`/MLA arch (task-13-adjacent). So DeepSeek streaming on 2 GPUs = layer
   mode only, today.
4. **Non-fatal gallocr OOM in multi-GPU streaming:** `ggml_gallocr_reserve_n_impl`
   attempts an oversized reserve (79 GB Qwen / 298 GB DeepSeek on one device),
   fails, and the run recovers with coherent output. Cosmetically alarming; a
   mis-sized worst-case reservation in the streaming + multi-GPU-split path to clean up.

**Recommendation:** ship single-GPU GPU-landing as the sweet spot. A real 2-GPU
streaming win needs GPU landing made multi-GPU-aware - per-GPU slot pools keyed by
(class, device) and engagement through the meta/layer offload path - which would
let a 2-GPU DeepSeek use ~24-48 GB of VRAM cache (higher hit -> past parity). That,
the `-sm tensor` DeepSeek load crash, and the gallocr reserve size are the multi-GPU
follow-ups.
