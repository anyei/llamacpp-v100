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

1. **Streamed host buffer type + "runs at all" (CPU landing, pull-compute-drop,
   no persistent cache).** New `ggml_backend_buffer_type` reporting `is_host`,
   backed by a `mmap(MAP_ANON|MAP_NORESERVE)` arena sized to the streamed
   tensors at their natural offsets; loading is metadata-only (record file
   offset, no copy). A pre-`MUL_MAT_ID` hook `pread`s (O_DIRECT, via the
   loader's `read_aligned_chunk`) the *selected* experts' slices into the arena,
   computes, then `MADV_DONTNEED`s them — RSS stays at the in-flight working
   set. `ffn_*_exps.weight` routed to the buft (auto-detect or `-ot`). **Gate:**
   (a) byte-exact temp-0 vs fully-resident on a small MoE that fits (GLM-4.7-Flash
   14 GB or Qwen3.6-35B-A3B 23 GB); (b) DeepSeek-V4-Flash 81 GB *runs* on 1 GPU
   with RSS bounded to the budget. Expected speed = the ~1.5 t/s streaming floor.
2. **SLRU expert cache in the arena (keep hot, evict LRU only under budget
   pressure).** `--ssd-stream-budget` bounds resident expert bytes. **Gate:**
   still byte-exact; measure real hit rate + t/s vs the increment-1 floor on
   DeepSeek.
3. **GPU landing + `MUL_MAT_ID` slot indirection** (hot experts in VRAM cache
   slots, GPU computes them), RAM demoted to L2 victim cache. **Gate:** byte-exact
   (reduction-order tolerance for GPU), t/s at GPU compute speed.
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
