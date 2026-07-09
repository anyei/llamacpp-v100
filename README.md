# llama.cpp — V100 Tensor-Parallel Fork

A performance-focused fork of [llama.cpp](https://github.com/ggml-org/llama.cpp)
tuned for **NVIDIA Tesla V100 (Volta, sm70, NVLink)** serving, with working
**tensor parallelism**, tuned **speculative decoding (MTP)**, and experimental
**distributed inference** (coordinator + workers, tensor-parallel "islands").

Reference hardware: 2x V100-SXM2-32GB (NVLink NV2), scaling plan for 4+2+1
GPUs across two machines. Models used for tuning and benchmarks:

- `Qwen3.6-27B-UD-Q4_K_XL-MTP.gguf` (16.4 GB) — primary serving model,
  hybrid attention + DeltaNet with an MTP head for speculation
- `GLM-4.7-Flash-REAP-23B-A3B-UD-Q4_K_XL.gguf` (~14 GB) — MoE / MLA
  experiments, single-GPU profile
- `Qwen3-0.6B-BF16.gguf` — small model for fast distributed/RPC iteration

## Benchmarks — single stream, max t/s per config

**2× V100-SXM2-32GB (NVLink)**, temp-0, `-n 128`, single stream, image built from this
tree. **MTP** = the model's built-in multi-token-prediction head used for self-speculation
(`--spec-type draft-mtp`). Numbers are decode (tg) / prefill (pp) t/s. Every change is
gated on byte-identical / PPL-neutral temp-0 output vs. baseline. Commands to reproduce
are below the tables.

**Qwen3.6-27B** — dense, MTP head (17.9 GB):

| Configuration | Prefill | **Decode** | | Configuration | Prefill | **Decode** |
|---|---:|---:|---|---|---:|---:|
| CPU (20t) | 10.1 | 1.4 | | 1 GPU + MTP | 106.6 | **51** ¹ |
| CPU + MTP | 9.7 | 3.4 | | 2 GPU layer + MTP | 100.9 | 52.6 |
| 1 GPU | 115.6 | 31.5 | | 2 GPU tensor | 100.5 | 48.8 |
| 2 GPU layer | 113.3 | 32.8 | | **2 GPU tensor + MTP** | 89.3 | **81** ¹ |

**Qwen3.6-35B-A3B** — MoE (3B active), MTP head:

| Configuration | Prefill | **Decode** | | Configuration | Prefill | **Decode** |
|---|---:|---:|---|---|---:|---:|
| CPU (20t) | 35.7 | 7.8 | | 1 GPU + MTP | 159.2 | **140** ¹ |
| CPU + MTP | 32.9 | 9.9 | | 2 GPU layer + MTP | 175.7 | 133.1 |
| 1 GPU | 203.3 | 100.9 | | 2 GPU tensor | 138.0 | 110.4 |
| 2 GPU layer | 200.3 | 100.1 | | **2 GPU tensor + MTP** | 122.8 | **165** ¹ |

**GLM-4.7-Flash-UD** — MoE / MLA (deepseek2 arch), no MTP head:

| Configuration | Prefill | **Decode** |
|---|---:|---:|
| CPU (20t) | 35.4 | 9.5 |
| 1 GPU | 185.0 | 89.5 |

Takeaways: MTP adds ~+40-50% (dense) up to ~+35% (MoE); tensor-split helps the dense 27B
(32.8 → 48.8) more than the already-fast MoE; the MoE + MTP stack holds **~166 t/s**
single stream, the dense 27B **~81 t/s** (server steady-state; both scale further at `-np 4`).

¹ MTP rows marked ¹ are the **production server steady-state** (`docker-compose.mtp.yml`,
full config + env gates, long generation, ~88.6% MTP draft acceptance, mean accepted len
3.66), read from `docker logs`: **35B tensor ~166** (peak `tg_3s` 174), **35B 1-GPU ~140**
(peak 148), **27B tensor ~81** (peak 87), **27B 1-GPU ~51** (peak 53). Unmarked MTP rows are `llama-cli` short-run
figures that measure the cold→warm ramp (tg climbs for the first ~500 tokens), so they
read several t/s low vs. the warmed server steady-state.

Reproduce (swap `M` per model; drop the `MTP` fragment for GLM, which has no MTP head):

```bash
MODELS=/mnt/models/ollama37-k80/.ollama/custom-models   # your GGUF dir
IMG=llamacpp-local-v100:latest                          # built --target full/server
M=Qwen3.6-27B-UD-Q4_K_XL-MTP.gguf   # or Qwen3.6-35B-A3B-UD-Q4_K_XL-MTP.gguf, GLM-4.7-Flash-UD-Q4_K_XL.gguf
BASE="-m /models/$M --no-mmap -c 4096 -n 128 --temp 0 --seed 1 -st -p 'Explain how a CPU pipeline works.'"
MTP="--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 1 --spec-draft-p-min 0.75"
run() { docker run --rm --gpus all "$@" -v $MODELS:/models:ro --entrypoint /app/llama $IMG cli; }

run -e CUDA_VISIBLE_DEVICES=    $BASE -ngl 0 -t 20                                   # CPU
run -e CUDA_VISIBLE_DEVICES=    $BASE -ngl 0 -t 20 $MTP                              # CPU + MTP
run -e CUDA_VISIBLE_DEVICES=0   $BASE -ngl 99                                        # 1 GPU
run -e CUDA_VISIBLE_DEVICES=0,1 $BASE -ngl 99 -sm layer                              # 2 GPU layer
run -e CUDA_VISIBLE_DEVICES=0 -e LLAMA_DECODE_GRAPH_CACHE=4   $BASE -ngl 99 $MTP     # 1 GPU + MTP
run -e CUDA_VISIBLE_DEVICES=0,1 -e LLAMA_DECODE_GRAPH_CACHE=4 $BASE -ngl 99 -sm layer $MTP   # 2 GPU layer + MTP
run -e CUDA_VISIBLE_DEVICES=0,1 -e GGML_CUDA_ALLREDUCE=p2p    $BASE -ngl 99 -sm tensor -ts 0.5,0.5   # 2 GPU tensor
run -e CUDA_VISIBLE_DEVICES=0,1 -e GGML_CUDA_ALLREDUCE=p2p -e LLAMA_DECODE_GRAPH_CACHE=4 \
    -e LLAMA_DECODE_GRAPH_CACHE_TOKENS=64 -e GGML_META_MAX_GRAPHS=8 \
    $BASE -ngl 99 -sm tensor -ts 0.5,0.5 $MTP                                        # 2 GPU tensor + MTP (peak)
```

Other measured highlights: KV cache q8_0 = **0.5×** memory (lossless), 27B on a remote
2-GPU RPC TP island **33 t/s** (loopback), and DeepSeek-V4 **81 GB streamed from SSD on
one 32 GB V100** (see the SSD-streaming section below).

## What this fork adds over upstream

### Backend-agnostic improvements (not V100-specific)

These live in backend-neutral code (`common/`, `src/llama-*`, `tools/server/`),
so they help speculative decoding, hybrid/recurrent models, and long-context
serving on **any** ggml backend — CPU, CUDA, Metal, ROCm, Vulkan, RPC — not
just CUDA/Volta. They were tuned and measured here on V100s, but nothing about
them is V100-specific.

- **Multi-shape decode graph cache** (`src/llama-context`) — an LRU of decode
  graphs keyed by ubatch shape, each with its own scheduler, so a workload that
  alternates between a few small shapes (speculative draft/verify, varying
  concurrency) replays cached graphs instead of rebuilding **and reallocating**
  every step. 100% steady-state reuse; build+alloc drops to ~0 ms. Env:
  `LLAMA_DECODE_GRAPH_CACHE` (entries, default 4), `LLAMA_DECODE_GRAPH_CACHE_TOKENS`
  (max cached ubatch size). (`GGML_META_MAX_GRAPHS` is the CUDA-tensor-split
  companion knob.)
- **Speculative draft padding** (`common/speculative`) — drafts padded to a
  fixed length so the verify batch shape stays constant across steps; this is
  what lets graph reuse and confidence-gated drafting (`--spec-draft-p-min`)
  actually pay off instead of forcing a rebuild every iteration. Kill switch:
  `LLAMA_SPEC_DRAFT_NO_PAD=1`.
- **Full-sequence equal ubatch splits** (`src/llama-batch`) — hybrid recurrent
  models (e.g. DeltaNet) no longer collapse to one-sequence-per-ubatch under
  speculative rollback; `split_equal(..., full_seqs=true)` groups whole
  sequences while preserving the rollback-snapshot invariant (the 64.7 → 125 t/s
  concurrent-spec fix).
- **Server prompt-batch defragmentation** (`tools/server`) — prompt processing
  breaks the batch only where a context checkpoint is actually created, not at
  every user message (large win for long chat-history prefill).
- **Prompt-cache checkpoint pruning + RAM bounds** (`tools/server`) — cached
  conversations keep only their newest checkpoints (hybrid-model checkpoints are
  ~150–230 MiB each regardless of token count), preventing multi-GiB cache
  entries and host OOM.
- **Adaptive speculative draft cap** (`common/speculative`, opt-in
  `LLAMA_SPEC_ADAPTIVE=1`) — caps each draft round near the measured acceptance
  EMA to skip cold draft passes. Measured tg-neutral on the MTP config here (the
  confidence gate already captures the value), so it ships **off by default**.
- **Robustness fixes** — clean failure on unreachable `--rpc` endpoints (was a
  silent CPU fallback), on failed context/lora init (was a null-pointer crash),
  and a lora-path double-free.
- **Env-gated diagnostics** (zero cost when unset): `LLAMA_DECODE_TIMING`
  (per-ubatch build/alloc/set/compute + reuse), `LLAMA_SPEC_TIMING`
  (draft/checkpoint/decode/accept phases).

### CUDA / Volta (V100) specific
- **One-shot P2P NVLink AllReduce** for 2-GPU tensor mode
  (`GGML_CUDA_ALLREDUCE=p2p`, NCCL fallback, cap `GGML_CUDA_AR_P2P_MAX_BYTES`).
- **MMVQ sm70 tuning** — a dedicated Volta parameter table for the quantized
  matrix-vector kernels; K-quant batch-1 decode uses `nwarps=2` (+1.8% nospec
  tg, perplexity-identical). Volta was previously served by the generic
  (untuned) path.
- **Quantized KV in tensor mode** — verified lossless at q8_0 (2x); mixed K/V
  types enabled via `GGML_CUDA_FA_ALL_QUANTS`. Note (measured): on Volta,
  quantized KV is for **capacity**, not long-context *speed* — the decode
  flash-attention kernel is compute-bound there, so quantizing KV slightly
  *increases* the long-context penalty.
- **MLA tensor mode** (`deepseek2` family: GLM-4.7-Flash, DeepSeek V2/V3/R1,
  Kimi K2) — attention runs mirrored, FFN/experts split; validated by
  perplexity (statistically identical to single-GPU). Temp-0 text can diverge
  from single-GPU runs (MoE-router-amplified reduction noise) without affecting
  quality. Single GPU remains fastest when the model fits.
- **Meta tensor-split backend diagnostics**: `GGML_META_DEBUG=1|2` (split-state
  + per-src resolution), `GGML_META_DEBUG_REDUCE` (AllReduce boundary
  placement), `GGML_META_TIMING` (compute-vs-reduce wall-time attribution),
  `LLAMA_DEBUG_DUMP_DIR`/`_FILTER` (full-tensor eval-callback dumps),
  `LLAMA_META_DUP_DEVICE` (genuine n-way tensor splits on one GPU).

### Distributed inference (experimental)
- **TP islands**: `ggml-rpc-server --tensor-parallel` exposes all local GPUs
  as one tensor-parallel device over RPC; the coordinator automatically
  computes and uploads per-tensor split states (weights, KV, recurrent-state
  caches). A 27B hybrid model loads sharded (9.2 GB per island GPU,
  slice-packed allocation) and generates coherently, driven entirely over
  the network; reloads take ~40-50 s with the worker weight cache.
- **State integrity over islands**: prompt checkpoints and slot save/restore
  work across RPC (views are resolved to their root tensors on the wire; a
  501 MB / 5259-token 27B state round-trips with byte-identical generation).
- RPC protocol 4.2: split-state upload, device descriptions, buffer-usage
  mirroring, meta-aware (logical-address) sanitization, async command
  markers + events (scheduler pipeline parallelism engages across RPC
  devices), multi-connection workers with per-connection buffer reclaim,
  and fenced worker-to-worker activation transfers (`GGML_RPC_NO_W2W=1`
  to force the old coordinator-bridged copies).
- Pipeline over RPC measured at ~3%/token protocol cost — cross-machine
  `-sm layer` is practical on ordinary Ethernet.

### 🚧 Coming: `--ssd-streaming` — run MoE models bigger than VRAM **+** RAM

> **Status: design validated, build in progress (task 15).** The feasibility
> is measured and committed; the flag and the buffer type are not landed yet.

**The use case.** Today a model has to fit in VRAM, or in VRAM + system RAM
(spilled via `-ngl` / CPU offload). When it doesn't fit *even in VRAM + RAM
combined*, you're stuck — mmap demand-paging thrashes the disk and collapses
to a fraction of a token/sec. `--ssd-streaming` adds the **SSD as a managed
third tier**: the small always-needed weights live resident (GPU/RAM), and the
bulk of the model is held on NVMe and pulled in on demand. The SSD is the
holding hand that lets a model far larger than your memory actually *run* at a
usable speed.

**Why MoE is the sweet spot (any MoE architecture, not just ours).** A
Mixture-of-Experts model activates only a few experts per token, so per step
you only need to *read* that token's experts — not the whole model. That turns
"stream 80 GB per token" into "stream the few MB of experts this token
actually uses," backed by a hot-expert cache in RAM/VRAM. This is
architecture-agnostic: it targets **any** MoE GGUF (Qwen3-MoE, Mixtral,
GPT-OSS, GLM, DeepSeek, …). **GLM-4.7-Flash and DeepSeek-V4-Flash are our two
first-class targets** (and we're validating on DeepSeek first) — but nothing in
the design is specific to them.

**Measured feasibility (this box: 2× V100 = 64 GB VRAM, 46 GB RAM, one NVMe).**
Reference model: `DeepSeek-V4-Flash-IQ2XXS`, **81 GB** — larger than VRAM *and*
RAM, so it can only run this way. 77.9 GB of it is experts (256 per layer,
6 active/token); 8.8 GB is always-resident. Random O_DIRECT reads on the NVMe
sustain **~2.7 GB/s** (they *bypass* the page cache, avoiding the reclaim trap
that sinks naive prefetching). Projected decode, expert-IO-bound:

| Expert-cache hit rate | Projected t/s | vs. today (mmap thrash) |
|---|---|---|
| 0% (pure stream, no cache) | ~1.5 | ~1.1 |
| ~44% (RAM cache, uniform routing) | ~2.6 | — |
| ~80% (realistic routing skew) | ~7.4 | — |
| ~95% (strong skew) | ~30 → compute-bound | — |

Even a cold pure-stream already beats mmap thrash; a hot-expert cache and real
routing locality multiply from there. Design, numbers, and the reproducible
benchmark (`scripts/ssd-stream-bench-odirect.cpp`) are in
[`docs/ssd-streaming-plan.md`](docs/ssd-streaming-plan.md).

## Fork knobs (env gates)

Everything this fork adds is **off by default** and gated by an env var (or CLI
flag), so a stock run behaves like upstream. Full rationale, measured effects, and
usage examples in [`docs/env-gates.md`](docs/env-gates.md); the complete list:

**SSD streaming — CPU tier**

| Env gate | Default | What it does |
|---|---|---|
| `LLAMA_SSD_STREAM_BUFFER` | off | Master switch: stream MoE expert weights from the GGUF on demand (run models bigger than VRAM+RAM). |
| `LLAMA_SSD_STREAM_BUDGET` | 8192 MiB | RAM expert-cache byte budget (LRU-bounded). |
| `LLAMA_SSD_STREAM_READ_THREADS` | 1 | Parallelize SSD miss-path preads (prefill/long-prompt win; decode-neutral). |
| `LLAMA_SSD_STREAM_SLRU` | off | Segmented LRU for the RAM cache instead of plain LRU (measured no-win; opt-in). |
| `LLAMA_SSD_STREAM_PROTECTED_PCT` | 80 | Protected-segment size for the RAM SLRU. |
| `LLAMA_SSD_STREAM_SERIAL` | off | Node-at-a-time execution. **Required for `-ngl 0`** (pure CPU). |
| `LLAMA_SSD_STREAM_NO_ODIRECT` | off | Force buffered reads instead of O_DIRECT (kill-switch / A-B). |
| `GGML_SSD_STREAM_DEBUG` | off | Periodic hit/miss/evict/hit-rate + miss-path timing log. |
| `LLAMA_SSD_STREAMING` | off | Legacy advisory-mmap "residency director" (phase-1 negative result; prefer the streamed buffer). |

**SSD streaming — GPU landing** (VRAM expert-slot cache; single-GPU)

| Env gate | Default | What it does |
|---|---|---|
| `LLAMA_SSD_STREAM_GPU` | off | Compute streamed experts on the GPU via a VRAM slot cache + id→slot indirection. Needs `…_BUFFER=1`. |
| `LLAMA_SSD_STREAM_VRAM_BUDGET` | 4096 MiB | Total VRAM budget for the slot pools. |
| `LLAMA_SSD_STREAM_VRAM_POOLS` | auto | Override the expert-slice class count the budget splits across (auto-detected). |
| `LLAMA_SSD_STREAM_GPU_SLRU` | on | Segmented LRU (scan resistance) for the VRAM cache; `=0` = plain LRU. |
| `LLAMA_SSD_STREAM_GPU_PROTECTED_PCT` | 80 | Protected-segment size for the VRAM SLRU. |
| `LLAMA_SSD_STREAM_GPU_NO_RECLAIM` | off | Kill-switch for the `input_cpy` VRAM reclaim (default shrinks the dead-weight copy). |
| `GGML_OP_OFFLOAD_MIN_BATCH` | 32 | *(upstream)* Min tokens for a `MUL_MAT_ID` to offload; not needed with GPU landing (auto-offloads at batch-1). |

**Speculative decoding / MTP**

| Env gate | Default | What it does |
|---|---|---|
| `LLAMA_SPEC_DRAFT_NO_PAD` | off | Kill-switch for MTP draft-length padding (padding keeps batch shape stable — the default win). |
| `LLAMA_SPEC_ADAPTIVE` | off | Cap each draft round by an acceptance EMA (measured tg-neutral). |
| `LLAMA_SPEC_TIMING` | off | Log server draft/verify/accept timings. |
| `--spec-draft-p-min` *(flag)* | — | Confidence gate: stop drafting below this probability. |

**Tensor-parallel AllReduce (2-GPU tensor split)**

| Env gate | Default | What it does |
|---|---|---|
| `GGML_CUDA_ALLREDUCE=p2p` | NCCL | One-shot P2P NVLink AllReduce for 2 GPUs (falls back to NCCL). |
| `GGML_CUDA_AR_P2P_MAX_BYTES` | 4 MB | Size cap above which P2P defers to NCCL. |
| `GGML_CUDA_FORCE_GRAPHS` | off | Force CUDA graphs on Volta (cc<8.0; no measured gain here). |

**Meta tensor-split backend**

| Env gate | Default | What it does |
|---|---|---|
| `GGML_META_MAX_GRAPHS` | 8 | Shadow-container slots (raise if many decode graph shapes are cached). |
| `GGML_META_DEBUG` | off | Split-state diagnostics (`=2` also prints per-source resolution). |
| `GGML_META_DEBUG_REDUCE` | off | Print AllReduce boundary placement. |
| `GGML_META_TIMING` | off | Per-step compute-vs-reduce timing. |
| `LLAMA_META_DUP_DEVICE` | 1 | Duplicate the device list N× so one GPU runs a genuine N-way split (validation harness). |

**Decode graph cache · Distributed/RPC · KV cache**

| Env gate | Default | What it does |
|---|---|---|
| `LLAMA_DECODE_GRAPH_CACHE` | 4 | Cached small-batch decode graphs for steady-state reuse (`=0` disables). |
| `LLAMA_DECODE_GRAPH_CACHE_TOKENS` | 64 | Max ubatch size eligible for the cache. |
| `GGML_RPC_NO_W2W` | off | Disable direct worker-to-worker tensor pull (bridge through the coordinator). |
| `GGML_RDMA_DEV` / `GGML_RDMA_GID` | auto | RDMA device / GID selection for the RPC transport. |
| `LLAMA_ATTN_ROT_DISABLE` | off | Opt out of Hadamard-rotated KV quantization (auto-on when KV is quantized). |

**Diagnostics** (zero cost unless set): `LLAMA_DECODE_TIMING`, `LLAMA_BATCH_DEBUG`,
`LLAMA_DEBUG_DUMP_DIR`/`_FILTER`, `LLAMA_DSV4_COMPRESS_DEBUG`, `GGML_SCHED_DEBUG`, `GGML_RPC_DEBUG`.

## Documentation map

| Doc | Contents |
|---|---|
| [`docs/perf-tuning-v100.md`](docs/perf-tuning-v100.md) | consolidated results, per-change details, Volta facts, deployment plan |
| [`docs/distributed-inference-guide.md`](docs/distributed-inference-guide.md) | how to run coordinator/workers/TP islands |
| [`docs/distributed-inference-plan.md`](docs/distributed-inference-plan.md) | the distributed design rationale |
| [`docs/validation-playbook.md`](docs/validation-playbook.md) | test scenarios + exact commands used to validate all of this |
| [`docs/ssd-streaming-plan.md`](docs/ssd-streaming-plan.md) | SSD streaming: design, measured results, CPU + GPU-landing tiers (task 15) |
| [`docs/env-gates.md`](docs/env-gates.md) | every fork env gate + CLI flag, grouped, with usage examples |
| [`docs/dev-workflow.md`](docs/dev-workflow.md) | the dev image, how runs/tests are done, correctness gates, the iterative loop |
| [`REBUILD-IMAGE.md`](REBUILD-IMAGE.md) | building the production Docker image |
| [`TASKS.md`](TASKS.md) | full task history with measurements and open items |

## Quick start (single machine, 2x V100)

Build the image and serve with the tuned profile:

```bash
docker build -f .devops/cuda.Dockerfile --target server -t llamacpp-local-v100:latest .

# point the profiles at your GGUF directory (or create llama.cpp/.env with
# MODELS_DIR=... — the compose files default to ./models)
export MODELS_DIR=/path/to/your/ggufs

docker compose -f docker-compose.mtp.yml up -d      # MTP speculation: fastest
# or
docker compose -f docker-compose.nospec.yml up -d   # plain serving
# or
docker compose -f docker-compose-glm-4.7.nospec.yml up -d  # MoE on a single GPU
```

Key flags the profiles use (see the compose files for the full set):
`-sm tensor -ts 0.5,0.5 -fa on` (tensor parallel), `-ctk q8_0 -ctv q8_0`
(lossless half-size KV), `--spec-type draft-mtp --spec-draft-n-max 3
--spec-draft-n-min 1 --spec-draft-p-min 0.75` (speculation),
`--cache-ram 12288 --ctx-checkpoints 8` (bounded prompt-cache RAM).

For multi-machine setups see the
[distributed guide](docs/distributed-inference-guide.md) and
`docker-compose.rpc-worker.yml`.

## Pending items

Tracked in detail in [`TASKS.md`](TASKS.md):

- **SSD streaming** (task 15) — **beta; usable via CLI flags.** Stream MoE
  experts from SSD to run models larger than VRAM+RAM: `--ssd-streaming`
  (RAM-cached expert tier, O_DIRECT), plus `--ssd-stream-gpu` to compute the
  hot experts on the GPU via a persistent VRAM slot cache (auto-offload +
  auto-sized pools). DeepSeek-V4-Flash **81 GB runs on one 32 GB V100 + 46 GB
  RAM**. GPU landing is **byte-exact and a ~3x decode win where the cache covers
  the hot expert set** (Qwen-35B-A3B 2.5 → ~9 t/s); for the >>VRAM extreme
  (DeepSeek) it reaches CPU parity. **GPU landing is single-GPU today** (no-op
  under `-sm tensor`/`-sm layer`); the CPU-tier `--ssd-streaming` works on 2 GPUs
  (layer-split, DeepSeek 81 GB runs) but buys no speedup over one card, and
  `-sm tensor` + DeepSeek crashes at load (meta split-state). Remaining:
  multi-GPU GPU landing, prefetch/overlap, DeepSeek-scale tuning — see
  `docs/ssd-streaming-plan.md §8` and `docs/env-gates.md`.
- **Distributed, hardware-gated** — two-box measurement of the
  worker-to-worker transfers; phase 3 cross-host NCCL (only worth it with
  RDMA / 25 GbE+).
- **Distributed polish** — worker fault tolerance (a dead worker still
  aborts the coordinator); RPC auth/TLS (WireGuard covers this in practice).
- **Minor debt** — CUDA OOM during meta buffer allocation asserts instead
  of erroring cleanly; `-ot` cannot target non-default buffer types.
- Tasks 13 (MLA tensor mode) and 14 (init-failure crashes) closed 2026-07-07.
- Token-generation round closed 2026-07-08: **17** overlap AllReduce with
  compute — negative, reverted (`97dffd25f`); **18** MMVQ sm70 tuning — +1.8%
  batch-1 nospec decode, ppl-identical (`b912d1b1e`); **19** FA long-context
  decay — sized, compute-bound, quant-KV counterproductive for speed, deep
  kernel work deferred (`723cf9fed`); **20** adaptive speculative draft cap —
  measured tg-neutral, ships off (`bc52cf1ea`).

## Reference hardware & platform

Everything in this fork was developed, measured, and validated on one box:

| Component | Detail |
|---|---|
| GPUs | 2x NVIDIA Tesla V100-SXM2-32GB (Volta, cc 7.0, 32 GB each) on an SXM2 carrier |
| GPU interconnect | NVLink NV2 between the pair (`nvidia-smi topo -m`: `NV2`) — this is what makes `-sm tensor` + NCCL fast |
| GPU driver | 580.x (CUDA 13 userspace; images build against CUDA 12.8) |
| CPU | Intel Core i9-10850K, 10c/20t @ 3.6 GHz |
| RAM | 46 GiB (+ zram swap) — note the 27B CPU-side config needs ~11 GiB anonymous memory beyond weights |
| Storage | Models on a consumer NVMe (measured 1.6 GB/s O_DIRECT sequential); bulk HDDs for everything else |
| OS / runtime | Linux (LTS kernel 6.18), Docker 29 with the NVIDIA container runtime; serving and dev builds both containerized |
| Network | Single host — all RPC/distributed numbers are loopback; the 4+2+1 two-box topology is the planned expansion (see the deployment plan in `docs/perf-tuning-v100.md`) |

**Linux-only, and deliberately so.** This fork is developed and tested
exclusively on Linux inside Docker. Upstream llama.cpp supports Windows and
macOS, but nothing added here has ever been run there, and there are no
positive expectations that it works: the distributed transport, `O_DIRECT`
I/O paths, the cgroup-based memory-cap testing methodology, and the compose
deployment profiles are all Linux-specific, and the tuning targets
(NVLink SXM2 V100s) barely exist outside Linux servers anyway. If you try it
on Windows, you are on your own — report findings, but expect breakage.

## Provenance and caveats

- Forked from upstream llama.cpp (see git history for the merge base).
- Tuned specifically for Volta (cc 7.0): CUDA graphs are unavailable there,
  MMQ/MMVQ dispatch thresholds and measured curves are V100-specific.
- The RPC protocol has **no authentication or TLS** — private networks only.
- Everything measured on the reference hardware above; your numbers will vary
  with interconnect, model, and quantization.
