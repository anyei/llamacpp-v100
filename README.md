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

## Headline results (vs. this fork's own starting point, July 2026)

| Metric | Before | After |
|---|---|---|
| Generation, single stream, MTP spec | 33.6 t/s | **66 t/s** |
| Generation, 4 concurrent, MTP spec (aggregate) | 32.5 t/s | **125.8 t/s** |
| Prefill, long chat history | 860 t/s | **1237 t/s** |
| KV cache memory (lossless q8_0) | 1x | **0.5x** |
| GLM-class MoE, single V100 | — | 79 t/s |
| 27B on a remote 2-GPU TP island (RPC, loopback) | crash | 33 t/s, 9.2 GB/GPU |

Every change was gated on **byte-identical temp-0 output** vs. baseline.

## What this fork adds over upstream

### Serving performance
- **Multi-shape decode graph cache** — LRU of small-batch decode graphs, each
  with its own scheduler; 100% graph reuse for speculative workloads.
  (`LLAMA_DECODE_GRAPH_CACHE`, `LLAMA_DECODE_GRAPH_CACHE_TOKENS`,
  `GGML_META_MAX_GRAPHS`)
- **MTP draft padding** — fixed-length drafts keep batch shapes stable so
  graph reuse works with confidence-based drafting (`--spec-draft-p-min`).
- **Full-sequence equal ubatch splits** — hybrid (recurrent) models no longer
  fall back to one-sequence-per-ubatch under speculative rollback; the 4-slot
  verify runs as one batch (the 64.7 → 125 t/s fix).
- **Prompt-batch defragmentation** — prompt processing no longer breaks the
  batch at every user message, only where a context checkpoint is created.
- **Prompt-cache checkpoint pruning + RAM bounds** — cached conversations keep
  only their newest checkpoints (hybrid-model checkpoints are ~150-230 MiB
  each); compose profiles bound cache/checkpoint RAM to prevent host OOM.
- **One-shot P2P NVLink AllReduce** for 2-GPU tensor mode
  (`GGML_CUDA_ALLREDUCE=p2p`).

### Correctness / robustness
- Quantized KV verified in tensor mode (q8_0 lossless at 2x; mixed K/V types
  enabled via `GGML_CUDA_FA_ALL_QUANTS`).
- **MLA tensor mode** (`deepseek2` family: GLM-4.7-Flash, DeepSeek V2/V3/R1,
  Kimi K2): supported — attention runs mirrored, FFN/experts split; validated
  by perplexity (statistically identical to single-GPU). Temp-0 text can
  diverge from single-GPU runs (MoE-router-amplified reduction noise) without
  affecting quality. Single GPU remains fastest when the model fits.
- Clean failure paths: unreachable `--rpc` endpoints, failed context creation,
  and a lora-path double-free all fixed (previously silent CPU fallback /
  null-pointer crashes).
- Diagnostic instrumentation: `LLAMA_DECODE_TIMING`, `LLAMA_SPEC_TIMING`,
  `GGML_META_DEBUG=1|2` (split-state + per-src resolution tracing),
  `GGML_META_DEBUG_REDUCE` (AllReduce boundary placement),
  `LLAMA_DEBUG_DUMP_DIR`/`_FILTER` (full-tensor dumps from the eval callback),
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

## Documentation map

| Doc | Contents |
|---|---|
| [`docs/perf-tuning-v100.md`](docs/perf-tuning-v100.md) | consolidated results, per-change details, Volta facts, deployment plan |
| [`docs/distributed-inference-guide.md`](docs/distributed-inference-guide.md) | how to run coordinator/workers/TP islands |
| [`docs/distributed-inference-plan.md`](docs/distributed-inference-plan.md) | the distributed design rationale |
| [`docs/validation-playbook.md`](docs/validation-playbook.md) | test scenarios + exact commands used to validate all of this |
| [`docs/ssd-streaming-plan.md`](docs/ssd-streaming-plan.md) | SSD streaming: design, measured results, and usage of the experimental `LLAMA_SSD_STREAMING` director (task 15) |
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

- **SSD streaming** (task 15) — the investigation is complete (dense CPU
  streaming: no win over kernel readahead; MoE under a memory cap: 10x
  headroom exists but page-cache steering is self-defeating). The remaining
  build is a userspace expert cache (pread/O_DIRECT arena + SLRU, one
  design for CPU and GPU tiers) — see `docs/ssd-streaming-plan.md`.
- **Distributed, hardware-gated** — two-box measurement of the
  worker-to-worker transfers; phase 3 cross-host NCCL (only worth it with
  RDMA / 25 GbE+).
- **Distributed polish** — worker fault tolerance (a dead worker still
  aborts the coordinator); RPC auth/TLS (WireGuard covers this in practice).
- **Minor debt** — CUDA OOM during meta buffer allocation asserts instead
  of erroring cleanly; `-ot` cannot target non-default buffer types.
- Tasks 13 (MLA tensor mode) and 14 (init-failure crashes) closed 2026-07-07.

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

- Forked from upstream llama.cpp (see git history for the merge base);
  upstream remote kept as `origin` for future merges.
- Tuned specifically for Volta (cc 7.0): CUDA graphs are unavailable there,
  MMQ/MMVQ dispatch thresholds and measured curves are V100-specific.
- The RPC protocol has **no authentication or TLS** — private networks only.
- Everything measured on the reference hardware above; your numbers will vary
  with interconnect, model, and quantization.
