# llama.cpp — V100 Tensor-Parallel Fork

A performance-focused fork of [llama.cpp](https://github.com/ggml-org/llama.cpp)
tuned for **NVIDIA Tesla V100 (Volta, sm70, NVLink)** serving, with working
**tensor parallelism**, tuned **speculative decoding (MTP)**, and experimental
**distributed inference** (coordinator + workers, tensor-parallel "islands").

Reference hardware: 2x V100-SXM2-32GB (NVLink NV2), scaling plan for 4+2+1
GPUs across two machines. Primary model during tuning: Qwen3.6-27B (hybrid
attention + DeltaNet) with MTP speculation.

## Headline results (vs. this fork's own starting point, July 2026)

| Metric | Before | After |
|---|---|---|
| Generation, single stream, MTP spec | 33.6 t/s | **66 t/s** |
| Generation, 4 concurrent, MTP spec (aggregate) | 32.5 t/s | **125.8 t/s** |
| Prefill, long chat history | 860 t/s | **1237 t/s** |
| KV cache memory (lossless q8_0) | 1x | **0.5x** |
| GLM-class MoE, single V100 | — | 79 t/s |
| 27B on a remote 2-GPU TP island (RPC, loopback) | crash | 33.3 t/s, ~12 GB/GPU |

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
- MLA models (`deepseek2` family: GLM-4.7-Flash, DeepSeek V2/V3/R1, Kimi K2)
  rejected cleanly in tensor mode instead of a core dump (proper support is a
  pending item; single GPU or `-sm layer` works).
- Diagnostic instrumentation: `LLAMA_DECODE_TIMING`, `LLAMA_SPEC_TIMING`,
  `GGML_META_DEBUG=1|2` (split-state + per-src resolution tracing).

### Distributed inference (experimental)
- **TP islands**: `ggml-rpc-server --tensor-parallel` exposes all local GPUs
  as one tensor-parallel device over RPC; the coordinator automatically
  computes and uploads per-tensor split states (weights, KV, recurrent-state
  caches). A 27B hybrid model loads sharded (~12 GB per island GPU) and
  generates coherently, driven entirely over the network.
- RPC protocol 4.1: split-state upload, device descriptions, buffer-usage
  mirroring, meta-aware (logical-address) sanitization.
- Pipeline over RPC measured at ~3%/token protocol cost — cross-machine
  `-sm layer` is practical on ordinary Ethernet.

## Documentation map

| Doc | Contents |
|---|---|
| [`docs/perf-tuning-v100.md`](docs/perf-tuning-v100.md) | consolidated results, per-change details, Volta facts, deployment plan |
| [`docs/distributed-inference-guide.md`](docs/distributed-inference-guide.md) | how to run coordinator/workers/TP islands |
| [`docs/distributed-inference-plan.md`](docs/distributed-inference-plan.md) | the distributed design rationale |
| [`REBUILD-IMAGE.md`](REBUILD-IMAGE.md) | building the production Docker image |
| [`TASKS.md`](TASKS.md) | full task history with measurements and open items |

## Quick start (single machine, 2x V100)

Build the image and serve with the tuned profile:

```bash
docker build -f .devops/cuda.Dockerfile --target server -t llamacpp-local-v100:latest .
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

- **MLA tensor-parallelism** (task 13) — deepseek2-family models currently
  refuse `-sm tensor`; a WIP mirrored-attention path exists behind
  `LLAMA_TENSOR_MLA_WIP=1` but produces corrupted output. Matters if a
  DeepSeek-family model becomes the flagship.
- **Distributed phase 2** (task 12) — async RPC double-buffering to cut the
  per-token latency of the synchronous protocol (mainly benefits the
  whole-model-remote case; parked pending real cross-host hardware).
- **Distributed polish** — final E2E validation of slice-packed island
  generation (load path verified at ~12 GB/GPU); adopted-alias shadow
  lifetime hardening; worker fault tolerance (a dead worker still aborts the
  coordinator); RPC auth/TLS.
- **Meta-device teardown segfault** (task 14) — freeing a tensor-split model
  without a context crashes; only reachable through error paths.
- **Distributed phase 3** — cross-host NCCL tensor parallelism; only worth it
  with RDMA / 25 GbE+.

## Provenance and caveats

- Forked from upstream llama.cpp (see git history for the merge base);
  upstream remote kept as `origin` for future merges.
- Tuned specifically for Volta (cc 7.0): CUDA graphs are unavailable there,
  MMQ/MMVQ dispatch thresholds and measured curves are V100-specific.
- The RPC protocol has **no authentication or TLS** — private networks only.
- Everything measured on the reference hardware above; your numbers will vary
  with interconnect, model, and quantization.
