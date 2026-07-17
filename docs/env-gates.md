# Environment Gates — V100 Fork Reference

This fork adds a number of **environment-variable gates** to enable, tune, and
debug its features (tensor parallelism, MTP speculation, SSD streaming, the meta
tensor-split backend, distributed inference, and instrumentation). This page
consolidates them.

Conventions:
- **Boolean** gates are "set = on" unless noted: any non-empty value (often `=1`)
  enables; unset = default. A few default *on* and take `=0` to disable.
- **Value** gates take a number (MiB, bytes, count, ...).
- Defaults are what you get when the variable is **unset**.
- These are this fork's gates. The stock ggml backend knobs (`GGML_VK_*`,
  `GGML_METAL_*`, `GGML_HEXAGON_*`, `GGML_CUDA_*` upstream ones, `HF_*`, ...) are
  not listed here; a few upstream gates that matter for a fork feature are marked
  *(upstream)*.

---

## 1. SSD streaming — CPU tier (task 15)

Run MoE models whose expert weights exceed VRAM+RAM by streaming routed experts
from SSD into a bounded RAM cache. See `docs/ssd-streaming-plan.md`.

> The common ones have **CLI flags** (preferred over the env vars):
> `--ssd-streaming`, `--ssd-stream-budget <MiB>`, `--ssd-stream-gpu`,
> `--ssd-stream-vram-budget <MiB>`. E.g.
> `llama-server -m model.gguf -ngl 99 --ssd-streaming --ssd-stream-gpu --ssd-stream-vram-budget 6000`.
> The remaining knobs below are env-only tuning/diagnostics.

| Gate | Type | Default | What it does |
|---|---|---|---|
| `LLAMA_SSD_STREAM_BUFFER` | bool | off | Master switch: route `blk.*.ffn_*_exps.weight` to the streamed buffer type (RAM cache, filled on demand from the GGUF). |
| `LLAMA_SSD_STREAM_BUDGET` | MiB | 8192 | RAM expert-cache byte budget (LRU-bounded resident experts). |
| `LLAMA_SSD_STREAM_READ_THREADS` | count | 1 | Parallelize the SSD miss-path preads across N workers (each with its own O_DIRECT bounce). Helps the **batch-amortized** path — prefill/long prompts (+31% on DeepSeek-81GB), saturates at 2; single-stream decode is neutral. Default 1 = serial. |
| `LLAMA_SSD_STREAM_SLRU` | bool | off (LRU) | Use a segmented LRU (probation/protected) for the RAM cache instead of plain LRU. Measured no-win for DeepSeek; kept opt-in. |
| `LLAMA_SSD_STREAM_PROTECTED_PCT` | 0-100 | 80 | Protected-segment size for the RAM SLRU (only when SLRU on). |
| `LLAMA_SSD_STREAM_SERIAL` | bool | off | Force node-at-a-time execution. **Required for `-ngl 0`** (pure CPU), where the router and experts share a scheduler split. Not needed when `-ngl > 0`. |
| `LLAMA_SSD_STREAM_NO_ODIRECT` | bool | off | Force buffered reads instead of O_DIRECT (kill-switch / A-B diagnostic; O_DIRECT is the default and keeps the page-cache/cgroup charge bounded). |
| `GGML_SSD_STREAM_DEBUG` | bool | off | Periodic hit/miss/evict/hit-rate/bytes-read log line. |
| `LLAMA_SSD_STREAMING` | bool | off | **Legacy/experimental** advisory mmap "residency director" (phase-1 negative result). Distinct mechanism from the streamed buffer above; shares `LLAMA_SSD_STREAM_BUDGET`. Prefer the streamed buffer. |

## 2. SSD streaming — GPU landing (increment 3)

Compute streamed experts on the GPU with a persistent VRAM slot cache
(hot experts stay resident across tokens; a cache hit is a zero-copy GPU read).
Composes with the CPU tier above (RAM becomes the L2 victim tier).

> **Single-GPU only (today).** GPU landing is a no-op under multi-GPU (`-sm tensor`
> or `-sm layer`); the slot cache doesn't allocate. On 2 GPUs, use plain
> `--ssd-streaming` (CPU tier, layer-split) - it works and is coherent, but gives
> no speedup over one GPU. See `docs/ssd-streaming-plan.md §8.3`.

| Gate | Type | Default | What it does |
|---|---|---|---|
| `LLAMA_SSD_STREAM_GPU` | bool | off | Enable the VRAM expert-slot cache + id->slot indirection. Requires `LLAMA_SSD_STREAM_BUFFER=1`. |
| `LLAMA_SSD_STREAM_VRAM_BUDGET` | MiB | 4096 | **Total** VRAM budget for the slot pools. |
| `LLAMA_SSD_STREAM_GPU_NO_RECLAIM` | bool | off | Kill-switch: keep the full-size `input_cpy` gallocr reservation for streamed-expert matmuls. By default the copy is shrunk to one slice (its data is redirected to the slot pool anyway) — reclaiming ~0.3-0.8 GB of compute-arena VRAM. Set to A-B the reclaim. |
| `LLAMA_SSD_STREAM_VRAM_POOLS` | count | auto | Override the number of expert-slice classes the budget is split across. **Auto-detected** from the model (e.g. 2 for DeepSeek-V4, 4 for mixed-quant UD Qwen); only set this to override. |
| `LLAMA_SSD_STREAM_GPU_SLRU` | bool | on | Segmented-LRU (scan resistance) for the VRAM slot cache; `=0` forces plain LRU. (Measured neutral for DeepSeek; may help more-skewed models.) |
| `LLAMA_SSD_STREAM_GPU_PROTECTED_PCT` | 0-100 | 80 | Protected-segment size for the VRAM SLRU. |
| `GGML_OP_OFFLOAD_MIN_BATCH` | count | 32 | *(upstream)* Min tokens for a `MUL_MAT_ID` to offload to the GPU. **Not needed for GPU landing** - streamed-expert matmuls auto-offload at batch-1 when `LLAMA_SSD_STREAM_GPU=1`. |

## 3. Speculative decoding / MTP (tasks 1, 20)

| Gate | Type | Default | What it does |
|---|---|---|---|
| `LLAMA_SPEC_DRAFT_NO_PAD` | bool | off | Kill-switch: disable MTP draft-length padding to fixed `n_max`. Padding keeps the batch shape stable for graph reuse and is the default win (33.6 -> 62-65 t/s); only set this to A-B the effect. |
| `LLAMA_SPEC_ADAPTIVE` | bool | off | Cap each draft round by an acceptance EMA. Measured tg-neutral; kept off. |
| `LLAMA_SPEC_TIMING` | bool | off | Log server speculation phase timings (draft/verify/accept). |

## 4. Tensor-parallel AllReduce over NVLink (task 4)

| Gate | Type | Default | What it does |
|---|---|---|---|
| `GGML_CUDA_ALLREDUCE` | `p2p` | NCCL | `=p2p` uses a one-shot P2P NVLink AllReduce for 2 GPUs (falls back to NCCL). |
| `GGML_CUDA_AR_P2P_MAX_BYTES` | bytes | 4 MB | Size cap above which the P2P path defers to NCCL. |
| `GGML_CUDA_AR_COPY_THRESHOLD` | bytes | tuned | Byte threshold for the P2P AllReduce copy path (tuning knob). |
| `GGML_CUDA_AR_COPY_CHUNK_BYTES` | bytes | 0 (off) | Chunk size for the P2P copy path; 0 keeps it unchunked. |
| `GGML_CUDA_AR_BF16_THRESHOLD` | count | 1 | Element threshold above which the reduce runs in bf16. |
| `GGML_CUDA_FORCE_GRAPHS` | bool | off | Force CUDA graphs on Volta (cc<8.0, where they're runtime-disabled). Opt-in; no measured gain here. |

## 5. Meta tensor-split backend (tasks 2, 8, 13, 16, 17)

The meta backend wraps N GPUs as one device for tensor parallelism.

| Gate | Type | Default | What it does |
|---|---|---|---|
| `GGML_META_DEBUG` | 1 / 2 | off | Split-state diagnostics; `=2` also prints per-source resolution (`SRC_RESOLVE`). |
| `GGML_META_DEBUG_REDUCE` | bool | off | Print AllReduce boundary placement (`partial N -> boundary M`). |
| `GGML_META_MAX_GRAPHS` | count | 8 | Compute-ring shadow-container slots (raise if many decode graph shapes are cached). |
| `GGML_META_TIMING` | bool | off | Per-step compute vs reduce timing for the meta device. |
| `LLAMA_META_EP_ONLY` | bool | off | Expert-parallel split shape: segment only `ffn_*_exps` across meta members, mirror everything else (attention, dense FFN, shared experts, output head). Also a fault isolator for split-policy bugs (task 28). |
| `LLAMA_META_ATTN_OWNER` | index | -1 (off) | Dedicate attention/KV/router to meta member `<j>` (a local GPU): non-owners get zero-size attention slices, the exit projection is dot-dim split so its output derives PARTIAL, and the delayed AllReduce broadcasts the owner's activations back (x+0=x). The owner takes a **0** expert share. The core of the cross-box expert-parallel topology; composes with `LLAMA_META_EP_ONLY` (task 28 increment 3). **Constraint: exactly ONE local (non-RPC) member — the attention owner. A second local GPU as an expert member corrupts the reduce (garbage output) and is rejected at load (task 48); use both local GPUs via single-box `-sm tensor -ncmoe` instead, or put experts on RPC workers.** |
| `GGML_META_SURGICAL_MAX_STATE_MIB` | MiB | 64 | Cap on replayed non-weight state size for the meta backend's surgical re-provision path. |
| `GGML_META_NO_DELAY` | bool | off | Reduce at every PARTIAL node instead of delaying AllReduce past the expert-merge tree. Diagnostic. |
| `GGML_META_NO_STAR` | bool | off | Disable the star reduce (batched-read partials + host sum + async broadcast, used when a local member can root the reduce); fall back to the fold+butterfly. A/B + diagnostic. |
| `GGML_META_NO_FUSED` | bool | off | Disable the fused boundary pipeline (proto 4.5: reduced value + CHAIN of subgraphs up to the next reduce + next-partial request in ONE message per wire member per boundary). Also auto-disabled under GGML_META_TIMING, GGML_META_NO_STAR and against pre-4.5 workers. A/B + diagnostic. |
| `GGML_META_FUSED_BCAST` | 0/1/2 | 2 | Broadcast-boundary side of the fused pipeline: 0 = plain copies (no carriage), 1 = fused SET only, 2 = full chain carriage (default). Diagnostic A/B levels. |
| `LLAMA_TRUNC_ARR` | bool | off | Accept longer-than-n_layer per-layer arrays and partial tensor loads, so `--override-kv <arch>.block_count=int:N` can truncate a model into a small fast reproducer. Debug only. |
| `LLAMA_META_DUP_DEVICE` | count | 1 | Duplicate the device list N times so ONE physical GPU runs a genuine N-way split (validation harness; e.g. `=2` reproduces 2-GPU exactness on one card). |

## 6. Decode graph cache (task 8)

| Gate | Type | Default | What it does |
|---|---|---|---|
| `LLAMA_DECODE_GRAPH_CACHE` | count | 4 | Number of small-batch decode graphs to cache (each with its own scheduler). `=0` disables. |
| `LLAMA_DECODE_GRAPH_CACHE_TOKENS` | count | 64 | Max ubatch size eligible for the cache. |

## 7. Distributed inference / RPC (task 12)

| Gate | Type | Default | What it does |
|---|---|---|---|
| `GGML_RPC_NO_W2W` | bool | off | Disable direct worker-to-worker tensor pull; fall back to bridged copies through the coordinator. |
| `GGML_RPC_THREADPOOL_POLL` | 0-100 | unset (off) | Worker-side (`rpc-server`): attach a PERSISTENT threadpool to CPU backends (unset = a disposable pool is created+joined per subgraph - measurable per-boundary turnaround on EP workers). The value is the busy-poll level: 0 = persistent but condvar waits, >0 = spin between subgraphs (trades idle cores for wake latency). Compose: `WORKER_THREADPOOL_POLL`. Loopback A/B: +2% decode, token-identical. |
| `GGML_RPC_CACHE_LIMIT_MIB` | MiB | 0 (off) | Worker-side (`rpc-server -c`): cap the tensor-cache dir; after each save, evict least-recently-USED entries (serves refresh mtime) until it fits. Also enforced ONCE AT STARTUP (task 45) so an idle worker holding stale model generations trims before the next load. The beacon publishes the current cache size (`cache_mib=`, fleet UI "Cache" column). Compose: `WORKER_CACHE_LIMIT_MIB`. A deleted cache dir is also recreated on the next save now (used to silently kill persistence until restart). |
| `GGML_RPC_NO_SRC_HINT` | bool | off | Disable the proto-4.8 slice-provenance path (task 44): coordinator falls back to plain `SET_TENSOR_HASH` offers. With it ON (default), EP/tensor-mode weight slices carry (tensor name, offset, row geometry) so a `--model-dir` worker serves them by pread from its local GGUF regardless of where the `-ts` split boundaries fall - share changes no longer cold-stream. Every serve is hash-verified; a stale local file degrades to streaming, never corruption. |
| `LLAMA_FLEET_CAPACITY_CHECK` | bool | on | (server) Fleet capacity gate: when RPC devices are in the pipeline, refuse to START a load whose weights + KV reserve exceed the pooled free device memory - the server waits in `waiting-capacity` state (visible in `/fleet/status` + fleet UI) and, under `--rpc-discover`, exits 42 automatically once newly beaconing workers raise the pool enough (restart policy re-discovers and loads). `=0` disables. Single-box (no-RPC) runs are never gated (mmap paging / -ncmoe / ssd-streaming are supported over-capacity regimes). |
| `LLAMA_FLEET_KV_RESERVE_MB` | MiB | 20480 | Headroom the capacity gate adds on top of the model weight bytes (KV + compute buffers + fragmentation margin). |
| `LLAMA_RPC_NO_SURGICAL` | bool | off (surgical ON) | With `--rpc-reload`: disable the surgical re-provision (returned worker's share replayed from its own cache, ~2min for a 48GB share vs ~10+min reload; falls back to the reload on any failure) and always do the full in-process reload. `LLAMA_RPC_SURGICAL_WAIT_S` (120) = how long to wait for a dead endpoint to return; `GGML_RPC_JOURNAL_MAX_MIB` (4096) = small-write spill cap; `GGML_RPC_REPROVISION_VERIFY=1` = read back and hash-verify every replayed region. |
| `LLAMA_ARG_RPC_RELOAD` | bool | off | (= `--rpc-reload`, server only) On RPC worker loss: fail in-flight requests, then reload the model IN-PROCESS across the workers reachable at that moment (dead workers drop with their positional `-ts` shares; a returned worker is re-included by the next failure-triggered reload; all-dead degrades to local-only loudly; a load that fails - fleet-sized models - retries every 10s). Default off = #29b behavior: exit 42 for the restart policy. |
| `LLAMA_ARG_RPC_SKIP_UNAVAILABLE` | bool | off | (= `--rpc-skip-unavailable`) Drop unreachable `--rpc` servers with a warning and split the model across the remaining devices, instead of exiting with an error. Load-time; a worker dying mid-session is handled separately (29b: requests error cleanly, server exits for restart+rediscovery). |
| `LLAMA_ARG_RPC_DISCOVER` | bool | off | (= `--rpc-discover`) Discover RPC workers announcing themselves on the LAN (`rpc-server --announce`) and use them; composes with `--rpc`, duplicates skipped. Trusted networks only. |
| `LLAMA_KV_WORKER_HOST` | bool | off | KV annex (task 30): a worker exposing GPU+CPU (`rpc-server -d CUDA0,CPU`, e.g. `WORKER_EXTRA_ARGS="-d CUDA0,CPU"`) gets its CPU device reserved — no layers, but the KV cache of that worker's GPU layers lives there (worker RAM). Weights stay in VRAM → a small-VRAM card holds ~2× the layers; measured ~7% decode cost (0.6B/32k loopback). |
| `LLAMA_ARG_RPC_DISCOVER_GROUP` | str | built-in | (= `--rpc-discover-group`) Multicast group `ADDR:PORT` for discovery; must match the workers' `--announce-group`. |
| `GGML_CUDA_ERROR_CONTAIN` | bool | off (rpc-server sets 1) | CUDA errors throw to the RPC compute boundary instead of aborting the process. Armed automatically by `rpc-server` (task 29e: a compute error drops one connection, the worker lives; 3 consecutive failures = poisoned backend → clean exit for the restart policy). Set `=0` on a worker to restore abort-with-backtrace. Never armed in coordinators/tools. |
| `GGML_CUDA_INJECT_COMPUTE_FAIL` | int | off | Fault injection for the containment path: `N>0` fails the Nth graph compute and after (poisoned backend); `N<0` fails only the \|N\|th (isolated error — the worker must recover). |
| `GGML_RPC_TIMING` | bool | off | Worker-side (`rpc-server`): per-command latency breakdown - `lock avg` (cross-connection exec-mutex contention wait) and `exec avg` / `exec max` (the handler, i.e. graph_compute + response send) per command type, logged every 20k commands and at each connection close. Decomposes the coordinator's end-to-end RTT (the fleet UI's per-worker `ms`, from `rpc_ep_stat`) into worker-processing vs network+coordinator residual - the direct measurement of the EP boundary turnaround (task 28). Look at `GRAPH_RECOMPUTE_UID`/`GRAPH_FUSED` for the per-boundary compute, `GET_TENSOR` for star-reduce partial reads. |
| `GGML_RDMA_DEV` / `GGML_RDMA_GID` | str | auto | RDMA device / GID selection for the RPC transport (when built with RDMA). |
| `GGML_RPC_DEBUG` | bool | off | *(upstream)* RPC command logging. |
| `GGML_RPC_STATS` | bool | off | Client-side per-RPC-command call/byte counters, dumped to stderr at exit. Diagnostic; used to find the O(n^2) graph serialization and the per-row weight upload (task 28 increment 1). |
| `GGML_RPC_DEBUG_FAIL_ALLOC` | `EP:SKIP:COUNT` | off | Fault injection: on endpoint `EP`, skip the first SKIP buffer allocations, fail the next COUNT, pass the rest (and log every request). Reproduces a worker rejecting a compute-buffer alloc (task 37 Run-B scenario: fail the PP compute buffer, let the no-pipeline retry pass). |
| `GGML_RPC_DEBUG_BUF` | bool | off | Client-side buffer lifecycle log: every remote alloc (endpoint, remote_ptr, size) and free. Used to find the zero-size-chunk dummy buffer in task 37. |
| `LLAMA_RPC_AUTO_WEIGHT_RESERVE_MB` | MiB | 2% of model, clamped [512, 2048] | Override the per-device compute-buffer allowance `--rpc-auto-weight` subtracts from every device cap (task 37; KV is estimated from the GGUF header separately). Bump it if V4-class members OOM at decode (the default reserve can undershoot the actual compute buffer). |
| `LLAMA_ARG_RPC_AUTO_WEIGHT` | bool | off | (= `--rpc-auto-weight`) Fill an unset `-ts` by each device's measured bandwidth score instead of by free memory, water-filled against capacity; local GPUs are benchmarked at startup. Also applies to `-sm tensor` EP splits (task 28 increment 3). Explicit `-ts` always wins. |
| `GGML_RPC_SCORE` | bool | off | Worker-side (= `rpc-server --score`): run a ~1s matvec benchmark at startup (effective memory bandwidth, the decode-bound quantity) and publish the score in the discovery beacon + over RPC, for `--rpc-auto-weight`. Benched ONCE at startup — restart the worker when the box is idle if a busy start under-read it (TASKS.md #42). |
| `GGML_RPC_ALLOW_SHUTDOWN` | bool | off | Worker-side (= `rpc-server --allow-shutdown`): permit the coordinator to restart this worker over RPC (`POST /fleet/worker/restart`). Off = the worker refuses shutdown commands. |
| `LLAMA_ARG_FLEET_ADMIN` | bool | off | (= `--fleet-admin`, server) Enable `POST /fleet/worker/restart` and `POST /fleet/reload`. Requires an `--api-key` (these are remote-kill/reload primitives on an unauthenticated RPC fabric). |
| `LLAMA_ARG_FLEET_PREFLIGHT` | path | off | (= `--fleet-preflight <gguf>`, server) Before the main load, benchmark a small model across the SAME devices/split (times single-token decodes) and publish the result in `/fleet/status`. A small dense model's compute is negligible, so the number is the fleet's per-token boundary/latency floor — an upper bound for any model on this topology, not a throughput estimate. Never runs on a resume/failure reload. |

## 8. KV cache (task 10)

| Gate | Type | Default | What it does |
|---|---|---|---|
| `LLAMA_ATTN_ROT_DISABLE` | bool | off | Opt out of Hadamard-rotated KV quantization (auto-enabled whenever KV is quantized; the rotation is what makes q4_0 KV near-lossless). |

## 8b. CUDA FlashAttention kernel selection (task 32)

| Gate | Type | Default | What it does |
|---|---|---|---|
| `GGML_CUDA_FA_NO_MMA` | bool | off | Never select the MMA FlashAttention kernel (tile/vec instead). For devices that pass the cc gate but can't run it — GTX 16xx (TU116/117) is cc 7.5 without tensor cores. Set on the affected *worker*; without it the first failure self-heals per device (WARN + tile fallback) instead of aborting. |
| `GGML_CUDA_FA_MMA_FORCE_SMEM_FAIL` | bool | off | Fault injection: pretend the MMA kernel's shared-memory opt-in failed, to exercise the fallback path on healthy hardware. |

## 9. Instrumentation & debug

| Gate | Type | Default | What it does |
|---|---|---|---|
| `LLAMA_DECODE_TIMING` | bool | off | Per-ubatch build / alloc / set-inputs / compute timing. |
| `LLAMA_BATCH_DEBUG` | 1 / 2 | off | Dump batch contents (`=2` = full per-token dump). |
| `LLAMA_DEBUG_DUMP_DIR` | path | off | Dump full tensors to a directory for elementwise diffing (pairs with the eval-callback). |
| `LLAMA_DEBUG_DUMP_FILTER` | substr | none | Only dump tensors whose name matches this substring. |
| `LLAMA_DSV4_COMPRESS_DEBUG` | bool | off | DeepSeek-V4 KV-compression debug logging. |
| `GGML_SCHED_DEBUG` | 1 / 2 | off | *(upstream)* Scheduler split/backend-assignment dump. |

---

## Usage examples

### Run a huge MoE from SSD on one GPU (CPU-compute expert tier)

DeepSeek-V4-Flash 81 GB on a single 32 GB V100 + 46 GB RAM (experts stream to a
30 GB RAM cache, non-experts on the GPU):

```sh
docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES=0 \
  -e LLAMA_SSD_STREAM_BUFFER=1 \
  -e LLAMA_SSD_STREAM_BUDGET=30720 \
  -e GGML_SSD_STREAM_DEBUG=1 \
  -v /path/to/models:/models:ro \
  --entrypoint /app/llama llamacpp-local-v100:latest cli \
  -m /models/DeepSeek-V4-Flash-...gguf -ngl 99 --no-mmap -c 4096 -n 96 --temp 0 -st -v \
  -p "Explain how a CPU pipeline works."
```

Pure-CPU (no GPU) variant adds `-e LLAMA_SSD_STREAM_SERIAL=1` and `-ngl 0`.
Equivalent CLI flags (no env needed): `--ssd-streaming --ssd-stream-budget 30720`
(+ `--ssd-stream-gpu --ssd-stream-vram-budget <MiB>` for the VRAM slot cache).

### GPU landing (experts computed on the GPU via a VRAM slot cache)

Best on models whose hot expert set fits the VRAM cache (big win: Qwen-35B-A3B
2.5 -> 7 t/s). Note `VRAM_POOLS` should match the model's expert-slice classes:

Streamed-expert matmuls auto-offload to the GPU and the VRAM budget auto-splits
across the model's expert-slice classes, so only the two budgets are needed:

```sh
# Qwen-35B-A3B: cache covers the hot set -> big win (~9 t/s decode), byte-exact
-e LLAMA_SSD_STREAM_BUFFER=1 -e LLAMA_SSD_STREAM_BUDGET=24000 \
-e LLAMA_SSD_STREAM_GPU=1    -e LLAMA_SSD_STREAM_VRAM_BUDGET=6000

# DeepSeek-V4 81GB (experts >> VRAM): break-even with CPU; watch the hit rate
-e LLAMA_SSD_STREAM_BUFFER=1 -e LLAMA_SSD_STREAM_BUDGET=24000 \
-e LLAMA_SSD_STREAM_GPU=1    -e LLAMA_SSD_STREAM_VRAM_BUDGET=12000 \
-e GGML_SSD_STREAM_DEBUG=1
```

### 2-GPU tensor parallelism with fast NVLink AllReduce

```sh
-e CUDA_VISIBLE_DEVICES=0,1 -e GGML_CUDA_ALLREDUCE=p2p \
  ... -m model.gguf -ngl 99 -sm tensor -ts 0.5,0.5
```

### MTP speculation (production default)

Padding is on by default; no env needed. To measure its effect, A-B with the
kill-switch and the timing log:

```sh
-e LLAMA_SPEC_DRAFT_NO_PAD=1 -e LLAMA_SPEC_TIMING=1   # baseline (padding off)
```

### Reproduce a 2-GPU tensor-split on one physical GPU (validation)

```sh
-e CUDA_VISIBLE_DEVICES=0 -e LLAMA_META_DUP_DEVICE=2 \
-e GGML_META_DEBUG_REDUCE=1  ... -sm tensor
```

### Diagnose a numeric divergence (full-tensor dumps)

```sh
-e LLAMA_DEBUG_DUMP_DIR=/tmp/dump -e LLAMA_DEBUG_DUMP_FILTER=ffn_gate \
-e LLAMA_DECODE_TIMING=1
```

### Distributed inference (coordinator + RPC workers)

Worker exposes its GPUs as one tensor-parallel island; the coordinator pushes
split states. Direct worker-to-worker pull is on by default; disable to compare:

```sh
# worker
ggml-rpc-server --tensor-parallel -p 50060
# coordinator
CUDA_VISIBLE_DEVICES="" llama-server -m model.gguf --rpc host:50060 -ngl 99
# (optional) force bridged copies instead of W2W pull:
-e GGML_RPC_NO_W2W=1
```

See `docs/distributed-inference-guide.md` for the full topology and security notes.
