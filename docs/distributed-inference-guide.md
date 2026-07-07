# Distributed Inference — Usage Guide

How to run llama.cpp across multiple GPUs and machines with this fork's
coordinator/worker support. Design rationale: `distributed-inference-plan.md`.
Status/history: `TASKS.md` item 12. All numbers measured on 2x Tesla
V100-SXM2-32GB with Qwen3.6-27B-UD-Q4_K_XL unless noted.

## TL;DR — which mode for which situation

| Situation | Mode | Measured |
|---|---|---|
| All GPUs in one machine | single process, `-sm tensor` + NCCL — do NOT use RPC | 47.8 t/s |
| Model fits one machine, second box idle | leave it idle, or serve another model on it | — |
| Model does NOT fit one machine | pipeline between boxes: `-sm layer --rpc worker:port` | ~3%/token protocol cost + network RTT |
| Second box has multiple NVLinked GPUs | TP island: worker runs `--tensor-parallel`, coordinator pipelines to it | 33.3 t/s whole-model-remote (pessimal case) |
| Need max context, model fits | offload tail layers to remote box (frees local VRAM for KV) | pipeline cost only |

Rule of thumb: **tensor parallelism inside a box (NVLink), pipeline between
boxes (Ethernet)**. Never `-sm tensor` across a network.

## 1. Worker setup (the box that serves its GPUs)

The production image ships the worker binary. Compose reference:
`docker-compose.rpc-worker.yml`. Manual invocation:

```bash
# plain worker: each GPU exposed individually (pipeline stages)
ggml-rpc-server -H 0.0.0.0 -p 50052 -c

# TP island: all local GPUs exposed as ONE tensor-parallel device
ggml-rpc-server -H 0.0.0.0 -p 50052 -c --tensor-parallel
```

- `-c` enables the local weight cache — **always use it**: the first load
  streams the full model over the socket (~40 min for a 27B over the
  synchronous protocol); with the cache, every later load reads local disk.
- `--tensor-parallel` requires >= 2 local GPUs; the worker prints
  `tensor-parallel island: N devices exposed as one` and advertises itself
  as `Meta[N](...)`. NCCL/P2P AllReduce runs worker-locally between its GPUs.
- The model file is only needed on the coordinator.

**SECURITY**: the RPC protocol has no authentication or TLS. Bind only to a
private network / WireGuard overlay. Never a public interface.

## 2. Coordinator setup (the box running llama-server)

```bash
# pipeline across local GPUs + remote worker(s)
llama-server -m model.gguf --rpc workerA:50052,workerB:50052 -sm layer -ngl 99 ...

# everything on a remote TP island (no local GPUs used)
CUDA_VISIBLE_DEVICES="" llama-server -m model.gguf --rpc island:50052 -ngl 99 ...

# hybrid: local GPUs take most layers, island takes the tail
llama-server -m model.gguf --rpc island:50052 -sm layer -ngl 99 -ts <weights> ...
```

Nothing else is required. When the coordinator sees an island device it
automatically computes per-tensor split states from the model (weights,
attention KV cache, recurrent state cache) and uploads them before
allocation — look for `uploading N split states to TP island` in the log.
Without those uploads (e.g. an old coordinator) the island falls back to
mirrored weights: still correct, but no memory savings.

`-ts` splits by memory across devices as usual; an island counts as one
device with the combined VRAM of its GPUs.

## 3. What to expect (measured, loopback worst case)

| Scenario | tg t/s | Notes |
|---|---|---|
| 27B in-process `-sm tensor` (reference) | 47.8 | the ceiling |
| 27B entirely on a remote 2-GPU island | 33.3 | pessimal: every token round-trips inputs + 600 KB logits |
| 27B pipeline, 1 local GPU + 1 RPC GPU | 32.3 | vs 33.2 in-process layer split = ~3% protocol cost |
| 0.6B entirely on island | 148 | small models amplify per-token overhead |

Real networks add ~2x RTT per token on top (sub-millisecond on wired LAN).
The whole-model-remote numbers are the worst case — when the island only
holds a layer slab (the intended topology), the RPC cost stays the same
small constant while most compute remains local.

## 4. Operational notes

- **First load**: budget ~40 min for a 27B over the socket (one-time per
  model per worker with `-c`). Watch progress via worker-side VRAM:
  `nvidia-smi --query-gpu=memory.used --format=csv`.
- **Verifying real sharding**: after load, each island GPU should hold
  roughly `model_size / n_gpus` + KV share + compute buffers. If VRAM looks
  like a full copy per GPU, check the coordinator log for the
  `uploading ... split states` lines.
- **Env knobs on the worker**: `GGML_META_DEBUG=1|2` traces split-state
  decisions (needs debug log level); `GGML_CUDA_ALLREDUCE=p2p` enables the
  one-shot NVLink AllReduce inside the island.
- **Failure behavior**: the protocol has no fault tolerance — a dead worker
  aborts the coordinator (TASKS.md item 12 phase 4 tracks hardening).
- **Restart policy**: give workers `restart: always`; the coordinator's
  weight cache handshake (`SET_TENSOR_HASH`) makes reconnect loads cheap.

## 5. Known limitations

- Per-token latency of whole-model-remote serving is gated by the
  synchronous RPC protocol (phase 2, async double-buffering, is the
  planned fix and matters mainly for that pessimal topology).
- Island VRAM currently runs higher than ideal sharding predicts —
  some tensor classes still fall back to mirrored (audit tracked).
- MLA models (`deepseek2` family) do not support tensor mode at all yet
  (task 13) — islands included. Pipeline mode works for them.
- Protocol version must match between coordinator and worker builds
  (both sides from this repo's image; proto 4.1+).
