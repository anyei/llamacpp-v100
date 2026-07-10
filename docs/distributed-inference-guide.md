# Distributed Inference — Usage Guide

How to run llama.cpp across multiple GPUs and machines with this fork's
coordinator/worker support. Design rationale: `distributed-inference-plan.md`.
Status/history: `TASKS.md` item 12. All numbers measured on 2x Tesla
V100-SXM2-32GB with `Qwen3.6-27B-UD-Q4_K_XL-MTP.gguf` (16.4 GB, hybrid
attention + DeltaNet) unless noted; small-model checks use
`Qwen3-0.6B-BF16.gguf`. The image
`llamacpp-local-v100:distributed-inference` ships both sides
(coordinator `llama-server` + worker `ggml-rpc-server`).

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

## 0. How data moves — what crosses the network, and when

The lifecycle of a coordinator + RPC worker, with real sizes:

```
    COORDINATOR (has the .gguf)                WORKER (no model file)
    llama-server / llama-cli                   ggml-rpc-server -c
    ──────────────────────────                 ─────────────────────

 ①  FIRST LOAD ── the worker's share of the weights crosses ONCE
    ┌──────────┐                                ┌───────────────┐
    │ GGUF on  │ ══ set_tensor ════════════════▶│ worker RAM /  │
    │ disk     │    whole model: 23 GB (35B)    │ VRAM          │
    └──────────┘    ≈ 3.5 min @ GbE             └──────┬────────┘
                    layer split: just the slice        │ hash + persist
                                                       ▼
                                                  /cache volume

 ②  EVERY LATER LOAD ── hashes only, weights come from the worker's disk
    ┌──────────┐                                ┌───────────────┐
    │ GGUF     │ ── SET_TENSOR_HASH (~KB) ─────▶│ "have it"     │
    └──────────┘                                │ load /cache   │
                                                └───────────────┘

 ③  INFERENCE ── per token, only boundary tensors cross
    ┌──────────┐    activations (KB-tens of KB) ┌───────────────┐
    │ sampler, │ ──────────────────────────────▶│ layer compute │
    │ orchestr.│ ◀──────────────────────────────│ (KV cache     │
    └────┬─────┘    logits row (~0.6 MB =       │  lives HERE)  │
         │          vocab × 4 B)                └───────────────┘
         └──▶ sample token, repeat
              cost/token ≈ 2×RTT + transfer  (~0.5 ms on a quiet LAN)
```

1. **First load — the worker's share of the weights crosses the wire, once.**
   The coordinator reads the GGUF locally and `set_tensor`-streams every tensor
   the scheduler placed on the RPC device. Whole-model-remote (`-ngl 99`, all
   layers on the worker) sends the entire model (23 GB for the 35B ≈ ~3.5 min
   on GbE); a layer split (`-sm layer -ts …`) sends only the worker's slice.
   The worker never needs the model file.

2. **Worker weight cache — the transfer happens only once.** With `-c` (and
   `LLAMA_CACHE=/cache` + a volume in a container), the worker hashes and
   persists every tensor it receives. On every later load the coordinator sends
   just the hash (`SET_TENSOR_HASH`), the worker answers "have it," and loads
   from local disk — restart-safe, re-stream-free.

3. **Inference — only activations (and logits) cross, per step.** Per token
   the coordinator sends the worker's boundary input activations and receives
   its outputs: a few KB (0.6B) to tens of KB (35B) per hop. The largest
   per-token transfer is the **logits row back to the sampler** (~vocab × 4 B
   ≈ 0.6 MB for Qwen), which is why *small fast* models feel the network more
   than big slow ones. In whole-model-remote even the KV cache lives on the
   worker; the coordinator just orchestrates + samples. At sub-millisecond
   LAN RTT this adds ~2×RTT + transfer per token — noise against a 130 ms CPU
   token, a few percent against a fast GPU token.

Measured cross-network (coordinator ↔ CPU worker at 0.15 ms RTT, 2026-07-09),
whole-model-remote, decode t/s:

| Model | loopback | network cold | network warm | network tax |
|---|---:|---:|---:|---|
| Qwen3-0.6B | 20.9 | 20.1 | 19.9 | ~4% |
| Qwen3.6-35B-A3B (MoE) | 7.7 | 7.6 | 7.4 | ~2-4% |

The 35B's cold run spent ~178 s streaming its 23 GB (≈130 MB/s, GbE-class
saturation); the warm run loaded from the worker's cache instead. Decode speed
is identical cold vs warm — the transfer is purely a load-time cost.

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
  **In a container, also set `LLAMA_CACHE=/cache`** and mount a volume there —
  without the env, the cache lands inside the container and dies with it.
- `--tensor-parallel` requires >= 2 local GPUs; the worker prints
  `tensor-parallel island: N devices exposed as one` and advertises itself
  as `Meta[N](...)`. NCCL/P2P AllReduce runs worker-locally between its GPUs.
- The model file is only needed on the coordinator.

**SECURITY**: the RPC protocol has no authentication or TLS. Bind only to a
private network / WireGuard overlay. Never a public interface.

### 1b. CPU-only worker (a box with no GPU)

Validated 2026-07-09 (loopback). A GPU-less box can serve its **CPU + RAM** as a
worker. Build the CPU image on that box from this tree (no CUDA anywhere):

```bash
docker build -f .devops/cpu.Dockerfile --target rpc-worker -t llamacpp-cpu:rpc-worker .
docker compose -f docker-compose.rpc-worker-cpu.yml up -d
```

The coordinator connects exactly like a GPU worker (`--rpc host:50052 -sm layer
-ngl 99`); layers assigned to the RPC device compute on the worker's CPU.

Measured (loopback, worker + coordinator sharing one 20-core box — treat as the
no-network upper bound):

| Model | local CPU `-ngl 0 -t 20` | whole model on RPC CPU worker |
|---|---:|---:|
| Qwen3-0.6B | 16.5 t/s | 17.8 t/s |
| Qwen3.6-35B-A3B (MoE) | 6.5 t/s | 7.7 t/s |

- **The protocol hop is free at CPU speeds** (~3% on GPU workers; unmeasurable
  when a token costs 130+ ms). Real networks add ~2× RTT/token — still minor
  for `-sm layer` on 1-10 GbE.
- **Worker threads**: rpc-server's default (half the cores) beat all-cores on
  loopback (7.7 vs 7.0 t/s — oversubscription). On a dedicated worker box,
  sweep half/physical/all (`WORKER_THREADS` in the compose) and keep the winner.
- Use cases: pipeline a model's tail layers onto the remote box's RAM
  (`-sm layer -ts …`), or serve small models entirely from the CPU box.

### 1c. Vulkan worker (Intel Arc / iGPU, AMD via Mesa)

Same pattern with the Vulkan image — the GPU enters the container via
`/dev/dri`, not the nvidia runtime:

```bash
docker build -f .devops/vulkan.Dockerfile --target rpc-worker -t llamacpp-vulkan:rpc-worker .
docker compose -f docker-compose.rpc-worker-vulkan.yml up -d
```

Start with a small model (Qwen3-0.6B) to validate the path — an Intel iGPU/Arc
is far slower than a datacenter GPU, so it's a topology/protocol test bed, not
a throughput contributor.

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
| 27B entirely on a remote 2-GPU island | 33.3 (29.2 @ 5.2k ctx) | pessimal: every token round-trips inputs + 600 KB logits |
| 27B pipeline, 1 local GPU + 1 RPC GPU | 32.3 | vs 33.2 in-process layer split = ~3% protocol cost |
| 0.6B entirely on island | 148 | small models amplify per-token overhead |

Island VRAM for the 27B at 8k context: **9.2 GB per GPU** (weights are
slice-packed — each GPU stores only its half plus KV share and compute
buffers). Load times: ~40 min first time over the socket, **40-50 s** on
every later load with the worker weight cache (`-c`).

Real networks add ~2x RTT per token on top (sub-millisecond on wired LAN).
The whole-model-remote numbers are the worst case — when the island only
holds a layer slab (the intended topology), the RPC cost stays the same
small constant while most compute remains local.

## 4. Session state over islands (save/restore, prompt checkpoints)

Server state features work transparently across RPC and are verified
byte-exact on the 27B:

- **Prompt checkpoints** (hybrid/recurrent models create these automatically
  during long prefills) read the recurrent + KV state from the island; a
  149.6 MiB checkpoint over loopback is effectively free within a 5k-token
  prefill.
- **Slot save/restore** (`--slot-save-path` + `/slots/:id?action=save|restore`)
  round-trips a 501 MB / 5259-token state in ~0.5-0.6 s over loopback with
  byte-identical continuation.
- Under the hood, raw-byte tensor reads/writes on *views* are translated to
  their root tensors before crossing the wire (view links don't survive
  serialization); the worker refuses loudly rather than guessing if it ever
  sees an unresolvable alias.
- A state file saved on an island restores exactly on the same topology.
  Restoring across topologies (island ↔ single GPU) works too, but the KV
  payload carries low-mantissa AllReduce noise between tensor-parallel and
  single-GPU configs — harmless in practice, just not bit-identical.

## 5. Async RPC & multi-worker pipelines (proto 4.2)

Everything here is automatic — no new flags on the happy path:

- **Pipeline parallelism engages** when a layer split spans multiple RPC
  devices: the coordinator log prints `pipeline parallelism enabled` and the
  scheduler double-buffers ubatches (`n_copies=4`). RPC devices now report
  async + events capability.
- **Worker-to-worker transfers**: with two or more 4.2 workers in a layer
  split, cross-stage activations flow directly between workers (a fenced
  pull) instead of bouncing through the coordinator:

  ```
      llama-server --rpc A:50052,B:50052 -sm layer

      COORDINATOR                WORKER A                 WORKER B
      (sampler, embed)           (layers 0..N/2)          (layers N/2..N)
      ────────────┬──            ───────┬────────         ───────┬───────
                  │  activations        │                        │
                  ├────────────────────▶│ compute stage 1        │
                  │                     │                        │
                  │              proto 4.2 (default):            │
                  │                     │══ direct, fenced ═════▶│ B pulls A's
                  │                     │   activation pull      │ output, then
                  │                     │                        │ computes
                  │  GGML_RPC_NO_W2W=1 (or peer unreachable):    │ stage 2
                  │◀╌╌╌ stage-1 out ╌╌╌╌┤                        │
                  ├╌╌╌╌ stage-1 out ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌▶  (2 hops
                  │                     │                        │  instead
                  │◀─────────────────── logits ──────────────────┤  of 1)
                  ▼ sample, repeat
  ```

  Requirements:
  - workers must be able to reach *each other* at the exact endpoint strings
    the coordinator uses (`--rpc a:port,b:port`) — same network/overlay, no
    NAT between them. If a worker cannot reach its peer, the copy silently
    falls back to the coordinator-bridged path (the worker logs
    `cannot reach source worker`).
  - `GGML_RPC_NO_W2W=1` on the **coordinator** forces the old bridged copies
    (debugging escape hatch).
- **Multiple coordinators per worker** are now possible (the worker serves
  connections concurrently; each connection's buffers are freed when it
  disconnects) — but two coordinators must not share the same *device*:
  the second one's graph replays fail loudly by design.
- **Unreachable workers fail fast**: `llama-server --rpc bad:port` now exits
  with `failed to connect to RPC server` instead of silently degrading to
  CPU-only inference (pre-4.2 behavior — worth knowing if you relied on it).
- **Version compatibility**: a 4.2 coordinator works with 4.1 workers (new
  features simply stay off). A 4.1 coordinator refuses 4.2 workers —
  upgrade the coordinator first, or both sides from the same image
  (`llamacpp-local-v100:distributed-inference` and later ship 4.2).

## 6. Operational notes

- **First load**: budget ~40 min for a 27B over the socket (one-time per
  model per worker with `-c`). Watch progress via worker-side VRAM:
  `nvidia-smi --query-gpu=memory.used --format=csv`.
- **Verifying real sharding**: after load, each island GPU should hold
  roughly `model_size / n_gpus` + KV share + compute buffers (9.2 GB/GPU
  for the 27B at 8k context). If VRAM looks like a full copy per GPU, check
  the coordinator log for the `uploading ... split states` lines.
- **Env knobs on the worker**: `GGML_META_DEBUG=1|2` traces split-state
  decisions (needs debug log level); `GGML_CUDA_ALLREDUCE=p2p` enables the
  one-shot NVLink AllReduce inside the island.
- **Failure behavior**: the protocol has no fault tolerance — a dead worker
  aborts the coordinator (TASKS.md item 12 phase 4 tracks hardening).
- **Restart policy**: give workers `restart: always`; the coordinator's
  weight cache handshake (`SET_TENSOR_HASH`) makes reconnect loads cheap.

## 7. Known limitations

- Per-token latency of whole-model-remote serving is still gated by the
  synchronous per-token input/logits round trips (async double-buffering
  helps pipelines, not the single-stage whole-model-remote case).
- On loopback with a shared GPU, worker-to-worker transfers measure the
  same as bridged copies — the win is on real networks (one direct hop
  instead of two through the coordinator) and has not been measured on
  physical two-box hardware yet.
- MLA models (`deepseek2` family) run tensor mode with mirrored attention
  (task 13 closed): quality validated by perplexity, but temp-0 outputs can
  diverge from single-GPU runs (MoE-router-amplified reduction noise) — do
  not gate MLA tensor/island setups on byte-exactness.
- No fault tolerance: a dead worker aborts the coordinator. No auth/TLS:
  private networks only — and workers now also connect to each other, so
  the whole worker set must share the trusted network.
