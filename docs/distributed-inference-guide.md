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

All decode t/s below are `Qwen3.6-27B-UD-Q4_K_XL` on 2× V100 (loopback, no
speculation) unless a different model is named; the 27B's *production* number
with MTP speculation is higher (~81 t/s — see the README benchmarks).

| Situation | Mode | Measured (model) |
|---|---|---|
| All GPUs in one machine | single process, `-sm tensor` + NCCL — do NOT use RPC | 47.8 t/s (27B, nospec) — the ceiling reference |
| Model fits one machine, second box idle | leave it idle, or serve another model on it | — |
| Model does NOT fit one machine | pipeline between boxes: `-sm layer --rpc worker:port` | 32.3 vs 33.2 t/s in-process (27B, 1 local + 1 RPC GPU) → ~3%/token + network RTT |
| Second box has multiple NVLinked GPUs | TP island: worker runs `--tensor-parallel`, coordinator pipelines to it | 33.3 t/s (27B entirely on a remote 2-GPU island — pessimal case) |
| Second box has no GPU (CPU + RAM only) | CPU worker: `cpu.Dockerfile --target rpc-worker` (§1b) | 35B-A3B MoE: 7.4-10.4 t/s by CPU; 0.6B: ~20 t/s; ~2-4% network tax (§3b) |
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

 ①b FIRST LOAD, worker has its own copy ── nothing big ever crosses
    (rpc-server --model-dir /models — the worker indexed its GGUFs by
     tensor-content hash at startup)

    ┌──────────┐                                ┌───────────────┐
    │ GGUF on  │ ── SET_TENSOR_HASH (~KB) ─────▶│ hash in local │
    │ disk     │                                │ model index?  │
    └──────────┘                                └──────┬────────┘
                                            hit        │        miss (stale /
                                    ┌───────────────┐  │        different file)
                                    │ worker's own  │◀─┤
                                    │ .gguf ── pread│  └──▶ falls back to the
                                    │ at NVMe speed │       ① stream for that
                                    └──────┬────────┘       tensor only
                                           ▼
                                      worker RAM / VRAM
    23 GB slice: ~3.5 min of ① streaming becomes seconds of local reads;
    the coordinator can't even tell the difference

 ②  EVERY LATER LOAD ── hashes only, weights come from the worker's disk
    ┌──────────┐                                ┌───────────────┐
    │ GGUF     │ ── SET_TENSOR_HASH (~KB) ─────▶│ "have it"     │
    └──────────┘                                │ load /cache   │
    (same story with --model-dir: the local     └───────────────┘
     index serves whatever the /cache misses)

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

   **1b. Worker-local models (`--model-dir`) — even the first stream disappears.**
   If the worker box has its own copy of the GGUF (any directory of GGUFs),
   `rpc-server --model-dir <dir>` indexes every tensor by content hash at
   startup (one read of each file, ~seconds/10 GB on NVMe; the index persists
   in the cache dir, so later starts are instant). The coordinator's
   `SET_TENSOR_HASH` then hits the index and the worker preads the bytes from
   its *own* GGUF instead of receiving the stream — first load at local-disk
   speed instead of network speed. Every read is re-hash-verified, and a
   stale or different local file simply hash-misses and streams as before —
   there is no version-skew failure mode, and the coordinator cannot tell the
   difference.

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
- `-md`/`--model-dir DIR` kills even the *first*-load stream when the worker
  has its own copy of the GGUF: the worker indexes every GGUF under DIR by
  tensor-content hash at startup (reads each file once, ~seconds/10 GB on
  NVMe; the index persists in the cache dir so later starts are instant) and
  serves hash-matched tensors from local disk instead of asking the
  coordinator to stream them. A stale or different local file simply
  hash-misses and streams as before — no version-skew failure mode. Only
  tensors > 10 MiB go through the hash path (same threshold as the weight
  cache), and tensor-parallel islands still stream (they receive slices,
  which don't hash-match whole tensors).
- `--tensor-parallel` requires >= 2 local GPUs; the worker prints
  `tensor-parallel island: N devices exposed as one` and advertises itself
  as `Meta[N](...)`. NCCL/P2P AllReduce runs worker-locally between its GPUs.
- `-a`/`--announce` (opt-in) broadcasts a presence beacon every ~2 s over UDP
  multicast so a coordinator running `--rpc-discover` finds this worker
  without a hard-coded `--rpc` list. The beacon egress is pinned to the
  interface of `-H`, so it never widens exposure beyond where the RPC port
  already listens. `--announce-group ADDR:PORT` overrides the default group
  (must match the coordinator's `--rpc-discover-group`).
- **KV annex for small-VRAM cards (task 30)**: run the worker with
  `-d CUDA0,CPU` and the coordinator with `LLAMA_KV_WORKER_HOST=1` — the
  worker's CPU device takes no layers and instead hosts the KV cache of its
  GPU layers in worker RAM, so the whole VRAM goes to weights (a 6 GB
  1660 Ti holds ~2× the layers). Attention for those layers runs on the
  worker CPU; GPU↔RAM handoffs stay worker-local. Measured cost: ~7%
  decode (0.6B @32k, loopback). Without the env the extra CPU device is
  just another (slow) layer target — don't expose it unless annexing.
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

**Dynamic fleets** (workers come and go): instead of — or alongside — a
static `--rpc` list, pass `--rpc-discover` and start the workers with
`--announce`. The coordinator listens for beacons for ~2.5 s at startup,
registers every worker it hears (duplicates against `--rpc` are skipped),
and warns-and-continues if a discovered worker vanished before connect.
Combined with `--rpc-skip-unavailable` for the static part of the list, the
fleet is fully dynamic: whoever is present at load time gets used, nobody
missing blocks startup. `--rpc-discover-group ADDR:PORT` selects a custom
multicast group. Trusted networks only — beacons are unauthenticated, like
the RPC protocol itself. Compose reference:
`docker-compose.fleet-coordinator.yml` (discovery + skip-unavailable + an
optional static list), paired with the `docker-compose.rpc-worker*.yml`
workers, which all `--announce` by default.

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

### 3b. Measured: a real heterogeneous CPU/iGPU worker fleet (2026-07-10)

**Split policy for CPU fleets (measured 2026-07-11, TASKS.md #31):** CPU decode
is memory-bandwidth-bound, so a pipeline across CPU boxes *aggregates
bandwidth* — and the optimal `-ts` is proportional to each box's **memory
bandwidth**, not its RAM size (the default) or CPU specs. 35B across
i7-8550U (DDR4) + i5-3210M (DDR3): all-on-fastest 0.88 t/s · memory-default
1.2 · even 1.72 · **bandwidth-proportional `-ts 3,2` = 2.26** (the DDR4:DDR3
ratio). Also measured: `-np` batching barely raises MoE aggregate on CPU
(+27%) because batched tokens hit distinct experts. GPU-rich boxes are the
opposite regime: never distribute a model that fits fast VRAM.


First cross-host measurements on real hardware. Fleet: coordinator (V100 box),
worker A = **i7-9750H** laptop (64 GB, `10.5.5.15`), worker B = **Core Ultra 9
285H** (64 GB, Arc 140T iGPU, `10.5.5.11`); quiet LAN, 0.15-0.44 ms RTT.
Decode / prefill t/s, whole-model-remote unless noted:

| Config | 0.6B | 35B-A3B (MoE) |
|---|---|---|
| worker A CPU (9750H) | 19.9 / 134 | 7.4 / 11.3 |
| worker B CPU (285H) | — | **10.4** / 10.7 |
| worker B iGPU (Arc 140T, Vulkan) | 26.3 / 39.5 | 10.0 / 5.3 |
| **both workers, `-sm layer` even split** | 17.9 / 86 | 7.4 / 10.7 |
| both workers, `GGML_RPC_NO_W2W=1` | — | 7.8 / 9.1 |

What these numbers teach (they generalize):

- **The default split is memory-proportional, not speed-aware** — both boxes
  report ~64 GB, so layers split ~50/50 and the pipeline runs at the *slower*
  box's pace (7.4 = worker A's solo speed; worker B alone does 10.4). Weight
  with `-ts`, or better: if the model fits the fastest box, run it there alone
  — single-stream pipeline time is the *sum* of stage times, so adding a
  slower box to a model that already fits can only slow the stream down.
  A pipeline earns its keep for **capacity** (model > one box's RAM) and
  **throughput** (stages overlap across concurrent requests), not latency.
- **Worker-to-worker vs coordinator-bridged is indistinguishable at CPU
  speeds** (7.4 vs 7.8, noise): the direct pull saves ~one LAN hop (~0.5 ms)
  against ~135 ms tokens. W2W matters for fast GPU workers, not CPU fleets.
  (`pipeline parallelism enabled` confirmed in the coordinator log — the
  async machinery does engage.)
- **An iGPU worker ≈ its host's CPU for decode** (10.0 vs 10.4): both drain
  the same LPDDR5X pool and decode is bandwidth-bound. Its prefill is 2x
  *worse* under Mesa/ANV Vulkan (no XMX matrix engines; see TASKS.md #27 for
  the tentative SYCL/coopmat follow-up). Use the CPU worker unless you need
  the cores free; only a **discrete** GPU (own VRAM bandwidth) meaningfully
  beats its host CPU as a worker.
- **Small models feel the network, big models don't**: 0.6B lost ~10% going
  from one worker to a 2-hop pipeline (17.9 vs 19.9) and ~4% to the network
  itself; the 35B lost ~2-4%.

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
- **Failure behavior**: at load time, an unreachable worker errors the
  coordinator out by default; pass `--rpc-skip-unavailable` (env
  `LLAMA_ARG_RPC_SKIP_UNAVAILABLE`) to drop dead workers with a warning and
  split the model across whoever is present — the right mode for fleets
  where laptops come and go. Caveat: an explicit `-ts` maps to devices
  positionally and will not account for the dropped servers, so prefer the
  default memory-proportional split with this flag. At runtime, a
  worker-side compute error (e.g. a CUDA fault) no longer kills the worker
  process: it drops that coordinator's connection and keeps serving others,
  and after 3 consecutive failures (poisoned CUDA context) it exits cleanly
  so `restart: always` brings up a fresh process (task 29e). The
  *coordinator* survives a worker dying mid-session too (task 29b): the
  failed endpoint is poisoned (never aborts the process), in-flight
  requests receive proper error responses (HTTP 500 "Compute error."), and
  llama-server then exits with code 42 — a lost worker makes the layer
  assignment unrecoverable in-process, so the restart policy reloads and
  `--rpc-discover` re-splits across whatever workers are alive. Together:
  a fully self-healing fleet with no manual intervention. Live
  redistribution without a restart remains TASKS.md #29 (c).
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
- No *runtime* fault tolerance: a worker that dies mid-session aborts the
  coordinator (load-time degradation exists — see `--rpc-skip-unavailable`
  in §6). No auth/TLS:
  private networks only — and workers now also connect to each other, so
  the whole worker set must share the trusted network.
- **GTX 16xx workers (TU116/117)**: those cards are cc 7.5 WITHOUT tensor
  cores; the flash-attention MMA kernel selected by CC used to kill the
  worker process (`cudaFuncSetAttribute: invalid argument` → abort,
  TASKS.md #32). Fixed 2026-07-10: the failed shared-memory opt-in now
  disables the MMA kernel on that device (one-time WARN) and falls back to
  the tile kernel, so FA can stay on. Set `GGML_CUDA_FA_NO_MMA=1` in the
  worker's environment to skip even the first failed attempt. `-fa off` on
  the coordinator remains the workaround for images older than this fix
  (verified: 35B layers on the 1660 Ti run with `-fa off`). Real-hardware
  confirmation of the new fallback on the 1660 Ti is still pending (needs a
  fresh `CUDA_DOCKER_ARCH=75` image on that box).
