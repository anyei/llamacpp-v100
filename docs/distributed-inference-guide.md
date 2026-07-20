# Distributed Inference — Usage Guide

How to run llama.cpp across multiple GPUs and machines with this fork's
coordinator/worker support. **Visual overview:** `architecture-diagrams.md`
(placement decision map, pipeline/EP topologies, load paths, fleet lifecycle).
Design rationale: `distributed-inference-plan.md`.
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
boxes (Ethernet)** — with one exception: a MoE whose *experts* fit no single
box runs cross-box as **expert-parallel** (`-sm tensor` + dedicated attention),
because its per-layer reduce is a small vector sum, not a GEMM reduce. Side-by-
side comparison and spin-up commands for all three modes: **§2b**.

## What this fork adds over upstream llama.cpp RPC

Everything in this guide builds on upstream's `rpc-server`/`--rpc` wire
concept, but most of what the guide describes does not exist in mainline.
Upstream RPC is a static list of remote devices spoken to over a blocking,
single-connection protocol: the whole model re-streams on every load, the
async/events interface is unimplemented (so pipeline parallelism never
engages over RPC), cross-stage copies bridge through the coordinator, and a
worker dying mid-serve kills the run. The one-line difference: **upstream
gives you a wire protocol; this fork built a serving fleet on top of it** —
remote devices are treated as a *population* with measured speeds, link
qualities, caches and lifecycles, not as GPUs that happen to be far away.

| Area | Upstream mainline | This fork (proto 4.2 -> 4.9) |
|---|---|---|
| Topologies | layer pipeline | + in-process tensor-parallel (meta backend), **expert-parallel** (meta-over-RPC, attention owner, star reduce - §2b), TP islands (§2/§4), KV annex |
| Wire | blocking round trips | async + events (pipeline engages), worker-to-worker fenced pulls, fused boundary command (one message per worker per MoE layer), uid-keyed graph cache (16 B/subgraph steady state) |
| Load path | full re-stream each load; cache serves unverified entries | verified content-addressed cache (atomic write + re-hash on read), `--model-dir` local sourcing, slice provenance (§0.1b) - cold 40 min -> warm minutes, `-ts`-change-proof |
| Fleet ops | static `--rpc` list | discovery beacons, bandwidth scoring + `--rpc-auto-weight`, capacity gate, fleet UI/API, preflight bench, worker log/restart/include (§2) |
| Failure | run dies with the worker | CUDA-error containment (drops one connection, not the worker), journal + surgical re-provision (~2 min), `--rpc-reload` re-split over survivors (§6) |

Deliberately unchanged: the trust model (unauthenticated, trusted-LAN-only,
same as upstream) and wire-level compatibility discipline — every fork
feature is protocol-version-gated, so a mixed fleet of old and new workers
degrades feature-by-feature instead of breaking.

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

   Since proto 4.8 (TASKS.md #44) this also covers **slices**: in EP/tensor
   mode the coordinator's offers carry source provenance (tensor name, byte
   offset, row geometry), so the worker assembles the exact slice from its
   local GGUF by pread no matter where the split boundaries fall — a `-ts`
   or auto-weight share change no longer cold-streams `--model-dir` boxes
   (measured -71% wire bytes on a boundary-moved 9B tensor-mode load). Same
   trust model: serve only on hash match. Kill switch: `GGML_RPC_NO_SRC_HINT=1`
   on the coordinator.

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
  Housekeeping (TASKS.md #45): `GGML_RPC_CACHE_LIMIT_MIB` caps the dir
  (least-recently-USED entries evicted; serves refresh mtime), enforced on
  every save AND once at startup — an idle worker trims stale model
  generations before its next load. The beacon publishes the current size
  (`cache_mib=`), shown in the fleet UI's Discovered table.
- `-md`/`--model-dir DIR` kills even the *first*-load stream when the worker
  has its own copy of the GGUF: the worker indexes every GGUF under DIR by
  tensor-content hash at startup (reads each file once, ~seconds/10 GB on
  NVMe; the index persists in the cache dir so later starts are instant) and
  serves hash-matched tensors from local disk instead of asking the
  coordinator to stream them. A stale or different local file simply
  hash-misses and streams as before — no version-skew failure mode. Only
  tensors > 10 MiB go through the hash path (same threshold as the weight
  cache). Since proto 4.8, EP/tensor-mode SLICES are served from the local
  GGUF too (the offer carries source provenance; see section 0 point 1b) —
  before that, slices only hit the received-bytes cache and any `-ts` change
  cold-streamed them.
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

**Fleet capacity gate** (TASKS.md #50): when RPC devices are in the pipeline,
the coordinator refuses to START a load whose weights + KV reserve
(`LLAMA_FLEET_KV_RESERVE_MB`, default 20 GiB) exceed the pooled free device
memory — a `--no-mmap` fleet load over capacity is a guaranteed
minutes-later OOM mid-stream. Instead the server waits in a visible
`waiting-capacity` state (orange chip + banner in the fleet UI,
`/fleet/status.capacity` = required/available MiB) and, under
`--rpc-discover`, watches the beacons: as soon as newly appeared workers
raise the joinable pool past the requirement it exits 42 so the restart
policy re-discovers the grown fleet and loads automatically — power on
another box and walk away. Static `--rpc` configs wait with a
restart-with-more-workers message. Single-box (no-RPC) runs are never gated
(mmap paging, `-ncmoe`, ssd-streaming are supported over-capacity regimes);
`LLAMA_FLEET_CAPACITY_CHECK=0` disables, and hybrid fleet+CPU-offload
setups that deliberately place weights in coordinator RAM should disable it
(the pool math counts devices only).

**Fleet web UI + API** (TASKS.md #35): the server web UI has a **Fleet**
section (sidebar network icon, `#/fleet`) backed by three endpoints:

- `GET /fleet/status` — per-device cards: memory, split share, layers,
  transfer counters (bytes, EWMA round-trip latency), speed score, plus every
  worker beaconing on the LAN (in the pipeline or not, with its beacon-carried
  `cache_mib` tensor-cache size). Answers **during** model load (reports
  fleet-wide load progress) and during `--rpc-reload` recovery; while the
  capacity gate holds a load, `server_state` is `waiting-capacity` and a
  `capacity` object carries required/available MiB.
- `GET /fleet/worker/log?ep=host:port` — tails the worker's in-memory log
  ring over a dedicated RPC connection (proto 4.7 `GET_LOG`, served outside
  the worker's exec lock so it answers even mid-compute).
- `POST /fleet/worker/rescore {"endpoint": ...}` — re-runs the worker's ~0.5 s
  bandwidth benchmark on demand (proto 4.9; a busy worker refuses rather than
  under-read). `--rpc-auto-weight` also re-benches every 4.9 worker at split
  time automatically, so stale startup scores no longer starve members. UI:
  per-worker Re-score button; shares apply at the next (re)load.
- Device cards carry a persistent health badge (healthy / recovering /
  degraded, with failure counts kept 5 min past recovery) and the fleet page
  banners recent failures — a crash-looping worker no longer shows green
  between crashes. The loading page shows per-worker readiness: two-tone
  progress bars (cached vs streamed, VRAM-delta for local GPUs) with
  waiting / streaming / loading-from-cache / READY chips.
- `POST /fleet/worker/restart {"endpoint": ...}` — asks the worker to
  `exit(0)` so its restart policy brings it back and `--rpc-reload`
  re-provisions it. Requires `--fleet-admin` AND an `--api-key` (it is a
  remote-kill primitive; RPC itself is unauthenticated).

**Speed scores + auto-weighted splits** (TASKS.md #31/#35f): workers started
with `--score` (all `docker-compose.rpc-worker*.yml` do) run a ~1 s matvec
benchmark at startup — effective memory bandwidth, the quantity that bounds
decode — and publish it in their beacon and over RPC. The coordinator flag
`--rpc-auto-weight` (`COORD_AUTO_WEIGHT=1` in the compose) then fills an
unset `-ts` proportionally to the scores, water-filled against each device's
free memory so shares never exceed capacity; local GPUs are benchmarked at
startup. #31 measured bandwidth-weighted splits ~2x faster than the
free-memory default on a heterogeneous CPU fleet. An explicit `-ts`/`COORD_TS`
always wins, and a warning fires when the model would fit the local GPUs
alone (#31 law 1: distribution loses in that regime).

## 2b. Choosing a split mode: layer vs tensor vs expert-parallel

Three ways to spread one model across devices. They differ in *what* each
device holds and *how* a token's work composes across them — which is what
decides whether distribution helps or hurts.

| | `-sm layer` (pipeline) | `-sm tensor` (tensor-parallel) | `-sm tensor` + EP (expert-parallel) |
|---|---|---|---|
| Each device holds | whole layers | a slice of *every* tensor | attention on ONE member, expert slices on the rest |
| Per-token comms | activations at each stage boundary (KB) | an AllReduce **every layer** | one reduce per MoE boundary (star/fused) |
| Composition law | **SUM** of stage times (+ hops) | max + per-layer reduce | **max over contacted workers** (#28) |
| Scales by adding boxes | worse (more stages) single-stream; helps capacity/throughput | only inside a box | capacity + batch throughput; single-stream is latency-bound |
| Good across a network? | **yes** (Ethernet-friendly) | **NO** — AllReduce/layer over Ethernet is ~100x too slow | yes — the reduce is a small k-vector sum, not a GEMM reduce |
| Use when | model/KV doesn't fit one box; tail-offload for context | ≥2 NVLinked GPUs in **one** box | a MoE whose **experts** fit no single box |
| Measured here | CPU fleet 1.2-2.0 t/s (35B); law: best-single-box wins if it fits | 2x V100 in-box: 108 t/s (35B), 47.8 (27B) — the ceiling | V4 86.7 GB RAM-resident on the fleet: 2.4-2.7 t/s single-stream |

```
 -sm layer (pipeline)          -sm tensor (in-box TP)      -sm tensor + EP (cross-box)
 token ─▶[L0-15]▶[L16-31]▶      token ─▶┌GPU0 slice┐        attn ─▶ CUDA0 (owner)
          box A     box B               │  reduce  │─▶       experts ─┬▶ worker .11
 time = A + B (+ hops)                  └GPU1 slice┘         (routed) ├▶ worker .15
 SUM of stages                    every layer, NVLink        time = MAX over the
                                  never Ethernet             workers a token touches
```

### Spin-up — `-sm layer` (pipeline across boxes)

Capacity / tail-offload. Each RPC worker computes whole layers; `-ts` weights
the split (auto-weight fills it by measured bandwidth):

```bash
# hand-weighted: worker .11 gets 3 shares, .15 gets 2, local GPUs the rest
llama-server -m model.gguf --rpc 10.5.5.11:50052,10.5.5.15:50052 \
  -sm layer -ngl 99 -ts 3,2,4,4

# zero-config dynamic fleet, bandwidth-weighted (workers run --score --announce):
COORD_AUTO_WEIGHT=1 docker compose -f docker-compose.fleet-coordinator.yml up
#   -> or the ready-made script:  ./run-fleet-deepseek-autoweight.sh
```

### Spin-up — `-sm tensor` (tensor-parallel inside one box)

The single-box ceiling. All local GPUs, NVLink AllReduce, one process, no RPC:

```bash
CUDA_VISIBLE_DEVICES=0,1 llama-server -m model.gguf \
  -sm tensor -ts 0.5,0.5 -ngl 99            # + GGML_CUDA_ALLREDUCE=p2p (env-gates §3)
```

To expose a multi-GPU box as ONE tensor-parallel device *to a remote
coordinator* (a "TP island"), run the worker with `--tensor-parallel` (§1) and
point the coordinator's `--rpc` at it — the AllReduce stays NVLink-local on the
island; only the island's boundary activations cross the network.

### Spin-up — `-sm tensor` + expert-parallel (a MoE across the fleet)

The flagship distributed mode (TASKS.md #28; design in
`docs/expert-parallel-plan.md`). Attention/KV/router live on ONE member
(`LLAMA_META_ATTN_OWNER`, a local V100), the routed experts are segmented
across the members (`LLAMA_META_EP_ONLY`). This is how the 86.7 GB
DeepSeek-V4-Flash runs RAM-resident across boxes that individually hold none of
it:

```bash
# env selects the EP shape; --device lists the meta members in -ts order
LLAMA_META_EP_ONLY=1 LLAMA_META_ATTN_OWNER=0 \
llama-server -m DeepSeek-V4-Flash.gguf \
  --rpc 10.5.5.11:50052,10.5.5.15:50054 \
  --device CUDA0,RPC0,RPC1 -sm tensor -ts 0,3,2 \
  -ngl 99 --no-mmap -c 4096 -ub 256 -b 256

# ready-made (hand -ts):        ./run-ep-fleet-deepseek.sh
# ready-made (auto-weighted):   EP_AUTO_WEIGHT=--rpc-auto-weight ./run-ep-fleet-deepseek.sh
```

- `-ts 0,3,2` gives the attention owner a **0** expert share (it owns attention
  instead); `--rpc-auto-weight` sizes the rest by score, capped by memory.
- **Exactly ONE local GPU** (the attention owner). A second local GPU as an
  expert member corrupts the reduce and is rejected at load (TASKS.md #48).
  Got two local GPUs? Use them via single-box `-sm tensor -ngl 99 -ncmoe N` —
  that's *faster* for a model whose experts fit CPU RAM+NVMe (V4: 3.54 vs
  ~2.5 t/s), since EP single-stream is latency-bound anyway.
- **Single-stream is latency-bound, not compute-bound** (#28 attribution): a
  token pays one round trip per MoE boundary, so **fewer computing members is
  faster** single-stream (adding a fast V100 expert member *lowered* single-
  stream 2.69 -> 2.06 t/s by adding a boundary participant). The wins are
  **capacity** (a model no box holds) and **batch throughput** (the boundary
  cost is paid once per step regardless of batch — `-np 4` aggregate scales
  ~x2.85, crossing the single-box bar at B=4).
- `-ub 256 -b 256` keeps the DSA compute reserve small (~258 MiB vs ~67 GB
  virtual at the default ub 512). Give each member real memory headroom — V4's
  decode compute buffer OOMs a member sized too tightly (leave margin beyond
  `LLAMA_RPC_AUTO_WEIGHT_RESERVE_MB`).

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

### 3c. Measured: Hunyuan 3 (295B MoE) served across the fleet (2026-07-17)

The largest model this fleet has run: `hy3-1M-MTP-Q4_K_M.gguf` — **295B total
/ ~21B active** (192 experts top-8 + 1 shared, 80 layers + 1 MTP NextN layer,
GQA 64/8, 1M native context), Q4_K_M ~171 GB. Fits nowhere singly; served
RAM/VRAM-resident across 5 boxes + 2 V100s (TASKS.md #51; arch ported from
upstream #25395, image `llamacpp-local-v100:75737b40b`).

**The fleet** (auto-weighted shares from measured bandwidth, capacity-capped):

| Device | Box | Score | Share |
|---|---|---:|---:|
| RPC0 | 10.5.5.11 — Core Ultra 9 285H, 64 GB | 51.5 GB/s | 29.2% |
| RPC1 | coordinator-local CPU worker (i9-10850K) | 30.1 GB/s | 19.0% |
| RPC2 | 10.5.5.15 — i7-9750H, 64 GB | 16.3 GB/s | 11.2% |
| RPC3 | 10.5.5.25 — i7-8550U, 32 GB (100-Mbit link) | 10.8 GB/s | 7.4% |
| RPC4 | 10.5.5.30 — i5-3210M (2012, 2C/4T), 16 GB (100-Mbit link) | 20.8 GB/s | 4.9% |
| CUDA0/1 | coordinator — 2x Tesla V100-SXM2-32GB (NVLink) | ~820 GB/s | 14.1% each |

Coordinator box: i9-10850K (10C/20T), 46 GiB DDR4, the two V100s. Note the
box contributes BOTH its GPUs (as pipeline stages) and its RAM (as a
`--announce`d local CPU worker) — 33% of the model lives on the coordinator
host in total. The weakest box on the LAN (.26, 8.6 GB/s) was deliberately
left out: one fewer stage beats its capacity contribution.

**Coordinator invocation** (`run-fleet-hy3-autoweight.sh`, compose
`docker-compose.fleet-coordinator.yml`):

```
llama-server -m hy3-1M-MTP-Q4_K_M.gguf -ngl 99 -sm layer
  -c 4096 -ub 256 -b 256 --no-mmap --port 8095
env: LLAMA_ARG_RPC_DISCOVER=1  LLAMA_ARG_RPC_SKIP_UNAVAILABLE=1
     LLAMA_ARG_RPC_AUTO_WEIGHT=1  LLAMA_ARG_FLEET_PREFLIGHT=Qwen3-0.6B
```

**Measured (temp 0.2, output read and coherent — native Hy3 chat template):**

| Metric | Value |
|---|---|
| Load (warm worker caches, HDD source) | 42.7 min |
| Preflight floor (0.6B over this topology) | 11.0 t/s |
| Decode, single-stream (67-110 tok gens) | **1.90-1.93 t/s** |
| Prompt processing (short prompts) | 3.1-3.4 t/s |
| Decode graph reuse | engaged (186 graphs by req 3) |

Context for the number: ~21B active at Q4_K_M = **~12.5 GB of weight reads
per token**, spread over the stages' memory systems (sum-of-stages ~360 ms
+ hops) — 1.9 t/s is the expected physics, not overhead. The same session
exercised the capacity gate twice for real (held at 177 < 194.6 GiB until
two more boxes beaconed, then auto-restarted and loaded) and survived a
beaconing-but-firewalled worker being dropped. Next lever: hy3's MTP head
via the speculative stack (`p-min 0.75` mandatory — upstream measured
acceptance collapsing at the default and 88-97% at 0.75), plus `-np`
batching for aggregate throughput.

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
  llama-server then either exits with code 42 (default — the restart policy
  reloads and `--rpc-discover` re-splits across whatever workers are alive)
  or, with `--rpc-reload` (env `LLAMA_ARG_RPC_RELOAD`), reloads the model
  IN-PROCESS across the workers reachable at that moment (task 29c): dead
  workers are dropped together with their positional `-ts` shares, a worker
  that comes back is re-included by the next failure-triggered reload,
  all-workers-dead degrades to local-only (loudly), and a load that cannot
  succeed yet (fleet-sized model, no workers) retries every 10s. Same
  process, HTTP endpoints and queue throughout — no orchestrator needed.
  Together: a fully self-healing fleet with no manual intervention.
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
