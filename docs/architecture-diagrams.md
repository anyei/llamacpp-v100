# Architecture diagrams

Visual companion to [distributed-inference-guide.md](distributed-inference-guide.md),
[expert-parallel-plan.md](expert-parallel-plan.md) and [ssd-streaming-plan.md](ssd-streaming-plan.md).
Each diagram answers one architectural question; measured numbers are from
TASKS.md and are V100-fleet-specific (your hardware will differ, the *shapes* won't).
GitHub renders the mermaid blocks natively.

## 1. Which mode do I run? (placement decision map)

The first question is always capacity: where do the weights fit?

```mermaid
flowchart TD
    A{"model fits one box's VRAM?"} -->|yes| B["single box, -sm tensor or -sm layer<br/>(fastest: 35B on 2xV100 = 108 t/s;<br/>never distribute what fits - law #31.1)"]
    A -->|no| C{"fits VRAM + RAM of one box?"}
    C -->|yes| D["-ngl 99 -ncmoe: experts in CPU RAM<br/>(V4 87GB on 46GB RAM box = 3.54 t/s<br/>with NVMe paging behind it)"]
    C -->|no| E{"fits the fleet's pooled RAM+VRAM?"}
    E -->|yes| F["fleet: -sm layer + --rpc-auto-weight<br/>(V4 fewer-boxes run = 4.6 t/s)<br/>or EP -sm tensor for MoE batching"]
    E -->|no| G["--ssd-streaming: expert cache<br/>over NVMe (runnability, 1.5-3 t/s)<br/>or grow the fleet"]
```

Rules of thumb baked into the arrows: distribution *loses* whenever a smaller
config holds the model (sum-of-stages); the fleet's win over `-ncmoe` is
removing the disk from the token loop, not adding compute.

## 2. Layer-split pipeline (`-sm layer`)

One contiguous slab of layers per device, in device-list order (RPC workers
first). Per token, the hidden state walks the stages in order — time is the
SUM of stages plus hops, so slow boxes tax every token.

```mermaid
flowchart LR
    subgraph coordinator
        E[embed] --> R0
        CU0["CUDA0<br/>layers 44-56"] --> CU1["CUDA1<br/>layers 57-80<br/>+ output head"]
        CU1 --> S[sampler]
    end
    R0["worker .11<br/>layers 0-23"] --> R1["worker .15<br/>layers 24-35"]
    R1 --> R2["worker .25<br/>layers 36-43"]
    R2 --> CU0
```

- Activation per hop: hidden x 2-4 B (tens of KB) — the network carries almost
  nothing per token; bandwidth matters at *load*, latency at *decode*.
- The split is share-proportional: `--rpc-auto-weight` sizes slabs by each
  box's measured memory bandwidth, capacity-capped (KV + compute reserve).
- Composition law: t/s ~ 1 / SUM(slab_i / bw_i + hop_i). Removing a slow
  stage speeds up every token (measured: 9 stages 1.0 -> 5 stages 3.4 ->
  3 stages 4.6 t/s on the same V4 model).

## 3. Expert-parallel (`-sm tensor` + `LLAMA_META_ATTN_OWNER`)

For MoE models: only the routed experts are split; one local GPU owns
attention/KV/router; workers hold expert shards. The composition law flips
from SUM-over-stages to MAX-over-contacted-workers.

```mermaid
flowchart TD
    subgraph coordinator box
        A["CUDA0 = attention owner<br/>attention + KV + router + dense<br/>(0% expert share)"]
    end
    A -->|"activations (KB)"| W1["worker .11<br/>expert shard ~60%"]
    A -->|"activations (KB)"| W2["worker .15<br/>expert shard ~40%"]
    W1 -->|"partial expert sums"| A
    W2 -->|"partial expert sums"| A
    A -->|"star reduce: host sum,<br/>broadcast identical bytes"| A
```

- Per MoE layer boundary: ONE fused message per worker (proto 4.5/4.6
  `GRAPH_FUSED`: carries the previous reduced value + the next subgraph chain
  + returns the boundary partial in the same response).
- Single-stream is RTT-serialized (~43 boundaries x RTT; workers idle >90%).
  Batching is the scaling axis: B=8 measured x2.85 aggregate.
- Constraint: exactly ONE local GPU may be a member (the owner); a second
  local GPU as expert member corrupts the reduce (#48, rejected at load).
  REMOTE GPUs (a worker box's GPU) are legal members — they are just RPC
  devices — but a small-VRAM card contributes little expert *capacity* while
  adding a serialized hop, so worker RAM is usually the better member.
- Slow-LINK boxes must stay out of the ring entirely: the cost is per
  boundary, not per byte (100-Mbit member = 0.42 vs 1.82 t/s, #28).

### 3b. The capacity law: whose memory counts (layer vs EP)

EP pools only what can hold *experts* — the owner takes no expert share and
the second local GPU is excluded (#48) — while `-sm layer` pools every
device. Worked example, GLM-5.2 Q2_K_XL (226.9 GiB + 8 GiB reserve =
**240.5 GiB required**) on this fleet, 2026-07-21:

```mermaid
flowchart TB
    subgraph EP["EP pool = worker RAM + owner only -> 178.6 GiB : HOLDS (62 GiB short)"]
        direction LR
        O1["CUDA0 owner<br/>~30 GiB"]:::gpu ---
        E1["local worker<br/>~38 GiB"] --- E2[".11<br/>~58 GiB"] ---
        E3[".25<br/>~28 GiB"] --- E4[".30<br/>~13 GiB"]
        X1["CUDA1 32 GiB<br/>EXCLUDED (#48)"]:::dead
        X2["1660 Ti 6 GiB<br/>legal, ~nil capacity"]:::dead
    end
    subgraph LAYER["-sm layer pool = every device -> 231.5 GiB : 9 GiB short, auto-starts when a box joins"]
        direction LR
        L0["CUDA0<br/>~30 GiB"]:::gpu --- L1["CUDA1<br/>~29 GiB"]:::gpu ---
        L2["1660 Ti<br/>~5 GiB"]:::gpu --- L3["5 CPU workers<br/>~167 GiB"]
    end
    classDef gpu fill:#8ecae6,color:#000
    classDef dead fill:#e5e5e5,color:#888,stroke-dasharray: 5 5
```

- The capacity gate prints exactly this math and, under `--rpc-discover`,
  holds the load and starts it automatically when a new box beacons.
- Sharded GGUFs are sized as the SUM of all `-NNNNN-of-NNNNN` siblings
  (fixed 2026-07-21: 0-based `llama_split_prefix` — before that every
  sharded model was silently sized as shard 1 alone, and auto-weight could
  hand a worker more than its RAM).

## 4. Weight distribution on load (proto 4.8)

Per tensor/slice > 10 MiB, the coordinator offers a content hash before
streaming. The worker satisfies it from the cheapest source it has; every
serve is re-hash-verified, so a stale file degrades to streaming, never to
corruption.

```mermaid
sequenceDiagram
    participant C as coordinator
    participant W as worker
    C->>W: SET_TENSOR_HASH2 {hash, src name+offset+rows}
    alt disk cache has the hash file
        W->>W: read cache-rpc/hash-file, verify
        W-->>C: have it (result=1)
    else local GGUF has the whole tensor (hash index)
        W->>W: pread + verify
        W-->>C: have it
    else local GGUF + provenance (4.8): rebuild the SLICE
        W->>W: pread rows at src offset/stride, verify
        W-->>C: have it
    else nothing matches
        W-->>C: miss (result=0)
        C->>W: SET_TENSOR (stream full bytes)
        W->>W: save to cache-rpc/hash-file
    end
```

The 4.8 slice branch is what makes `-ts`/auto-weight share changes
stream-free on `--model-dir` boxes: boundaries move, hashes change, but the
provenance (tensor name + row range) still resolves against the local GGUF
(measured -71% wire bytes on a boundary-moved load). Older workers simply
never see HASH2 (version-gated) and keep the first three branches.

## 5. Worker cache lifecycle (housekeeping, #45)

```mermaid
flowchart LR
    S["stream received<br/>(> 10 MiB)"] -->|"save as fnv-hash of bytes"| E["cache-rpc/hash-file"]
    E -->|"served on a later load"| T["mtime refreshed<br/>(entry is HOT)"]
    E -->|"model/-ts changed:<br/>new bytes = new hashes"| Z["entry never asked<br/>for again (goes cold)"]
    W["over GGML_RPC_CACHE_LIMIT_MIB?"] -->|"evict oldest-mtime first"| X[deleted]
    Z -.-> W
    N["worker startup"] --> W
    V["every save"] --> W
```

Entries are never *invalidated* — they are content-addressed, so correctness
is settled by the verify-on-read; they just go cold when nothing offers their
hash anymore. The LRU cap (enforced on every save AND once at startup) is
what reclaims them; the beacon's `cache_mib=` makes the pressure visible in
the fleet UI.

## 6. Fleet lifecycle (discovery, capacity gate, recovery)

```mermaid
stateDiagram-v2
    [*] --> discovering: startup - listen 2.5s for beacons
    discovering --> capacity: register workers, auto-weight scores
    capacity --> waiting_capacity: pooled memory < weights + reserve
    waiting_capacity --> waiting_capacity: recheck every 3s, UI shows shortfall
    waiting_capacity --> exit42: fresh beacons cover the shortfall
    exit42 --> discovering: restart policy relaunches
    capacity --> preflight: pool sufficient
    preflight --> loading: split chosen, stream/cache/model-dir
    loading --> serving: model loaded
    serving --> surgical: worker connection lost
    surgical --> serving: worker returned, journal replayed in ~2 min
    surgical --> reloading: worker stayed dead / replay failed
    reloading --> discovering: exit 42 or in-process re-split
```

The same exit-42 edge powers three flows: the capacity gate, the fleet UI's
Include button, and `--rpc-reload` worker-loss recovery — one mechanism,
"restart and re-discover whoever is present".

## 7. MTP speculative decoding under tensor-split

MTP (multi-token prediction) models carry a trained NextN/MTP layer. The fork
runs it as a self-speculative draft: the MTP head drafts N tokens, the target
verifies them in ONE batched decode. Two things make this work on the
2-GPU tensor-split (meta) backend:

```mermaid
flowchart TD
    subgraph "per round"
        D["MTP head drafts up to n_max tokens<br/>(padded to FIXED n_max)"] --> V["target verifies draft batch<br/>in one decode (batch = n_max+1)"]
        V -->|"accepted prefix"| O[emit tokens]
        V -->|"first mismatch"| R["rollback + continue<br/>from corrected token"]
    end
    V --> C{"graph cache:<br/>seen this ubatch shape?"}
    C -->|hit ~100%| G["reuse cached decode graph<br/>(build+alloc = 0 ms)"]
    C -->|miss| B["build + cache<br/>(LRU, N shapes, own scheduler)"]
```

- **Draft padding**: variable-length drafts would change the batch shape every
  round and defeat graph reuse; padding to a fixed `n_max` keeps ONE shape
  (33.6 -> 62-65 t/s single-stream when introduced).
- **Multi-shape decode graph cache**: each cached graph carries its own
  scheduler + meta shadow ring slot, so the tensor-split backend replays
  without rebuilding split states (prod: 27B ~81 t/s, 35B ~166 t/s serving).
- Acceptance is the whole game: ~88% on the prod models; hy_v3 needs
  `p-min 0.75` (its single-depth head collapses at the default).

## 8. TP island (worker-side tensor parallel)

A worker with >= 2 GPUs can expose them as ONE device (`--tensor-parallel`);
the coordinator uploads per-tensor split states so the island shards
weights/KV internally (NVLink/P2P AllReduce inside the box, one RPC endpoint
outside).

```mermaid
flowchart LR
    C[coordinator] -->|"split states by tensor name,<br/>then normal RPC traffic"| I
    subgraph I["island worker - 2 GPUs, one RPC device"]
        G0[GPU0 shard] <-->|"AllReduce<br/>(NVLink/P2P)"| G1[GPU1 shard]
    end
```

## 9. ssd-streaming (3-tier expert placement, single box)

For MoE models bigger than RAM: non-expert weights stay resident; routed
experts stream on demand with two caches in front of the SSD.

```mermaid
flowchart TD
    R["router picks k experts<br/>for this token"] --> V{"VRAM slot cache<br/>(id -> slot indirection)"}
    V -->|hit| K["compute on GPU<br/>(zero copy)"]
    V -->|miss| M{"RAM arena (LRU)"}
    M -->|hit| H["H2D upload to a slot"] --> K
    M -->|miss| S["O_DIRECT pread from SSD<br/>(parallel readers)"] --> H
```

Hit rates beat projections (73% at 30 GB cache for V4-class routing skew);
the regime is IO-bound runnability, not speed — the fleet (diagram 1) is the
faster answer whenever pooled RAM exists.
