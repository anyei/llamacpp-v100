# Beating 6 t/s: ceiling math and the next architecture (2026-08-13)

Purpose: given the new rig - 3x local V100 32GB + local CPU (DDR4-2666) + 1
remote worker V100 32GB (128 GB VRAM total) - derive the decode-speed ceiling
from first principles, then the architecture that reaches it. Model of record:
DeepSeek-V4-Flash (~86.7 GB on disk, profile research/v4flash-80g-profile-2026-08-13.json).
All formulas shown; every constant is either measured in this repo or labeled
as an assumption. The fork's own laws apply: decode is memory-bound; the fleet
exists only for over-local-VRAM models (#31); single-stream fleet decode is
boundary-serialized (#49).

---

## 1. Constants

Hardware (datasheet + in-repo measurements):

| Resource | Value | Source |
|---|---|---|
| V100 32GB HBM2 bandwidth | 900 GB/s theoretical, ~800 effective (GEMV streams) | datasheet, assumption on effective |
| Local VRAM (3 GPUs) | 96 GB | rig |
| Worker VRAM (1 GPU) | 32 GB | rig |
| DDR4-2666 (CPU RAM) | 41.6 GB/s dual-ch, 85.3 quad-ch theoretical; ~0.8x effective | datasheet |
| LAN to worker | 10 GbE TCP, RTT ~300 us; RoCE planned 10-20 us | #60/#117 (46 MB/s = 4% of 10G) |
| NVLink (local GPU pair) | ~50 GB/s/dir assumed for the bridged pair; 3rd GPU interconnect UNKNOWN | assumption - open question 4 |
| Measured boundary turnaround (fused EP path) | ~4.6 ms/boundary at 5.06 t/s (198 ms/token / 43) | EP+dspark FB validation |
| Ping overhead inside the fused path | ~104 pings/token ~= 30 ms pure RTT | 43-boundary attribution (#84 era) |

Model (V4-Flash, from the 2026-08-13 profile + GGUF facts):

- 43 layers, 256 routed experts/layer, **top-6 routing** (`n_expert_used: 6`).
- Routed experts ~= 74 GiB total (~80 GiB upper bound in notes) => **~6.8 MiB
  per expert per layer** (74 GiB / 43 / 256).
- Dense remainder (MLA attention, delta-net blocks, embeddings, norms):
  ~= 86.7 - 79.4 ~= **7 GB**, almost all read every token.
- Hybrid gated-delta-net: recurrent layers carry fixed-size state, so KV grows
  only on the full-attention layers (MLA-compressed). VRAM headroom matters -
  see risks.

Per-token bytes (the only arithmetic that matters for decode):

```
expert bytes/token = 43 layers x 6 active x 6.8 MiB        = 1.76 GiB
dense  bytes/token ~= full dense set                        ~= 7 GB
total              ~= 8.8 GB/token (any placement that keeps everything in HBM)
```

## 2. Ceiling table - what binds each candidate config

Ceiling = 1 / max(stage times); the wire stages are serialized, not overlapped,
in the current implementation (that is the #49 law).

| Config | Expert-read stage | Dense stage | Boundary stage | Ceiling (math) |
|---|---|---|---|---|
| A. Current fleet record (RAM experts, TCP) | ~1.6 GiB from DDR4 ~= 40-50 ms + CPU compute | owner on HBM | 43 x ~4.6 ms ~= 198 ms serialized | **~5-6 t/s = today** |
| B. 4-way EP over TCP (all VRAM) | 1.76 GiB/4 @800 = 0.55 ms | owner 7 GB @800 = 8.75 ms | 43 x ~1-4.6 ms = 43-200 ms | ~5-23 t/s - wire still binds |
| C. 4-way EP over RoCE (all VRAM) | 0.55 ms | 8.75 ms (owner) / 2.9 ms (split 4) | 43 x ~0.1-0.3 ms = 4-13 ms | ~30-90 t/s |
| D. **In-box 3-GPU EP (all VRAM, no wire)** | 1.76 GiB/3 @800 = 0.73 ms | 7 GB/3 @800 = 2.9 ms (dense split) | 43 x ~0.05-0.2 ms local = 2-9 ms | **~70-140 t/s** |
| E. Aggregate-bandwidth bound (the user's "what the math allows") | (1.76+7) GB / (3 x 800 GB/s) = 3.7 ms | - | 0 (in-box, ideal) | ~270 t/s theoretical max |
| F. Same with CPU experts in the decode path | any byte from DDR4 costs 25x HBM; 10% of expert bytes on CPU = +4.4 ms | - | - | drops to ~40-90 t/s - keep the CPU OUT of decode |

Rows D/E are the answer to "what the math allows": with the model fully in
local HBM and zero wire in the decode path, the floor is the in-box boundary
machinery (43 local reduces), not bandwidth. 270 t/s is the unreachable ideal;
**70-140 t/s is the honest theoretical band, and 30-60 t/s is a defensible
engineering target** (40-50% of band, matching the fork's historical
efficiency ratios).

## 3. The regime change the rig creates

V4-Flash (86.7 GB) now fits the LOCAL box alone: 96 GB VRAM >= 86.7 + KV +
compute reserve (tight, ~5-9 GB headroom - see risks). Per the fork's own #31
law 1 ("never distribute a model that fits fast local VRAM - 108 t/s local vs
1.9 fleet"), the correct V4 placement moved from "fleet model" to "in-box
model" the day the 3rd GPU landed. Every millisecond of the current 6 t/s
record that is wire (43 boundaries x turnaround, ~104 pings of pure RTT) is
self-inflicted in the new regime.

The 4th GPU (remote worker) does not help single-stream V4 latency: adding it
re-imports 43 wire boundaries per token (row B vs row D). It helps everywhere
else - throughput batching, GLM-class models (227 GB), KV annex, drafter host.

## 4. Proposed architecture A - in-box all-VRAM EP (ship-it-now)

Shape: `--device CUDA0,CUDA1,CUDA2` + `LLAMA_META_ATTN_OWNER=0,1,2` (owner
group of 3, already legal after the #118 guard relaxation) with #74/#75
frequency-derived expert placement across the 3 members. This is exactly the
eplocal machinery (#118/#122) minus the CPU member.

- Attention/dense: owner group of 3, dense tensors interleaved across owners
  (the #70 dual-role treatment that holds the 5.67-6.22 t/s fleet records) -
  dense reads split 3 ways, reduces stay in-box.
- Uniform CUDA roster => the backend's full-set comm context initializes
  (no RPC member to veto it): AllReduce over NVLink/P2P instead of host
  staging - the thing #105 built the subgroup path for, here granted for free.
- CPU member: OUT of the decode path (row F). Uses: LOCAL_DRAFT drafter host
  (+41% measured when coordinator-local), page cache, KV overflow only if
  forced.
- Speculation: dflash drafter pinned (CPU via LOCAL_DRAFT, or a GPU0 share),
  **NMAX=2** - the union law now runs on HBM: w2 costs 1.62-1.70x reads vs
  E[tokens/pass] 1.8-1.9 at the measured 83-93% acceptance => +10-15%
  (research/ep-spec-improvement-directions-2026-08-05.md direction #1, never
  A/B'd at n2).
- The worker: excluded from the V4 roster. Alternatives in section 6.

Expected ladder (each step gated byte-exact c80261ff + PPL before banking):

1. In-box 3-GPU EP, target-only: math band 70-140 t/s; expect **25-45 t/s**
   first working point (boundary machinery + MLA compute eat the rest; the
   single-box V4 kernel burst gives ~9.7 t/s at -ncmoe 37 with DDR4 experts,
   and every one of those DDR4 reads just moved to HBM).
2. + dspark/dflash NMAX=2: **x1.10-1.15**.
3. + boundary-count diet (below, P3): each fused pair of boundaries removed is
   ~0.1-0.2 ms/token in-box; the whole diet is worth ~2-5 ms/token.
4. Throughput mode (serving): -np 2-4 batching amortizes the 43 boundaries
   across sequences - the old EP batching curve (1.66->4.73 t/s at B=1..8 on
   RAM experts) steepens sharply when members are HBM-bound.

## 5. Proposed architecture B - two-plane fleet (for GLM-class and the worker)

For models that still do not fit locally (GLM-5.2 227 GB, Kimi2.7), and to
monetize the 4th GPU, the boundary chain - not bandwidth - remains the enemy.
Ordered by ROI, all within existing fork machinery:

1. **P1 batching first** (measured, zero build): boundary cost is per-graph,
   not per-sequence. -np 2-4 divides the 43-boundary chain by the batch.
2. **P2 fix the ping/fused-drain ratio** (code, no hardware): 2.4 pings per
   fused call ~= 30 ms/token of pure RTT against a design goal of 1 message
   per member per boundary. `rpc_drain_pings` call-site audit around the fused
   path. Direct reclaim on every fleet config.
3. **P3 boundary-count diet:** (a) fuse the attention-exit and expert-exit
   reduces of one layer where both fire (proto already has fused boundary
   commands); (b) generalize GGML_META_EXPERT_DEFER (+34% on loopback) from
   loopback to wire members - consume the expert sum one layer late so the
   boundary response leaves the critical path; (c) NMAX=2 keeps verify passes
   cheap (union law on HBM members).
4. **P4 RoCE when hardware lands** (#60): RTT 300 us -> 10-20 us; only binds
   after P1-P3, then takes the wire floor from ~77 t/s to ~1500 t/s - i.e. out
   of the way permanently.
5. **Worker roles that pay today:** zero-share drafter (`-ts 0` member - the
   #114 shape now that ffn_down is an island exit; TRUE-zero owner residual
   still open), KV annex for long-context V4 (frees local VRAM headroom),
   dedicated GLM expert member, or second-model host.

## 6. Explicitly rejected (with the reason on record)

- **4-way EP over TCP for V4 single-stream** (row B): re-imports the wire into
  every token for +33% bandwidth the model does not need - 5-23 t/s ceiling.
- **CPU expert member in the V4 decode path** (row F): DDR4-2666 is 25x slower
  per byte than HBM; the eplocal 3.04 t/s first-serve was the thread-starved
  preview of this. CPU RAM's role is drafter + cache, not weights.
- **LP pair-fusion** (arXiv 2502.02790): halves boundaries, closed
  quality-catastrophic in-fork; the paper's own GSM8K caveat agrees.
- **Block-parallel / spec-over-boundary on MoE fleets:** closed negative (#85,
  #71 stage-1 ratio 1.00); the union law only relaxes on HBM members (hence
  NMAX=2 above, not n3+).

## 7. Risks and open questions

1. **VRAM headroom on the local box is ~9 GB** (96 - 86.7): KV for long
   context + compute reserve must fit. Gate architecture A at the production
   context size, not at 4k. If it does not fit: KV annex to the worker (B5)
   or shave the context, not CPU experts.
2. **3rd-GPU interconnect unknown**: if GPU2 is PCIe-attached (no NVLink to
   the pair), dense-split AllReduce traffic crosses PCIe x16 (~13 GB/s real):
   ~105 MB/token of TP traffic = ~8 ms/token - halves the ceiling. Placement
   answer: owner group = the NVLink pair, GPU2 experts-only with a smaller
   share (the wizard already supports per-device expert shares). Measure the
   link first (`nvidia-smi topo -m`).
3. **In-box boundary cost is assumed 0.05-0.2 ms** (local fused reduce). If it
   measures near the 4.6 ms fleet number, the whole thesis shifts to P3/P4 -
   measure one in-box EP serve before building anything.
4. **Draft VRAM/draft-on-CPU cost** at NMAX=2 with LOCAL_DRAFT: measured for
   n_max 3-5 on loopback, not on the 3-GPU roster - part of the A ladder.
5. **Where the worker V100 goes once V4 leaves the fleet** - GLM-5.2 EP debut
   with the staged profile, or drafter/KV annex. A fleet decision, not a code
   one.

## Bottom line

The math allows ~70-140 t/s for V4-Flash on this rig; the path there is not a
new subsystem but a placement regime change (in-box all-VRAM EP, owner group
of 3, CPU out of decode) plus the already-filed boundary-chain fixes (P1-P3)
for the models that still need the fleet. The first engineering target is
**25-45 t/s target-only**, gated byte-exact - 4-7x the current record - with
NMAX=2 speculation and the boundary diet as the follow-on multipliers.
