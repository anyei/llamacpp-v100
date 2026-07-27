# True Distributed Inference: Coordinator + N Workers

Status: analysis / design document. Nothing here is implemented yet beyond what is
marked "exists today". Target deployment: 1 coordinator docker instance + N worker
docker instances, each bound to its own GPU(s), on one or several hosts.
Reference hardware: Tesla V100-SXM2 (Volta, sm70, NVLink NV2 pairs), scaling from
2 to 4+ GPUs.

## 1. What exists today

### 1.1 RPC backend (worker side: `rpc-server`, `ggml/src/ggml-rpc/`)

- A worker runs `rpc-server` and exposes all its local accelerators as ggml
  devices over a hand-rolled binary TCP protocol (optional RDMA/RoCEv2).
- The coordinator is a normal `llama-server`/`llama-cli` started with
  `--rpc hostA:port,hostB:port,...`. Remote devices then behave like local ones
  for every split mode.
- Model weights and the KV cache for layers assigned to a worker are allocated
  and stay on the worker (`RPC_CMD_ALLOC_BUFFER`); only activations at split
  boundaries cross the wire. Weight upload can be skipped on restart via a
  worker-side file cache keyed by tensor hash (`SET_TENSOR_HASH`, `-c` flag).
- Graph shipping is optimized: the full graph is serialized once, and unchanged
  graphs re-run via a tiny `GRAPH_RECOMPUTE` message.

### 1.2 Split modes across devices

- `-sm layer` (pipeline): each device owns a contiguous layer slab + its KV.
  One activation handoff (n_embd x n_tokens) per stage boundary per microbatch.
  Works over RPC today and is the only mode that is *practical* over a network.
- `-sm tensor` (tensor parallel, meta backend `ggml-backend-meta.cpp`): each
  layer's weights are sharded across all devices; ~2 AllReduces per layer per
  step. On a single host this uses NCCL (or an internal CUDA AllReduce) and is
  the best mode for token-generation latency (measured on 2x V100: tg 47.8 t/s
  vs 33.2 for layer; pp 1576 vs 1096).

### 1.3 Hard limitations for distributed use (verified in source)

> Historical baseline (2026-07-08, design time). Several rows have since
> shipped fixes: worker-to-worker copies (proto 4.2, fenced pulls), async RPC
> (markers + events), fault tolerance (#29: surgical re-provision +
> `--rpc-reload`), capability negotiation (op-fingerprint HELLO, proto 4.11).
> Current state: `docs/distributed-inference-guide.md`.

| Limitation | Where |
|---|---|
| NCCL init is single-process only (`ncclCommInitAll`) | `ggml-cuda.cu:1393-1414` |
| Meta-backend AllReduce over RPC falls back to a butterfly that stages every slice through the coordinator host (GET_TENSOR + host add + SET_TENSOR per step) | `ggml-backend-meta.cpp:2073-2217`, `ggml-backend.cpp:460-519` |
| RPC backend exposes no `comm_init`/`comm_allreduce` proc address, so the fast AllReduce path is never taken | `ggml-rpc.cpp:1883-1893` |
| Worker-to-worker copies refused (`COPY_TENSOR` only within one endpoint) | `ggml-rpc.cpp:501-521` |
| RPC is fully synchronous: no async ops, one in-flight command per socket | `ggml-rpc.cpp:652-655, 1798-1803` |
| No fault tolerance: any bad response aborts the coordinator | `ggml-rpc.cpp:30` |
| No auth/TLS ("never run on an open network") | `tools/rpc/README.md`, `rpc-server.cpp:301-309` |
| `supports_op` always true; capability negotiation absent | `ggml-rpc.cpp:1822-1827` |

## 2. Network budget: why mode choice matters

Per-token traffic for Qwen3.6-27B class models (n_embd = 5120, ~64 layers,
fp16/bf16 activations, single stream):

- Tensor parallel: 2 AllReduces per layer x 64 layers = 128 sync points per
  token. Payload per AllReduce at batch 1 is n_embd x 2 B = 10 KiB - trivial
  bandwidth, but each sync is a round trip. Over TCP with ~50-100 us RTT that
  is 128 x (RTT + software overhead) = 10-25 ms per token of pure latency,
  i.e. it would erase the entire 21 ms/token budget. Over NVLink/NCCL on one
  host the same syncs cost ~20-40 us each - that is why single-host TP works.
- Pipeline (layer split): N_stages - 1 handoffs per token (e.g. 3 for 4 GPUs),
  each n_embd x 2 B. Latency adds ~3 x RTT per token: sub-millisecond even on
  ordinary Ethernet. But batch-1 latency does not improve with more stages
  (each token still visits every layer sequentially); only throughput under
  concurrent load improves via pipelining.

Conclusion: **tensor parallel must stay inside a fast-interconnect domain
(NVLink/PCIe within a host); pipeline parallelism is the only sane cross-host
mode.** The right multi-host architecture is hierarchical: TP within each
worker, pipeline between workers.

## 3. Important caveat for the 4x V100 single-host plan

If all 4 V100s end up in ONE host, "distributed" (coordinator + worker
containers) is the wrong tool: a single llama-server process with
`-sm tensor -ts 0.25,0.25,0.25,0.25` and NCCL will beat any multi-container
split, because NCCL AllReduce over NVLink is 2-3 orders of magnitude lower
latency than loopback TCP RPC. Containers add nothing but isolation overhead
on one host. The coordinator/worker design pays off only when GPUs live in
different hosts.

Caveat inside the caveat: V100-SXM2 NVLink topology is usually pairwise (NV2
between neighbors, PCIe across pairs). `nvidia-smi topo -m` on the 4-GPU host
will tell whether all pairs are NVLinked; if only pairs are linked, expect
AllReduce over 4 GPUs to be bottlenecked by the PCIe hops (NCCL handles this
with ring construction, but per-sync latency rises). Measure tg with
`-sm tensor` on 4 GPUs vs 2 before committing.

## 4. Proposed architecture

```
                +--------------------+
                |    coordinator     |   llama-server (HTTP API, sampling,
                |  (may own GPUs 0,1)|   tokenizer, scheduler, spec decode)
                +---------+----------+
                          | RPC control channel (graph, activations)
          +---------------+----------------+
          |                                |
 +--------+---------+            +---------+--------+
 |     worker 1     |            |     worker 2     |
 | rpc-server       |            | rpc-server       |
 | GPUs 2,3 (TP via |            | GPUs 4,5 (TP via |
 | local NCCL)      |            | local NCCL)      |
 +------------------+            +------------------+

 pipeline split BETWEEN boxes, tensor split WITHIN each box
```

Key idea: make one RPC worker with K local GPUs look like ONE fast device to
the coordinator (a "TP island"), instead of K independent slow devices.

## 5. Implementation roadmap

### Phase 0 - works today, no code changes (baseline to measure)

- Coordinator: `llama-server --rpc worker1:50052,worker2:50052 -sm layer -ngl 99`
- Workers: `rpc-server -H 0.0.0.0 -p 50052 -c /cache` (one per host, all GPUs).
- Docker: `--network host` (RPC has no TLS; keep it on a private network or
  WireGuard overlay), `--gpus` per worker, model file only needed on the
  coordinator (weights stream once, then hash-cached on workers).
- Expected: correct, tolerable prefill, but generation latency gated by
  synchronous handoffs; no TP inside workers (each GPU is its own stage).

### Phase 1 - TP islands: meta backend inside the worker (medium effort)

Today `rpc-server` registers each GPU as a separate device. Add a worker flag
`--tensor-parallel` that wraps all local GPUs in a `ggml_backend_meta_device`
(same code path `-sm tensor` uses in-process) and exposes ONE device over RPC.
- Touchpoints: `tools/rpc/rpc-server.cpp` (device enumeration),
  `ggml/src/ggml-rpc/ggml-rpc.cpp` server init; meta backend is already a
  normal ggml device so most of it composes.
- Result: coordinator does `-sm layer` across workers; each worker internally
  runs NCCL TP across its GPUs. This is the hierarchical architecture with
  minimal new protocol work, and it is the highest-value phase.

### Phase 2 - async RPC + double-buffered pipeline (medium effort)

- Implement `set_tensor_async` / `graph_compute` overlap in the RPC client and
  a second in-flight command slot on the server, so stage N+1's input upload
  overlaps stage N's compute. `ggml-backend-sched` already double-buffers
  (`n_copies=4`) when backends report async capability; the RPC backend just
  needs to report and honor it (`ggml-rpc.cpp:1798-1803`).
- Add TCP_NODELAY + connection keepalive audit; optional RDMA where NICs allow.
- Result: multi-request throughput scales with number of workers (true
  pipelining), single-stream latency roughly unchanged.

### Phase 3 - cross-host tensor parallel, only if a fast fabric exists (large)

Only worth it with RDMA/RoCE or >= 25 GbE + kernel-bypass; otherwise skip.
- Replace `ncclCommInitAll` with `ncclCommInitRank` + bootstrap: coordinator
  generates `ncclUniqueId`, distributes it over the RPC control channel, each
  worker process joins the clique; NCCL then does allreduce worker-to-worker
  directly (NCCL supports sockets and IB transports natively).
- Touchpoints: `ggml-cuda.cu:1390-1460` (comm init), new RPC messages for
  bootstrap exchange, meta backend gains a "remote member" mode where
  subgraph dispatch goes over RPC but reductions go over NCCL.
- This also cleanly covers the "multi-container, single host" case (NCCL
  works across processes on one host via SHM/P2P transports, containers need
  `--ipc=host` and shared /dev/shm).

### Phase 4 - operational hardening (incremental, parallel to any phase)

- Worker health endpoint + coordinator-side reconnect/retry instead of
  `GGML_ABORT` (`ggml-rpc.cpp:30`); graceful degradation is model-fatal but
  should produce an HTTP 503, not a dead coordinator.
- Authentication token on the RPC handshake (HELLO message already exists,
  `ggml-rpc.cpp:330-347`) + optional TLS via the existing httplib/openssl dep.
- Docker compose reference: coordinator service + N worker services,
  healthchecks, `NCCL_SOCKET_IFNAME`, `--ipc=host` guidance, private overlay
  network. Worker image = current cuda image + `rpc-server` binary (already
  built; just not shipped in the server image target - add to
  `.devops/cuda.Dockerfile`).

## 6. What to measure at each phase (the numbers that matter)

- tg t/s single stream and pp t/s at 4k/32k, per phase, vs the single-process
  2-GPU baseline (tg 47.8, pp 1576 on this fork, build 2da668617).
- Per-token added latency of the RPC hop: run worker on same host over
  loopback first (isolates protocol cost from network cost).
- 4x V100 single host: `-sm tensor` 4-way vs 2 TP islands of 2 (`-sm layer`
  between islands) - the NVLink pair topology decides the winner.

## 6b. Measured update 2026-07-27: the fleet is boundary-bound (gate 5, #75)

Hot-expert placement A/B on the record hy3 EP roster: uniform 3.43 t/s vs
placed 3.29 t/s - a null result with the implementation verified correct.
Cutting the slow members' expert bytes ~3.6x moved nothing, so the binding
constraint is the per-layer boundary cost (~294 ms/token / 80 layers ~= 3.7
ms/boundary, one GbE RTT + reduce), not bandwidth. A GGML_META_TIMING
decomposition (TASKS #7) splits compute vs reduce vs wire before further
roadmap commitment.

Consequence - "more workers -> faster single-stream tokens" has exactly three
escapes (full survey: docs/research/2026-07-24-horizontal-scaling.md section 9
and the 2026-07-27 addendum in 2026-07-parallel-decoding-and-distribution.md):

| Escape | Mechanism | Leading candidates | State (2026-07-27 eve) |
|---|---|---|---|
| (a) cheaper boundaries | cut the per-hop latency floor / byte cost | **f16 wire format LANDED (proto 4.12, +19.6%)**; q8_0 wire next if PPL allows; soft-RoCE (rxe) prototype, then cheap ConnectX (#60); UCCL-EP CPU-proxy pattern | f16 measured, PPL gate pending |
| (b) fewer boundaries | cut sync points or participants | **boundary fusion LANDED (GGML_META_BCAST_FUSE, +15% pooled)**; Layer Parallelism pair-fusion (80->40 syncs, quality-gated); METRO-style member-skipping reduce | fusion measured + default-on in fleet scripts |
| (c) fill the boundaries | overlap the wait with useful work | spec-over-boundary (#71 stage 1), SpecPipe/PipeInfer pipeline filling, ktransformers Expert Deferral (corroborated by APEX's deferred-sync) | next frontier once (a)/(b) plateau |

The dated notes below are a chronological ledger - each is superseded by
later ones. Naming: "TASKS #9" in these notes refers to the 2026-07-24/27
session tracker's boundary-fusion task, NOT item 9 of TASKS.md (which is the
unrelated GPU-sampling investigation). Bottom line as of 2026-07-27 evening:
record-roster decode went 2.97 (all off) -> 3.42 (BCAST_FUSE=2) -> 4.09 t/s
(+ GGML_RPC_WIRE_F16), all legs coherence-read.

**#7 MEASURED 2026-07-27 (GGML_META_TIMING on the record hy3 EP roster):**
compute 403-405 ms/graph, reduce 180-186 ms/graph over **159 boundaries/graph
= 1.99 per layer** (attention-owner broadcast + expert-sum reduce), reduce
cost **1.15 ms/boundary**. Instrumentation serializes overlap (instrumented
sum 587 ms = 1.70 t/s vs 292 ms = 3.43 t/s uninstrumented), and reduce is
synchronous network work that does not compress when uninstrumented, so the
inferred production split is **reduce ~183 ms (63%) / compute ~109 ms (37%)
of each 292 ms token**. This confirms the boundary-bound hypothesis AND
explains the gate-5 null quantitatively: placement attacked a slice of the
37%. Ceilings from the same numbers: halve the boundaries (2/layer -> 1 by
fusing the owner broadcast into the expert reduce) -> **5.0 t/s**; free
boundaries (RDMA-class) -> **9.2 t/s**. The per-layer boundary pair is the
single most concrete new target: escape (b) has a named, code-only first
step (boundary fusion) before any hardware.

**Per-kind split (2026-07-27b):** attn-broadcast 79 ms/graph over 80
boundaries (0.99 ms each) vs true reduce 110 ms/graph over 79 (1.39 ms each).
Production time ~= instrumented compute + B1 + B2 (109+79+110 ~= 292), so the
pipeline has effectively NO overlap: the RPC client issues writes async, but
per-socket ordering queues the next boundary's READ behind them. Fusion plan
refined accordingly (TASKS #9): (1) let the delay walker push a
single-contributor PARTIAL (the owner-broadcast pattern) across the mirrored
residual ADD so B1 broadcasts ffn_inp instead of attn_out; (2) make B2
gather-only, delivering just to the members that compute before the next
broadcast (the next layer's owner - an NVLink write). Workers' residual copies
then go permanently stale by design. Estimated -40-60 ms/token -> ~4.1-4.3 t/s.

**Fusion design status (2026-07-27c, TASKS #9):** design complete, paused
before code on an honest measurement gap. The delivery of a boundary value to
a wire member rides fused_send, which bundles value + next-subgraph trigger in
ONE message - so the B2 writeback may already cost zero extra round trips, and
the per-kind timing above cannot distinguish pipelined from serialized cost
(its per-boundary drains serialize everything they measure). Before any code:
(i) a wall-time-only A/B toggling GGML_META_NO_STAR, (ii) a probe leg with the
B2 writeback payload zeroed inside fused_send to price it truly, (iii)
async-completion timestamps. If the writeback prices at ~0, the remaining
levers are the B2 GATHER (the only irreducible read RTT per layer) and escape
(c) fill-the-bubble. Walker-crossing + delivery-skip design details and the
cross-piece safety question are recorded in TASKS #9.

**Probe (ii) MEASURED 2026-07-27d - the writeback costs ~47 ms/token:**
GGML_META_PROBE_NO_WRITEBACK=1 (chain-only fused messages, reduced values
never delivered; garbage output by design) on the record roster: eval
244.9/227.7/260.9 ms/token = mean 244.5 vs 291.5 baseline -> the boundary
writeback's true critical-path price is ~47 ms/token (~16%), i.e. ~4.1 t/s
if eliminated where unconsumed. So fused_send does NOT hide it in production
(per-socket ordering queues the next gather behind it, as suspected). The #9
build decision is GO: walker crossing + delivery skip, prize ~+20% decode,
with the value-less fused message path already validated by this probe (it IS
one). Caveat: garbage activations could shift CPU compute time slightly
(denormals/NaN); direction and magnitude match the wire arithmetic, so
accepted. Cross-piece consumer safety remains the one open design question
before delivery-skip ships.

**#9 BUILT AND MEASURED 2026-07-27e - boundary fusion lands +15% pooled
(conservative +7-9%):** `GGML_META_BCAST_FUSE` (default OFF; env-gates.md) in
ggml-backend-meta.cpp. Level 1 walker crossing: a single-contributor PARTIAL
(owner-broadcast) crosses the mirrored residual ADD in get_i_delayed, so B1
broadcasts `ffn_inp` instead of `attn_out` (guarded against ADD_ID partial
merges - after one, the owner's local value is no longer the logical sum).
Level 2 delivery skip: rebuild-time consumer DFS per reduce boundary per WIRE
member (view-chain aware; unsafe on reduce-contribution / OUTPUT / in-place /
piece-end contamination); withheld values ride value-less fused messages.
Cross-piece consumers (the open design question) SOLVED by a stale-value
registry keyed (buffer uid, data offset) + fingerprint, with repair copies at
piece start from a member holding the true value - engagement visible via
`SKIP-WB:`/`REPAIR:` under GGML_META_DEBUG_REDUCE=1. Gates: CPU loopback
byte-exact (pre-change == off == 1 == 2, sha a9294d12, trunc-hy3 2 RPC
members); CUDA trunc roster (CUDA0,CUDA1,RPC0,RPC1) deterministic per config,
crossing on all layers, skip engages exactly where safe (refuses layer-3 cell
whose closure reaches result_output). Fleet A/B (record hy3 roster, same
binary, 6x 100-tok greedy each, coherence READ on every leg): off 2.79 then
3.16 (warm control) vs fuse=1 3.44, fuse=2 3.37 - pooled +15% (t~3.6),
conservative vs warm control +7-9%. Delivery skip is NEUTRAL on top of the
crossing on this roster: the skippable writebacks (mid-layer expert reduces to
wire members) were already piggybacked on chain messages, and the mid-layer
ffn_inp broadcasts cannot be skipped (experts consume them) - consistent with
the 2026-07-27c pause note. The 47 ms probe priced ALL writebacks; the SAFE
subset prices near zero. Next lever on the reduce share: the B2 gather
(irreducible read RTT/layer) and escape (c) fill-the-bubble.

**Boundary traffic census 2026-07-27g (GGML_META_BOUNDARY_STATS, record
roster, fuse=2, production speed - pure counters, no drains):** per token: 80
bcast1 (B1) + 79 star (B2); wire partials arrive **96-98% via the fused
pre-request** (212-218 fused vs 3.5-8.6 plain reads/graph) - the B2 gather has
essentially NO request-RTT left to remove; delivery skip removes **all 237
B2 wire writebacks** (79 x 3 wire members), the 240 delivered are exactly the
B1 ffn_inp broadcasts (~4.3 MiB/graph); repairs 0 (single-piece decode
graphs). Total boundary wire bytes ~8 MiB/token ~= **~68 ms/token of GbE byte
time** (23% of the 292 ms token) split evenly between B1 broadcasts out and
fused gather responses in. => The next lever is **wire-format compression of
boundary payloads** (f16 halves: ~-34 ms/token ~ +13%; q8_0: ~-49 ms/token ~
+20%; distributed-llama ships q80 sync as precedent) - needs an RPC proto
bump + worker roll (both directions convert worker-side). After that, only
arrival-latency overlap (escape (c)) remains.

**f16 wire format MEASURED 2026-07-27h - +19.6% on top of boundary fusion:**
proto 4.12 (`GGML_RPC_WIRE_F16`, commit bc4387b95) rides fused SET payloads
and FETCH responses as f16. Record-roster A/B, same binary and freshly rolled
workers (local + .11 on rpc-worker-bc4387b95; .15 still proto-old = f32
fallback on that connection): f32 wire 3.07-3.71 mean 3.42 vs f16 wire
3.69-4.34 **mean 4.09 t/s**, both legs coherent. Cumulative today: 2.97
(fuse off) -> 3.42 (BCAST_FUSE=2) -> 4.09 (+ f16 wire) = **+38%**, with .15's
roll still pending for full f16 coverage. The gain EXCEEDS the ~+13%
serialized-bytes projection - halving payloads also cuts response arrival
latency and per-socket contention at the star root. Quality: PPL A/B legs on
the record roster (this doc's next dated note) gate any default-on decision.
Gates so far: env-off CPU loopback byte-identical (a9294d12); f16 trunc smoke
token-identical to f32. **PPL GATE PASSED 2026-07-27i (record roster, wikitext
8 chunks, -c 512, .15 on 4.12 for the f16 leg): f32 wire 3.7895 +/- 0.2116 vs
f16 wire 3.7669 +/- 0.2074** - delta ~9x inside the error bar, per-chunk
values track within ~1%. f16 boundary payloads are quality-neutral on the
production model; enabled in the fleet run scripts alongside BCAST_FUSE
(WIRE_F16=0 / COORD_WIRE_F16=0 for A/B legs).

**MTP prod-config A/B 2026-07-27f - fuse is a verified no-op there:** Qwen3.6
-27B-MTP, 2x V100 `-sm tensor -ts 0.5,0.5`, full prod compose flags, same #9
binary, 3x 500-token temp-0 gens per leg: off 66.5-66.7 vs fuse=2 66.3-66.5
t/s AND byte-identical responses on all three pairs (multi-contributor
attention means the crossing never fires; no wire members means delivery skip
never engages; NCCL/P2P allreduce path bypasses the fallback entirely).
Setting GGML_META_BCAST_FUSE globally is safe for the local serving configs -
the flag only acts on EP/dedicated-attention fleet topologies.

distributed-llama's RPi5 result (1->4 workers, 5.95->13.68 t/s over plain TCP,
q80 sync, star, similar-speed nodes) is the existence proof the goal is sound;
its "similar-speed nodes" condition maps to keeping stragglers off the
critical path (replica/class routing for the slow boxes). Placement-family
work (static, v2 budgets, adaptive re-place, GLM artifact) is CLOSED on this
fleet unless #7 contradicts the reduce/wire-dominance expectation.

## 7. Decision summary

| Scenario | Recommended mode | Needs code? |
|---|---|---|
| 2-4 GPUs, one host | single process, `-sm tensor` + NCCL | no |
| GPUs spread over hosts, 1-10 GbE | pipeline between hosts, TP inside each (Phase 1) | yes (Phase 1) |
| Multi-host + RDMA fabric | hierarchical or full TP (Phase 3) | yes (Phases 1-3) |
| Many independent requests, many hosts | N independent replicas + external LB (no llama.cpp changes; vLLM-style disaggregation is out of scope) | no |
