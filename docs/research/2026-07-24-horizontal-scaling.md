# Horizontal distributed inference: field sweep and steal list (2026-07-24)

Iteration 3 of the #67 research track. Question: how does the fork scale
horizontally - more machines helping, with minimal network-latency impact?
Method: 4-agent web sweep (2023-2026 literature, ~50 systems/papers), filtered
against the fleet's measured physics. Prior rounds: `2026-07-parallel-decoding-
and-distribution.md` (iterations 1-2). Caveat: arXiv IDs/links are as reported
by the sweep; re-verify the specific ones when a task is picked up.

Fleet physics (measured, the filter for everything below): single-stream
decode is LATENCY-bound. Layer mode token time = sum of stage times. EP is
serialized by ~43 sequential boundary RTTs/token and is insensitive to member
bandwidth (an 823 GB/s member changed nothing). RTT 0.15-3.4 ms, GbE with
100-Mbit segments. Best V4: single-box 9.8 t/s, layer fleet 4.6-4.8, EP 2.5-2.8.

## 0. The verdict in one page

Horizontal scaling has THREE separate axes and the field buys each with
different mechanisms:

| Axis | Status today | What the field says buys it |
|---|---|---|
| Capacity (model fits nowhere else) | WORKS (pooled RAM) | free - the fleet's whole point |
| Throughput (streams/s, prefill/s) | partial | replicas + request-class routing; micro-batch overlap; PD disaggregation with compressed KV handoff |
| Single-stream latency | the hard one | fewer + cheaper boundaries (placement, star reduce, wire quant, RDMA); or fill the pipeline with speculative work |

The one law every area converged on: **at KB-scale payloads, per-message
latency (alpha) dominates bandwidth (beta)**. TPI-LLM's star-beats-ring
result, exo's TCP/RDMA cliff, Petals' few-fat-boundaries, and our own EP
measurements are the same finding. Hop count and the transport latency floor
are the design variables, not GB/s.

Corollary that inverts datacenter intuition: fast-link systems hide
communication under compute (FLUX, COMET); a slow-link fleet must do the
opposite - hide compute under communication, or fill the communication
bubbles with speculative work. Adding a machine to the latency-critical path
makes single-stream slower (exo's README says it outright; our 9->5->3-stage
law measured it). Added machines must either absorb traffic classes
(replicas, batch, prefill) or pass a latency bar before joining the critical
path.

## 1. Wide-area / heterogeneous serving systems

- **Petals** (Yandex/HSE, ACL'23/NeurIPS'23, arXiv 2312.08361) - pipeline
  swarm over the public Internet, DHT picks lowest-RTT server per block,
  client-side re-route on failure. 6 tok/s Llama-2-70B over WAN. Proof that
  sequential-block PP survives WAN RTTs; greedy placement is its documented
  weakness (solvers beat it 2-3x).
- **SWARM Parallelism** (ICML'23, arXiv 2301.11913) - randomized temporary
  pipelines + stage replicas; failure = re-route, not stall. Trained at
  <200 Mb/s links (our 100-Mbit regime) via compressed boundary activations.
- **Helix** (CMU, ASPLOS'25, arXiv 2406.01566) - placement as MAX-FLOW over a
  GPU+link graph, MILP; per-request pipelines (TP inside node, PP across).
  3.3x throughput, -24% decode latency on 24-42 mixed-GPU nodes. The
  principled form of auto-weight; link heterogeneity is IN the objective.
- **Parallax** (Gradient/HKUST, arXiv 2509.26182, code) - closest match to
  this fleet. Phase 1: DP places layer slices with a latency-dominant
  heuristic (minimize stages per replica, maximize replicas; Z(k) =
  k^a/(T_comp + stages*RTT)) + water-filling by measured FLOPs. Phase 2:
  per-request min-latency chain over a DHT of live per-layer latencies +
  pairwise RTTs (1-2 s refresh). 3.6x throughput vs HexGen at ~10 ms links;
  scheduler <9 ms at 256 GPUs.
- **HexGen-2** (arXiv 2502.07903) - PD disaggregation on heterogeneous
  pools; KV transfer path inside the solver. 1.3-2.0x at equal price.
- **Hetis** (SC'25, arXiv 2509.08309) - offloads ATTENTION at per-head
  granularity to weak GPUs (weight-light, scales by head count). 2.25x. The
  most credible job for our 6 GB GPU / Arc iGPU workers.
- **SpotServe** (ASPLOS'24, arXiv 2311.15566) - re-parallelize on
  join/leave; migration planning as Kuhn-Munkres matching; token-granular
  stateful recovery. The algorithm for "worker joined, rebalance without a
  full restart".
- **exo** (exo labs) - memory-weighted ring partitioning (= our auto-weight)
  and the candid law: "adding less capable devices slows individual latency
  but increases cluster throughput". EXO 1.0 pivoted to PD split (prefill on
  DGX Spark, decode on Mac Studio, KV over 10 GbE).
- **distributed-llama** (b4rtaz, active 2026) - TP over plain TCP, sync
  buffers quantized q80. RPi5 1->4 nodes: decode 5.95 -> 13.68 t/s (~55-60%
  efficiency per added node). Cleanest evidence that TP over Ethernet CAN
  speed single-stream - with quantized sync, few similar-speed nodes.
- **EdgeShard** (IEEE IoT-J'24, arXiv 2405.14371) - DP joint device
  selection + partition; 50% latency cut on a real heterogeneous prototype.
- **Edge-MoE "Prism"** (arXiv 2508.12851; NOT the 2505.04021 Prism) - EP
  across edge servers at tc-limited 500 Mbps: placement objective = expected
  remote-expert invocations weighted by activation frequency;
  entropy-proportional per-layer expert counts; submodular top-frequency
  assignment; periodic re-placement if C(new)+T_mig < C(old). Up to 30.6%
  lower latency than EPLB on DeepSeek-V2-Lite. The most directly applicable
  paper in the sweep - our #74/#75 made adaptive, at our bandwidth.
- **DeServe** (arXiv 2501.14784) - request-level parallelism: replicate,
  route whole requests, zero cross-node decode traffic. 6.7-12.6x in
  high-latency networks. The strongest argument that multi-stream throughput
  comes from replicas, not deeper pipelines.
- Also: Tessera (kernel-granularity disaggregation), Cronus (partial prefill
  offload to the weak box, overlapped), NetKV (network-cost oracle in
  routing), Melange (route by request class), QLM, PolyServe (segregate
  latency-SLO requests from throughput streams - the right pattern for a
  small fleet).

## 2. Expert-parallel / MoE systems

- **MegaScale-Infer** (ByteDance, arXiv 2504.02263) - attention/expert
  disaggregation (= our EP shape) + PING-PONG pipeline parallelism:
  micro-batches shuttle between attention and expert pools so one
  micro-batch's compute hides the other's communication. 1.9x per-GPU
  throughput. The overlap mechanism is interconnect-agnostic - it helps
  MORE when RTTs are large.
- **vLLM wide-EP + DBO** (blog 2025-12) - dual-batch overlap: two decode
  micro-batches offset so one's all-to-all hides the other's compute.
  Ping-pong productized. Also: EPLB rebalance as a LIVE weight shuffle
  driven by a sliding-window load counter (recipe for #75's cadence).
- **METRO** (NVIDIA/Yale, arXiv 2512.09277) - in memory-bound decode,
  token-balancing EPLB BACKFIRES (more distinct experts per GPU = more
  weight bytes). Route to minimize distinct experts/members touched per
  token instead. -11-22% decode latency on 8xA100. A routing-policy change,
  nearly free to try.
- **MoETuner** (arXiv 2502.06643) - cross-layer routing dependency: a
  token's layer-l expert predicts a small set in layer l+1; place affine
  chains on the same member. 17.5% multi-node. Computable offline from our
  LLAMA_EXPERT_PROFILE histograms - attacks the 43-hop count itself.
- **GRACE-MoE** (arXiv 2509.25041) - expert grouping to cut cross-node
  traffic + locality-aware routing (prefer the local replica). 4.66x.
- **ViBE** (arXiv 2606.00735) - speed-aware placement: hot experts to
  faster devices, recalibrate under drift. Our fleet is far more skewed
  than their nominally-identical GPUs; the gain should be larger.
- **Director** (arXiv 2607.08782) - re-place/migrate experts DURING
  compute-bound prefill windows (near-zero downtime). Pairs with our
  worker-to-worker transfer + weight caches.
- **ktransformers SOSP'25 facts**: at low batch, compute offload beats
  weight offload (compute cold experts where they live; transfer weights
  only for persistently hot ones). Expert Deferral: defer some experts to
  overlap the next layer's attention - fills the RTT bubble, +33% decode,
  <=0.5% accuracy cost (needs eval gating).
- **ProMoE** (arXiv 2410.22134) - predicts next-layer experts from
  intermediate activations; preemptible chunked prefetch; 2.07x decode vs
  reactive. ALREADY HAS llama.cpp INTEGRATION HOOKS.
- **Gate-similarity prediction family** (AdapMoE, FATE 78.8-97.2% hit,
  DuoServe, Pre-gated without retraining) - ~90% next-layer prediction from
  residual-stream similarity, no model change. A miss costs the status quo.
- **MoE-Infinity** (arXiv 2401.14361) - personal machines, BATCH 1, per-token
  latency headline: 3.1-16.7x via sparsity-aware activation-trace cache.
- **HybriMoE** (DAC'25, arXiv 2504.05897) - "GPU transfer time is
  transaction-dominated, not byte-dominated" = our LAN finding; prefetch by
  impact (activation prob x time saved), not raw frequency.
- **DeepEP/EPLB** - DeepEP needs RDMA+SM90 (does not transfer; note its
  "low latency" is 163 us ON RDMA - our GbE RTTs are 1-2 orders worse per
  hop, which is why EP-over-Ethernet numbers don't exist in the literature).
  UCCL-EP (Berkeley, arXiv 2512.19849) is the portable pattern: CPU proxies
  issue RDMA for GPUs, works on cheap dumb NICs.
- **ScMoE** (arXiv 2404.05019) - shortcut-connected EP breaks the
  layer-serial comm dependency (needs training; existence proof only).
- The GbE-Ethernet EP regime is essentially UNRESEARCHED - our measured
  facts are a literature gap; overlap + hop-count reduction are the only
  levers, and both are available.

## 3. Latency hiding and transport

- **exo RDMA cliff**: TCP/Thunderbolt ~300 us/hop made cross-device TP
  actively harmful; RDMA at 3-9 us/hop gave 1.8x (2 devices) / 3.2x (4).
  OCI reports ~2 us on small ConnectX+RoCE clusters. This quantifies the
  cliff our 0.15-3.4 ms boundaries sit on the wrong side of. (#60's prize,
  measured by others.)
- **UCCL-EP / UCCL** - CPU-proxy async transfer engine: GPU hands compact
  routing commands to multithreaded CPU proxies that issue the RDMA; works
  on AWS-EFA/Broadcom-class cheap NICs. Exactly what a ggml RPC backend can
  implement without GPU-NIC co-design.
- **NIXL / Mooncake Transfer Engine** - one async point-to-point transfer
  API with pluggable backends (RDMA/GPUDirect/NVMe/TCP fallback). Adopt the
  ARCHITECTURE (single async transfer abstraction in the RPC layer, TCP
  today, verbs tomorrow, per-link path selection), not the libraries.
- **On-the-wire activation quantization**: Apple (arXiv 2411.07942) 16 ->
  4.2 bits avg on allreduce payloads, 98-99.5% task retention;
  Hansen-Palmus (arXiv 2411.09510) 3.5-4.5x activation reduction; DeepEP
  FP8 dispatch; distributed-llama q80 sync. On a 100-Mbit link a 16 KB bf16
  hidden costs ~1.3 ms serialization; q8/q4 cuts it 2-4x. Apply per-link
  adaptively; near-no-op on GbE.
- **TPI-LLM** (arXiv 2410.00531) - 70B tensor-parallel across 4 laptops on
  WIFI: star-topology allreduce beats ring at KB payloads. Direct
  confirmation for our collective paths.
- **Galaxy** (INFOCOM'24) - tile-based fine-grained comm/compute overlap
  for edge clusters: tile the GEMM, pipeline each tile's allreduce behind
  the next tile's compute - hides compute UNDER communication (the correct
  direction for slow links).
- **FLUX / COMET** - SM-level fusion hiding 96% of comm (A100/H100 NVLink).
  Mechanism does not port to sm70/CPU; the IDEA ports as tile-level
  send/compute interleave in the RPC layer.
- **Speculative pipeline filling**: PipeInfer (SC'24, arXiv 2407.11798)
  continuous async speculation + early cancellation, 2.15x, explicitly
  tolerant of low bandwidth; SpecPipe (arXiv 2504.04104) 4.19-5.53x TBT on
  an 8-stage 70B pipeline, pruning propagation keeps wasted work off slow
  links. Attacks "token time = sum of stage times" directly; our MTP is the
  draft-token source these papers need.
- **Negative checks**: ZeRO-style weight-gather over the network is
  bandwidth-prohibitive at batch 1 (81 GB over GbE ~ 650 s/token).
  DistriFusion-style stale-activation reuse has no AR-decode analog (the
  analog is speculative decoding - already in the fork).

## 4. Throughput scaling: disaggregation, KV streaming, pipelining

- **DistServe** (OSDI'24) / **Splitwise** (ISCA'24) / **Mooncake** (Kimi,
  FAST'25-era, arXiv 2407.00079) - PD disaggregation: 7.4x goodput / 2.35x
  at equal cost / +525% simulated long-context. All aggregate-throughput
  numbers on RDMA-class fabrics; Splitwise explicitly requires a "fast
  back-plane" for state transfer. KV-as-first-class-resource tiered
  HBM->DRAM->SSD is the mainstream direction (our SSD-streaming instincts,
  applied to KV).
- **PrfaaS** (arXiv 2604.15039) - PD split ACROSS DATACENTERS on commodity
  Ethernet: offload only long uncached prefills, bandwidth-aware routing.
  Works because hybrid attention shrinks KV; dense-attention KV keeps P and
  D coupled. The break-even model for "prefill on the fleet, decode on the
  V100s".
- **CacheGen** (SIGCOMM'24, arXiv 2310.07240) - KV encoder: 3.5-4.3x
  smaller, 3.2-3.7x lower fetch delay, bandwidth-adaptive, designed for
  WAN. The missing piece that makes PD split / request migration viable on
  GbE; our q8 KV formats are the substrate.
- **KVServe** (arXiv 2605.13734) - controller picks KV compression level
  from LIVE link measurements. Matches our measurement-driven culture.
- **Arrow** (arXiv 2505.11916) - dynamically rebalances prefill:decode
  instance ratio, 2.55x. Our roles are static today.
- **Dynamo / llm-d** - steal the KV-aware router (route to the member
  holding the prefix) and the scorer (prefix-hit + load + SLO); skip the
  platforms (K8s/Rust control planes are disproportionate for 5-7 boxes).
- **Sarathi-Serve** (OSDI'24) - chunked prefill + stall-free batching: the
  lightweight PD alternative on the coordinator; chunked prefill flowing
  through pipeline stages also pipelines prefill across nodes.
- **Llumnix** (OSDI'24) - live request migration with KV; on GbE only
  worth it for long decodes, and only after CacheGen-class compression.
- **Break-even calibrations (read before building)**:
  - "Where Edge-Cloud Speculative Decoding Actually Pays Off" (arXiv
    2606.25091): single-request benefit of cross-node SD is bounded by RTT;
    pipelined DSD beats colocated only when RTT < drafting window; the real
    win is multi-tenant capacity ((1+g*t_d/t_v)x more clients). Evaluate
    cross-node drafting by THROUGHPUT, not single-stream latency.
  - "Revisiting Disaggregated LLM Serving" (arXiv 2601.08833):
    disaggregation benefits are NOT guaranteed - they depend on load and
    the KV transfer medium, and cost more energy. On 100-Mbit links an
    uncompressed PD split LOSES to colocated.

## 5. Cross-field consensus

1. Placement is a GLOBAL optimization (max-flow/DP/submodular), never a
   greedy per-node heuristic; solvers beat greedy 2-3.6x, and link
   heterogeneity belongs in the objective.
2. Stage count is the latency currency; replica count is the throughput
   currency. Minimize cross-slow-link boundaries on the critical path;
   scale throughput with replicas and micro-batch overlap.
3. TP stays inside high-bandwidth domains; across slow links use PP, EP, or
   replication. Widening a synchronous group across a slow link is strictly
   harmful.
4. In the decode/memory-bound regime minimize DISTINCT EXPERTS (weight
   bytes) touched per token path, not tokens per device (METRO, MoETuner,
   GRACE - the same law three ways).
5. Expert prediction is a solved sub-problem (~90% next-layer from gate
   similarity); every serious offloading system prefetches rather than
   reacts, and re-places during compute-bound windows.
6. Bytes on the wire are cheap to cut (4-16x activation/KV compression,
   negligible quality loss) and underexploited at inference.
7. Everyone hides communication; nobody eliminates it. Speculation fills
   single-stream bubbles; micro-batching fills multi-stream ones.
8. Fault tolerance = fast re-route + incremental recovery (DHT expiry,
   Kuhn-Munkres re-placement, randomized re-pipelining), never full
   restart. We have surgical re-provision; the solver-based re-place is
   the upgrade.

## 6. The steal list, ranked for this fleet

1. **Cheap RDMA transport + CPU-proxy async transfer engine** (#60
   hardware). Prototype zero-cost with soft-RoCE (`rxe`) first, then used
   ConnectX-4/5 10/25 GbE for the boxes that matter (.11 + coordinator
   first). Expected: boundary RTT 0.15-3.4 ms -> ~5-30 us; EP's 43
   serialized RTTs/token collapse from ~13-146 ms to ~1 ms class - the
   single largest lever for BOTH EP and layer mode. Effort: $30-80/node +
   medium transport work in the RPC layer (UCCL-EP pattern).
2. **Speculative pipeline filling** (SpecPipe/PipeInfer x our MTP). Push
   draft tokens through the layer pipeline so stages stop idling; early
   cancellation keeps waste off slow links. Expected: single-stream layer
   mode 4.6-4.8 -> plausibly 6-10 t/s (their 2-5x TBT, discounted for
   GbE); the ONLY technique found that attacks single-stream latency
   without new hardware. Stacks with #71. Effort: medium-high (draft
   propagation + rollback over RPC).
3. **Ping-pong / dual-batch micro-batching in EP** (MegaScale-Infer, vLLM
   DBO). Overlap one stream's boundary RTTs with another's compute; MTP
   verify batches can synthesize the second stream. Expected: EP aggregate
   scales with stream count instead of pinning at 2.5-2.8 t/s. Effort:
   medium-high (double-buffered RPC + micro-batch scheduler).
4. **Placement solver v2 ("auto-place")** (#70 tail). Parallax two-phase:
   DP allocation with the latency-dominant heuristic (min stages per
   replica, max replicas) + water-fill by measured score; then per-request
   min-latency chain over the live latency/RTT table we already publish.
   Helix shows links belong in the objective; our fleet is small enough
   for near-exhaustive search. Expected: adding a box stops slowing anyone
   down; multi-stream scales with replicas. Effort: medium.
5. **Expert placement v2 on the #74 histograms** (extends #75). MoETuner
   cross-layer affinity chains + METRO min-members-per-token routing +
   ViBE speed-proportional assignment + Edge-MoE-Prism adaptive re-place
   (C(new)+T_mig < C(old), during prefill windows per Director). Expected:
   fewer boundary hops/token, shorter straggler tail; the online form of
   the GO'd static placement. Effort: low-medium (offline solver over
   existing histograms).
6. **Gate-similarity expert prefetch** (ProMoE - already has llama.cpp
   hooks - + HybriMoE impact metric). Predict next-layer experts (~90%
   hit), prefetch from disk/peer during current-layer compute. Expected:
   removes expert-miss stalls from the SSD/peer streaming critical path; a
   miss costs the status quo. Effort: low. (This is the current-token
   variant of the router-lookahead lever #55 left on the table.)
7. **On-the-wire activation quantization** (Apple/CacheGen/distributed-
   llama q80). q8 default, adaptive q4 on 100-Mbit segments. Expected:
   2-4x less serialization delay on the slow links; near-no-op on GbE so
   gate per measured link. Effort: low (kernels exist; verify accuracy per
   model).
8. **KV-compressed handoff + selective PD offload** (CacheGen/KVServe/
   PrfaaS). Quantize+delta KV (reuse q8 formats), probe per-link bandwidth,
   offload only long uncached prefills, V100s stay decode-dedicated.
   Expected: prefill parallelism across the fleet becomes viable on GbE;
   enables migration later. Effort: low-medium. Read the two break-even
   papers first - this is a throughput-under-load play.
9. **Request-level replication + class routing** (DeServe, PolyServe,
   Melange). Where a model fits on 1-2 boxes, run replicas and route whole
   sessions by class (interactive short -> fast box; batch/long -> fleet).
   Expected: near-linear multi-stream throughput per added box, zero
   network on the decode path. Effort: low-medium (routing/affinity layer
   on the existing fleet server).
10. **Hetis-style attention-head offload for weak-GPU workers**. Give the
    6 GB card / Arc iGPU a few attention heads' KV instead of a pipeline
    stage. Expected: the weak GPUs stop being zero-share passengers.
    Effort: medium (new split-policy case in the meta backend).

Dependencies: 1 makes 2-5 cheaper; 4-5 share the histogram/solver infra;
6 and the #55 prefetcher share the streaming path; 8 precedes any PD or
migration work. Not listed: things already covered by iteration 2
(spec-over-boundary, EAGLE-3 head, hot-expert static placement) - those
stand as ranked there.

## 7. What NOT to build (negative checks from the field)

- ZeRO-style weight-gather-per-layer over the network - bandwidth-
  prohibitive at decode batch sizes; our SSD streaming is the right form.
- Stale-activation async (DistriFusion) - no AR-decode analog.
- TP widened across 100-Mbit links (HexGen's straggler law) - pipeline or
  replica boundaries belong there.
- DeepEP/NCCL-GIN-style GPU-initiated RDMA - SM90+ only; UCCL-EP's
  CPU-proxy is the portable form.
- Full PD disaggregation without KV compression - loses to colocated on
  slow links (Revisiting, PrfaaS's caveat).
- Token-balancing EPLB for decode - METRO proves it backfires in the
  memory-bound regime; balance distinct experts, not tokens.
- Kubernetes-class control planes (Dynamo/llm-d wholesale) - steal the
  router/scorer ideas only.
- KTransformers' whole-graph CUDA-graph trick - cc<8.0; our decode graph
  cache is the Volta answer.

## 8. Watch list additions (merge-cycle checks)

- UCCL-EP / UCCL releases (portable EP transport on cheap NICs) - when it
  matures, re-evaluate before writing our own verbs path.
- SpecPipe/PipeInfer code releases - none found yet; re-check.
- vLLM wide-EP DBO and EPLB live-rebalance implementation details.
- Edge-MoE "Prism" (2508.12851) artifacts - adaptive placement reference.
- distributed-llama releases - the TP-over-TCP efficiency curve on
  homogeneous nodes is a datapoint worth tracking for the 2-V100 + future
  4-GPU expansion.
- llama.cpp upstream #24675 (RPC async/events) and #25818 (remote spec)
  - already tracked from iteration 2; unchanged.
