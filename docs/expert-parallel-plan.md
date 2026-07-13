# Expert-Parallel Distributed Workers — Design & Feasibility (TASKS.md #28)

Phase 0 deliverable (2026-07-11): design, measured-fact projections, and gated
increments for distributing MoE inference with the **expert** as the unit, across
the real 4-worker fleet (see `distributed-inference-guide.md` §3b for the fleet).

## 1. Why the expert is the right unit — the composition law

Task #31 measured the fundamental limit of layer pipelines: per-token time is the
**SUM of stage times plus hops**, so every box — including the slowest — is on the
critical path of every token (35B across 5 machines: 1.9 t/s vs 108 on the V100s
alone). Expert parallelism changes the composition law:

```
pipeline:         t_token = Σ_stages t_stage           (slow box taxes EVERY token)
expert-parallel:  t_token = t_attn + Σ_moe_layers max_{contacted workers} t_expert
                            (a worker is contacted only if the router picked
                             one of ITS experts for this token)
```

Consequences (the "smallest computable chunk" principle):
- A slow box owning **few, cold** (rarely-routed) experts contributes capacity
  while almost never gating a token. Its cost when contacted is bounded by ONE
  expert FFN + one activation round trip — not a model slab.
- The tunables are *how many* and *which* experts each box owns. Placement is a
  continuous dial between "contributes nothing, costs nothing" and "full member."
- What matters for a slow box is its **latency and link**, not its FLOPS: the
  per-contact payload is tiny (activations), the compute is one small FFN.

## 2. What crosses the network (sizes from the real models)

Per MoE layer, per token, per contacted worker:
- coordinator → worker: the token's hidden state, `n_embd × 2-4 B` ≈ **8-16 KB**
  (Qwen3.6-35B n_embd=4096 f16 ⇒ 8 KB; batch B tokens ⇒ B× that, shared per worker)
- worker → coordinator: `(experts owned ∩ routed) × n_embd × 4 B` ≈ **16-48 KB**
- merge on the coordinator: weighted add of k vectors — trivial.

Per token totals (Qwen3.6-35B-A3B: 40-ish MoE layers, k=8; DeepSeek-V4-Flash:
43 layers, k=6+1): ~0.7–2 MB/token spread over all contacted workers — GbE does
~120 MB/s ⇒ **wire time ~6–16 ms/token**, and the *latency* floor is
`layers × RTT` ≈ 43 × 0.3 ms ≈ **13 ms/token** on this LAN (sequential layers).
Compare the targets below: this tax is decisive for fast models, minor for the
big-model capacity play.

## 3. The prize (projections from measured numbers)

Reference vehicle: **DeepSeek-V4-Flash 81 GB** (77.9 GB experts, 8.8 GB
non-expert; measured layout in `ssd-streaming-plan.md` §8). It fits NOWHERE in
the fleet today except SSD-streamed on one V100 box at **2.7-3.1 t/s** (73% RAM
hit @30 GB budget).

Expert-parallel across the fleet's pooled RAM (64+64+32+16 = **176 GB — the
whole expert set becomes RAM-resident**):
- per-token expert bytes touched: ~1.83 GB (cold) spread across owners in
  parallel: `t_compute ≈ max_i(bytes_i / bw_i)`; balanced by bandwidth
  (~34+21+60 GB/s CPU boxes + V100 coordinator share) ⇒ **~15-30 ms**
- + activation wire/latency ≈ 15-25 ms ⇒ **~25-55 ms/token ⇒ 18-40 t/s**
  vs 2.7 t/s today — a **7-15× runnability win** on hardware already owned.
That is the case that justifies the build. For models that fit one fast box
(35B on V100s), EP is still a loss — same conclusion as #31.

## 4. Design

Coordinator (V100 box) runs everything except routed-expert FFNs: attention,
norms, router, shared experts, KV. Workers own **expert subsets** per layer and
serve `expert_ffn(x, expert_id)` calls.

Key insight for the implementation route: **the meta tensor-split backend
already implements intra-layer expert segmentation** (`ffn_*_exps` split
AXIS_1-segmented across devices, `mul_mat_id` PARTIAL + delayed AllReduce —
validated on Qwen A3B and GLM in tensor mode). The missing piece is running the
meta device over **RPC member devices** instead of local CUDA devices. The RPC
backend now has async + events (proto 4.2) and per-endpoint containment (29b).
The per-layer "AllReduce" for a segmented expert sum is just the sum of each
member's partial expert-output vector — **k × n_embd × 4 B per layer**, NOT a
row-split GEMM reduce; Ethernet handles that fine (this is why #17/#31's
"Ethernet AllReduce is dead" verdict does not apply: that verdict was for
per-GEMM row splits).

Placement policy (the user-guidance dial, informed by #31):
1. **Static bandwidth-weighted** (increment 1): expert count per worker ∝
   measured per-box memory bandwidth; attention/shared experts mirrored on the
   coordinator (never cross the wire).
2. **Hot/cold by routing stats** (increment 2): the ssd-streaming work already
   proved routing skew is measurable online (LRU hit rates by expert). Hot
   experts → coordinator VRAM/fast boxes; the cold tail → slow boxes (the i5
   holds experts it serves a few times a minute — capacity, ~zero gating).
3. **Dynamic migration** (increment 3, stretch): move experts when observed
   contact latency × frequency says so; the beacon (29d) carries free_mib and
   has room for a bandwidth score.

## 5. Gated increments

- **0. This document** — design + projections from measured facts. ✅
- **1. Meta-over-RPC spike**: build a coordinator-side meta device whose
  members are RPC devices (CPU workers); run a MoE model with experts
  segmented across 2 workers, attention local. Gates: coherent temp-0 output;
  PPL == single-box reference; per-token wire bytes within 2× of §2 estimates.
  Feasibility probe (2026-07-11): **GO** — `ggml-backend-meta.cpp:2416` has a
  generic `allreduce_fallback` for members without the backend-specific comm
  proc (only CUDA provides `ggml_backend_comm_allreduce_tensor`), so RPC/CPU
  members reduce through it (coordinator-bridged — correct, one extra hop;
  optimize later via W2W). Remaining risks: fallback perf, graph-build cost
  per layer, meta shadow bookkeeping over RPC buffer semantics (the TP-island
  work solved the inverse direction and is the reference).

  **DONE (2026-07-11).** `-sm tensor` + `--rpc a,b --device RPCx,RPCy` builds
  `Meta(RPC,RPC)` with no new subsystems; vehicle = GLM-4.7-Flash (MLA policy
  already mirrors attention and splits only `ffn_*_exps` = exactly the EP
  shape; the fallback reduce runs W2W via COPY_FROM_REMOTE, not bridged).
  Bugs found and fixed on the way (all in the working tree):
  1. *Meta static shadow-arena retirement* (`ggml-backend-meta.cpp`): a KV ctx
     that exactly fills its arena (pure-attention models: 2*(1+n_stream)*n_layer
     tensors) retired the static arena mid-creation; the per-member alloc walk
     only sees the current arena -> silent NULL -> assert. Fixed: retirement
     headroom + views-only contexts take the dummy-buffer branch. Hybrids
     (27B/35B) undershoot the arena, which is why prod never hit it.
  2. *O(n^2) graph serialization* (`ggml-rpc.cpp`): `serialize_graph` walked
     src closures past subgraph boundaries, so each of the meta backend's
     ~2/layer subgraph messages carried the whole model-graph closure:
     measured **53 MB/token** on GLM (GGML_RPC_STATS). Fixed: leaf-cut
     serialization (out-of-graph tensors serialize as data-only leaves; safe
     for any server version).
  3. *Graph re-serialization every token*: the single `last_graph_uid` reuse
     slot never matches when ~96 subgraphs cycle per token. Fixed: proto 4.3
     `GRAPH_COMPUTE_UID`/`GRAPH_RECOMPUTE_UID` + per-(device,uid) LRU cache
     (server cap 512, client tracks half) — steady-state decode sends 16 B
     per subgraph instead of the graph.
  4. *Per-row weight upload*: the meta slicer's `set_tensor_2d` fell back to
     one RPC SET_TENSOR per row — 12.0M calls / +4.3 GB descriptor overhead
     per GLM load, and it defeated SET_TENSOR_HASH/--model-dir. Fixed: RPC
     buffer `set_tensor_2d` gathers rows client-side into 32 MB chunks
     (2064 calls; hash path engages again on whole/mirrored tensors).

  **Gate results** (2 CPU workers .11 + .15, GbE; coordinator dev box):
  - *Coherence*: PASS — 0.6B dense loopback and GLM fleet runs produce fluent
    on-topic temp-0 output.
  - *PPL* (wiki.test, 8 chunks): meta 8.1284 +/- 0.51 vs single-box 8.0294
    +/- 0.49 (BOTH .11-only and .15-only give the identical 8.0294 — single-box
    is hardware-independent). Attribution: LLAMA_META_DUP_DEVICE=2 against ONE
    worker (same 2-way split arithmetic, no cross-worker wire) reproduces
    **8.1284 exactly** — the delta is inherent to the 2-way split math
    (segmented expert sums reorder additions -> near-tied router picks flip),
    and the cross-worker transfer adds zero numerical error. PASS.
  - *Wire bytes/token* (RPC_STATS, -n 1 vs -n 129 subtraction): mixed-proto
    fleet (workers still 4.2 -> no uid cache): 2.9 MB/token coordinator tx,
    2.75 MB of it legacy graph serialization. With proto 4.3 on both ends
    (validated on loopback): ~140 KB/token tx + ~616 KB logits rx + ~770 KB
    W2W reduce payloads = **~1.5 MB/token — inside the §2 band**, PASS.
  - *Perf trail* (not a gate): GLM decode 0.7 -> 4.2 t/s from leaf-cut alone
    (mixed fleet); loopback 0.6B 11.6 -> 18.1 t/s. Latency floor (§2) is now
    the dominant cost: ~95 sequential subgraph+reduce boundaries/token.
  - Fleet worker image with proto 4.3 pushed as
    `10.5.5.1:5000/llamacpp-cpu:rpc-worker-uidcache`; workers need a manual
    `docker compose up -d` with that WORKER_IMAGE (not yet applied).
- **2. DeepSeek-81GB across the fleet**: the prize run. Gates: loads with
  experts RAM-resident across ≥3 workers (--model-dir makes this cheap);
  decode ≥ 3× the 2.7 t/s SSD baseline; coherent output; PPL sane.

  **GATE RESULT (2026-07-11 evening): capacity PROVEN, perf gate NOT met.**
  With clean caches (see the cache-poisoning saga below) the 86.7 GB model
  runs **coherently** RAM-resident across .11+.15+.25 (`-ts 4,3,1.5`):
  fluent on-topic CPU-pipeline explanation, uid-cache healthy (13.4k
  recomputes vs 564 full sends). Decode **0.4 t/s** vs the 8.1 gate (and
  below the 2.7 SSD baseline). Clean 2-worker baseline (.11+.15, GbE only,
  `-ts 3,2`): **1.0 t/s coherent** - META_TIMING ~175 ms compute +
  ~40 ms reduce per graph piece, ~5 pieces/token, 10.8 boundaries/piece.
  Attribution, all actionable:
  1. **.25's 100-Mbit link sits in EVERY reduce boundary** - 3 members means
     2 butterfly steps/boundary and .25 carries transfers in both; #31's
     bandwidth-proportional law applies to LINKS, not just RAM: a
     100-Mbit box must hold a tiny/cold share (true expert-parallel
     placement, increment 3) or stay out of the hot path.
  2. Sequential fenced W2W pulls (~3.4 ms/boundary at 2 members, worse at
     3) - parallelize the pulls in allreduce_fallback.
     **DONE (2026-07-12, proto 4.4):** the pulls of one reduce step now go out
     as a batch - all fences captured BEFORE any request is sent (an early
     fence is safe: it already covers the compute that produced the src, and
     excluding the sibling pulls is exactly what removes the serialization),
     then all COPY_FROM_REMOTE requests, then the acks. Needed a server-side
     fix: the 4.3 server held its per-process execution lock across the whole
     pull, so two crossing in-flight pulls ABBA-deadlocked ACROSS processes
     (each server's fenced-read serving thread needed the lock its own
     COPY_FROM_REMOTE held while waiting on the other worker; found the hard
     way - the first batched run wedged the loopback pair for its full
     timeout). 4.4 servers run the remote fetch without the lock and take it
     only for the local write, the same treatment GET_TENSOR_FENCED already
     had. The client batches only when BOTH ends of every pull are >= 4.4;
     mixed fleets silently keep the old sequential path (validated against a
     4.3 server: no deadlock, no crash; an old dst pulling from a NEW src
     refuses HELLO and degrades to the bridged copy until the fleet is
     rolled). Gates: 0.6B loopback coherent 18.4 t/s (was 18.1); truncated-V4
     meta run token-identical to single-CPU.
     **Fleet A/B (2026-07-13, fleet on 4.4 images, V4 2-worker .11+.15
     `-ts 3,2`): reduce 40 -> 25.5-28.5 ms/graph piece (-35%, ~2.4 ms/boundary
     at 10.8 boundaries/piece). Decode unchanged (timed 0.98 vs 1.0; untimed
     1.20 vs 1.2-1.3): compute at 155-223 ms/piece is ~85% of token time, so
     the reduce cut is masked - the multi-build subgraph cache (next lever) is
     now decisively the bottleneck.** Note for reruns: on .15 pick its CPU
     device explicitly (`--device RPC0,RPC2` - RPC1 is the 6 GB 1660 Ti and
     `-ts` shares against it fail to alloc); every load re-verifies the
     worker tensor caches (~9-17 min for .15's 36 GB share - the price of
     verify-on-read).
     3-worker rerun (+.25, `-ts 4,3,1.5`): 0.35 t/s (was 0.4), reduce
     144 ms/piece (13.4 ms/boundary vs 2.4 on 2-worker) and compute
     562 ms/piece - .25's 100-Mbit link and slow CPU still gate everything,
     confirming increment 3 (tiny/cold share for slow-link boxes) over any
     further transport tuning.
     **Perf lever 2 landed (2026-07-13, c20e432b4): multi-build subgraph
     cache.** Builds keyed by a content hash of the outer graph (sched pieces
     get a FRESH uid per split rebuild, so uid equality never matched and
     every call redid shadow re-registration + split walk + subgraph
     construction, ~30 ms/piece); shadow-ring rotation is the validity
     horizon, and the galloc-facing init_tensor memoizes unchanged
     re-registrations (otherwise per-token allocations grow the shadow arenas
     forever once hits stop ring rotation). Fleet 2-worker V4: 0.98 -> 1.16
     t/s timed / 1.20 -> 1.25 untimed, 128/128 hits steady-state; loopback
     token-identical to the previous commit on 0.6B and V4-trunc. NOTE the
     rebuild never showed in META_TIMING's compute bucket (tm_last starts
     after the build block) - it was invisible inter-graph overhead, and in
     untimed runs partially overlapped with async worker compute, which is
     why the end-to-end gain is modest. The ~800 ms/token that remains is
     per-boundary worker compute (mirrored attention runs FULLY on every
     member + expert slices), so the next levers are increment 3 placement
     and reduce/compute overlap, not client-side bookkeeping.
  3. Per-token client-side subgraph rebuild (uid==0 outer graphs) -
     multi-build cache.
     **Perf lever 3 landed (2026-07-13): star reduce (dispatch pipelining).**
     On a topology with a local root (dedicated-attention Meta(CUDA0,RPC,RPC)),
     a reduce boundary was a chain of sequential blocking steps: fold GET,
     CUDA ADD, butterfly GET+SET pair, more ADDs, copy-out - each its own
     round trip or device sync, ~2 ms/boundary x ~21.5 boundaries/graph.
     Now: one batched read of all computing partials (new reg proc
     `ggml_backend_get_tensor_batch` sends every GET_TENSOR request before
     reading any response - the transfers and the workers' compute tails
     overlap, and in-order serving is the fence), host f32 sum in the same
     order the fold+butterfly used (fp add commutativity keeps it
     bit-identical), then async SET broadcast of identical bytes to every
     member. No protocol change; workers untouched; all-wire fleets keep the
     butterfly; `GGML_META_NO_STAR=1` for A/B. Also fixed: batch procs were
     looked up from member 0's reg only, so Meta(CUDA0,...) never found them.
     Gates: loopback 12L V4 truncation token-identical star vs butterfly;
     full-V4 fleet outputs byte-identical. Fleet decode 2.08/2.24 t/s vs
     1.30/1.69 same-binary controls (prior best 1.82); prefill ~8.3 vs 7.55.
  Also hit and fixed on the way: a fresh cache-miss load writes the
  worker's full share to the cache dir and the DIRTY page cache stacks on
  top of the weights - OOM-killed .15 (36 GB share, 64 GB box) at end of
  load; fsync+fadvise(DONTNEED) per entry now caps the dirty set, and a
  member alloc failure fails the load cleanly instead of aborting the
  coordinator (c18e3dbbc).

  **The debugging story that got here (for the record):**

  *The model is 86.7 GB on disk (not the 81 GB quoted above).* It loads
  RAM-resident: 3 workers (.11+.15+.25, `-ts 4,3,1.5`) in ~47 min — gated by
  .25's 100-Mbit link (TCP backpressure on .25's ~22 GB share serializes the
  whole per-tensor upload loop); 2 workers (.11+.15 GbE, `-ts 3,2`) in ~13 min.
  Slice-aware local sourcing is the fix (a worker can only hash-match WHOLE
  tensors today; the split-state push used by TP islands is the template).
  A worker indexes /local-models at STARTUP - restart it after staging a file.

  *deepseek4 is DSA, not plain MLA — and that difference drove every failure:*
  - GLM-4.7-Flash (deepseek2 = actual MLA) is fully coherent under this
    machinery, so MLA per se is fine. DeepSeek-V4-Flash (deepseek4 = DSA:
    MLA latents + per-token top-k indexer, custom `llama_kv_cache_dsv4`,
    SWA semantics) does NOT report `is_mla()`, so the split policy's
    mirror-attention branch never engaged.
  - `attn_sinks.weight` then took the head-split path whose axis reference
    (`attn_output.weight`) does not even exist in this arch -> load abort.
    FIXED: DSA archs (DEEPSEEK4/DEEPSEEK32/GLM_DSA) mirror attention
    explicitly, same rationale as MLA (llama-model.cpp `mirror_attn`).
  - The deepseek4 graph is 33.6k nodes and the scheduler cuts it into ~5
    pieces per token (the indexer bounces through the coordinator; GLM = 2).
    Two meta-backend assumptions broke:
    (a) a scheduler piece can END on a view of a coordinator-side host tensor
        -> the subgraph walk never closed the trailing subgraph
        (`i_start == n_nodes` assert). FIXED: close it, no split state.
    (b) the meta backend cached ONE subgraph build keyed by the outer graph
        uid; with multiple meta pieces cycling per token it rebuilt every
        call with FRESH uids -> the proto-4.3 graph cache never hit -> a
        full re-serialization per subgraph per token (measured 24 MB/token,
        ~200 ms/graph). FIXED: subgraph uids are now a CONTENT HASH (ops,
        shapes, op_params, flags, data + src addresses) - identical rebuilds
        keep their uid, so RECOMPUTE_UID hits across rebuilds. Measured after:
        14758 recomputes vs 564 full sends, compute 200 -> ~124 ms/graph,
        decode 0.8 -> 1.2 t/s (2 workers).
  - *SOLVED: the "word-salad/DDDD" corruption was NEVER a split bug — it was
    the rpc-server tensor cache (`-c`) serving unverified poisoned entries.*
    The hunt, for the record (each step ~3 min once the model was truncatable
    with `LLAMA_TRUNC_ARR=1` + `--override-kv deepseek4.block_count=int:6`):
    EP_ONLY split: still garbage -> NO_DELAY (new `GGML_META_NO_DELAY=1`,
    reduce at every PARTIAL): still garbage -> GGML_RPC_NO_W2W: still garbage
    -> .11 SINGLE-DEVICE (no meta at all): STILL garbage, split exonerated ->
    same single-device through the same worker image locally: CLEAN ->
    .15 single-device: garbage; fresh Q2_K 0.6B on .11: first run plausible,
    every later run catastrophic (PPL = n_vocab = uniform logits) -> the
    common factor: fleet workers run `-c`; repeated/killed uploads during the
    day's crashed loads left truncated cache entries, and `set_tensor_hash`
    served whatever bytes were in the file - no size or content check (the
    request does not even carry a size; upstream's cache has the same hole).
    GLM survived because its cache entries were written by clean runs.
    **Fix (verified end-to-end): cache writes go through tmp+rename (a partial
    write can never land under the final name), and reads re-hash the entry
    before serving - mismatch deletes the file and falls back to streaming,
    so poisoned fleet caches self-heal on the next load.** Validated locally:
    truncating a 127 MB entry to 1 KB -> "fails verification" log, identical
    output, entry rewritten to full size on the same load.
    Side finding: V4's split math is CLEAN - loopback dup-device (same split,
    same RPC serialization) is token-identical to single-CPU; .11 and .15
    agree with each other on fresh streams (cross-box variant rounding is a
    legit small delta, GLM PPL was exactly equal either way).
    Debug enablers added along the way: `LLAMA_TRUNC_ARR=1` (accept longer
    per-layer arrays + partial tensor load, so `--override-kv
    <arch>.block_count` can truncate any model into a minutes-scale
    reproducer) and `GGML_META_NO_DELAY=1`.
  - *Perf reality check vs §2:* META_TIMING shows ~3.4 ms per reduce boundary
    (~54 boundaries/token = ~185 ms/token of reduce alone) vs the 0.3 ms RTT
    the §3 projection assumed - the gap is per-boundary orchestration: the
    allreduce fallback's fenced W2W pulls are SEQUENTIAL synchronous RPCs.
    Next levers, in expected order of payoff: parallelize the fenced pulls in
    `allreduce_fallback` (halves boundary latency) - **landed 2026-07-12 as
    proto 4.4 batched pulls, see attribution item 2 above** - multi-build
    subgraph cache (kills the per-token client rebuild of 33.6k shadows),
    EP batching (inc 4).
    Even at zero overhead the 2-worker compute floor is ~4-5 t/s; the 8.1 t/s
    gate likely needs 3 workers + the reduce-latency lever.
- **3. Placement policy**: bandwidth-weighted then hot/cold (routing-stat
  export from the router — the ssd-stream debug counters are the template).
  Gate: measurable tail-latency reduction vs uniform placement.

  **Attribution result that reshapes this increment (2026-07-13).** Truncated
  6-layer V4 on the fleet, three configs (single .11 = A+E; 2-worker
  `-ts 1,1` = A+E/2+OH; `-ts 3,1` = A+0.75E+OH) solve the split:
  E(experts) ~40 ms, A(mirrored attention/dense/indexer) ~47 ms,
  OH(meta+RPC) ~63 ms per token. Scaled to the full 61-layer model this is
  **attention ~473 ms (59%) / experts ~246 ms (31%) / overhead ~79 ms (10%)**
  of the measured 798 ms/token - and mirrored attention is a CONSTANT: it
  runs in full on every member (gated by the slowest), so no expert
  re-weighting can push the CPU fleet past ~2.1 t/s, below the 3.54 t/s
  single-box bar (2x V100 + -ncmoe). Conclusion: increment 3 must implement
  the ORIGINAL design sketch - attention/router/dense LOCAL on the
  coordinator's V100s (the ~7 GB non-expert stack fits one card), only
  expert slices on the workers - i.e. a Meta(CUDA0,RPC,RPC) whose placement
  DEDICATES attention to member 0 instead of mirroring it (the zero-share
  slice machinery exists; the known `-ts 0` segfault, TASKS #30 note, is on
  this path). Projection: GPU attention ~30-80 ms + expert share ~120-205 ms
  + overhead ~80 ms = ~2.6-3.7 t/s at 2-4 workers, crossing the single-box
  bar with placement + batching still to come.
- **4. Batching/multi-stream**: EP naturally batches per-worker (all tokens'
  hits on a worker's experts in one call) — revisit the #31 -np finding here;
  EP is the path where MoE batch throughput CAN scale.

  **MEASURED 2026-07-13 - confirmed, no code needed (a property of the
  design).** llama-batched-bench, full V4, dedicated-attention fleet config
  (npp 64, ntg 32): TG aggregate 1.66 / 2.71 / 3.82 / 4.73 t/s at
  B=1/2/4/8 - x2.85 at B=8 where the #31 pipeline measured +27%. The
  boundary cost (reduce + turnaround, the fleet's dominant tax) is paid
  once per decode STEP regardless of B, so batch tokens ride the same
  boundaries; expert compute grows sublinearly (per-worker batched
  mul_mat_id). Aggregate crosses the 3.54 single-box single-stream bar at
  B=4. Coherence: llama-parallel -np 4, all four concurrent temp-0
  responses fluent/on-topic. Caveat kept honest: the same bench on the
  single box (2x V100 -ncmoe 99) scales x2.17 (3.85/5.37/7.07/8.35) and
  stays ahead in absolute V4 throughput - V4 half-fits in coordinator RAM
  with NVMe behind it. EP's absolute win is the capacity regime (no box
  holds the experts); its batching win is the steeper scaling slope.
  Slow-link async participation (batch-only .25) stays parked - the
  per-boundary latency law keeps it out of the sync ring, and V4 does not
  need its capacity.

## 6. Open questions (tracked, not blocking increment 1)

- Router on coordinator needs each token's router logits before expert calls —
  already local (router weights are non-expert, mirrored).
- KV/prefill: prefill ships B×8 KB per layer per worker — fine (batch-amortized).
- Failure: a dead expert-owner mid-token → 29b containment fails the request;
  re-placement of its experts = restart with rediscovery (29b loop) — experts
  reload from --model-dir locally.
- Quantized expert compute on old CPUs (i5-3210M lacks AVX2 VNNI etc.): its
  per-expert FFN cost is the "contacted-worker max" — measure in increment 1.
