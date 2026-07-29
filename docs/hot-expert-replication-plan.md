# Hot-expert replication / gather leg-skip (TASKS #71, member-compute axis)

Status: design v1, 2026-07-29. Follows the #75 double-null and deferral v3
green (fill-the-bubble-plan 2.2b/c). Companion: docs/expert-placement-plan.md
(the machinery this reuses), docs/expert-profiling.md (the hotness source).

## 1. Why this and not more placement

Measured constraints the design must respect:

- **Member per-layer time is insensitive to routed byte mix.** Placement
  nulled twice on the record roster: gate 5 (-4%, latency-bound regime) and
  the v3 re-test (4.68 vs 5.04 t/s, and the DECISIVE tell: ready rate
  74.9-75.4% placed vs 76.0% control - members did not get faster per layer
  when their cold-expert bytes shrank). Whatever a wire member pays per
  boundary is dominated by per-leg fixed costs (thread wake, gather/encode
  of the full boundary tensor, socket service), not expert-row reads.
- **A zero-routed member still pays the full leg today.** The B2 contributor
  set is graph-build-static: `ggml/src/ggml-backend-meta.cpp:4102-4110`
  collects every member whose boundary shadow has `GGML_TENSOR_FLAG_COMPUTE`
  (set once at 1918-1936), and the star gather (4145-4348) pulls
  `ggml_nbytes(boundary)` from each - under placement, a member none of
  whose experts were routed this token ships an exactly-zero tensor over
  the wire and the owner waits for it (or, under v3, defers it as late
  mass). Nothing consults per-token routing.
- **The boundary wait IS the member gap** (WAIT_US=3000 leg: at defer rate
  5.9%, t/s == control). Shrinking what a member computes does not pay;
  removing the member from the boundary entirely does - it drops out of
  the max() the owner waits on, and under v3 it stops contributing late
  mass.

Conclusion: the lever is **contributor count (legs), not bytes**. "Hot-expert
replication" decomposes into two separable mechanisms:

1. **Dynamic leg skip** - per-token, drop zero-routed members from the
   gather. Exact by construction (their contribution is identically zero).
2. **Hot-mass concentration** - a layout in which the owner covers most of
   the routed mass, so (1) fires often.

(2) without (1) is placement - measured null. (1) without (2) is small
(see table). Together they are the mechanism the ledgers call replication.

## 2. The numbers (profiles/hy3-full.json, 5502 tokens, 79 MoE layers, k=8 of 192)

Routing skew (mean over MoE layers, fraction of routed slots):

| hot set | top-16 | top-32 | top-48 | top-64 |
|---|---|---|---|---|
| coverage | 0.712 | 0.855 | 0.900 | 0.922 |

Expected gather legs per decode boundary (multinomial model, record shares
21,21,46,50,27, wire members = 46/50/27; today's actual value is 3.00 - every
member contributes every boundary):

| layout | mean legs /3 | P(boundary fully local) |
|---|---|---|
| contiguous split + leg skip | 2.54 | 0.001 |
| **#75 hot-first artifact + leg skip** | **0.60** | **0.537** |
| eviction variants H=16/32 | 1.23 / 0.73 | - / 0.485 |

The load-bearing row is the second: **the existing, already-gated #75
artifact** (placements/hy3-record-21-21-46-50-27.json, hottest-first perm,
owner = top-48/layer, Cov@25.5% = 0.913 per expert-profiling.md) **already
concentrates the hot mass. No new layout, no VRAM math, no permuted-upload
changes are needed - the entire build is the leg-skip primitive**, and it
turns the twice-null placement layout into: 54% of decode boundaries need
no wire wait at all, mean legs drop 3.00 -> ~0.60.

Model caveats (inc-0 measures ground truth): multinomial independence
(token-level routing correlation can push either way), profile-workload
drift (5502-token profile), and hy3-specificity - GLM-5.2 profiled
essentially uniform (expert-profiling.md), so this axis is worthless there.

## 3. Design

### 3.1 inc-0 - instrument (counters only, zero behavior change)

**BUILT 2026-07-29 (as-built differs from the sketch below):** gather-time
reads of the topk tensor return ring-recycled bytes (the member arena
reuses the slot intra-subgraph and ignores FLAG_OUTPUT), so the ids are
captured at COMPUTE time instead - an eval callback (`llama_zl_ids_cb`,
profiler pattern) notes each layer's decode ids via
`ggml_backend_meta_note_routed_ids`; the gather consults the last-noted
layer (dataflow: topk-L computes, then L's reduce, then topk-(L+1)).
Ownership host copy registered at table build via
`ggml_backend_meta_set_expert_ownership`. Counters + `META_ZL` dump line
under GGML_META_BOUNDARY_STATS; everything gated on GGML_META_ZL_STATS=1
(a MEASUREMENT tool: the callback forces sched splits, so instrumented
legs are batch-invariance-class, not baselines - same caveat family as
GGML_META_TIMING). Loopback (trunc stub, PLACE=1, hot-first artifact):
owner slot share 48-54% live (61% profile-predicted - profile-vs-live
drift is exactly what this instrument measures), zero-leg rates grade
monotonically with placement rank (m1 4-9%, m2 31-40%, m3 58-70%).

Original sketch (superseded): at the star gather, read the layer's routed
ids host-side and consult a host-side `member_of[e]` table (derivable at
table-build time, src/llama-expert-placement.cpp:257-301). New
BOUNDARY_STATS counters, decode-shaped boundaries only:

- `zl_boundaries[j]` - boundaries where member j had zero routed experts
- `zl_free` - boundaries where ALL wire members had zero
- owner routed-slot coverage (validates Cov@25.5% on live traffic)

Gate: counters match the section-2 model within reason on a PLACE=1 fleet
serve; all other output byte-identical (it is read-only).

### 3.2 inc-1 - dynamic leg skip (the build)

**FLEET A/B MEASURED 2026-07-30 - NULL; the section-4 stop rule FIRES
and the replication/leg-skip lane CLOSES on this roster.** Placed +
EXPERT_DEFER=1 + zero short replies (workers local/.11 on proto 4.14,
-t8/-t6): 4.61 t/s mean (4.07-4.88, 10 runs, coherent, LOST 0,
markers ACTIVE on both 4.14 connections) vs same-day default-EP v3
control 4.97-5.21. Defer rate 64.6-69.7% = UNCHANGED vs control 65-70%
- the loopback signature (rate 51.5% -> 14.1%) did NOT transfer. Reading:
a wire member's lateness is PIPELINE PHASE LAG - it executes its fused
chain pieces in order and reaches each fetch a boundary behind the
owner - so shrinking the reply payload (57KB -> 1B) changes nothing the
owner waits on; the wait was never byte-bound (consistent with q8==f16).
Per the stop rule: per-leg cost is not the binding term either; the
member-compute axis is exhausted on this fleet (placement null twice,
threads null, leg-skip null). Proto 4.14 STAYS (exact, free, engaged -
harmless default). Remaining levers: faster member lanes (V4 dual-role
CUDA1 expert share - user-suggested, filed), fewer boundaries (LP
pair-fusion, parked), #60-class transport. Load note: the first placed
load hit the #8 stale-manifest wedge (V4's 144GB LRU-evicted the placed
slices; 9/9-miss batches streamed slowly, then .15 refused the final
121KB table alloc after 28 min); the immediate RETRY loaded clean -
wedge cost ~30 min, filed under #72(a).

**inc-1 BUILT + LOOPBACK-GATED 2026-07-30 (proto 4.14 zero short reply,
as corrected below):** client sets fused flag bit 128 when the worker
speaks minor >= 14; the worker's FETCH branch scans the payload for
bitwise zero (u64 words, early-out) and answers with a 1-byte 0x5A
marker instead of the encoded payload; the response FIFO accepts
out_size==1 only for zero_ok entries and stashes logical_size zeros.
Per-connection: pre-4.14 workers keep full payloads. Gates (trunc stub,
3 loopback workers, fuse=2 + q8): place-exact = 6/6 byte-identical
c80261ff with markers ACTIVE x3; default-EP = 6/6 c80261ff, markers 0
(never fires - row-slices are not bitwise zero); place+EXPERT_DEFER=1 =
markers ACTIVE, LOST 0, and the stub defer rate FELL 51.5% -> 14.1% -
zero legs arrive fast enough to be consumed exactly instead of deferred.
Remaining: worker image rebuild + fleet A/B (PLACE=1 EXPERT_DEFER=1 vs
the default-EP v3 plateau 4.97-5.21).

**TIMING CORRECTION (2026-07-29, found while building inc-0):** the
coordinator-side skip described below cannot work as written - the fused
FETCH for boundary L is pre-issued inside boundary L-1's fused message,
BEFORE layer L's routing exists anywhere. The owner cannot decide "don't
request member m's L-contribution" in time. The correct mechanism is
WORKER-side: the worker already knows its own routed count at compute
time (every non-owned lane carries the mul_mat_id skip sentinel), so a
member whose lanes are ALL sentinel for a boundary answers the
pre-issued fetch with a 1-byte ZERO marker instead of the encoded
payload (proto rev, "zero-contribution short reply"). The owner treats
the marker as exact zeros: no payload encode on the member, ~57 KB -> 1 B
on the wire, near-zero arrival latency, and under v3 the marker is
consumed instantly (never deferred, no late mass). Graph topology,
manifests and the response FIFO ordering are all untouched - the reply
is just shorter. inc-0's counters (below) still price the opportunity
exactly; the coordinator-side eval-callback id capture they use
(GGML_META_ZL_STATS) stays a measurement-only tool.

At gather time (ggml-backend-meta.cpp:4145+), with placement tables loaded:
compute this boundary's zero-routed member set from the ids (host read is
tiny - 8 x i32 per layer; D2H once per MoE layer per token, ~1-2 ms/token
total against a 120-140 ms latency term) and drop those members from
`part[]` for THIS execution only:

- their fused pre-request FETCH is never issued (the response FIFO stays
  consistent - no request, no expected response);
- their contribution is not summed (it is identically zero: sentinel +
  mask guarantee, same invariant the #75 audits enforce);
- under v3, they are neither `ready` nor `defer` - they leave the
  denominator entirely (new counter `skipped_legs` alongside ed_*).

Crucially the graph-static structures are untouched: COMPUTE flags,
manifests, subgraph partitions, fused-chain shapes all stay as built -
the skip is a runtime subset decision inside the gather, so no
manifest/cache churn (#8/#72 class) and no per-member graph divergence
(which the meta backend structurally cannot do).

B1 is NOT skipped: members need ffn_inp to run their (structurally
identical) graphs; a zero-routed member simply computes ~nothing (all its
mul_mat_id lanes carry the -1 sentinel already) and its outputs are never
pulled. Wire savings ride along on B2 (`bs_gather_bytes` should drop
roughly with legs).

Applicability: **placement regime (AXIS_2) only.** Default EP splits every
expert across every member (AXIS_1/AXIS_0, src/llama-model.cpp:657-668),
so no member is ever zero-routed there. The track therefore adopts
PLACE=1 as its serve layout and must beat the DEFAULT-EP v3 plateau, not
the placed one - the placed layout's measured 0 to -7% is the entry fee
the leg-drop must overcome.

Gates, in order:
1. off = byte-identical (env `GGML_META_LEG_SKIP=0` default-off while
   experimental);
2. loopback PLACE=1, deferral OFF: leg skip ON vs OFF **byte-identical**
   (this is an exact transform - stronger than PPL; any drift = a real bug
   in the zero-contribution invariant, run the #75 pair audit);
3. fleet exact-mode A/B (PLACE=1, EXPERT_DEFER=0): t/s + coherence +
   byte-class stability; **measure prefill pp too** - the hot-first layout
   concentrates ~90% of routed mass on the owner in prefill as well, a
   cost the placed A/B never separated out (mitigation if bad: inc-3);
4. fleet v3 A/B (PLACE=1 EXPERT_DEFER=1 LEG_SKIP=1 vs today's default-EP
   v3 plateau): primary = plateau t/s + coherence reads; mechanism
   instruments = `skipped_legs` (expect ~2.4/boundary), defer rate over
   the REMAINING legs, and PPL (prefill legs are skipped exactly, so PPL
   stays a valid regression net for the exact path).

### 3.3 inc-2 - artifact refresh (cheap, optional)

scripts/expert-placement.py already emits hottest-first; regenerate from a
fresher/larger profile if inc-0 shows drift (re-profile is a serve flag,
LLAMA_EXPERT_PROFILE). Eviction-style variants (owner keeps top-H hot +
its own hottest fill) are strictly worse than the shipped artifact per the
section-2 table - only revisit if VRAM shrinks.

### 3.4 inc-3 - true replication (phase masks; only if prefill pays the toll)

If gate 3.2(3) shows the placed layout materially hurts prefill: keep
member copies of the hot experts resident (schema v2: per-member expert-id
lists, counts may sum > n_expert, perm no longer a bijection) and ship TWO
mask/remap table pairs - decode masks prefer the owner replica, prefill
masks prefer the member copy - selected by the same ne[1]==1 gate v3 uses.
Masks stay constant tensors (no in-graph dynamism); every audit becomes
per-phase ("masks sum to ones per phase", extending
llama-expert-placement.cpp:161-205); the -ts consistency warning
(src/llama.cpp:436-458) needs a replication-aware denominator; auto-weight
is blind to replica bytes (same known gap as #70 dual-role - explicit -ts
is the workaround). This is the only increment that adds VRAM cost
(top-48 replicas ~ up to ~30 GiB if fully doubled - so in practice
replicate a SUBSET, top-16 ~ 10 GiB, coverage 0.71).

### 3.5 Sequencing vs the worker thread sweep

The thread/affinity sweep stays first (agreed #71 order): it needs no
build, and its result prices this track - if member layer time turns out
thread-elastic, the ready rate rises for free and the residual-leg wait
shrinks under both designs. The sweep's instrument (ready-rate movement)
is the same one gate 3.2(4) uses.

## 4. Expected value and falsifiers

Honest ceiling: arrival latency ~120-140 ms/token at 3.00 legs. Legs ->
0.60 with 54% of boundaries fully local does not scale latency by 0.2 (the
wait is a max over present legs, and the slowest member still anchors the
boundaries where it appears), but the fully-local half of boundaries pays
~zero wire wait. Exact-mode target: ~180-200 ms/token (~5-5.5 t/s at
exact-class quality - i.e. v3-class speed WITHOUT deferral's decode-path
perturbation); composed with v3 on the residual legs it should press the
probe ceiling (~5.3+) with strictly less late mass than v3 alone.

Falsifiers / stop rules:
- inc-0 live zero-leg frequency far below model (routing correlation or
  profile drift) -> re-price before building inc-1.
- Gate 3.2(3) exact-mode A/B: if dropping ~2.4 legs/boundary moves t/s by
  less than fleet noise AND (under v3) the ready rate on remaining legs
  does not rise, then per-leg cost was not the binding term either - the
  member-compute axis is dead on this fleet; remaining levers are member
  hardware and #60-class transport.
- Placed-layout entry fee: if PLACE=1+skip cannot beat default-EP v3 even
  with legs at 0.6, close the track and record why (the -7% placed tax
  exceeded the leg win).

Operational cost to budget per placed leg: permuted single-threaded load
(~35 min) + the #72(b) cache-limit mitigation (WORKER_CACHE_LIMIT_MIB
raised: local 96G, .11 128G; local recreate needs WORKER_PORT=50053).

## 5. Prior art (one line each)

- DeepSeek EPLB: replicates hot experts for load balance across EP ranks -
  same skew premise, throughput-batch regime.
- ktransformers/Fiddler: static profiled placement works; our twist is the
  latency-regime consequence (legs, not bytes).
- MegaScale-Infer: disaggregates attention/FFN and micro-batches to fill
  waits - the -np>1 cousin, filed separately in fill-the-bubble 2.3.
