# Fill the bubble: overlap boundary waits with useful work (TASKS #71, escape c)

Status: design v1, 2026-07-27 night. Follows the gate-5 null (#75) and the
wire-format ladder (proto 4.12/4.13). Companion ledgers:
distributed-inference-plan.md 6b (measured escapes table),
research docs 2026-07-24-horizontal-scaling.md section 9 and
2026-07-parallel-decoding-and-distribution.md (2026-07-27 addendum).

## 1. The measured starting point

Record hy3 EP roster, all default-on levers (BCAST_FUSE=2, wire q8_0 > f16 >
f32): ~4.0 t/s = ~250 ms/token. Decomposition (from the #7 timing split plus
the boundary census, both 2026-07-27):

- member compute: ~109 ms/token (instrumented, does not compress uninstrumented)
- boundary wire bytes: were ~68 ms at f32; q8_0 cut ~3/4 of them -> ~17 ms
- boundary arrival latency: the remainder, ~120-140 ms/token - 159
  boundaries/graph (80 B1 broadcasts + 79 B2 gathers) at ~0.8-0.9 ms average,
  serialized because per-socket ordering queues the next gather behind the
  previous writes. Fusion already removed the removable writebacks; the census
  shows 96-98% of gather values arrive via the fused pre-request, so there is
  essentially no request-RTT left to shave.

The wire-byte lever is exhausted (q8 == f16 within fleet noise). What remains
is compute (37%) + arrival latency (~50%): the per-layer wait for the slowest
member's contribution to clear the wire. Escape (c) = keep the fleet busy
DURING that wait instead of shrinking it.

## 1b. ESCAPES END-STATE (2026-08-01)

All three escapes are now fully measured on the record roster:

- **(a) cheaper boundaries: CLOSED.** q8/f16 wire (PPL-neutral, default-on)
  exhausted the byte axis (q8 == f16); proto 4.14 zero short replies
  (57KB -> 1B) moved nothing - the wait was never byte-shaped.
- **(b) fewer boundaries: CLOSED.** LP pair fusion halved boundaries
  exactly (79 -> 41) for only +8-13%: members compute both layers'
  experts per lap, so per-lap latency doubles as laps halve - only fixed
  per-boundary costs are saved - and train-free pairing is quality-
  catastrophic on a reasoning model (lp-pair-fusion-plan 3e).
- **(c) fill the bubble: BANKED.** Deferral v3 +34% (serve candidate,
  soak pending) and -np 2 +40% aggregate at no idle cost. Spec lane
  shelved (ratio <= 1.0).

Cross-cutting verdict: a wire member's lateness is fused-chain PIPELINE
PHASE LAG. Levers that survive it: transport-class latency reduction
(#60 - LP priced the per-lap fixed cost at 10-15% for a mere halving),
faster member lanes (hardware; the V4 dual-role/lean-roster wins are
this lever), and -np throughput scaling. Placement (x2), threads,
leg-weight, boundary-count and spec are all measured dead on this
hardware.

## 2. Candidates, ranked

### 2.1 Probe 0 first: re-measure spec-over-boundary on the current stack

Stage 1 (LLAMA_META_LOCAL_DRAFT=1, commit a11a99f00 on `parallel-inference`)
localizes the MTP draft graph to coordinator members; measured 2026-07-23/24
at spec/plain ~0.85 and shelved. Two of the three cost terms behind that 0.85
have since collapsed:

1. Draft fleet traffic: already fixed by stage 1 itself (draft cost 2949 ->
   222 ms/64 iters, 13x).
2. Verify-batch wire bytes: an n-lane verify multiplies boundary payloads
   ~n-fold; at f32 that was ~68 ms/token of byte time scaling with n. Now q8
   cuts those bytes 4x, so the verify byte penalty shrank 4x.
3. Verify-batch member compute: the #52 law (distinct expert reads multiply
   with chain length) still stands - this term did NOT change.

Cost model (record roster, C = 109 ms compute, L = ~140 ms boundary):

    plain:      C + L                    = ~250 ms/token
    spec pass:  C*f(n) + L + eps_bytes   per (1 + mean accepted) tokens

At the measured acceptance (83%, mean chain 2.17 at n-max 3), the pass
amortizes L across ~2.17 tokens. Break-even is f(3) <= ~2.8; the stage-1
worker-op counts suggest f(3) ~ 2-2.5, which lands at ~5.3-6.1 t/s
(+30-50%). The 0.85 verdict was priced on a byte-heavy boundary that no
longer exists - this is a measurement gap, not a design bet.

Action: merge `expert-placement` (fusion + wire) into `parallel-inference`,
re-gate loopback byte-identity, then one fleet A/B (plain vs MTP=1 +
LOCAL_DRAFT=1), 6x 100-tok greedy per leg, coherence-read. Decision rule:

- ratio >= 1.05: spec is live again -> tune n-max/p-min, then compose with 2.2
- ratio ~ 1.0: bubble eaten by verify compute growth -> measure f(n) with
  BOUNDARY_STATS, decide between draft shortening and 2.2
- ratio < 1.0: latency does not amortize as modeled -> instrument, then 2.2

### 2.2 Expert Deferral (ktransformers pattern; APEX deferred-sync)

The only candidate that attacks the arrival-latency term directly with no
second stream and no new hardware. Mechanism: at layer L's expert reduce, do
not wait for the slow (wire) members - the owner proceeds into layer L+1 with
the partial sum it already holds (own experts + any contributions that
arrived), and late contributions are injected into the residual stream when
they land (one layer late, or dropped under a staleness bound). Field numbers:
ktransformers +33% decode at <=0.5% accuracy; APEX corroborates the
never-stall-the-critical-path pattern independently.

Why it fits this fleet specifically:

- The delayed-reduce machinery already moves reduce points across the graph;
  the #75 ownership masks/tables already describe exactly which member holds
  which contribution. Both repoint to deferral with modest surgery.
- The residual stream makes one-layer-late addition algebraically clean: a
  deferred contribution d added after layer L+1's attention differs from the
  true value only through attention's nonlinearity on (x vs x+d); with 8-of-256
  routing the wire members' share of any single layer's FFN output is small.

Quality risk is real and the gates are mandatory: PPL on the record roster
(wikitext 8x -c 512, the wire-gate protocol) + coherence-read + a
deferral-rate counter (fraction of layer-reduces that proceeded partial).
Prototype vehicle: trunc-hy3 loopback with an artificial worker delay, then
the fleet. Tunables: defer only wire members (never owner contributions),
staleness cap (contribution lands at most 1 layer late, else stall), off
switch per layer range (first/last k layers full-sync - the field notes deep
and shallow layers tolerate deferral differently).

Ceiling: removing the full wait would be the "free boundaries" ceiling (~9
t/s from the #7 numbers); a realistic partial deferral (wire members only,
1-layer staleness) targets the ktransformers-class +30% -> ~5.2 t/s, and it
COMPOSES with 2.1 (deferral cuts L; spec amortizes what remains).

**Ceiling MEASURED 2026-07-28 (GGML_META_PROBE_DEFER_GATHER, commit
649742b3b + the response-FIFO transport fix 95db33ef6): probe plateau
5.0-5.55, mean ~5.3 t/s vs 4.00 control = +32%, zero worker errors over 12
runs.** The probe zeroes wire contributions outright (output garbage by
design), so +32% is the no-quality-cost bound; real deferral (late
injection) captures some fraction of it. The number matches ktransformers'
+33% claim for the same pattern. Build is GO. The transport half is DONE
and general: the RPC client now keeps a per-socket in-order response FIFO
(pings + fused fetches; early-read fetches stash) - a deferred fetch no
longer desyncs interleaved traffic, which is the async-arrival primitive
the real mechanism needs. Remaining: the graph half (inject the drained
value into the residual stream one layer late, staleness cap, deferral-rate
counter, PPL + coherence gates).

**Graph half BUILT + QUALITY-GATED 2026-07-28 (GGML_META_EXPERT_DEFER,
f5dd4e1fb):** an eligible reduce sums local contributions and distributes;
wire partials drain at the next multi-contributor star reduce (the response
FIFO makes the interleaved traffic safe) and are ADDED there - the stream is
corrected additively one reduce late, staleness structurally capped at one
boundary. Eligibility requires a next same-size star reduce, so the last
reduce (logits path) always gathers exactly. Gates: off-gate byte-identical
(sha 91a2d41a); engagement defers == injects, LOST 0.00; trunc PPL +1.8%
inside its error bar (5-layer stub exaggerates per-boundary share);
record-roster fleet PPL pending on a CORRECT binary - the first attempt
(3.7819 +/- 0.211) ran a pre-deferral CUDA build with the env silently
ignored, so it is another BASELINE point (usefully: baseline family now
3.7669-3.7950), not a deferral measurement. Known caveat: the first
execution of a newly built graph shape partitions subgraphs differently
from the cached one, so run 1 of a shape differs textually from runs 2+
(deterministic thereafter; batch-invariance class). Trap for the gate list:
before ANY fleet leg, verify the binary carries the feature (strings/commit
stamp) - an env the binary does not know is silently ignored and the leg
measures the baseline.

**v1 FLEET MEASURED 2026-07-28 (correct binary, engagement 212-216
defers==injects/graph, LOST 0, zero worker errors): decode 4.43-5.43 mean
5.07 t/s vs 4.00 control = +27%, capturing most of the +32% probe ceiling -
but the COHERENCE GATE FAILS: the output collapses into a repetition
spiral.** Diagnosis: v1 defers EVERY wire partial = the majority of each
layer's expert mass arrives one layer late (the three wire members hold
~123 GiB of the experts); ktransformers' quality result defers a small
subset. The trunc PPL +1.8% was the honest early warning. **v2 design =
READINESS-GATED deferral: at the gather, poll each wire member's socket -
consume exactly (no defer) when the fused response has already arrived,
defer only actual stragglers.** Quality perturbation then scales with real
lateness (rare, straggler-sized) instead of total wire mass, while the
speedup lives exactly where the wait was. Fallbacks if quality still moves:
layer-range cap (full sync first/last k layers), member cap (defer only the
slowest member). Transport needs one new primitive: a non-blocking
"response head ready" check on the socket fd (poll(2); once the head has
arrived the remaining KBs follow at wire speed, so blocking on the tail is
cheap).

**v2 BUILT 2026-07-28:** `socket_t::recv_ready()` (poll(2)/WSAPoll, 0
timeout) + `ggml_backend_boundary_fused_ready` on the RPC client (walks the
response FIFO consuming entries only while the socket reports readable
bytes; true once the pending FETCH payload is stashed) + the meta gather
polls each candidate and defers only stragglers. `GGML_META_EXPERT_DEFER=1`
is now v2 readiness-gated; `=2` preserves the v1 defer-all behavior for
measurement. Two quality fallbacks shipped alongside so a failed fleet gate
does not cost another build cycle: `GGML_META_EXPERT_DEFER_WAIT_US` (busy-
poll grace window per boundary before a straggler defers) and
`GGML_META_EXPERT_DEFER_SYNC_EDGE` (first/last k multi-contributor reduces
always exact). New `ready` counter in the `META_EXPERT_DEFER` line: defer
rate = defers/(defers+ready) is the primary fleet instrument - if it sits
near 100%, readiness gating has degenerated to v1 and quality will follow;
the honest expectation on the record roster is HIGH (the wire members'
per-layer time exceeds the owners', so their responses are rarely early) -
the WAIT_US/SYNC_EDGE knobs exist to buy quality back at measured latency
cost. Worker-side test hook `GGML_RPC_DEBUG_FUSED_DELAY_US` fakes a
straggler on loopback. **Loopback gates PASSED 2026-07-28**
(71-defer-v2-gate.sh, trunc-hy3, 3 loopback workers, fuse=2 + q8):
off = stable byte-identical c80261ff (no-op when off); m1 engages at
defers 3.0 / ready 2.8 per graph (51.5% - loopback workers are genuinely
marginal at gather time), injects==defers, LOST 0; m1d (3 ms delay on one
worker) rate rises to 66.7% = the straggler defers; m1dw (delay +
WAIT_US=10000) defers 0.0, STABLE, byte-identical - the grace window fully
recovers exactness at ~5% loopback t/s cost; m2 reproduces v1 (rate 100%,
text 45a9d603); m2e (SYNC_EDGE=3) syncs all ~6 stub reduces, byte-identical.
v2 output is run-to-run NONDETERMINISTIC by design (readiness races);
byte-identity applies only to 0-defer configs. Harness trap that burned a
cycle: stale relative-path workers from the prior session held 50901/50902
past an anchored pkill - gate run 1 measured yesterday's worker binary.

**v2 FLEET LEG 1 MEASURED 2026-07-28 (record roster, commit 6229e354a,
binary env-verified): pure readiness gating = t/s WIN, PPL FAIL.**
Same-day warm control 3.67 t/s (8 runs, 3.15-3.98, pre-v2 binary, 8h-warm
serve; the overnight 15-run plateau was 4.00). EXPERT_DEFER=1: plateau
mean 5.17 t/s over 11 runs (4.79-5.57) = +41% vs same-day control, +29%
vs 4.00 - above v1's +27% and at/above the +32% probe ceiling band.
Defer rate 65.5-69.1% (defers ~139-149, ready ~67-74 /graph), LOST 0.
Coherence READ: genuinely coherent physics reasoning (nothing like v1's
repetition collapse), BUT a rare single-CJK-token intrusion when quoting
the prompt (2 of 13 reads: "in这样.", "in其 detail") - a visible quality
scar. Fleet PPL (wikitext 8x -c 512): **4.5735 +/- 0.27 vs baseline
family 3.7669-3.7950 +/- 0.21 = ~+21% - OUTSIDE the bar, do not serve
pure readiness gating.** Lesson repeated: the coherence read alone is not
a quality gate; 66% of wire mass one layer late reads fluent and still
costs a fifth of the model's likelihood. Sweep in progress: SYNC_EDGE=8
leg (protect shallow/deep reduces at ~10% of boundaries' t/s cost), then
WAIT_US ~400 - walking the frontier between exact +0% and v1's
incoherent +27%.

**Sweep leg A (SYNC_EDGE=8) MEASURED: t/s 4.82 mean over 12 runs (+31%
vs 3.67), defers 123/graph rate 71%, all 12 coherence reads CLEAN (the
CJK-intrusion artifact is gone - the edges were its source), but PPL
4.3910 +/- 0.26 = still ~+16% outside the bar.** Reading: the PPL damage
is DISTRIBUTED over the ~123 middle-layer defers, not concentrated at
the edges (edge protection removed only 0.18 of the 0.78 PPL excess).
Linear damage model: PPL-neutral needs defers down to roughly ~35/graph
(rate ~20-25%). Leg C = SYNC_EDGE=8 + WAIT_US=600 (grace window sized
to ~70% of the 0.85 ms mean boundary wait) - measures how much of the
+31% survives at a materially lower defer rate.

**Leg C (WAIT_US=600 + SYNC_EDGE=8) NULL: rate 73.2%, defers 123.8 -
the sub-ms grace window catches essentially nothing.** In the deferred
steady state the pipeline phase-shifts: wire responses run MILLISECONDS
behind the owner's gather (member layer time exceeds owner layer time,
and once deferring the member stays a boundary behind), so readiness is
a CLIFF, not a slope, at sub-ms windows. Early t/s 3.93/4.80 (failed
waits still bill ~600us x straggler boundaries). Leg D = WAIT_US=3000:
a window that actually catches the response should RE-SYNCHRONIZE the
pipeline (the member stops falling behind - the loopback m1dw leg showed
exactly this flip: rate 66.7% -> 0 at 5% t/s cost), so the rate may
collapse rather than shave. If it lands ~20-30% rate at t/s >= ~4.3,
that is the PPL-candidate frontier point; if it regresses to control
t/s, the readiness dial has no servable middle on this fleet and the
verdict follows plan section 4 (fall back to escape (b)).

**Leg D (WAIT_US=3000 + SYNC_EDGE=8): the resync flip is REAL - decode
defer rate collapsed 73% -> 5.0-5.9% (defers ~10/graph) - but t/s 3.90
mean (8 runs, 3.64-4.20) = control, and PPL came back 4.3965 +/- 0.26 =
IDENTICAL to edge8's 4.3910 at 71% rate. ~10 defers/graph cost the same
+0.6 PPL as 123. THE LINEAR DAMAGE MODEL IS FALSIFIED - and the failure
mode identifies itself: PPL is an all-PREFILL instrument, and a late
injection during prefill corrupts the KV that every subsequent position
of the chunk conditions on, so even rare prefill defers poison the whole
context (the serve-side CJK artifact was the same mechanism - it struck
exactly where the model quotes its own prompt, i.e. corrupted prompt
KV; note the serve counters I steered by were decode-graph rates, while
the PPL vehicle's own defer rate went unrecorded - the BOUNDARY_STATS
dump needs >=128 graphs and an 8-chunk run never prints one).**

### 2.2b Deferral v3: decode-only (prefill always exact)

One-line mechanism change: deferral eligibility additionally requires
the boundary tensor to be DECODE-shaped (ne[1] == 1);
GGML_META_EXPERT_DEFER_PREFILL=1 re-enables prefill deferral for
measurement. Rationale: prefill already amortizes each boundary across
the whole ubatch (the per-token boundary cost that deferral attacks is
a decode phenomenon), and prefill exactness keeps the KV - and thus
everything the model conditions on - bit-clean. Expected: the decode
speedup survives (~+30-40%), the prompt-KV artifact class disappears,
and fleet PPL returns to baseline BY CONSTRUCTION - which also means
PPL STOPS BEING THE QUALITY GATE for v3 (it no longer exercises the
lossy path); the decode-side quality evidence must come from generative
reads and long-generation degeneration checks. Loopback regate: off
byte-identical c80261ff; engagement decode-only (defers 2.3/graph vs
3.0 with prefill included), LOST 0.

**v3 FLEET MEASURED 2026-07-28 - ALL GATES GREEN. This is the serve
candidate.** Decode defer rate 66.6% (defers 139 / ready 70 per graph),
LOST 0; plateau 4.93 t/s mean over 12 runs (4.60-5.35, last five 5.08-
5.35 still warming) = +34% vs the same-day 3.67 control, ~+25% vs the
overnight 4.00 plateau; coherence 14/14 CLEAN reads including two
300-token generations (coherent essay + well-formed poem, zero CJK
intrusions, no degeneration - the prompt-KV artifact class is gone with
prefill exactness); fleet PPL 3.7819 +/- 0.21 = the baseline family
dead-center (and identical to the earlier mislabeled 'pre-deferral
binary' run - both measure the exact-prefill path, a tidy cross-check
that the decode-only gate engages on the PPL vehicle). Honest residual
risk: PPL cannot see the decode-path perturbation by construction; the
generative evidence is 14 greedy reads - recommend a real-workload
quality soak before making EXPERT_DEFER=1 the default. Sweep artifacts
(v2 knobs) remain available: WAIT_US=3000 gives exact-class behavior at
control speed (rate 5.9%); SYNC_EDGE composes if a decode-quality issue
ever surfaces.

### 2.2c After v3: what the sweep re-ranks (escape (b) design frame)

The WAIT_US=3000 leg is the load-bearing measurement for the roadmap: at a
5.9% defer rate t/s equals control, so the per-boundary wait is the
MEMBER-COMPUTE GAP (wire members' per-layer expert time exceeding the
owners' layer time), not fixed transport latency. Consequences:

1. **#75 hot-expert placement is RE-OPENED under v3.** Its gate-5 null (-4%)
   was measured in the exact regime, where the fleet was latency-bound and
   cutting member bytes could not shorten the critical path. Under v3 the
   owner is drain-throttled by member per-layer time - exactly the term
   placement shrinks (fewer cold expert bytes per member per token). The
   machinery is DONE and gated (skip sentinel, record artifact
   placements/hy3-record-21-21-46-50-27.json); the A/B is zero new code:
   `PLACE=1 EXPERT_DEFER=1 STATS=1 ./run-ep-fleet-hy3-spec.sh` vs today's
   v3 plateau (4.93-5.35), defer rate as the secondary instrument (if
   placement speeds members up, ready-rate should RISE). ~1.5 h fleet
   window (2 loads + runs).
2. **Layer Parallelism pair-fusion's 1.9x ceiling (addendum pricing) is
   STALE.** It assumed ~3.7 ms/boundary of removable cost; the sweep shows
   most of that is elastic member compute, which pairing does NOT reduce
   (members do both layers' experts per cycle). LP's real exact-path win is
   the halved count of FIXED per-boundary costs (socket turnaround, host
   reduce, payload latency) - order +10-15%, and it composes POORLY with
   v3 (deferral already hides per-boundary fixed costs one layer deep).
   LP also carries the known reasoning-quality collapse risk and a
   model-graph rewrite (residual rewiring is builder-side, not a GGUF
   transform alone). VERDICT: parked again until the member-compute axis
   is exhausted. Cheapest ceiling probe if revisited: 40-layer
   block_count-truncated record serve (upper bound: halves compute AND
   boundaries; if THAT is not >= ~1.5x, LP is dead here).
3. **The member-compute axis is now the multiplier lane**: placement (1),
   member CPU upgrades/threads, and expert replication (hot experts on
   multiple members so gathers shrink). In that order of cheapness.

**(1) MEASURED 2026-07-28 late night - placement NULL under v3 too;
#75 closed on this roster in both regimes.** Placed 4.68 t/s mean
(12 runs, 4.01-5.18) vs same-session control 5.04 (4.61-5.50), both
coherent, LOST 0. The mechanism-level tell: defer rate 74.9-75.4% vs
76.0% - the ready rate did not rise, i.e. placement did not measurably
shorten member per-layer time. Whatever the wire members' bottleneck is
per layer (RAM bandwidth on the full routed set, thread scheduling, the
q8 encode), it is not the cold-expert byte mix placement optimizes.
Remaining member-compute levers: worker thread/affinity tuning, expert
REPLICATION (different mechanism: replicas let gathers shrink - fewer
contributors per boundary), faster member hardware. Also note the
placed leg pays a real operational tax: permuted single-threaded loads
(~35 min) and a second cache layout (the #72(b) wedge needed the cache
limits raised to load at all).

**Worker thread/affinity sweep MEASURED 2026-07-29 (v3 serve, record
roster, 8x100 greedy + defer counters per leg, cold-vs-cold legs,
coherence-read all): the thread lever is EXHAUSTED at the current
config.** L0 baseline (-t 8 local / -t 8 .11, disposable per-subgraph
pool) 4.97 t/s rate 65-70%; L1 (+GGML_RPC_THREADPOOL_POLL=0 both) 3.98
(-20%) rate unchanged - the persistent condvar pool is a fleet LOSS
despite the +2% loopback prior; L2 (-t 12 local / -t 14 .11) 1.79
(-64%!) - local-box oversubscription slows the OWNER (defer rate fell
to 60% while t/s collapsed = members looked "readier" because the
coordinator got slower) and .11's hybrid Ultra 9 (6P+8E) pays E-core
barrier drag on every op; L3 (-t 8 local / -t 6 .11 = P-cores only)
5.21 t/s (4.91-5.35) rate 68.8-71.9% - marginal WIN, adopted as the
serve config. CALIBRATION VERDICT: member per-layer time is not
thread-elastic upward; ready-rate never materially rose in any leg ->
the member-compute gap stands, leg-skip/replication (below) is the
remaining lever. (.11 corrected: 62 GB RAM box - the "128G" in prior
notes is its DISK cache limit.)

**-np 2 UNDER v3 MEASURED 2026-07-29 (record roster, winning -t8/-t6
worker config): AGGREGATE +40%.** Single-stream inside the -np 2 serve:
5.11-5.68 t/s = NO idle penalty vs -np 1. Two concurrent 100-tok greedy
streams: per-stream 3.6-3.9, aggregate 7.2-7.7 t/s (warm rounds) vs
~5.2-5.3 single = the second stream genuinely fills boundary bubbles
(defer rate ~71%, LOST 0, both streams coherent). Production
implication: -np 2 is strictly better (parallel capacity at no
idle-stream cost); per-stream latency under simultaneous load costs
~30%.

**LP CEILING PROBE MEASURED 2026-07-30 (the 2.2c item-2 probe): GO -
the fewer-boundaries axis is ALIVE.** 40-layer block_count-truncated
record serve (same roster, v3 stack, wall-time-only): 8.85-9.67 t/s
mean ~9.3 over 7 valid runs = ~1.8x the 4.97-5.21 full-model plateau,
above the >=1.5x go/no-go line. Defers 72.5/graph (half of the full
model's ~148 - boundary count halved as expected), LOST 0. Read with
the leg-skip null (member lateness = chain phase lag): boundary COUNT
multiplies the lag laps, and halving it nearly doubles decode - LP
pair-fusion comes OFF the parked list; its realizable fraction of the
1.8x is the boundary share (the probe also halves compute). Next:
LP design (residual rewiring is builder-side; reasoning-collapse risk
gates apply per the addendum).

**Replication design doc LANDED 2026-07-29:
docs/hot-expert-replication-plan.md.** Key reframe: replication =
hot-mass concentration (the shipped #75 artifact already does this,
Cov@25.5% = 0.913) + a DYNAMIC LEG SKIP at the star gather - the piece
placement never had (the B2 contributor set is graph-build-static;
a zero-routed member still ships an exactly-zero tensor and the owner
waits on it). Modeled on the hy3 profile: mean legs 3.00 -> 0.60,
54% of decode boundaries fully wire-free, exact math (byte-identity
gate). Sequencing unchanged: thread sweep first, then inc-0 counters,
then the leg-skip build.

### 2.3 Later / other axes

- SpecPipe / PipeInfer continuous speculation with early cancellation:
  strongest published numbers (2-5x TBT) but assumes a layer PIPELINE; our
  record roster is a star-EP topology. Revisit for layer-mode fleets or
  after 2.1 establishes the draft plumbing under continuous load.
- Ping-pong / dual-batch micro-batching (MegaScale-Infer, vLLM DBO): fills
  boundary bubbles with a SECOND stream - a throughput lever for -np > 1
  serving, not single-stream latency. File separately from #71.
- Stage 2 self-speculative MoE (draft on the VRAM expert set): stays shelved.
  Its premise (cut draft-side expert reads) targets the byte axis that gate 5
  measured non-binding, and hy3's native MTP head already gives 83%
  acceptance. Reopen only if probe 0 shows spec live but draft cost binding.
  (Recon 2026-07-29: verdict confirmed pre-build, code map banked - see 2.4.)
- Layer Parallelism pair-fusion is escape (b) (fewer boundaries), tracked in
  the research addendum - not this doc.

### 2.1a Probe 0 MEASURED 2026-07-27/28 - ratio 1.00, spec is still shelved
(at n-max 3 / p-min 0.75)

Merged tree (37cf1c45e), record roster, both legs coherence-read. The 6-run
protocol proved too short - the fleet warms over a serve's first ~20 minutes
- so both legs were run to plateau: control 4.00 t/s (15 runs, 3.62-4.36),
spec + LLAMA_META_LOCAL_DRAFT 3.99 t/s (9 runs, 3.60-4.69). Acceptance
84-97%, and the loopback vehicle (no wire cost) shows +58% for the same
config - so the fleet-side loss is in the verify pass itself: an n-lane
verify multiplies BOTH member expert reads (the #52 law) and per-boundary
payload bytes (B1/B2 carry n lanes; q8 shrank the unit cost, not the
scaling), and together they currently price at exactly the amortization won.

Learned along the way: per-request speculative.n_max/p_min overrides are
IGNORED for draft-mtp (server-schema.cpp "disabled for now" TODO) - only the
server flags (NMAX/PMIN in run-ep-fleet-hy3-spec.sh) change the draft shape;
and single-serve measurements drift +0.5 t/s from cold to warm, so only
plateau-vs-plateau comparisons are honest.

Follow-up (ii) MEASURED same night and it CLOSES the spec lane: NMAX=6
PMIN=0.3 (server flags) = 2.34 t/s mean (1.87-2.80, 12 runs, coherent),
acceptance 27-45% at 125-172 drafted per 100 generated. Rejected lanes pay
full verify-batch bytes + reads, so aggressive drafting is a 0.59 ratio.
Spec on this fleet: best-acceptance config breaks even, anything looser
loses - SHELVED until the per-lane boundary cost falls (RDMA-class
transport, #60) or tree-verify with expert-reuse-aware selection changes
lane economics. The BOUNDARY_STATS split (i) is now optional - the sweep
demonstrates lane-cost dominance behaviorally. Expert Deferral (2.2) is the
active escape-(c) lever.

### 2.4 Stage-2 self-spec RECON 2026-07-29 (no build): stays shelved by its
own kill math; code map banked; one live spec-family gap found (defer-blind
verify)

Recon pass before opening the stage-2 lane (docs digest + full code map, no
code written). Verdict first, then the banked map.

**Kill math - the premise cannot move the measured bottleneck.** Stage 2's
mechanism is "make the draft cheaper" (route the draft only to VRAM-resident
experts; zero extra weight reads). But probe 0 (2.1a) already measured the
economics of a nearly-free draft: LOCAL_DRAFT cut draft cost 13x
(2949 -> 222 ms/64 iters), acceptance ran 84-97%, and the ratio was still
1.00. An even cheaper draft with at-best-equal acceptance is bounded by the
same number - the whole loss lives in the VERIFY pass (n-lane member expert
reads, the #52 law), which self-spec by definition leaves full. The stated
reopen gate ("spec live but draft cost binding", 2.3) is unmet on both arms.
The second reopen key - "VRAM expert fraction grows a lot" - has not moved:
hy3 owners already held 42 GiB (25.5%) when the null was measured, the V4
dual-role shape holds ~36 GB (same class), and ~123 GiB of experts still
live in member RAM. Growing it a lot means hardware, not config. Verdict:
STAYS SHELVED; unlocks unchanged (#60 transport, big VRAM growth,
tree-verify with expert-reuse-aware selection per MoE-Spec/EcoSpec).

**Code map banked for whenever an unlock lands** (smallest-diff build path,
so the future session starts warm):

- The restriction primitive already exists twice: the #75 per-member
  exp_remap/exp_mask tables consumed in build_moe_ffn
  (src/llama-graph.cpp:1930-1962, sentinel -1 lanes compute nothing), and
  the group-limited routing pattern (src/llama-graph.cpp:1898-1911) is the
  exact template for a pre-topk -INFINITY owner-only mask. build_moe_ffn
  also has an unused `selected_experts_in` injection parameter.
- Owner-residence is one lookup at table-build time: the
  ggml_backend_meta_set_expert_ownership registry + wire_member[] (RPC vs
  in-process) already distinguish VRAM members
  (ggml/src/ggml-backend-meta.cpp:2102-2131, 2743-2815).
- Accidental near-miss: under LLAMA_META_LOCAL_DRAFT a localized draft graph
  resolves the MIRRORED exp_remap/exp_mask to the first local member's copy
  - i.e. a placed serve would restrict the draft to that member's owned set
  FOR FREE - but it cannot fire today: placement arrays are sized
  hparams.n_layer() (src/llama.cpp:426) so nextn layers are never placed,
  and the MTP builder's il >= n_layer() fails the table-bounds guard
  (src/llama-graph.cpp:1930). Closing that gap = extend the artifact/tables
  over nextn layers, or ship a draft-specific mask.
- A full-trunk self-draft context (non-MTP) is the expensive variant: a
  second llama_context over the same model works (that IS draft-mtp,
  common/speculative.cpp:2338), but ctx_other KV sharing allowlists only
  GEMMA4_ASSISTANT/EAGLE3/DFLASH (src/llama-context.cpp:302-320), so
  hy3/V4 would duplicate the full KV and pay catch-up decodes. The MTP head
  stays the right draft vehicle. A per-context expert-subset knob also
  needs a cparams field wired into graph-reuse identity (precedent:
  nextn_layer_offset, src/llama-graph.h:776).

**The gap the recon DID find (live, unexplored, attacks the binding cost):
verify boundaries are defer-blind.** v3 deferral and the ZL counters both
gate on single-token boundaries (ne[1]==1,
ggml/src/ggml-backend-meta.cpp:4430 and :4354), so a spec verify batch
(ne[1] = 1+n_draft) silently disengages deferral. Two consequences:
(1) every spec A/B ever run had its verify passes fully exposed to member
phase lag - the exact term v3 hides on decode; (2) today's production
baseline INCLUDES v3 (+34%), so a spec config now pays a defer-loss on
every verify pass relative to it - the wash would re-measure as a loss.
The one spec-family probe left with a live mechanism:
GGML_META_EXPERT_DEFER_VERIFY - widen the v3 gate to verify-shaped
boundaries (ne[1] <= 1 + n_draft), letting straggler member partials inject
late during verify exactly as they do on decode. Cheap (gate widening + one
env; the FIFO/readiness transport already handles multi-token fetches).
Quality risk to state up front: late injection perturbs intra-batch
conditioning (the prefill-poisoning class, but depth <= n_draft = 3, not a
512-token chunk) and perturbed verify logits move ACCEPTANCE decisions -
rejected lanes cost speed not correctness, and accepted-token KV
perturbation is the same class v3 already ships on decode. Gate protocol =
section 3 ladder: loopback off-gate byte-identity (c80261ff), engagement
counters on verify-shaped graphs, then fleet A/B
(SPEC=1 EXPERT_DEFER=1 + DEFER_VERIFY=1) vs the v3 plateau (4.97-5.21),
acceptance + coherence-read every leg, long-generation reads. Decision
rule: only if it clears the v3 plateau by >= +5% does any stage-2 draft
work (VRAM-subset masks) become worth pricing again.

**DEFER_VERIFY MEASURED 2026-07-29 night - PROBE FAILS, spec family stays
closed, and the falsification is mechanism-level.** Built as
GGML_META_EXPERT_DEFER_VERIFY (61aaa24a5; ne[1] <= value widening, parse-
VALUES, injection-shape walk unchanged). Loopback gates first: off =
c80261ff 6/6; padded-spec legs proved the defer-blind claim quantitatively
(v3 under spec = 0.1 defers/graph; =4 restores 3.7-3.9; z-leg =0 == unset;
LOST 0 everywhere; server stable 12/12). Fleet A/B, record roster,
same-session back-to-back, workers identical (-t8/-t6), coherence-read
every leg:
- control (EXPERT_DEFER=1 STATS=1): warm 5.25-5.72 (8 runs), plateau
  5.14-5.96 MEAN 5.49 (10 runs), defers 141.6-152.6/graph rate 65.7-70.8%
  LOST 0, physics-reasoning read clean. (Above the historical 4.97-5.21 -
  good fleet day; warm caches.)
- test (SPEC=1 NMAX=3 LOCAL_DRAFT=1 + v3 + DEFER_VERIFY=4): warm
  4.22-5.95 mean 4.88, measured 4.18-4.89 MEAN 4.53 (10 runs), acceptance
  76-98%, defers 49.3-65.9/graph rate 65.5-66.9% LOST 0 (ENGAGED - without
  the widening this sits near zero under spec), reads clean.
- RATIO 0.83 (4.53/5.49) - the >= +5% rule fails by ~22 points, outside
  fleet noise by 3x. Deferral verifiably engaged on verify boundaries and
  the speed did not come back: hiding member ARRIVAL lateness cannot
  recover the verify pass, because the cost is member COMPUTE scaling with
  chain width (the #52 law), which deferral does not reduce. The 0.83
  ratio matches the pre-fusion stage-1 ratio (~0.85): defer-verify closes
  the v3-handicap confound and the spec economics are unchanged.
Verdict: the defer-blind-verify gap is now MEASURED CLOSED as a spec
recovery lever. DEFER_VERIFY stays in tree default-off (exact-when-off
gate-proven; useful instrument for any future multi-token-decode work,
e.g. tree-verify or block decode). Spec-family reopen keys shrink to #60
transport and tree-verify-with-expert-reuse economics; "compose spec with
deferral" is no longer an untested rival explanation.
Incident color for #72(a): the first control load OOM-killed the LOCAL
worker mid-stream (kernel global OOM, worker anon-rss 50.7GB while the
coordinator's no-mmap cold-stream peaked; V4->hy3 layout churn forced the
full streaming path) - decode -3 + connection-lost + untrusted in-process
reload followed; the clean relaunch on warmed page caches loaded fine and
the worker peaked at its normal ~46GiB share. Watch worker RSS on the
FIRST post-churn load.

### 2.5 INDEPENDENT VERIFICATION 2026-07-30 (4-agent literature sweep; no code,
no fleet legs) — stage-1/2 verdicts CONFIRMED, two premises revised, three
tasks filed (#84 verify-side economics, #85 block-decode stage 3, #86
pre-scheduling recon)

Method: two web agents hunted counter-evidence against the stage-1 and
stage-2 verdicts independently; one agent swept the blockwise/parallel-decode
SOTA; one re-derived the ledger's own record (the #52 law's provenance:
TASKS #52 2026-07-18 verify pass 1.17s vs 0.53s plain, worker exec 140ms vs
71ms floor, the 96%-acceptance run the SLOWEST; ancestor #31 LAW 3 -np 4
aggregate only +27%).

**Confirmed (the field reproduced us):** every 2025-26 MoE-speculation paper
prices verify as the UNION of experts across verified lanes — EcoSpec
(2607.12696) states it verbatim (dense lanes reuse weights, MoE cost = the
union); MoESD (2505.19645) and Cascade (2506.20675, 2-3x verify-time growth,
outright slowdowns from naive SD on MoE) and DraftExpert (2607.24434,
acceptance 22-46% band for naive drafting = our 0.59 leg) all match our
1.00 / 0.59 / 0.83 ladder in kind. The Cohere MoE+SD study places our regime
(bs=1, partial expert residency) exactly in the band where "verification adds
extra expert weight loading, limiting SD gains". No paper composes deferral
with verification (the DEFER_VERIFY 0.83 experiment appears novel) and no
paper tests experts sharded across multi-node CPU RAM over Ethernet — the
topology itself is unpublished territory.

**Revision 1 — the multiplier arithmetic (feeds #84 probe 1).** Measured
cross-lane routing overlap is substantial: Cohere (8-of-128, K=3 spec)
measures 30-38% adjacent-lane expert overlap → a 4-lane verify activates
~20.4 unique experts vs 29.1 uniform, i.e. ~2.5x not 4x; MoE-offload caching
work corroborates ~2x-over-independence consecutive-token reuse. Our n=3
member-read multiplier is therefore likely ~2.2-2.6x, not the 3x the
break-even model assumed. If break-even at ratio 1.00 required the full 3x,
a second cost is hiding (per-lane boundary fixed cost, lanes verified past
the acceptance point, router divergence). One counter decides it: distinct
expert IDs touched per member per verify graph vs the single-token floor.
No published number exists at 8-of-256 sparsity — our counter would be the
first.

**Revision 2 — "verify is untouchable" is now contradicted four ways
(feeds #84 probes 2-3, = spec REOPEN KEY 3):** MoE-Spec (2602.16052)
budgeted verify (cap the per-layer verify union to top-B by tree-aggregated
router probability; -1.4% acceptance, +10-30% t/s, training-free); SS-MoE
(WWW 2026, paywalled — claims unverified) confidence-gated
accept-WITHOUT-verify (verifies/token < 1 breaks the ratio-1.00 bound
structurally; 3.72x claimed); EVICT (2605.00342) lossless utility-optimal
tree truncation (-32.5% activated experts); EcoSpec expert-reuse-aware lane
selection — our own stated reopen key, now published (1.36x batch-1 vs
EAGLE-3's 1.22x). Favorable prior: our placement skew coverage@25.5% = 0.913
exceeds every published testbed (SpecMoE 2604.10152 best case: NLLB 0.84 →
4.3x). Standing risk law: lossy acceptance damage GROWS with task hardness
(2607.26627: +0.38pp GSM8K → +6.67pp AIME) — PPL cannot see it; hard-task
coherence reads are the gate.

**Stage-2 kill-math: right as specified, safe on its subtle point.** No
restricted-expert draft in the literature exceeds 84-91% acceptance (our
84-97% MTP is already at/above the published band), so "deeper chains at
very high acceptance" stays dead. Exact residual/delta-verify (reuse draft
hot-expert outputs in verify) exists nowhere and is structurally unsound —
hidden states diverge from the first restricted layer, so draft outputs are
computed on the wrong state for exact verify; approximate versions collapse
into lossy verification (probe 3's territory). The shelving stands; only
the verify-side keys above reopen anything.

**Block-decode sweep (stage 3, first enumeration — filed #85):** the only
published shape whose economics fit this fleet is CLP-class (2606.10935)
verification-free MTP emission — a tiny gate accepts k MTP tokens directly,
no verify batch. Recon caveat found before building: accepted tokens still
need trunk KV, so the catch-up ingest is an n-lane pass paying the same
union cost as a verify — CLP ≈ spec at 100% effective acceptance, and #84's
union counter gates whether that clears the null. Ruled out: Jacobi/
lookahead (56-120x extra FLOPs/step), Medusa trees (maximal union), raw
dLLMs and AR→diffusion conversion (RND1: 500B tokens, quality loss). Parked
with entry conditions: SBD/Fast-dLLM-v2 (~1B-token finetune, 3-5x fewer
passes at parity), OEA (2511.02237) batch-aware re-routing as a union
compressor, judge/relaxed acceptance (ICLR'25) as a zero-read stack-on.

**2026-07-31 MEASURED (probe 1 = the union counter, GGML_META_UNION_STATS,
c4c0c94b6):** loopback stub verify union multiplier **w4 = 3.06-3.14x** vs
the w1 floor of 8.00 (w2 1.8-1.9, w3 2.3-2.75, 7-tok prefill 5.1-5.3 vs 7).
Break-even was f(3) <= 2.8: **measured ~3.1 closes the stage-1 arithmetic -
ratio ~1.0 is the predicted outcome, no hidden second cost, and Revision 1's
overlap hope does not rescue spec at 8-of-256 sparsity.** The fleet number is
pending an instrument redesign: the routed-id eval callback CORRUPTS
meta-fleet serve quality on fused AND plain stacks (2x reproduced,
discriminated; full detail TASKS #84c) - the record-roster curve will come
from an offline per-position profiler dump on a single-box V4 -ncmoe prefill.
Bonus from the discriminator serve: post-merge fleet-spec regate GREEN
(coherent, 4.01-4.90 t/s cold, acceptance 83-93%).

## 3. Probe ladder (each gated before the next)

1. Merge expert-placement -> parallel-inference; CPU loopback byte gate:
   LOCAL_DRAFT=0 == pre-merge behavior == LOCAL_DRAFT=1 (trunc-hy3-mtp,
   2 loopback RPC members, sha protocol from dev-workflow).
2. Fleet A/B probe 0 (section 2.1) on the record roster, current default
   stack, workers restarted first (stale-manifest trap #8). Coherence-read
   every leg.
3. If spec >= +5%: acceptance/chain-length sweep (n-max 2/3/4, p-min
   0.6/0.75) - chain length trades f(n) against amortization; the model in
   2.1 predicts the optimum shifts longer as L/C grows.
4. Expert Deferral prototype behind GGML_META_EXPERT_DEFER (default off):
   loopback correctness harness with injected worker delay, deferral-rate
   counter, then fleet PPL + t/s A/B.
5. Compose the winners; update distributed-inference-plan.md 6b and TASKS #71.

## 4. What would falsify the frame

- Probe 0 lands ~1.0 AND BOUNDARY_STATS shows verify passes are NOT
  compute-inflated: then L does not amortize with batch depth, meaning the
  boundary cost is per-TOKEN not per-PASS somewhere in the stack (scheduler,
  KV plumbing, draft loop) - instrument before building anything else.
- Deferral prototype shows PPL moving outside the wire-gate error bar even at
  wire-members-only, 1-layer staleness: drop 2.2, fall back to escape (b)
  (Layer Parallelism pair-fusion) as the remaining latency lever.
