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
