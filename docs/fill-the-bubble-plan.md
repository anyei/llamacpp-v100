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
