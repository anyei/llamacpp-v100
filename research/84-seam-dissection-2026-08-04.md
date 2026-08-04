# Seam dissection round 2 (2026-08-04): trunc20 union corruption is EXECUTION-TIME, not placement

Continues the 2026-08-04 checkpoint (TASKS #71). Vehicle: full hy3 + `--override-kv
hy_v3.block_count=int:20` + `LLAMA_TRUNC_ARR=1`, lean-GPU fleet pair shape
(`-ts 5,21,55,56,28`, workers 50053/.11/.15). Scripts: `84-trunc20-dissect.sh`
(legs A/B = union, C = ctl, all with `GGML_META_DEBUG_REDUCE=1`, bodies captured),
`84-trunc20-ringtest.sh` (legs D/E = union + `GGML_META_MAX_GRAPHS=48`).
Artifacts: `/tmp/t20d/`, `/tmp/t20r/`.

## Verdict 1: the placement hypothesis is REFUTED

The checkpoint's fix direction ("deterministic callback split placement - fixed op
boundaries instead of timing-dependent landing spots") assumed the eval callback
makes sched-split/reduce placement vary. Measured: it does not.

- REDUCE traces (boundary placement + rebuild order) are byte-identical between
  serve A and serve B up to B's crash point (471/471 lines, diff = 0).
- Within serve A, all four probes produce byte-identical 621-line REDUCE
  sequences (probe1 == probe2 == probe3 == probe4, diff = 0).
- Placement under the callback IS different from ctl (chunk-forced subgraph
  closure at every `ffn_moe_topk-N`, ~3 boundaries/layer vs ctl's 2, and
  perpetual rebuilds: 2530 REDUCE lines in A vs 114 in C), but it is
  deterministic - same landing spots every serve, every probe.

## Verdict 2: the corruption is execution-time (values, not structure)

With placement byte-identical:

- A probe3 content sha 08330f89 vs probe4 f65d1d74 (temp 0, cache_prompt off,
  same serve) - token stream diverges mid-generation.
- A probes 1-2 = HTTP 500 "The model produced output that does not match the
  expected Content-only format" (`common/chat.cpp:3285`) - babble emitting
  parser-rejected output. This also resolves the old 7f8998ec sha-pollution:
  those were error bodies, and the 500s themselves are a vehicle artifact
  (parse layer), not a separate corruption class.
- Serve B: CUDA illegal memory access on device 1, surfaced in
  `ggml_backend_cuda_synchronize` during a probe-1 decode chunk - i.e. a kernel
  launched within the current chunk read a bad pointer. Placement identical to
  A at that point; A survived 4 probes. Memory corruption, nondeterministic
  across serves.
- Ctl (C) reproduces its historical shas exactly: probe1 bac4252d (the known
  cross-serve-stable first-decode sha), probes 2-4 1f211a37 (late-stable).

## Mechanism frame (post-dissection)

The callback path chops each meta split into ~21 chunk graphs (`ggml_graph_view`,
uid==0) per token. Each chunk is its own meta build; the build cache caps at
`ring_slots-1 = 7` (`GGML_META_MAX_GRAPHS` default 8), so every chunk build is
evicted before reuse -> permanent rebuild churn (visible as the 2530-line REDUCE
stream). The sched fully synchronizes members after every chunk
(`ggml_backend_meta_synchronize` syncs all members), so cross-chunk async racing
is excluded; the corruption must be a stale/dangling reference baked into or
consulted by chunk builds (freed-memory reads explain nondeterministic VALUES
under deterministic control flow, and the occasional IMA).

Suspects, each with a one-env discriminator:

1. Build-ring exhaustion/aliasing under churn -> test `GGML_META_MAX_GRAPHS=48`
   (all chunk builds fit, no eviction). Legs D/E.
2. `GGML_META_BCAST_FUSE=2` skip-writeback vs eval-callback readers - a
   DOCUMENTED unsupported combo (env-gates.md: "do not combine with debug
   readers of pre-boundary tensors (eval-callback) at level 2"). The B2 safety
   closure is chunk-aware in the conservative direction (piece-end = unsafe) and
   registers cross-piece repairs, but 21 pieces/graph x 20 layers amplifies any
   hole. -> test union leg with `GGML_META_BCAST_FUSE=0` if (1) comes back
   still-broken.

## Ringtest result (legs D/E): RING EXHAUSTION CONFIRMED

Union leg + `GGML_META_MAX_GRAPHS=48` (everything else identical to A/B, FUSE=2
kept): D = E = `77a08819 00d69d3b 00d69d3b 00d69d3b` - serve-to-serve
DETERMINISTIC and steady after the known first-decode warmup step (same
stability pattern as ctl's bac4252d/1f211a37). No 500s, no crash. Suspect (2)
BCAST_FUSE=2 is exonerated at this scale (it was active in both stable legs).

Mechanism: TWO-CLOCK MISMATCH in the build cache. Builds aged (and capped) by
`rebuild_seq` (increments per cache miss), but shadow-ring slots are only
actually recycled when the ring INDEX advances - which happens in init_tensor
(sched allocation events), not per rebuild; between advances the per-rebuild
clear re-clears the same already-empty slot. Under callback chunking (~21
uid==0 pieces per outer graph vs a `ring_slots-1 = 7` cap) every chunk build
was evicted before reuse -> permanent rebuild churn -> stale reads somewhere in
the rebuild path corrupt values nondeterministically (freed-memory contents),
occasionally fatally (leg B's CUDA IMA). Ctl never rebuilds in steady state,
which is why it was always immune.

## Fix (this commit)

`ggml-backend-meta.cpp`:

1. Build aging re-clocked from rebuild count to `reset_seq` = count of FRESH
   ring clears (a rebuild's clear targets `cur+1`; it only counts when that
   slot differs from the previous clear, i.e. after an init_tensor advance).
   This is the clock that actually invalidates shadows.
2. LRU size cap split per class: outer builds (uid != 0) keep the
   `ring_slots-1` cap; chunked pieces (uid == 0, eval-callback chunking) get
   their own pool capped at `max(256, 8*ring_slots)` so a whole chunk set
   (2 shapes x ~n_layers+1 pieces) stays cached.

With the fix, the union leg reaches build-cache steady state like ctl - the
churn regime (and whatever stale read it exposes) is no longer entered. The
underlying churn-time stale read is NOT root-caused node-by-node; it is now
unreachable in supported configs but remains a latent hazard if some future
regime again outruns the cache (residual noted in TASKS #71).

## Gates on the fixed build

- c80261ff byte-identity 6/6 (trunc5 loopback stub, default stack, no callback).
- trunc5 seam repro 5/5 legs (ctl/union/zl/ring/plain) all c80261ff distinct=1.
- trunc20 regate (84-trunc20-regate.sh, legs F/G union NO override + H ctl):
  F == G == `77a08819 00d69d3b 00d69d3b 00d69d3b` (serve-to-serve deterministic,
  steady after the first-decode warmup step, no 500s, no crash - identical to
  the MAX_GRAPHS=48 signature), H == `bac4252d 1f211a37 1f211a37 1f211a37`
  (ctl bit-exact vs pre-fix history). The eval-callback instrument path is now
  deterministic on the 20-layer vehicle; the final quality gate for lifting the
  fleet eval-callback caveat remains a coherence read on a real (non-babble)
  model at the next fleet window.

# Fleet coherence window (2026-08-04 evening): CAVEAT STAYS - deterministic WRONG math at scale

Real-model gate on the hy3 record roster (full hy3-1M-MTP, fleet
CUDA0,CUDA1,RPC0-2, production stack). Four loads, findings in order:

1. **Rotation-pin share-balance regression (NEW, blocks the old record -ts)**:
   the proven maxed split `-ts 21,21,46,50,27` now OOMs CUDA1 at warmup
   (cudaFuncGetAttributes OOM, 2/2, with AND without CUDA graphs) - first
   record-roster load since the 25647fa7c rotation pin. Resized
   `-ts 21,19,47,51,27` loads but the layout is grossly lopsided: CUDA0 4.7 GB
   (~zero experts) vs CUDA1 32.2/32.7 GB. Pinning rotation=0 for _exps makes
   the same member absorb the rounding remainder on EVERY layer instead of
   rotating it. CORRECTNESS is unaffected (ctl coherence passes on the lopsided
   layout); it is a balance/capacity regression. Fix direction: keep rotation
   fixed per tensor KIND (layer-uniform, which is all the degenerate-propagation
   fix needs) but stagger kinds across members (e.g. gate/up/down at 0/1/2).
2. **Union-callback VRAM overhead**: on maxed shapes the callback's chunked
   execution needs headroom the maxed split does not have;
   GGML_CUDA_DISABLE_GRAPHS=1 alone does NOT save it.
3. **The caveat STAYS - and is sharpened**: union leg on the resized layout =
   deterministic GARBAGE (identical bytes across two serves, 3.05-3.27 t/s);
   ctl same layout = COHERENT (90km/h 60 km correct + planets correct,
   4.97 t/s). **EXPERT_DEFER exonerated**: union with deferral OFF produces
   BYTE-IDENTICAL garbage to the deferral leg. So the chunk-path corruption is
   deterministic wrong math, defer-independent, CUDA-graph-independent.
4. **Counter signature of the chunk regime**: per-chunk BOUNDARY_STATS show
   `parts fused 0.0` (vs ctl 112.8/graph - the fused-arrival machinery is
   entirely bypassed under chunking), `deliveries skipped 0.0` (vs 237), and no
   META_EXPERT_DEFER line ever prints (deferral never engages on chunk graphs).

**Where that leaves the bug**: trunc5 loopback proves the chunk path CAN be
exact (union leg == c80261ff, the no-callback sha); full hy3 (32+nextn layers,
67+ chunks) is deterministically wrong. Scale-dependent correctness break
between 5 and 33 layers. NEXT DISSECTION (loopback-only, no fleet window
needed): byte-identity union-vs-ctl at trunc10/15/20 on the CPU loopback rig to
find the breaking scale, then diff the chunk builds at the first wrong layer.
Prime suspects: chunk-end forced boundaries on topk VIEWs, the stale/repair
registry across 67 pieces, or the wire GET path that replaces fused arrivals.
Logs: /tmp/hy3-union-coherence{,2,3}.log, /tmp/hy3-ctl-coherence.log,
/tmp/hy3-union-nodefer.log.

# Loopback trunc ladder (2026-08-04 late): the break is PROMPT-DEPENDENT, present at 5 layers

Host-loopback ladder (84-trunc-ladder.sh, full hy3 + block_count override, CPU +
3 loopback workers, default stack, 2 prompts x 6 probes per leg):

- NL=10: both legs internally deterministic ACROSS SERVES (ctl d53bb253 /
  union d5a01673 reproduced run-to-run) but union != ctl (divergent parse
  outcomes both prompts; ctl-p2 steady 4574313f clean 6/6, union-p2 500s 6/6).
- NL=5 (full-file override - kills the stub-vs-override vehicle confound):
  - prompt 1 ("CPU pipeline"): union == ctl == c80261ff 6/6 BOTH - exact,
    matches the historic stub sha.
  - prompt 2 ("The capital of France is"): ctl 6a8f1a83 -> e2f6c6ea steady;
    union e2f6c6ea FIRST (= ctl's steady sha!) -> af6720d6 steady. DIVERGENT.

VERDICT: the chunk-path wrong math is NOT layer-count-gated - it is
prompt/routing-DEPENDENT and already present at 5 layers; more layers just make
more prompts hit it (why trunc20/full-33 look worse). Every historical trunc5
"exact" gate used only prompt 1 - the 5/5 c80261ff verdicts were
prompt-1-blind. The union-p2 first-probe-matches-ctl-then-flips pattern says
the wrong value enters via serve STATE built during the first p2 cycle (build
cache / registry contents), not via the p2 graph itself. ZL/UNION noted-id
consumers audited: pure counters, no math influence.

REPRO OF RECORD (cheapest yet): NL=5, single France prompt, ~29 GB host
loopback, deterministic divergence in 2 probes. Tracer pair banked:
84-n5-tracer.sh (ctl vs union, DEBUG_REDUCE + GGML_META_DEBUG=1 SPLIT_STATE).

## Per-token logit profile (84-n5-logits.sh): CONSTANT PER-DECODE-PASS ERROR

n_probs=20 over 24 tokens, ctl vs union, NL=5, both prompts:

- tok0 (prefill pass): top-20 logprobs BIT-IDENTICAL (max|d| = 0.0) both prompts.
- every decode pass after: constant max|d| ~4e-2..2e-1, NOT growing, both
  prompts, chosen tokens mostly unflipped (pipeline 24/24 same -> its sha
  "exactness" was flip-luck; france flipped at tok23 -> its sha divergence).

REFRAME: the corruption is NOT prompt-dependent and NOT accumulating - EVERY
DECODE-SHAPED chunked graph carries a fixed ~5e-2-logit wrong contribution
while prefill-shaped chunked graphs are exact. At 33 layers the per-pass error
is ~6x larger -> immediate garbage (the fleet observation). Rules out: KV
corruption (delta would grow), fp reassociation (would be ~1e-5), routing
dependence (both prompts equally perturbed).

Next discriminator: BCAST_FUSE=0 pair at NL=5 (84-n5-logits-f0.sh) - delta
gone = fuse/skip/repair machinery under chunking; delta stays = fuse exonerated
on loopback, shrink repro to NL=2 for node-level dissection.

## Minimal repro ladder result: TWO ADJACENT MoE LAYERS

- FUSE=0 pair at NL=5: per-pass delta PERSISTS (france max 1.11) - the whole
  BCAST_FUSE/skip/repair subsystem is exonerated on loopback too.
- NL=2 (1 dense + 1 MoE): BIT-EXACT, 24/24 tokens, both prompts, max|d| 0.0.
- NL=3 (2 MoE layers): per-pass delta returns (max 2.9e-2).

MINIMAL REPRO: two adjacent MoE layers, decode-shaped graphs only (prefill
bit-exact at every NL). The defect is a cross-layer interaction: MoE layer L's
chunk output feeding layer L+1's chunk. Scripts: 84-n{2,3,5}-logits.sh (+
-f0 variant), ~12-29 GB host loopback, ~10-20 min per pair.

NEXT DISCRIMINATOR (build, ~10 lines): a chunk-only eval callback
(LLAMA_CB_CHUNK_ONLY=1 in llama-context.cpp: answer ask=true on ffn_moe_topk-*
but skip the tensor_get read). Reproduces delta = pure graph-chunking bug in
the meta decode path; exact = the callback READS (meta get_tensor gathers
mid-graph) mutate member state. Then node-level diff on the NL=3 graph.
