# Layer-Parallel pair fusion (TASKS #71, escape (b): fewer boundaries)

Status: design v1, 2026-07-30, written on the day the ceiling probe passed.
Sources: 2502.02790 (TMLR 2026) via docs/research/2026-07-parallel-decoding-
and-distribution.md (2026-07-27 addendum re-rank); measured frame from
fill-the-bubble-plan 2.2c and the 2026-07-30 nulls.

## 1. Why now - the measured case

Three results this week point every remaining watt at boundary COUNT:

1. **Member lateness is pipeline phase lag, not weight.** The leg-skip
   fleet A/B shipped 1-byte replies instead of 57 KB payloads and moved
   NOTHING (4.61 vs 4.97-5.21, defer rate unchanged) - a wire member
   executes its fused-chain pieces in order and reaches each fetch a
   boundary behind the owner. Payload size, routed bytes (placement x2),
   and thread count (sweep) are all falsified levers.
2. **The 40-layer ceiling probe is GO: ~9.3 t/s = ~1.8x** the 4.97-5.21
   plateau (defers halved to 72.5/graph, LOST 0). Halving the lap count
   nearly doubles decode.
3. Deferral v3 (+34%) already harvests the overlap available WITHIN the
   current boundary structure; its WAIT_US=3000 leg showed the residual
   wait is structural. Fewer laps is the axis deferral cannot reach.

Honest LP ceiling: the probe halves compute AND boundaries; LP pairing
halves boundaries only (members compute BOTH layers' experts per cycle).
From the #7 split (~250 ms token = ~110 compute + ~120-140 boundary at
4.0 t/s; proportionally ~192 ms at today's 5.2), removing half the
boundary term prices LP at **~+45-55% decode** before quality costs -
on top of v3, with which it composes (deferral still applies to the
remaining boundaries).

## 2. Mechanism

The paper's transform, translated to this fleet's graph:

    sequential (today):   x1 = x0 + attn_a(x0) + ffn_a(x0 + attn_a(x0))
                          x2 = x1 + attn_b(x1) + ffn_b(x1 + attn_b(x1))
    paired (LP):          x2 = x0 + [attn_a(x0) + ffn_a(...)]
                              + [attn_b(x0) + ffn_b(...)]

Both layers of a pair consume the SAME residual input; their deltas add.
Weights are untouched - this is pure dataflow, so it is a BUILDER-side
rewrite (llama-graph), not a GGUF transform. No new proto, no meta
backend surgery:

- **One B1 per pair**: the pair's shared input broadcasts once. Members
  compute both layers' routed experts from it (2x k routed ids; layer
  b's router sees x0 instead of x1 - an inherent, priced part of the
  paper's accuracy cost) and **sum the two FFN contributions member-side
  into ONE boundary tensor** - one B2 gather per pair, same payload
  size as today. 79 expert boundaries -> ~40.
- **Attention pairs on the owner**: attn_a and attn_b both read x0 and
  can run concurrently on the owner GPUs (independent KV per layer,
  unchanged). The delayed-reduce walker already crosses the mirrored
  residual ADDs (BCAST_FUSE machinery); the pair ADD tree is the same
  shape one level deeper.
- **Deferral composes**: the paired boundaries remain star reduces;
  EXPERT_DEFER=1 keeps its decode-only readiness gating on them.

## 3. Build plan (increments, each gated)

1. **inc-0 builder rewire behind `LLAMA_LP_PAIRS` (default off, =N pairs
   depth-2 only):** llama-graph builds pair blocks for the middle layers;
   `LLAMA_LP_SYNC_EDGE=k` keeps the first/last k layers sequential (the
   deferral sweep's edge-protection instinct; the paper also finds deep/
   shallow layers tolerate pairing worst). Off = byte-identical (gate).
2. **inc-1 stub bring-up:** trunc-hy3 loopback, LP=on - correctness is
   "builds, runs, deterministic, finite", NOT byte-identity (the math
   changes by design). Boundary count per graph via BOUNDARY_STATS is
   the engagement instrument (star reduces ~halve).
3. **inc-2 quality gates on the record roster (the REAL gates):**
   - fleet PPL wikitext 8x -c 512 vs baseline family 3.7669-3.7950
     (paper prices -1.5-4% overall quality; PPL should move but
     modestly);
   - REASONING reads: hy3 is a reasoning model and GSM8K-class
     collapse is the documented failure mode - the coherence gate for
     LP must include math/multi-step prompts, not just prose;
   - sweep LLAMA_LP_SYNC_EDGE (0/4/8) and pair only middle layers.
4. **inc-3 fleet t/s A/B:** LP=on + EXPERT_DEFER=1 vs today's 4.97-5.21
   plateau; target >= +30% to justify the quality tax; also re-measure
   -np 2 aggregate (fewer boundaries shrink the bubble the second
   stream fills - aggregate gain may compress).

## 4. Risks and falsifiers

- **Reasoning collapse** (the paper's GSM8K result) even at edge-
  protected middle-layer pairing: the lane dies without a finetune,
  which is out of scope - fall back to transport (#60) and V4 fleet
  levers.
- Routing drift: layer b's router on x0 changes expert selection; on a
  192-expert 8-of model the paper's tolerance may not transfer -
  watch PPL first, it is the cheap early warning.
- KV correctness: attn_b consumes x0 but writes its own KV as today;
  positions/streams unchanged. Prefill pairs identically (no decode-only
  gate needed for exactness - LP is uniformly lossy by design, which is
  why the quality gates carry the whole decision).
- Owner VRAM: pair blocks double peak attention concurrency on the
  owner; hy3's owner group (0,1) interleaves layers - pairing must keep
  a pair's two layers on the SAME owner or pay a cross-GPU hop (pair
  assignment follows `owners[pair % n]`).

## 5. Prior art (one line each)

- 2502.02790 (TMLR 2026): the transform itself, 1.19-1.46x on NVLink
  TP; our hops cost ~1000x more, hence the outsized ceiling here.
- METRO-style member-skip reduce + the #75 masks: orthogonal, parked.
- distributed-llama: the existence proof that plain-Ethernet fleets can
  scale single-stream when sync points are few and cheap.
