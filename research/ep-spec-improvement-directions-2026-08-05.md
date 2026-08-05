# Improving speculative decoding for the EP fleet (research synthesis, 2026-08-05)

Question: dspark/MTP spec measured NEGATIVE on the 0731 EP fleet (#80: EP
target-only 4.19 vs EP+dspark n3 3.37). Given how EP works here, what could
make speculation pay? Literature sweep + synthesis against our measured physics.

## Why spec fails HERE and works in big-lab EP serving (the reconciliation)

- Our fleet decode is EXPERT-WEIGHT-bandwidth-bound on RAM members: each verify
  lane multiplies DISTINCT expert reads. Measured union curves: hy3 w4 2.89x,
  V4 w4 2.65x (~0.55-0.60x linear) vs break-even ~2.8 -> verify eats the
  boundary amortization (#52 law; #71 stage-1 ratio 1.00).
- MagicDec's "verification is nearly free" result holds when the dominant
  memory traffic is the KV CACHE - KV loads amortize across draft lanes. Our
  dominant traffic is expert weights, which only amortize when lanes route to
  the SAME experts (8-of-256 routing -> near-disjoint by default).
- DeepSeek/SGLang production EP+MTP (1.8x TPS, 85-90% acceptance) runs HUGE
  batches over many GPUs: expert weights are batch-amortized, so verify rides
  mostly-amortized traffic. Our -np 1-2 has no such amortization.
- CONCLUSION: on this fleet, spec pays only if (a) the verify expert-union
  shrinks, (b) batching amortizes expert reads, or (c) chains stay in the
  concave-cheap part of the union curve.

## Ranked directions (cheapest-to-test first)

1. **NMAX=2 A/B (zero build, one fleet window)**: the union curve is concave -
   w2 costs only 1.62-1.70x reads. At the measured 83-93% acceptance,
   E[tokens/pass] ~ 1.8-1.9 for 1.6-1.7x reads = +10-15% hypothesis. The #80
   5-leg tested n3/n5 but NEVER n2. Protocol: plateau-vs-plateau on the live
   0731 config, NMAX=2 PMIN high (0.8+).
2. **Expert-reuse-aware lane gating (EcoSpec-class, the real build)**:
   published 1.62x on DeepSeek-V3.1-671B/Qwen3-235B by scoring draft
   candidates with predicted marginal expert-activation cost and favoring
   drafts that REUSE the experts already in the verify set (arXiv 2607.12696).
   OUR ADVANTAGE: with LLAMA_META_LOCAL_DRAFT the draft trunk runs
   coordinator-local, so the drafter's router ids per draft token are known
   BEFORE verify - NO eval callback needed (not blocked by the seam bug).
   Build sketch: in the speculative accept/draft loop, after drafting token k
   compute marginal_new_experts(k) from the draft's topk ids vs the chain's
   union; stop the chain when marginal read cost exceeds expected acceptance
   value. Instruments exist (union counters); gates = union/pass drops while
   acceptance holds, then fleet plateau A/B.
3. **Adaptive verification length via drafter confidence (EVICT-class,
   small build)**: lossless truncation of the verify set - drop tail lanes
   whose joint acceptance probability is low before they enlarge the union
   (arXiv 2605.00342). We already measured dspark's confidence head is
   informative (acceptance 45.7->95.7% under threshold sweeps); today conf_min
   is per-request - make chain length per-STEP adaptive.
4. **spec x np>=2 (zero build, unmeasured #71 thread)**: two slots' verify
   unions overlap and expert reads amortize across slots - the MagicDec
   mechanism starts applying to expert weights. np2 alone is +40% aggregate;
   spec on top never measured.
5. **#60 transport (RDMA/NIC)**: cheaper per-lane boundary shifts break-even -
   standing reopen key, hardware-shaped.
6. **Long-context KV term**: MagicDec's regime (KV-bound at long ctx) is
   weakened here by MLA/DSA tiny KV - not a lever for hy3/V4.
7. **Owner-VRAM-routed drafting (stage-2/#75)**: draft-side expert masking to
   VRAM residents is acceptance-filtered (lossless) but draft cost is already
   ~free after LOCAL_DRAFT - does not attack the verify term; keep shelved.

## Suggested order

(1) NMAX=2 fleet A/B and (4) spec x np2 - both zero-build, one window;
then (2) EcoSpec lane gating build if either shows the union economics moving;
(3) rides along as a small patch on the same code path.

Sources: [EcoSpec](https://arxiv.org/abs/2607.12696) ·
[EVICT adaptive verification](https://arxiv.org/html/2605.00342v1) ·
[SGLang large-scale EP + MTP](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/) ·
[SGLang MTP](https://www.lmsys.org/blog/2025-07-17-mtp/) ·
[DSpark](https://arxiv.org/html/2607.05147v1) ·
[MagicDec](https://arxiv.org/pdf/2408.11049) ·
[HyperDFlash](https://arxiv.org/pdf/2606.26744)
