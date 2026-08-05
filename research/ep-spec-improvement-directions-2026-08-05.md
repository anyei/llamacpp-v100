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

## NMAX=2 A/B RESULT (2026-08-05): NEGATIVE - direction 1 closed

Live 0731 serve, same shares/env both legs, 8x200-token probes, plateau reads:
- Leg A target-only: 4.35 t/s mean (4.15-4.50, 8/8 clean).
- Leg B dspark NMAX=2 PMIN=0.8 draft@CUDA0 LOCAL_DRAFT: 3.88 t/s mean over the
  6 valid runs (3.47-4.52) = -11%. Acceptance 60-65%, mean len 2.2 - the
  concave-union hypothesis needed >=80% acceptance; real chat sits at ~62%
  even at p_min 0.8. Chain-length tuning cannot rescue spec on this fleet.
- VERDICT: #80's EP-target-only keeper STANDS at every n_max. The live spec
  path is #107 (expert-reuse-aware lane gating) which attacks the union term
  itself, optionally composed with #108 per-step confidence adaptivity.
- Bonus capture: a #104 drop struck mid-leg-B with FIN pollers armed - the
  coordinator's ephemeral probe FINs sit in FIN-WAIT-2 locally while .11 holds
  the mirror CLOSE-WAITs for 10+ minutes -> the worker does NOT close probed
  connections on client FIN. Suspect chain: ephemeral-probe CLOSE-WAIT/fd
  accumulation (worker soft limit 1024) -> fd pressure -> compute-connection
  failure ~hourly (the #39 fd-leak class, again). fd-trend poller armed on .11
  (/tmp/fdtrend-11.log) - the accumulation curve decides.

## Post-roll validation audit (2026-08-05 morning)

- #104 fix: deployed worker binary md5-IDENTICAL to the fixed image on local
  and .11; fd trend on .11 flat at 6 fds / 0 CLOSE_WAIT for ~4h post-roll;
  ZERO 'failed in ggml_backend_rpc' drops in 90+ min of serving (pre-fix
  cadence ~1 per 30 min). Coordinator image carries BSUM + CB_CHUNK_ONLY
  (grep on /app/libggml-base.so.0 + libllama.so.0). NOTE: aggregate
  /fleet/status probe latency still alternates ~2-3s (uncached) vs ~2ms
  (60s cache) - consistent with .15 still running the OLD worker image and
  queueing its probe on the exec lock; TESTABLE PREDICTION: sub-second
  uncached probes once .15 is recreated on rpc-worker-aea8de1da.
- Fleet Unload button: headless-verified on the LIVE launcher (puppeteer +
  chromium): button present via textContent match, click opens the
  confirmation dialog with the correct model name and copy, Cancel path
  leaves the serve 'loaded', 0 JS errors. Confirm path intentionally NOT
  exercised (would stop the user's serve).
