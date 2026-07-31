# #67 research iteration 1 (2026-07-22): parallel decoding x distributed serving

Scope: the user's `research/true-parallel-inference/` papers + a web sweep over
diffusion/two-tower models, MTP-successor retrofits, and the distributed field.
Target hardware: 2x V100 32GB (sm70) + heterogeneous CPU LAN fleet, GbE,
measured bottleneck = per-token boundary LATENCY (~43 sequential hops for EP;
proven insensitive to member bandwidth - adding an 823 GB/s VRAM expert member
changed nothing).

## The named references, resolved

- **"Gemma diffusion" = DiffusionGemma-26B-A4B-it** (Google, 2026-06-10, Apache
  2.0, open weights + unsloth GGUFs). Gemma-4 MoE 25.2B-A3.8B, block-AR
  "multi-canvas" decode (256-token canvases denoised in parallel, KV-cacheable).
  Headline >1,100 t/s is H100 FP8; NO V100/consumer numbers exist anywhere.
  Quality trails AR Gemma 4 (MMLU-Pro 77.6 vs 82.6). llama.cpp support is an
  UNMERGED PR (#24423), diffusion-cli only - no llama-server path.
- **"Nemotron two towers" = Nemotron-Labs-TwoTower-30B-A3B** (NVIDIA, ~2026-07,
  TiDAR lineage: frozen AR context tower + diffusion drafter tower, blocks of
  16). 2.42x at 98.7% AR quality - measured on 2x H100 BF16, ~118 GB weights,
  no ggml path at all.

Verdict on both: architecturally the future, practically not deployable on this
rig today. WATCH (esp. PR #24423 + llama-server support), don't build.

## The user's papers

- **APEX** (2506.03296): train-free CPU-attention offload with deferred sync
  (GPU consumes CPU attention results one iteration late, never stalls). Wins
  +11-96% but assumes PCIe-local CPU; dead on arrival across GbE. Portable
  bits: local-host attention offload for high -np long-context (breaks the
  VRAM KV ceiling), and the analytical placement inequality for our
  auto-weight. Niche.
- **Layer Parallelism** (2502.02790, TMLR 2026): run consecutive layer PAIRS in
  parallel post-hoc; win comes from HALVING sequential sync points (1.19-1.46x
  on TP nodes). Train-free at -1.5-4% accuracy, math/reasoning degrades hard
  (GSM8K collapses; light finetune recovers half). Portable as an offline GGUF
  layer-fusion transform + graph-builder changes; on our fleet, halved depth =
  halved boundary hops, and our hops cost ~1000x theirs. High ceiling, real
  quality risk - prototype AFTER the spec-decode route, gate on ppl + tasks.
- The other two PDFs (2014 parallel Lasso; scikit-learn n_jobs tuning) are
  off-topic - no action.

## The retrofit-decoding lane (the actionable one)

- Upstream's unified spec framework is ALREADY IN OUR TREE since the 6d5a910
  merge: `--spec-type draft-simple | draft-eagle3 | draft-dflash | draft-mtp`
  plus train-free `ngram-cache/simple/map-k/k4v/mod`. EAGLE-3 merged upstream
  2026-06-12 (1.6-3.28x, 40-93% acceptance).
- **A trained EAGLE-3 head for Qwen3.6-27B exists** (Ex0bit/Qwen3.6-27B-PRISM-
  EAGLE3, 0.6B, 1.84-1.97x on SGLang, Apache 2.0) - needs GGUF conversion.
- DeepSeek's DeepSpec (2026-06-27) leads the field (DSpark +27-31% acceptance
  over EAGLE-3; DFlash block-diffusion drafter - our models dir already holds
  dflash-draft-3.6-q8_0.gguf for Qwen3.6).
- Cross-NETWORK speculative decoding is now a published pattern (PicoSpec:
  pipelined draft/verify so the drafter never idles during the round trip;
  DSD/SLED): draft locally, verify remotely in ONE batched round trip.
- MoE caveat: verification batches activate more experts; net spec gains on
  MoE are smaller than dense - measure, don't assume.

## The distributed lane (upstream watch + field)

- **Upstream's meta backend is JohannesGaessler's PR #19378** (merged Apr 2026)
  - local multi-GPU tensor parallel only; RPC stays layer-split; hybrid
  TP+PP-across-nodes is an open request (#23568). Upstream explicitly leaves
  orchestration to external layers: the fleet (discovery, auto-weight, W2W,
  caches, recovery, EP-over-RPC) has NO upstream competitor. Watch: whether
  meta-backend+RPC composition lands upstream - that's the collision risk.
- **prima.cpp** (ICLR'26) is the closest peer: pipelined-ring parallelism +
  Halda placement solver; 5-17x lower TPOT than llama.cpp-RPC. Its two ideas
  worth stealing: ring overlap (device works on multiple in-flight cycles) and
  solver-based placement. Notably its best numbers ALSO come from spec decode.
- **EP-specific**: EPLB-style hot-expert replication (replicate the hottest
  experts on multiple members -> fewer forced hops) and router-lookahead
  prefetch (layer-N router predicts layer-N+1 experts) are productized in vLLM
  and directly adaptable to our EP auto-weight.
- Wire compression of activations helps bandwidth, not RTT count - low priority
  on GbE (RTT-dominated), revisit if #60 RDMA lands.

## Ranked shortlist

1. **Speculative decoding ACROSS the fleet boundary** (chosen first experiment).
   Mechanism: draft on the coordinator (V4's NATIVE MTP head via in-tree
   `--spec-type draft-mtp`; hy3 idem per #51/#52), verify the whole draft chain
   through the fleet in ONE pipeline/EP pass -> tokens-per-boundary-chain
   multiplied by accepted length. Everything needed is in-tree TODAY; zero
   training; composes with EP and layer fleets. Expected: V4 EP 5.15 ->
   ~8-11 t/s at the 85-95% acceptance MTP shows locally (MoE verify caveat
   applies). EXPERIMENT: A/B V4 EP fleet +/- draft-mtp (n-max 3, p-min 0.75),
   coherence-gated; then hy3 MTP=1 rerun (its #52 single-depth-head caveats).
2. **EAGLE-3 head for Qwen3.6-27B** - convert the PRISM head to GGUF, A/B vs
   the 87 t/s MTP baseline single-box; also try dflash-draft (already on disk)
   via draft-dflash. Small effort, possibly beats MTP.
3. **Hot-expert replication + router-lookahead prefetch for EP** - fleet moat
   work; attacks the boundary count itself (skip hops whose experts are
   replicated locally). Medium effort, novel-on-LAN.
4. **Layer Parallelism GGUF transform** - highest ceiling per boundary math,
   real quality risk; prototype on dense 27B with ppl gate after 1-2 land.
5. **prima.cpp-style ring overlap** for the layer fleet - scheduling-only,
   pairs with 1.
6. WATCH: DiffusionGemma llama.cpp server support (PR #24423), TwoTower ggml
   ports, DeepSpec/DSpark heads for our models, upstream meta-backend x RPC.

## Cadence decision

Upstream merge cadence: WEEKLY (17 days of drift cost 31% on V4, #65). Each
merge cycle re-runs the watch list above (#67a).

## Addendum 2026-07-24: research/true-parallel-inference folder review (#71)

Reviewed the remaining unread items in `research/true-parallel-inference/`:

- **Median Selection Subset Aggregation** (`9b81f...-Paper.pdf`, NeurIPS 2014,
  Wang/Peng/Dunson): parallel *statistical* inference - Lasso/GIC feature
  selection fitted per data subset, combined by median inclusion + coefficient
  averaging. "Inference" = parameter estimation; terminology collision, no
  applicability to generative decoding. OFF-TOPIC.
- **Parallel Inference for Real-Time ML Applications** (Al Bayyat et al. 2024):
  sklearn `RandomizedSearchCV(n_jobs=-1)` Random-Forest hyperparameter tuning
  benchmark. OFF-TOPIC.
- **`5-288`**: dead page capture (tracking scripts only, no article body).
- **Defeating Nondeterminism in LLM Inference** (`index.html`, Thinking
  Machines Lab): RELEVANT - batch-invariance. Explains the temp-0 output drift
  we measure across split configs and between spec-verify batches and plain
  decode (reduction order changes with batch shape -> logit drift -> token
  divergence). Consequences adopted:
  1. Byte-identity gates are SHAPE-LOCAL: only compare runs with identical
     split + batch shape; cross-shape drift at temp-0 is expected physics, not
     corruption (nearly mis-diagnosed twice on 2026-07-23).
  2. Batch-invariant kernels are a #71 stage-2+ enabler: bitwise-equal
     verify-batch vs single-token logits would make speculation provably
     identity-preserving, restoring the byte-identity gate for all spec work.
     Re-derivation cost for sm70 unknown; watch for upstream/community
     batch-invariant kernel work before building.

Collection note: filter future paper hauls on "parallel/speculative DECODING"
or "token generation" - "parallel inference" collides with statistics and
classic-ML serving literature.

## Iteration 2 — 2026-07-24 (post-#70/#71-stage-1 refocus)

### (a) Upstream watch (merge base ~Jul 5 -> Jul 24)
- **RPC core stagnant** (nothing merged since May). Issue #25890 (15-min serialized
  535 GB loads) is third-party validation of our caching/manifest moat. WATCH:
  **#24675** (RPC async/events -> pipeline parallelism over RPC - the one PR that
  narrows the gap), **#25818** ("remote speculative decoding via ethernet",
  draft on a separate llama-server - the FIRST upstream distributed-spec
  feature; absorb or differentiate at merge time).
- **Meta backend**: #19378 merged Apr (LOCAL TP only); July hardening (TP+ncmoe
  MoE fix #25028, DSV4 fused ops #25585). No sign of meta+RPC composition.
  Merge-friction risk concentrates in ggml-backend-meta.cpp + spec sidecar
  auto-config churn in common/ (#25811/#25955/#25989).
- **Upstream circles hot-experts single-box**: #25932 (--pin-hotexperts, usage
  tracking + mlock top-N) and #26003 (--lazy-experts, page-cache prefetch of
  routed experts). Read both before building ours; nothing cross-node.
- Spec framework: DSpark drafter PR #25173 (DFlash + semi-AR Markov head,
  60-85% over MTP-1 in DSV4 production; DFlash itself already merged).

### (b) Hot-expert placement — GO, gated on a 1-day profiling counter
Field consensus: balancing losses equalize GLOBAL expert load, not per-layer/
per-domain load ("globally balanced, locally imbalanced"). Measured coverage at
a 25% budget on balanced-trained MoEs: **37-53%** (MoE-Sieve: OLMoE 53%, Qwen-MoE
37.4%, DeepSeek-MoE 42.6%; CRAFT: per-layer peak-to-mean 2.5x-27x on R1/Kimi-K2).
Frequency beats recency (LFU +84% over LRU, CMU 2511.05814); static profiled
placement is the working floor (Fiddler, ktransformers, SlimCaching, Prism).
FOR US: uniform owner slice = 25.5% VRAM hit fraction by construction; frequency
placement projects **37-43% conservative** (+12-18 pp) -> CPU expert bytes/token
drop ~20-25% -> up to ~1.2-1.3x decode on the record config (5.06-5.22 -> ~6-6.7
ceiling). Per-layer top-k frequency ranking suffices for static placement
(submodular-greedy optimal, Prism); uniform per-layer budget first, entropy-
weighted budgets as v2. Risks to verify on-workload: hy3's shared experts may
have absorbed the skew; mixed traffic flattens hot sets. DECISIVE MEASUREMENT
(filed as #74): per-layer router-selection counters, few thousand decode tokens
of representative traffic, compute top-25.5% coverage per layer + cross-domain
stability. An afternoon of counter code, zero placement changes.

### (c) Watch list resolutions
- **MoE x speculation expert-read explosion is now a NAMED problem**: MoE-Spec
  (2602.16052, +10-30% via expert budgeting), EcoSpec (2607.12696, expert-reuse-
  aware draft selection, 1.62x), Utility-Driven SD (2506.20675). Our #71 stage-1
  parity-minus verdict independently reproduced by the field; their mitigations
  are the stage-2 toolbox.
- **Self-Speculative MoE (WWW 2026, 3.72x claimed): draft with a reduced expert
  subset of the SAME model — zero extra weight reads.** Fused with dual-role:
  draft by routing ONLY to VRAM-resident experts (no RAM traffic), verify full.
  Marries hot-expert placement and speculation into one mechanism — adopted as
  the #71 STAGE-2 CANDIDATE (after #74/#75 land, which also make the reduced
  set accurate).
- Batch-invariance: vLLM mode is sm80+ (no V100); llama.cpp PR #16016 (covers
  mul_mat_id) is a maintainer-rejected draft -> fork-carry option if/when we
  want provable spec identity.
- DiffusionGemma: GGUFs exist (unsloth 26B-A4B), llama.cpp still cli-only draft
  (#24423/#24427, "diffusion server" at design stage). TiDAR: still paper-only.
  LLaDA2.X: open diffusion MoEs (16B/100B) with cli-path support.

### Iteration-2 prioritized candidates
1. **#74 expert-frequency profiling** (1 day): router counters + coverage report
   -> gates #75. 2. **#75 hot-expert placement**: per-layer expert-ID scatter
   list replacing the contiguous owner slice (+20-30% decode expected).
3. **#71 stage 2 = draft-on-VRAM-experts self-speculation** (needs #75).
4. Absorb DSpark when merged; track #25818 + #24675 at each weekly merge.

## Addendum 2026-07-24b: Brakel et al., "Model Parallelism on Distributed
## Infrastructure" (arXiv 2403.03699) — folder review

Survey of model parallelism (intra-op vs inter-op taxonomy, auto-parallelisation
search: Alpa/FlexFlow-SOAP/PipeDream/FTPipe/Metis; training-era case studies).
No new runnable technique, two takeaways adopted:
1. Independent confirmation of the fleet laws the guide already states
   (intra-op needs NVLink-class links; inter-op pipelines tolerate Ethernet;
   micro-batching amortizes bubbles).
2. **Auto-parallelisation-as-search is the right frame for #70-tail
   auto-weight v2 ("auto-place")**: today's auto-weight is a greedy
   bandwidth-proportional splitter, blind to owner groups/dual-role/expert
   scatter. Formulate placement (per-device role + byte share + hot-expert
   list) as a search over our MEASURED cost model (worker scores, capacities,
   link RTTs, #74 coverage curves) — Alpa-style two-level, but our fleet is
   small enough for near-exhaustive search. FTPipe's non-adjacent-layer
   assignment also generalizes our layer-mode contiguous slabs, though #61
   showed layer mode is near-physics, so low priority.

## Addendum 2026-07-27: folder re-read after the gate-5 null result (#75/#67)

Gate 5 measured hot-expert placement at -4% vs uniform on the record roster:
the fleet is boundary-cost-bound, not byte-bound. Re-ranking of this folder's
verdicts under that measurement ("more workers -> faster tokens" lens):

- **Layer Parallelism (2502.02790): PROMOTED from parked to candidate.** Its
  entire win is halving sequential sync points; it was reviewed when bytes
  still looked binding and its 1.19-1.46x was priced on ~us NVLink hops. Our
  hops cost ~1000x more, so depth-halving attacks OUR binding constraint
  disproportionately: 80 -> 40 boundaries at ~3.7 ms/boundary is a ~1.9x
  ceiling on the fleet before quality costs. Gates unchanged and mandatory:
  offline GGUF pair-fusion transform, ppl + task suite (GSM8K collapses
  without a light finetune; hy3 is a reasoning model), and #7's timing split
  must first confirm the boundary share. Prototype vehicle: trunc-hy3 pairs.
- **APEX (2506.03296): still DOA across GbE as proposed, but its deferred-sync
  pattern (consume a subordinate result one iteration late; never stall the
  critical path) is the same mechanism as ktransformers' Expert Deferral
  (+33% decode) promoted in the horizontal-scaling addendum. Two independent
  sources on defer-and-overlap upgrade that candidate's confidence.**
- **distributed-llama stands as the existence proof** that more workers CAN
  speed single-stream over plain Ethernet (RPi5 1->4: 5.95 -> 13.68 t/s,
  q80 sync, star, similar-speed nodes). Its conditions map to our roadmap:
  similar-speed nodes := don't put stragglers on the critical path (replica/
  class routing for the slow boxes), q80 sync := wire activation quant
  (steal #7), star := already our reduce shape.
- Brakel/TML verdicts unchanged (frame + gate methodology respectively).

Net: the folder's contribution to "more workers -> faster token generation"
is mechanism (b) of the three escapes - FEWER boundaries (Layer Parallelism)
- plus corroboration for (c) fill-the-bubble (APEX pattern -> Expert
Deferral). Mechanism (a), cheaper boundaries, lives in the horizontal doc
(soft-RoCE/UCCL). All three wait on #7's compute/reduce/wire split.

## Addendum 2026-07-30: independent verification sweep - the MoE-speculation
## and block-decode paper haul (feeds TASKS #84/#85/#86; synthesis lives in
## fill-the-bubble-plan 2.5)

Four-agent sweep run as an adversarial check on the #71 stage-1/stage-2
verdicts plus the first enumeration of the block/parallel-decode family
(Jacobi/Medusa/CLLM/lookahead had zero prior hits in this ledger). Verdict
summary is in fill-the-bubble-plan 2.5; this addendum is the citation bank.

### MoE x speculation (verify economics - the #52-law literature)

- EcoSpec, arXiv 2607.12696 - expert-reuse-aware draft/lane selection
  (score candidates by marginal new-expert cost vs a running buffer);
  1.36x batch-1 vs EAGLE-3 1.22x on Qwen3-235B; states the dense-vs-MoE
  verify distinction verbatim. = our "tree-verify with expert-reuse-aware
  selection" reopen key, published.
- MoE-Spec, arXiv 2602.16052 - budgeted verify: cap per-layer verify union
  to top-B by tree-aggregated router prob, truncate/substitute the rest;
  -1.4% acceptance, quality within noise, +10-30% t/s, training-free.
- EVICT, arXiv 2605.00342 - utility-optimal draft-tree truncation before
  verify: -74.7% verified tokens, -32.5% activated experts, LOSSLESS,
  avg 1.21x over EAGLE-3.
- MoESD, arXiv 2505.19645 - formalizes SD-net-negative in memory-bound MoE
  (expert-union growth with n); independent reproduction of our ratio 1.00.
- Cascade / Utility-Driven SD for MoE, arXiv 2506.20675 - measures 2-3x
  verify-time growth vs dense, 1.5x SLOWDOWNS from naive SD; fix = dynamic
  spec on/off + K tuning (limits loss to 5%).
- DraftExpert, arXiv 2607.24434 - trained per-layer resident draft expert
  lifts tiny-footprint drafts to 84-87% acceptance; naive drafting sits at
  22-46% (= our 0.59 leg's band).
- SpecMoE, arXiv 2604.10152 (DAC'26) - restricted-expert self-draft (N=4 of
  128 pinned) reaches ~85-88% per-token acceptance; skew table: NLLB 0.84
  -> 4.3x, Mixtral 0.32 -> 2.17x. Our coverage@25.5%=0.913 exceeds their
  best testbed. Caveat: batch<=4 loses to plain hot-expert caching - the
  coalescing win is a large-batch effect.
- SS-MoE (WWW 2026, DOI 10.1145/3774904.3792218) - confidence-gated
  accept-WITHOUT-verify on a restricted-expert self-draft; 3.72x claimed
  "nearly lossless". PAYWALLED - unverified beyond abstract.
- Revisiting Lossy Verification, arXiv 2607.26627 - lossy acceptance damage
  grows with task hardness (+0.38pp GSM8K -> +6.67pp AIME): the standing
  risk law for every skip/relaxed scheme; PPL cannot see it.
- Cross-lane overlap data: Cohere MoE+SD blog (8-of-128: 30-38%
  adjacent-lane overlap; 4-lane verify ~2.5x unique experts, not 4x);
  MoE-Infinity 2401.14361 + ST-MoE prefetch 2606.15453 (~2x-over-
  independence consecutive-token reuse). No published number at 8-of-256 -
  our #84 counter would be first.
- OEA, arXiv 2511.02237 - training-free batch-aware RE-ROUTING (tokens
  piggyback on experts already loaded for the batch): 15-39% latency cut,
  no significant accuracy loss; candidate union-compressor for any
  multi-lane pass (lossy-gate discipline applies).
- Speculative MoE, arXiv 2503.04398 - token->expert route PREDICTION
  pre-gating (89% / 96.3% static top-k precision) + co-activation-clustered
  placement; gains grow on slow interconnects (2.34-4.3x PCIe/UPI vs
  1.49-1.72x NVLink) -> #86 recon.
- Dead ends confirmed: SpecExec 2406.02532 / Sequoia 2402.12374 tree sizes
  are DENSE-offload economics (verify lanes reuse the same weights - the
  amortization that does not exist in sparse MoE); exact residual/delta
  verify for expert-subset drafts exists nowhere and is structurally
  unsound (hidden-state divergence from the first restricted layer).

### Block / parallel decode (stage-3 family - first enumeration)

- CLP, arXiv 2606.10935 - verification-free MTP emission: ~5K-param linear
  gate on backbone state accepts k MTP tokens directly, NO verify batch;
  1.14-1.29x on dense 7B. The one published shape that fits this fleet
  (-> #85, with the KV-catch-up caveat: accepted tokens still need an
  n-lane trunk ingest, so it prices as spec-at-100%-acceptance).
- Judge Decoding (ICLR 2025) + SelfJudge 2510.02329 - relaxed/judge
  acceptance via tiny probe: more accepted tokens per verify pass at zero
  extra reads; cheapest stack-on found. Same lossy-risk law.
- MARS, arXiv 2604.07023 - multi-token emission via continued training of
  the SAME model (no heads): 1.5-1.7x, needs a finetune we cannot afford.
- Set Block Decoding, arXiv 2509.04185 + Fast-dLLM v2, arXiv 2509.26328 -
  cheapest credible AR->block conversion (~1B-token finetune, 3-5x fewer
  passes at parity, exact KV). PARKED: finetune budget + no MoE-scale
  replication.
- Jacobi family: Lookahead (ICML'24, 56-120x extra FLOPs/step - fatal
  here), CLLMs 2403.00835, Jacobi Forcing 2512.14681 (best-in-family,
  3.8-4x, but distillation finetune + compute-for-latency by design).
  RULED OUT for this fleet.
- Medusa 2401.10774 / Jakiro 2502.06282 - tree verify = maximal expert
  union. RULED OUT.
- Diffusion: LLaDA-MoE 2509.24389, LLaDA2.0 2512.15745, RND1 (AR->diffusion
  conversion of Qwen3-30B-A3B: 500B tokens, still below AR quality) - no
  cheap conversion path exists mid-2026; APD 2506.00413 (dLLM + tiny-AR
  mixture accept rule) is the acceptance primitive to remember IF a
  block-capable model ever lands.
- Production MTP serving datum: SGLang MTP (lmsys 2025-07-17) caps draft at
  2-4 tokens (accept len 2.18-2.44, +60% single-node, +14% at scale); no
  per-rank expert-read cost model published - the gap our counter fills.
