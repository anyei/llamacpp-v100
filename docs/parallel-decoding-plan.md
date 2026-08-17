# Parallel Decoding Plan (TASKS #127)

Status: OPEN 2026-08-14 - decisions (a)/(c)/(d) taken same day: UD-Q4_K_XL
in-box = spine A endgame (rung A6), #108 PEARL build APPROVED (#107 still
lands first), GLM Q4-467 PARKED. Goal (user, 2026-08-14): 20 t/s minimum,
30 = happy, on the over-VRAM spines - the ~160 GB DeepSeek-V4-Flash production
checkpoint and GLM-5.2 (Q2_K_XL 243.6 GB in-box / UD-Q4_K_XL 467 GB
fleet-only). This doc owns the seven-paper parallel-decoding
study (research/true-parallel-inference/) and the ladder that spends its
findings. Companions: docs/expert-parallel-plan.md (EP machinery),
docs/expert-placement-plan.md (#74/#75), TASKS #107/#108 (spec-stack vehicles
the papers map onto), #60 (RoCE), #28 (W2W), #31 (fleet laws), #110 (quality
battery protocol).

## 0. Frame

"True parallel inference" = parallelism whose units do not talk during decode.
Three dimensions, in the order they pay on this rig:

1. **Time-parallel** (draft-then-verify): the drafter materializes future
   tokens so the target verifies them in one pass. Already production machinery
   (MTP/NextN #124, dspark, dflash). The papers upgrade THIS dimension:
   draft/verify overlap + adaptive length (PEARL -> #108) and entropy gating
   (Cerberus -> #107).
2. **Sequence-parallel** (-np batching): boundary and weight-read cost is
   per-graph, not per-sequence. Measured EP curve 1.66 -> 4.73 t/s at B=1..8 on
   RAM experts (old roster). Zero build cost, throughput-mode lever.
3. **Space-parallel** (EP/tensor split): pays in-box or behind cheap
   boundaries only. 2026-08-13 decomposition on the 5-member placed-EP roster:
   compute 190-198 ms/graph = 88% (CPU member expert term), reduce 14 ms over
   43 boundaries (0.33 ms each), attn-bcast 7.4 ms. The wire theory is
   FALSIFIED on this topology - the enemy is the DDR4/CPU expert term, and the
   lever is placement/coverage, not boundary count, until the CPU term falls
   under ~25 ms/token.

Non-negotiable gates carry over unchanged: coherence eyeball before trusting
any t/s (coherence-gate memory), BOUNDARY_STATS/defers==injects structural
pass, byte gate c80261ff for hot-path code changes, PPL family for math
changes, leg protocol per the fleet-measure skill.

## 1. Paper ledger (all PDFs in research/true-parallel-inference/)

| Paper | Verdict | What we take |
|---|---|---|
| 2601.09921 QEC parallel window decoding | FRAME | The body concept: parallel units, zero inter-unit communication at inference, consistency by construction, outputs combine trivially; throughput scales by stacking units at constant latency. LLM-legal implementations = draft-then-verify (time) and batching (sequence). Its core enabler is TRAINED-IN self-coordination - out of reach for a serving fork (see non-goals). |
| 2403.03699 model-parallelism survey | DIAGNOSIS | Intra-operator parallelism is only ever run over NVLink-class links (Megatron 8-way in-node, PaLM TPU pods); nobody runs it over Ethernet. Names the quadrant our TCP wire-EP lived in. Supports: in-box shapes first, wire members only behind cheap boundaries (already measured cheap on the current roster) or RoCE (#60). |
| 2506.03296 APEX CPU-GPU overlap | DISCIPLINE | (a) Deferred one-step sync: consume the slow unit's layer-i result only when the fast unit needs it next iteration - independent confirmation of GGML_META_EXPERT_DEFER (+34% loopback; P3b wire generalization when wire members matter again). (b) Pay-gate inequality: offload only when a profiled analytical condition holds (their CPU attention <10% of GPU speed = pipelining rarely pays; async overlap instead). Template for ANY slow-member decision here. (c) CPU KV+attention annex under VRAM pressure = a throughput-mode candidate, parked behind a pay-gate (section 8). |
| 2408.11850 PEARL (ICLR'25) | ADOPT -> #108 | Training-free, LOSSLESS (standard speculative verification - byte-gate compatible at temp 0). pre-verify: target forwards the bare prefix concurrently with drafting, verifying draft token 1 early; reject -> skip the whole verify pass. post-verify: drafter keeps drafting the next window during target verify; full accept -> skip the next draft phase. Together: drafter and target busy at every timestamp, draft length adapts (segmented drafting). Evidence: MAT 5.3-8.3 -> 26.5-39.9, up to 1.50x over vanilla SD, 4.43x over AR (A100s, compute-adequate - temper for our memory-bound regime). Ablation: post-verify dominates when acceptance is high, pre-verify saves the wasted verify when acceptance is low - both regimes exist here (85.2% EP+dspark real workload vs 55-62% temp-0 probes). gamma ~= round(c), c = target/draft speed ratio. |
| 2410.13344 Cerberus | GATE ONLY -> #107 | Their trained heads we do not need (NextN/MTP heads exist: V4 #124, GLM/hy3 native). The transferable piece: entropy-based gating - skip or shrink speculation on high-entropy steps. Their measurement: 18.38% of Medusa speculation steps accept ZERO tokens (up to 24.84% on hard tasks) = pure waste; entropy of the last hidden state negatively correlates with accepted count. Their gate ablation is modest (+2.4% t/s on A100), but our failed speculation costs union reads on DDR4 (w2 = 1.62-1.70x reads), so the gate's value scales up here. Threshold is manual per model/task (their stated limitation) - treat as a tunable, sweep it. |
| 2405.15208 LUD | REJECT as-is | Requires continual training (PAD-token data reconfiguration) - a train-time capability, not a serving feature. Salvage: (a) accepted-span distribution is heavily 1-2 tokens (~67-75% single) - independent confirmation that small n_max captures most of the mass; (b) beta-confidence span acceptance is LOSSY (their code quality -3%) - allowed only as an explicit opt-in behind the #110 KL battery; (c) repetition-halt guard (stop accepting when x_i repeats/extends x_{i-1}) - cheap, borrow freely; (d) code decodes far more parallel than prose - expect task-dependent acceptance. |
| 12863 ParaDecode (ICLR'25 sub, anonymous; file 12863_Fast_and_Accurate_Langua.pdf) | REJECT as-is | Trains lightweight intermediate-layer LM heads (16M/head, T^(i) reusing the frozen final head, KL-distilled, backbone frozen, ~15K samples) to emit tokens early; the next token's early layers run as batched matmuls alongside the current token's remaining KV-fill layers; final-layer verification guarantees parity. 1.15-1.53x on A100/H100-class compute-bound rigs. Train-time capability -> not a serving add (LUD family). Salvage: (a) verification is NON-NEGOTIABLE - their ablation drops consistency 97.2% -> 23.8% without it (independent proof for the coherence-gate discipline); (b) untrained early exit = 0.863x SLOWDOWN, and their training-free Self-SpecDecode baseline (layer-skip self-drafting) = only 1.03-1.14x plus per-task Bayesian search - closes the layer-skip-drafter lane here with the paper's own numbers; (c) gamma=0 (always early-predict) was their fastest setting once heads were calibrated - keep the #107 gate simple before investing in threshold sweeps. |
| 2502.02790 LP pair-fusion | CLOSED (prior) | docs/lp-pair-fusion-plan.md - quality-catastrophic in-fork; stands. |

## 2. Mapping into filed tasks

- **#108 (per-step adaptive chain) absorbs PEARL.** Scope: restructure the
  server speculative loop so drafter and target run concurrently - pre-verify
  (target forward on prefix during drafting; on reject, skip verify and
  redraft) + post-verify (drafter continues past the window during verify;
  on full accept, chain windows without a draft stall). Both are lossless =
  the temp-0 byte leg must hold vs the serialized loop. Adaptive length falls
  out (no fixed n_max stall). Implementation notes: drafter already
  device-pinned (LOCAL_DRAFT / --spec-draft-device); needs an async draft
  thread + draft-KV rewind past unverified windows (rewind machinery exists in
  the spec path); gamma init from measured c, then acceptance-driven.
- **#107 (union-aware lane gating) absorbs the Cerberus gate.** Signal:
  drafter logit entropy (available at the coordinator, zero extra forwards) or
  last-hidden entropy; above threshold -> target-only step (no draft, no
  union verify read). Directly attacks the 38-45% temp-0 rejection regime that
  made #80 go target-only. Sweep the threshold on the real serve; keep the
  kill switch.
- **Order of operations**: #107's gate is small and independent - it can land
  first and de-risk re-enabling spec on rosters where #80 said target-only.
  #108 is the structural win on top.

## 3. Spine A - V4-Flash ~160 GB, target 20-30 t/s

Where the lane stands (2026-08-14, next-steps memory): placed NATIVE-profile
fleet-EP serve (CUDA0,CUDA1,CUDA2 + RPC0 worker V100 + CPU, -ts 15,15,15,18,37)
launching; projection 9-11 t/s vs the 5.76-5.86 measured baseline; MTP drafter
(#124 head) queued to stack after. Decomposition says 88% of the token is the
CPU member's expert term.

Ladder (each rung: coherence -> structural counters -> 3+ run legs vs the
named baseline; spec rungs add the temp-0 spec-vs-nospec byte leg):

- **A1 (queued, zero build)**: measure the placed NATIVE serve. Gate the 9-11
  projection. This is the plan's calibration point - it fixes the real CPU
  read share and the residual per-stage budget.
  **MEASURED 2026-08-14: 5.3-5.5 t/s - projection MISS, placement null #3.**
  Two 8-run legs (fresh 5.25-5.38 mean 5.34; plateau at ~35 min 5.31-5.49
  mean 5.35 - no warm-up drift, load was cache-warm in 4.5 min), 100-tok
  greedy, cache_prompt false, config = the user's roster (-ts 15,15,15,18,37,
  -c 32768) + full EP env + v4-0731-native-place-15-15-15-18-37.json.
  Coherence PASS (74.63 km/h reasoning probe exact; prose clean). Engagement
  PROVEN: all 12 env vars read off the child process, and placement load is
  throw-on-mismatch (llama.cpp:443ff) - a decoding serve means the artifact
  was validated + installed. vs the 5.76-5.86 baseline (ctx 8192): parity
  minus the ctx delta. CONSEQUENCE: native placement joins hy3's two nulls -
  routing skew on V4-0731 (proxy Cov@25.5% 0.583) is too soft for placement
  to move the CPU term; levers are now A2 coverage, A3 spec, A6 Q4-in-box.
  INSTRUMENT GAP: launcher-spawned serves swallow child stderr at default
  verbosity (LOG() forward filtered) - DEFER/BOUNDARY counter gates need a
  script-launched serve or a launcher verbosity bump; A1 ran on coherence +
  legs only.
- **A2 (zero build)**: roster A/B - in-box only (drop RPC0: CUDA0-2 + CPU
  eplocal) vs A1's 5-member shape. Decomposition says the wire is cheap
  (0.33 ms/reduce), so the worker's +32 GB coverage may WIN despite the wire;
  the in-box leg quantifies exactly what the worker buys. Also: KV q8_0 and
  context trim legs to grow GPU expert shares (coverage is the only V4 lever
  that moves the 88% term).
  **MEASURED 2026-08-14 (ctx 8192, 8-run legs, coherence PASS on every leg):**
  ctx-parity 5-member placed EP = 5.83-6.15, mean 6.03 (A1's 5.35 was the
  ctx-32768 tax, +12.7% recovered; placement speed-null re-confirmed vs the
  5.76-5.86 old-artifact baseline). **In-box layer+ncmoe (-ncmoe 28,
  -ts 29,7,7, classic path, zero meta machinery) = 9.43-9.82, mean 9.64 -
  the shape verdict: +60% over the best EP shape.** In-box eplocal (EP CPU
  member, placed 15,15,15,55) = 5.49-5.66, mean 5.60. The ncmoe/eplocal
  ratio 1.72x reproduces the IQ2XXS precedent (11.3/7.0 = 1.61x):
  **the meta-EP member machinery costs ~1.7x on single-stream decode vs the
  classic resident-CPU-expert path, even with 65% vs 37% of expert mass on
  DDR4.** The worker's +32GB coverage only recovers fleet-EP to 6.03.
  Consequences: A6's serving shape = ncmoe (not eplocal) pending a -np
  ranking check; A3 rides the ncmoe shape. Dead levers, with mechanism:
  GPU share raise OOMs (17% shares -> one member gets a ~32GiB buffer,
  share-asymmetry class), and the placement artifact is speed-null but
  LOAD-BEARING for fit (no-placement loads at the proven 15% shares also
  OOM one member at ~32GiB - the rotation-pin remainder). VRAM at
  -ncmoe 28: 17.1/25.7/23.2 GiB used -> tuning leg -ncmoe 24 -ts 27,8,8
  (4 more GPU expert layers, GPU0 keeps drafter headroom), est ~10.
- **A3 (queued)**: MTP drafter stack on the winner (--spec-type draft-mtp,
  #124 head, NMAX=2 first). The 2026-08-05 NMAX=2 NEGATIVE (3.88 vs 4.35,
  62% temp-0 acceptance, dspark, old roster) is NOT a prior for this leg:
  different drafter (native MTP head), placed roster, and real-workload
  acceptance measured far higher (85.2%). Measure TRUE acceptance + temp-0
  byte identity.
  **MEASURED 2026-08-14 on the tuned ncmoe shape (target-only ref: -ncmoe 24
  = 9.82-10.00, mean 9.95; VRAM 23.0/29.2/26.7):**
  (a) **draft-mtp #124 head, n-max 2: NEGATIVE** - acceptance 39% (30/76,
  deterministic at temp 0), 8.94 mean = -10% vs target-only. The native
  head is weak at depth 2 on this checkpoint.
  (b) **draft-dspark Q8, n-max 3, conf-min 0.3 (needs -ncmoe 25 - the
  10.4 GiB drafter + -ncmoe 24 OOMs GPU0; MTP head 5.7 GiB fits): 9.60-11.35,
  mean 10.12, probe acceptance 53-70%.** Parity-plus on the temp-0 probe;
  by the established probe-vs-real pattern (55% -> 85%) real-workload
  expectation is 11-13. Keeper decision = user's real usage.
  (c) **Byte-identity spec-vs-nospec FAILS BENIGNLY**: divergence is a
  mid-sentence near-tie fork ("to efficiency" vs "to the efficiency of a
  refrigerator", both coherent) - the fp-batch-shape class (verify batches
  hit different kernels; argmax flips on near-ties). Spec byte gates cannot
  be like-vs-like across batch shapes on this stack; #110 KL battery is the
  quality arbiter if suspicion arises. conf-min 0.3 is already a crude #107
  gate; the entropy-gate build refines exactly this knob.
- **A4 (build)**: **#107 increment 1 LANDED 2026-08-14**: `--spec-draft-entropy-max`
  (Cerberus-class confidence gate; entropy in bits over the draft candidate
  distribution, 0 = off). Sites: draft-simple/eagle3/mtp loops; dflash/dspark
  keep their trained confidence head; inert under pad_drafts (same contract
  as p_min). Gates green: canonical off-leg c80261ff stable, engagement
  draft_n 63->0 @ 0.05 / unchanged @ 30 on the trunc-mtp stub (no-pad).
  **REAL-SERVE RESULTS 2026-08-14 (image 4e19f362d rolled to the X99
  launcher; keeper baseline on it = 9.98 mean, reproduces 10.12-class):**
  (a) **#107 entropy gate on draft-mtp (n2, no-pad, entropy-max 2.5):
  8.94 -> 9.28 mean, acceptance 39% -> 44%, drafts/100tok 76 -> 63** - the
  gate works exactly as designed (trims wasted drafts, recovers speed),
  but draft-mtp stays below the 9.95 target-only: DSpark (10.12) keeps
  the crown; entropy gate = the tool for rescuing weak drafters, not for
  beating a strong one. (b) **#108 PEARL post-verify is STRUCTURALLY
  limited to independent drafters (draft-simple)**: on the DSpark serve
  it produced "Invalid input batch" 500s - feature-conditioned drafters
  (dspark/dflash/eagle3/mtp) consume the target's hidden states, which do
  not exist yet for the next window; draft-ahead is impossible in
  principle for them. Type-gate added (PEARL requires draft-simple in the
  spec types). The PEARL piece that fits feature drafters is the
  PRE-verify arm (overlap the target's next forward with drafting) = the
  #108 v2 direction. Production impact today: none (opt-in env, default
  off); the fork's spec stack keeps DSpark + conf-min as the production
  configuration. Original A4 text follows:
  #107 gate, then #108 PEARL loop, A/B'd separately on the A3
  serve. Expected from paper evidence tempered to a memory-bound target:
  +10-25% over the serialized spec loop at >=75% acceptance, plus recovery of
  most zero-accept step waste at low acceptance. Anything beyond that is
  upside, not plan-of-record.
- **A5 (zero build)**: -np 2-4 batching curve on the best single-stream shape
  (throughput mode; the old 1.66->4.73 curve steepens as members leave the
  RAM-bound regime).
  **PREVIEW MEASURED 2026-08-14 (dev box, X99 off; RAM-bound MoE vehicle =
  Qwen3.6-35B-A3B Q4, GPU attention + --cpu-moe experts):** np1 27.5; np2
  aggregate FLAT 27.5-27.9 (per-stream 13.8 = time-slicing, hybrid-vehicle
  scheduling quirk); **np4 aggregate 84.7-85.8 = 3.1x at -22% per-stream**.
  Sequence-parallel amortization of CPU expert reads is real and large ->
  the keeper (~10 single-stream) extrapolates to ~30 aggregate at np4:
  the 20-30 goal in aggregate terms (decision (b)) is plausibly in reach
  today. V4-arch curve (256-expert union, skewed routing) = first leg when
  the X99 returns. Harness /work/np-curve.sh.
  **A5-ON-V4 MEASURED 2026-08-16 (X99, keeper shape -ncmoe 25 -ts 27,8,8
  -c 8192 --parallel 4, image 5091efd76-era): TARGET-ONLY np1 10.11 (4 runs,
  10.02-10.15) / np2 ~14.6 aggregate (1.44x) / np4 19.3-19.9 aggregate mean
  ~19.6 = 1.94x at -52% per-stream, zero failures, coherence PASS** — the
  20-goal minimum is effectively reached in aggregate terms on the 162GB
  production checkpoint, single box. **SPEC×BATCHING INTERACTION (the
  headline mechanism): with DSpark ON the same np4 collapses to ~12.4
  aggregate (= np2-class) AND 1-2 rounds per 5 die with 500s** — CUDA0
  compute-graph OOM (~1.1GiB gallocr reserve at 4-wide prefill; drafter's
  10.4GiB + zero headroom; `[spec] failed to measure draft model memory`
  at load = the reserve shortfall; -ub 512 is default = no-op, fragmentation
  keeps it intermittent even at 1271MiB free). Speculation is a single-stream
  lever, batching an aggregate lever - they do not compose on this shape.
  Production menu: np1+DSpark 11.56 single / np4 target-only ~19.6 aggregate;
  ADAPTIVE-under-load A/B = the candidate bridge.
  **#108 v2 NOTE (same day): the pre-verify arm is killed by paper-math on
  memory-bound targets** - a decode pass costs ~one weight sweep regardless
  of position count, so splitting the (n+1)-pass into 1 + n doubles target
  reads on accepted rounds; spec levers on this rig are acceptance-side:
  LLAMA_SPEC_ADAPTIVE=1 (in-tree, default off - zero-build A/B next X99
  window at higher n-max), span acceptance (#110-gated), better drafters.
- **A6 (DECIDED 2026-08-14 - the spine A endgame, the 20-30 carrier)**:
  UD-Q4_K_XL 144.5 GiB served in-box on the X99 - eplocal
  CUDA0,CUDA1,CUDA2,CPU, no wire member, native-profile placement. VRAM
  expert fraction ~55-60% by size (96 GB minus dense minus KV/reserve, over
  ~132 GB expert mass) with SKEWED routing -> the coverage curve prices the
  read-hit; pre-spec paper band 20-40 t/s single-stream, to be pinned by R0
  and the curve, spec stack (A3/A4 rungs re-run on this shape) on top.
  Bring-up discipline: trunc-6 gate vehicle first (#124 loopback recipe),
  then the full serve. TRAP guard: if this build is a different checkpoint
  than the serve the NATIVE profile was collected on, collect a fresh profile
  and regenerate the placement artifact (cross-checkpoint placement is a
  known trap); R0 artifact-pinning resolves the 0731-vs-Q4_K_XL identity
  question first.

Honest band math (all inputs to be re-pinned at A1): placement projection
9-11; coverage moves (A2 legs) are worth what they measure - no credible
paper-math takes the CPU term below ~45-60 ms/token at current VRAM fraction;
spec stack x1.10-1.30 -> **12-15 t/s single-stream is the paper-supported
band for the current checkpoint+quant on this rig**. Reaching 20-30
single-stream needs a coverage step-change, which means a decision (section
8a): the UD-Q4_K_XL 144.5 GiB build in-box (96 GB VRAM holds a majority of its
expert mass with skewed routing - the NATIVE coverage curve prices this
exactly), or accepting aggregate throughput (A5) as the 20-30 metric. The
plan does not decide this; it produces the numbers that let the user decide.

## 4. Spine B - GLM-5.2, target 20-30 t/s

Facts that shape everything (models skill): glm-dsa, 256 experts, routing
profiled near-UNIFORM -> placement/replication levers are WORTHLESS here.
Miss-rate ~= RAM share of expert mass, full stop. Native MTP head exists
(has_mtp). Two builds: Q2_K_XL 243.6 GB (fits X99 96 VRAM + 251 RAM in-box),
UD-Q4_K_XL 467 GB (exceeds any single box - W2W fleet or parked for RoCE).

Ladder:

- **B0 (zero build)**: pin arch numbers - n_expert_used, layer count, expert
  bytes/token, dense mass (gguf dump + a short LLAMA_EXPERT_PROFILE run to
  CONFIRM uniformity on real traffic, Cov@25.5% ~= 0.255 expected). Without
  E_tok the band math is fiction; with it, it is one division.
- **B1**: Q2 in-box bring-up: glm52-trunc{6,18}-q2 gate vehicles first
  (load + byte-stability), then full eplocal (CUDA0-2 + CPU, uniform shares -
  no placement artifact, it cannot help).
- **B2**: measure vs the uniform-miss formula: t_ram ~= E_tok x ram_share /
  BW_eff. With ~65-70% of expert mass in DDR4, the pre-spec single-stream
  band is wide (8-20 t/s) until B0/R0 pin E_tok and BW_eff - do not promise
  20 before those land.
- **B3**: spec stack (native MTP head + #107/#108) - same legs as A3/A4.
  LUD's task-dependence note applies: expect code >> prose acceptance.
- **B4**: worker member A/B (+32 GB VRAM coverage vs wire cost - same
  experiment as A2, uniform-routing edition). RoCE (#60) re-prices this rung
  when hardware lands.
- **B5**: -np batching curve; uniform routing makes the batch expert-union
  grow faster than on skewed models - measure, do not assume the V4 curve.
- **Q4-467 feasibility note (no rung until B1-B3 report)**: needs W2W across
  boxes (#28 machinery) with most expert mass on DDR4 across TWO boxes;
  single-stream 20-30 on TCP is not credible on paper-math; revisit post-RoCE
  or as an aggregate-throughput target only. Worker-box RAM size is an R0
  unknown for this note.

## 5. R0 - instruments and unknowns (cheap, before/alongside A1)

- DDR4 effective bandwidth on the X99 (STREAM-class probe; channel population
  check) - every band in sections 3-4 divides by it.
- Spine artifact pinning: exact quant + on-disk size for "the 160 GB V4"
  serve config and both GLM builds, recorded here at first measurement.
- GLM arch numbers + uniformity confirmation (B0).
- In-box 4-member boundary cost (META_TIMING on an A2 leg) - the fleet number
  (0.33 ms/reduce) says boundaries are a rounding error today; verify that
  survives the eplocal shape.
- Worker-box RAM size (GLM Q4 note).
- c (target/draft speed ratio) per spine on the real serves - sets PEARL's
  initial gamma.

### R0 results (2026-08-14, first pass)

- **DDR4 effective bandwidth (X99, 56T/251GiB)**: OpenMP bench, flat from 8
  to 56 threads: copy (STREAM convention) 35.1-35.4 GB/s, **pure-read 60.3
  GB/s**. Use 60 for expert-read terms, 35 for mixed r/w. (The plan's old
  33 GB/s plug was ~1.8x pessimistic on reads.)
- **Spine artifacts pinned**: spine A serve vehicle = DeepSeek-V4-Flash-0731
  **UD-Q8_K_XL, 150.8 GiB weights (161,869,615,520 bytes, 5 shards; the
  models dir shows 181G with extras)**, launcher metadata: deepseek4, 43
  blocks, 256 experts, has_mtp=false (head stripped from this export - the
  #124 drafter file carries it, which is exactly A3's config). So the A6
  endgame build (UD-Q4_K_XL 144.5 GiB) is a different checkpoint AND quant:
  A6 needs its own profile+placement (trap confirmed, not hypothetical). GLM-5.2:
  UD-Q4_K_XL 436 GiB (X99 + /mnt/files local); Q2_K_XL 6 shards at
  /mnt/full-models on the local box - must be copied to the X99 for B1.
  IQ2XXS keeper 81 GiB on X99.
- **GLM-5.2 arch (B0, gguf header)**: glm-dsa, block_count 79 = 3 leading
  dense + 76 MoE trunk (nextn/MTP head is extra, hy3 pattern;
  nextn_predict_layers=1 -> spec stack applies). 256 experts, **top-8 + 1
  SHARED always-active** (pin shared experts in VRAM: ~0.8 GiB at Q2).
  expert_ffn 2048 x embd 6144 = 37.75M params/expert (~21 MiB at Q4,
  ~10.6 MiB at Q2). **E_tok ~= 76 x 9 x per-expert ~= 7.2 GiB/token at Q2**
  - GLM's active expert mass is ~2.2x V4's. Group routing off.
- **B2 band refresh with measured numbers**: Q2 in-box (VRAM ~75 GiB of
  ~202 GiB expert mass = ~37% coverage; uniform routing -> miss ~63%): RAM
  term ~4.6 GiB/token @ 60 GB/s ~= 77-82 ms -> **~9-11 t/s pre-spec, 10-14
  with the spec stack**. GLM 20-30 single-stream is NOT paper-supported
  in-box on this rig - spine B's 20-30 needs the aggregate metric (decision
  (b)), worker VRAM behind a dieted wire, or W2W/RoCE. Spine A (A6) remains
  the 20-30 carrier.
- **A1 blocker status (2026-08-14)**: the placed serve was killed by the
  launcher recreate at 12:48Z (VRAM free, all endpoints healthy, catalog
  says unloaded; the earlier "loaded" reading was manifest optimism). The
  roster-matched artifact v4-0731-native-place-15-15-15-18-37.json IS in
  the X99 launcher cache (Aug 14 04:13). Relaunch is the user's call per
  the standing serve rule; measurement resumes the moment it is up.

## 6. Non-goals (reasons on record)

- **Trained self-coordination** (2601.09921's core mechanism, LUD training,
  CLLM-class consistency training): requires training the served model;
  serving-fork scope excludes it. The #85/#71 block-parallel closures stand -
  the papers explain WHY untrained variants failed (the consistency was never
  trained in), they do not reopen the lane.
- **beta-confidence (lossy) span acceptance** except as an explicit opt-in
  gated by the #110 KL battery. Default acceptance stays lossless.
- **Tree / n>=3 wide speculation on RAM-heavy rosters**: union-read law
  (w2 = 1.62-1.70x; wider = worse on DDR4). Width stays <=2; PEARL adds
  depth-adaptivity instead of width.
- **LP pair-fusion**: closed, quality-catastrophic (own plan doc).
- **CPU experts for models that fit VRAM**: the #31 law. (On THESE spines the
  CPU term is unavoidable - the plan shrinks it, it cannot remove it.)

## 7. What lands where

- Code rungs (#107 gate, #108 PEARL loop): commits per rung with gate
  evidence; env kill-switches documented in docs/env-gates.md the same
  commit; TASKS #107/#108 entries updated at the front, this doc's rung
  status updated same session.
- Measurement rungs (A1-A5, B0-B5): verdicts in TASKS #127 entry +
  next-steps handoff; numbers with config, run count, coherence verdict, and
  named baseline per the ledger skill - "faster" without a baseline is not a
  result.

## 8. Decision points (status as of 2026-08-14)

- **(a) DECIDED 2026-08-14: UD-Q4_K_XL in-box variant** is the spine A
  endgame - formalized as rung A6 above. A1-A5 on the current fleet
  checkpoint remain the calibration ladder.
- **(b) DECIDED 2026-08-17 (user): the 20-30 goal is SINGLE-STREAM,
  strictly.** The measured np4 target-only ~19.6 aggregate (A5-on-V4) is a
  capability, not the goal; honest single-stream band on the current
  checkpoint stays 12-15 and the lane still needs single-stream levers.
  Consequence: batching headroom (4x positions ~= 2x cost, measured) should
  be spent INSIDE one stream -> tree/multi-candidate speculation and the
  #132 RPC-drafter architecture become the primary levers.
- **(c) APPROVED 2026-08-14: #108 PEARL build green-lit.** Engineering order
  unchanged: #107 gate first (small, independent), #108 on its evidence.
  Both carry the temp-0 spec-vs-nospec byte leg as the landing gate.
- **(d) PARKED 2026-08-14: GLM Q4-467** - no W2W bring-up spend; revisit
  post-RoCE (#60) or when spine B's Q2 ladder reports.
- **(e) OPEN - APEX-style KV/attention annex** (worker or CPU) for
  throughput mode: only enters the ladder with a written pay-gate inequality
  and a measured profile behind it; propose-then-approve.
