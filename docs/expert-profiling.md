# Expert-Frequency Profiling → Hot-Expert Placement (TASKS #74/#75)

> **Status (2026-07-24): #74 DONE on hy3** — coverage@25.5% = **0.913** vs
> 0.255 uniform, cross-domain 0.912 (worst layer 0.56, median 0.94); emphatic
> GO (bar was ~0.35). hy3 has a structurally-cold expert tail: ~75% of expert
> bytes serve <9% of reads in any domain. **#75 v1 landed** (gates 1-4 passed:
> byte-exact, PPL-neutral, ownership audit) — fleet A/B is the remaining gate.
> GLM-5.2 remains to be profiled per the policy below.

How to measure per-layer router expert-selection frequencies on a serving
model and turn them into a frequency-ranked VRAM placement. General workflow
first, then the model-specific procedures (hy3, GLM-5.2).

## Why

Load-balancing losses equalize each expert's GLOBAL training load, not its
per-layer/per-domain inference load ("globally balanced, locally imbalanced").
Field measurements on balanced-trained MoEs put top-25% expert coverage at
37-53% of activations (MoE-Sieve; CRAFT measured per-layer peak-to-mean up to
27x on DeepSeek-R1/Kimi-K2). Our dual-role owners hold a UNIFORM slice today,
so their VRAM hit fraction equals their byte fraction by construction;
frequency placement converts the same VRAM into a higher hit fraction, which
directly reduces member-RAM expert reads — the fleet's decode bottleneck.
Full evidence review: `research/2026-07-parallel-decoding-and-distribution.md`
(iteration 2).

## General workflow

1. **Collect** — serve the model with the #74 profiler enabled:
   `LLAMA_EXPERT_PROFILE=/path/model-profile.json` on the coordinator.
   The profiler hooks the MoE top-k tensor per layer (imatrix-style eval
   callback), accumulates `counts[layer][expert]`, and refreshes the JSON
   every ~500 decode tokens. Works in ANY split mode — the router runs
   everywhere — so the profiling serve does not need to be the final layout.
2. **Traffic** — a few thousand decode tokens of REPRESENTATIVE prompts:
   15-25 prompts x 256 tokens across the domains you actually serve (chat,
   code, summarization, reasoning). Collect TWO domain buckets separately
   (two profile files) for the stability check.
3. **Analyze** — `scripts/expert-coverage.py <profile.json> --budget <frac>`:
   per-layer and aggregate coverage of the top-`frac` experts vs the uniform
   baseline (`frac` itself). Cross-domain stability: coverage on traffic B of
   the top set computed from traffic A. Stable + above-uniform = GO.
4. **Place (#75)** — per-layer top-k expert-ID lists sized to the owners'
   VRAM budget replace the contiguous uniform slice (scatter list in the
   split state). Uniform per-layer budget v1; entropy-weighted budgets v2.
5. **Cadence** — static placement, re-profile weekly or on domain shift.
   Do NOT chase per-batch hotness (micro-batch hot sets are noise — GEM).

## hy3 procedure (the #75 gate)

- Serve: the record EP config (`run-ep-fleet-hy3.sh`, ts 21,21,46,50,27) with
  the profiler env added — one warm reload (~15 min).
- Budget to evaluate: owners hold 42/164.9 GiB = **25.5%** → the decision
  number is per-layer Cov@25.5% vs the uniform 25.5%.
- GO bar: aggregate Cov@25.5% >= ~35% and cross-domain drop <= ~5 pp.
- Risk to check: hy3's shared experts (always resident, outside the routed
  pool) may have absorbed the skew — the counters answer this directly.

## GLM-5.2 procedure (policy: profile BEFORE the EP debut)

GLM EP is capacity-borderline (experts 216.2 GiB vs ~218-225 GiB pool) and
even more member-RAM-bound than hy3, so its FIRST EP layout should already be
frequency-placed. Two ways to get the profile, choose by box availability:

- **Path A — profile on a cheap full-model serve first (preferred when the
  fleet is free):** the known-good GLM layer fleet
  (`run-fleet-glm52-autoweight.sh`, ~247 GiB pooled, ~0.5-1.5 t/s) with the
  profiler on. Profiling needs ~4-6k decode tokens ≈ 1-1.5 h of decode at
  layer-fleet speed. Routing decisions are placement-independent, so this
  profile is exactly valid for the EP layout.
- **Path B — uniform-EP shakedown, then re-place (when the layer fleet is
  too expensive to stand up):** bring GLM EP up with a capacity-fitted
  UNIFORM split (gets ~5x the decode speed of layer mode, so 4-6k tokens
  ≈ 20-30 min), profiler on from the first token; then RELOAD with the
  frequency placement (warm caches make the re-place reload cheap). You pay
  one extra reload, not an extra serve.
- **Never profile a truncated GLM** — truncation changes hidden states and
  therefore routing; per-layer profiles from truncated runs are not valid.
- Budget to evaluate: owners contribute ~40-46 GiB of 216.2 = **18.5-21%**
  → decision number is Cov@~20%. Field prior for DeepSeek-lineage routing
  says this is where skew is largest (up to 27x peak-to-mean per layer), so
  expect a bigger relative lift than hy3.
- The placement feeds the same #75 scatter-list mechanism; combined with the
  capacity math redo (diagrams §3b), it decides whether GLM EP lands at a
  usable t/s.

## Interactions

- **#71 stage 2** (draft-on-VRAM-experts self-speculation): the profiled hot
  set IS the reduced draft expert set — same artifact, second consumer.
- **Auto-weight (#70 tail)**: once placement is frequency-based, auto-weight's
  owner shares should be expressed in "effective hit fraction," not bytes.
- **Upstream**: #25932 (--pin-hotexperts) and #26003 (--lazy-experts) are the
  single-box cousins — check both at each weekly merge for convergence.
