---
name: llamacpp-v100-fleet-measure
description: Measurement discipline for llamacpp-v100 true-parallel-inference fleet experiments - leg protocol, coherence and quality gates, counter instruments, PPL protocol, reference baselines, and the noise/warm-up traps that invalidate comparisons. Use before ANY fleet t/s, PPL, or A/B measurement, and when interpreting results.
---

# Fleet measurement: protocol, gates, baselines

## The leg protocol

```bash
RUNS=8 ./scripts/measure-fleet-leg.sh <leg-name>       # PORT=8098 default
```
6-10 × 100-token greedy completions (`temperature 0, cache_prompt false`), prints
per-run decode t/s and the FULL run-1 text. Then pull counters:
`docker logs <serve> 2>&1 | grep -a 'META_EXPERT_DEFER' | tail -3`.

**Rules that make legs comparable:**
- The fleet WARMS ~+0.5 t/s over a serve's first ~20 min → only
  plateau-vs-plateau or cold-vs-cold comparisons are honest. 6-run legs on a
  fresh serve read low.
- Fleet noise is ±0.3 t/s between same-config runs; only back-to-back
  same-session A/Bs are trustworthy; differences under ~0.4 mean NULL.
- Nothing else may compete for CPU during a leg: no image builds, no compiles
  (the local worker and the coordinator host-reduce share this box's cores —
  a `-j18` build once turned a leg into a -64% artifact).
- Workers restarted between legs reset their warm state; keep worker config
  IDENTICAL across the legs of one A/B (threads, image, cache limits).
- Before any leg for a NEW feature: verify the serve binary carries it
  (`strings` check) — silently-ignored envs measure the baseline.

## The coherence gate (mandatory, every leg)

HTTP 200 + plausible t/s is NOT correctness. READ the output:
- hy3: run-1 text from the measure script (expect coherent reasoning prose;
  repetition spirals and CJK intrusions are known failure signatures).
- V4-Flash: bare `/completion` produces a Q&A-list continuation artifact (base-
  completion framing, NOT damage) — judge coherence via `/v1/chat/completions`.
- For LOSSY features (LP pairing, deferral variants): add math/multi-step
  reasoning prompts (train catch-up, GSM8K-style) — fluent prose survives damage
  that reasoning does not. PPL alone misses decode-path damage (v3 lesson) and
  prefill damage dominates PPL (all-prefill instrument).

## Counter instruments

- `META_EXPERT_DEFER: defers D ready R (rate %) injects I LOST L` — engagement +
  health: `defers == injects` and `LOST 0.00` are structural passes; the
  defer/ready rate is the member-lateness instrument (rises when members slow).
- `META_BOUNDARY_STATS` — star/bcast1 counts (boundary structure), gather/deliver
  bytes. Prints every 128 graphs; an 8-chunk PPL run never prints it.
- Never baseline with `GGML_META_TIMING` (serializes) or `GGML_META_ZL_STATS` /
  `LLAMA_EXPERT_PROFILE` (eval callbacks force sched splits; ~-1.5 t/s).

## -np 2 aggregate measurement

Two concurrent curls per round, sum `predicted_per_second` of the pair; also run
sequential singles inside the same serve (no idle penalty expected). 4+ rounds.
Per-stream ~-30% under simultaneous load is normal; aggregate is the headline.

## PPL protocol

wikitext, 8 chunks, `-c 512`, on the record roster. Baseline family
**3.7669–3.7950 ± 0.21** (f32/f16/q8 wire all inside). Outside the bar = FAIL
regardless of how coherent reads looked (v2 deferral: fluent at PPL 4.57).
PPL is all-prefill: it cannot see decode-only damage — pair it with reads.

## Reference baselines (record roster unless noted)

| Config | Number |
|---|---|
| default-EP v3 serve (`EXPERT_DEFER=1`, -t8/-t6 workers) | 4.97–5.21 t/s plateau |
| v3 overnight plateau (2026-07-28) | 4.93–5.35 |
| pre-deferral control | 4.00 plateau / 3.67 same-day-warm |
| -np 2 aggregate under v3 | 7.2–7.7 (per-stream 3.6–3.9) |
| 40-layer trunc ceiling probe | 8.85–9.67 (~1.8×) — LP upper bound |
| deferral probe ceiling (garbage-by-design) | ~5.3 (+32%) |
| V4-Flash lean dual-role (production, cold) | 5.67–6.22 |
| V4-Flash 5-member dual-role (cold) | 5.23–6.23 |
| V4-Flash original CUDA0-only (warming) | 4.47–5.34 |
| loopback stub exact sha (default stack) | `c80261ff`; LP-on sha `2265c9e7` |

Closed lanes (do not re-measure without new mechanism): placement (null ×2),
worker threads beyond -t8/-t6 (all worse), leg-payload size (proto 4.14 null —
member lateness is chain phase lag), spec/draft-mtp on this fleet (ratio ≤ 1.0).

## Recording

Every measured leg lands in the ledger: `TASKS.md` #71 entry + the relevant
`docs/*-plan.md` section, with config, runs, mean, counters, coherence verdict,
and the comparison baseline. Negative results are recorded with the same rigor —
the closed-lane list above is the roadmap's most valuable asset.
