---
name: llamacpp-v100-quality-battery
description: Cross-stack inference-QUALITY comparison for the llamacpp-v100 fork - KL-divergence protocol against an upstream llama.cpp logit base, noise-floor controls, PPL anchors, exact-answer decode probes, and interpretation thresholds. Use when someone suspects a serving strategy (EP tensor, fleet, gates) degrades output quality vs another stack, or to regate quality after meta/kernel changes. First run + reference numbers: TASKS.md #110 (2026-08-06).
---

# Quality battery: does stack X degrade inference vs stack Y?

Answers "is the math wrong?" with measurements, not vibes. Three levels, strongest
first; run all three — each sees damage the others are blind to.

## Level 1: KL divergence (the mathematical "same result" test)

One leg SAVES its full logit stream, every other leg REPLAYS the same text and
measures per-token distribution distance:

```bash
# base leg (the control stack):
llama-perplexity -m <model> -f /work/wikitext-2-raw/wiki.test.raw -c 512 --chunks 16 \
  <stack flags> --save-all-logits /work/<name>-base.dat
# each comparison leg:
llama-perplexity -m <model> -f ... -c 512 --chunks 16 <other stack flags> \
  --kl-divergence-base /work/<name>-base.dat --kl-divergence
```

- Base file ~1.05 GB at 16×512 tokens (f16-quantized logits; the base-derived
  PPL reads ~0.7% below the live run — an artifact, identical for all replays,
  so comparisons stay apples-to-apples).
- The base file is CROSS-BINARY compatible (upstream c8e03ce81 base replayed
  fine under the fork) — verify loudly-erroring load before trusting a new pair.
- **MANDATORY noise-floor control**: replay the base config against its own base.
  Classic GPU+CPU offload measured EXACTLY 0.000000 mean KLD (max 5.5e-5 = f16
  storage rounding) — deterministic. Do NOT assume other configs are; fleet legs
  have their own jitter. A leg's result is only interpretable relative to the
  measured floor of its own config.

**Thresholds** (mean KLD, same weights): ≤ noise floor = identical; ~1e-3 =
fp-reordering territory; ~1e-2–3e-2 = a Q4-quant-magnitude distribution shift —
check PPL and top-1 agreement before calling it damage; ≫ 0.05 = a stage is
damaging the math. Always report: mean/median/99% KLD, Same-top-p %, and the
PPL ratio ± CI. PPL inside the CI + top-1 ≥ ~94% + probes clean = quality-neutral.

## Level 2: PPL anchor

Same runs emit it free (Mean PPL(Q) vs PPL(base)). Absolute quality per stack;
catches prefill damage. It CANNOT see decode-only damage (all-prefill — v3
lesson) — never conclude from PPL alone.

## Level 3: exact-answer decode probes (the only level that sees decode damage)

KL and PPL are teacher-forced prefill. GGML_META_EXPERT_DEFER acts at DECODE
ONLY — invisible to levels 1–2. Serve each stack, run 10–15 greedy exact-answer
prompts (arithmetic, GSM8K-style, factual) via **/v1/chat/completions** (bare
/completion on chat models produces Q&A-list artifacts), `temperature 0`,
`max_tokens ≥ 400`, score by expected-answer substring.

- **Scorer trap (burned)**: match number formats loosely — "6400" vs "6,400"
  vs "$6,400" produced a false FAIL. READ every failing output before calling
  it a model error.
- Short probes (≤400 tok) cannot see slow-accumulating drift — pair with the
  long-horizon extension below when that is the suspicion.

## Reference numbers (first run, 2026-08-06, IQ2XXS keeper 86.7GB, TASKS #110)

| Leg | mean KLD vs upstream | top-1 | PPL | probes |
|---|---|---|---|---|
| upstream classic (base) | — | — | 5.2011 ± 0.21 | 12/12 |
| upstream self-replay | 0.000000 | ~100% | 5.2011 | — |
| fork classic layer+ncmoe | 0.0176 | 95.8% | 5.1686 | — |
| fork EP tensor clean (FUSE=2) | 0.0220 | 94.7% | 5.1590 | — |
| fork EP + WIRE_F16+Q8 | 0.0242 | 94.5% | 5.1463 | — |
| fork EP production (incl DEFER) | — | — | — | 12/12 |

Verdict: EP/meta/wire stages add ~0.007 KLD over fork baseline = quality-neutral.
**Known open lead**: the 0.0176 fork-vs-upstream drift exists in CLASSIC mode
(no meta code in path) — kernel-level, V100 FA path prime suspect; PPL-neutral.
Instrument: rerun the pair with `-fa off` both legs.

## Kept assets

- Upstream base: `/work/v4kl-upstream-base.dat` (keeper, 16×512, upstream c8e03ce81)
- Upstream build: `/work/build-upstream/` (CUDA arch 70); source worktree
  `llama.cpp-work/upstream-src` (repo remote `upstream`). Rebuild: cmake
  `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DLLAMA_CURL=OFF`, ~8 min -j16.
- Leg runner + probe scripts pattern: session scratchpad `v4-quality-legs.sh`,
  `decode-probes.py` (recreate from this skill if gone).
- Keeper model: `/mnt/models/ollama37-k80/.ollama/custom-models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf`
  (NVMe, 80.8 GiB, deepseek4, classic shape `-sm layer -ts 36,7 -ncmoe 32`;
  NEVER tensor+ncmoe, #82).

## Process traps (each cost real time)

1. Run legs via a bash-shebang script or explicit flags — zsh does NOT
   word-split `$VAR` args ("too many colons" docker error).
2. Serve/leg watchers must tolerate the first ~10 s of empty `docker ps`
   status (first-poll race reads as DIED while the container loads fine).
3. Wizard sizes are GiB; `du`/API are GB — an "80GB" model is ~86.7e9 bytes
   (a 60–120 *GiB* find filter missed it).
4. Upstream flag is `--n-cpu-moe` (alias `-ncmoe` exists); `--flash-attn on`
   both legs; identical `-c`, `--chunks`, corpus, and model file across ALL legs.
5. Wire-gate engagement is only provable via the `rpc: compressed boundary
   payloads ACTIVE` INFO line (needs `-v`) — absence of KLD change is not proof
   the gate was exercised.
6. GPU legs need the serving GPUs free — coordinate the production serve
   window (unload → battery → relaunch with the exact captured args/env).

## Extensions not yet run (for the "EP has something weird at scale" claim)

- KL protocol at long context: `-c 8192+`, fewer/longer chunks.
- Teacher-forced long-decode divergence hunt: 2000+-token greedy decode, dump
  per-token logprobs on both stacks, find where |Δ| GROWS (accumulation) vs
  stays flat (benign jitter).
- Owner-count sweep 1/2/4 on the 4×V100 rig — non-root-owner count is the
  suspected scaling axis (open non-root-attn-owner × chunking residual).
