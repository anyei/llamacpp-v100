---
name: llamacpp-v100-ledger
description: Recording conventions for the llamacpp-v100 fork - where results, designs, verdicts and traps go (TASKS.md, docs/*-plan.md, memory, commits) and the required rigor for measured claims. Use when landing any feature, measurement, or verdict, and at session handoffs.
---

# Ledger discipline

## Where things go

| Artifact | Home |
|---|---|
| Task state + measured verdicts (dense, newest-first inside each entry) | `TASKS.md` — one giant line per task; append updates as `**BOLD DATED HEADERS**` at the FRONT of the entry body |
| Design + build state + gate results for a lane | `docs/<lane>-plan.md` (fill-the-bubble, lp-pair-fusion, hot-expert-replication, expert-placement, launcher-wizard…) — add dated `## 3x.` sections; when a lane closes, update the `Status:` header line to **LANE CLOSED <date>** with a pointer |
| Env gates | `docs/env-gates.md` — every new env gets a row (name, type, default, one-liner incl. serve-safety); note "parse VALUES: =0 is off" |
| Session handoff | memory `next-steps.md`: new `# RESUME HERE (<date>)` block at top, demote the old one to `# PREVIOUS`; update the frontmatter `description:` AND the MEMORY.md index line |
| Cross-escape summary | fill-the-bubble-plan `## 1b ESCAPES END-STATE` |

## Rigor rules for any measured claim

- Record: config (exact envs/flags), run count, per-run numbers or range+mean,
  counters (defer/ready/LOST, star/bcast1), coherence verdict, and the NAMED
  baseline compared against. "Faster" without a baseline is not a result.
- Negative results get FULL rigor — the closed-lane list (placement ×2, threads,
  leg-weight, LP, spec) is the roadmap's most valuable asset. When a design's
  own falsifier fires, say so explicitly ("section-4 stop rule fired").
- Traps discovered get recorded twice: in the session handoff AND in the
  relevant skill/plan doc.

## Commit style

- Subject: `area: what changed - key result` (e.g. `meta: LP pair boundary
  merge works - name-tag derivation + reduce suppression (stub star 3->2,
  byte-exact vs unmerged)`).
- Body: mechanism in 2-4 lines + gate evidence. Docs-only commits prefix `docs:`.
- Every substantive commit gets pushed to `v100` remote, branch
  `parallel-inference`, same session. Ledger updates ride WITH the code commit
  or immediately after — never leave measured results uncommitted overnight.

## Reference shas / signatures worth citing

- Loopback stub exact reference: `c80261ff` (default stack); LP-on: `2265c9e7`.
- PPL baseline family (record roster, wikitext 8x -c 512): 3.7669–3.7950 ±0.21.
- Structural passes: `defers == injects`, `LOST 0.00`.
