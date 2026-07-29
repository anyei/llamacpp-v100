---
name: llamacpp-v100-upstream-merge
description: The recurring defensive upstream merge for the llamacpp-v100 fork (#67) - what to merge, known conflict hotspots, and the mandatory regate ladder afterward. Use when merging upstream llama.cpp into parallel-inference or evaluating upstream features.
---

# Upstream merge (#67 recurring)

## Policy

Defensive-only: merge to stay current on kernels/models/server, never to adopt
upstream's distributed roadmap (the fork's meta/EP/RPC layer IS the moat).
Last merge: `47bd90033` (upstream 6d5a910: V4 fused ops, Volta CUDA graphs,
n_keep_tail). Weekly cadence intended; check `git log --oneline -5 upstream/master`.

## Conflict hotspots (fork-heavy files)

`ggml/src/ggml-backend-meta.cpp` (entirely fork), `ggml/src/ggml-rpc/ggml-rpc.cpp`
(proto 4.x ladder — upstream rpc changes need manual weaving; our minor version
must stay >= upstream's), `ggml/include/ggml-backend.h` + `ggml-rpc.h` (fork API
tails), `common/arg.cpp` (fleet/auto-weight args, allow_cpu gate),
`src/llama-model.cpp` (split policies), `src/llama-graph.cpp` (moe remap/mask,
name tags — upstream renames of `ffn_moe_topk`/`ffn_moe_weighted` BREAK the
name-tagged derivations and the profiler; grep for both after merge),
`src/models/hy-v3.cpp` (LP lambda restructure), `tools/server/*` (router/wizard/
fleet endpoints), `.devops/*.Dockerfile`.

## Name-tag contract check (easy silent breakage)

After merge, verify these strings still exist in `src/llama-graph.cpp` and the
meta backend still references them: `ffn_moe_topk-`, `ffn_moe_weighted_placed`,
`lp_moe_pair`. Also `GGML_TENSOR_FLAG_COMPUTE` semantics and the mul_mat_id
skip-sentinel contract (`ggml/include/ggml.h` around 1444).

## Mandatory regate ladder (in order, cheap → expensive)

1. Build: `build-cpu` + `build-cuda75` clean; feature strings present
   (`WIRE_Q8`, `EXPERT_DEFER`, `BCAST_FUSE` in libs).
2. Loopback byte-identity: default stack off-gate must reproduce `c80261ff`
   6/6 on the trunc stub (see llamacpp-v100-dev-workflow). If upstream changed
   kernels the sha may legitimately move — establish and record the NEW
   reference sha only after verifying stability + coherence.
3. Loopback engagement: `EXPERT_DEFER=1` counters sane (defers==injects, LOST 0).
4. MTP local serving A/B: the single-box Qwen3.6-27B MTP compose config must be
   byte-identical/t/s-neutral (production regression net, historically ~73 t/s).
5. Fleet smoke: one hy3 record-roster serve, 6-run leg + coherence read vs the
   current plateau (4.97–5.21 as of 2026-08); worker images rebuilt only if
   rpc/ggml-cpu changed (proto bump rules in llamacpp-v100-deploy).
6. Ledger: TASKS #67 entry + note in the session handoff; new upstream features
   worth absorbing get their own TASKS entries, not silent adoption.

## Rollback

The merge lands as ONE merge commit on `parallel-inference`; if gates fail and
the weave is deep, `git revert -m 1 <merge>` and file the blockers in #67
rather than shipping a half-gated tree.
