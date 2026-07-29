---
name: llamacpp-v100-triage
description: Failure triage playbook for the llamacpp-v100 fleet - decode -3, reload segfaults, stale manifests, worker drops, OOM classes, and quality-failure signatures with the instrument to reach for. Use when a serve, load, or measurement fails or output quality looks wrong.
---

# Triage playbook

## Load / serve failures

| Symptom | Diagnosis | Remedy |
|---|---|---|
| `batched placement: N/N entries missed ... manifest went stale` repeating | model-layout churn LRU-evicted worker cache slices (#8/#72a) | it SELF-HEALS by streaming (watch worker RSS grow); expect +10-20 min. Only intervene if it also drops the connection |
| `connection lost or worker crashed` during provisioning, but the worker container is `Up` | worker disk-thrash stalls (rpc-timing shows multi-second `SET_TENSOR_HASH` exec max) tripped the coordinator's socket timeout — worker did NOT crash | `docker rm -f` the serve, relaunch immediately — the failed pass warmed page caches; retry loads clean (proven 3×) |
| tiny alloc refused at end of a long load (e.g. 121 KB on .15) | long-uptime worker RAM bloat on the memory-capped box | immediate retry; if recurrent, ask the user to restart .15's worker |
| serve exits 139 during `--rpc-reload` recovery | FILED BUG: full-reload fallback segfaults after surgical cache-miss | never trust in-process reload after a worker drop — clean relaunch |
| decode `ret = -3` every request, serve "healthy" | graph references a failed endpoint (degraded load) OR an unsupported topology | check load log for an earlier endpoint failure; if topology: `n_local > n_owners` (non-owner second GPU) is a FILED CRASH on DSA — use owner groups instead |
| server up but first request 500s once then works | warmup graph race — retry before diagnosing |

## Quality-failure signatures (read the OUTPUT, not the status code)

| Signature | Known cause |
|---|---|
| repetition spiral | v1 defer-all class: too much expert mass arriving late |
| rare single-CJK-token intrusions when quoting the prompt | corrupted PROMPT KV (prefill-path perturbation) |
| token salad (CJK/fragment mix, unparseable) | structural graph damage — LP-class lossy transform too aggressive |
| `output does not match expected Content-only format` 500s | degenerate generations tripping the response parser — quality red flag on real models (cosmetic only on trunc stubs) |
| Q&A-list continuations on V4 via `/completion` | base-completion framing artifact, NOT damage — verify via chat endpoint |

## "My change did nothing" checklist

1. Port squatter serving old binary? `pgrep -a -x llama-server` and check cmdline.
2. Env silently ignored? binary `strings` check + `=0`-parses-as-ON audit.
3. Reading a recycled tensor? gather-time reads see ring-recycled bytes —
   capture at compute time (eval callback) instead.
4. Wrong layer of the system? Split-state derivation, window walk
   (`get_i_delayed`), and boundary registration are THREE separate mechanisms —
   `GGML_META_DEBUG_REDUCE=1` (REDUCE:/PM- lines) shows where boundaries
   actually land; instrument before iterating blind (rule learned the hard way).

## Instruments quick-pick

boundary placement → `GGML_META_DEBUG_REDUCE=1`; engagement/health →
`GGML_META_BOUNDARY_STATS=1` (needs 128 graphs); rpc traffic → `GGML_RPC_STATS=1`
(client) / `WORKER_RPC_TIMING=1` (worker, shows the disk-thrash stalls);
first faulting CUDA node → `GGML_CUDA_SYNC_NODES=1` + `GGML_CUDA_DISABLE_GRAPHS=1`.
