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
| serve exits 139 during `--rpc-reload` recovery | FIXED 2026-07-29 (HTTP-thread vocab use-after-free during reload; pre-task ctx guard in server_queue). Pre-fix binaries still crash | on a pre-fix binary: clean relaunch; on current binaries the reload holds requests and completes |
| decode `ret = -3` every request, serve "healthy" | graph references a failed endpoint — a mid-LOAD `batched placement: N/M entries missed ... failing the endpoint` line survives to a healthy-looking serve (#72(a); this was the whole "DSA n_local>n_owners crash", RESOLVED 2026-07-30 — the topology is valid) | grep the load log for `failing the endpoint`; on current binaries the first failed request arms surgical->reload and the serve SELF-HEALS (~25 min reload); pre-fix binaries segfault in that reload — relaunch instead |
| server up but first request 500s once then works | warmup graph race — retry before diagnosing |
| load crash `illegal memory access` at `ggml_backend_cuda_synchronize`, stack fingers `ggml_backend_rpc_benchmark_device` (any `-sm tensor` shape) | ROOT-CAUSED 2026-08-19 (#135=#136): pre-#136 images bench the composite `Meta(...)` device — the bench graph computes through the meta backend with raw unregistered tensors, member kernels dereference HOST pointers. `-sm tensor` itself WORKS (proven coherent 32 t/s leg, X99). Still degraded by design: params_fit unimplemented, backend sampling unsupported, cache_reuse disabled | roll to >= 8eb2026b0 (bench skips Meta devices, contains CUDA faults, error_tail keeps first error lines). On older images: no workaround env exists — use `-ts` shares on layer split, or roll forward |
| load SEGFAULT (exit 139, NO error output) right after `initializing, n_slots` on a meta-EP roster (EP_ONLY/loopback class) | same #131b bench-on-Meta bug, CPU-member symptom class | fixed 2026-08-19 (>= 8eb2026b0); on older images avoid meta-EP shapes whose model device list is the meta device, or roll forward |

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
