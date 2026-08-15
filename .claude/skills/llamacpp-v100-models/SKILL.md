---
name: llamacpp-v100-models
description: Model-zoo facts for the llamacpp-v100 fleet - paths, sizes, arch quirks (hy3 MTP, V4 DSA, GLM routing), test vehicles, and the profiling/placement artifacts. Use when picking a model for a serve or experiment, or reasoning about arch-specific behavior.
---

# Model zoo and arch facts

## Serve/measurement models

| Model | Path | Size | Facts |
|---|---|---|---|
| hy3 (record roster vehicle) | `/mnt/files/hy3-1M-MTP-Q4_K_M.gguf` | 183 GB | `hy_v3`: 80 layers (79 MoE, layer 0 dense) + 1 nextn/MTP, 192 experts k=8, 1M ctx, REASONING model (quality gates must include math reads). All #71 baselines live here |
| DeepSeek-V4-Flash (production) | `/mnt/files/DeepSeek-V4-Flash/DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf` | 144.5 GiB, 5 shards (shard 1 near-empty) | DSA arch: dedicated attention owner; owner-GROUP layer-interleave WORKS (dsv4 caches are layer-resolved, #70); production shape = lean dual-role (see llamacpp-v100-fleet-serve). Native MTP head. Chat model — bare `/completion` gives Q&A-list artifacts |
| GLM-5.2 | `/mnt/files/GLM-5.2/` (UD-Q4_K_XL 11 shards, 467 GB) + `/mnt/full-models/GLM-5.2-Q2_K_XL-*` (6 shards, 243.6 GB) | | `glm-dsa`, 256 experts; routing profiled near-UNIFORM → skew-based levers (placement/replication) are worthless on it |
| Qwen3.6-27B MTP | (single-box production, compose files) | | the MTP spec-decode regression net (~73 t/s, 2×V100 tensor-split) |

## Test vehicles

- **Trunc stub**: `/work/hy3-trunc5-mtp.gguf` + `--override-kv hy_v3.block_count=int:5`
  + `LLAMA_TRUNC_ARR=1`. Reports n_layer **4** to placement validation. Output is
  degenerate BY DESIGN — only byte-stability/sha and counters mean anything.
  4-member loopback artifact: `/work/trunc5-place-1111.json`.
- Other truncs: `glm52-trunc{6,18}-q2.gguf` in `/mnt/files`.
- 40-layer ceiling probes: full model + `--override-kv hy_v3.block_count=int:40`
  (wall-time only; early runs 500 on degenerate output — expected).

## Profiling / placement artifacts

- `profiles/hy3-full.json` — router histogram (5502 tokens): top-16/192 experts
  = 71% of routed slots, top-48 = 90%. Collected via `LLAMA_EXPERT_PROFILE`.
- `placements/hy3-record-21-21-46-50-27.json` — hot-first placement for the
  record roster (owner = top-48/layer, Cov@25.5% = 0.913). Roster-specific:
  member count must match the serve. Placement measured NULL both regimes but
  the artifact + machinery are reusable.
- `/mnt/files/sweeps.json` — measured-speed registry the wizard reads.

## Arch quirk cheat-sheet

- Qwen3.8-27B (arch qwen35, native MTP): the chat template defaults
  reasoning_effort to XHIGH and force-opens `<think>` - real prompts think for
  thousands of tokens and exhaust max_tokens inside the think block ("never
  stops thinking"). Knobs: `--chat-template-kwargs '{"reasoning_effort":"medium"}'`
  (low/medium/xhigh only; "high" silently maps to xhigh), `--reasoning-budget N`
  hard cap (+ optional --reasoning-budget-message), per-request
  chat_template_kwargs {"enable_thinking": false}.

- hy3 MTP: decode graphs can be 2-wide (nextn) — `ne[1]==1` gates miss them;
  spec/draft-mtp per-request overrides are silently ignored (server flags only).
- V4/DSA: attention owner takes 0 expert share ONLY as a default — dual-role
  owners measured fine; `n_local > n_owners` non-owner GPU = filed crash.
- Wizard/scan: shard sets collapse to `-00001-of-` (name-stripped); sharded
  `size_bytes` sums all shards (`n_shards` field).
- PPL vehicles: an 8-chunk run never prints BOUNDARY_STATS (needs 128 graphs) —
  don't steer by serve counters during PPL.
