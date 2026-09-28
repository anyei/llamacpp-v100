---
name: llamacpp-v100-models
description: Model-zoo facts for the llamacpp-v100 fleet - paths, sizes, arch quirks (hy3 MTP, V4 DSA, GLM routing, Qwen3.8-Flash-Next qwen4exp PLE + head-file MTP), test vehicles, MoE-cache vehicles, and the profiling/placement artifacts. Use when picking a model for a serve or experiment, or reasoning about arch-specific behavior.
---

# Model zoo and arch facts

## Serve/measurement models

| Model | Path | Size | Facts |
|---|---|---|---|
| hy3 (record roster vehicle) | `/mnt/files/hy3-1M-MTP-Q4_K_M.gguf` | 183 GB | `hy_v3`: 80 layers (79 MoE, layer 0 dense) + 1 nextn/MTP, 192 experts k=8, 1M ctx, REASONING model (quality gates must include math reads). All #71 baselines live here |
| DeepSeek-V4-Flash (production) | `/mnt/files/DeepSeek-V4-Flash/DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf` | 144.5 GiB, 5 shards (shard 1 near-empty) | DSA arch: dedicated attention owner; owner-GROUP layer-interleave WORKS (dsv4 caches are layer-resolved, #70); production shape = lean dual-role (see llamacpp-v100-fleet-serve). Native MTP head. Chat model — bare `/completion` gives Q&A-list artifacts |
| GLM-5.2 | `/mnt/files/GLM-5.2/` (UD-Q4_K_XL 11 shards, 467 GB) + `/mnt/full-models/GLM-5.2-Q2_K_XL-*` (6 shards, 243.6 GB) | | `glm-dsa`, 256 experts; routing profiled near-UNIFORM → skew-based levers (placement/replication) are worthless on it |
| Qwen3.6-27B MTP | (single-box production, compose files) | | the MTP spec-decode regression net (~73 t/s, 2×V100 tensor-split) |
| Qwen3.8-Flash-Next (single-box Flash serve, #143-#151) | X99 launcher `/models/Qwen3.8-FLash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` (dir name carries the capital L) + MTP head `mtp-Qwen3.8-Flash-Next-Mtp-Q8_0.gguf` in the same dir (a copy sits in the coordinator models root) | ~111 GB incl. the ~97 GiB PLE table; routed experts 71.7 GiB (3.1-3.6 MB/expert, ~1.5 GB/token) | arch `qwen4exp` (GDN + QSA + PLE + hyper-connections + MRoPE; port of upstream PR 27742, audit in research/qwen4exp-port-audit.md); 48 MoE layers; the MTP head is a SEPARATE file (`--model-draft`, 55-100% acceptance, n-max 3); production = `-ngl 99 -ncmoe 48 --moe-cache 15000` + MTP n3 (see llamacpp-v100-fleet-serve) |
| Qwen3.6-35B-A3B UD-Q4_K_XL (#151 quality vehicle) | dev `/work/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`; K80 box models dir; staged in X99 tmpfs (`/dev/shm`) for batteries | ~22 GB | the MoE-cache KL/decode battery vehicle (one K80 die, and the V100); `-ub 16` legs bit-exact, K80 `-ub 512` non-deterministic run-to-run |
| MiMo-V2.6-Flash-RL + GLM-5.3-Flash GSQ-RCO 3.5bit (#152, NOT loadable yet) | `/mnt/models-b/MIMO2.6/MiMo-V2.6-Flash-RL-GSQ-RCO-3.5bit.gguf` (135 GB, arch `mimo2`) and `/mnt/models-b/GLM-5.3-flash/GLM-5.3-Flash-GSQ-RCO-3.5bit.gguf` (137 GB + 1.2 GB mmproj, arch `glm5-next`) | | route decided: #67 upstream merge first, then PR 27773's diff; neither GGUF has nextn tensors; GSQ-RCO is a quant RECIPE (stock kernels load it); the K80 box (78 GB RAM) cannot hold them |

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
- `profiles/flash-next-q4-2026-09-06{,-bucketA,-bucketB}.json` — Flash-Next
  router histograms (#148): Cov@25.5% 0.584 merged but 0.354 cross-domain =
  domain-sensitive, no one-domain placement; the MoE cache's warm-start prior.

## Arch quirk cheat-sheet

- Qwen3.8-27B Q4_K_XL single-V100 MAX-T/S KNOBS (swept 2026-08-15, temp-0
  6x128tok legs): `-ngl 99 -fa on -c 8192 --spec-type draft-mtp
  --spec-draft-n-max 2` = **44.7 t/s stable** (target-only 33.7; n1 43.7@81%,
  n3 40.7@54%, n4/n5 negative; NO_PAD/adaptive/entropy all LOSE to padded
  fixed-n graph reuse; ctx 32768 costs ~-30%; KV q8_0 on V100 FA costs ~-30%
  - avoid both for speed). Padding makes p-min/entropy flags INERT on mtp -
  don't cargo them.
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
- Qwen3.8-Flash-Next (`qwen4exp`): the ~97 GiB PLE table rides a `-sm tensor`
  split (`-ts` splits the full 111 GB; `-ot` to CPU registers but does not
  move it) and the load counts it as CPU model bytes in the fleet UI (#147);
  multi-GPU LAYER splits needed the #148 sched graph-inputs fix; `--mmproj` on
  a tensor split ABORTS the load when CUDA0's share fills (#144, 862 MiB, hard
  assert - leave headroom or drop it); prefill routes <= 2 tokens wide
  regardless of `-ub`/`-np` = ~36 t/s prompt (#150, open; upstream #28136 is
  the likely fix); `--rpc` must precede `--device ...,RPC0`; `--device ...,CPU`
  needs `LLAMA_META_LOCAL_DRAFT` PRESENT in the env; `LLAMA_MMAP_RANDOM=1`
  advises the PLE table random (off by default, costs cold prefill on
  sequential models). Fork backend/loader INFO lines need `-lv 4`.
- Any speculation on a recurrent-state target (Flash-Next GDN) takes the
  rollback snapshots, so weightless drafters (ngram) count too.
