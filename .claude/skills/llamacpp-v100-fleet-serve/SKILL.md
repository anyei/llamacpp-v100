---
name: llamacpp-v100-fleet-serve
description: Launch, watch, and stop the llamacpp-v100 true-parallel-inference fleet serves (hy3 record roster and DeepSeek-V4-Flash production shapes) - env gate stacks, load-time expectations, stale-manifest wedge handling, and health-watcher patterns. Use when starting/stopping any fleet serve or diagnosing a load that will not come up.
---

# Fleet serves: launch, watch, recover

## Rosters and known-good shapes

**hy3 record roster** (the measurement vehicle, `hy3-1M-MTP-Q4_K_M.gguf`, 183 GB):
`./run-ep-fleet-hy3-spec.sh` — env knobs: `EXPERT_DEFER=1 STATS=1` (v3 serve),
`PLACE=1` (placed artifact), `NP=2`, `SPEC=1 NMAX= PMIN=`, `BCAST_FUSE/WIRE_F16/WIRE_Q8`
(default on). Roster: CUDA0+CUDA1 owners (`ATTN_OWNER=0,1`, dual-role) + workers
local:50053/.11:50052/.15:50055, `-ts 21,21,46,50,27`. The script waits for
workers to listen. Port 8098, api key `anyei`.

**V4-Flash production shape** (measured best 2026-07-31, ~5.9 t/s cold): dual-role
owners, LEAN roster (no .15):
```bash
nohup docker run --name llama-ep-v4dual3 --gpus all \
  -v "$HOME/server/git-projects/llama.cpp-work/build-cuda75:/srcbin:ro" \
  -v "$PWD:/repo:ro" -v /mnt/files:/models:ro --network host \
  -e LD_LIBRARY_PATH=/srcbin/bin \
  -e LLAMA_META_EP_ONLY=1 -e LLAMA_META_ATTN_OWNER=0,1 \
  -e CUDA_VISIBLE_DEVICES=0,1 -e LLAMA_API_KEY=anyei \
  -e LLAMA_FLEET_KV_RESERVE_MB=6144 -e LLAMA_FLEET_CAPACITY_CHECK=0 \
  -e GGML_META_BCAST_FUSE=2 -e GGML_RPC_WIRE_F16=1 -e GGML_RPC_WIRE_Q8=1 \
  -e GGML_META_EXPERT_DEFER=1 -e GGML_META_BOUNDARY_STATS=1 \
  --entrypoint /srcbin/bin/llama-server nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  -m "/models/DeepSeek-V4-Flash/DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf" \
  --rpc 127.0.0.1:50053,10.5.5.11:50052 \
  --device CUDA0,CUDA1,RPC0,RPC1 -sm tensor -ts 18,26,60,46 \
  -ngl 99 --no-mmap --rpc-reload \
  -c 8192 -ub 256 -b 256 --host 0.0.0.0 --port 8098 -np 2 -fit off \
  > /tmp/serve.log 2>&1 & disown
```
Facts: DSA/V4 IS owner-group capable (dual-role owners fine). A second local GPU
as a NON-owner expert member (`n_local > n_owners` + `ALLOW_MULTI_LOCAL`) CRASHES
at first decode on V4 (ret -3, bug filed) — don't use that shape. Lean rosters
beat wide ones (every member adds gather legs; dropping .15 GAINED throughput).
`-np 2` is production default: no idle-stream penalty, +40% aggregate under load.

## Gate stacks

Production/default: `BCAST_FUSE=2  WIRE_F16=1  WIRE_Q8=1  EXPERT_DEFER=1
BOUNDARY_STATS=1  KV_RESERVE_MB=6144  CAPACITY_CHECK=0`. Quality fallback if a
serve reads off: `GGML_META_EXPERT_DEFER_WAIT_US=3000` (exact-class at control
speed). Never serve: `EXPERT_DEFER=2`, `EXPERT_DEFER_PREFILL=1`,
`PROBE_DEFER_GATHER=1`, `GGML_META_TIMING=1`, `GGML_META_ZL_STATS=1` (last two =
measurement-only). LP gates (`LLAMA_LP_PAIRS`, `GGML_META_PARTIAL_MERGE`) are
LOSSY and quality-ungated on the fleet — measurement only until the ladder passes.

## Health watcher pattern

Launch detached (`nohup ... & disown`, never a raw background task — the ~10-min
cap kills docker clients), then watch in <10-min re-armed windows:
```bash
for i in $(seq 110); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer anyei" http://127.0.0.1:8098/health)
  [ "$code" = "200" ] && { echo UP; exit 0; }
  st=$(docker ps -a --filter name=<container> --format '{{.Status}}')
  case "$st" in Up*) ;; *) echo "DIED: $st"; docker logs <container> 2>&1 | grep -aE " E " | tail -5; exit 1;; esac
  sleep 5
done; echo STILL_LOADING
```
The container-name check must tolerate the first ~5 s (docker run races the
first poll — treat empty status as "not yet", not death).

## Load-time expectations and the stale-manifest wedge

- Warm worker caches, same layout as last serve: 15-25 min (183 GB hy3) /
  20-30 min (144 GiB V4). Placed (`PLACE=1`) loads: single-threaded permuted,
  ~35 min, and need the raised cache limits.
- **Layout churn = slow loads**: switching models/layouts LRU-evicts the previous
  layout's slices from worker disk caches. Symptom:
  `batched placement: N/N entries missed on <ep> - manifest went stale, failing the endpoint`
  repeating — this is the #8/#72(a) wedge, it SELF-HEALS by streaming (watch
  worker RSS grow); expect +10-20 min.
- **.15 end-of-load alloc refusal**: after ~28 min a placed load can die with
  `failed to allocate ... buffer of size <tiny>` on .15 (long-uptime worker RAM
  bloat). The immediate RETRY loads clean (the failed attempt warmed the caches).
- The serve exits on load failure; container shows `Exited` — `docker rm -f` and
  relaunch. With `--rpc-reload` a worker loss mid-serve triggers in-process
  reload; a reload after cache-miss can SEGFAULT (exit 139, bug filed) — treat a
  dead serve container as relaunch-from-scratch.

## Stopping and etiquette

- `docker rm -f <serve-container>` — workers free their share on disconnect
  (worker processes stay up; disk caches persist).
- The GPUs/workers are shared with the user's soaks and wizard tests: don't
  relaunch over a serve the user asked to keep down; don't restart workers or
  run CPU-heavy builds during anyone's measurement window.
- Serve inventory: `docker ps --format '{{.Names}}' | grep llama-ep`.
