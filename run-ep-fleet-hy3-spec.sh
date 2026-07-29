#!/usr/bin/env bash
# TASKS #71 probe 0: hy3 EP record config, plain decode vs MTP spec with the
# coordinator-local draft (LLAMA_META_LOCAL_DRAFT=1), on the current default
# stack (BCAST_FUSE=2 + q8_0/f16 wire ladder). Same binary, same roster, same
# -ts - the ONLY variable is SPEC. docs/fill-the-bubble-plan.md section 2.1.
#
# The 2026-07-24 spec verdict (ratio ~0.85, shelved) was measured before
# boundary fusion and wire compression; the boundary the verify chain
# amortizes is now mostly arrival latency, so re-measure before any escape-(c)
# build. Gate discipline: 6x 100-tok greedy, cache_prompt false, workers
# restarted before each leg (#8 stale-manifest trap), coherence-READ every leg.
#
# Usage: SPEC=0 ./run-ep-fleet-hy3-spec.sh   # control (plain decode)
#        SPEC=1 ./run-ep-fleet-hy3-spec.sh   # MTP draft, coordinator-local
set -euo pipefail

SRC=${SRC:-$HOME/server/git-projects/llama.cpp-work/build-cuda75}
REPO=${REPO:-$HOME/server/git-projects/llama.cpp}
MODELS_DIR=${MODELS_DIR:-/mnt/files}
MODEL=${MODEL:-hy3-1M-MTP-Q4_K_M.gguf}
PORT=${PORT:-8098}
API_KEY=${API_KEY:-anyei}

SPEC_ARGS=()
SPEC_ENV=()
if [[ "${SPEC:-0}" == "1" ]]; then
    SPEC_ARGS=(--spec-type draft-mtp --spec-draft-n-max "${NMAX:-3}" --spec-draft-n-min 1 --spec-draft-p-min "${PMIN:-0.75}")
    SPEC_ENV=(-e LLAMA_SPEC_DRAFT_NO_PAD=1 -e LLAMA_SPEC_TIMING=1 -e LLAMA_META_LOCAL_DRAFT=1)
fi
# DEFER=1: GGML_META_PROBE_DEFER_GATHER wall-time probe - OUTPUT IS GARBAGE BY
# DESIGN (star gather stops waiting for wire partials); prices the Expert-
# Deferral ceiling. Never serve users with it.
if [[ "${DEFER:-0}" == "1" ]]; then
    SPEC_ENV+=(-e GGML_META_PROBE_DEFER_GATHER=1)
fi
# EXPERT_DEFER: the real mechanism - wire partials injected one reduce late
# (see fill-the-bubble-plan 2.2/2.2a). 1 = v2 readiness-gated (defer only
# stragglers), 2 = v1 defer-all (coherence FAILS - measurement only).
# DEFER_WAIT_US / DEFER_SYNC_EDGE pass the quality-fallback knobs through.
if [[ "${EXPERT_DEFER:-0}" != "0" ]]; then
    SPEC_ENV+=(-e GGML_META_EXPERT_DEFER="${EXPERT_DEFER}")
    [[ -n "${DEFER_WAIT_US:-}" ]]   && SPEC_ENV+=(-e GGML_META_EXPERT_DEFER_WAIT_US="${DEFER_WAIT_US}")
    [[ -n "${DEFER_SYNC_EDGE:-}" ]] && SPEC_ENV+=(-e GGML_META_EXPERT_DEFER_SYNC_EDGE="${DEFER_SYNC_EDGE}")
fi
# PLACE=1: hot-expert placement artifact (#75). Gate 5 measured it null in the
# EXACT latency-bound regime; v3 deferral (EXPERT_DEFER=1) makes the fleet
# member-THROUGHPUT-bound, which is placement's premise - re-test combined.
if [[ "${PLACE:-0}" == "1" ]]; then
    SPEC_ENV+=(-e LLAMA_META_EXPERT_PLACEMENT=/repo/placements/hy3-record-21-21-46-50-27.json)
fi
# STATS=1: GGML_META_BOUNDARY_STATS counters (production-valid, no drains)
if [[ "${STATS:-0}" == "1" ]]; then
    SPEC_ENV+=(-e GGML_META_BOUNDARY_STATS=1)
fi
# LP=1: Layer-Parallel pair fusion (docs/lp-pair-fusion-plan.md) - LOSSY,
# quality-gate measurement only until the ladder passes. LP_EDGE = sync edge.
if [[ "${LP:-0}" == "1" ]]; then
    SPEC_ENV+=(-e LLAMA_LP_PAIRS=1 -e LLAMA_LP_SYNC_EDGE="${LP_EDGE:-2}" -e GGML_META_PARTIAL_MERGE=1)
fi

# wait for every roster worker to LISTEN: a freshly restarted rpc-server
# benchmarks for ~15-20 s before binding, and --rpc aborts on the first
# unreachable endpoint (raced twice on 2026-07-29)
for ep in 127.0.0.1:50053 10.5.5.11:50052 10.5.5.15:50055; do
    h=${ep%:*}; p=${ep#*:}
    for i in $(seq 1 24); do
        timeout 3 bash -c "echo > /dev/tcp/$h/$p" 2>/dev/null && break
        [[ $i == 24 ]] && { echo "worker $ep not listening after 2 min" >&2; exit 1; }
        sleep 5
    done
done

docker rm -f llama-ep-hy3 >/dev/null 2>&1 || true
exec docker run --name llama-ep-hy3 --gpus all \
  -v "$SRC:/srcbin:ro" -v "$REPO:/repo:ro" -v "$MODELS_DIR:/models:ro" --network host \
  -e LD_LIBRARY_PATH=/srcbin/bin \
  -e LLAMA_META_EP_ONLY=1 -e LLAMA_META_ATTN_OWNER=0,1 -e LLAMA_META_ALLOW_MULTI_LOCAL=1 \
  -e CUDA_VISIBLE_DEVICES=0,1 -e LLAMA_API_KEY="$API_KEY" \
  -e LLAMA_FLEET_KV_RESERVE_MB=6144 -e LLAMA_FLEET_CAPACITY_CHECK=0 \
  -e GGML_META_BCAST_FUSE="${BCAST_FUSE:-2}" -e GGML_RPC_WIRE_F16="${WIRE_F16:-1}" -e GGML_RPC_WIRE_Q8="${WIRE_Q8:-1}" \
  "${SPEC_ENV[@]}" \
  --entrypoint /srcbin/bin/llama-server \
  nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  -m "/models/$MODEL" \
  --rpc 127.0.0.1:50053,10.5.5.11:50052,10.5.5.15:50055 \
  --device CUDA0,CUDA1,RPC0,RPC1,RPC2 -sm tensor -ts 21,21,46,50,27 \
  -ngl 99 --no-mmap --rpc-reload \
  "${SPEC_ARGS[@]}" \
  -c 4096 -ub 256 -b 256 --host 0.0.0.0 --port "$PORT" -np "${NP:-1}" -fit off
