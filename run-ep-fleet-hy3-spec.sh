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
  -c 4096 -ub 256 -b 256 --host 0.0.0.0 --port "$PORT" -np 1 -fit off
