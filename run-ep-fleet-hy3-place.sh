#!/usr/bin/env bash
# TASKS #75 fleet A/B: hy3 EP record config, uniform expert split vs
# frequency-ranked placement (placements/hy3-record-21-21-46-50-27.json).
# Same binary, same roster, same -ts - the ONLY variable is
# LLAMA_META_EXPERT_PLACEMENT (PLACE=1). docs/expert-placement-plan.md section 4/6.
#
# Gate discipline (dev-workflow section 5 + coherence-gate memory): READ the output
# text (Rayleigh prompt) before trusting any t/s; placement changes WHICH
# member holds each expert, so warm worker caches partially miss on the
# first placed load (permuted slice bytes differ; see plan section 2b) - budget a
# longer load, warm after that.
#
# Usage: PLACE=0 ./run-ep-fleet-hy3-place.sh   # control (uniform)
#        PLACE=1 ./run-ep-fleet-hy3-place.sh   # placement
set -euo pipefail

SRC=${SRC:-$HOME/server/git-projects/llama.cpp-work/build-cuda75}
REPO=${REPO:-$HOME/server/git-projects/llama.cpp}
MODELS_DIR=${MODELS_DIR:-/mnt/files}
MODEL=${MODEL:-hy3-1M-MTP-Q4_K_M.gguf}
PORT=${PORT:-8098}
API_KEY=${API_KEY:-anyei}

PLACE_ENV=()
if [[ "${PLACE:-0}" == "1" ]]; then
    PLACE_ENV=(-e LLAMA_META_EXPERT_PLACEMENT=/repo/placements/hy3-record-21-21-46-50-27.json)
fi

docker rm -f llama-ep-hy3 >/dev/null 2>&1 || true
exec docker run --name llama-ep-hy3 --gpus all \
  -v "$SRC:/srcbin:ro" -v "$REPO:/repo:ro" -v "$MODELS_DIR:/models:ro" --network host \
  -e LD_LIBRARY_PATH=/srcbin/bin \
  -e LLAMA_META_EP_ONLY=1 -e LLAMA_META_ATTN_OWNER=0,1 -e LLAMA_META_ALLOW_MULTI_LOCAL=1 \
  -e CUDA_VISIBLE_DEVICES=0,1 -e LLAMA_API_KEY="$API_KEY" \
  -e LLAMA_FLEET_KV_RESERVE_MB=6144 -e LLAMA_FLEET_CAPACITY_CHECK=0 \
  -e GGML_META_BCAST_FUSE="${BCAST_FUSE:-2}" -e GGML_RPC_WIRE_F16="${WIRE_F16:-1}" -e GGML_RPC_WIRE_Q8="${WIRE_Q8:-1}" \
  "${PLACE_ENV[@]}" \
  --entrypoint /srcbin/bin/llama-server \
  nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  -m "/models/$MODEL" \
  --rpc 127.0.0.1:50053,10.5.5.11:50052,10.5.5.15:50055 \
  --device CUDA0,CUDA1,RPC0,RPC1,RPC2 -sm tensor -ts 21,21,46,50,27 \
  -ngl 99 --no-mmap --rpc-reload \
  -c 4096 -ub 256 -b 256 --host 0.0.0.0 --port "$PORT" -np 1 -fit off
