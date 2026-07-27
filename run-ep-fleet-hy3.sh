#!/usr/bin/env bash
# Serve hy3 (hy3-1M-MTP Q4_K_M, 182.5 GB MoE) across the fleet — #70 owner-GROUP
# expert-parallel with DUAL-ROLE owners. Measured 2026-07-23 (coherence-gated):
#   plain decode 5.06-5.22 t/s single-stream (layer baseline 2.74; ts 18,18,52,50,27 measured 4.61-4.80)
#   CPU-only-experts variant (owners ts 0):  3.53-3.80 t/s
# Both V100s form the attention owner group (layers interleaved il % 2, all
# attention syncs on NVLink) AND each holds a 21 GB expert share in leftover
# VRAM (~800 GB/s expert reads - the dual-role win, TASKS.md #70).
#
# SIZING (GGUF-parsed, do this again before changing -ts):
#   experts = 164.9 GiB   attention = 3.3 GiB (owners)   mirror = 1.9 GiB/member
#   -ts values are EXPERT BYTES in GiB; member footprint = slice + 1.9 + ~1.5.
#   Fit each share to the box: local 73 GiB avail, .11 ~59 (its worker DIES on
#   overshoot - abort-on-alloc-fail, #68b/#72), .15 ~48 (user dev laptop).
#
# Prerequisites:
#   - Workers up: local :50053, .11 :50052 (intel-full image: SYCL0=RPC1 is
#     SKIPPED, CPU=RPC2 is the member), .15 CPU :50055.
#   - Local worker cache (cache-rpc/) LRU limit must exceed its slice or
#     manifest-stale endpoint-fails are STRUCTURAL (#72b); --rpc-reload below
#     is what recovers them - never run EP loads without it (#72a).
#   - Capacity gate must be OFF: it does not yet understand dedicated-split
#     sizing (#70 tail). Cold load ~25-30 min; warm (cached slices) ~15-20.
#
# BINARY: this recipe predates an image carrying #70 - it runs the local build
# (merge-test worktree @1980084e6, see next-steps memory). Swap --entrypoint
# for the image once one is built with #70+UI.
set -euo pipefail

SRC=${SRC:-/tmp/claude-1000/-home-anyei-server-git-projects-llama-cpp/7c2996c9-82cb-42db-b7cb-20d14bc60707/scratchpad/merge-test}
MODELS_DIR=${MODELS_DIR:-/mnt/files}
MODEL=${MODEL:-hy3-1M-MTP-Q4_K_M.gguf}
PORT=${PORT:-8098}
API_KEY=${API_KEY:-anyei}

# MTP=1 adds draft-mtp speculative decoding (the model ships a native MTP head).
SPEC_ARGS=()
SPEC_ENV=()
if [[ "${MTP:-0}" == "1" ]]; then
    SPEC_ARGS=(--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 1 --spec-draft-p-min 0.75)
    SPEC_ENV=(-e LLAMA_SPEC_DRAFT_NO_PAD=1 -e LLAMA_SPEC_TIMING=1)
fi

docker rm -f llama-ep-hy3 >/dev/null 2>&1 || true
exec docker run --name llama-ep-hy3 --gpus all \
  -v "$SRC:/src:ro" -v "$MODELS_DIR:/models:ro" --network host \
  -e LLAMA_META_EP_ONLY=1 -e LLAMA_META_ATTN_OWNER=0,1 -e LLAMA_META_ALLOW_MULTI_LOCAL=1 \
  -e CUDA_VISIBLE_DEVICES=0,1 -e LLAMA_API_KEY="$API_KEY" \
  -e LLAMA_FLEET_KV_RESERVE_MB=6144 -e LLAMA_FLEET_CAPACITY_CHECK=0 \
  -e GGML_META_BCAST_FUSE="${BCAST_FUSE:-2}" -e GGML_RPC_WIRE_F16="${WIRE_F16:-1}" -e GGML_RPC_WIRE_Q8="${WIRE_Q8:-1}" \
  "${SPEC_ENV[@]}" \
  --entrypoint /src/build-cuda/bin/llama-server \
  nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  -m "/models/$MODEL" \
  --rpc 127.0.0.1:50053,10.5.5.11:50052,10.5.5.15:50055 \
  --device CUDA0,CUDA1,RPC0,RPC2,RPC3 -sm tensor -ts 21,21,46,50,27 \
  -ngl 99 --no-mmap --rpc-reload \
  "${SPEC_ARGS[@]}" \
  -c 4096 -ub 256 -b 256 --host 0.0.0.0 --port "$PORT" -np 1 -fit off
