#!/bin/bash
# TASKS #71 crash-bug (a) fleet re-test: full V4, single attention owner
# (CUDA0), CUDA1 as PURE EXPERT MEMBER (n_local=2 > n_owners=1,
# ALLOW_MULTI_LOCAL=1) - the exact filed crash shape, on the fixed binary.
# 5-member roster keeps the local worker share BELOW production (~52GB) to
# stay clear of the #79 OOM cliff.
set -u
SRC=$HOME/server/git-projects/llama.cpp-work/build-cuda75
REPO=$HOME/server/git-projects/llama.cpp
MODEL=/models/DeepSeek-V4-Flash/DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf

exec docker run --name llama-ep-v4pm --gpus all \
  -v "$SRC:/srcbin:ro" -v "$REPO:/repo:ro" -v /mnt/files:/models:ro \
  --network host \
  -e LD_LIBRARY_PATH=/srcbin/bin \
  -e LLAMA_META_EP_ONLY=1 -e LLAMA_META_ATTN_OWNER=0 \
  -e LLAMA_META_ALLOW_MULTI_LOCAL=1 \
  -e CUDA_VISIBLE_DEVICES=0,1 -e LLAMA_API_KEY=anyei \
  -e LLAMA_FLEET_KV_RESERVE_MB=6144 -e LLAMA_FLEET_CAPACITY_CHECK=0 \
  -e GGML_META_BCAST_FUSE=2 -e GGML_RPC_WIRE_F16=1 -e GGML_RPC_WIRE_Q8=1 \
  -e GGML_META_EXPERT_DEFER=1 -e GGML_META_BOUNDARY_STATS=1 \
  --entrypoint /srcbin/bin/llama-server nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  -m "$MODEL" \
  --rpc 127.0.0.1:50053,10.5.5.11:50052,10.5.5.15:50055 \
  --device CUDA0,CUDA1,RPC0,RPC1,RPC2 -sm tensor -ts 0,32,60,46,28 \
  -ngl 99 --no-mmap --rpc-reload \
  -c 8192 -ub 256 -b 256 --host 0.0.0.0 --port 8098 -np 2 -fit off
