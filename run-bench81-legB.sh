#!/bin/bash
# TASKS #81 leg B: the FORK's full stack on the identical 2-box shape.
# CUDA0 only + .11 fork worker (:50052). Meta EP, single attention owner,
# fuse=2 + q8/f16 wire + deferral v3.
set -u
SRC=$HOME/server/git-projects/llama.cpp-work/build-cuda75
MODEL=/mnt/models/ollama37-k80/.ollama/custom-models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf

exec docker run --name llama-bench81b --gpus all \
  -v "$SRC:/srcbin:ro" \
  -v /mnt/models/ollama37-k80/.ollama/custom-models:/models:ro \
  --network host \
  -e LD_LIBRARY_PATH=/srcbin/bin \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e LLAMA_META_EP_ONLY=1 -e LLAMA_META_ATTN_OWNER=0 \
  -e LLAMA_FLEET_KV_RESERVE_MB=2048 -e LLAMA_FLEET_CAPACITY_CHECK=0 \
  -e GGML_META_BCAST_FUSE=2 -e GGML_RPC_WIRE_F16=1 -e GGML_RPC_WIRE_Q8=1 \
  -e GGML_META_EXPERT_DEFER=1 -e GGML_META_BOUNDARY_STATS=1 \
  --entrypoint /srcbin/bin/llama-server nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  -m "/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf" \
  --rpc 10.5.5.11:50052 \
  --device CUDA0,RPC0 -sm tensor -ts 28,59 \
  -ngl 99 --no-mmap --rpc-reload \
  -c 4096 -ub 256 -b 256 \
  --host 127.0.0.1 --port 8099 -np 1 -fit off
