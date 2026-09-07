#!/bin/bash
# TASKS #81 leg A: VANILLA UPSTREAM llama.cpp over RPC, true 2-box shape.
# CUDA0 only (no CUDA1, no local CPU-RAM layers) + .11 upstream rpc-server
# (:50060) carrying the rest in CPU RAM. -sm layer (upstream's RPC mode).
set -u
SRC=$HOME/server/git-projects/llama.cpp-work/build-upstream-cuda
MODEL=/mnt/models/ollama37-k80/.ollama/custom-models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf

exec docker run --name llama-bench81a --gpus all \
  -v "$SRC:/srcbin:ro" \
  -v /mnt/models/ollama37-k80/.ollama/custom-models:/models:ro \
  --network host \
  -e LD_LIBRARY_PATH=/srcbin/bin \
  -e CUDA_VISIBLE_DEVICES=0 \
  --entrypoint /srcbin/bin/llama-server nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  -m "/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf" \
  --rpc 10.5.5.11:50052 \
  -ngl 99 -sm layer -ts 66,34 \
  -c 4096 -ub 256 -b 256 \
  --host 127.0.0.1 --port 8099 -np 1
