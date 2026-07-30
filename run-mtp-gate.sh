#!/bin/bash
# #67 merge gate: Qwen3.6-27B MTP single-box serving A/B vehicle (compose
# docker-compose.mtp.yml flags, but the MERGED build-cuda75 binaries).
# Historical net: ~73-81 t/s single-stream steady-state.
set -u
SRC=$HOME/server/git-projects/llama.cpp-work/build-cuda75
MODELS=/mnt/models/ollama37-k80/.ollama/custom-models

exec docker run --name llama-mtp-gate --gpus all \
  -v "$SRC:/srcbin:ro" -v "$MODELS:/models:ro" --network host \
  -e LD_LIBRARY_PATH=/srcbin/bin \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e GGML_CUDA_ALLREDUCE=p2p \
  -e LLAMA_DECODE_GRAPH_CACHE=4 \
  -e LLAMA_DECODE_GRAPH_CACHE_TOKENS=64 \
  -e GGML_META_MAX_GRAPHS=8 \
  -e GGML_CUDA_DISABLE_GRAPHS=1 \
  --entrypoint /srcbin/bin/llama-server nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  -m /models/Qwen3.6-27B-UD-Q4_K_XL-MTP.gguf \
  --mmproj /models/qwen3.6-27b-mmproj-f16.gguf \
  --host 127.0.0.1 --port 8097 \
  -ngl 99 -sm tensor -ts 0.5,0.5 -t 10 -np 1 \
  -c 262000 --batch-size 4096 --ubatch-size 4096 \
  --cont-batching --timeout 10 --no-mmap \
  --cache-prompt --cache-ram 12288 --ctx-checkpoints 8 \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 1 --spec-draft-p-min 0.75 \
  -fit off -fa on --no-prefill-assistant --reasoning-preserve
