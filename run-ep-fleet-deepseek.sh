#!/usr/bin/env bash
# Serve DeepSeek-V4-Flash (86.7 GB MoE) across the fleet — dedicated-attention EP
# topology. Coordinator = this box (2x V100): attention/router/KV live on the local
# GPU, the routed experts split across the RPC workers. Measured 2.36-2.56 t/s decode
# single-stream (arc: 1.82 layer -> 2.24 star-reduce -> 2.56 fused-boundary).
#
# Prerequisites:
#   - Workers up:  .11 (RPC0) and .15 (RPC2 = its CPU; RPC1 is its 6 GB 1660 Ti, unused
#     for experts). Roll them with docker-compose.rpc-worker-*.yml. .11 is already on
#     the 4.7 image; .15 serves fine on 4.6 but roll it to -68d91c065 for the UI's
#     per-worker logs/score/restart.
#   - First load streams the expert slices to each worker unless the GGUF is staged
#     locally there (--model-dir /local-models). The workers' -c cache makes every
#     later load (and surgical recovery) fast.
set -euo pipefail
cd /home/anyei/server/git-projects/llama.cpp

COORD_IMAGE=llamacpp-local-v100:2a64d2cc6 \
COORD_API_KEY=anyei \
EP_MODEL=DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
EP_WORKERS=10.5.5.11:50052,10.5.5.15:50052 \
EP_DEVICES=CUDA0,CUDA1,RPC0,RPC2 \
EP_TS=0,0,3,2 \
EP_CTX=4096 \
EP_PORT=8098 \
COORD_GPU=0,1 \
  docker compose -f docker-compose.ep-fleet.yml up --force-recreate

echo
echo "Coordinator up. Fleet UI + chat:  http://$(hostname -I | awk '{print $1}'):8098/"
echo "Follow the load:                  docker logs -f llama-ep-fleet"
echo "Stop:                             docker compose -f docker-compose.ep-fleet.yml down"
