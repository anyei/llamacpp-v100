#!/usr/bin/env bash
# Serve DeepSeek-V4-Flash (86.7 GB MoE) across the fleet — dedicated-attention EP
# topology. Coordinator = this box (2x V100): attention/router/KV live on the local
# GPU, the routed experts split across the RPC workers. Measured 2.36-2.56 t/s decode
# single-stream (arc: 1.82 layer -> 2.24 star-reduce -> 2.56 fused-boundary).
#
# Prerequisites:
#   - Workers up: .11 CPU (:50052 -> RPC0) and .15 CPU (:50054 -> RPC1). Since 2026-07-16
#     .15 runs its GPU (:50052) and CPU (:50054) as SEPARATE workers - the 6 GB 1660 Ti
#     stays OUT of the EP ring (mirrored-set + share would overflow it, TASKS.md #39a).
#   - First load streams the expert slices to each worker unless the GGUF is staged
#     locally there (--model-dir /local-models). The workers' -c cache makes every
#     later load (and surgical recovery) fast.
set -euo pipefail
cd /home/anyei/server/git-projects/llama.cpp

# EP uses ONE local GPU as the dedicated attention owner (CUDA0); a SECOND local
# GPU as an expert member corrupts the reduce (TASKS.md #48) and is rejected at
# load. For both V100s on V4, single-box '-sm tensor -ngl 99 -ncmoe N' is faster
# (3.54 vs ~2.5 t/s). Experts go on the RPC workers.
COORD_IMAGE=llamacpp-local-v100:bc12761c5 \
COORD_API_KEY=anyei \
EP_MODEL=DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
EP_WORKERS=${EP_WORKERS:-10.5.5.11:50052,10.5.5.15:50054} \
EP_DEVICES=${EP_DEVICES:-CUDA0,RPC0,RPC1} \
EP_TS=${EP_TS:-0,3,2} \
EP_CTX=${EP_CTX:-4096} \
EP_PORT=${EP_PORT:-8098} \
EP_PARALLEL=${EP_PARALLEL:-1} \
COORD_GPU=0,1 \
  docker compose -f docker-compose.ep-fleet.yml up --force-recreate

echo
echo "Coordinator up. Fleet UI + chat:  http://$(hostname -I | awk '{print $1}'):8098/"
echo "Follow the load:                  docker logs -f llama-ep-fleet"
echo "Stop:                             docker compose -f docker-compose.ep-fleet.yml down"
