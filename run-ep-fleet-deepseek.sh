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
# (9.8 vs ~2.5 t/s post-#54; EP is for models that exceed local RAM+VRAM).
# Experts go on the RPC workers.
#
# FIXED ROSTER (2026-07-22): the three FASTEST boxes only - every extra member
# adds boundary RTTs, and the slow stragglers (.25/.26) serialize the chain
# (5-member run measured 0.95 t/s vs 2.5+ with a lean fast set):
#   RPC0 = local cpu worker :50053 (i9, ~34 GB/s)
#   RPC1 = .11 :50052 (Core Ultra, ~57 GB/s - the fleet's fastest CPU box)
#   RPC2 = .15 :50055 (i7, ~28-30 GB/s)
# Capacity ~105-115 GB pooled >= V4's 87 GB + reserve.
# NOTE (#68): .11 must run its worker with WORKER_DEVICES=CPU (or SYCL0,CPU) -
# its Arc iGPU (SYCL0) over-reports free memory, takes an oversized expert
# share, and dies at init_tensor. EP decode is RAM-bandwidth-bound, so CPU
# mode loses nothing vs SYCL there.
COORD_IMAGE=${COORD_IMAGE:-llamacpp-local-v100:2cb391528} \
COORD_API_KEY=anyei \
EP_MODEL=DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
EP_WORKERS=${EP_WORKERS:-127.0.0.1:50053,10.5.5.11:50052,10.5.5.15:50055} \
EP_DEVICES=${EP_DEVICES:-CUDA0,RPC0,RPC1,RPC2} \
EP_AUTO_WEIGHT=${EP_AUTO_WEIGHT:---rpc-auto-weight} \
EP_CTX=${EP_CTX:-4096} \
EP_PORT=${EP_PORT:-8098} \
EP_PARALLEL=${EP_PARALLEL:-1} \
COORD_GPU=0 \
  docker compose -f docker-compose.ep-fleet.yml up --force-recreate -d

echo
echo "Coordinator up. Fleet UI + chat:  http://$(hostname -I | awk '{print $1}'):8098/"
echo "Follow the load:                  docker logs -f llama-ep-fleet"
echo "Stop:                             docker compose -f docker-compose.ep-fleet.yml down"
