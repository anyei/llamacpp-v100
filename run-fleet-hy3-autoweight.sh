#!/usr/bin/env bash
# Serve Hunyuan 3 (hy_v3, 295B MoE 192x top-8, 80 layers + 1 MTP/NextN, 1M ctx,
# Q4_K_M ~171 GB) with a SCORE-AUTO-WEIGHTED LAYER split over the discovered fleet.
# Sibling of run-fleet-glm52-autoweight.sh (GLM on :8097); this one runs on :8095.
#
# REQUIREMENTS
#   - hy_v3 arch support: image >= e3b4f019c (upstream #25395 port).
#   - Capacity: 171 GB weights + LLAMA_FLEET_KV_RESERVE_MB (default 20 GB) needs
#     ~191 GB pooled -> EVERY worker box online (.11, .15 both, .25, .30, .26).
#     The new capacity gate holds the load (state: waiting-capacity) and starts
#     automatically once enough boxes beacon.
#   - Model lives on /mnt/files (HDD): the cold local read adds ~15-20 min on top
#     of the LAN stream to cold worker caches.
#
# MTP NOTE: first runs are plain decode. The MTP head is single-depth trained -
# upstream measured draft acceptance collapsing at default p-min and ~88-97%
# at p-min 0.75 (same as our prod MTP config). Wire spec flags only after the
# plain fleet run is proven coherent.
#
# COHERENCE GATE: read the actual output text before trusting any t/s.
set -euo pipefail
cd /home/anyei/server/git-projects/llama.cpp

if docker ps --format '{{.Names}}' | grep -q '^llama-fleet-coordinator$'; then
  echo "WARNING: llama-fleet-coordinator is running - it will be replaced. Ctrl-C within 5s to abort."
  sleep 5
fi

MODELS_DIR=/mnt/files \
COORD_API_KEY=anyei \
COORD_IMAGE=llamacpp-local-v100:e3b4f019c \
COORD_MODEL=hy3-1M-MTP-Q4_K_M.gguf \
COORD_AUTO_WEIGHT=1 \
COORD_GPUS=0,1 \
COORD_CTX=4096 \
COORD_UB=256 \
COORD_BATCH=256 \
COORD_PORT=8095 \
COORD_FLEET_ADMIN=1 \
COORD_PREFLIGHT=/models/Qwen3-0.6B-BF16.gguf \
  docker compose -f docker-compose.fleet-coordinator.yml up

# Fleet UI + chat:  http://<this-box>:8095/
# Chosen split:     docker logs llama-fleet-coordinator | grep -i auto-weight
# Stop:             docker compose -f docker-compose.fleet-coordinator.yml down
