#!/usr/bin/env bash
# Serve DeepSeek-V4-Flash with a SCORE-AUTO-WEIGHTED LAYER split (the A/B counterpart
# to run-ep-fleet-deepseek.sh, which is the hand-tuned EP topology). Nothing is pinned:
# workers are discovered on the LAN and -ts is computed by --rpc-auto-weight from each
# worker's measured bandwidth score, capped by free memory. Runs on a SEPARATE port so
# you can keep the static EP script around and compare the two later.
#
# IMPORTANT — this is a DIFFERENT topology from the EP script, not just "EP minus -ts":
#   - -sm layer (a naive per-layer pipeline), because --rpc-auto-weight only applies to
#     layer mode. It is NOT the dedicated-attention EP config; expect lower t/s
#     (sum-of-stages law, TASKS #31). The point here is to see what the automatic
#     score-based balancing does on its own.
#   - Auto-weight engages ONLY if EVERY discovered worker reports a score, i.e. all of
#     them are on the new image AND started with --score. If any is unscored (e.g. a box
#     still on a 4.6 image), it logs a warning and falls back to the memory-proportional
#     default. Check the "--rpc-auto-weight:" lines in the startup log to see which split
#     it actually used.
#   - The score is COMPUTE bandwidth, not link speed. A fast-CPU / slow-link box (.25/.30)
#     gets a high score and a big share, but its slow link still gates the pipeline. That
#     mismatch is the thing worth observing in this experiment.
set -euo pipefail
cd /home/anyei/server/git-projects/llama.cpp

COORD_IMAGE=llamacpp-local-v100:94b7ba5dc \
COORD_MODEL=DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
COORD_AUTO_WEIGHT=1 \
COORD_GPUS=0,1 \
COORD_CTX=4096 \
COORD_UB=256 \
COORD_BATCH=256 \
COORD_PORT=8099 \
COORD_API_KEY=anyei \
COORD_FLEET_ADMIN=1 \
  docker compose -f docker-compose.fleet-coordinator.yml up

# Fleet UI + chat:  http://<this-box>:8099/   (EP static run stays on :8098)
# The split it chose is in the log:  docker logs llama-fleet-coordinator | grep -i auto-weight
# Stop:             docker compose -f docker-compose.fleet-coordinator.yml down
