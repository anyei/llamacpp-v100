#!/usr/bin/env bash
# Serve GLM 5.2 (glm-dsa, 744B/40B-active MoE, 79 layers) with a SCORE-AUTO-WEIGHTED
# LAYER split over the discovered fleet — the GLM counterpart to
# run-fleet-deepseek-autoweight.sh.
#
# MODEL: SixVolts/GLM-5.2-ewaste-edition-GGUF Q2_K_XL, 227 GB in 6 shards
# (Q2_K experts + IQ2_XXS on the 13 coldest layers, Q4_K non-experts —
# K-quants chosen for fast dequant on the fleet's pre-AVX-512 CPU boxes).
# NOTE: the earlier 2.244bpw file was ik_llama-only (ggml types 133/153) and
# could not load here — deleted 2026-07-17, replaced by this one.
#
# CAPACITY WARNING — this model does NOT fit the small fleet:
#   227 GB model + KV + reserves needs ~235 GB vs pooled ≈ 240 GB with EVERY box up:
#   .11 (~61) + .15cpu (~50) + .15gpu (~5.5) + .25 (~30) + .30 (~14) + .26 (~15)
#   + 2x V100 (~64). ALL workers must be online BEFORE starting; discovery only
#   runs at startup (restart the container after the fleet changes). The slow-link
#   boxes gate the pipeline (#31 sum-of-stages) but are a capacity necessity.
#   Expect the first load to stream ~170 GB to cold worker caches (the 100-Mbit
#   boxes are the long pole; possibly 1h+, fast once their -c caches are warm).
#
# NOTES
#   - Model lives in /mnt/full-models (MODELS_DIR override below); llama.cpp
#     auto-loads shards 2-6 from the first shard's directory.
#   - glm-dsa arch support is in image bc12761c5 (verified ancestor).
#   - Same container name as the DeepSeek scripts — starting this REPLACES any
#     running fleet coordinator. Port 8097 so old bookmarks don't lie.
#   - -c 4096 -ub 256 -b 256: the config proven stable for the V4 big-model runs;
#     MLA KV is small so ctx can grow later once it's known-good.
#   - COHERENCE GATE: read the actual output text before trusting any t/s.
set -euo pipefail
cd /home/anyei/server/git-projects/llama.cpp

if docker ps --format '{{.Names}}' | grep -q '^llama-fleet-coordinator$'; then
  echo "WARNING: llama-fleet-coordinator is running (probably the DeepSeek run) — it will be replaced. Ctrl-C within 5s to abort."
  sleep 5
fi

MODELS_DIR=/mnt/full-models \
COORD_API_KEY=anyei \
COORD_IMAGE=llamacpp-local-v100:bc12761c5 \
COORD_MODEL=GLM-5.2-Q2_K_XL-00001-of-00006.gguf \
COORD_AUTO_WEIGHT=1 \
COORD_GPUS=0,1 \
COORD_CTX=4096 \
COORD_UB=256 \
COORD_BATCH=256 \
COORD_PORT=8097 \
COORD_FLEET_ADMIN=1 \
COORD_PREFLIGHT=/models/Qwen3-0.6B-BF16.gguf \
  docker compose -f docker-compose.fleet-coordinator.yml up

# Fleet UI + chat:  http://<this-box>:8097/
# Chosen split:     docker logs llama-fleet-coordinator | grep -i auto-weight
# Stop:             docker compose -f docker-compose.fleet-coordinator.yml down
