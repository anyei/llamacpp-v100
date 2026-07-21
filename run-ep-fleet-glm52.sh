#!/usr/bin/env bash
# GLM-5.2 (744B MoE, Q2_K_XL 227 GB, 6 shards) as an EXPERT-PARALLEL fleet:
# CUDA0 = dedicated attention owner (dense/attn/router, no expert share),
# CPU workers hold the routed experts. Serves on :8097.
#
# CAPACITY REALITY (why this may hold at the gate): EP places experts in WORKER
# RAM only - the owner takes no expert share and #48 forbids the 2nd local GPU
# as a member. GLM's ~210 GiB of experts vs ~160-170 GiB of pooled worker caps
# (.11+.15cpu+.25+.30+local) is SHORT unless every box is up. The capacity gate
# (fixed in #39) HOLDS the load and prints required vs available - power on
# more workers (or run the LAYER fallback, run-fleet-glm52-autoweight.sh, which
# adds both V100s' 64 GiB to the pool).
#
# Perks on: auto-weight + split-time rescore (4.9), proto 4.10 manifest
# handshake (zero-offer warm loads on 4.10 workers), --rpc-reload surgical
# recovery, trimmed KV reserve (MLA KV is tiny - the 20 GiB default would
# waste a tight pool). MTP=1 adds GLM's NextN speculative decoding (p-min 0.75
# + no-pad, the #52 recipe) - run PLAIN first, coherence-gate, then A/B MTP.
set -euo pipefail
cd "$(dirname "$0")"

EXTRA_ARGS=""
if [ "${MTP:-0}" = "1" ]; then
  EXTRA_ARGS="--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 1 --spec-draft-p-min 0.75"
  export COORD_SPEC_NO_PAD=1
  echo "MTP speculative decoding ENABLED (n-max 3, p-min 0.75, no-pad)"
fi

MODELS_DIR=/mnt/full-models \
COORD_IMAGE=${COORD_IMAGE:-llamacpp-local-v100:0a20aab2f} \
COORD_API_KEY=anyei \
EP_MODEL=GLM-5.2-Q2_K_XL-00001-of-00006.gguf \
EP_WORKERS=${EP_WORKERS:-127.0.0.1:50053,10.5.5.11:50052,10.5.5.25:50052,10.5.5.30:50052} \
EP_DEVICES=${EP_DEVICES:-CUDA0,RPC0,RPC1,RPC2,RPC3} \
EP_AUTO_WEIGHT=--rpc-auto-weight \
EP_CTX=${EP_CTX:-4096} \
EP_PORT=8097 \
EP_PARALLEL=${EP_PARALLEL:-1} \
COORD_GPU=0 \
COORD_KV_RESERVE_MB=${COORD_KV_RESERVE_MB:-8192} \
COORD_FLEET_ADMIN=1 \
  docker compose -f docker-compose.ep-fleet.yml up -d --force-recreate

echo
echo "Fleet UI + chat:  http://$(hostname -I | awk '{print $1}'):8097/"
echo "Follow the load:  docker logs -f llama-ep-fleet   (watch 'fleet capacity' - a hold names its own fix)"
echo "Add workers:      EP_WORKERS=...,host:port EP_DEVICES=CUDA0,RPC0,...,RPCn ./run-ep-fleet-glm52.sh"
echo "Stop:             docker compose -f docker-compose.ep-fleet.yml down"
