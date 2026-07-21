#!/usr/bin/env bash
# SSD-streaming showcase, one command: DeepSeek-V4-Flash (86 GB) on ONE 32 GB
# V100 + RAM + NVMe, every measured-good knob on (TASKS.md #15 + #55).
#
#   ./run-ssd-deepseek.sh                 # serve on :8098
#   SSD_MODEL=other.gguf ./run-ssd-deepseek.sh
#
# What "all the perks" means here (see docker-compose.ssd.yml for the knobs):
#   - three-tier streaming: VRAM slot cache (GPU landing) -> RAM SLRU -> O_DIRECT NVMe
#   - parallel miss reads (SSD_READ_THREADS=4): #55 measured +17% decode on the
#     CPU tier and +31% prefill - MB-scale random preads are latency-bound at QD1
#   - input_cpy VRAM reclaim, SLRU scan resistance, O_DIRECT: default-on
#   - debug counters ON: watch `GPU cache hit=` / `miss-path` / `prefetch` lines
#     (`docker logs -f llama-ssd-deepseek`) - hit rate is everything, raise
#     SSD_VRAM_MIB / SSD_RAM_MIB until just below OOM
#   - SSD_PREFETCH stays 0: the previous-token lookahead measured NO-WIN for
#     DeepSeek-class routing (TASKS.md #55) - set SSD_PREFETCH=6 to experiment,
#     the log then reports its prediction accuracy
#
# IMAGE NOTE: the #55 knobs need an image built from commit >= 342d8f040
# (REBUILD-IMAGE.md; `latest` from an older tree still runs, minus the new reads).
#
# The model must live on the NVMe you want to stream from (MODELS_DIR).
set -euo pipefail
cd "$(dirname "$0")"

MODELS_DIR=${MODELS_DIR:-/mnt/models/ollama37-k80/.ollama/custom-models} \
SSD_IMAGE=${SSD_IMAGE:-llamacpp-local-v100:latest} \
SSD_VRAM_MIB=${SSD_VRAM_MIB:-16384} \
SSD_RAM_MIB=${SSD_RAM_MIB:-30720} \
SSD_READ_THREADS=${SSD_READ_THREADS:-4} \
SSD_PREFETCH=${SSD_PREFETCH:-0} \
  docker compose -f docker-compose.ssd.yml up -d

echo
echo "Serving on   http://$(hostname -I | awk '{print $1}'):8098/  (first load streams non-experts, ~1 min)"
echo "Counters:    docker logs -f llama-ssd-deepseek | grep ssd_stream"
echo "Stop:        docker compose -f docker-compose.ssd.yml down"
