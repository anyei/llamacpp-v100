#!/usr/bin/env bash
# TASKS.md #54 live measurement: single-box V4 (2x V100 + -ncmoe) vs the fleet.
# Stops the fleet coordinator + local CPU worker (frees both GPUs and ~33 GiB
# RAM), serves V4 single-box, measures decode t/s + NVMe read B/s per token,
# then restores the fleet. Requires explicit consent because it takes down
# production serving for the duration (~30-60 min):
#
#   FLEET_STOP_OK=1 ./run-ncmoe-54-measure.sh            # baseline: -ncmoe 99
#   FLEET_STOP_OK=1 NCMOE_LAYERS=20 ./run-ncmoe-54-measure.sh   # VRAM-heavy variant
#
# Context (groundwork measured 2026-07-22, non-disruptive):
#   - fleet reference same day: 4.79 t/s decode, coherent (96-tok temp-0)
#   - NVMe (nvme0n1) random reads over the V4 gguf, O_DIRECT:
#       256K QD1 0.67 GB/s | 1M QD1 1.67 GB/s | 1M x4 2.51 GB/s | 4M x4 2.65 GB/s
#   - implication: at ~1.8 GB/token expert demand, a 50% cache-hit rate would
#     need >3.5 GB/s effective disk to explain the old 282 ms/token - the disk
#     cannot do that, so the historical 3.54 t/s ALREADY ran at a much higher
#     effective hit rate (routing skew; ssd-streaming measured 73% hit @30 GiB).
#     With the fleet's local worker stopped, ~70 GiB of the ~80 GiB of experts
#     fit in page cache - the single-box number with ALL local RAM is the real
#     #54 question, not a rerun of the 46 GiB-constrained one.
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${FLEET_STOP_OK:-0}" != "1" ]]; then
    echo "This stops llama-fleet-coordinator + llama-rpc-worker-cpu for the run."
    echo "Set FLEET_STOP_OK=1 to consent."
    exit 1
fi

IMAGE=${IMAGE:-llamacpp-local-v100:b80ba3114}
MODEL=${MODEL:-DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf}
MODELS_DIR=${MODELS_DIR:-/mnt/models/ollama37-k80/.ollama/custom-models}
NCMOE_LAYERS=${NCMOE_LAYERS:-99}
PORT=${PORT:-8097}
N_GEN=${N_GEN:-5}
GEN_TOKENS=${GEN_TOKENS:-128}

restore() {
    echo "--- restoring fleet"
    docker rm -f llama-ncmoe-54 >/dev/null 2>&1 || true
    docker start llama-rpc-worker-cpu llama-fleet-coordinator || true
}
trap restore EXIT

echo "--- stopping fleet (coordinator + local cpu worker)"
docker stop llama-fleet-coordinator llama-rpc-worker-cpu
free -g | sed -n 2p

echo "--- starting single-box V4 (-ngl 99 -ncmoe ${NCMOE_LAYERS}, mmap ON, no warmup)"
docker run -d --name llama-ncmoe-54 --gpus all \
    -e CUDA_VISIBLE_DEVICES=0,1 \
    -v "${MODELS_DIR}:/models:ro" \
    -p "${PORT}:8080" \
    --entrypoint /app/llama-server \
    "${IMAGE}" \
    -m "/models/${MODEL}" \
    -ngl 99 -ncmoe "${NCMOE_LAYERS}" -t 10 -c 8192 --no-warmup \
    --host 0.0.0.0 --port 8080 -np 1 -fit off

echo "--- waiting for the model (first tokens are cold; cache warms as it decodes)"
until curl -sf --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
    if [[ "$(docker inspect -f '{{.State.Running}}' llama-ncmoe-54 2>/dev/null)" != "true" ]]; then
        echo "--- FAILED: container died during load; last log lines:"
        docker logs --tail 12 llama-ncmoe-54 2>&1
        exit 1
    fi
    sleep 5
done

read_sectors() { awk '{print $3}' /sys/block/nvme0n1/stat; }

for i in $(seq 1 "${N_GEN}"); do
    s0=$(read_sectors); t0=$(date +%s.%N)
    out=$(curl -s --max-time 1200 "http://127.0.0.1:${PORT}/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"prompt\":\"Write a short paragraph about the history of computing. Attempt ${i}:\",\"max_tokens\":${GEN_TOKENS},\"temperature\":0}")
    t1=$(date +%s.%N); s1=$(read_sectors)
    python3 - "$out" "$s0" "$s1" "$t0" "$t1" <<'EOF'
import json, sys
d = json.loads(sys.argv[1]); s0, s1 = int(sys.argv[2]), int(sys.argv[3])
el = float(sys.argv[5]) - float(sys.argv[4])
t = d.get("timings") or {}
n = t.get("predicted_n") or 1
disk_gb = (s1 - s0) * 512 / 1e9
print(f"gen: {t.get('predicted_per_second', 0):.2f} t/s decode | prefill {t.get('prompt_per_second', 0):.1f} t/s | "
      f"disk {disk_gb:.2f} GB in {el:.0f}s = {disk_gb/el:.2f} GB/s = {disk_gb/n*1e3:.0f} MB/token")
print("  text:", (d["choices"][0].get("text") or "")[:160].replace("\n", " "))
EOF
done

echo "--- page cache after run:"; free -g | sed -n 2p
echo "--- done (fleet restored by trap)"
