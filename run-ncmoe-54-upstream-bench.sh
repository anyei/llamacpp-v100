#!/usr/bin/env bash
# Upstream comparison for TASKS.md #54 / docs/v4-single-box-benchmark.md:
# run the SAME single-box V4 MoE benchmark on an UNMODIFIED upstream llama.cpp.
# Benchmark-only - upstream is cloned to a sibling dir and never patched; the
# build and the server run inside a CUDA 12.8 container because the host's
# CUDA 13 toolchain dropped sm70 (Volta).
#
#   FLEET_STOP_OK=1 ./run-ncmoe-54-upstream-bench.sh                # -ncmoe 37
#   FLEET_STOP_OK=1 NCMOE_LAYERS=99 ./run-ncmoe-54-upstream-bench.sh
#
# Stops the fleet coordinator + local CPU worker for the run (same reason as
# run-ncmoe-54-measure.sh: the benchmark needs both GPUs and the page cache),
# restores them on exit.
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${FLEET_STOP_OK:-0}" != "1" ]]; then
    echo "This stops llama-fleet-coordinator + llama-rpc-worker-cpu for the run."
    echo "Set FLEET_STOP_OK=1 to consent."
    exit 1
fi

UPSTREAM_DIR=${UPSTREAM_DIR:-/home/anyei/server/git-projects/llama.cpp-upstream-bench}
BUILD_IMAGE=${BUILD_IMAGE:-nvidia/cuda:12.8.1-devel-ubuntu24.04}
MODEL=${MODEL:-DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf}
MODELS_DIR=${MODELS_DIR:-/mnt/models/ollama37-k80/.ollama/custom-models}
NCMOE_LAYERS=${NCMOE_LAYERS:-37}
PORT=${PORT:-8096}
N_GEN=${N_GEN:-5}
GEN_TOKENS=${GEN_TOKENS:-128}

if [[ ! -d "${UPSTREAM_DIR}" ]]; then
    git clone --depth 50 https://github.com/ggml-org/llama.cpp "${UPSTREAM_DIR}"
fi
echo "--- upstream commit: $(git -C "${UPSTREAM_DIR}" log -1 --format='%h %ci %s' | cut -c1-100)"

if [[ ! -x "${UPSTREAM_DIR}/build/bin/llama-server" ]]; then
    echo "--- building upstream (CUDA sm70, containerized)"
    docker run --rm -v "${UPSTREAM_DIR}:/src" -w /src "${BUILD_IMAGE}" bash -c "
        set -e
        apt-get update -qq && apt-get install -y -qq cmake git >/dev/null 2>&1
        cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release
        cmake --build build --target llama-server -j20"
fi

restore() {
    echo "--- restoring fleet"
    docker rm -f llama-upstream-54 >/dev/null 2>&1 || true
    docker start llama-rpc-worker-cpu llama-fleet-coordinator || true
}
trap restore EXIT

echo "--- stopping fleet (coordinator + local cpu worker)"
docker stop llama-fleet-coordinator llama-rpc-worker-cpu || true

echo "--- starting upstream llama-server (-ngl 99 -ncmoe ${NCMOE_LAYERS}, mmap ON, no warmup)"
docker run -d --name llama-upstream-54 --gpus all \
    -e CUDA_VISIBLE_DEVICES=0,1 \
    -v "${UPSTREAM_DIR}:/src:ro" \
    -v "${MODELS_DIR}:/models:ro" \
    -p "${PORT}:8080" \
    --entrypoint /src/build/bin/llama-server \
    "${BUILD_IMAGE}" \
    -m "/models/${MODEL}" \
    -ngl 99 -ncmoe "${NCMOE_LAYERS}" -t 10 -c 8192 --no-warmup \
    --host 0.0.0.0 --port 8080 -np 1

echo "--- waiting for the model"
until curl -sf --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
    if [[ "$(docker inspect -f '{{.State.Running}}' llama-upstream-54 2>/dev/null)" != "true" ]]; then
        echo "--- FAILED: container died during load; last log lines:"
        docker logs --tail 15 llama-upstream-54 2>&1
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
      f"disk {disk_gb:.2f} GB in {el:.0f}s = {disk_gb/n*1e3:.0f} MB/token")
print("  text:", (d["choices"][0].get("text") or "")[:160].replace("\n", " "))
EOF
done

echo "--- coherence gate (chat endpoint)"
curl -s --max-time 600 "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Explain in two sentences why the sky is blue."}],"max_tokens":96,"temperature":0}' \
    | python3 -c "
import json, sys
d = json.load(sys.stdin)
t = d.get('timings') or {}
print('chat:', (d['choices'][0]['message']['content'] or '')[:300])
print('chat decode:', round(t.get('predicted_per_second', 0), 2), 't/s')"

echo "--- done (fleet restored by trap)"
