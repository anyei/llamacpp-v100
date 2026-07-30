#!/bin/bash
# Relaunch-until-clean loop for the V4 production serve: a post-churn load can
# trip #72(a) (manifest-stale endpoint failure) several times while the worker
# cache re-converges on the layout; each pass streams more slices in. Retries
# bounded; state written to STATE_FILE for the session watcher.
set -u
STATE_FILE=${STATE_FILE:-/tmp/v4-restore-state.txt}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-6}

launch() {
  docker rm -f llama-ep-v4dual3 2>/dev/null
  sleep 2
  docker run -d --name llama-ep-v4dual3 --gpus all \
    -v "$HOME/server/git-projects/llama.cpp-work/build-cuda75:/srcbin:ro" \
    -v "$HOME/server/git-projects/llama.cpp:/repo:ro" -v /mnt/files:/models:ro \
    --network host \
    -e LD_LIBRARY_PATH=/srcbin/bin \
    -e LLAMA_META_EP_ONLY=1 -e LLAMA_META_ATTN_OWNER=0,1 \
    -e CUDA_VISIBLE_DEVICES=0,1 -e LLAMA_API_KEY=anyei \
    -e LLAMA_FLEET_KV_RESERVE_MB=6144 -e LLAMA_FLEET_CAPACITY_CHECK=0 \
    -e GGML_META_BCAST_FUSE=2 -e GGML_RPC_WIRE_F16=1 -e GGML_RPC_WIRE_Q8=1 \
    -e GGML_META_EXPERT_DEFER=1 -e GGML_META_BOUNDARY_STATS=1 \
    --entrypoint /srcbin/bin/llama-server nvidia/cuda:12.8.1-devel-ubuntu24.04 \
    -m "/models/DeepSeek-V4-Flash/DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf" \
    --rpc 127.0.0.1:50053,10.5.5.11:50052 \
    --device CUDA0,CUDA1,RPC0,RPC1 -sm tensor -ts 18,26,60,46 \
    -ngl 99 --no-mmap --rpc-reload \
    -c 8192 -ub 256 -b 256 --host 0.0.0.0 --port 8098 -np 2 -fit off
}

for attempt in $(seq 1 $MAX_ATTEMPTS); do
  echo "attempt $attempt launching $(date +%H:%M:%S)" >> $STATE_FILE
  launch > /dev/null
  while true; do
    sleep 15
    code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer anyei" http://127.0.0.1:8098/health)
    if [ "$code" = "200" ]; then
      echo "UP after attempt $attempt $(date +%H:%M:%S)" >> $STATE_FILE
      exit 0
    fi
    if docker logs llama-ep-v4dual3 2>&1 | grep -aqE "failing the endpoint"; then
      docker logs llama-ep-v4dual3 2>&1 | grep -aE "failing the endpoint" | tail -1 >> $STATE_FILE
      break # next attempt
    fi
    st=$(docker ps -a --filter name=llama-ep-v4dual3 --format '{{.Status}}' | head -1)
    case "$st" in
      Up*) ;;
      *) echo "DIED attempt $attempt: $st" >> $STATE_FILE; break ;;
    esac
  done
done
echo "EXHAUSTED $MAX_ATTEMPTS attempts" >> $STATE_FILE
exit 1
