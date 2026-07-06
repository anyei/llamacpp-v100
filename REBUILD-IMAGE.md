# Rebuilding the `llamacpp-local-v100` Docker image

Step-by-step guide for building this repo into the production server image
used by `docker-compose.yml` / `docker-compose.mtp.yml` / `docker-compose.nospec.yml`.

Everything runs from the repo root:

```bash
cd ~/server/git-projects/llama.cpp
```

---

## 1. Check what you are about to build

`docker build` copies the **working tree** (not just committed files), so
whatever is on disk right now is what goes into the image.

```bash
git log --oneline -3   # confirm the commits you expect are present
git status             # uncommitted changes WILL be included - make sure that's intended
```

Current flags already baked into `.devops/cuda.Dockerfile` (nothing to add manually):

| Flag | What it gives you |
|---|---|
| `CMAKE_CUDA_ARCHITECTURES=70` | V100 (compute capability 7.0) — the default, no build-arg needed |
| `GGML_CUDA_NCCL=ON` | NCCL AllReduce over NVLink for tensor-parallel |
| `GGML_CUDA_FA_ALL_QUANTS=ON` | mixed K/V cache quant types (e.g. `-ctk q8_0 -ctv q4_0`) |
| `GGML_CUDA_GRAPHS=ON` | harmless on V100 (CUDA graphs are disabled at runtime for cc < 8.0) |

## 2. Keep a rollback copy of the current image

```bash
docker tag llamacpp-local-v100:latest llamacpp-local-v100:previous
```

## 3. Build

```bash
docker build \
  -f .devops/cuda.Dockerfile \
  --target server \
  -t llamacpp-local-v100:latest \
  .
```

Notes:
- `--target server` builds only the `llama-server` image (what the compose files run).
- First build takes roughly **30–60 minutes** (FA_ALL_QUANTS instantiates many kernels).
  Later builds are much faster thanks to Docker layer caching, *but* any source file
  change invalidates the `COPY . .` layer, so the C++/CUDA compile always reruns.
- The UI stage needs network access once (npm packages); it is layer-cached afterwards.
- Optional: append a version tag as well, so images are identifiable later:

```bash
docker tag llamacpp-local-v100:latest llamacpp-local-v100:$(git rev-parse --short HEAD)
```

## 4. Restart the server on the new image

```bash
# stop whichever profile is running
docker compose -f docker-compose.mtp.yml down

# start it again (picks up llamacpp-local-v100:latest automatically)
docker compose -f docker-compose.mtp.yml up -d
```

(Substitute `docker-compose.nospec.yml` or `docker-compose.yml` as appropriate.)

## 5. Verify

```bash
# wait for model load (first cold load of the 27B takes ~8-9 minutes), then:
curl -s http://localhost:8097/health          # expect: {"status":"ok"}

# confirm the build revision inside the image matches your checkout:
docker compose -f docker-compose.mtp.yml logs | grep -m1 -i "build"

# quick generation smoke test + speed reading:
curl -s http://localhost:8097/completion \
  -d '{"prompt":"Say hello.","n_predict":32}' | python3 -c \
  "import json,sys; r=json.load(sys.stdin); print(r['content']); print(round(r['timings']['predicted_per_second'],1),'t/s')"
```

Expected tokens/s on 2x V100, Qwen3.6-27B, tensor split (short context):
- MTP profile: ~60–66 t/s single stream
- nospec profile: ~46–48 t/s single stream

## 6. Rollback (if something is wrong)

```bash
docker compose -f docker-compose.mtp.yml down
docker tag llamacpp-local-v100:previous llamacpp-local-v100:latest
docker compose -f docker-compose.mtp.yml up -d
```

---

## Appendix: fast iteration without rebuilding the image

For development/testing there is a persistent build container (`llamacpp-dev`,
repo mounted at `/src`, incremental build dir at `/build` with ccache — rebuilds
take ~1–3 minutes instead of an hour):

```bash
# rebuild binaries after editing source
docker exec llamacpp-dev bash -c 'cd /src && cmake --build /build -j18'

# run the freshly built server ad-hoc on port 8098 (GPUs must be free)
docker exec -d llamacpp-dev /build/bin/llama-server \
  -m /models/Qwen3.6-27B-UD-Q4_K_XL-MTP.gguf \
  -sm tensor -ts 0.5,0.5 -c 32768 -fa on -fit off \
  --host 0.0.0.0 --port 8098
```

Use this to validate changes quickly; rebuild the real image (steps above) only
when you want them in production.
