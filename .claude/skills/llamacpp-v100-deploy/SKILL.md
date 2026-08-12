---
name: llamacpp-v100-deploy
description: Build and roll out the llamacpp-v100 fork's docker images locally - coordinator (llamacpp-local-v100), CPU RPC worker (llamacpp-cpu), and the launcher/wizard service - including registry push, worker recreation on all boxes, tag hygiene, and verification. Use for any image rebuild, registry push, worker upgrade, or launcher roll.
---

# Deployments: images, registry, rollouts

## The three images

| Image | Dockerfile | Target | Contents |
|---|---|---|---|
| `llamacpp-local-v100:<sha>` | `.devops/cuda.Dockerfile` | `server` | coordinator llama-server, V100 (arch 70), NCCL, FA_ALL_QUANTS, embedded web UI incl. wizard.html |
| `llamacpp-cpu:rpc-worker-<sha>` | `.devops/cpu.Dockerfile` | `rpc-worker` | `ggml-rpc-server` only |
| launcher | (same coordinator image) | — | run via `docker-compose.launcher.yml` in router mode |

Tag with `git rev-parse --short HEAD`. `docker build` copies the WORKING TREE —
check `git status` first; uncommitted changes ship.

## Build (always detached — the ~10-min background cap kills clients)

**MANDATORY pre-build gate (#96):** `python3 scripts/gen-wizard-flags.py --check`
must pass before any coordinator image build — a stale FLAGS catalog vs arg.cpp
fails the build here, not silently in the shipped wizard. On drift: run the
generator, `cp tools/ui/static/wizard.html tools/ui/dist/wizard.html`, commit.

```bash
nohup docker build -f .devops/cuda.Dockerfile --target server \
  -t llamacpp-local-v100:$(git rev-parse --short HEAD) . > /tmp/build.log 2>&1 & disown
# watch by ARTIFACT, not by process: loop `docker image inspect <tag>` in a
# <10-min background watcher; re-arm until IMAGE_READY
```
- Coordinator: ~10 min warm layer cache (only common/server changed), 35-60 min
  when ggml/CUDA sources changed. Worker image: ~5-10 min.
- Verify features in the built image before rolling:
  `docker run --rm --entrypoint bash <img> -c 'grep -l GGML_RPC_WIRE_Q8 /app/libggml-rpc.so'`
  (the `libcuda.so.1` error without `--gpus` is expected noise).

## Registry (10.5.5.1:5000)

- Push via the `127.0.0.1:5000/...` alias — pushing to `10.5.5.1:5000` directly
  FAILS (https-vs-http; the daemon only trusts 127.0.0.0/8):
```bash
docker tag <img>:<sha> 127.0.0.1:5000/<img>:<sha>
docker push 127.0.0.1:5000/<img>:<sha>
curl -s http://127.0.0.1:5000/v2/<img>/tags/list   # verify
```
- X99 (10.5.6.2) pulls over the 10G as `10.5.6.1:5000/<img>:<tag>` (its
  daemon.json trusts that address; the coordinator wlan/192.168.68.67 path is
  dead). **TRAP (2026-08-12): docker-29 + compose-5.4 `compose pull` WEDGES
  against registry:2.8.3** (referrers-API 404 retry loop) - `docker pull` the
  exact tag directly first, then `compose up`; or upgrade the registry to :3.
- Worker boxes PULL as `10.5.5.1:5000/<img>:<tag>`. A NEW box needs the
  insecure-registry config in `/etc/docker/daemon.json` (merge into existing
  keys, then `systemctl restart docker` — restart KILLS running containers,
  do it in the same window as the worker recreate):
  `{"insecure-registries": ["10.5.5.1:5000"]}`
  The coordinator box itself does NOT need it (loopback push alias).
- **Moving `:latest` aliases (since 2026-08-06)**: `llamacpp-local-v100:latest`
  and `llamacpp-cpu:rpc-worker-latest` point at CURRENT, locally AND in the
  registry; the compose defaults resolve to them (launcher `COORD_IMAGE`
  default = `llamacpp-local-v100:latest`, worker `WORKER_IMAGE` default =
  `llamacpp-cpu:rpc-worker-latest`; the bare `rpc-worker` tag is retired).
  **Every roll MUST re-tag and re-push both aliases** or they silently rot —
  a stale `:latest` was exactly the footgun that prompted this convention.
- Tag hygiene: keep exactly CURRENT + one ROLLBACK per image locally (plus their
  registry aliases and the `:latest` aliases); `docker rmi` the rest. Only
  `llamacpp*`/`ds4` images may ever be deleted — never base/infra/upstream
  images. Disk target: keep / below ~90%.

## Worker rollout (NEVER while a fleet serve is up — restarts drop the serve)

Current standing config (sweep-tuned): local `-t 8`, .11 `-t 6` (P-cores only).

```bash
# local worker (compose project = THE REPO dir; every var must be re-passed or
# compose reverts it - the WORKER_PORT trap has burned real sessions)
WORKER_PORT=50053 WORKER_CACHE_LIMIT_MIB=98304 \
  docker compose -f docker-compose.rpc-worker-cpu.yml up -d --force-recreate

# .11 (ssh anyei@10.5.5.11; compose at ~/server/llama/llamacpp-v100)
ssh anyei@10.5.5.11 'cd ~/server/llama/llamacpp-v100 && \
  WORKER_IMAGE=10.5.5.1:5000/llamacpp-cpu:rpc-worker-<sha> \
  WORKER_CACHE_LIMIT_MIB=131072 WORKER_THREADS=6 \
  docker compose -f docker-compose.rpc-worker-cpu.yml pull -q && \
  WORKER_IMAGE=... WORKER_CACHE_LIMIT_MIB=131072 WORKER_THREADS=6 \
  docker compose -f docker-compose.rpc-worker-cpu.yml up -d --force-recreate'

# .15 (:50055): USER-ONLY box - no ssh works from here. Ask the user; workers on
# older protos interoperate per-connection (wire ladder + zero-reply degrade
# gracefully), so .15 lagging an image generation is fine. One-liner for them:
#   WORKER_IMAGE=10.5.5.1:5000/llamacpp-cpu:rpc-worker-latest \
#     docker compose -f docker-compose.rpc-worker-cpu.yml up -d --force-recreate
# (plus the box's usual port/thread vars)
```
# X99 (10.5.6.2, 2x V100 SXM2 32GB + 251GB RAM; verified 2026-08-12): TWO
#   workers - :50052 CUDA (the COORDINATOR image run as an arch-70 worker,
#   CUDA_VISIBLE_DEVICES=0,1; baked healthcheck reads "unhealthy" - cosmetic)
#   and :50054 CPU (rpc-worker image, -t 30). A multi-GPU worker consumes
#   SUCCESSIVE RPC device names in endpoint order - put the RAM endpoint
#   first in --rpc if RPC0 must be the 256GB device.
#   Pulls via 10.5.6.1:5000 (see registry TRAP).

Verify after recreate: `docker inspect llama-rpc-worker-cpu --format '{{.Config.Image}} {{.Config.Cmd}}'`
(check image tag, `-p`, `-t`). Workers re-benchmark ~15-20 s before listening.
Box facts: local 78 GB RAM / 20 cores; .11 62 GB RAM (its "128G" is the DISK
cache limit) / 16 cores hybrid 6P+8E; .15 64 GB, ~28 GB/s, slowest.

## Launcher / wizard roll

```bash
docker rm -f llama-launcher
COORD_IMAGE=llamacpp-local-v100:<sha> docker compose -f docker-compose.launcher.yml up -d
# (bare compose up uses the :latest alias - fine ONLY right after a roll
#  re-pointed it; pin the sha when rolling to be exact)
```
- The launcher-managed production serve dies with the launcher container —
  recreating it costs a full model reload (~20-30 min V4); bundle launcher
  rolls with a serve-restart window and relaunch the serve after (POST
  /models/load with the exact captured args/env).
- `llamacpp_launcher-cache` volume persists the user's saved model dirs — never
  delete it. Compose is host-network with a PORT-aware healthcheck (the image's
  baked healthcheck probes :8080 and reads unhealthy otherwise).
- Verify: `curl -s --compressed http://127.0.0.1:8399/wizard.html | head -c 100`
  (embedded assets are gzipped — without `--compressed` you get 415),
  `/wizard/dirs` (saved dirs intact), `/models` (count sane), `/wizard/hw`
  (GPUs + `discovered` workers; the beacon merge holds 90 s, first call after a
  cold start may be short one worker).

## Coordinator serve binary vs image

Fleet serves run the DEV binary (`build-cuda75` bind-mounted), not the image —
so a serve picks up changes after `cmake --build /work/build-cuda75`, no image
rebuild needed. Images matter for: the launcher/wizard, compose-based production
serves, and worker boxes. Rebuild the image when server/common/UI changes should
reach those.
