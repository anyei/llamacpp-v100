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
After any `docs/env-gates.md` edit also run `python3 scripts/gen-wizard-gates.py`
(GATES catalog, same file + same cp); it WARNs on rows missing from its CLASS
map - add them, an unclassified row ships as "diag (unclassified)".

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

## Kepler (sm_37, K80 box) image variant

```bash
nohup docker build -f .devops/cuda.Dockerfile --target server \
  --build-arg CUDA_VERSION=11.8.0 --build-arg UBUNTU_VERSION=22.04 --build-arg GCC_VERSION=11 \
  --build-arg CUDA_DOCKER_ARCH=37 --build-arg PURGE_CUDA_COMPAT=1 \
  -t llamacpp-local-v100:$(git rev-parse --short HEAD)-kepler . > /tmp/build-kepler.log 2>&1 & disown
```
- ~7 min, 5.1 GB. `PURGE_CUDA_COMPAT=1` removes the base image's cuda-compat
  libcuda: it outranks the host driver in ldconfig order and on driver 470
  (K80) CUDA then enumerates ZERO devices while nvidia-smi still works.
- Alias `:kepler` (local + registry) mirrors `:latest` for the K80 box; the
  local K80 launcher runs `llamacpp-local-v100:<sha>-kepler` with `MODELS_DIR`
  re-passed (compose reverts it otherwise). The worker image needs no Kepler
  variant. The harness classifies launcher recreates as production deploys -
  expect one denial, re-run after the user's explicit go.
- State 2026-10-01 ~19:00 (#156 7.4 roll): `:latest` = 1300d2bef-156p4 (sha256:57d5bf36; 156p3 + the exact-plan
  allocator rule GGML_ALLOC_EXACT_PLAN default on; rollback 1300d2bef-156p3), `:kepler` = 1300d2bef-156p4-kepler
  (sha256:9b1ba57b; rollback 1300d2bef-156p3-kepler). Both launchers recreated and verified (X99: 23 models, V100,
  wizard 145 gates, Flash-Next config gateOn keeps LLAMA_QSA_BLOCK_TOPK=1; K80 box: 74 models, both dies). Old tags
  1300d2bef-156p (both boxes) and 1300d2bef-156p2-kepler removed (registry keeps them). Build-cache prune after the
  builds (16.2 GB, user's OK); / at 94 %. X99 bench156/arm.sh now uses 1300d2bef-156p3 as its runtime image.
- State 2026-10-01 ~17:50 (#156 7.3 roll): `:latest` = 1300d2bef-156p3 (sha256:7591b92b; #156 port + PR #26385 + 7.5 +
  the 7.3 block-level QSA top-k, uncommitted tree; rollback 1300d2bef-156p), `:kepler` = 1300d2bef-156p3-kepler
  (sha256:7d3cf363; rollback 1300d2bef-156p2-kepler). Both launchers recreated on the pinned tags and verified (X99: 23
  models, V100, wizard knows LLAMA_QSA_BLOCK_TOPK; K80 box: 74 models, both dies). X99 saved Flash-Next config
  55366197782 gateOn += LLAMA_QSA_BLOCK_TOPK=1 (backup wizard-configs.json.bak-20261001-qsa). Old tags removed:
  063bdc2a0-db (coordinator + X99), e117ee884-widefix-kepler (registry copies kept). Disk: two user-approved
  build-cache prunes around the two builds (16.4 + 12.8 GB); / at 94 % after. Each coordinator + Kepler build pair
  adds ~13 GB of build cache - prune (with the user's OK) before and after.
- State 2026-10-01 (#156 roll): `:latest` = 1300d2bef-156p (sha256:26ed6b8d; the #156 port + upstream PR #26385
  softmax-race fix on HEAD 1300d2bef, uncommitted tree; rollback 063bdc2a0-db), `:kepler` = 1300d2bef-156p2-kepler
  (same tree + the 7.5 BF16->F32 cuBLAS fix for cc < 6.0, which makes MTP work on the K80; rollback
  e117ee884-widefix-kepler). Both launchers recreated on the pinned tags and verified (X99: 23 models, V100 visible;
  K80 box: 74 models, both dies). X99 saved Flash-Next config 55366197782 now runs CUDA graphs ON (gateOff
  GGML_CUDA_DISABLE_GRAPHS; backup wizard-configs.json.bak-20261001). The V100 image predates the 7.5 line (no effect
  at cc 7.0). Old tags removed: 99b3c21ee-q2avx2 (coordinator + X99), e117ee884-kepler, test images keplerfix /
  156p-kepler. Disk: / still 97 % after two Kepler builds (build cache pruned once, 25 GB, user's call).
  The harness REFUSES launcher recreates (production deploy) until the user says go in the conversation - even a
  `docker ps | grep llama-launcher` was refused once the recreate had been denied.
- State 2026-09-30 eve: `:latest` = 063bdc2a0-db (sha256:a47b290d, the #154 PORT on HEAD 063bdc2a0, committed by the user as 1300d2bef: MoE doorbell +
  prefill stream ring, both env-gated default off; rollback 99b3c21ee-q2avx2; t3k2/t3k3 removed locally and on the X99,
  still in the registry), `:kepler` = e117ee884-widefix-kepler (rollback e117ee884-kepler; no Kepler rebuild - the
  doorbell needs a CUDA device with pinned host memory, the K80 box would need its own gate). X99 launcher recreated on
  the pinned tag, verified; the saved Flash-Next config (55366197782) turns on `LLAMA_MOE_DOORBELL=1`,
  `GGML_SCHED_PREFILL_STREAM=1` and `-ub 4096 -b 4096` (backup `wizard-configs.json.bak-20260930b`).
  TRAP (2026-09-30): the coordinator box's / is at 97 % (build cache ~90 GB, ~23 GB reclaimable); check `df -h /`
  before any image build.
  RPC: the port kept the op-set fingerprint (patch 3) - the two new ops are appended and never sent (the RPC client
  refuses them); deployed workers verified compatible (loopback gate sha c80261ff), no worker roll needed.
  Suffixed tags (`<sha>-t3k`, `<sha>-widefix`) = built from a working tree the
  user had not committed yet; the X99 launcher compose lives in
  `/home/anyei/server/services/llamacpp-v100/` (defaults match the running
  config, only `COORD_IMAGE` needs pinning).
  cc 3.7 has no dp4a: the MoE cache is speed-neutral there (correctness only).

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
#   TRAP (2026-09-28): X99 holds ONLY the registry-prefixed tags - a bare
#   `docker run llamacpp-local-v100:<tag>` there tries Docker Hub (exit 125);
#   always name `10.5.6.1:5000/llamacpp-local-v100:<tag>` in run/compose.

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
Gate legs on the X99 without an image build: ship `build-cuda75/bin` there
(the `devbins151` pattern) and run inside the launcher image with
`-v <devbins dir>:/devbins:ro -e LD_LIBRARY_PATH=/devbins --entrypoint
/devbins/llama-perplexity` - dev libs == image libs on the stock path was
verified 0.000000 KLD (moe-cache-plan 12.14).
