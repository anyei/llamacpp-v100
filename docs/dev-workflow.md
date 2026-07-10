# Development workflow (V100 fork)

How this fork is actually built, run, and validated day-to-day. Everything here is
specific to this box (2× V100-32GB, sm70) and the custom features that live only in
this fork (SSD streaming, meta tensor-split, MTP spec, etc.). If you are picking the
work back up cold, read this first — most of it is hard-won and not obvious from the
upstream docs.

Related references:
- [`docs/env-gates.md`](env-gates.md) — every `LLAMA_*` / `GGML_*` env toggle.
- [`docs/ssd-streaming-plan.md`](ssd-streaming-plan.md) — the streaming increment staircase.
- [`docs/perf-tuning-v100.md`](perf-tuning-v100.md) — perf levers and measured baselines.
- [`TASKS.md`](../TASKS.md) — the running log of what's done / measured / open.

---

## 1. The development image

All development runs inside a CUDA container built for **sm70 (Volta / V100)**. We do
**not** run binaries on the host — the host CUDA/toolchain does not match, and the
container pins the exact build flags this fork needs.

### Build command

```bash
docker build -f .devops/cuda.Dockerfile --target full \
  -t llamacpp-local-v100:<tag> .
```

- `--target full` produces the image that ships every tool (`llama-cli`,
  `llama-perplexity`, `llama-server`, `llama-bench`, `llama-batched-bench`, …) plus
  the Python conversion scripts.
- The Dockerfile hard-codes the flags that matter here (see `.devops/cuda.Dockerfile`):
  `-DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_NCCL=ON -DGGML_CUDA_FA_ALL_QUANTS=ON
  -DGGML_RPC=ON -DGGML_NATIVE=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DGGML_BACKEND_DL=OFF
  -DLLAMA_BUILD_TESTS=OFF`.
- `CUDA_DOCKER_ARCH` defaults to `70`. Leave it — Volta is the target.

### Cost & caching

- The build does `COPY . .` then a **clean `cmake --build`** — there is **no ccache and
  no BuildKit cache mount**, so *every* build recompiles from scratch. `GGML_CUDA_FA_ALL_QUANTS=ON`
  makes the flash-attention kernels the dominant compile cost. Budget **several minutes
  per build** and always run it in the background (`run_in_background`), then continue
  prepping the test/verification while it compiles.
- Because it is a full rebuild, **uncommitted working-tree changes are included** (the
  `COPY . .` copies them). You do *not* need to commit before building — build from the
  dirty tree, verify, *then* commit. The banner's `build : bXXXX-<sha>` shows the last
  commit only (the `APP_REVISION`), not the dirty state, so don't trust it to tell you
  whether your edit is in — trust that `COPY . .` copied it.

### Tag convention

Tag by the increment being tested, not `latest`, so A/B comparisons keep both images:
`inc3-base`, `inc3-reclaim`, `inc3-rt`, `v100-bigbro2`, … Keep the last known-good image
around; a new feature image should be diff-able against it. `:latest` is whatever was
built last and is *not* reliable — always name the tag explicitly in a run.

### 1b. Local registry — workers pull, they don't build (TASKS.md #33)

The worker boxes are far slower builders than this machine, and worker images are not
box-specific (the CUDA `ARG CUDA_DOCKER_ARCH` cross-builds any sm from here, and the
CPU image uses `GGML_CPU_ALL_VARIANTS=ON` runtime dispatch). So this box runs a local
registry and the fleet pulls from it; a worker builds locally only for a target this
box genuinely can't produce (e.g. a future Intel Arc SYCL image).

**Registry** (once, on this box): `docker compose -f docker-compose.registry.yml up -d`
— `registry:2`, `restart: always`, volume `llamacpp-registry-data`, bound to
`127.0.0.1:5000` (push side) **and** `10.5.5.1:5000` (LAN pull side only — no auth/TLS,
same trusted-network model as the RPC workers).

**Push from this box** — use `localhost:5000` (docker's TLS exemption covers only
localhost; do NOT add insecure-registries here, the dockerd restart would kill the
prod inference containers):

```bash
docker tag  llamacpp-cuda:75-mytag localhost:5000/llamacpp-cuda:75-mytag
docker push localhost:5000/llamacpp-cuda:75-mytag
```

**Pull on a worker** — one-time setup: add this box to `/etc/docker/daemon.json` and
restart dockerd (`{"insecure-registries": ["10.5.5.1:5000"]}`), then:

```bash
docker pull 10.5.5.1:5000/llamacpp-cuda:75-mytag
# same repo the localhost push created — the registry ignores the hostname part
```

Browse what's available: `curl http://10.5.5.1:5000/v2/_catalog` and
`curl http://10.5.5.1:5000/v2/<repo>/tags/list`.

**Disk discipline**: the registry volume stores its own copy of every pushed image on
this already-tight disk (~10 GB per CUDA image; the volume rides `/var/lib/docker`).
Push only images a worker will actually pull, delete stale tags, and reclaim with the
registry's garbage collector (deletes are enabled):

```bash
docker exec llamacpp-registry registry garbage-collect /etc/docker/registry/config.yml
```

### 1c. Fast iteration: incremental builds in a dev container

The image build is a clean rebuild every time (§1). For a code-compile-test loop that
is *seconds* (CPU) / *minutes-once-then-seconds* (CUDA), run a persistent dev container
with the repo mounted read-only and an out-of-tree build dir (validated 2026-07-10 —
the whole RPC feature batch was developed this way):

```bash
# CPU loop (arg parsing, RPC protocol, anything backend-agnostic):
docker run -d --name llama-devcpu --network host \
  -v "$PWD":/src:ro -v /path/to/work:/work \
  -v /mnt/models/ollama37-k80/.ollama/custom-models:/models:ro \
  ubuntu:24.04 sleep infinity
docker exec llama-devcpu bash -c 'apt-get update -qq && apt-get install -y -qq build-essential cmake git'
docker exec llama-devcpu bash -c 'cmake -S /src -B /work/build-cpu -DCMAKE_BUILD_TYPE=Release \
  -DGGML_RPC=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF && \
  cmake --build /work/build-cpu --target llama-cli ggml-rpc-server -j $(nproc)'

# CUDA loop: same pattern with the devel base image (already pulled) + --gpus all:
#   nvidia/cuda:12.8.1-devel-ubuntu24.04, cmake flags from .devops/cuda.Dockerfile
#   (-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DGGML_NATIVE=ON; skip
#   GGML_CUDA_FA_ALL_QUANTS unless testing quantized-KV FA — it dominates compile time)
```

Traps: `pkill -f ggml-rpc-server` from a `docker exec bash -c` kills the exec shell
itself (the pattern matches its own command line) — anchor it
(`pkill -f "^/work/build-cpu/bin/ggml-rpc-server"`). Background a server with
`docker exec -d`, never `nohup ... &` inside a foreground exec. These containers are
throwaway — the real gate before commit is still the production image build (§1).

---

## 2. Models & data

Models live at:

```
/mnt/models/ollama37-k80/.ollama/custom-models/
```

Mount it **read-only** into the container at `/models`. The workhorses this session:

| Model | File (basename) | Notes |
|---|---|---|
| Qwen3.6-35B-A3B (MoE, MTP) | `Qwen3.6-35B-A3B-UD-Q4_K_XL-MTP.gguf` | ~23 GB, fits 1 GPU. The **fast correctness gate** for streaming (byte-exact / PPL). |
| GLM-4.7-Flash-UD (deepseek2/MLA) | `GLM-4.7-Flash-UD-Q4_K_XL.gguf?download=true` | 17.5 GB, 47L, full 202K ctx fits 1 GPU. **Note the `?download=true` suffix — quote the whole filename.** |
| DeepSeek-V4-Flash (81 GB MoE) | `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf` | 81 GB experts, fits neither VRAM nor RAM → the **SSD-streaming reference vehicle**. |

Perplexity corpus (wikitext-2-raw) lives in the session scratchpad
(`.../scratchpad/wikitext-2-raw/wiki.test.raw`) and is mounted at `/corpus`.

---

## 3. Running the tools

The image entrypoint dispatches subcommands: `cli`, `perplexity`, `server`, `bench`, …

### Canonical single-run

```bash
docker run --rm --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v /mnt/models/ollama37-k80/.ollama/custom-models:/models:ro \
  --entrypoint /app/llama llamacpp-local-v100:<tag> cli \
  -m "/models/<MODEL>.gguf" \
  -ngl 99 --no-mmap -c 4096 -n 128 --temp 0 --seed 1 -st -v \
  -p "Explain how a CPU pipeline works, step by step." </dev/null
```

Flags that are **not optional** for scripted runs:

- **`</dev/null`** — without stdin closed, `llama-cli` drops into an interactive
  `> ` prompt and loops forever on EOF (this once produced a 90 MB "output" of spinner
  frames). Always redirect stdin from `/dev/null`.
- **`-st`** (single-turn) — generate once and exit, no chat loop.
- **`-v`** — **required to see `GGML_LOG_INFO`**, which is where all the streaming
  stats live (`ggml_ssd_stream: …`, KV/compute buffer sizes, pool allocations). Without
  `-v` the log is just the banner + generated text and you will wrongly conclude a
  feature "didn't engage." (Burned an entire profiling run on this.)
- **`--temp 0 --seed 1`** — greedy, reproducible sampling for A/B comparison.
- **`--no-mmap`** — streaming and several fork features assume non-mmap tensor loading.
- **`-ngl 99`** offloads all layers to GPU; **`-ngl 0`** is the CPU path (see the
  determinism note below).

### Selecting GPUs

- `CUDA_VISIBLE_DEVICES=0` — single GPU.
- `CUDA_VISIBLE_DEVICES=0,1` **+** `-sm tensor -ts 0.5,0.5` — tensor parallelism (meta
  backend); `-sm layer` — pipeline/layer split.
- The box is **shared with other inference tenants** — always pin the device(s) you
  intend and remember that another process may be on the other card.

### Enabling fork features

Features are **env-gated** (see `docs/env-gates.md`). Example — SSD streaming with GPU
landing and parallel reads:

```bash
-e LLAMA_SSD_STREAM_BUFFER=1 -e LLAMA_SSD_STREAM_BUDGET=30000 \
-e LLAMA_SSD_STREAM_GPU=1   -e LLAMA_SSD_STREAM_VRAM_BUDGET=14000 \
-e LLAMA_SSD_STREAM_READ_THREADS=4 -e GGML_SSD_STREAM_DEBUG=1
```

There are also `--ssd-streaming` / `--ssd-stream-gpu` / `--ssd-stream-budget` /
`--ssd-stream-vram-budget` CLI flags that set the envs for you.

---

## 4. Gotchas / hard-won lessons

These are the traps that have cost real time. Internalize them.

1. **`-v` gates the INFO logs.** No `-v` ⇒ no `ggml_ssd_stream:` lines ⇒ you can't tell
   if the feature engaged. Always `-v` when inspecting behavior.
2. **`</dev/null` or the CLI hangs interactively.** Non-negotiable for scripts.
3. **`nvidia-smi memory.used` is GPU-global.** On a shared box it includes other
   tenants, so it is **useless as an isolated VRAM measurement**. For real per-run VRAM,
   read the tenant-independent numbers from the load log with `-v`: `CUDA0 model buffer
   size`, `CUDA0 KV buffer size`, `CUDA0 compute buffer size`. (These are how the
   `input_cpy` reclaim was measured: `407.00 → 73.69 MiB`, not by polling smi.)
4. **`-ngl 99` GPU-MoE decode is run-to-run non-deterministic.** Near-tied expert
   routing flips depending on non-deterministic GPU reduction order, so **greedy temp-0
   text can differ between two identical runs** — a byte-for-byte sha mismatch does *not*
   prove a code change altered the math. Confirm by running the *same* config twice; if
   it self-differs, the mismatch is noise. Use **PPL** (below) as the real gate.
5. **Byte-exactness is only meaningful on `-ngl 0` (CPU) with `LLAMA_SSD_STREAM_SERIAL=1`**
   — the deterministic path. GPU-landing features have no deterministic path (they need
   the GPU), so PPL is their gate.
6. **Concurrent background jobs must not write the same output paths.** Two runs tee-ing
   to the same file corrupt each other's results. Use unique per-run tags; run
   GPU-contending jobs **sequentially**, not in parallel.
7. **81 GB DeepSeek runs are slow** (decode ~1.5–2.6 t/s) and load ~9 s. A 96-token
   probe is several minutes. Batch what you need into one run; don't spin many.
8. **`?download=true` in a model filename** must be quoted everywhere (`"$MODEL"`), or
   the shell treats `?` as a glob and `&`/`=` oddly.

---

## 5. Correctness gates

A change to a compute or memory path is **not done until it passes a correctness gate**.
Pick the gate by what the change can and cannot affect:

- **Byte-exact (strongest, deterministic paths only).** `-ngl 0` CPU +
  `LLAMA_SSD_STREAM_SERIAL=1`, temp-0, compare a `sha256` of the generated text against
  the resident/reference run. Use when the path is deterministic.
- **PPL (the workhorse for GPU paths).** `perplexity -f /corpus/wiki.test.raw --chunks N`
  (16 chunks is enough for a 4-decimal match; the fork's headline comparisons use ~40).
  PPL averages over the routing noise, so **identical PPL == quality-neutral** even when
  greedy text differs. Example gate for the `input_cpy` reclaim: reclaim ON `7.0898` ==
  OFF `7.0898`.
- **A/B via kill-switch.** Every new feature ships with an env to disable it
  (`LLAMA_SSD_STREAM_GPU_NO_RECLAIM=1`, `LLAMA_SSD_STREAM_READ_THREADS=1`, …). Run ON vs
  OFF on the *same* image and compare — this isolates the feature from every other
  variable (build, model, box load).
- **Controls to localize a bug.** When streaming produced PPL garbage, the `-cmoe`
  control (put experts on a *normal* CPU buffer, same partial offload) scored correct,
  which pinned the fault to the streamed-copy path rather than the offload. Reach for a
  control that changes exactly one thing.
- **Coherence smoke.** For a quick "did I break generation," eyeball the `-st` output —
  it should be fluent, on-topic prose. Garbage/repetition = something is wrong.

Run the **cheap, fast gate first** (Qwen PPL, or Qwen byte-exact) before spending a slow
DeepSeek run. Qwen fits one GPU and is the quick canary; DeepSeek is the extreme.

---

## 6. Performance measurement

- **Prefer tenant-independent metrics.** `CUDA0 compute buffer size` (arena), the
  internal `ggml_ssd_stream:` timers (`read … H2D-issue …`), and hit-rate counters are
  isolated from other GPU tenants; `nvidia-smi` totals are not.
- **Instrument before optimizing.** Before writing the parallel-read path, an env-gated
  timer split the miss path into *SSD read (26.7 s)* vs *H2D-issue (12.2 s)* — which is
  what justified attacking the reads first and not the H2D. Measure where the time is;
  don't guess.
- **t/s is noisy on a shared box.** Report generation t/s from the `[ Prompt: … |
  Generation: … ]` line, but treat small deltas as noise and lean on the isolated
  internal metrics for the real signal. Idle the other GPU when you can.
- **Warm up.** GPU-landing and the caches warm over the first ~tens–hundreds of tokens;
  a 40-token run understates steady-state. Use `-n 96`+ for a cache-warming perf read.
- **Negative results are results.** This project records losers (adaptive draft cap,
  AllReduce overlap, page-cache steering, SLRU-on-GPU-pool) with the measured numbers in
  `TASKS.md`/the plan doc so nobody re-attempts them. If a lever doesn't pay, write down
  *why* and move on.

---

## 7. The iterative loop

The actual per-change cycle:

```
 1. edit  ──────────────────────────────────────────────────────────────┐
 2. build in background (docker build … -t …:<new-tag>)  ~minutes        │
 3. while building: write/stage the verification script                  │
 4. correctness gate FIRST (cheap: Qwen PPL / byte-exact ON vs OFF)      │
 5. if correct → perf measurement (sweep the knob; isolated metrics)     │
 6. if win + neutral → commit (code + plan-doc writeup + TASKS.md)       │
    if regression/negative → record the negative result, revert, ────────┘
                             or keep env-gated-off with a note
```

Principles that keep this safe and fast:

- **Gate new behavior behind an env, defaulted to the old behavior.** A new feature
  should be a *no-op* unless explicitly enabled (`READ_THREADS` default `1`,
  `NO_RECLAIM` off, etc.). This makes every change (a) shippable immediately, (b)
  A/B-testable against itself, (c) instantly revertible in prod without a rebuild.
- **Correctness before speed, always.** A faster wrong answer is worthless. Never report
  a perf number for a config that hasn't passed its correctness gate.
- **One variable at a time.** Same image, same model, same box state — flip only the
  feature env. Cross-image or cross-model comparisons hide confounds.
- **Build from the dirty tree, commit after green.** Don't commit speculative code; get
  the gate green first, then commit with the measured numbers *in the message* and a
  writeup in the plan doc / `TASKS.md`.
- **Write the measurement down where the next person will look.** `TASKS.md` for the
  running log, the plan doc for the design + result, commit message for the headline
  numbers. Future-you will not remember why `SLRU` is off by default unless it's written.
- **Don't re-run a search you already delegated.** Wait for the background build/run to
  notify; prep the next step meanwhile instead of polling.

---

## 8. Scratchpad harness scripts

Reusable run harnesses live in the session scratchpad (regenerate as needed):

- `ssd_run.sh TAG MODEL NGL GPUS PROMPT NGEN "ENVS" [extra…]` — one streamed run, strips
  spinner frames, prints t/s + hit-rate + a text `sha`.
- `ppl.sh` / `ppl_reclaim.sh` — resident-vs-streamed (or ON-vs-OFF) perplexity gate.
- `verify_reclaim2.sh` — byte-exact + compute-buffer + pool-sizing gate for a feature.
- `ds_readthreads.sh` — a knob sweep on DeepSeek with the internal timer breakdown.

Pattern for all of them: `docker run --rm --gpus all … --entrypoint /app/llama <img>
cli … </dev/null > out.txt 2>&1`, then `grep -aE` the `ggml_ssd_stream:` / `[ Prompt:`
lines out of the log. Poll a background run's condition rather than `sleep`-ing.

---

## 9. Quick reference

```bash
# Build (background; ~minutes; full recompile)
docker build -f .devops/cuda.Dockerfile --target full -t llamacpp-local-v100:mytag .

# Fast correctness canary: Qwen PPL, feature ON vs OFF (same image)
docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES=0 \
  -e LLAMA_SSD_STREAM_BUFFER=1 -e LLAMA_SSD_STREAM_GPU=1 \
  -v /mnt/models/ollama37-k80/.ollama/custom-models:/models:ro \
  -v .../wikitext-2-raw:/corpus:ro \
  --entrypoint /app/llama llamacpp-local-v100:mytag perplexity \
  -m "/models/Qwen3.6-35B-A3B-UD-Q4_K_XL-MTP.gguf" -ngl 99 --no-mmap \
  -f /corpus/wiki.test.raw --chunks 16 </dev/null 2>&1 | grep -aE "Final estimate"

# Perf read with the streaming stats (note -v)
docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES=0 -e GGML_SSD_STREAM_DEBUG=1 \
  -e LLAMA_SSD_STREAM_BUFFER=1 -e LLAMA_SSD_STREAM_GPU=1 -e LLAMA_SSD_STREAM_VRAM_BUDGET=14000 \
  -v /mnt/models/ollama37-k80/.ollama/custom-models:/models:ro \
  --entrypoint /app/llama llamacpp-local-v100:mytag cli \
  -m "/models/<MODEL>" -ngl 99 --no-mmap -c 4096 -n 96 --temp 0 --seed 1 -st -v \
  -p "..." </dev/null 2>&1 | grep -aE "GPU cache hit=|miss-path|CUDA0 compute buffer|Prompt:"
```
