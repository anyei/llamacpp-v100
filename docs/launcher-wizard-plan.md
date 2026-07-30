# Launch wizard: guided model serving on the router (design + increments)

Status: SHIPPED 2026-07-29 (increments 1-3 + extras; section 5 = as-built,
section 6 = #78 env-gate catalog + 2026-07-30 launch-button fix pass). Originally design v1,
2026-07-28. UX spec = the "Model Launch Wizard" artifact
(interactive mock, draft 3): 3 steps (model -> run mode -> launch), smart
recommendations from model metadata + hardware, measured-over-estimated
speeds, full env-gate surface (mode cards + advanced drawer).

## 1. What already exists (survey 2026-07-28)

The in-tree multi-model ROUTER covers most of the machinery:

- Router mode = `llama-server` started with NO model; `--models-dir` scans
  `*.gguf` (subdirs pair `mmproj*` automatically), presets carry per-model
  args, `POST /models/load` spawns a child server on a free port and proxies
  to it, `GET /models/sse` streams status. Child lifecycle, LRU unload,
  log capture, orphan cleanup: all done (server-models.cpp).
- `GET /models` has the exact hook point for richer metadata: the TODO at
  server-models.cpp:1697 ("may require reading GGUF metadata").
- Offline gguf HEADER reads (no tensor data, no backend touch) are an
  established pattern: common/arg.cpp:1415-1445 reads arch, block_count,
  context_length, head counts and computes KV bytes/token - the wizard's
  sizing routine essentially exists and needs lifting into a shared helper.
- A self-contained static page is served with ZERO C++ changes by dropping
  it in tools/ui/static/ (loading.html precedent) - embedded, API-key-exempt.
- `GET /fleet/status` already returns per-device memory, scores, reachability
  and beacon-DISCOVERED workers - but it is NOT wired in router mode (runs
  against the router's empty context; devices come back empty).

Consequences for the design:

- **No LLAMA_LAUNCHER env needed.** Router mode IS the gate. The entry
  compose is just: router mode + `--models-dir /models` + host/port.
- **Launching = presets.** The wizard composes per-model args and loads via
  the existing spawn path; no new process machinery.

## 2. Entry compose

```yaml
# docker-compose.launcher.yml (increment 2)
services:
  llama-launcher:
    image: llamacpp-local-v100:latest
    network_mode: host
    volumes: [ "${MODELS_DIR:-/mnt/files}:/models:ro" ]
    entrypoint: /app/llama-server
    command: --models-dir /models --host ${HOST:-0.0.0.0} --port ${PORT:-8080}
```

Wizard at `http://host:port/wizard.html`; the normal chat UI stays at `/`.

## 3. Increments

### inc-1: gguf metadata in GET /models  (this session)

- New helper (common/gguf-meta.h or server-side): offline header read ->
  { size_bytes, arch, n_ctx_train, block_count, n_expert, kv_bytes_per_token,
    rope_scaling_orig_ctx?, has_nextn (MTP head) }. Reuse the
  common/arg.cpp:1415 pattern (gguf_init_from_file no_alloc).
  MTP detection: nextn/MTP tensors or arch-specific key (verify against
  hy3 + Qwen3.6-MTP ggufs which are on disk).
- Populate in `server_model_meta` next to update_caps() (models-dir +
  cache sources; lazy on first /models call or at scan, measure cost - a
  header read is ms-scale, dirs are small).
- Emit under `"metadata"` in get_router_models (the :1697 TODO).
- Family pairing surfaced: `"pairs": { mmproj: <name|null>, dflash: <name|null> }`
  by filename-family fuzzy match (mmproj subdir pairing already exists;
  add top-level fuzzy fallback: qwen >= 3.5 vision-capable rule lives in
  the UI, the API just reports what files matched).
- Gate: single-model mode untouched; router /models JSON adds keys only.

### inc-2: wizard.html + launch-with-args

- Port the mock to tools/ui/static/wizard.html; replace demo SCAN with
  GET /models?reload=1, keep all selection logic client-side.
- Launch: extend `POST /models/load` with optional `"args": ["-c","65536",...]`
  overlaid on the model's preset (after reserved-args stripping; router CLI
  overlay order preserved). SSE drives the launch progress states the mock
  fakes today.
- Serve the fleet composes case (distributed modes) v1 as: emit the exact
  `EXPERT_DEFER=1 ... ./run-ep-fleet-hy3-spec.sh`-equivalent command for
  copy/paste; child-spawned fleet serving needs the router to pass RPC env
  (works - children inherit environ) but keep it explicit first.

### inc-3: hardware + fleet discovery for the router

- New `GET /wizard/hw`: RAM (/proc/meminfo), models-dir disk stats,
  discovered RPC workers (rpc_discover_listen - network-only, respects the
  router's no-GPU invariant), and GPU inventory via a short-lived PROBE
  child (the router must never init CUDA; a child --probe-hw one-shot that
  prints device name/total/free and exits, cached).
- Alternatively proxy /fleet/status to a loaded child when one exists.

### inc-4: measured-speed registry + task folds

- sweeps.json in the models dir (or server cache): per model+mode measured
  t/s; wizard overlays "measured" over "est." exactly like the mock.
- Fold #73 (worker score staleness: re-bench idle / bench-at-connect) and
  #68 (auto-weight must distrust integrated-GPU memory reports) into the
  fleet-card data path.

## 4. Gates

- inc-1: build clean; router smoke on a dir containing hy3 + Qwen3.6-MTP +
  mmproj + dflash files -> metadata JSON correct (moe/mtp/nctx/pairs);
  single-model serve byte-untouched (no code path shared).
- inc-2: end-to-end launch of a small model (gemma-4b) from the wizard on
  the dev box; child appears in /models with wizard args; chat works.
- inc-3: /wizard/hw returns GPUs+RAM+beacons on the V100 box with the
  production fleet up (workers must appear via beacons).
- Every increment: the wizard page must render with JS disabled-gracefully
  (plain /models JSON link) and in both themes.

## 5. As-built (2026-07-29, commits f4414d169..597270e8e)

Landed beyond the original increments:

- `--models-dir` accepts a comma list (earlier dir wins on name collisions,
  bad dirs warn-and-skip) and the scan RECURSES: nested folders emit every
  gguf (shard sets collapse to -00001- with the suffix stripped; one-model
  dirs keep the dir name + mmproj pairing; multi-model dirs emit by filename
  and surface their mmproj separately). dflash files classify as kind=draft.
- Runtime sources: GET/POST /wizard/dirs with persistence in the llama
  cache (router-models-dirs.txt; CLI flag beats the UI list) + mounted-
  storage suggestions from /proc/mounts (real fs, directories only,
  scaffolding mounts and device aliases filtered, shallow gguf counts).
  The UI opens on the sources screen when nothing is scanned; the header
  "sources" button reopens it.
- GET /wizard/hw: RAM, per-dir disk, GPUs via nvidia-smi subprocess (the
  router never inits CUDA), workers via beacon listen (k=v parsed).
- GET /wizard/sweeps: merged sweeps.json from every source dir (measured
  t/s per model|mode|spec keys; /mnt/files/sweeps.json seeded).
- POST /models/load: extra_args + extra_env overlay (LLAMA_SERVER_* refused).
- wizard.html embedded in the image UI assets (no-cache so browsers track
  updates); docker-compose.launcher.yml = image + /models + host /mnt tree
  (ro, rslave) + launcher-cache volume + HOST/PORT.
- Coordinator images: 64c68933b (first embed) -> 0ad6f9327 (no-cache) ->
  597270e8e (recursive scan + mount filter). Registry pushes go via the
  127.0.0.1:5000 alias (the daemon only trusts 127.0.0.0/8 as insecure).

## 6. #78 env-gate catalog + launch-button fix pass (2026-07-30)

- Environment-gate catalog (TASKS #78): `scripts/gen-wizard-gates.py` parses
  docs/env-gates.md into 95 gates embedded in wizard.html (GATES-CATALOG
  markers; re-run after doc edits). Advanced drawer group with class chips
  (serve/tune/diag/danger/worker), value editors, mode-stack pre-seeding,
  danger hard-confirm (incl. GGML_META_EXPERT_DEFER=2 value-level), worker
  rows inert-annotated. Overlay rewrites the resolved flags -> rides the
  existing extra_env; no server change.
- Launch button verified end-to-end (headless chromium vs a real router,
  Qwen3-0.6B: Launch -> loaded -> ready link, 0 JS errors). Fixed on the way:
  `-ncmoe auto` invalid -> computed layer estimate (metadata now carries
  block_count + gguf path); mmproj/dflash substring family match (qwen3 vs
  qwen3.5 n_embd crash) -> exact equality; `--ctx-shift` -> `--context-shift`;
  hardcoded /models paths -> metadata paths; fleet --rpc from discovered
  beacons; Copy button actually copies; <meta charset>; details drawers
  survive re-render; trunc vehicles annotated.

Later same day: "Running now" panel on the model step (loaded/loading models
with status chip, open-chat link, Unload/Stop button posting /models/unload
with an async-shutdown poll; live rows tinted in the list); gates catalog
v100/upstream badges (from the doc's *(upstream)* markers).

Remaining polish (none blocking):

- wizard link from the main webui; fleet-mode launches now launch through the
  router like any mode (flags parse-validated) but a real fleet launch from
  the wizard is still unproven live.
- fold #73 (score staleness) + #68 (auto-weight iGPU trust) into the fleet
  card data path.
