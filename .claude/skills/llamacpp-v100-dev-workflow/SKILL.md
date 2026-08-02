---
name: llamacpp-v100-dev-workflow
description: Develop and gate changes for the llamacpp-v100 fork (true-parallel-inference work) using the docker dev containers - incremental builds, loopback gate harnesses, byte-identity protocol, debug instruments, and the process traps that repeatedly cost hours. Use whenever editing C++/CUDA in this repo or running loopback correctness gates.
---

# Dev workflow: containers, builds, gates

## Layout

| What | Where |
|---|---|
| Repo (source of truth) | `~/server/git-projects/llama.cpp` — mounted **read-only** at `/src` in the dev container |
| Build trees | `~/server/git-projects/llama.cpp-work` (host bind) = `/work` in the container: `build-cpu` (CPU-only, fast iteration) and `build-cuda75` (V100 arch 70, the fleet-serve binary) |
| Dev container | `llama-devcuda` (long-running; `docker exec` into it). GPUs visible. `fuser` is NOT installed. |
| Loopback gate scripts | `/work/71-*.sh` (host: `llama.cpp-work/71-*.sh`) |
| Trunc test vehicle | `/work/hy3-trunc5-mtp.gguf` + `--override-kv hy_v3.block_count=int:5` + env `LLAMA_TRUNC_ARR=1` (reports n_layer **4** to placement validation, not 5) |

## Build commands

```bash
# CPU build, incremental (~1-3 min) - the iteration loop
docker exec llama-devcuda bash -c 'cd /src && cmake --build /work/build-cpu -j12 --target llama-server 2>&1 | grep -cE " error"; true'

# CUDA fleet binary (rebuild BEFORE any fleet serve after meta/rpc changes; can
# exceed 120s - run in background)
docker exec llama-devcuda bash -c 'cd /src && cmake --build /work/build-cuda75 -j18'
```

- `--target llama-server` rebuilds all deps (ggml libs) — sufficient for meta/rpc/llama changes.
- **Verify the binary carries a new feature before any measured leg** — an env the
  binary doesn't know is silently ignored and the leg measures the baseline:
  `strings /work/build-cuda75/bin/libggml-base.so | grep -c MY_NEW_ENV`
- Host can run `/work/build-cpu/bin/*` directly:
  `W=~/server/git-projects/llama.cpp-work/build-cpu/bin; LD_LIBRARY_PATH=$W $W/llama-server ...`
- The build dir may hold stale lib generations (`.so.0.15.3` next to `.so.0.17.0`) —
  the `.so.0` symlinks decide; check `ls -la /work/build-cpu/bin/ | grep libggml` if
  behavior looks stale.

## Loopback gate harness (the correctness protocol)

Three loopback RPC workers + the trunc vehicle, all inside `llama-devcuda`:

```bash
pkill -x ggml-rpc-server 2>/dev/null; pkill -x llama-server 2>/dev/null; sleep 1
for p in 50901 50902 50903; do /work/build-cpu/bin/ggml-rpc-server -H 127.0.0.1 -p $p -t 4 > /dev/null 2>&1 & done
sleep 2
LLAMA_META_EP_ONLY=1 LLAMA_META_ATTN_OWNER=0 LLAMA_META_LOCAL_DRAFT=0 LLAMA_TRUNC_ARR=1 \
GGML_META_BCAST_FUSE=2 GGML_RPC_WIRE_F16=1 GGML_RPC_WIRE_Q8=1 GGML_META_BOUNDARY_STATS=1 \
/work/build-cpu/bin/llama-server -m /work/hy3-trunc5-mtp.gguf --override-kv hy_v3.block_count=int:5 \
  --rpc 127.0.0.1:50901,127.0.0.1:50902,127.0.0.1:50903 \
  --device CPU,RPC0,RPC1,RPC2 -sm tensor -ts 1,1,1,1 -ngl 99 --no-mmap \
  -c 4096 -ub 256 -b 256 --host 127.0.0.1 --port 8199 -np 1 -fit off > /work/srv.log 2>&1 &
for i in $(seq 1 90); do grep -q "model loaded" /work/srv.log && break; sleep 2; done
```

- `--device CPU` is REJECTED unless the env `LLAMA_META_LOCAL_DRAFT` is PRESENT
  (even `=0`) — `common/arg.cpp` gates `allow_cpu` on it.
- Byte-identity protocol: 6 × 24-token greedy completions (`temperature 0,
  cache_prompt false`), sha256 the `content`. **Reference sha `c80261ff`** = the
  exact-sum baseline on this stub with the default stack. Any off-gate for a new
  feature must reproduce it 6/6.
- Nondeterministic-by-design features (deferral readiness races): byte-identity
  applies only to 0-engagement configs; judge engaged legs by stability + counters.
- The stub has 4 layers (layer 0 dense, 3 MoE): `BOUNDARY_STATS` shows
  `bcast1 4.0 star 3.0` per graph at the default stack. Feature engagement is read
  from those counters, not from t/s.

## Debug instruments (each with its caveat)

| Env | Shows | Caveat |
|---|---|---|
| `GGML_META_BOUNDARY_STATS=1` | boundary census + `META_EXPERT_DEFER` line, every **128 graphs** | production-valid (pure counters); a short PPL run never reaches 128 graphs and prints nothing |
| `GGML_META_DEBUG_REDUCE=1` | `REDUCE: partial N -> boundary M` placement lines + `PM-*` pair-merge probes | first tool to reach for on boundary-placement questions |
| `GGML_META_TIMING=1` | compute vs reduce split | SERIALIZES — instrumented legs are never baselines |
| `GGML_RPC_STATS=1` | client per-command counters at exit | the right engagement instrument when INFO logs are invisible |
| `GGML_META_ZL_STATS=1` | zero-leg routing counters (`META_ZL` line) | installs an eval callback → forces sched splits → batch-invariance-class, never a baseline; mutually exclusive with `LLAMA_EXPERT_PROFILE` |
| `GGML_CUDA_SYNC_NODES=1` | names the first faulting CUDA node | combine with `GGML_CUDA_DISABLE_GRAPHS=1` |

## Traps (every one of these has cost an hour+)

1. **`pkill -f` self-kill**: a `docker exec bash -c` whose cmdline mentions the
   pattern kills its own shell (exit 143). Use `pkill -x <name>` or kill by PID.
   `fuser` does not exist in llama-devcuda.
2. **Port squatters**: stale servers from earlier runs hold 8199/50901-3 and serve
   your curls with the OLD binary — results look like "my change did nothing".
   Before trusting a null: `pgrep -a -x llama-server` and check the cmdline.
   `<defunct>` zombies in the container are unreaped-but-dead — harmless, no ports.
3. **Env gates parse VALUES**: `getenv() != nullptr` treats `=0` as ON. All new
   gates must `atoi(getenv(...)) != 0`. Gate scripts that pass `VAR=0` legs will
   silently run instrumented otherwise.
4. **Background-task cap**: long `docker build`/`docker run` clients started as
   harness background tasks get killed at ~10 min. `nohup ... & disown` from a
   normal shell, then watch by artifact existence (image exists, health 200).
5. **Verbosity**: this fork's CLI needs `-st` and `</dev/null`; `GGML_LOG_INFO`
   lines need `-v`/`-lv`, but `-lv 1` makes fleet loads unusably slow — prefer
   stderr prints / counters for must-see engagement evidence.
6. Server readiness = `grep -q "model loaded" <log>`, then still `sleep 1`.
8. **Wizard static changes don't reach dev builds by themselves**: npm is NOT
   in llama-devcuda, so the ui-assets provisioning reuses `tools/ui/dist/`
   as-is. After editing `tools/ui/static/wizard.html`: `cp tools/ui/static/wizard.html
   tools/ui/dist/wizard.html` (host), rebuild `llama-ui llama-server`, and test
   WITHOUT `--no-webui` (it 404s every UI asset - burned 30 min as "stale embed").
   Image builds run vite and need none of this. After flag changes also run
   `python3 scripts/gen-wizard-flags.py` (and `--check` = the #96 drift gate).
7. Reading intermediate tensors at execution time returns **ring-recycled bytes**
   (the member arena ignores FLAG_OUTPUT). Capture values at compute time via the
   eval-callback pattern (`llama_expert_profile_cb` / `llama_zl_ids_cb` in
   `src/llama-context.cpp`) and hand them over via a registry.

## Gate ledger discipline

Every feature lands with: off-gate byte-identity leg, engagement leg (counters),
and a note in the relevant `docs/*-plan.md` + `TASKS.md` entry with measured
numbers. Reference shas and counter signatures go in the plan doc so the next
session can regate without re-deriving.
