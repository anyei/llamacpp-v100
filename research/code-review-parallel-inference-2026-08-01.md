# Code review — `parallel-inference` branch (2026-08-01)

High-effort multi-agent review of `git diff master...HEAD` (426 changed files).
38 candidates found, adversarially verified: 29 kept (collapsing to ~19 distinct defects), 9 refuted.
The 10 most severe are listed below — all correctness-class. Cleanup-class findings
(ASCII-rule violations, getenv duplication, dead variables, oversized `graph_compute`,
unchecked file writes) verified but fell below the report cap.

Status legend: `[ ]` open · `[x]` fixed · `[-]` won't fix / accepted

---

## [x] 1. Server-wide HTTP hang: unbounded `ctx_hold` + untimed `ctx_guard_enter()`

**FIXED 2026-08-01:** timed guard (`CTX_GUARD_HOLD_TIMEOUT_MS=30s`) +
`server_unavailable_exception` → 503 via `ex_wrapper`; reload retry loop kept
infinite (self-heal). Gated via loopback wedge repro (dead worker + hidden
model file): /completion 503 at 30.0s, /health <1ms throughout, self-heal
serves 200 after file restore.

**File:** `tools/server/server-context.cpp:5074` (and `:1763`) — **CONFIRMED**

The `bypass_sleep |= sleep_idle_seconds < 0` fast path was deleted, so every
non-bypass HTTP handler now blocks in `server_queue::ctx_guard_enter()` — an
untimed `condition_tasks.wait(lock, [&]{ return !ctx_hold; })` — even when
idle-sleep is disabled, and nothing bounds how long `ctx_hold` stays set.

**Failure scenario:** An RPC worker drops and `--rpc-reload` runs
`queue_tasks.ctx_hold_begin(15000)` (`server-context.cpp:1763`) before its
`for (int attempt = 1;;)` retry loop; if the workers stay unreachable the loop
retries every 10s forever and `ctx_hold_end()` (line 1860) is never reached.
Every request through `create_response()` (completions, /metrics, /props,
/slots, /v1/models) parks forever inside `ctx_guard_enter` with no timeout and
no `should_stop` poll, exhausting the HTTP thread pool: clients get no response
and no error, and even `create_response(true)` bypass routes become unreachable
once the threads are consumed. Before the change, with the default
`sleep_idle_seconds < 0`, those requests were never blocked on lifecycle state.

---

## [x] 2. Deprecated `-ndio` / `--no-mmap` negations clobber `load_mode` (mmap silently disabled)

**FIXED 2026-08-01:** all three shims (`--mlock`, `--mmap/--no-mmap`,
`-dio/-ndio`) now compose with the current `load_mode` instead of overwriting
it; `-ndio` restores mmap only from DIRECT_IO. Gated: 7 new composition cases
in test-arg-parser, all pass.

**File:** `common/arg.cpp:3135` (same pattern at `:3117` `--mlock`, `:3126` `--no-mmap`) — **CONFIRMED**

The deprecated `-ndio/--no-direct-io` negation writes the whole `load_mode`
field, resetting it to `LLAMA_LOAD_MODE_NONE` and thereby silently disabling
the default mmap.

**Failure scenario:** `llama-server -m big.gguf -ndio` (or a compose file with
`LLAMA_ARG_DIO=0`, which fleet stacks set to pin direct-IO off) previously left
the default `LLAMA_LOAD_MODE_MMAP` untouched. Now the handler sets
`params.load_mode = LLAMA_LOAD_MODE_NONE`, so `llama_model_loader` computes
`use_mmap == false` (`src/llama-model-loader.cpp:559`). The model is read fully
into anonymous RAM: load time goes from seconds to minutes and peak RSS becomes
the full model size, OOM-killing boxes sized on file-backed weights. `--no-mmap`
at `:3126` likewise clobbers a previously supplied `--mlock`.

---

## [x] 3. Data race on `base_params.models_dir` (router UB/segfault)

**FIXED 2026-08-01:** `load_models()` snapshots `models_dir` under `mutex`
before the Phase-1 unlocked scan (all callers enter unlocked, so no deadlock).
Build-gated; runtime TSAN gate skipped as disproportionate.

**File:** `tools/server/server-models.cpp:475` (and `:470`) — **CONFIRMED**

`server_models::load_models()` reads `base_params.models_dir` outside the mutex
("Phase 1 ... no lock needed"), but the new `set_models_dirs()` rewrites that
same `std::string` under the mutex — a concurrent dir change and a reload race
on the string buffer.

**Failure scenario:** Client A POSTs `/wizard/dirs` → `set_models_dirs()`
assigns `base_params.models_dir` while holding `mutex`. Concurrently client B
GETs `/models?reload=1` → `load_models()` runs
`string_split<std::string>(base_params.models_dir, ',')` at line 475 with no
lock. The reader walks the string's heap buffer while the writer frees and
reallocates it: the router segfaults (or splits garbage paths and drops every
model). The launcher dies and all child instances become unreachable.

---

## [x] 4. Recursive models-dir scan silently collapses same-named models

**FIXED 2026-08-01:** collision now warns with both paths and keeps the first
entry (matches the router's cross-dir "earlier wins" policy) instead of the
map silently keeping the last. Gated: vendorA/vendorB `Qwen3-30B` scratch
repro logs the warning, one deterministic entry served.

**File:** `common/preset.cpp:467` — **CONFIRMED**

`load_from_models_dir` replaced the flat one-level scan with a `depth < 4`
recursive walk that names each directory's single model after that directory,
but results are still collapsed through `out[preset.name] = preset` on a
`std::map`, so same-named nested directories silently overwrite each other.

**Failure scenario:** `--models-dir /models` containing
`/models/vendorA/Qwen3-30B/model.gguf` and `/models/vendorB/Qwen3-30B/model.gguf`
both emit name `Qwen3-30B`; the map keeps whichever `fs_list` returned last.
`/v1/models` shows one entry and a request routed to that name loads the other
vendor's weights — the other model is unreachable and the user silently gets
the wrong one.

---

## [x] 5. hy-v3 LP pair-fusion early return skips `build_cvec` + `l_out` callback

**FIXED 2026-08-01:** layers steered by an active control vector are excluded
from pairing (fall back to the exact unpaired path, which applies
`build_cvec` + `l_out`), with a one-time warning. Unsteered paired graphs are
untouched (`tensor_for` null → predicate false). Gated: build clean,
LP_PAIRS=1 serve generates, no spurious warning without cvec.

**File:** `src/models/hy-v3.cpp:222` — **CONFIRMED**

The LP pair-fusion path added an early `return nullptr` from `build_layer` that
skips the function tail, dropping the `build_cvec(lcur, il)` call (and the
`l_out` callback) that every non-paired layer still applies.

**Failure scenario:** hy-v3 with `--control-vector` and `LLAMA_LP_PAIRS=1`:
for every layer in `[lp_edge, n_layer - lp_edge)` where both layers are MoE,
the control vector is never added to the residual — only the few edge layers
are steered, with no warning. `cb(..., "l_out", il)` also disappears for those
layers, so eval-callback/debug instruments see a different node set for paired
vs unpaired runs.

---

## [x] 6. `--rpc-auto-weight` low fit estimate is now fatal `exit(1)` at parse time

**FIXED 2026-08-01 (user decision, revisits TASKS #90):** restored
warn-and-continue — a low fit estimate keeps the default split and logs a
loud warning naming the OOM risk (with the #90 39.7GB/32GB precedent) and the
remedies (free memory, shrink, retry once workers settle, explicit `-ts`).
Rationale: estimates can be transiently low after a worker restart; the user
chose availability over fail-fast. Gated: build clean, test-arg-parser OK.

**File:** `common/arg.cpp:1594` — **CONFIRMED**

`apply_rpc_auto_weight` no longer warns and falls back to the default split
when `share_sum < 0.999`; it now calls `exit(1)`, so a low auto-weight fit
estimate is fatal at argument-parse time instead of advisory.

**Failure scenario:** A fleet worker is restarted or still releasing buffers,
so its reported free memory is transiently low and the 90%-headroom sum lands
at e.g. 0.97. `llama-server --rpc-auto-weight` now exits 1 during
`common_params_parse_ex` before the model is ever probed — the serve never
starts (under the router the child dies immediately with a failed status) —
where previously it logged a warning, kept the default split, and loaded.

> Note: this may be intentional hardening (see #95/auto-weight history) —
> decide deliberately whether fail-fast or warn-and-fallback is wanted.

---

## [x] 7. Wizard hardware-edit assigns `HW` before validating; partial JSON bricks the page

**FIXED 2026-08-01:** edit now merges over `HW_DEFAULT` (same as bootstrap),
renders before persisting, and rolls `HW`/`totalVram` back on any throw;
honest error alert. Gated: node syntax check + 4-shape logic harness
(partial/garbage/bad-gpus/bad-fleet) all behave.

**File:** `tools/ui/static/wizard.html:764` — **CONFIRMED**

The hardware-edit handler assigns the parsed JSON to the global `HW` before
validating it and without merging `HW_DEFAULT`, so a valid-but-partial edit
permanently corrupts the running page.

**Failure scenario:** Submitting `{"gpus":[{"name":"A100","vram":80}],"ram":256}`
(no `fleet` key): `HW = JSON.parse(v)` succeeds, the partial blob is persisted
to `localStorage`, then `chips()` dereferences `HW.fleet.workers` and throws.
The catch shows the misleading `alert("invalid JSON")`, but `HW` is now
globally missing `fleet`, so every subsequent `render()` throws and the wizard
stops responding until reload. Omitting `gpus` breaks `HW.gpus[0].vram` at
lines 515/613/1059 the same way.

---

## [x] 8. `/models/load` `extra_args` unvalidated: `--port` override breaks routing; non-strings → 500

**FIXED 2026-08-01:** extra_args now validated like extra_env: non-string
entries → 400 (was 500), `--port`/`--host` rejected 400 (router owns the
child's binding); extra_env also gets the non-string 400. Gated live: all
three rejections + benign-args acceptance verified over HTTP.

**File:** `tools/server/server-models.cpp:1816` — **CONFIRMED**

`extra_args` from the request body is appended to the child's argv with no
validation (unlike `extra_env`), and non-string array elements throw an
uncaught json `type_error`.

**Failure scenario:** POST `/models/load` with
`{"model":"x","extra_args":["--port","9999"]}`: later flags win, the child
binds 9999 while `inst.meta.port` holds the router-allocated port. Every proxy
request hits the wrong port, health polling reports the model failed though the
child is up — unrecoverable without an unload. Separately, `{"extra_args":[123]}`
makes `a.get<std::string>()` throw outside the `res_err` validation path,
turning a malformed request into a 500 instead of the 400 the adjacent
`extra_env` check produces.

---

## [x] 9. Wizard `extra_env` appended *after* inherited env — child `getenv` keeps the old value

**FIXED 2026-08-01:** confirmed empirically (C execve/getenv repro: duplicate
envp → child reads FIRST entry). extra_env now erases matching `NAME=` entries
from the inherited env before appending. Gated end-to-end: router with
`GATE_TEST_FOO=1`, wizard override `=2` → child `/proc/<pid>/environ` shows
exactly one entry, value 2 (also exercised extra_args+extra_env overlay
loading the trunc stub through /models/load).

**File:** `tools/server/server-models.cpp:983` — **PLAUSIBLE**

`child_env` starts as a copy of the router's own environ; the wizard overlay is
appended after. `execve` keeps duplicates in order and glibc's `getenv` returns
the FIRST match, so an override of a variable already present in the router's
env is ignored by the child.

**Failure scenario:** Router started with `GGML_META_BCAST_FUSE=1` in compose
env; wizard sets `GGML_META_BCAST_FUSE=2`. The child receives both entries and
reads `1`. The UI reports success, the serve silently runs with the old gate
value — an A/B measurement that is actually the same configuration twice. The
code comment ("env entries reach the child through its environment like any
gate") asserts the opposite.

> ⚠️ Direct threat to A/B measurement validity (fleet gate experiments).

---

## [x] 10. Fused-FETCH stash is untagged FIFO — co-hosted meta members can swap partials

**FIXED 2026-08-01:** trace confirmed the mechanism (deferral skips j1's recv,
j2 pops the FIFO front; same-shape sizes pass). Latent only — current fleet is
1 device/endpoint. Stash + rsp_fifo entries now carry the owning backend;
fused_recv/fused_ready claim only their own payloads (single-owner behavior
identical by construction). Gated: canonical 3-worker loopback byte-identity
**6/6 c80261ff** with fused engagement (`bcast1 4.0 star 3.0`, parts fused
8.8/graph); build-cpu + build-cuda75 clean. Live 2-device-endpoint defer A/B
needs a fleet window (no local rig).

**File:** `ggml/src/ggml-rpc/ggml-rpc.cpp:2408` — **PLAUSIBLE**

The new per-socket fused-FETCH stash is a strict FIFO with no member/device
tag, but the meta backend's expert-deferral path recvs members out of send
order, so two meta members sharing one RPC endpoint (multi-GPU worker behind
one cached socket) can swap each other's partials.

**Failure scenario:** A 2-GPU worker added as a single endpoint: both RPC
devices share one cached socket / one `ggml_backend_rpc_async_state`. With
`GGML_META_EXPERT_DEFER=1` (or `GGML_META_PROBE_DEFER_GATHER`), the star gather
(`ggml-backend-meta.cpp:4600-4655`) defers member j1 but recvs j2 immediately;
j2's `fused_recv` pops j1's payload from `st.fetch_stash` (size check passes —
same boundary shape). Partials are summed into the wrong slots: silently wrong
activations, incoherent output, no error logged.

> Current fleet topology note: only relevant if any endpoint exposes >1 device
> behind one socket AND a defer gate is on. Verify against the actual roster.

---

## [x] 11. NEW (found 2026-08-01 during #10 validation): meta serve over a multi-device endpoint fails at warmup

Attempting #10's real-topology validation (one `ggml-rpc-server` exposing both
V100s → `--device CPU,RPC0,RPC1 -sm tensor` meta serve) failed every decode with
`batched placement: entries missed ... manifest went stale` (49 misses) →
decode -3. Pre-existing (identical with defer off and with the 0498a482b
rollback client); never exercised — fleet workers are single-device.

**ROOT CAUSE + FIX 2026-08-01:** the client optimistically inserted every
STREAMED hash into its per-socket manifest ("the stream makes the worker cache
it") — only true when the worker runs `-c`. A multi-device endpoint uploads
identical mirrored bytes twice on one socket, so the second copy batch-placed
against a cache that was never populated. Proven: same topology with `-c`
worked immediately. Fix: the manifest now only ever contains what the worker
itself reported; repeat identical uploads stream instead (cost: mirrored
tensors stream once per device on multi-device endpoints only).
Gates: (1) cacheless 2-device repro loads + serves, 0 misses; (2) canonical
3-worker byte-identity 6/6 c80261ff; (3) **#10 topology validation unblocked
and run**: defer mode 1 (30/30 stable, defers 0 = null for swap path on fast
loopback) and mode 2 all-defer (defers 3.8/graph rate 100%, injects 3.8,
**LOST 0.00**, 29/30 stable + 1 in-family variant) — concurrent multi-member
stash on one socket exercised, owner claims correct. Residual: the mixed
straggler swap-differential (old-vs-new) needs artificial per-member latency;
not built.

## [ ] 12. FILED 2026-08-02 (fleet validation): dspark/dflash draft context cannot couple to a meta-split target

With the target on the EP/meta device, a `--spec-type draft-dspark` draft
context aborts in `resolve_fused_ops`/`graph_reserve`:
`pre-allocated tensor (output.weight) in a buffer (Meta(...)) that cannot run
the operation (NONE)` — the DFlash-family draft graph references TARGET model
tensors (`ctx_other` coupling, anchor/head reads), which live in the meta
buffer; the draft's single-device scheduler cannot address them. Note
`draft-simple` does NOT couple (loopback-gated fine). Fix direction: when
`ctx_other`'s model is meta-resident, substitute the draft-device MEMBER copy
of the referenced tensors (under EP_ONLY `output.weight` is mirrored on every
member, so a zero-copy member-view API in ggml-backend-meta suffices), or
materialize plain-device copies at draft-context build.

**UPDATE 2026-08-02 late:** NOT meta-specific. On the LAYER fleet the same
abort fires when the target's `output.weight` lives on a remote worker (layer
split puts it on the LAST device): `pre-allocated tensor (output.weight) in a
buffer (RPC0[10.5.5.15:50055]) that cannot run the operation`. The coupling
constraint is general: **every target tensor the DFlash-family draft graph
references must live on a device inside the draft's scheduler.** Config
workaround (validated leg in flight): order the device list so a LOCAL GPU is
last (`--device RPC0,RPC1,RPC2,CUDA0,CUDA1` + reordered `-ts`) so the output
head co-resides with the pinned drafter. Real fix unchanged: member-view /
copy substitution at draft-context build. Also validated this leg: the
eviction-defer fix (a8472e5bd) held on all three workers - zero manifest
misses with warm caches (45 GiB served from cache on .11 + local).
Precedes: spec fixes 0bc0775d4 + b2f8d55b3 (draft split inheritance), which
this sits behind.

## Refuted candidates (for the record)

| File | Claim | Why refuted |
|---|---|---|
| `ggml-rpc.cpp:1023` | `fetch_stash` never cleared on ping-failure path | Failure path consequences don't materialize as claimed |
| `server-models.cpp:1786` | Detached proxy disables client-disconnect cancel | Trigger misstated: only load-wait path sets `waited` |
| `llama-graph.cpp:2188` | `weight_before_ffn` branch missing PARTIAL tag | Both branches use the same arch enum; premise wrong |
| `arg.cpp:1080` | `LLAMA_META_LOCAL_DRAFT=0` accepted as enabled | — |
| `ggml-backend-meta.cpp:1053` | Name-based split-state special cases fragile | Style concern, not a defect |
| `server-models.cpp:1919` | Hand-rolled CSV split duplicates `string_split` | Cleanup-class, refuted as filed |
| `server-models.cpp:1849` | Duplicate fleet-status/worker-log handlers | Cleanup-class, refuted as filed |
| `gen-wizard-gates.py:126` | Unicode ellipsis/emdash in generator | Cleanup-class, refuted as filed |
| `llama-expert-placement.cpp:138` | regex vs sscanf parser divergence | Cleanup-class, refuted as filed |

## Review stats

- Diff: `git diff master...HEAD`, 426 files
- 4 finders (3 correctness angles + 1 cleanup), 37 verifier agents
- 38 candidates → 29 verified kept → 10 reported (cap), 9 refuted
