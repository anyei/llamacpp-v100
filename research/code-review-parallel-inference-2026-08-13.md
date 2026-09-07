# Code review - `parallel-inference` window 2026-08-02..2026-08-13 (2026-08-13)

Single-reviewer pass (subagent quota unavailable) over
`git diff 5770c21c3..09edfff24` - the 131 commits of fork work since the
2026-08-01 review (research/code-review-parallel-inference-2026-08-01.md, whose
findings are all fixed; the fixes are in-window and were re-reviewed).
Lens: the fork's purpose - true parallel inference, faster decode than the
current ~6 t/s fleet record - so perf regressions on hot paths rank alongside
correctness. Angles: correctness, edge cases, wire/API contracts, regressions +
invariants. UI (tools/ui) and docs reviewed only for API-contract surface.

Status legend: `[ ]` open · `[x]` fixed · `[-]` won't fix / accepted

Findings: 2 MEDIUM, 4 LOW. All other areas verified clean (list at the end).

---

## [ ] 1. Unguarded static `probe_cache` in the `/wizard/hw` handler (race, UB)

**File:** `tools/server/server-models.cpp:2025` (`get_wizard_hw`) - **CONFIRMED**

`static std::map<std::string, std::pair<json, int64_t>> probe_cache;` lives
inside the lambda body and is read/mutated (`find`, `insert` via `[]`, timestamp
rewrite) with no lock. httplib dispatches handlers on its thread pool, so two
concurrent `/wizard/hw` requests (browser poll + a second client, or the wizard
and the fleet screen open together) iterate and insert on the same std::map
concurrently - UB, realistically a crashed router mid-session.

This is the same defect class the 2026-08-01 review's finding #3 fixed in this
same file (`base_params.models_dir` race).

**Failure scenario:** two HTTP workers enter the handler in the same 60 s
window; one inserts `probe_cache[ep] = {...}` while the other is inside
`probe_cache.find(ep)` -> iterator invalidation / tree corruption -> segfault.

**Fix direction:** guard the cache with a small mutex (or hoist it into
`server_models` state already under its mutex).

---

## [ ] 2. `/wizard/placements/generate` reads/writes arbitrary paths - no roots pin

**File:** `tools/server/server-models.cpp:2137` (`post_wizard_placement_generate`) - **CONFIRMED**

`profile` comes straight from the request JSON. The handler reads that file
(echoing its `model`/`tokens_profiled` fields into the response) and writes a
derived artifact to `<profile_dir>/<stem>-place-<shares>.json` - anywhere the
server process (root in the fleet containers) can write. The sibling endpoint
`post_wizard_placement_remove` *does* pin the path under the scanned roots
(`server-models.cpp` weakly_canonical + `rfind(rc + "/", 0)` check); generate
does not. Inconsistent, and on anything but a trusted LAN it is an arbitrary
JSON-read + limited-content file-write primitive.

**Failure scenario:** any client that can reach the server port POSTs
`{"profile": "/etc/cron.d/x.json", "ts": [1]}` (after planting a JSON) and the
server writes `/etc/cron.d/x-place-1.json` as root. Realistic exposure is low on
the private fleet, but the endpoint is unauthenticated.

**Fix direction:** reuse the remove endpoint's canonical-path roots check on
`profile` before reading; reject paths outside the models dirs + cache.

---

## [ ] 3. `g_rpc_session_model` is process-global; concurrent loads mislabel caches

**File:** `ggml/src/ggml-rpc/ggml-rpc.cpp:1343`, set from `common/common.cpp:1252` - **CONFIRMED, severity depends on load serialization**

`common_init_result` overwrites the process-wide `g_rpc_session_model` on every
model load. Every RPC socket opened afterwards announces that id. If the router
ever loads two models concurrently (or a new load starts while another model's
tensors are still streaming over already-open sockets is NOT the issue - the
risk is two loads interleaving socket creation), model A's tensors land in
model B's per-model folder on the worker (#103).

Impact is bounded: every cache serve is hash-verified, so this misfiles bytes
rather than corrupting them; the cross-model isolation #103 was built for
(poison containment) weakens silently. If router model loads are fully
serialized on the model thread, downgrade to a note.

**Fix direction:** pass the id per-load (e.g. a loader-scoped guard or a
parameter through the RPC device init) instead of one global, or confirm/load-
serialize and document.

---

## [ ] 4. Placements `mtime` uses the impl-defined file-clock epoch

**File:** `tools/server/server-models.cpp:2109` - **LOW CONFIDENCE**

`last_write_time(p).time_since_epoch().count()` has an unspecified epoch before
C++20 `clock_cast`; the value round-trips fine for the newest-first sort (same
process, same clock) but the wizard UI may render nonsense dates if it treats
the number as Unix time.

**Fix direction:** convert via `std::chrono::clock_cast` (C++20) or
`to_time_t`-style portable path before serializing.

---

## [ ] 5. `/fleet/status` exposes LLAMA_*/GGML_* env var VALUES

**File:** `tools/server/server-context.cpp:5909` (launch capture) - **CONFIRMED**

The launch block serializes every `LLAMA_*`/`GGML_*` environment variable with
its value into an unauthenticated endpoint response. The fleet's gates are
plain config today, but any secret passed via env (API keys, tokens in a
`LLAMA_`-prefixed var) leaks to anyone who can reach the port.

**Fix direction:** redact values matching `*KEY*`/`*TOKEN*`/`*SECRET*` at
capture time, or ship names-only.

---

## [ ] 6. `is_recr_layer` can throw on malformed tensor names

**File:** `src/llama-model.cpp:566` - **CONFIRMED, narrow**

`std::stoul(tensor_name.substr(pos))` throws `std::invalid_argument` if a
`blk.*` or `cache_l*` tensor name has no digit run at `pos`. Model names are
GGUF-controlled, so this needs a malformed/adversarial GGUF to trigger, but the
failure is an uncaught exception at load instead of a clean error. #109 itself
is correctly gated (verified).

**Fix direction:** wrap the stoul in a digit check or try/catch -> treat as
non-recurrent on parse failure.

---

## Open questions (perf / purpose-driven)

1. **`ffn_down` island-exit (6f89ad52e) has no layer-fleet t/s gate.** The
   byte gates (6/6 c80261ff) and the -ts 0,1 / -ts 1,19 loopback repros cover
   EP rosters only. In steady-state layer mode the same-owner degenerate
   shortcut should rarely fire for ffn_down (the activation arrives mirrored),
   but a one-line A/B on the hy3 layer fleet (record 4.61-4.80 t/s dual-role)
   would close this - it is a per-block-exit derivation change on the hottest
   path. Same question, smaller, for 1-2 member meta rosters (eplocal), which
   now derive PARTIAL at every block exit.
2. **Unpinned draft under a layer-split target still inherits the fleet
   spread** (`common/speculative.cpp`: the decoupling only fires for pinned
   drafts or tensor targets). Layer+dspark measured 2.81 t/s (#80) - is that
   the accepted shape, or should unpinned+layer also demote the draft to the
   main GPU?
3. **#105 local-subgroup pre-reduce sits opt-in behind GGML_META_LOCAL_COMM**
   with a banked +2.4% (p~0.12) on the OLD 2-local-GPU roster. The new rig has
   3 local GPUs - the subgroup grows from 2 to 3 members and the host-star
   saving triples (3 D2H -> 1). Re-run the A/B on the new roster before the
   next fleet tuning round.
4. **Known residuals already tracked in TASKS (no action, confirming sight):**
   TRUE-zero owner share still aborts the dedicated-attention chain (#114
   residual - blocks the drafter-on-owner config); #114 cleanup-abort
   residual; #119 ssd-streaming V4 Q8 incoherence (open bug); #117 cold-stream
   ~46 MB/s cap (4% of 10G) - the tiled-upload work (#113) cut bytes but not
   per-tile serialization, which is the stated next lever for load time.
5. **Worker cache-cap enforcement now only runs at full idle** (eviction
   deferred while any coordinator is connected, catch-up on last disconnect).
   A long-lived coordinator means the worker disk can overshoot for days on
   multi-model fleets. Acceptable, or should enforcement also run between
   loads (manifest handshake boundary)?

## Checked and found CLEAN

- **RPC wire compat (proto 4.16):** `RPC_CMD_SESSION_MODEL` appended before
  `RPC_CMD_COUNT` (no id shift), coordinator sends it only when
  `server_minor >= 16`, worker behavior unchanged without it; both mixed-version
  directions degrade to legacy flat-cache behavior by design.
- **Eviction deferral:** counter is atomic, `rpc_cache_enforce_limit` re-checks
  it at entry, catch-up races a new connection safely.
- **EINTR retry on send/recv + TCP keepalive (60/10/3):** correct, best-effort,
  non-fatal; no busy-spin risk beyond signal storms.
- **GET_DEVICE_MEMORY off the execution lock:** the handler only touches
  backend memory queries (cudaMemGetInfo / proc reads) - thread-safe against
  concurrent compute; `mark_executed` kept in lockstep.
- **ggml-alloc reserve-failure invalidation** (`n_nodes = n_leafs = 0`) and the
  matching `res->reset()` on alloc/localize failure in `llama-context.cpp` -
  converts a next-decode segfault into a clean re-reserve/failure.
- **Meta build cache rework (#84 seam fix):** `park_active()` nulls
  `build_active` before the registration path, so the new per-class LRU loop
  cannot erase it (no dangling); `reset_seq` counts only fresh ring clears;
  the ring-advance/wrap accounting holds per buffer.
- **#105 subgroup pre-reduce:** folded members' scratch slots are zeroed and
  skipped (no double count); the representative stays in the plain path;
  requires `comm_ctx == nullptr` so the shared `comm_allreduce` pointer never
  mixes contexts; all-present gate falls back cleanly.
- **#118 thread forwarding:** the RPC backend does not export
  `ggml_backend_set_n_threads`, so `ggml_backend_meta_set_n_threads` reaches
  only the local CPU member - remote workers cannot be clobbered by the
  coordinator's `-t`.
- **#95 auto-weight floor:** `share[]` is function-local; the `n_nonzero < 2`
  bail returns before `params.tensor_split` is written - no partial state;
  dropped bytes redistribute only to uncapped members.
- **#114 one-granule floor:** rotation only permutes which member receives
  slice j, so the axis-space guard (`ne_s - g_s*(n_devices-1-j)`) is correct in
  cut terms; true-zero shares keep zero width (the drafter case) by the
  `frac_j > 0` test.
- **#113 tiled uploads:** div-by-zero on `tile_unit` guarded; the tile grid is
  anchored in root coordinates (split-invariant) as designed; asserts hold for
  unit-aligned AXIS_1/AXIS_2 splits.
- **#12 ctx_other mirrors (DFlash/EAGLE3):** one-time snapshot at draft-context
  init is safe (weights immutable at runtime); mirror precedence in the graph
  builders matches the creation condition; buffer lifetime tied to the context.
- **#48 guard relaxation:** owner-group comma math preserves the old verdicts;
  local CPU members bypass only the GPU branch, as intended (#118).
- **Chunked graphs off the fused pipeline (`cgraph->uid != 0`):** correctness
  fix per research/84-seam-dissection-2026-08-04.md; the perf cost is the
  documented trade.
- **Docs/env-gates and UI types:** additive fields only (`memory_breakdown`,
  `launch`, `speculative`, `devices[]`) - backward compatible.

## Verdict

**SHIP-AFTER-FIXES** if any of these hosts is reachable beyond the trusted
fleet LAN: fix #1 (probe_cache race) and #2 (generate path pin). On the private
fleet as-is: SHIP, with #1/#2 scheduled - both are small, local fixes. The two
perf open questions (1: layer-fleet gate for the ffn_down island-exit; 3: #105
subgroup A/B on the 3-GPU roster) are the only items touching the decode-speed
mission and should be answered with measurements, not code.
