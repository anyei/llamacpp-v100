# Code review — `master..pi-code-review` (2026-08-19)

Max-effort multi-agent review of `git diff master...pi-code-review`
(master `752ba7ef6` → pi-code-review `3d6d89392`; 483 commits, 459 files,
+80220/−6102). Recall mode: 10 finder angles (5 correctness + reuse +
simplification + efficiency + altitude + conventions) × up to 8 candidates,
a removed-behavior (Angle B) pass, and a Phase-3 gap sweep — every surviving
candidate adversarially verified by one verifier (3-state: CONFIRMED /
PLAUSIBLE / REFUTED).

This document is a findings ledger for later work — **not** PR comments, and
**no fixes are applied**. Each finding carries the concrete trigger and the
load-bearing lines the verifier quoted, so the fix discussion can start from
evidence.

**50 findings:** 2 CRITICAL, 10 HIGH, 17 MEDIUM, 21 LOW, plus a 12-item
refuted list recorded so nothing gets re-raised.

> **Provenance note.** The review ran on 2026-08-19 but was interrupted by a
> session limit during final assembly. This document was reconstructed from the
> completed subagent transcripts (13 finders + 10 verifier clusters, all
> recovered) plus a re-run of the one finder (Angle B) and the gap sweep that
> the interruption had killed, and a verifier pass over the three new candidates
> those two produced. Line numbers are working-tree lines at `3d6d89392`;
> re-confirm before editing.

Severity legend: **CRITICAL** (memory-safety / crash on a routine path) ·
**HIGH** (silent correctness corruption, server-wide abort, or secret
disclosure — realistic trigger) · **MEDIUM** (correctness/robustness, narrower
trigger or display) · **LOW** (bounded / opt-in / cosmetic / cleanup).

---

## Walkthrough checklist (fix scope decided 2026-08-23: CRITICAL+HIGH+MEDIUM; LOW deferred)

`[ ]` open · `[x]` fixed (commit in the note) · `[-]` won't-fix (reason in the note)

- [ ] 1 PEARL worker races main-thread ctx_dft
- [ ] 2 PEARL decode-error throw before join / reload UAF
- [x] 3 /fleet/status leaks LLAMA_API_KEY — fixed 2026-08-23 (redact env values + --api-key argv; live loopback gate: NO-LEAK during load window)
- [ ] 4 meta early-error abandons in-flight fused FETCH
- [ ] 5 deferred cache eviction races new manifest
- [ ] 6 WIRE_F16/Q8 presence-gate, =0 is a no-op
- [x] 7 gguf_get_val_str(general.architecture) no type guard — fixed 2026-08-23 (type guard matching the chat_template pattern; gated with crafted UINT32-arch gguf, router scans it clean)
- [ ] 8 /wizard/placements/generate over-read + OOM + unconfined write
- [ ] 9 DSpark conf gate reads stale encoder features
- [ ] 10 mirror-strip renumber vs full-batch extraction (tree)
- [ ] 11 adopted PEARL rounds skip update_tgt
- [ ] 12 ctx_hold_begin teardown with guards held (PLAUSIBLE, repro first)
- [x] 13 process_rows branch-heal misindexes MTP/EAGLE3 — fixed `c2f43e1f5` (2026-08-21, #137 step 0)
- [ ] 14 spec-tree row demand vs batch capacity
- [ ] 15 star-gather ignores fused_recv failure
- [ ] 16 fused MUL_MAT_ID+GLU/ADD_ID pre-zero; add-id.cu sentinel
- [ ] 17 dspark markov head unequal per-slot blocks
- [ ] 18 RPC graph-uid cross-process collisions
- [ ] 19 RPC cache cap unenforced while conns live
- [ ] 20 TCP_KEEPIDLE breaks macOS build
- [ ] 21 wizard kind classifier hides target models
- [ ] 22 conv_models.remember() replace → spurious 400
- [ ] 23 CHECK_IDS aborts under CUDA graphs
- [ ] 24 fleet totals count CPU-holder rows as VRAM
- [ ] 25 header Unload target vs MRU proxy target
- [ ] 26 greeting no-model CTA during fetch/after failure
- [ ] 27 FleetDeviceCard labels weights share as % experts
- [ ] 28 meta PARTIAL set_tensor over-read (PLAUSIBLE)
- [ ] 29 SESSION_MODEL once per socket (PLAUSIBLE, latent)
- 30-50 LOW: deferred by scope decision 2026-08-23

---

## CRITICAL

### 1. PEARL worker thread races main-thread `ctx_dft` mutations in the same pass
**File:** `tools/server/server-context.cpp:4407` (worker launch) — **CONFIRMED**

`LLAMA_SPEC_PEARL=1` launches a `std::thread` running
`common_speculative_draft(spec)` on `ctx_dft` in `pre_decode()` (`:4407`), joined
only at `decode()` `:5132`. Between launch and join the main thread mutates the
same `ctx_dft` with no synchronization: the empty-draft checkpoint branch
`llama_memory_seq_rm(..., slot.id, ckpt.pos_max+1, -1)` (`:4444`), the
pending-prompt loop `slot.mem.seq_rm/seq_add` (`:4652/:4825`), `load_dft`
(`:4753`), and `common_speculative_set_state` (`:4755`). The only guard,
`if (ctx_dft && !slot.spec_ahead_live)` (`:4439`), excludes *ahead* slots only —
but an empty-draft slot is never marked ahead (`:4379`), so it runs `seq_rm`
concurrently with the worker's `llama_decode`.

**Failure:** `LLAMA_SPEC_PEARL=1`, draft-simple, `-np ≥ 2`, one slot generating
(worker live) while another slot starts/continues a prompt → concurrent
`llama_decode` + `llama_memory_seq_rm`/state-set on one `llama_context` → data
race on KV-cache structures → crash or silent KV corruption.

### 2. PEARL decode-error path throws before join; reload frees `spec`/`ctx_dft` under the live worker
**File:** `tools/server/server-context.cpp:5117` (throw) / `:1429` (destroy) — **CONFIRMED**

On a decode error, `decode()` runs `slot.prompt_clear()` (`:5089`, → `mem.seq_rm`
→ `ctx_dft`) for every processing slot — including ahead slots the worker is
actively decoding — then `throw` (`:5117`), *before* the PEARL join at `:5132`.
The catch does `abort_all_slots; break` with no join. On RPC loss the next
`update_slots` runs `rpc_reload_and_resplit()` (`:4058`) → `handle_sleeping_state`
→ `destroy()` (`:1429`), which frees `spec`/`ctx_dft` — the exact objects the
detached worker captured — with **no join on any reload path**.

**Failure:** PEARL on + any decode error in a round where a worker was launched
(the worker deliberately overlaps the failing target decode, so it is live with
high probability); with `--rpc-reload`, the freed-`spec`/`ctx_dft` use-after-free
follows on the reload path.

---

## HIGH

### 3. `/fleet/status` leaks `LLAMA_API_KEY` (and every `LLAMA_*`/`GGML_*` var) to unauthenticated clients
**File:** `tools/server/server-context.cpp:6585` — **CONFIRMED**

The launch-command capture builds `body["launch"].env` from an environ loop whose
only filter skips the router-plumbing prefix and admits everything else:
`if (kv.rfind("LLAMA_SERVER_", 0) == 0) continue;` then
`if (kv.rfind("LLAMA_", 0) == 0 || kv.rfind("GGML_", 0) == 0) env.push_back(kv);`
(`:6585`). `LLAMA_API_KEY` matches the `LLAMA_` prefix, so its full `KEY=value`
is pushed into the response; it is the documented api-key source
(`arg.cpp:4033`, `--api-key ... .set_env("LLAMA_API_KEY")`). The `--api-key
SECRET` CLI form leaks identically through the argv capture (`:6576`). And
`/fleet/status` is **deliberately key-exempt while loading**:
`server-http.cpp:225` `if (!is_ready.load() && req.path == fleet_status_path)
return true;` inside `middleware_validate_api_key`.

**Failure:** operator sets the key (env or CLI, as the fleet composes do) → during
any model load / `--rpc-reload` recovery (the loading page's normal multi-minute
poll window) an unauthenticated `GET /fleet/status` returns the secret in
`launch.env`; after load, any single-key holder still sees it (and all other
`LLAMA_*`/`GGML_*` secrets) in the FleetScreen copy-pasteable launch panel.
**Fix direction:** drop keys containing `KEY`/`TOKEN`/`SECRET`/`PASSWORD` from the
env loop and redact the value after `--api-key`/`--api-key-file` in the argv
capture.

### 4. Meta backend: early graph-compute error abandons an in-flight fused FETCH → permanent silent logit corruption
**File:** `ggml/src/ggml-backend-meta.cpp:5219` (and `:4996/:5140/:5165/:5244`) — **CONFIRMED**

An early error `return status` from `ggml_backend_meta_graph_compute` leaves the
socket-side fused-FETCH entry in `rsp_fifo` (`ggml-rpc.cpp:2384`). The protocol
records `{kind, fmt, zero_ok, logical_size, owner}` and the stash `{owner,
payload}` — **no graph id / sequence tag** — and nothing drains or resets the FIFO
on graph start or on the error return. The next graph's `fused_recv` claims the
first stash entry with matching owner (the stale previous-graph payload); the
only guard is a size check, and decode-graph star boundaries all share `nbytes`
(`n_embd*1*4`), so the stale claim is silent and every later `fused_recv` runs one
boundary late — **permanently**.

**Failure:** a transient mid-graph child error (local CUDA alloc failure, fill/
butterfly path) while a wire member's fetch is pending → `llama_decode` fails
once, server retries on the same sockets → all subsequent decode output silently
corrupted (stale activations summed). A later shape change surfaces it as a
spurious endpoint failure via the size check.
**Fix direction:** tag fetches with a graph sequence id, or drain/invalidate
`rsp_fifo`+`fetch_stash` on any early error return from meta graph_compute.

### 5. Worker deferred cache eviction races a new coordinator's manifest → endpoint hard-fail on load
**File:** `ggml/src/ggml-rpc/ggml-rpc.cpp:3445` — **CONFIRMED**

`rpc_cache_enforce_limit` checks `g_rpc_active_conns` once at entry (`:3445`),
then scans and deletes cache files (`:3455`/`:3482`) with **no lock and no
recheck** across the entire multi-second scan+delete. It fires on the last-conn
close (`:5439`, off `exec_mutex`) while the accept loop stays live and bumps the
counter for a new connection that serves `RPC_CMD_MANIFEST` from a live dir scan.
If the manifest is served mid-eviction it lists entries about to be deleted; the
client batch-places by hash, `set_tensor_hash_batch` reports `n_miss>0`, and the
client hard-fails: “manifest went stale, failing the endpoint” (`:966`).

**Failure:** the normal fleet pattern — coordinator restart = last conn closes
(deferred eviction fires against an over-limit cache; default
`GGML_RPC_CACHE_LIMIT_MIB=65536`) while the restarted coordinator reconnects
seconds later. **Plausible root cause of the known stale-manifest wedge.**
(See also finding #19 — the same function's cap is unenforced while connections
are live, a distinct failure mode.)

### 6. `GGML_RPC_WIRE_F16`/`WIRE_Q8` gate on env *presence*, not value → documented `=0` disable is a no-op
**File:** `ggml/src/ggml-rpc/ggml-rpc.cpp:2253` — **CONFIRMED**

`static const bool wire_f16 = getenv("GGML_RPC_WIRE_F16") != nullptr;` (and
`wire_q8`) — no value parse. The same file value-parses the analogous
`GGML_RPC_TIMING` and documents exactly this trap (`:4633`: compose passes
`NAME=` (empty) through, and presence-gating turned it on fleet-wide). Worse than
a latent bug: the repo's own production composes default these **on** and
document `=0` as the disable knob (`docker-compose.ep-fleet.yml:65`,
`docker-compose.fleet-coordinator.yml:46`). Setting `COORD_WIRE_Q8=0` yields
`GGML_RPC_WIRE_Q8=0` in the container → non-null → lossy q8_0 wire **still
enabled**.

**Failure:** every past “compression off” A/B leg run this way actually measured
on-vs-on; the disable knob never worked. **Fix direction:** parse the value like
`rpc_timing_enabled` does.

### 7. `gguf_get_val_str(general.architecture)` with no type guard → whole router aborts on one malformed GGUF
**File:** `tools/server/server-models.cpp:253` — **CONFIRMED**

`server_model_read_gguf_meta` reads `general.architecture` via
`gguf_get_val_str` with **no** `gguf_get_kv_type == GGUF_TYPE_STRING` check — the
immediately-preceding chat_template read *does* guard (`:245`). `gguf_get_val_str`
asserts type (`gguf.cpp:193`) → `GGML_ABORT` → `abort()` (SIGABRT), which the
scan's `catch (const std::exception&)` (`:339`) **cannot** catch. `gguf_init`
returns non-null for any valid header regardless of the KV's type.

**Failure:** one crafted/corrupt `.gguf` in any scanned models dir (declaring
`general.architecture` as e.g. UINT32 or an array) aborts the entire multi-model
router at startup scan and on every reload/`get_meta` refresh (`:790`).

### 8. `/wizard/placements/generate`: heap over-read + attacker-driven OOM + unconfined write
**File:** `tools/server/server-models.cpp:2243` (`post_wizard_placement_generate`) — **CONFIRMED**

Three distinct defects in one unauthenticated endpoint. **(a) Heap over-read:**
`perm` is sized `n_expert` and sorted with a comparator reading `c[a]`/`c[b]`
(`:2255`), but `c` is built from `prof["counts"][il]` with only what's present; a
counts row with positive sum but shorter than `n_expert` indexes past
`c.size()` (UB). **(b) OOM:** `n_layer` comes from the profile JSON (`:2215`);
`prof` is a non-const `json`, so `prof["counts"][il]` auto-expands with nulls
toward `n_layer` entries — `n_layer ≈ INT_MAX` drives a multi-GB allocation.
**(c) Unconfined write:** `prof_path` is caller-supplied (`:2184`), read as an
arbitrary file (`:2205`), and the artifact is written to `pp.parent_path()/base`
(`:2278`) with **no** roots check — the sibling `remove` handler (`:2306`) *does*
canonicalize + reject paths outside the models dirs, proving confinement is the
expected pattern.

**Failure:** any client reachable on the port POSTs a crafted profile → heap
over-read / ~34 GB allocation / root-owned write outside the models roots.

### 9. DSpark confidence gate reads stale encoder features when the conf head is absent
**File:** `common/speculative.cpp:1483` — **CONFIRMED**

`const float * conf = params.conf_min > 0.0f ? llama_get_embeddings_nextn(ctx_dft)
: nullptr;` — keyed only on the CLI param, **no `has_conf` flag anywhere**.
`dspark_conf_proj` is optional (`dflash.cpp:88`, `TENSOR_NOT_REQUIRED`) and no
`llama.h` API exposes it. `get_embeddings_nextn` returns the persistent buffer
unconditionally (`llama-context.cpp:1342`), which is written only when the graph
produced a nextn tensor (`:2521`) — but `process()`'s encode already filled it
with encoder features (`dflash.cpp:208`). So without a conf tensor the gate
compares stale RMS-normed encoder components against `conf_min ∈ (0,1]` and
`break`s at `i=0` ~half the time → `result.size() < n_min` → draft cleared.
`conf_min` is also **client-settable per request** (`server-schema.cpp:350`,
`{"speculative":{"conf_min":0.5}}`) — nothing at load or per-request rejects
`conf_min>0` on a headless drafter.

**Failure:** `--spec-draft-conf-min 0.5` (or a client sending `conf_min`) with a
DSpark GGUF lacking `conf_proj` (explicitly permitted) → drafting silently
collapses every round. Also reachable **with** a conf-head model when
`build_dspark_markov_head` early-returns on unequal blocks (`dflash.cpp:233`) →
`t_h_nextn` unset → same stale gate (see finding #17).

### 10. Mirror-strip renumbering vs full-batch extraction buffer corrupts dspark/dflash drafter state
**File:** `common/speculative.cpp:1289` (with `tools/server/server-context.cpp:5147`) — **CONFIRMED**

The server strips branch-seq rows and renumbers the survivors before
`common_speculative_process` (`server-context.cpp:5147`, no index translation
recorded). dspark/dflash `process()` reads `llama_get_embeddings_layer_inp(ctx_tgt)`
by the **mirror** batch's own row positions (`:1289`), but the extraction buffer
is laid out by the **full** decoded batch (branch rows interleaved between slots;
`llama-context.cpp:2512`, `split_simple` preserves order). The dflash
`process_rows` override indexes `layer + rows[...]` with full-batch indices,
proving full-batch indexing is the author's own contract.

**Failure:** `LLAMA_SPEC_TREE=1`, `n_parallel ≥ 2`, dspark/dflash (in the allowed
roster), any earlier slot armed → each later slot's `i_batch_beg` is smaller than
its true extraction row by `(1 + n_draft)` per preceding armed slot → the drafter
encodes and injects the wrong slot's hidden states into its persistent decoder KV
→ silent per-slot acceptance collapse on essentially every armed multi-slot round
(output text stays correct via target verification, so it is silent).

### 11. Adopted PEARL rounds run `update_pos` without `update_tgt` → stale checkpoint restore on SWA/recurrent targets
**File:** `tools/server/server-context.cpp:4318` — **CONFIRMED**

On adoption (`:4285`) the slot takes the else branch at `:4315` (guarded
`!adopted`), so `update_pos` refreshes `n_tokens`/`pos_min`/`pos_max` to the
current round (`:4318`) but the slot is kept **out** of the drafting list
(`:4329`) and `update_tgt` runs nowhere else. Nothing re-captures the target
checkpoint after adoption; `data_tgt` still holds the previous round's blob
(without the tokens accepted that round). On a partial rejection of the adopted
draft with a checkpoint-class target (FULL/RS rm-type, reachable — the PEARL gate
constrains only the *drafter's* rm-type), `load_tgt` (`:5486`) writes the stale
round-N blob while `seq_rm`/`keep_first` (`:5492`/`:5494`) use round-N+1
`pos_max`/`n_tokens`.

**Failure:** `LLAMA_SPEC_PEARL=1` + draft-simple + SWA or recurrent target + an
adopted round followed by a partial rejection → silently corrupted slot state /
degraded generations.

### 12. `ctx_hold_begin` proceeds with teardown while request threads still hold guards
**File:** `tools/server/server-queue.cpp:159` — **PLAUSIBLE (HIGH)**

`ctx_hold_begin(15000)` (`:156`) warns and returns normally on timeout with
`n_ctx_guards > 0` still held; its sole caller (`server-context.cpp:1891`)
immediately runs `handle_sleeping_state(true)` → `destroy()` → `llama_init.reset()`
freeing model/ctx/vocab. New entrants are correctly blocked once `ctx_hold=true`,
so the hole is an **existing** holder exceeding 15 s (chat-template render +
tokenize on the HTTP thread) that then keeps reading freed vocab/model — the exact
UAF class the guard was added to fix. It is a deliberate liveness tradeoff (untimed
waits starved the HTTP pool); the only open question is whether a real pre-task
window exceeds 15 s (multi-MB prompt, heavy jinja over a huge conversation, or a
CPU-starved box during a worker-loss incident).
**Confirm by:** inject a 16 s pre-task sleep and kill an RPC worker with
`--rpc-reload`; expect the `QUE_WRN` line then a segfault in tokenize/template.

---

## MEDIUM

### 13. Default `process_rows` branch-heal misindexes MTP/EAGLE3 extraction buffers (via secondary-drafter path)
**File:** `common/speculative.cpp:333` — **CONFIRMED (trigger corrected)**

The default `process_rows` renumbers the branch sub-batch `0..n-1` (`:333`), but
MTP (`:1758`) and EAGLE3 (`:813`) index the target's per-row extraction buffers by
batch-row position, which are laid out by the full verify batch. The **primary**
roster gates MTP/EAGLE3 out of tree mode (`server-context.cpp:2209`), but the
`LLAMA_SPEC_DRAFT2_TYPE=draft-mtp`/`draft-eagle3` secondary-drafter path
registers after `spec_tree_active` is resolved and the gate inspects only
`params_base.speculative.types` — so it bypasses the gate, and
`common_speculative_process_rows` heals **all** impls.

**Failure:** `LLAMA_SPEC_TREE=1 --kv-unified` + primary draft-simple/dspark/dflash
+ `LLAMA_SPEC_DRAFT2=<model> LLAMA_SPEC_DRAFT2_TYPE=draft-mtp` + a branch taken →
the secondary drafter's mirrored KV / pending_h is rebuilt from another slot's
hidden states → silent drafter-state corruption.

### 14. Spec-tree doubles per-round row demand but not batch capacity → assert abort + unguarded logits read
**File:** `tools/server/server-context.cpp:624` (and `:5406`) — **CONFIRMED**

Server batch capacity stays `n_batch` (`:2488`) while armed slots consume
`2 + 2*n_draft` rows instead of `1 + n_draft`. Only the branch rows fail
gracefully (`:612`); the main anchor+draft rows feed `GGML_ASSERT(add_ok ...)`
(`:624`). Separately, tree sampling uses full-batch indices without the view guard
(`:5406` vs the non-spec `:5272` `i_batch - off`), and the post-decode guard
covers only `spec_i_batch`.

**Failure:** `LLAMA_SPEC_TREE=1` with large `n_parallel*n_draft` vs `n_batch`
(e.g. `n_batch=2048, n_parallel=64, n_draft=16`: baseline 1088, fully-armed 2176)
→ hard `GGML_ASSERT` abort on a later slot's main row (a config that served fine
pre-tree); or, on a split round, `llama_get_logits_ith` with an index `≥ n_outputs`
→ nullptr/abort.

### 15. Meta star-gather ignores `fused_recv` failure → one silently corrupted token
**File:** `ggml/src/ggml-backend-meta.cpp:4908` — **CONFIRMED**

The reduce ignores `fused_recv`'s bool (`:4907`); on failure `fused_recv`
zero-fills the slot and returns false, the reduce sums the zeros, broadcasts the
wrong total (`:4963`), and returns `GGML_STATUS_SUCCESS` (`:4964`). A mid-graph
failure is usually caught one dispatch later, but a failure on the **last** reduce
(the logits boundary) completes the graph with half-sum logits.

**Failure:** socket failure on the final logits-boundary reduce → at least one
silently corrupted token before the failure surfaces elsewhere.
**Fix direction:** check `fused_recv`'s return in the reduce path and abort the
graph.

### 16. Fused `MUL_MAT_ID+GLU`/`+ADD_ID` bypasses `dst` pre-zero; `add-id.cu` has no sentinel guard at all
**File:** `ggml/src/ggml-cuda/ggml-cuda.cu:3572` / `:3923`; `ggml/src/ggml-cuda/add-id.cu:14` — **CONFIRMED**

Fused dispatches call the vec kernels directly with `ids` and `dst` = the GLU/bias
node (`:3572`), skipping `ggml_cuda_mul_mat_id`'s `cudaMemsetAsync` pre-zero; the
mmvq/mmvf wrappers do no zeroing, and the sentinel early-return (`mmvq.cu:554`)
assumes a pre-zero that didn't happen. For GLU-only fusions the garbage rows are
provably never read (down-proj skips sentinel lanes), so **no** output corruption
on bias-free models — but the `MUL_MAT_ID+ADD_ID` fusion (`:3923`, per-expert-bias
models like gpt-oss) *does* read the fused bias-node dst → NaN/Inf survives the 0.0
mask. **Bonus (separate finding):** `add-id.cu:14/:21` has no sentinel guard —
`i11 == -1` reads the row before the bias tensor **even unfused**, so bias-model +
placement is broken independently.

### 17. DSpark markov head assumes equal per-slot block sizes → misaligned bias/conf rows
**File:** `src/models/dflash.cpp:237` — **CONFIRMED**

Slots share one decode (`speculative.cpp:1460`); per-slot block size is
`min(n_max, dp.n_max)` and `dp.n_max` genuinely differs (a slot near its context
limit gets a smaller block, `server-context.cpp:545`). `build_dspark_markov_head`
takes `n_blocks = ubatch.n_seqs_unq`, `n_tok = base->ne[1]`, guards only
`n_tok % n_blocks != 0` and `block_drafts > block_size` — blocks of 3+5
(`n_tok=8, n_blocks=2`) pass both; the uniform strided views read block B's markov
anchor from a MASK token and scatter bias/conf onto wrong logit columns for both
blocks. The non-dividing case silently no-ops instead, feeding finding #9.

**Failure:** ≥2 slots drafting with unequal budgets → silently corrupted draft
logits (target still verifies output text; this is silent draft-quality loss).

### 18. RPC graph-uid cache: uid is a per-process counter → cross-process collisions kill connections
**File:** `ggml/src/ggml-rpc/ggml-rpc.cpp:4324` — **CONFIRMED**

The server's shared per-device uid cache evicts in global FIFO order, never
refreshing on hit (`:4324`); the client `/8` cap bounds only each connection's own
inserts. Critically, the uid is a **per-process monotonic counter** (`ggml.c:56`),
not a content hash — two coordinator *processes* on one worker count from 1 and
collide by construction. A server-side miss returns false → connection closed
mid-serve; a re-upload transfers `owner_conn` (`:4288`), so a legitimate same-uid
second connection gets its `RECOMPUTE` rejected (`:4345`). Stale header comment at
`:391` still claims eviction “only costs a re-send.”

**Failure:** ≥2 client processes (or endpoint-string aliases — IP vs hostname) on
one worker device → dropped connections / rejected recomputes. Single-coordinator
topology is provably safe, which is why production hasn't hit it.

### 19. RPC cache cap unenforced while connections are live → unbounded disk growth
**File:** `ggml/src/ggml-rpc/ggml-rpc.cpp:3445` — **CONFIRMED**

Distinct from finding #5 (the eviction-vs-manifest *race*): here the cap invariant
itself is **suspended** whenever `g_rpc_active_conns > 0` (early-return at `:3445`).
`cache_store` calls enforce on every save (`:3592`) but that call is neutered by
the same early-return during a serve, so writes never trim while connected;
enforcement runs only at startup (`:5401`) or as idle catch-up on the **last**
closing connection (`:5439`). With overlapping/persistent coordinator connections
(router-managed swaps that keep the compute socket open, or a new load connecting
before the old drops) the count never reaches 0, so the cap is never enforced.
Second defect: idle enforcement reads a `thread_local t_rpc_session_model` (`:3392`)
to pick the protected folder (`:3468`), but it runs on the last-closer's thread —
possibly empty or a different model — so “active model's folder evicts last”
protects the wrong folder.

**Failure:** back-to-back router loads with overlapping connections → cache dir
grows past `GGML_RPC_CACHE_LIMIT_MIB` without bound → ENOSPC (degrades
`cache_store`, starves co-tenant processes); idle trims may evict the still-in-use
model's slices. **Fix direction:** trim non-active/older slices even while
`conns>0`, and pass the intended active-model name in explicitly rather than
reading the last-closer's thread-local.

### 20. `TCP_KEEPIDLE` breaks the macOS build of the RPC backend
**File:** `ggml/src/ggml-rpc/transport.cpp:641` — **CONFIRMED**

`setsockopt(..., TCP_KEEPIDLE, ...)` under a bare `#ifndef _WIN32` with no
`__APPLE__` branch (`:639`). macOS defines `TCP_KEEPALIVE`, not `TCP_KEEPIDLE`, so
the identifier is undeclared; `transport.cpp` is built unconditionally with the RPC
backend (the CMake file already has explicit Apple handling for RDMA, so Apple +
`GGML_RPC=ON` is contemplated). **Fix direction:** `#ifdef TCP_KEEPIDLE` guard or
an `__APPLE__` branch using `TCP_KEEPALIVE`.

### 21. Wizard `kind` classifier hides legitimately-named target models from the launchable list
**File:** `tools/server/server-models.cpp:2621` — **CONFIRMED**

`model_info["kind"]` is set by lowercased-name substring: any name containing
`draft` or `dflash` → `"draft"` (`mmproj` likewise). The wizard UI **filters**,
not hints: `MODELS = SCAN.filter(x => x.kind==="model")` (`wizard.html:316`) is the
launchable picker source.

**Failure:** a target like `DeepSeek-R1-Draft-Qwen-32B` contains `draft` →
excluded from the launchable list.

### 22. `conv_models.remember()` unconditional replace → spurious 400 for concurrent same-conversation POSTs
**File:** `tools/server/server-models.cpp:1849` — **CONFIRMED**

`remember` overwrites the entry+ticket unconditionally (`server-models.h:142`).
In `proxy_post`, a request registers a ticket, parks in `ensure_model_ready`
(model-load wait), then checks `alive(conv_id, ticket)` (`:1849`). Two concurrent
POSTs with the same `X-Conversation-Id` during a load: req A gets T1, req B
overwrites with T2, both unblock, A's `alive(T1)` sees T2 → spurious HTTP 400
(“request cancelled by a stop”), though no stop occurred. The real cancel path is
`forget()`; `remember`'s replace conflates a duplicate with a cancellation.

### 23. `GGML_CUDA_DEBUG_CHECK_IDS` aborts under default-on CUDA graphs
**File:** `ggml/src/ggml-cuda/ggml-cuda.cu:1947` — **CONFIRMED**

`GGML_CUDA_CHECK_IDS=1` runs `cudaMemcpyAsync` D2H + `cudaStreamSynchronize` at
the top of `ggml_cuda_mul_mat_id` (`:1945`). CUDA graphs stay **enabled** for
quantized MoE decode (`:2614`), and `cudaStreamSynchronize` on a capturing stream
returns a capture error → `CUDA_CHECK` aborts a few tokens in. `env-gates.md`
documents this hazard for `SYNC_NODES` (`:88`) but not for `CHECK_IDS` (`:90`).

**Failure:** the `#75` diagnostic aborts the process exactly on the quantized-MoE
decode it was built to debug, unless `GGML_CUDA_DISABLE_GRAPHS=1` is also set.
Opt-in debug only. **Fix direction:** skip while `cudaStreamIsCapturing`, or force
graph-disable; add the caveat to `env-gates.md`.

### 24. Fleet totals count CPU-holder rows as VRAM
**File:** `tools/ui/.../fleet/FleetScreen.svelte:98` — **CONFIRMED**

`fleetTotals` classifies rows only by `worker_is_cpu` (`:97`). The server emits a
`#131a` CPU-offload holder row with `is_rpc=false` → `worker_is_cpu=false`
(`server-context.cpp:6470`) filled from host RAM (`:6460`).

**Failure:** any ncmoe/eplocal serve sums the coordinator's host RAM (~251 GiB on
X99) into the “VRAM” tile. Display-only.

### 25. Fleet header Unload targets `loadedModelIds[0]`, page shows the MRU proxy target
**File:** `tools/ui/.../fleet/FleetScreen.svelte:172` — **CONFIRMED**

`unloadTarget = loadedModelIds[0]` (listing order, no MRU sort), while the page's
fleet data proxies to `fleet_proxy_target` (loading child, else most-recently-used;
`server-models.cpp:1912`). With ≥2 loaded models these diverge; the confirm dialog
names the victim (`:691`), which is the mitigation. Display/action mismatch.

### 26. Greeting shows “No model is loaded” + wizard CTA during fetch and after a transient failure
**File:** `tools/ui/.../chat/ChatScreen/ChatScreenGreeting.svelte:15` — **CONFIRMED**

`noModel` has no “fetched yet” / loading / error guard (`:15`); there is no
`hasFetched` flag in the store. On every cold load `routerModels` is `[]` between
the `serverStore.fetch()` (which flips role to ROUTER) and the router-models await
→ the full-screen CTA flashes while a model is serving. Worse, `fetchRouterModels`'
catch sets `routerModels = []`, so one transient `/models` failure makes the CTA
**sticky** until the next successful refetch.

### 27. `FleetDeviceCard` labels total-weights share as “% experts”
**File:** `tools/ui/.../fleet/FleetDeviceCard.svelte:271` — **CONFIRMED**

The card prefers `model_frac` (device's share of **all** model bytes, incl.
attention/router/output) over `split_frac` (the EP expert distribution) and renders
`attention owner + ${splitPercent}% experts` (`:270`). A **pure** attention owner
(`split_frac 0`, zero experts) still has `model_frac > 0` and is tagged with a
nonzero “% experts.”

### 28. Meta PARTIAL `set_tensor` over-reads the caller buffer on sub-range writes
**File:** `ggml/src/ggml-backend-meta.cpp:2427` — **PLAUSIBLE**

The `GGML_BACKEND_SPLIT_AXIS_PARTIAL` branch reads `ne` floats from the caller's
pointer regardless of `(offset, size)` (`:2422`); unlike the AXIS_0/1 branches it
has no offset/size alignment assert. A sub-range write over-reads by
`ne*4 - size` bytes. No in-tree writer does a sub-range `set_tensor` on a PARTIAL
tensor today, so it is a latent memory-safety bug.
**Fix direction:** read `size/sizeof(float)` floats, or assert `offset==0 &&
size==ggml_nbytes(tensor)`.

### 29. `SESSION_MODEL` sent once per socket → drafter cache mis-scoped under the main model
**File:** `ggml/src/ggml-rpc/ggml-rpc.cpp:1423` — **PLAUSIBLE**

`SESSION_MODEL` is sent only under a once-per-socket latch (`:1423`), reset only on
fresh-socket creation. A second model announced over a persistent socket is
dropped. The claimed “live process switching models” trigger doesn't exist
(model switching spawns subprocesses; drafter loads never set a second session id),
so no stale-manifest failure today — but a drafter sharing a worker with the main
model gets its tensors cached under the **main** model's folder, weakening the
`#103` cross-model isolation. Latent fragility if any future flow loads a second
model over a live socket.

---

## LOW

- **30. `comm_ctx_sub` never freed → NCCL communicator leak per teardown.**
  `ggml/src/ggml-backend-meta.cpp:3001` — CONFIRMED. The destructor frees only
  `comm_ctx` (`:3000`); `comm_ctx_sub` (created under `GGML_META_LOCAL_COMM`,
  `:2982`) has no free site anywhere → NCCL comm + GPU resources leak on every
  meta teardown/reload. Opt-in, one leak per reload.

- **31. `LLAMA_META_LOCAL_DRAFT` presence-vs-value parse divergence.**
  `common/arg.cpp:1090` — CONFIRMED. `parse_device_list` gates on presence
  (`!= nullptr`) while the three consumers (`llama-model.cpp:602`/`:2436`,
  `llama-context.cpp:659`) parse `atoi != 0`. `LLAMA_META_LOCAL_DRAFT=0
  --device CPU,CUDA0` admits a CPU device the guard meant to reject, with the
  feature off → slow-but-functional. One-line fix.

- **32. Shard-size summation triplicated with divergent missing-shard behavior.**
  `tools/server/server-context.cpp:1111` (+ `:91`, `server-models.cpp:222`) —
  CONFIRMED. Two hand-rolled sums skip a missing shard (partial total); the third
  (`fleet_model_weight_bytes`, canonical `llama_split_*`) returns **0** on any
  missing shard. Divergent semantics for the same task.

- **33. Load-mode string→enum map triplicated; exported canonical helper is dead.**
  `common/arg.cpp:3267` (+ `llama-bench.cpp:766`) — CONFIRMED. Three copies over
  the same 5 modes; `llama_load_mode_from_str` (`llama.h:214`, `LLAMA_API`) has
  **zero callers** repo-wide.

- **34. `add_drafter` switch duplicates `common_speculative_init`'s impl construction.**
  `common/speculative.cpp:2919` — CONFIRMED. A new draft type wired into `init`
  but not `add_drafter` hits `default: return false` → the secondary-drafter path
  fails at runtime while the primary works. Drift-prone.

- **35. `GGML_VBUF_DEBUG_FAIL` function-local statics race under multi-threaded reserve.**
  `ggml/src/ggml-alloc.c:439` — CONFIRMED. Plain C statics (`fail_from`, `seen`),
  no atomics/mutex, reached concurrently by `test-thread-safety` and main+draft
  reserves → nondeterministic injection ordinal (defeats `#112`-style exact-ordinal
  repro) + formal data race. Debug-instrument only.

- **36. Unconditional `mul_mat_id` `dst` pre-zero — guarantee isn't even uniform.**
  `ggml/src/ggml-cuda/ggml-cuda.cu:1933` (+ `ggml-cpu.c:1628`, `repack.cpp:4457`) —
  CONFIRMED (deliberate). Measured 0.24% vs 2.8% noise on GPU decode
  (`env-gates.md:89`); CPU serial full-`dst` memset before the threadpool barrier
  is unmeasured, and the fused paths skip the memset (finding #16) so the “always
  zeroed” guarantee it pays for doesn't hold uniformly. Efficiency note; a
  build-time placement flag would gate it.

- **37. `spec_mirror_trim` runs unconditionally every verify round for the primary drafter.**
  `common/speculative.cpp:241` — CONFIRMED. First statement of every `process()`;
  the server already rolls back the primary drafter's memory (acknowledged in the
  function's own comment). Bounded cost (≤ n_seq map entries, cheap per-cell scan);
  a per-seq `pos_max` no-op check removes the per-round scans.

- **38. Wizard `/wizard/hw` holds a mutex across sequential 800 ms endpoint probes.**
  `tools/server/server-models.cpp:2019` — CONFIRMED. `beacon_mtx` held through the
  per-endpoint probe loop; a never-probed endpoint re-pays 800 ms under the lock
  each request, serializing concurrent `/wizard/hw` (UI fires 5 rescans at 2.5 s).
  Admin endpoint, 60 s probe cache caps it.

- **39. `ChatScreenModelLoading` dismissed set never cleared → later failures suppressed.**
  `tools/ui/.../chat/ChatScreen/ChatScreenModelLoading.svelte:22` — CONFIRMED. The
  `SvelteSet` is only added to (`:59`), never `.delete()`/`.clear()`; the component
  is mounted persistently, so a dismissed model that fails again (same `m.id`) has
  its banner + new `error_tail` silently suppressed for the route lifetime. Still
  visible in wizard/loading detail. Fix: delete the id when the model re-enters
  LOADING.

- **40. `formatFileSize` renders “X undefined” at ≥ 1 TB.**
  `tools/ui/src/lib/utils/formatters.ts:23` — PLAUSIBLE. `sizes` ends at `GB`; with
  `k=1000`, `bytes ≥ 1e12` → `sizes[4]` undefined. Needs a ≥1 TB model (none in the
  current zoo); overflow mostly pre-existed the `k=1000` switch. Add `TB`/clamp.

- **41. Fresh-loaded model shows no speculation badge until a full refetch.**
  `tools/ui/.../models/ModelsSelectorOption.svelte:68` — CONFIRMED. The SSE
  `status_change` carries `info` (`server-models.cpp:1275`) but the client merges
  only `status` (`models.svelte.ts:845`); the fallback `model?.info?.meta?.speculative`
  is dead (rows nest speculative at `row.meta`, not `info.meta`). Transient badge
  gap only.

- **42. Dangling backslash in fleet `launchText` for non-Linux coordinators.**
  `tools/ui/.../fleet/FleetScreen.svelte:71` — CONFIRMED. `cmd` is filled only
  `#ifdef __linux__`; a non-Linux coordinator with any env var emits
  `{cmd:[], env:[...]}`, every env line gets a trailing `\`, and the argv line is
  cmd-gated → pasteable text ends in a dangling backslash. Fleet deploys Linux-only.

- **43. `extract-mtp-head.py` scans only shards 4–5 with no found-all check.**
  `scripts/extract-mtp-head.py:117` — CONFIRMED. Unpinned `resolve/main` URLs; an
  upstream re-shard that moves a wanted tensor out of shards 4–5 yields a drafter
  GGUF silently missing tensors (fails only later at model load).

- **44. `extract-mtp-head.py` short-read guard is an assert (stripped under `-O`).**
  `scripts/extract-mtp-head.py:179` — PLAUSIBLE. The retry path is correct; exposure
  needs `PYTHONOPTIMIZE` **plus** an `IncompleteRead`-evading truncated body.

- **45. `measure-fleet-leg.sh` suppresses the draft readout at 0% acceptance.**
  `scripts/measure-fleet-leg.sh:28` — CONFIRMED. `if acc and dn` is falsy when
  `draft_n_accepted == 0` → a 0%-acceptance leg looks like speculation-off. Fix:
  `if acc is not None and dn`.

- **46. `--draft-p-min` is silently ignored for dspark drafters.**
  `common/speculative.cpp:1485` — CONFIRMED (removed-behavior). The dspark branch of
  `draft()` truncates only on the confidence head (`:1488`) and pushes every sampled
  token; the parallel dflash branch keeps `if (cur_p->data[0].p < params.p_min)
  break;` (`:1520`), as do all other impls (`:519/:997/:1913`). `--draft-p-min` maps
  to `params.p_min` (`arg.cpp:4698`) and the ctor still logs it (`:1177`), but it
  never applies for dspark → full-length drafts including low-confidence tails,
  rejected at verification (wasted verify compute; **output stays correct**).
  DSpark's design-intended knob is `conf_min`, so this is a logged-but-inert
  inconsistency, not a correctness bug. Fix: add the `p_min` break to the dspark
  loop, or warn at ctor that `p_min` is inert for dspark.

- **47. `cache_enforce_idle` can run after `rpc_server` is torn down (use-after-scope).**
  `ggml/src/ggml-rpc/ggml-rpc.cpp:5441` — CONFIRMED-adjacent (Angle B, low). On the
  accept-fails-> supervisor-restart path (`:5423`) `ggml_backend_rpc_start_server`
  returns and destroys the stack-local `rpc_server`, while a last-client thread
  (whose connection died in the same fd storm) may still be inside
  `cache_enforce_idle` scanning a multi-GB cache dir → use-after-scope on the
  server's `cache_dir`. Narrow (fd exhaustion + large cache); latent.

- **48. Spec-tree mirror always sets `batch_mirror.logits` to a heap array.**
  `tools/server/server-context.cpp:5150` — CONFIRMED-adjacent (Angle B, low/latent).
  The mirror replaces `batch_view` with a stripped batch whose `logits` is always a
  heap array (zeros when `batch_view.logits` was null), dropping the previous
  null-`logits` default semantics. Benign today (the server always populates
  `batch.logits`); latent if an all-rows-output null-logits path is ever introduced.

- **49. Side finding: eagle3 `draft()` never calls `spec_alt_begin`.**
  `common/speculative.cpp:946` — surfaced during K7 verification. Under
  `LLAMA_SPEC_ALT_STATS=1` (no roster gate) the per-seq alt capture grows unbounded
  and the stats mis-index into stale rounds. Not a correctness bug on the serve
  path; alt-stats instrument only.

- **50. A typo in `LLAMA_SPEC_DRAFT2_TYPE` throws instead of hitting its own
  "not recognized" handler.** `tools/server/server-context.cpp:2382` — CONFIRMED
  (found 2026-08-20 while documenting the second-drafter knobs). The call
  `common_speculative_types_from_names({... : "draft-simple"})` sits **outside** the
  `try` block, which only opens at `:2386`. That function throws on an unknown name
  (`common/speculative.cpp:2534`, `throw std::invalid_argument("unknown speculative
  type: " + name)`), so a misspelled type escapes as an unhandled exception and kills
  startup — while the `SRV_WRN("LLAMA_SPEC_DRAFT2_TYPE '%s' not recognized - second
  drafter disabled")` handler at `:2384`, clearly written to degrade gracefully, is
  unreachable for exactly the case it targets. Its `d2_types.empty()` leg is also
  dead (the vector is always built from one name); only the literal `none` spelling
  reaches the warning via the `COMMON_SPECULATIVE_TYPE_NONE` leg. Fix: move the
  `types_from_names` call inside the `try`, or use the non-throwing
  `common_speculative_type_from_name` (`:2540`, returns `..._COUNT` on miss).

---

## Refuted (recorded so they are not re-raised)

Each was flagged by a finder and knocked down by its verifier with a quoted guard.

- **K3** `msa_strict_slots` unified+`np>1` — MiniMax-M3 (the only k_idx-cache arch)
  falls back to dense with a logged warning for exactly that layout
  (`minimax-m3.cpp:184`); the `msa_strict_slots` condition is consistent with the
  MSA gate.
- **K4** `n_ctx/n_seq_max` halving via `n_seq_extra` — unified KV bypasses the
  division (`llama-context.cpp:613`) and `n_seq_extra` is set only in the
  unified-required branch (`server-context.cpp:2215`).
- **K6** branch-seq `seq_rm` assert — server strips branch rows before any drafter
  `process()`; unified draft cache makes `seq_to_stream.size()==LLAMA_MAX_SEQ` so
  branch ids pass the assert as a no-op.
- **K7** stale alt capture consumed as a wrong-token arm — blocked by the length
  check (`speculative.cpp:82`), roster gate, and `add_drafter` type rejection.
- **K10** `stoul` cache-name throw (`llama-model.cpp:574`) — every existing
  `cache_*` name parses; DSA caches take the `dsv4_` prefix branch. Angle B
  independently re-flagged this in the EP-only/attn-owner framing and confirmed
  current in-tree names all parse. **Stays refuted** (latent only for a
  hypothetical future cache name).
- **K11** `stoi` `dflash.block_size` throw — runs first in the ctor's mandatory
  `graph_reserve` inside `llama_init_from_model`'s try/catch → clean init failure,
  never mid-decode.
- **S1c** PEARL `pos_max` vs `update_tgt` race — `update_tgt` writes only `data_tgt`;
  `pos_max` is written solely by `update_pos` before the thread launch. Distinct
  members, no formal race (fragility note only).
- **P3** `gguf-truncate.py` alignment — `gguf.cpp:770`'s exact-cumulative-offset
  enforcement makes the raw-size accumulation reproduce the required invariant for
  any loadable source.
- **P4** `xdraft-join.py` empty-accepted — the writer asserts `accepted.size() >= 1`
  (`server-context.cpp:5345`); only a torn final line could break it.
- **E6** `spec_mirror_trim` multi-seq mis-trim — the server's sole row builder is
  single-seq and every drafter entry either rebuilds single-seq or asserts; latent
  only (draft_simple is the one path missing the assert).
- **G8** `ggml_set_output` view-chain walk — upstream commit `da5b44862` (tag
  b10166), not fork code; appears in the diff only because the review base predates
  that merge. **Drop from scope.**
- **G10** meta `set_tensor_2d_async` offset mistranslation — guarded by
  `GGML_ASSERT(offset == 0)` and no in-tree caller can pass a nonzero offset.

Angle B additionally traced ~15 deletions to re-established replacements (protection
intact, do not re-hunt): the kv-cache `n_tokens > cells.size()` guard (moved earlier
in `find_slot`), the DFlash block-size clamp (generalized for DSpark), the dsv4
`n_embd_head`/`n_groups` asserts (`deepseek4.cpp:992`), subprocess handle guards
(into `common_subproc`), the `pipe_t` mutex/queue (renamed `server_pipe`), the
`ctx_dft` seq_rm/add pairs (folded into `common_memory`), the LRU build-cap +
shadow-ring staleness redesign, the eagle3 layer-extraction guards (widened with
`GGML_ABORT`), the `GET_DEVICE_MEMORY` handler relocation, and the
mlock→load_mode / direct-io / manifest-insert / drain-pings→FIFO / reset_seq
reworks.

---

## Coverage notes

- All 13 finder angles + the Angle B removed-behavior pass + the Phase-3 gap sweep
  completed; every surviving candidate got one adversarial verifier vote. The three
  candidates surfaced by the recovery re-runs (findings #3, #19, #46) were verified
  to the same standard; the two low-confidence latent items (#47, #48) are Angle B
  finder-level, marked as such.
- Line numbers are working-tree lines at `3d6d89392`; the diff base is `752ba7ef6`.
  A few candidates cite lines in files pulled in by an in-window upstream merge
  (e.g. G8) — those are called out and excluded.
- No fixes applied and nothing committed to code; this is a ledger only. The
  fix-triage workflow (confirm-before-fix, one-finding-one-commit, regate ladder)
  lives in the `code-review-binary-focus` skill.
