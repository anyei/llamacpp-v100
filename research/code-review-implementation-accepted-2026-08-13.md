# Code review - implementation acceptance record (2026-08-13)

Decision log for the findings in research/code-review-parallel-inference-2026-08-13.md
(review window 5770c21c3..09edfff24 on `parallel-inference`). Each item is
discussed one at a time; this document records the verdict, the agreed fix
shape, and - after landing - the gate evidence.

Status legend: `pending` - not yet discussed · `accepted` - will fix, shape agreed ·
`landed` - fix committed + gated · `deferred` - real but scheduled later ·
`rejected` - won't fix (rationale recorded)

| # | Finding (review doc ref) | Severity | Status |
|---|---|---|---|
| 1 | probe_cache race in /wizard/hw (server-models.cpp:2025) | MEDIUM | accepted |
| 2 | /wizard/placements/generate path not root-pinned (server-models.cpp:2137) | MEDIUM | accepted |
| 3 | g_rpc_session_model process-global (ggml-rpc.cpp:1343, common.cpp:1252) | LOW | rejected - not triggerable |
| 4 | placements mtime file-clock epoch (server-models.cpp:2109) | LOW | rejected - cosmetic |
| 5 | /fleet/status env value exposure (server-context.cpp:5909) | LOW | deferred |
| 6 | is_recr_layer stoul throw (llama-model.cpp:566) | LOW | accepted |
| Q1 | ffn_down island-exit: no layer-fleet t/s gate | open question | accepted - schedule hy3 A/B |
| Q2 | unpinned draft under layer-split target inherits fleet spread | open question | rejected - current shape is a measured win |
| Q3 | #105 subgroup pre-reduce A/B on the 3-GPU roster | open question | accepted - folded into arch-A bring-up |
| Q5 | worker cache-cap enforcement only at full idle | open question | deferred - fix shape recorded |

(Q4 in the review doc is a sighting of already-tracked TASKS residuals, not an
action item - no row needed.)

---

## Item 1 - probe_cache race in /wizard/hw

Status: **accepted** (2026-08-13)

Agreed shape: small dedicated mutex next to the function-local static cache;
the lock covers only map reads/writes, never the 800ms endpoint probe itself
(duplicate concurrent probes of the same endpoint accepted as harmless).
Not hoisted into server_models state - keeps model operations clear of probe
latency.

Gate: build-gated; concurrent-curl smoke loop optional; deterministic race
repro skipped as disproportionate (same standard as the 2026-08-01 #3 fix).

## Item 2 - /wizard/placements/generate path pinning

Status: **accepted** (2026-08-13)

Agreed shape: extract the remove endpoint's canonical-path + roots-prefix check
into one shared helper in server-models.cpp; apply to `profile` at the top of
generate with the same "outside the scanned roots" rejection. No UI behavior
change (the picker only surfaces root-listed files by construction).

## Item 3 - g_rpc_session_model process-global

Status: **rejected** (2026-08-13) - not triggerable today

Evidence gathered during review discussion: server_models serializes all model
loads (`is_reloading` + cv, "load() blocks on !is_reloading",
server-models.cpp:808/943), and the manifest handshake that consumes the global
runs inside the owning load's weight-upload path. Two loads cannot interleave,
so the wrong-id window does not exist in current usage. Action: rationale
recorded here; one documenting comment at the g_rpc_session_model declaration
noting the serialized-loads assumption (no behavior change).

## Item 4 - placements mtime file-clock epoch

Status: **rejected** (2026-08-13) - cosmetic, no consumer

Checked during discussion: neither wizard.html nor the Svelte app reads the
`mtime` field - it exists only as the endpoint's own newest-first sort key,
where a consistent-but-unspecified clock epoch is harmless (same process, same
clock domain). No constructible trigger today; revisit only if a UI date
display or a cross-language consumer is added.

## Item 5 - /fleet/status env value exposure

Status: **deferred** (2026-08-13)

Threat model today: private fleet LAN; a client able to reach /fleet/status can
already exercise more dangerous admin surface (wizard delete/load). Realistic
secret vectors in the LLAMA_*/GGML_* namespace are thin (LLAMA_API_KEY implies
auth, which then gates this endpoint). Agreed fix shape when it lands: redact
values at capture for names matching *KEY*/*TOKEN*/*SECRET*/*PASS* (~4 lines),
keeping gate values visible for reproducibility.

## Item 6 - is_recr_layer stoul throw

Status: **accepted** (2026-08-13)

Agreed shape: digit check at the parse position before stoul; a name with no
digit run is treated as non-recurrent (defined behavior) instead of throwing
std::invalid_argument at load. Trigger needs a crafted/corrupt GGUF, but extra
tensors do get instantiated and reach the split-state derivation, so the path
is reachable. ~2 lines.

## Open questions - decisions

- **Q1 ffn_down island-exit layer-fleet gate: accepted - schedule the hy3
  layer-fleet A/B** (t/s before/after 6f89ad52e against the 4.61-4.80 t/s
  dual-role record). The only perf-regression doubt in the window; closed by
  measurement, not argument.
- **Q2 unpinned draft under a layer-split target: rejected.** The inherited
  fleet spread is a measured win (layer+dspark 2.81 vs 2.01 t/s target-only,
  +40%). Main-GPU demotion recorded as a possible future A/B, no action now.
- **Q3 #105 subgroup pre-reduce: accepted - folded into the architecture-A
  bring-up** (research/true-parallel-next-architecture-2026-08-13.md). The
  uniform CUDA roster gets the full-set comm context without the opt-in; the
  subgroup path remains relevant for mixed rosters (eplocal CPU member).
- **Q5 cache-cap enforcement at idle only: deferred.** Fix shape on record:
  enforce at manifest-handshake time, evicting only non-current-model entries
  (safe: the client has not fetched its manifest yet). Land when worker disk
  pressure actually appears.

---

## Implementation queue (agreed 2026-08-13)

1. Item 1: probe_cache mutex in get_wizard_hw (server-models.cpp:2025) -
   lock map access only, probe outside the lock. Build-gated.
2. Item 2: shared canonical-path roots check applied to /wizard/placements/
   generate (server-models.cpp:2137), same rejection as remove.
3. Item 6: digit guard before stoul in is_recr_layer (llama-model.cpp:566).
4. Item 3 (documentation only): comment at g_rpc_session_model
   (ggml-rpc.cpp:1343) noting the serialized-loads assumption.
