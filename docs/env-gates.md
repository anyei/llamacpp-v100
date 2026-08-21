# Environment Gates — V100 Fork Reference

This fork adds a number of **environment-variable gates** to enable, tune, and
debug its features (tensor parallelism, MTP speculation, SSD streaming, the meta
tensor-split backend, distributed inference, and instrumentation). This page
consolidates them.

Conventions:
- **Boolean** gates are "set = on" unless noted: any non-empty value (often `=1`)
  enables; unset = default. A few default *on* and take `=0` to disable.
- **Value** gates take a number (MiB, bytes, count, ...).
- Defaults are what you get when the variable is **unset**.
- These are this fork's gates. The stock ggml backend knobs (`GGML_VK_*`,
  `GGML_METAL_*`, `GGML_HEXAGON_*`, `GGML_CUDA_*` upstream ones, `HF_*`, ...) are
  not listed here; a few upstream gates that matter for a fork feature are marked
  *(upstream)*.

---

## 1. SSD streaming — CPU tier (task 15)

Run MoE models whose expert weights exceed VRAM+RAM by streaming routed experts
from SSD into a bounded RAM cache. See `docs/ssd-streaming-plan.md`.

> The common ones have **CLI flags** (preferred over the env vars):
> `--ssd-streaming`, `--ssd-stream-budget <MiB>`, `--ssd-stream-gpu`,
> `--ssd-stream-vram-budget <MiB>`. E.g.
> `llama-server -m model.gguf -ngl 99 --ssd-streaming --ssd-stream-gpu --ssd-stream-vram-budget 6000`.
> The remaining knobs below are env-only tuning/diagnostics.

| Gate | Type | Default | What it does |
|---|---|---|---|
| `LLAMA_SSD_STREAM_BUFFER` | bool | off | Master switch: route `blk.*.ffn_*_exps.weight` to the streamed buffer type (RAM cache, filled on demand from the GGUF). |
| `LLAMA_SSD_STREAM_BUDGET` | MiB | 8192 | RAM expert-cache byte budget (LRU-bounded resident experts). |
| `LLAMA_SSD_STREAM_READ_THREADS` | count | 1 | Parallelize the SSD miss-path preads across N workers (each with its own O_DIRECT bounce). Helps the **batch-amortized** path — prefill/long prompts (+31% on DeepSeek-81GB), saturates at 2; single-stream decode on the GPU tier is neutral. Since #55 the CPU tier (`-ngl 0` + SERIAL) also batches its per-node miss reads through this: measured **+17% decode** (1.11 -> 1.30 t/s, V4-86GB @12GB budget, rt=4) — MB-scale random reads are latency-bound at QD1. Default 1 = serial. |
| `LLAMA_SSD_STREAM_PREFETCH` | count | 0 (off) | (#55) Router-lookahead prefetch: predict the next N streamed tensors' expert sets from the PREVIOUS token's routing and pread them from a background thread (3 tensors ~= 1 MoE layer). **Measured ineffective for DeepSeek-V4: at practical budgets the prediction is redundant with the LRU (issued=0 — last-token experts are still resident), at starvation budgets accuracy drops to ~28% and the churn is a net loss (0.87 -> 0.63 t/s @2GB).** Kept opt-in as the harness for a future TRUE router-lookahead (compute layer L+1's router on layer L's hidden state — predicts the CURRENT token's experts, which LRU cannot already hold; needs graph surgery). The debug dump reports issued/filled/used (accuracy) + top-decile routing-skew share. |
| `LLAMA_SSD_STREAM_SLRU` | bool | off (LRU) | Use a segmented LRU (probation/protected) for the RAM cache instead of plain LRU. Measured no-win for DeepSeek; kept opt-in. |
| `LLAMA_SSD_STREAM_PROTECTED_PCT` | 0-100 | 80 | Protected-segment size for the RAM SLRU (only when SLRU on). |
| `LLAMA_SSD_STREAM_SERIAL` | bool | off | Force node-at-a-time execution. **Required for `-ngl 0`** (pure CPU), where the router and experts share a scheduler split. Not needed when `-ngl > 0`. |
| `LLAMA_SSD_STREAM_NO_ODIRECT` | bool | off | Force buffered reads instead of O_DIRECT (kill-switch / A-B diagnostic; O_DIRECT is the default and keeps the page-cache/cgroup charge bounded). |
| `GGML_SSD_STREAM_DEBUG` | bool | off | Periodic hit/miss/evict/hit-rate/bytes-read log line. |
| `LLAMA_SSD_STREAMING` | bool | off | **Legacy/experimental** advisory mmap "residency director" (phase-1 negative result). Distinct mechanism from the streamed buffer above; shares `LLAMA_SSD_STREAM_BUDGET`. Prefer the streamed buffer. |

## 2. SSD streaming — GPU landing (increment 3)

Compute streamed experts on the GPU with a persistent VRAM slot cache
(hot experts stay resident across tokens; a cache hit is a zero-copy GPU read).
Composes with the CPU tier above (RAM becomes the L2 victim tier).

> **Single-GPU only (today).** GPU landing is a no-op under multi-GPU (`-sm tensor`
> or `-sm layer`); the slot cache doesn't allocate. On 2 GPUs, use plain
> `--ssd-streaming` (CPU tier, layer-split) - it works and is coherent, but gives
> no speedup over one GPU. See `docs/ssd-streaming-plan.md §8.3`.

| Gate | Type | Default | What it does |
|---|---|---|---|
| `LLAMA_SSD_STREAM_GPU` | bool | off | Enable the VRAM expert-slot cache + id->slot indirection. Requires `LLAMA_SSD_STREAM_BUFFER=1`. |
| `LLAMA_SSD_STREAM_VRAM_BUDGET` | MiB | 4096 | **Total** VRAM budget for the slot pools. |
| `LLAMA_SSD_STREAM_GPU_NO_RECLAIM` | bool | off | Kill-switch: keep the full-size `input_cpy` gallocr reservation for streamed-expert matmuls. By default the copy is shrunk to one slice (its data is redirected to the slot pool anyway) — reclaiming ~0.3-0.8 GB of compute-arena VRAM. Set to A-B the reclaim. |
| `LLAMA_SSD_STREAM_VRAM_POOLS` | count | auto | Override the number of expert-slice classes the budget is split across. **Auto-detected** from the model (e.g. 2 for DeepSeek-V4, 4 for mixed-quant UD Qwen); only set this to override. |
| `LLAMA_SSD_STREAM_GPU_SLRU` | bool | on | Segmented-LRU (scan resistance) for the VRAM slot cache; `=0` forces plain LRU. (Measured neutral for DeepSeek; may help more-skewed models.) |
| `LLAMA_SSD_STREAM_GPU_PROTECTED_PCT` | 0-100 | 80 | Protected-segment size for the VRAM SLRU. |
| `GGML_OP_OFFLOAD_MIN_BATCH` | count | 32 | *(upstream)* Min tokens for a `MUL_MAT_ID` to offload to the GPU. **Not needed for GPU landing** - streamed-expert matmuls auto-offload at batch-1 when `LLAMA_SSD_STREAM_GPU=1`. |

## 3. Speculative decoding / MTP (tasks 1, 20)

| Gate | Type | Default | What it does |
|---|---|---|---|
| `LLAMA_SPEC_DRAFT_NO_PAD` | bool | off | Kill-switch: disable MTP draft-length padding to fixed `n_max`. Padding keeps the batch shape stable for graph reuse and is the default win (33.6 -> 62-65 t/s); only set this to A-B the effect. |
| `LLAMA_SPEC_ADAPTIVE` | bool | off | Cap each draft round by an acceptance EMA. Measured tg-neutral; kept off. |
| `LLAMA_SPEC_PEARL` | bool | off | (#108) PEARL post-verify draft-ahead: draft the NEXT window on a worker thread during the target's verify decode; adopted next round when the assumed prefix held. Requires an INDEPENDENT drafter (draft-simple) with a PART-type draft context - feature drafters (dspark/dflash/eagle3/mtp) need target hiddens and are excluded by the type gate. Tiny vehicles run slower by construction (nothing to hide the draft under). |
| `LLAMA_SPEC_ALT_STATS` | bool (value-parsed) | off | (#132) Instrument: score the drafter's runner-up candidates at every rejected draft position (`SPEC_ALT:` line every 16 rejections, needs `-v` for LOG_INF) — bounds the multi-candidate/tree upside. Capture lives in the sampler-based drafter loops. **CORRECTED 2026-08-21 (this row previously had it backwards):** capture IS present for **mtp** (`spec_alt_begin` speculative.cpp:1848 + `spec_alt_push` :1933), alongside draft-simple (:479/:539) and dspark/dflash (:1434/:1504,:1526) — so the instrument DOES bound the tree upside on MTP serves. **eagle3 is the broken one**: it calls `spec_alt_push` (:1017) but never `spec_alt_begin`, so its per-seq capture grows unbounded and the stats index into stale rounds (code-review finding #49). Measure at PRODUCTION temp (temp 1), never temp-0 probes. Serve-safe counters, but not a baseline instrument. |
| `LLAMA_SPEC_TREE` | bool (value-parsed) | off | (#132) Multi-candidate verification v1: one alt branch per verify round (drafter runner-up at draft position 0 + shared continuation) on a per-slot spare seq; rescued first-position rejections bank ~2 extra tokens. Requires --kv-unified + draft-simple-only roster + PEARL off (type-gated at load, WARN otherwise). Lossless: temp-0 byte gate holds. `SPEC_TREE:` counters every 64 armed rounds. |
| `LLAMA_SPEC_DUMP` | path | off | (#132) Append one record per fresh verify round (`anchor_pos;draft_tokens;accepted_tokens`) for offline cross-drafter/stream analysis. TRAP: temp-0 streams from different batch shapes fork at near-ties - never join dumps by absolute position; use per-request sequence alignment (xdraft-join3.py pattern). |
| `LLAMA_SPEC_DRAFT2` | path | off | (#132, experimental) Register a SECOND drafter (own model+context) behind the primary in the priority-fallback dispatch: later impls only draft sequences the earlier ones left empty. With `LLAMA_SPEC_DRAFT2_TYPE` (draft-simple/eagle3/mtp/dflash/dspark; default draft-simple) and optional `LLAMA_SPEC_DRAFT2_DEVICE` (e.g. `RPC0` = a remote worker). draft-simple secondary owns its mirror hygiene (self-trims, survives ckpt restores); mtp/eagle3 as secondary UNGATED on ckpt-class targets (their process may desync after restores - gate before trusting). Verified: ngram primary + remote draft-simple secondary drafts 35/35 on fresh prose, byte-stable. |
| `LLAMA_SPEC_TIMING` | bool | off | Log server speculation phase timings (draft/verify/accept). |

## 4. Tensor-parallel AllReduce over NVLink (task 4)

| Gate | Type | Default | What it does |
|---|---|---|---|
| `GGML_CUDA_ALLREDUCE` | `p2p` | NCCL | `=p2p` uses a one-shot P2P NVLink AllReduce for 2 GPUs (falls back to NCCL). |
| `GGML_CUDA_AR_P2P_MAX_BYTES` | bytes | 4 MB | Size cap above which the P2P path defers to NCCL. |
| `GGML_CUDA_AR_COPY_THRESHOLD` | bytes | tuned | Byte threshold for the P2P AllReduce copy path (tuning knob). |
| `GGML_CUDA_AR_COPY_CHUNK_BYTES` | bytes | 0 (off) | Chunk size for the P2P copy path; 0 keeps it unchunked. |
| `GGML_CUDA_AR_BF16_THRESHOLD` | count | 1 | Element threshold above which the reduce runs in bf16. |
| `GGML_CUDA_FORCE_GRAPHS` | bool | off | Force CUDA graphs on Volta (cc<8.0). Historical: Volta graphs were runtime-disabled pre-merge; since 601ad05a9 upstream enables them by default on Volta, so this is a no-op on current builds (kept for older images). |
| `GGML_CUDA_SYNC_NODES` | bool | off | **(task 75)** Synchronize after every graph node and abort naming the FIRST faulting node (with its srcs' names/shapes/pointers). Compute only, so unlike `CUDA_LAUNCH_BLOCKING=1` the weight upload keeps full speed - that difference is what made it usable (launch-blocking turned a 7 s load into 20+ min). This is what named `ffn_moe_gate-1 (MUL_MAT_ID)` in the #75 gate-5 hunt; a sticky CUDA error otherwise surfaces at an unrelated later call (`cublasCreate`, `cudaFuncGetAttributes`, an NCCL allreduce...). Combine with `GGML_CUDA_DISABLE_GRAPHS=1`: the sync is illegal while a CUDA graph is capturing. |
| `GGML_CUDA_MMID_NO_DST_ZERO` | bool | off | **(task 75)** Skip the dst pre-zeroing in `mul_mat_id`. Measurement aid only - the zeroing exists so skip-sentinel lanes cannot surface recycled garbage, so this is ONLY valid where no sentinel can appear (expert placement off). Measured cost on the MTP spec-decode config (Qwen3.6-27B, 2x V100, -c 32768): tg 73.61 t/s with vs 73.79 without, i.e. 0.24% against a ~2.8% run-to-run spread - no detectable regression. |
| `GGML_CUDA_CHECK_IDS` | bool | off | **(task 75)** Bounds-check `mul_mat_id` expert ids against `src0->ne[2]` at the point of use and report node/device/offending value. Note it only checks the VALUES; the #75 crash was in-range ids that were not DISTINCT per token. **Combine with `GGML_CUDA_DISABLE_GRAPHS=1`, same as `GGML_CUDA_SYNC_NODES`:** the check does a D2H `cudaMemcpyAsync` + `cudaStreamSynchronize` at the top of `ggml_cuda_mul_mat_id`, which is ILLEGAL while a CUDA graph is capturing — and graphs stay ENABLED for quantized MoE decode (the very case this gate exists to debug), so without the companion gate it aborts the process a few tokens in. |
| `GGML_CUDA_NO_CONCURRENT_STREAMS` | bool | off | **(task 75)** Keep every node on the main stream (no fork/join across the concurrent-stream scheduler). A/B switch for isolating cross-stream ordering bugs. |
| `GGML_CUDA_DISABLE_GRAPHS` | bool | off | *(upstream)* Disable CUDA graphs. **Set in the MTP production compose**: draft/verify shape churn makes graph re-capture a ~5% net loss for spec decode (measured 75-78 t/s off vs 70-73 on). Leave ON (default) for plain decode - Volta graph coverage is part of the upstream V4 win (#65). |

## 5. Meta tensor-split backend (tasks 2, 8, 13, 16, 17)

The meta backend wraps N GPUs as one device for tensor parallelism.

| Gate | Type | Default | What it does |
|---|---|---|---|
| `GGML_META_DEBUG` | 1 / 2 | off | Split-state diagnostics; `=2` also prints per-source resolution (`SRC_RESOLVE`). With expert placement active (TASKS #75), `>=1` also runs the gate-4 ownership audit: per member, mul_mat_id (slot, row) pairs computed vs owned vs SKIPPED (sentinel lanes), `EXPERT_AUDIT:` lines per graph + finals at free, loud WARN per non-owned pair computed off the dummy slot (pins the flat id/mask gathers as graph outputs so the readback is valid); `=2` additionally runs the audit selftests (injected bad table / bad pair must be caught). **Caveat (2026-07-26): the runtime counters are UNRELIABLE ON CUDA** - the end-of-graph readback of the flat id/mask gathers reports violations that `GGML_CUDA_CHECK_IDS` (no bad ids at the point of use), the CPU audit (0 violations) and the CUDA PPL gate all contradict. Trust the CPU audit; treat CUDA violation output as a broken instrument until the readback is fixed or dropped. |
| `GGML_META_DEBUG_REDUCE` | bool | off | Print AllReduce boundary placement (`partial N -> boundary M`). |
| `GGML_META_CHUNK_SYNC` | bool | off | Drain every member (`ggml_backend_synchronize`) before each chunked graph piece runs — discriminates cross-piece ORDERING faults from wrong VALUES. Value-parsed (`=0` is off). SERIALIZES the member pipeline, so an instrumented leg is never a baseline. |
| `GGML_META_DEBUG_KVSUM` | substring | off | Per-member checksums of cache leafs whose name contains the given substring, after every compute call (two-owner seam dissection: does a non-root owner's KV *write* go wrong under chunking, or only its *read*?). Also traces per-member COMPUTE flags of the matching `SET_ROWS` writes at build time. Takes a NAME SUBSTRING, not a boolean — any non-empty value selects. |
| `GGML_META_DEBUG_BSUM` | bool | off | Per-boundary value checksums (`BSUM:` lines): per-member partial sums + FINAL at star reduces, delivered value at bcast1 boundaries. Printed from the reduce itself, so ctl and eval-callback-chunked legs are observable without perturbing graph structure (the 2026-08-04 seam dissection instrument). Adds a sync+read per bcast1 - not a baseline. |
| `LLAMA_CB_CHUNK_ONLY` | bool | off | Install an eval callback that chunks the sched at every base `ffn_moe_topk-*` exactly like ZL/UNION_STATS but never reads the tensor - separates graph-chunking effects from gather reads (seam-dissection discriminator). Chunked graphs run the plain delivery path (see `GGML_META_NO_FUSED` note). |
| `GGML_META_MAX_GRAPHS` | count | 8 | Compute-ring shadow-container slots (raise if many decode graph shapes are cached). |
| `GGML_META_TIMING` | bool | off | Per-step compute vs reduce timing for the meta device. Adds device syncs - instrumented legs are NOT baselines. |
| `GGML_META_BOUNDARY_STATS` | bool | off | Structural tallies of boundary traffic every 128 graphs (boundary kinds, partials via fused pre-request vs plain reads, wire deliveries vs skips, bytes, repairs). Pure counters, no drains - valid on a production-speed run, unlike GGML_META_TIMING. |
| `GGML_RPC_WIRE_F16` | bool | off | **(proto 4.12)** Boundary payloads in the fused pipeline ride f16 instead of f32 (SET compressed by the coordinator + expanded by the worker; FETCH response compressed by the worker + expanded by the paired recv). Per-connection: workers on proto <=4.11 silently stay f32. LOSSY but PPL-neutral (3.7669 vs 3.7895 +/- 0.21, record roster); fleet ~+15-20% decode. Engagement: one-time `rpc: compressed boundary payloads ACTIVE` INFO line (needs `-v`/`-lv` to surface). **PRESENCE-GATED, NOT value-parsed: `=0` still ENABLES it.** The fleet compose files (`docker-compose.ep-fleet.yml`, `docker-compose.fleet-coordinator.yml`) document `COORD_WIRE_*=0` as the disable knob, which is a NO-OP — any past "compression off" A/B leg run that way actually measured on-vs-on. To disable, UNSET the variable. (code-review 2026-08-19 finding #6) |
| `GGML_RPC_WIRE_Q8` | bool | off | **(proto 4.13)** q8_0 boundary payloads (~3.76x cut, 34 B per 32 values; sizes not divisible by 32 fall back to f16/f32). Takes precedence over WIRE_F16 on workers speaking minor 13 - set BOTH so each connection rides the best format it can. PPL-neutral (3.7950 vs f32 3.7895 +/- 0.21); fleet mean 4.01 t/s, tightest spread of the wire-format legs. **PRESENCE-GATED, NOT value-parsed: `=0` still ENABLES it.** The fleet compose files (`docker-compose.ep-fleet.yml`, `docker-compose.fleet-coordinator.yml`) document `COORD_WIRE_*=0` as the disable knob, which is a NO-OP — any past "compression off" A/B leg run that way actually measured on-vs-on. To disable, UNSET the variable. (code-review 2026-08-19 finding #6) |
| *(proto 4.14 zero short reply)* | automatic | on | **(TASKS #71 inc-1)** A member whose fused FETCH payload is bitwise zero (all mul_mat_id lanes sentinel under placement) answers with a 1-byte marker; the client stashes zeros. Requested per fetch (flag bit 128) when the worker speaks minor >= 14; pre-4.14 workers keep full payloads. Exact and free; fleet-measured NULL for t/s (member lateness is chain phase lag, not payload bytes) - kept as default behavior. Engagement: stderr `rpc: proto 4.14 zero short replies ACTIVE`. |
| `LLAMA_META_EP_ONLY` | bool | off | Expert-parallel split shape: segment only `ffn_*_exps` across meta members, mirror everything else (attention, dense FFN, shared experts, output head). Also a fault isolator for split-policy bugs (task 28). |
| `LLAMA_META_ATTN_OWNER` | index list | -1 (off) | Dedicate attention/KV/router to meta member `<j>` (a local GPU): non-owners get zero-size attention slices, the exit projection is dot-dim split so its output derives PARTIAL, and the delayed AllReduce broadcasts the owner's activations back (x+0=x). The core of the cross-box expert-parallel topology; composes with `LLAMA_META_EP_ONLY` (task 28 increment 3). DSA/V4 topology: the owner takes a **0** expert share. **Owner GROUP (task 70):** accepts a comma list (e.g. `0,1` = both local V100s over NVLink) for standard (non-DSA) archs - whole LAYERS interleave across the owners (`owners[il % n]`), so mirrors bigger than one device fit (hy3's 34.7 GB > 32 GB) without breaking the single-latent-cache invariants (feature-dim splitting across owners is NOT viable). Owners are dual-role there: they also hold expert shares (hy3 record: `-ts 21,21,46,50,27`). Guard: passes while n_local <= n_owners; beyond that a second local GPU as an expert member is rejected at load (task 48) unless `LLAMA_META_ALLOW_MULTI_LOCAL=1`. |
| `LLAMA_META_ALLOW_MULTI_LOCAL` | bool | off | Re-test gate for the task-48 corruption after major merges: allows more local members than the owner group spans (with a WARN). Outputs MUST pass the coherence gate before trusting such a config. |
| `LLAMA_META_EXPERT_PLACEMENT` | path | off | **(task 75)** Hot-expert placement: frequency-ranked whole-expert assignment of routed experts to meta members, from an artifact JSON generated by `scripts/expert-placement.py` out of a `LLAMA_EXPERT_PROFILE` histogram (see `placements/`, `docs/expert-placement-plan.md`). Unset = bit-for-bit today's uniform behavior. Artifact is roster-specific (member count must match the serve; regenerate on any `-ts` retune). Gates 1-4 passed (byte-exact + PPL-neutral + ownership audit); combine with `GGML_META_DEBUG>=1` for the `EXPERT_AUDIT` counters. |
| `GGML_META_SURGICAL_MAX_STATE_MIB` | MiB | 64 | Cap on replayed non-weight state size for the meta backend's surgical re-provision path. |
| `GGML_META_NO_DELAY` | bool | off | Reduce at every PARTIAL node instead of delaying AllReduce past the expert-merge tree. Diagnostic. |
| `GGML_META_NO_STAR` | bool | off | Disable the star reduce (batched-read partials + host sum + async broadcast, used when a local member can root the reduce); fall back to the fold+butterfly. A/B + diagnostic. |
| `GGML_META_LOCAL_COMM` | bool | off | **(TASKS #105)** Pre-reduce the LOCAL same-backend members (e.g. an NVLinked CUDA pair) among themselves via the backend comm path (NCCL/p2p), then send one representative into the host/wire star. Without it a MIXED roster (CUDA + RPC) gets no comm context at all - `ggml_backend_cuda_comm_init` refuses any non-all-CUDA member list - so every boundary stages through host RAM while NVLink idles. Engagement: `META_LOCAL_COMM: N subgroup pre-reduces/graph`. Measured: fleet 5-member A/B +2.4% (3.94 -> 4.03 t/s, p~0.12 = trend, not significance; the fleet is wire-dominated); stub output BYTE-IDENTICAL to the host path. Opt-in until the single-box EP A/B (CUDA0,CUDA1,CPU vs the 6.95 t/s baseline) - that shape is where host staging dominates. |
| `GGML_META_NO_FUSED` | bool | off | Disable the fused boundary pipeline (proto 4.5: reduced value + CHAIN of subgraphs up to the next reduce + next-partial request in ONE message per wire member per boundary). Also auto-disabled under GGML_META_TIMING, GGML_META_NO_STAR, against pre-4.5 workers, and for CHUNKED graphs (uid 0 eval-callback graph views - the pipeline's piece-boundary invariant breaks there and desynced wire members; 2026-08-04 root cause, research/84-seam-dissection-2026-08-04.md). Fused and plain delivery are legitimately fp-different (both self-consistent): byte-identity gates must compare like-vs-like paths. A/B + diagnostic. |
| `GGML_META_FUSED_BCAST` | 0/1/2 | 2 | Broadcast-boundary side of the fused pipeline: 0 = plain copies (no carriage), 1 = fused SET only, 2 = full chain carriage (default). Diagnostic A/B levels. |
| `GGML_META_BCAST_FUSE` | 0/1/2 | 0 | **(TASKS #9 boundary fusion)** 1 = walker crossing: a single-contributor PARTIAL (owner-broadcast pattern, dedicated attention) may cross an ADD with a MIRRORED operand, so B1 broadcasts `ffn_inp` instead of `attn_out`. 2 = additionally withhold a reduce boundary's writeback from wire members that provably never consume it (gather-only B2; fused messages go value-less, the shape the `GGML_META_PROBE_NO_WRITEBACK` probe priced at ~47 ms/token). Cross-piece readers are covered by a stale-value registry + repair copies at piece start. Delivery skip requires member 0 local (host reads + repair source); butterfly (`GGML_META_NO_STAR`) always delivers. `GGML_META_DEBUG_REDUCE=1` prints `SKIP-WB:`/`REPAIR:` engagement evidence. Gates: CPU loopback byte-exact (off==1==2==pre-change); do not combine with debug readers of pre-boundary tensors (eval-callback) at level 2. |
| `GGML_META_PROBE_DEFER_GATHER` | bool | off | **(TASKS #71 Expert Deferral probe)** The star gather stops WAITING for wire members' partials: fused responses are drained one reduce boundary late (values discarded, slot zeroed), plain wire reads skipped - sums hold local contributions only. **OUTPUT IS GARBAGE BY DESIGN**; wall-time-only probe pricing the deferral ceiling (owner compute overlapping wire arrival). Never serve with it. `DEFER=1` in run-ep-fleet-hy3-spec.sh. Measured ceiling on the record roster: +32%. |
| `GGML_META_EXPERT_DEFER` | 0/1/2 | 0 | **(TASKS #71 Expert Deferral)** Eligible star reduces sum LOCAL contributions and distribute; wire partials drain at the next multi-contributor star reduce and are ADDED there (one-reduce staleness; last reduce always exact - the logits path never defers; `META_EXPERT_DEFER` counter line under BOUNDARY_STATS, defers==injects and LOST 0 is the structural pass). **`=1` (v3, READINESS-GATED + DECODE-ONLY): each candidate's socket is polled at the gather (`fused_ready`, non-blocking poll(2) on the response FIFO) - a response that already arrived is consumed exactly, only actual stragglers defer, and ONLY on decode-shaped boundaries (ne[1]==1) - prefill is always exact (prefill defers poison the chunk KV: measured +0.6 PPL at even a 5.9% rate). Record roster: 4.93 t/s (+34%), 14/14 clean reads, PPL 3.7819 = baseline; counter line shows `ready` + defer rate. `=2` (v1, defer-ALL): measurement only - fleet +27% (5.07 vs 4.00) but COHERENCE FAIL (repetition collapse - it defers ALL wire mass one layer late); never serve with it.** `EXPERT_DEFER=` in run-ep-fleet-hy3-spec.sh. |
| `GGML_META_EXPERT_DEFER_PREFILL` | bool | off | Measurement override: allow `EXPERT_DEFER=1` deferral on prefill-shaped boundaries too (the v2 behavior). Poisons chunk KV - PPL 4.39-4.57 vs 3.78 baseline on the record roster. Never in production. |
| `GGML_META_EXPERT_DEFER_WAIT_US` | us | 0 | Grace window for `EXPERT_DEFER=1`: a not-yet-ready response is busy-polled up to this budget (shared per boundary) before it is declared a straggler and deferred. Trades boundary latency back for exactness; 0 = pure readiness. |
| `GGML_META_EXPERT_DEFER_SYNC_EDGE` | count | 0 | The first/last k multi-contributor star reduces of every graph never defer (field notes: shallow/deep layers tolerate deferral worst). Quality fallback if readiness gating alone fails coherence/PPL. |
| `GGML_META_EXPERT_DEFER_VERIFY` | width | 0 | **(fill-the-bubble 2.4 spec probe)** Widen the v3 decode-only gate to spec-verify-shaped boundaries: ne[1] <= value also defers (set to 1 + n_draft max, e.g. 4 for NMAX=3). 0/absent = decode-only; prefill stays excluded; parse-VALUES (=0 off). Loopback gated 2026-07-29: under padded spec v3 collapses to 0.1 defers/graph (verify is defer-blind), =4 restores 3.7-3.9 defers/graph, LOST 0, injects==defers. Fleet A/B 2026-07-29: PROBE FAILED ratio 0.83 (4.53 vs 5.49 v3 control, engaged, LOST 0, coherent) - verify cost is member COMPUTE, not arrival lateness; spec family stays closed. Keep default-off; instrument-class for future multi-token decode work. `DEFER_VERIFY=` in run-ep-fleet-hy3-spec.sh. |
| `LLAMA_META_LOCAL_DRAFT` | bool | off | **(TASKS #71 stage 1)** Localize the MTP draft graph to coordinator members (draft cost 13x lower over the fleet). Its PRESENCE (even `=0`) also unlocks `--device CPU` in arg parsing (loopback gate harnesses need it). Spec lane itself is measured shelved on this fleet (ratio <= 1.0). |
| `LLAMA_LP_PAIRS` | bool | off | **(TASKS #71 escape (b), LANE CLOSED 2026-08-01)** Layer-Parallel pair fusion: consecutive layer pairs run from the SAME residual input, deltas add (hy-v3 builder). LOSSY by design. Fleet verdict: boundary halving works (79 -> 41 stars) but only +8-13% t/s (per-lap member latency doubles) at CATASTROPHIC quality - measurement/research only, never serve. |
| `LLAMA_LP_SYNC_EDGE` | count | 2 | First/last k layers stay sequential under `LLAMA_LP_PAIRS` (min 1: the last layer must stay sequential for out_ids gating). |
| `GGML_META_PARTIAL_MERGE` | bool | off | **(TASKS #71 LP inc-1b)** Two expert partials share ONE reduce boundary: builder-tagged `lp_moe_pair` ADDs derive PARTIAL by name and their upstream tree boundaries skip the reduce (split kept). Exact (stub byte-identical to unmerged). Reusable for any two-partials-one-boundary shape; off = production untouched. |
| `GGML_META_ZL_STATS` | bool (parse value; `=0` off) | off | **(TASKS #71 replication inc-0)** Zero-leg counters: per-member zero-routed gather frequency + owner routed-slot share (`META_ZL` line under BOUNDARY_STATS). Installs an eval callback (routed-id capture at compute time) -> forces sched splits: MEASUREMENT ONLY, never a baseline; mutually exclusive with `LLAMA_EXPERT_PROFILE`. Needs PLACE tables. **2026-07-31 (#84c): eval-callback instruments corrupt meta-FLEET serve quality - loopback only until the split-seam bug is fixed.** |
| `GGML_META_ZL_DEBUG` | bool | off | One-shot diagnostics for the ZL machinery (registration + gather decisions). |
| `GGML_META_UNION_STATS` | bool (parse value; `=0` off) | off | **(TASKS #84)** Expert-union width curve: `META_UNION` line under BOUNDARY_STATS - per lane-width bucket the distinct-expert union per boundary + multiplier vs the width-1 floor (the verify member-read multiplier the #52 law priced as linear); `META_UNION_MEM` adds per-member owned-uniq (needs PLACE tables). Widens the routed-id eval-callback capture to ne1<=16. **FLEET-QUALITY-UNSAFE (2026-07-31, #84c): the callback's sched splits CORRUPT meta-serve output on fused AND plain stacks (repetition spiral; discriminated vs no-UNION coherent) - loopback stub gates only; the fleet number comes from the offline per-position profiler route.** Mutually exclusive with `LLAMA_EXPERT_PROFILE`. |
| `LLAMA_EXPERT_MASK` | path | off | **(TASKS #84 probe 2)** Per-layer routing-budget mask: file lines `il id id ...` = allowed experts; -INF on SELECTION scores pre-topk (gating weights of allowed experts untouched, cvec-style per-layer tensors). LOSSY measurement instrument for budgeted-verify pricing (argmax-flip vs budget via perplexity --kl-divergence) - never serve users with it. Unset = bit-for-bit untouched. |
| `LLAMA_EXPERT_PROFILE_IDS` | path | off | **(TASKS #84)** With `LLAMA_EXPERT_PROFILE` set: also dump PER-POSITION routed ids (binary int32 records: il, k, n_tok, ids lane-major) for the offline union-curve analysis (`scripts/union-curve-from-ids.py`). Use on SINGLE-BOX runs only (the profile callback is meta-fleet-quality-unsafe, see ZL/UNION rows). Note the 2026-07-31 profiler fix: topk is a strided VIEW - pre-fix prefill profiling read argsort-permutation bytes (decode profiling, e.g. the #74 artifact, was correct: [k,1] views are contiguous). |
| `GGML_RPC_DEBUG_FUSED_DELAY_US` | us | 0 | **Worker-side test hook:** delay every fused FETCH response by this much before sending, faking a straggler so the loopback gate can exercise `EXPERT_DEFER=1` deferral. Never in production. |
| `LLAMA_TRUNC_ARR` | bool | off | Accept longer-than-n_layer per-layer arrays and partial tensor loads, so `--override-kv <arch>.block_count=int:N` can truncate a model into a small fast reproducer. Debug only. |
| `LLAMA_META_DUP_DEVICE` | count | 1 | Duplicate the device list N times so ONE physical GPU runs a genuine N-way split (validation harness; e.g. `=2` reproduces 2-GPU exactness on one card). |

## 6. Decode graph cache (task 8)

| Gate | Type | Default | What it does |
|---|---|---|---|
| `LLAMA_DECODE_GRAPH_CACHE` | count | 4 | Number of small-batch decode graphs to cache (each with its own scheduler). `=0` disables. |
| `LLAMA_DECODE_GRAPH_CACHE_TOKENS` | count | 64 | Max ubatch size eligible for the cache. |

## 7. Distributed inference / RPC (task 12)

| Gate | Type | Default | What it does |
|---|---|---|---|
| `GGML_RPC_NO_W2W` | bool | off | Disable direct worker-to-worker tensor pull; fall back to bridged copies through the coordinator. |
| `GGML_RPC_THREADPOOL_POLL` | 0-100 | unset (off) | Worker-side (`rpc-server`): attach a PERSISTENT threadpool to CPU backends (unset = a disposable pool is created+joined per subgraph - measurable per-boundary turnaround on EP workers). The value is the busy-poll level: 0 = persistent but condvar waits, >0 = spin between subgraphs (trades idle cores for wake latency). Compose: `WORKER_THREADPOOL_POLL`. Loopback A/B: +2% decode, token-identical. |
| `GGML_RPC_CACHE_LIMIT_MIB` | MiB | 0 (off) | Worker-side (`rpc-server -c`): cap the tensor-cache dir; after each save, evict least-recently-USED entries (serves refresh mtime) until it fits. Also enforced ONCE AT STARTUP (task 45) so an idle worker holding stale model generations trims before the next load. The beacon publishes the current cache size (`cache_mib=`, fleet UI "Cache" column). Compose: `WORKER_CACHE_LIMIT_MIB`. A deleted cache dir is also recreated on the next save now (used to silently kill persistence until restart). |
| `GGML_RPC_NO_SRC_HINT` | bool | off | Disable the proto-4.8 slice-provenance path (task 44): coordinator falls back to plain `SET_TENSOR_HASH` offers. With it ON (default), EP/tensor-mode weight slices carry (tensor name, offset, row geometry) so a `--model-dir` worker serves them by pread from its local GGUF regardless of where the `-ts` split boundaries fall - share changes no longer cold-stream. Every serve is hash-verified; a stale local file degrades to streaming, never corruption. |
| `LLAMA_FLEET_CAPACITY_CHECK` | bool | on | (server) Fleet capacity gate: when RPC devices are in the pipeline, refuse to START a load whose weights + KV reserve exceed the pooled free device memory - the server waits in `waiting-capacity` state (visible in `/fleet/status` + fleet UI) and, under `--rpc-discover`, exits 42 automatically once newly beaconing workers raise the pool enough (restart policy re-discovers and loads). `=0` disables. Single-box (no-RPC) runs are never gated (mmap paging / -ncmoe / ssd-streaming are supported over-capacity regimes). |
| `LLAMA_FLEET_KV_RESERVE_MB` | MiB | 20480 | Headroom the capacity gate adds on top of the model weight bytes (KV + compute buffers + fragmentation margin). |
| `LLAMA_FLEET_LOCAL_BENCH` | bool | on | (server, TASKS #136) The #131b load-time local-device bench (same matmul bench the workers run for `--score`; fills the `/fleet/status` score column for local devices). `=0` disables it. Even when on, a device reporting < 192 MiB free is auto-skipped (another serve may hold it near-full), and the bench runs behind scoped CUDA error containment: a failed bench costs only its score row, never the load. |
| `LLAMA_RPC_NO_SURGICAL` | bool | off (surgical ON) | With `--rpc-reload`: disable the surgical re-provision (returned worker's share replayed from its own cache, ~2min for a 48GB share vs ~10+min reload; falls back to the reload on any failure) and always do the full in-process reload. `LLAMA_RPC_SURGICAL_WAIT_S` (120) = how long to wait for a dead endpoint to return; `GGML_RPC_JOURNAL_MAX_MIB` (4096) = small-write spill cap; `GGML_RPC_REPROVISION_VERIFY=1` = read back and hash-verify every replayed region. |
| `LLAMA_ARG_RPC_RELOAD` | bool | off | (= `--rpc-reload`, server only) On RPC worker loss: fail in-flight requests, then reload the model IN-PROCESS across the workers reachable at that moment (dead workers drop with their positional `-ts` shares; a returned worker is re-included by the next failure-triggered reload; all-dead degrades to local-only loudly; a load that fails - fleet-sized models - retries every 10s). Default off = #29b behavior: exit 42 for the restart policy. |
| `LLAMA_ARG_RPC_SKIP_UNAVAILABLE` | bool | off | (= `--rpc-skip-unavailable`) Drop unreachable `--rpc` servers with a warning and split the model across the remaining devices, instead of exiting with an error. Load-time; a worker dying mid-session is handled separately (29b: requests error cleanly, server exits for restart+rediscovery). |
| `LLAMA_ARG_RPC_DISCOVER` | bool | off | (= `--rpc-discover`) Discover RPC workers announcing themselves on the LAN (`rpc-server --announce`) and use them; composes with `--rpc`, duplicates skipped. Trusted networks only. |
| `LLAMA_KV_WORKER_HOST` | bool | off | KV annex (task 30): a worker exposing GPU+CPU (`rpc-server -d CUDA0,CPU`, e.g. `WORKER_EXTRA_ARGS="-d CUDA0,CPU"`) gets its CPU device reserved — no layers, but the KV cache of that worker's GPU layers lives there (worker RAM). Weights stay in VRAM → a small-VRAM card holds ~2× the layers; measured ~7% decode cost (0.6B/32k loopback). |
| `LLAMA_ARG_RPC_DISCOVER_GROUP` | str | built-in | (= `--rpc-discover-group`) Multicast group `ADDR:PORT` for discovery; must match the workers' `--announce-group`. |
| `GGML_CUDA_ERROR_CONTAIN` | bool | off (rpc-server sets 1) | CUDA errors throw to the RPC compute boundary instead of aborting the process. Armed automatically by `rpc-server` (task 29e: a compute error drops one connection, the worker lives; 3 consecutive failures = poisoned backend → clean exit for the restart policy). Set `=0` on a worker to restore abort-with-backtrace. Never armed in coordinators/tools. |
| `GGML_CUDA_INJECT_COMPUTE_FAIL` | int | off | Fault injection for the containment path: `N>0` fails the Nth graph compute and after (poisoned backend); `N<0` fails only the \|N\|th (isolated error — the worker must recover). |
| `GGML_RPC_TIMING` | bool | off | Worker-side (`rpc-server`): per-command latency breakdown - `lock avg` (cross-connection exec-mutex contention wait) and `exec avg` / `exec max` (the handler, i.e. graph_compute + response send) per command type, logged every 20k commands and at each connection close. Decomposes the coordinator's end-to-end RTT (the fleet UI's per-worker `ms`, from `rpc_ep_stat`) into worker-processing vs network+coordinator residual - the direct measurement of the EP boundary turnaround (task 28). Look at `GRAPH_RECOMPUTE_UID`/`GRAPH_FUSED` for the per-boundary compute, `GET_TENSOR` for star-reduce partial reads. The fleet UI surfaces the busiest graph command's `exec avg` as a per-worker badge (slowest-above-median highlighted), parsed from these dumps via `GET_LOG` (#58). Value-parsed since 12ee78006 (`=0`/empty = off; before that, a compose-style empty `GGML_RPC_TIMING=` turned the dump flood ON). |
| `GGML_RDMA_DEV` / `GGML_RDMA_GID` | str | auto | RDMA device / GID selection for the RPC transport (when built with RDMA). |
| `GGML_RPC_DEBUG` | bool | off | *(upstream)* RPC command logging. |
| `GGML_RPC_STATS` | bool | off | Client-side per-RPC-command call/byte counters, dumped to stderr at exit. Diagnostic; used to find the O(n^2) graph serialization and the per-row weight upload (task 28 increment 1). |
| `GGML_RPC_DEBUG_FAIL_ALLOC` | `EP:SKIP:COUNT` | off | Fault injection: on endpoint `EP`, skip the first SKIP buffer allocations, fail the next COUNT, pass the rest (and log every request). Reproduces a worker rejecting a compute-buffer alloc (task 37 Run-B scenario: fail the PP compute buffer, let the no-pipeline retry pass). |
| `GGML_RPC_DEBUG_BUF` | bool | off | Client-side buffer lifecycle log: every remote alloc (endpoint, remote_ptr, size) and free. Used to find the zero-size-chunk dummy buffer in task 37. |
| `GGML_META_TILED_UPLOAD` | bool | **on** | (#113) Tiled weight uploads to RPC members: cut each member's split segment on a fixed grid anchored in root-tensor coordinates (AXIS_1 = dim-1 column tiles ~256KiB, e.g. gate/up_exps strips; AXIS_2 = whole-expert slabs, placed layouts) instead of hashing the whole segment as one blob. Interior tiles keep their content hash across `-ts` changes, so reloads after a share retune batch-place from the worker disk cache and only boundary/moved tiles stream. AXIS_0 segments (sub-row slivers, e.g. down_exps) keep the bulk path. `=0` restores whole-segment uploads. |
| `GGML_META_DEBUG_UPLOAD` | bool | off | (#113 diag) Log the scatter path (MULTISEG/SWITCH, axis, segments) for every expert-tensor upload plus per-member tiling decisions (usage, piece vs tile bytes, tiled?) for blk.0 gate_exps. The instrument that caught the 2MiB-tile-floor no-engagement bug. |
| `GGML_VBUF_DEBUG_FAIL` | `N` | off | Fault injection at the graph-allocator layer: log every vbuffer (compute-buffer) alloc and fail every one from the Nth on (0-based), regardless of backend. Use when `GGML_RPC_DEBUG_FAIL_ALLOC` cannot fire: meta-EP member compute buffers allocate through the Meta buft, never the plain RPC buft. Timeline on the trunc-V4 loopback: calls 0-3 = load/init (failing those = clean context-init throw), calls 4+ = decode-time reserves; `=5` reproduced the 2026-08-10 poisoned-galloc crash pair (gallocr NULL-vbuffer segfault + decode-cache set_inputs abort). |
| `LLAMA_RPC_AUTO_WEIGHT_RESERVE_MB` | MiB | 2% of model, clamped [512, 2048] | Override the per-device compute-buffer allowance `--rpc-auto-weight` subtracts from every device cap (task 37; KV is estimated from the GGUF header separately). Bump it if V4-class members OOM at decode (the default reserve can undershoot the actual compute buffer). |
| `LLAMA_RPC_AUTO_WEIGHT_MIN_SHARE` | fraction | 0.05 | Auto-weight floor: a member whose computed share falls below this fraction is floored to EXACT 0 and its bytes handed to the remaining uncapped members (tiny shares round badly at segment granularity). `atof`-parsed, so `=0` means "no floor", not "off". |
| `LLAMA_ARG_RPC_AUTO_WEIGHT` | bool | off | (= `--rpc-auto-weight`) Fill an unset `-ts` by each device's measured bandwidth score instead of by free memory, water-filled against capacity; local GPUs are benchmarked at startup. Also applies to `-sm tensor` EP splits (task 28 increment 3). Explicit `-ts` always wins. |
| `GGML_RPC_SCORE` | bool | off | Worker-side (= `rpc-server --score`): run a ~1s matvec benchmark at startup (effective memory bandwidth, the decode-bound quantity) and publish the score in the discovery beacon + over RPC, for `--rpc-auto-weight`. Benched ONCE at startup — restart the worker when the box is idle if a busy start under-read it (TASKS.md #42). |
| `GGML_RPC_ALLOW_SHUTDOWN` | bool | off | Worker-side (= `rpc-server --allow-shutdown`): permit the coordinator to restart this worker over RPC (`POST /fleet/worker/restart`). Off = the worker refuses shutdown commands. |
| `LLAMA_ARG_FLEET_ADMIN` | bool | off | (= `--fleet-admin`, server) Enable `POST /fleet/worker/restart` and `POST /fleet/reload`. Requires an `--api-key` (these are remote-kill/reload primitives on an unauthenticated RPC fabric). |
| `LLAMA_ARG_FLEET_PREFLIGHT` | path | off | (= `--fleet-preflight <gguf>`, server) Before the main load, benchmark a small model across the SAME devices/split (times single-token decodes) and publish the result in `/fleet/status`. A small dense model's compute is negligible, so the number is the fleet's per-token boundary/latency floor — an upper bound for any model on this topology, not a throughput estimate. Never runs on a resume/failure reload. |

## 8. KV cache (task 10)

| Gate | Type | Default | What it does |
|---|---|---|---|
| `LLAMA_ATTN_ROT_DISABLE` | bool | off | Opt out of Hadamard-rotated KV quantization (auto-enabled whenever KV is quantized; the rotation is what makes q4_0 KV near-lossless). |

## 8b. CUDA FlashAttention kernel selection (task 32)

| Gate | Type | Default | What it does |
|---|---|---|---|
| `GGML_CUDA_FA_NO_MMA` | bool | off | Never select the MMA FlashAttention kernel (tile/vec instead). For devices that pass the cc gate but can't run it — GTX 16xx (TU116/117) is cc 7.5 without tensor cores. Set on the affected *worker*; without it the first failure self-heals per device (WARN + tile fallback) instead of aborting. |
| `GGML_CUDA_FA_MMA_FORCE_SMEM_FAIL` | bool | off | Fault injection: pretend the MMA kernel's shared-memory opt-in failed, to exercise the fallback path on healthy hardware. |

## 9. Instrumentation & debug

| Gate | Type | Default | What it does |
|---|---|---|---|
| `LLAMA_DECODE_TIMING` | bool | off | Per-ubatch build / alloc / set-inputs / compute timing. |
| `LLAMA_BATCH_DEBUG` | 1 / 2 | off | Dump batch contents (`=2` = full per-token dump). |
| `LLAMA_DEBUG_DUMP_DIR` | path | off | Dump full tensors to a directory for elementwise diffing (pairs with the eval-callback). |
| `LLAMA_DEBUG_DUMP_FILTER` | substr | none | Only dump tensors whose name matches this substring. |
| `LLAMA_DSV4_COMPRESS_DEBUG` | bool | off | DeepSeek-V4 KV-compression debug logging. |
| `GGML_SCHED_DEBUG` | 1 / 2 | off | *(upstream)* Scheduler split/backend-assignment dump. |
| `LLAMA_EXPERT_PROFILE` | path | off | **(task 74)** Per-layer router expert-selection profiler: hooks the MoE top-k tensor per layer, accumulates `counts[layer][expert]`, refreshes the JSON every ~500 decode tokens. Works in ANY split mode (the router runs everywhere), so profile on a cheap serve. Costs ~1.5 t/s while on - profile runs are not benchmarks. Feeds `scripts/expert-placement.py` (task 75); procedure: `docs/expert-profiling.md`. |

---

## Usage examples

### Run a huge MoE from SSD on one GPU (CPU-compute expert tier)

DeepSeek-V4-Flash 81 GB on a single 32 GB V100 + 46 GB RAM (experts stream to a
30 GB RAM cache, non-experts on the GPU):

```sh
docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES=0 \
  -e LLAMA_SSD_STREAM_BUFFER=1 \
  -e LLAMA_SSD_STREAM_BUDGET=30720 \
  -e GGML_SSD_STREAM_DEBUG=1 \
  -v /path/to/models:/models:ro \
  --entrypoint /app/llama llamacpp-local-v100:latest cli \
  -m /models/DeepSeek-V4-Flash-...gguf -ngl 99 --no-mmap -c 4096 -n 96 --temp 0 -st -v \
  -p "Explain how a CPU pipeline works."
```

Pure-CPU (no GPU) variant adds `-e LLAMA_SSD_STREAM_SERIAL=1` and `-ngl 0`.
Equivalent CLI flags (no env needed): `--ssd-streaming --ssd-stream-budget 30720`
(+ `--ssd-stream-gpu --ssd-stream-vram-budget <MiB>` for the VRAM slot cache).

### GPU landing (experts computed on the GPU via a VRAM slot cache)

Best on models whose hot expert set fits the VRAM cache (big win: Qwen-35B-A3B
2.5 -> 7 t/s). Note `VRAM_POOLS` should match the model's expert-slice classes:

Streamed-expert matmuls auto-offload to the GPU and the VRAM budget auto-splits
across the model's expert-slice classes, so only the two budgets are needed:

```sh
# Qwen-35B-A3B: cache covers the hot set -> big win (~9 t/s decode), byte-exact
-e LLAMA_SSD_STREAM_BUFFER=1 -e LLAMA_SSD_STREAM_BUDGET=24000 \
-e LLAMA_SSD_STREAM_GPU=1    -e LLAMA_SSD_STREAM_VRAM_BUDGET=6000

# DeepSeek-V4 81GB (experts >> VRAM): break-even with CPU; watch the hit rate
-e LLAMA_SSD_STREAM_BUFFER=1 -e LLAMA_SSD_STREAM_BUDGET=24000 \
-e LLAMA_SSD_STREAM_GPU=1    -e LLAMA_SSD_STREAM_VRAM_BUDGET=12000 \
-e GGML_SSD_STREAM_DEBUG=1
```

### 2-GPU tensor parallelism with fast NVLink AllReduce

```sh
-e CUDA_VISIBLE_DEVICES=0,1 -e GGML_CUDA_ALLREDUCE=p2p \
  ... -m model.gguf -ngl 99 -sm tensor -ts 0.5,0.5
```

### MTP speculation (production default)

Padding is on by default; no env needed. To measure its effect, A-B with the
kill-switch and the timing log:

```sh
-e LLAMA_SPEC_DRAFT_NO_PAD=1 -e LLAMA_SPEC_TIMING=1   # baseline (padding off)
```

### Reproduce a 2-GPU tensor-split on one physical GPU (validation)

```sh
-e CUDA_VISIBLE_DEVICES=0 -e LLAMA_META_DUP_DEVICE=2 \
-e GGML_META_DEBUG_REDUCE=1  ... -sm tensor
```

### Diagnose a numeric divergence (full-tensor dumps)

```sh
-e LLAMA_DEBUG_DUMP_DIR=/tmp/dump -e LLAMA_DEBUG_DUMP_FILTER=ffn_gate \
-e LLAMA_DECODE_TIMING=1
```

### Distributed inference (coordinator + RPC workers)

Worker exposes its GPUs as one tensor-parallel island; the coordinator pushes
split states. Direct worker-to-worker pull is on by default; disable to compare:

```sh
# worker
ggml-rpc-server --tensor-parallel -p 50060
# coordinator
CUDA_VISIBLE_DEVICES="" llama-server -m model.gguf --rpc host:50060 -ngl 99
# (optional) force bridged copies instead of W2W pull:
-e GGML_RPC_NO_W2W=1
```

See `docs/distributed-inference-guide.md` for the full topology and security notes.
