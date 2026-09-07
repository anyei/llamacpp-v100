# MoE expert cache (VRAM cache for CPU-resident experts) - investigation + proposal (TASKS #151)

Status: **PROPOSAL 2026-09-07, awaiting the user's pick** (section 7 decision points). No code
written. Companion lanes: docs/ssd-streaming-plan.md (Task 15, the in-tree SSD tier + its
single-GPU slot cache), docs/expert-profiling.md (#74 profiles = warm-start source),
docs/expert-placement-plan.md (#75 skip sentinel + remap tables, reused here).

## 0. TL;DR

- "MoE cache" = keep the routed experts in host RAM (what `-ncmoe` does today), and keep the
  HOT experts additionally resident in spare VRAM so the GPU computes them, while the CPU keeps
  computing the misses. Three forks have built this in 2026; the two that work share one law:
  **a miss is never fetched on the critical path.** Every design that fetched misses
  synchronously (including our own Task-15 GPU landing) lost or broke even on big models.
- The tree ALREADY has a slot cache (Task 15, `LLAMA_SSD_STREAM_GPU`), but it is the wrong
  shape for RAM-resident experts: it fills misses synchronously inside the scheduler's copy
  phase, mutates node sources per token, is single-GPU only, and requires the SSD arena buft.
- The forks: **leloch/llama.cpp** (EC3, upstream PR #24524, closed under the AI policy) and
  **TheTom/llama-cpp-turboquant** (`--moe-cache`, by giveen, PRs #284/#288, evolved from EC3)
  are the in-kernel hybrid: the CPU `mul_mat_id` op itself ships hit rows to the GPU and
  computes miss rows on the CPU threads. **markldn/llama.cpp-qwen4exp-lru-async** (and the OPEN
  upstream PR #27861 by csantiago78 it ports) is the graph-level dual chain: a second, GPU-side
  mul_mat_id chain over slot tensors, an id->slot table remapped with `get_rows`, the CPU chain
  skipping cached ids, both down outputs summed; fills by a background thread, tables published
  between decodes. The user's turboquant run on the X99 (2026-09-07) is TheTom's build.
- **Physics on the X99 decides the shape**: V100 on PCIe Gen3 x16 (~11 GB/s pinned, ~5 pageable)
  vs 100+ GB/s host RAM on the dual E5-2690 v4. A miss byte over PCIe costs ~10x a miss byte
  read by the CPU. Hence never-stall + async fills + pinned source memory are mandatory, and the
  win is bounded by how much of the token the CPU expert path is.
- Expected gains (from the forks' measurements, scaled by our expert sizes): **Flash-Next
  (0.9-1.7 MB experts, fast host) +5-25% decode at 1 GPU** (small experts are the forks' known
  weak case; markldn got +24% on a 6-core box); **DeepSeek-V4-Flash / GLM-5.2 / Ling (multi-MB
  experts) +20-50%**, more with 2-3 GPUs of cache. Numbers must be measured here first.
- **Recommendation**: build the graph-level dual chain (markldn / PR #27861 shape) natively in
  this fork (~1.2-1.5k lines, CUDA-graph and decode-graph-cache safe, no cross-backend calls),
  with the policy lessons from EC3/giveen (decode-only fill, admission throttle at capacity,
  bounded async fills, heat-aware eviction) and one fork-specific addition: **profile-seeded
  warm start** from the #74 artifacts. Phase 0 = a zero-code A/B on the X99 using TheTom's
  already-built binary, which the user can run today (section 8).

## 1. Why now

The user built `local/llama.cpp:turboquant` (TheTom fork, `feature/turboquant-kv-cache` @
407f3237b, 2026-09-06) on the X99 and launched Flash-Next with `--moe-cache 18000 --fit on -ngl
auto --no-mmap --spec-draft-model mtp-...` (compose at
/home/anyei/server/services/llama-cpp-turboquant). The container died in a restart loop and
was stopped (exit 137 = docker stop). From its log, two problems, neither of them the cache
mechanism itself (2 and 3 are one causal chain):

1. `llama_model_load: error loading model: check_tensor_dims: tensor 'blk.48.nextn.eh_proj.weight'
   not found` - TheTom's qwen4exp MTP expects a different head export than our #143/#149 head
   file (ours carries fc_embd/fc_hid). Fatal; the serve never started.
2. The cache was dormant before that: `[moe-cache] configured: ... min-expert=1024 KiB` is the
   pre-Ampere admission floor (their `on`/`N` modes on compute capability 7.0). Flash-Next's
   gate/up experts are 921,600 B = 900 KiB < 1 MiB, so their cache-aware fit counted that shape
   as permanently uncacheable (`common_moe_cache_plan_fit: ... some routed expert weights would
   remain permanently uncached`, fit.cpp:141-145 = supported_bytes != expert_bytes) and
   `kept stock placement`: `-ngl auto` filled the card with whole layers, leaving
   `free=1532 MiB` < the 3072 MiB reserve, hence `CUDA0 capacity: ... granted=0 MiB`. In forced
   mode the dormant state is silent (no `[moe-cache] enabled` line is the only evidence).
   Fix: `GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB=512` (their Ampere default; EC3 measured 512 KiB
   experts still profitable) - and/or force placement with `-ngl 99 -ncmoe 48`.
3. `--moe-cache auto` would not have helped either: auto needs compute capability 8.0; on the
   V100 only `on` or a fixed `N` MiB engages.

Section 8 gives the corrected command. The larger question the user asked - what architecture
actually works, and what should this fork build - is the rest of this document.

## 2. What the tree already has (and why it is not the answer)

| Piece | Where | Reuse |
|---|---|---|
| Task 15 streamed buft + LRU/SLRU RAM cache + **single-GPU VRAM slot cache** (`gpu_bind`: touch pool, H2D misses synchronously, remap ids into a GPU scratch, alias `input_cpy` to the pool, swap `node->src[2]`) | `ggml/src/ggml-ssd-stream.cpp:1050-1237`, hook in `ggml-backend.cpp:1636-1730` | Policy engine (`gpu_slot_pool`, offline-tested), pool sizing by slice class, MMQ pad-slot trap, recycled-node trap. NOT the fill path. |
| Skip sentinel: negative id = lane uses no expert, dst rows pre-zeroed, on CPU and every CUDA mmid path | `ggml.h:1449`, `ggml-cuda.cu:1925-2049`, `mmvq.cu:555,786`, `ggml-cpu.c` gather loop | The GPU chain marks misses with the sentinel instead of a dummy zero slot (markldn needs the dummy slot; we do not). |
| #75 remap tables via `ggml_get_rows` on I32 per-layer tensors inside `build_moe_ffn` | `src/llama-graph.cpp:1946-2036` | Same op shape as the id->slot remap. |
| #74 profiler + artifacts (`profiles/flash-next-q4-2026-09-06*.json`, hy3, V4) | `src/llama-context.cpp:30-193`, `docs/expert-profiling.md` | Warm start: seed the per-layer fill queue with the top-K of the merged profile. |
| Decode-graph cache (one sched per graph shape, replayed) | `src/llama-context.cpp:1339-1430` | Static graph is a hard requirement -> favours the dual-chain design. |
| CUDA host pinning API | `ggml-cuda.cu:4691` `ggml_backend_cuda_register_host_buffer` (cudaHostRegister Portable+ReadOnly) | Pin the offloaded expert tensors in place at load. Unused by our loader today. |
| Scheduler decode/prefill asymmetry: `MUL_MAT_ID` offloads to CUDA only at >= 32 tokens (`GGML_OP_OFFLOAD_MIN_BATCH`), copying used experts per batch; below that the whole op runs on the CPU with activations D2H / results H2D | `ggml-backend.cpp:943-963`, `ggml-cuda.cu:5383-5394` | The cache is a decode feature; prefill keeps the stock offload (faster from pinned memory: markldn +41% pp). |

Why the Task-15 landing is not it: (a) miss = synchronous `tensor_set_async` + `ggml_backend_synchronize`
per node, i.e. PCIe on the critical path (its own numbers: DeepSeek-81GB 1.5-2.5 t/s vs CPU 2.5 -
break-even at 64% hit; Qwen-35B 2.5 -> 7 t/s only because the hot set fit); (b) it rewrites
`node->src[2]` and `input_cpy` every token (incompatible with CUDA graphs, fragile under the
decode-graph cache - the recycled-node crash was exactly that); (c) single GPU by construction;
(d) it only engages for the streamed buft, not plain `-ncmoe` tensors. Its right role: the SSD
tier for models that do not fit RAM. The new cache targets RAM-resident experts.

Two documented traps that carry over: `LLAMA_SSD_STREAM_PREFETCH` (previous-token router
lookahead) was measured useless - LRU already holds the last token's experts; eval-callback
instruments corrupt meta-fleet serves (build on the graph/scheduler, never on `cb_eval`).

## 3. The forks

### 3.1 leloch/llama.cpp - "EC3" (branches moe-cache, moe-cache-v2-pr, v3-expert-cache; upstream PR #24524, 2026-06-12, closed the same day; RFC discussion #24528 still open, no maintainer engagement)

In-kernel hybrid. Inside the CPU `ggml_compute_forward_mul_mat_id`, thread 0 calls a
CUDA-registered API table (`ggml_expert_cache_v3_api`: begin/plan/dispatch/collect/end, plus
redirect_offer/finalize, glu_hits, invalidate, node_time, router_bias): `plan` marks hit rows
and enqueues async inserts for misses (worker threads, pinned staging, own streams,
`LLAMA_EC3_INSERTS` 8 per node visit, `LLAMA_EC3_THROTTLE` admit 1-in-8 at capacity), `dispatch`
issues ONE batched mmvq over the hit rows on the GPU while the other CPU threads compute the
miss rows, `collect` syncs and scatters the GPU rows into dst. Exact-stride per-(expert size,
type) slot pools; plain LRU. Decode-only fill (prompt-driven fills measurably polluted v2).
Extras: fused gate+up+SwiGLU dispatch, down-dst GPU-resident handoff, hot-set persistence,
baseline-sampled bail-out, multi-GPU striping by layer. The `moe-cache-v2-pr` branch (2026-08-06,
29 commits) is the cleaned v2 that giveen ported; v2 DROPPED the dst handoff, the idle-worker
prefetch backfill, the hot-set persistence and a cache-aware routing bias (all measured not to
pay) - the research tree `v3-expert-cache` still carries them.

Measured (4x RTX 3090, EPYC 7R13 8ch DDR4): GLM-5.1 754B IQ2_M 14.0 -> 19.2 t/s (+34%),
Qwen3.5-397B +18%, ERNIE-21B +44%, Llama-4-Scout +44%, OLMoE (tiny experts) +1-13%; 16/16
positive or parity in the forced `-ncmoe 99` regime; auto vs best static placement +7..+25%.
PPL parity; greedy near-tie flips. Lessons recorded in their own docs (EC3_READINESS,
EC3_OPTIMIZATION_ROADMAP) that we adopt:

- Sub-MB experts lose more to dispatch overhead than the GPU saves -> a per-tensor size floor
  and an economics rule (spill ratio >= 1.8 AND (>= 2 devices OR experts >= 2 MiB)).
- "Every per-node latency lever landed neutral at ~75% hit"; the binding floor is GPU chain
  exec + CPU miss work. Eviction-policy zoo (LFU, TinyLFU, SLRU, S3-FIFO) = +-3%. Capacity and
  ADMISSION are what matter. Loosening admission below 1-in-8 lost 1.7-2.4 t/s with no hit gain.
- Cross-layer router prefetch is structurally impossible (ids(L+1) depends on moe_out(L)); the
  measured hit rate already equals the destructive-mask ceiling. Only a learned predictor could
  beat temporal locality - none did.
- CUDA-graph capture of the dispatch chain was a net loss (exec-bound). Bigger total budget can
  LOSE (VRAM pressure ~10 ms/token hidden cost) - keep a reserve.
- Cold window is net-negative (tg100 -7%); fixed with idle-worker backfill + prompt-phase
  discovery. Fit's 1 GiB margin starves a 3 GiB reserve -> the cache must be fit-aware.
- Five blockers found by their audit (fused-GLU stale state, gate-defer corruption on models
  with readers between the MMIDs, F16 expert abort, process-global singleton across models,
  CUDA_CHECK aborts under self-made OOM) - all things a fresh design should avoid by
  construction: no cross-node learned state, type allowlist, per-context state, checked errors.

### 3.2 TheTom/llama-cpp-turboquant `--moe-cache` (giveen: PR #284 roll-up split into #288 "Pr/cuda moe cache" merged 2026-08-12; #300 heat-protected eviction merged 08-18; #327 adaptive cpu-overlap calibration merged 08-31; #291 prefetch NOT in the tree; #364 open = TQ-only pinned-host expert streaming with a device pointer table)

Same in-kernel hybrid lineage, productised: provider interface `ggml_moe_cache_api` (CUDA/HIP,
Metal, Vulkan providers; `moe-cache.cu` 3391 lines + 725-line common header + hooks in
ggml-backend.cpp, ggml-cpu.c, llama-context.cpp, fit.cpp), one session per scheduler, per-device
budget claimed once, pools per shape after a shape census, layer->device stable assignment,
demand fill only (never synchronous), admit after 1st miss (pool holds everything) or 2nd miss
(capacity-constrained), <= 8 fills per node, device queue 128 jobs / 512 MiB, one low-priority
fill stream per device, **heat-aware eviction** (LRU victim with resident hit count <= 4),
replacement throttle 8 fresh misses, max 8 tokens per node (so MTP/DSpark verify batches stay
eligible), fused gate/up/SwiGLU when both rows are resident, invalidation on host-buffer
writes, allocator trim hook, `-lv 4` statistics. Dropped from EC3: redirect, backfill,
persistence. Volta: `on`/`N` modes only (cc >= 7.0), 1 MiB expert floor.

Verified in the code (agent read of moe-cache.cu, ggml-cpu.c, fit.cpp at 407f323): the CPU
`mul_mat_id` stays on the CPU backend at decode (op-offload untouched, still >= 32 tokens);
thread 0 calls begin/plan, launches ONE async H2D of `[slot ids | act rows]` from a pinned
scratch + quantize + mmvq on a per-device non-blocking stream, all threads compute the miss
rows through the stock chunked path, then thread 0 `collect`s (D2H + `cudaStreamSynchronize` +
memcpy into dst rows). A miss NEVER yields a slot for the current token - even the path that
enqueues a fill returns -1. Fills: one worker thread per device, `memcpy` source -> per-worker
pinned staging buffer -> `cudaMemcpyAsync` on a lowest-priority stream + sync; no
`cudaHostRegister` of the weights anywhere (page faults on mmap'd sources are absorbed on the
worker). No events. No CUDA-graph interaction (cached nodes live in CPU splits). Eligibility:
<= 8 tokens AND <= 64 flattened routed rows per node (top-10 x 8 tokens = 80 rows would bypass;
top-10 x 6 = 60 fits). "cpu-overlap" is the opposite of what the name suggests: when every row
is a hit it hands some hit rows BACK to the CPU so both engines work, with an online
calibration over {formula,0,1,2,4,8} rows. Pools are per (device, expert size, type) slot slabs
aggregated across same-shape tensors, layer->device assignment sticky and capacity-weighted;
budget = min(N, free - reserve) computed lazily from `cudaMemGetInfo` at first eligible node
and latched (never re-probed). Eviction: intrusive LRU with an LFU veto (`uses <= hot` wins,
else LRU head), no aging, no per-layer reservation. Stats count residency probes, not bytes.
Dead/drifted bits found: `query_fused` never registered; header comment names dispatch tables
that do not exist; NVFP4 silently excluded from fusion; PR #291 prefetch is NOT in the tree.

Measured (4x RTX 3090, 48-core host): DeepSeek-V4-Flash Q8_K_XL canonical CPU experts 19.05 ->
22.54 (1 GPU) / 26.21 (2) / 27.47 (3) / 27.51 t/s (4); with DSpark 51.6 -> 70 t/s; GLM-5.2
IQ2_M + MTP 14.5 -> 28.3 t/s; Llama-4-Scout +81%; Qwen3-30B +2%, Qwen3.6-35B +8% (small experts
again); dormant when the model fits (parity). PPL equal within error. Their doc's rule: "Do
not infer a gain from hit rate alone: activation transfers, result transfers, fill traffic, GPU
speed, PCIe speed, and CPU memory bandwidth all affect the result." Closest data point to us:
their PR #340 thread runs **qwen4exp UD-Q4_K_XL with `--moe-cache auto` + MTP: 32.3 t/s baseline
-> 33.7 cold -> 53.0 warm, steady hit 83.4% after one ~273-token request, and a new topic drops
to 31.0** (Ampere-or-newer card, host unknown) - the warm/topic-switch shape we should expect.
Community results on Volta-class or older cards are the caution: GTX 1080 Ti measured **-31%**
(no spare VRAM, weak mmvq), and a matched-VRAM test showed resident layers beating the cache
1.8-2.3x when the budget already covers the working set - the cache pays only when VRAM budget
< expert working set.

### 3.3 markldn/llama.cpp-qwen4exp-lru-async (2026-09-03..05) and upstream PR #27861 (csantiago78, OPEN: "llama: GPU-resident LRU cache for host-offloaded MoE expert weights")

Graph-level dual chain, on OUR model (Qwen3.8-Flash-Next / qwen4exp, with MTP):

- Per host-resident layer, companion slot tensors `up_c/gate_c/down_c` of shape
  `[ne0, ne1, n_slots+1]` allocated in the buffer of that layer's router (`ffn_gate_inp`) - so
  multi-GPU placement under `-sm layer` falls out for free. Slot `n_slots` is all zeros.
- An I32 `table[n_expert]` (expert -> slot, or n_slots when uncached), one copy on device
  (read by `ggml_get_rows` to remap `selected_experts` for the GPU chain), one on host (read by
  the CPU `mul_mat_id` via `dst->src[3]` to SKIP cached ids, zeroing their rows).
- Graph: the stock CPU chain (gate/up/act/down) stays as is; a second GPU chain
  `up_g/gate_g -> swiglu -> down_g` over the slot tensors with the remapped ids runs on the
  device; `experts = add(down_cpu, down_g)`. Each expert is computed on exactly one chain, so
  the sum is exact. Enabled only for n_tokens <= 5 (validated with MTP n-max <= 4), SILU MoE,
  no expert biases/scales/LoRA.
- A miss is never fetched inline. `llama_moe_cache_step()` runs once per `llama_decode` after
  the graph finished: publish completed uploads (table update), evict LRU victims (clear their
  table entry now), hand slice copies to ONE background worker thread (throttled: 2 inserts
  per layer per step by default). A running graph can never observe a torn slot.
- All `-ncmoe` expert tensors are page-locked in place at load (cudaHostRegister/hipHostRegister,
  no copy): the sched's existing prefill offload copies became direct DMA: **+41% prefill**
  (253 -> 357 t/s, ~3600-token prompt) for a one-time +3.9 s at load, cache on or off.
- First version fetched misses in-graph and lost (35% of GPU time in copies); the rewrite is
  the never-stall one. Their earlier ggml-backend-based attempt (host readback of ids + host
  bitset every step) regressed - hence the device-side LRU kernel `moe-lru.cu` (201 lines).
- Upstream PR #27861 itself: draft since 2026-08-28, 12 files, +645 lines, no new kernels, no
  maintainer reply; Flash-Next UD-Q4_K_XL on 2x3090 18.4 -> 24.2 t/s (+31%) with 48 slots per
  layer (~4.1 GiB); its routing study found NO static skew (a top-32 hot list covers ~10% of
  held-out traffic) but strong temporal locality (LRU-64 ~67-81%, LRU-128 ~81-90%, 384 slots
  98.5%) - the same shape as our #148 cross-domain 0.354 result. Problems testers found, each
  a design input for us: (1) its `n_tokens == 1` guard excludes every MTP/spec verify batch
  (patches that repeat the table make 2-4 tokens byte-identical; >= 16 corrupts); (2)
  **duplicate dummy-slot ids break the CUDA batched mul_mat_id kernels (mmid/mmf/mmq) above the
  mmvq window of 8 tokens** - exactly the bug class our #75 skip sentinel was built to remove;
  (3) decode vs prefill graph topology differs -> repeated reallocs / VRAM shrink after big
  prompts (our decode-graph cache keeps one sched per shape, which contains this); (4)
  per-expert-scale wrappers (Gemma4 `down_exps_s`) need the skip on the inner mmid; (5) an
  upload backlog can starve new experts; (6) a rocprof trace showed 3x more stream syncs and
  ~10x more H2D ops than MTP-only, and dummy-slot rows still burn GPU kernels; (7) `--fit`
  ignores the cache. A sibling RFC (memoriaru #28248, ~700 lines) keeps ONE chain and fills
  synchronously in the split prologue: +84% on an RTX 4090 (PCIe Gen4) for Flash-Next Q3_K_XL
  with all-CPU experts - a strong-GPU/fast-PCIe result that does not transfer to Gen3 + a fast
  host (section 4), but it shows how much the hit path alone is worth on that class of box.

Measured (R9700 32 GB + RX 9070 16 GB on PCIe 3.0 x4, 6-core CPU, `-ncmoe 40`, 96 slots/layer):
decode 12.2 -> 15.1 t/s (+24%), prefill 207 -> 254 (+23%); with MTP + kernel work 21.2 t/s.
Greedy temp-0 output bit-identical vs `-ncmoe` on low-entropy prompts, with and without MTP.
Code size: 647 + 121 (cache) + 201 (LRU kernel) + graph/CPU/arg hooks.

### 3.4 Others seen (survey agent, 2026-09-07; details in its report, kept out of the tree)

- Same lineage: mclaudod `moe-hot-cache` (leloch v2 + direct-resident hits; RTX 3060 Qwen3.6-35B
  37 -> 65.7 -> 68.7 t/s; rejected seven hit-rate policies that did not move t/s; multi-GPU
  broken), sarashin65 (SYCL provider + STATIC domain hot sets; dynamic LFU/LRU raised hit
  65.6 -> 86.7% but promotion H2D competed with compute, shipped static only; domain Jaccard
  0.27-0.29 code x prose), simlu mirror.
- Upstream attempts, all closed or stalled: #17044 (new GGUF layout), #20757 issue (the request;
  Python PoC 12-14 t/s at 98% hit on an 8 GB card) + #21609/#21614 (scheduler input_cpy as an
  N-slot LFRU pool + FATE temporal prefetch; found MMQ over-reads past the last expert row ->
  quantized types need a pad slot), #23170 (copy only missing expert ranges - NaN from gallocr
  reuse, no-op when correct; post-mortem asks for a persistent slot buffer with explicit
  bookkeeping), #26563/#26824 miltos22 `-ehs` heat map + hot store (PP collapsed to CPU, closed
  for scope), #25294 freedomljc disk streaming with a custom remap op (GB10 GLM-5.2 decode
  2.4x), #23440 Metal disk pool, #21067 am17an `--prefetch-weights` (maintainer PoC: whole
  tensors double-buffered on a copy stream, pays only at ubatch >= 1024), #28414 prefill
  prefetch, #26003 `--lazy-experts` (closed 2026-09-07 "not worth it"), #28545 mlock RFC.
- Other shapes: ap03906101/moe-hotcache and TheTom #364 = a UVA pointer table inside mmvq
  (pinned experts resolve to VRAM slots, all others are read directly from pinned host over
  PCIe; retracted +67% after MMQ read raw host pointers, honest +15%) - misses over PCIe at
  Gen3 speed lose to our CPU (section 4); Lidenburg LFU-with-aging all-on-GPU variant (the
  only branch that works with `-sm tensor` on V4); JigSawPT static hot lists (+26%, -10% PP,
  measured async CPU/GPU overlap at scheduler granularity as a net loss on WDDM); ongunm FATE
  cross-layer prediction (99.5% hit on Qwen3-30B but prompt eval collapsed to 4 t/s);
  yalun753 cuBLAS-over-pinned-host (`cudaMemcpyAsync` fails across multiple
  `cudaHostRegister` ranges - register per tensor, never copy across a tensor boundary).
- Outside llama.cpp: ik_llama.cpp has no dynamic cache (ikawrakow: "copying MoE tensors to
  the GPU is basically never a good idea" for decode; its `--prefetch-experts` is a page-cache
  prefetcher for prefill), KTransformers static frequency placement + dynamic re-selection
  during long prefills + expert deferral, Fiddler (the copy-weights-vs-ship-activations rule),
  MoE-Infinity, mixtral-offloading, vLLM RFC #38256 (persistent in-place expert->slot int32 map
  = CUDA-graph friendly, LFRU, prefill dedup), papers ProMoE/HOBBIT/FATE (learned or gate-
  reuse predictors, the only prefetch that beats LRU) and arXiv 2608.07911 (replay evaluation
  inflates recency policies 27-29%).
- Upstream master (b10749, 2026-09-07) contains none of them; #27861 and #28248 are the live
  ones to watch at each weekly merge (#67). Note: upstream commit 4310aa4f87 "contrib: allow all
  AI-generated code in general (#26012)" (2026-07) may have relaxed the policy that closed
  #24524 - verify before assuming this fork's AGENTS.md copy is current upstream policy.

## 4. Physics on our hardware

Facts (verified 2026-09-07): X99 = dual E5-2690 v4 (56 threads), 251 GB RAM (scored 109 GB/s),
currently ONE V100-SXM2 32 GB visible at 03:00.0 on PCIe Gen3 x16 (the carrier with the other
two cards is off the bus again). Coordinator box now carries 2x K80 (not a serving target).

Flash-Next UD-Q4_K_XL geometry (GGUF headers read on the X99): 48 MoE layers x 512 experts,
top-10, n_embd 2560, n_ff_exp 640. Per expert per layer: gate Q4_K 0.92 MB + up Q4_K 0.92 MB +
down Q5_1 1.23 MB (43 layers) or Q8_0 1.74 MB (5 layers) = **3.07-3.58 MB per expert**;
**71.7 GiB of routed experts** (17.38 + 43.6 + 10.74 GiB across shards 2-4); shared experts
0.23 GiB; PLE table 28.8 GB IQ4_NL; trunk ~9 GB. Decode reads **~1.5 GB of expert weights per
token** (10 x 48 x ~3.1 MB) when everything misses.

| Path | Cost of 1.5 GB/token | Notes |
|---|---|---|
| CPU reads experts from RAM (today, `-ncmoe 48`) | ~15 ms bandwidth + vec_dot compute + 48 split boundaries | measured whole-token 80 ms (12.5 t/s, spec off, -t 24); where the other ~55-65 ms go is unmeasured (trunk kernels, PLE gather, per-split sync) - phase 0 item |
| VRAM hits | 1.5 GB at ~800 GB/s = ~2 ms + ~6 launches/layer | the prize |
| PCIe misses, pinned | 1.5 GB at ~11 GB/s = 136 ms | why sync fill is dead on arrival: at 70% hit still 41 ms |
| PCIe misses, pageable (default mmap) | ~2x worse | why pinning is not optional |

Consequences:

1. Hits pay ~10x less than the CPU path per byte; misses over PCIe pay ~10x MORE. A design is
   only ever "not worse than `-ncmoe`" if misses stay on the CPU and fills are asynchronous,
   rate-limited, and from pinned memory. Both working forks converge on exactly this.
2. The ceiling is the CPU expert share of the token. For Flash-Next on the X99 that share is
   maybe 20-30% (fast host, small experts), so a perfect cache yields at most ~+25-40% and a
   realistic 70-85% hit rate gives +10-25% - consistent with markldn's +24% on a much weaker
   host. For V4-Flash-0731 / GLM-5.2 / Ling the expert bytes per token are 2-4x larger and the
   share is 50%+, hence the forks' +30-50%.
3. Capacity: the cache pays only while VRAM budget < the expert working set; at equal bytes,
   resident layers beat a cache 1.8-2.3x (matched-VRAM test in the #24528 thread), so `auto`
   must compare against the best feasible static placement, never assume. With ~20 GB spare on
   one V100 the cache holds ~28% of Flash-Next's experts. The
   merged profile puts the static top-25.5% at 58% coverage; an LRU tracking the live domain
   does better (Task 15 measured 39% budget -> 73% hit on DeepSeek; EC3 saw 75-80% at ~30%).
   Three V100s would hold nearly all 71.7 GiB - at that point the honest comparison is a
   static 3-GPU layer split, not a cache.
4. The extra GPU chain is launch-bound at batch 1 (~6 small kernels per layer, ~0.05-0.1 ms
   without graphs -> 2-5 ms/token over 48 layers). CUDA graphs per split (the fork's
   FORCE_GRAPHS era knob) or the decode-graph cache take most of that back. The in-kernel
   design overlaps the GPU chain with the CPU misses (max instead of sum); on the X99 that is
   worth ~2-5 ms/token on Flash-Next - a v2 lever, not a v1 blocker.

## 5. Proposed architecture: dual chain, table-published, never-stall

Name: `--moe-cache` (CLI) / `LLAMA_MOE_CACHE_*` (env), lane doc = this file.

Data structures (per llama_context, NOT process-global - EC3 blocker B4):

- `moe_cache_layer{il, n_slots, up_src/gate_src/down_src (host tensors), up_c/gate_c/down_c
  (device slot tensors [ne0, ne1, n_slots] in the router's buffer), dev_table/host_table (I32
  [n_expert], -1 = not resident = our skip sentinel), slot_expert[], slot_last_use[],
  slot_hits[], slot_in_flight[], pending_misses[]}`. Slot tensors are allocated OUTSIDE gallocr,
  after `sched_reserve` (so KV + compute buffers come first and the budget is free VRAM minus a
  reserve, like both forks: reserve default 3 GiB, knob).
- Pools per layer (markldn) rather than per shape class (Task 15 / EC3): fixes the pool-ordering
  and role-starvation bugs by construction and gives a per-layer budget knob; the cost is the
  inability to shift capacity between layers (v2: profile-weighted per-layer budgets).

Graph (in `build_moe_ffn`, only when `n_tokens <= moe_cache_max_batch` (default 8, covers MTP
n-max 3 + ngram drafts), the layer is host-resident, arch path is plain SILU/SwiGLU with no
expert bias/scale/LoRA, no `gate_up_exps` fusion, no placement remap active):

```
ids        = ffn_moe_topk                                  (GPU, as today)
slot_ids   = get_rows(dev_table, ids)                      (GPU; -1 for misses)
GPU chain  : up_g = mmid(up_c, cur, slot_ids); gate_g = mmid(gate_c, cur, slot_ids)
             act_g = swiglu(gate_g, up_g);  down_g = mmid(down_c, act_g, slot_ids)
CPU chain  : unchanged (gate/up/act/down on the host tensors), but the down node (and the
             gate/up nodes) carry src[3] = host_table -> the CPU op skips ids whose table
             entry is >= 0 (rows stay zero)
experts    = add(down_cpu, down_g)                         (GPU)
```

Sentinel lanes cost nothing on the GPU (our mmid paths skip negative ids and pre-zero dst);
markldn's dummy zero slot is not needed - and the dummy slot is what breaks the CUDA batched
mmid/mmf/mmq kernels above 8 tokens in #27861 (duplicate ids), the exact bug class #75 removed. The split structure is identical to `-ncmoe` today
(one CPU split per MoE layer); the GPU chain rides in the surrounding GPU splits. Static graph:
only buffer CONTENTS (tables, slots) change between decodes, so the decode-graph cache and
CUDA graphs are untouched.

Per-decode step (`llama_context::decode`, after the graph completed - a point where no graph
references the tables): (1) publish completed uploads (write both tables), (2) for each layer,
pop the miss list recorded by the CPU op this step (ids seen with table < 0), apply admission,
pick victims (never an in-flight slot; never a slot published this step), clear their table
entry, enqueue `{layer, expert, slot}` on the fill queue, (3) one `tensor_set_async` per dirty
table + a sync on the cache backend. Fill worker: its own `ggml_backend_t` per device + own
stream; `tensor_set_async(slot_tensor, host_expert_ptr + e*nb[2], slot*nb[2], nb[2])` from
PINNED memory (true DMA), event/sync, then mark done. Bounded: `inserts_per_layer_per_step`
(default 2 like markldn; EC3 8 per node), queue cap in jobs and bytes, and a backlog rule: a
queued fill older than N steps is dropped so a burst cannot starve fresh misses (#27861 tester
finding).

Policy (v1, deliberately boring - the forks measured the zoo at +-3%):
- Admission: first miss when the pool is not full; at capacity, admit 1-in-N misses (N = 8,
  EC3/giveen) and prefer experts missed >= 2 times (giveen). Decode-only: ubatches above the
  batch cap never touch the tables or the miss lists (EC3's "prompt fill pollutes" result).
- Eviction: LRU with heat protection (skip victims with resident hit count > 4 unless all are
  hot). Slots hit this step are pinned for the step.
- **Warm start (fork-specific)**: at context init, if a #74 profile artifact is given
  (`--moe-cache-profile path.json` / `LLAMA_MOE_CACHE_PROFILE`), pre-fill each layer's queue
  with the top-K experts of the merged histogram (K = n_slots), streamed by the worker during
  the first prompts. Closes EC3's net-negative cold window without their idle-backfill
  machinery, and turns the #148 artifacts (Cov@25.5% 0.58 merged, but domain-sensitive) into a
  prior that the LRU corrects online. Later: dump the resident set at shutdown and reload it.
- Pinning: with `--no-mmap` (or `--load-mode none`) the loader already places CPU-overridden
  experts in the CUDA host buffer type = `cudaMallocHost` = pinned, no work needed (the user's
  turboquant compose and several launcher configs run this way). With mmap, `cudaHostRegister`
  every host-resident `*_exps` tensor range in place (Portable | ReadOnly; per tensor, never a
  copy across a range), independent of the cache (markldn: +41% prefill from the sched's own
  offload path; #25859: +21% PP). `--no-host` users lose nothing. Fallback if in-place
  registration misbehaves on a host: giveen's memcpy-to-pinned-staging on the worker (the
  cache still never stalls; only the sched's prefill path loses the DMA win). Knob to disable
  (`LLAMA_MOE_PIN_EXPERTS=0`); log the pinned GiB and seconds. Note: CUDA pinning does not
  count against RLIMIT_MEMLOCK (driver uses get_user_pages), so containers need no ulimit change.
- Multi-GPU: per-layer slot tensors follow the layer's router device (`-sm layer`), one worker
  backend per device, budgets per device. `-sm tensor` / meta backend: cache disabled (that
  combination is already broken for `-ncmoe`, common/arg.cpp:842, TASKS #82).
- Repack bufts: the cache requires the canonical CPU layout (`ggml_backend_cpu_buffer_type` or
  the CUDA host buft); with AMX/aarch64 repack bufts active the layer is skipped and logged.
- Safety: all CUDA calls in the worker checked -> on error, disable the layer (table all -1,
  slots released), never abort; the miss path is always the stock CPU path, so "disabled" ==
  `-ncmoe`. Model unload joins the worker and drains the queue before buffers are freed.
- Instruments: per-layer hit/miss/fill/evict counters, bytes uploaded, worker queue depth,
  per-token hit rate line every N steps (`LLAMA_MOE_CACHE_STATS=N`), and a one-line summary
  at teardown; wizard/fleet card shows "cache: X/Y GiB, hit Z%".

Eligibility gates (auto-off with a logged reason): experts smaller than 512 KiB per tensor
(EC3/giveen floor; Flash-Next gate/up = 900 KiB passes), fewer than 32 slots per layer, no
host-resident expert layer, unsupported quant for our mmid (allowlist = the types our mmvq
handles), unsupported MoE graph variant.

## 6. Alternatives considered

| Option | Shape | Size | Pros | Cons |
|---|---|---|---|---|
| A. Port TheTom/giveen `--moe-cache` | in-kernel hybrid, provider table | ~10k lines (CUDA subset ~5-6k) + hooks in ggml-cpu.c, ggml-backend.cpp, llama-context.cpp, fit.cpp | most validated (4-GPU V4/GLM/DSpark), fused gate/up, fit-aware, stats, tests | intrusive in the two files where this fork diverges most (meta backend, defer, ssd-stream, skip sentinel); Volta second-class (1 MiB floor, no auto); their fit/repack/moe-cache coupling drags in `fit.cpp` changes; giveen's tree is 3 days behind upstream, ours 560 commits - every hunk hand-woven |
| B. Port leloch EC3 (PR #24524 form) | in-kernel hybrid, CUDA-only | +2.2k lines / 11 files | smallest in-kernel port, same core idea | the readiness audit lists 5 blockers in that form (some fixed later in v3, which is bigger); process-global state; 20 env knobs; fused/defer learned-state hazards |
| C. Extend Task-15 GPU landing | sync fill in the sched copy phase | rewrite of its core anyway | policy engine reusable | wrong fill model (critical-path PCIe), per-token src mutation, single GPU, SSD-buft-only - every trait we need to remove |
| **D. Build the dual chain natively (recommended)** | graph-level, table-published, async worker | ~1.2-1.5k lines: `src/llama-moe-cache.{h,cpp}` (~700), `build_moe_ffn` hook (~80), CPU op skip via src[3] (~20 in the gather loop we already modified for the sentinel), loader pinning (~60), arg/env/stats (~100), unit test | static graph (decode-graph cache, CUDA graphs), no CPU->CUDA calls, uses stock ops + our sentinel, multi-GPU by placement, validated on qwen4exp by markldn (bit-identical greedy, +24%), an OPEN upstream PR of the same shape to converge with (#27861) | no CPU/GPU overlap inside a layer (2-5 ms/token on Flash-Next, section 4.4); no fused gate/up on the GPU chain (later: our #140 fused mmid can apply); the per-layer pool cannot shift capacity between layers (profile-weighted budgets = v2); the #27861 tester list (section 3.3) must be closed by design: sentinel not dummy slot, batch cap 8 = the mmvq window, decode-graph cache for shape stability, backlog rule, fit awareness |

Recommendation = D, with A's policy defaults and EC3's negative results as guard rails, and a
zero-code phase 0 that uses A's binary as the oracle for "what is the prize on this box".

## 7. Increments and gates (each its own commit + gate, per docs/dev-workflow)

0. **Measure before building (X99, no code)**. (a) Our image: `-ncmoe 48` vs `-ncmoe 40` vs
   `-ncmoe 44` at fixed `-t 24`, plus `-t 12/24/44` at `-ncmoe 48`, llama-bench tg 128 x 3 -
   gives the CPU expert cost per layer and whether it is bandwidth- or compute-bound. (b)
   TheTom's binary, cache off vs on, isolated arm exactly as their doc prescribes: see section
   8 - a real Flash-Next number on THIS V100 in an afternoon. (c) Same A/B on
   DeepSeek-V4-Flash-0731 (present on the X99) = the big-expert vehicle. Decision rule: if
   Flash-Next shows < +10% and V4 < +20% at 1 GPU, park the lane (record the negative with
   full rigor) - the fleet's static 3-GPU layers are the better lever. Otherwise continue.
1. **Pinning only** (loader): pin host-resident `*_exps` at load. Gate: bytes pinned logged, no
   RSS growth, byte-identical decode, prefill t/s on V4-0731 (Flash-Next prefill is #150-bound
   and will not move until that bug is fixed - say so in the result).
2. **Slot tensors + tables + dual-chain graph, static content** (no worker): fill a fixed set
   (profile top-K) at init, no eviction. Gates: unit test of table/skip logic on the CPU op;
   `-ncmoe` vs cache PPL (llamacpp-v100-quality-battery), greedy low-entropy identity, coherence
   read; decode-graph cache + FORCE_GRAPHS on; MTP n-max 3 verify batches on.
3. **Async worker + LRU + admission + step()**. Gates: torn-slot stress (tiny pool, inserts
   high), teardown clean (ASan), hit-rate ramp curve, tg300 x 3 vs `-ncmoe` per fleet-measure
   discipline (256 warm-up tokens: the forks' cold window is real).
4. **Budget/auto + reserve + stats + wizard exposure** (`--moe-cache off|<MiB>|auto`, env
   rows in docs/env-gates.md, fleet card). Gate: launcher configs round-trip, dormant when the
   model fits (parity within noise).
5. **Multi-GPU** when the X99 carrier returns (3 V100): per-device pools under `-sm layer`;
   compare against the static 3-GPU layer split (the honest baseline at that VRAM).
6. **v2 levers, each measured, each with a kill rule**: heat-aware eviction on/off; profile
   weighted per-layer budgets; fused gate/up on the GPU chain (#140 machinery); overlap via
   scheduler support (only if phase-0 decomposition shows the serialized chain matters);
   resident-set dump/reload; SSD arena as fill source for the Task-15 tier.

## 8. What to run today on the X99 (TheTom's build, no port needed)

**DONE 2026-09-07 13:05 at the user's request** - compose rewritten as below (backup
`docker-compose.yml.bak-2026-09-07`): cache ENGAGED (`min-expert=512 KiB`, granted 18000 MiB of
21626 free, pools q4_K 11655 slots / q5_1 5367 / q8_0 680 / q5_K 311 = 18.0 GB, `[moe-cache]
enabled`, 28.9 GiB VRAM used). Probe: coherent (code + arithmetic read), decode 18.4 t/s cold,
21.0 warm, 18.2 after a topic switch, 20.8 back on the warm topic; prompt 23-25 t/s (the #150
prefill class is present in their build too). Missing for a real A/B: the `--moe-cache off` arm
on the same binary (2-min restart) - the phase-0 measurement of section 7. Add
`GGML_CUDA_MOE_CACHE_STATS=200` to the compose environment to get hit-rate lines while running.

Corrected launch for an isolated cache A/B on Flash-Next (their doc, section "Benchmarking",
adapted to Volta; no MTP head because theirs rejects our export):

```
# arm 1: canonical CPU experts, cache off      arm 2: same + cache
GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB=512 /app/llama-bench -m .../Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -ngl 99 -ncmoe 48 -t 24 -fa 1 -p 0 -n 300 --n-gen-warmup 256 -r 3 \
  --moe-cache off,16000 --repack off -v -o json
```

llama-server equivalent: replace `--fit on --n-gpu-layers auto` with `-ngl 99 -ncmoe 48` (or keep
`--fit on` and rely on the 512 KiB floor making their fit choose the cache layout - explicit is
safer), keep
`--moe-cache 16000` (fixed budget; `auto` is Ampere-only), add
`GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB=512` to the environment, drop `--spec-draft-model` (or
convert the head their way), keep `-lv 4` and look for `[moe-cache] enabled` + the pool lines
and teardown `hits=` counters. Expect a 256-token warm-up before steady state. With `-c 262000`
keep the 3 GiB reserve: KV (q8_0/turbo3) took ~2.6 GiB in their log, fine.

## 9. Open questions for the discussion

1. Target vehicle: Flash-Next (the daily driver, smallest expected gain) or V4-0731/GLM/Ling
   (largest gain, models the fleet cannot serve fast today)? Phase 0 answers with numbers; the
   pick decides the order of increments 5-6.
2. Route: D (build) vs A (port giveen) - see the table. A is the choice if the user wants the
   fused path and 4-GPU validation NOW and accepts the merge burden in ggml-cpu.c/ggml-backend.cpp.
3. Slot granularity: per-layer pools (proposed) vs per-shape-class pools (Task 15 / EC3).
4. Whether pinning ships independently first (it is the one piece with a measured win of its
   own, but on Flash-Next it is blocked by #150).
5. CLI surface: mirror TheTom's `--moe-cache off|on|soft|auto|N` for wizard familiarity, or the
   smaller `off|N|auto`.
