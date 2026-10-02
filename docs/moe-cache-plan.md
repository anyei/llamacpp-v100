# MoE expert cache (VRAM cache for CPU-resident experts) - investigation + proposal (TASKS #151)

Status: **BUILT + ROLLED 2026-09-07 (sections 12.1-12.11); K80/Kepler battery 2026-09-27 found and FIXED the wide-node gather bug (section 12.12); V100 regate 2026-09-28 GREEN, image e117ee884-widefix, the V100 was never exposed (section 12.13); MAX_BATCH >= offload-width assert fixed by a clamp (12.14)**. Built on top (2026-09-30, TASKS #154): the MoE doorbell hands the cache's misses to a host executor without splitting the decode graph, and the prefill stream ring - `docs/strata-port-plan.md`. Originally: PROPOSAL 2026-09-07 (section 7 decision points). No code
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
auto --no-mmap --spec-draft-model mtp-...` (compose in the
user's turboquant service dir on the coordinator). The container died in a restart loop and
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
prefill class is present in their build too). **A/B DONE (user go, same evening): cache OFF (`--moe-cache off --no-repack`) = 13.8 / 13.9 /
14.2 / 14.0 t/s on the same four requests; cache ON = 18.4 / 21.0 / 18.2 / 20.8 -> +33% cold,
+50% warm on Flash-Next on ONE V100 with 18 GB of cache.** Prompt processing 23-25 t/s in both
arms. That is about 2x the section-4 estimate: the CPU expert path on this box costs more than
the pure bandwidth model (71 ms/token at 14 t/s), so the cache's prize is larger. Serve
restored to cache-on with `GGML_CUDA_MOE_CACHE_STATS=200` for hit-rate lines.

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

## 10. Decisions 2026-09-07 (user answers to section 9)

1. **Targets, in order: Qwen3.8-Flash-Next -> Ling-3.0-flash -> GLM-5.2.**
   - Flash-Next: section 4 geometry (71.7 GiB experts, 3.1-3.6 MB/expert, ~1.5 GB/token).
   - Ling-3.0-flash Q4_K_M (X99 `$MODELS_DIR/Ling-3.0-flash/Ling-3.0-flash/`, 2 shards,
     72.4 GiB): arch bailingmoe3, 43 blocks (2 leading dense = 41 MoE), 512 experts, top-8 with
     GROUP routing (8 groups, 4 used), sigmoid gating (func 2), expert_weights_scale 2.5 + norm,
     1 shared expert, n_embd 2560, n_ff_exp 768. Per expert: gate/up Q4_K 1.11 MB each + down Q6_K
     1.61 MB = **3.83 MB**; **69.7 GiB of routed experts**; ~1.26 GB/token. Same class as
     Flash-Next (slightly bigger experts, fewer per token). Design note: the group top-k runs
     BEFORE `selected_experts` in `build_moe_ffn`, so the dual-chain hook (which starts at
     `selected_experts`) is unaffected; the weights scale/norm apply after the down sum as today.
   - GLM-5.2: only the UD-Q4_K_XL set exists locally (coordinator `$MODELS_DIR/GLM-5.2`, 11 shards,
     ~436 GB) - it does NOT fit the X99's 251 GB RAM. The X99 vehicle is the Q2_K_XL (227 GB,
     used for Task 15/#55 streaming; recopy needed) or the SSD tier. Biggest expert share per
     token of the three (TheTom: 209 GiB experts at IQ2_M, 14.5 -> 28 t/s with MTP on 4x3090),
     so the largest expected gain, but capacity-borderline in RAM: plan it last, as decided.
2. **Route = decided by code review.** Two independent reviews are running on (A) giveen's
   in-kernel cache in TheTom's tree and (B) markldn's dual chain + its upstream ancestor
   PR #27861: severity-ranked findings, quality scores, port cost into this fork. Rule: AAA ->
   port; poor -> build (taking the proven pieces). Verdict lands in section 11.
3. **Pool granularity: per-layer pools, sized from the profile, is the better choice here.**
   Why: (a) heat differs per layer, not per shape - our #148 profile needs 244-367 of 512
   experts per layer for 90% of selections (peak/mean 4-16x), so a per-layer budget weighted by
   the profile captures most of what a shared pool would redistribute at runtime; (b) per-layer
   pools keep the layer -> device mapping trivial (the slots live where the layer's router
   lives) and remove the whole shape-census / pool-ordering / role-starvation bug class both
   in-kernel forks hit (EC3 M1, giveen's "wait for as many visits as tensors" rule); (c) the
   only thing lost is runtime migration of capacity between layers, which matters only if layer
   heat drifts faster than a serve restart - measurable later, not needed for v1. Shared-shape
   pools (EC3/giveen) are the right answer only when there is no profile and layers are
   heterogeneous in count (Llama-4's 16-expert tensors) - not our case.
4. **Pinning, in plain words** (the question was whether to ship it alone first): "pinning"
   means page-locking the expert weights in host RAM so the GPU can DMA them directly. Unpinned
   (default mmap) copies go through a driver bounce buffer at about half the PCIe speed. Today
   the scheduler already copies used experts to the GPU for big prompt batches, so pinning
   speeds up prefill on its own, cache or not (markldn measured +41%). Two facts make this a
   non-question: with `--no-mmap` (your turboquant compose, several launcher configs) the
   experts are ALREADY pinned - the loader puts CPU-overridden weights in CUDA pinned host
   memory; and on Flash-Next prefill does not batch at all because of #150, so pinning shows
   nothing there until that bug is fixed. Decision taken: no separate increment; pinning of
   mmap'd experts is a ~60-line part of the cache increment, and the launcher default for cache
   serves is `--no-mmap`.
5. **CLI = `--moe-cache off|N|auto`** (N = MiB per device; env alias `LLAMA_ARG_MOE_CACHE`;
   `auto` = free VRAM minus the reserve after KV and compute buffers, dormant when the routed
   experts already fit). No `on`/`soft` modes.

## 11. Code-review verdicts (2026-09-07, route decision per section 10.2)

### 11.A giveen `--moe-cache` in TheTom/llama-cpp-turboquant @ 407f323 (CUDA provider + hooks, ~4.6k lines read in full)

**Verdict: solid-but-not-AAA.** No defect found that corrupts output in the supported model
(one scheduler thread per session). The hard protocol is right and tested: slot pins +
generation counters, eviction never touches a copying or pinned slot, invalidation on every
buffer-write path, "restore rows to the CPU mapping before the barrier" holds on every failure
path including fused, hit rows use the STOCK GPU mmvq arithmetic (quality = `-ngl` offload, not
a new kernel path), byte math range-checked, Volta-safe (dp4a only, explicit VOLTA branch).

Majors (file:line in their tree):
- M1 every CPU worker thread takes a global registry mutex on EVERY MUL_MAT_ID node
  (`ggml_moe_cache_active()` ggml-backend.cpp:65-77 via `ggml_moe_cache_can_fuse` ggml-cpu.c:3276,
  called from all threads in `try_fuse_ops`) - a 47-thread convoy 3x per MoE layer per token,
  paid even with the cache off; workers and thread 0 can see different provider tables, a latent
  barrier-count deadlock.
- M2 budget latched once (`moe_cache_prepare_budget` :1247-1250, never re-probed); any fill error
  or allocator OOM trims the device and it stays dead for the process lifetime (:1203-1206,
  :3301-3340). A long-lived server loses the cache on the first big-prompt pool growth.
- M3 one session per scheduler, VRAM split N ways, cold N times: `ggml_backend_sched_new` creates
  a default session per scheduler (ggml-backend.cpp:2030-2040); OUR decode-graph cache creates a
  scheduler per graph shape -> N budgets and N cold caches; also their session only scans
  `ggml_backend_is_cuda` backends, and our scheduler list under the meta backend is {meta, CPU}
  -> no session at all without a member-unwrap patch.
- M4 structure: 3.4k-line accretion; ~600 lines of policy duplicated in
  `ggml-moe-cache-common.h` (three copies to keep in sync); `query_fused` declared, documented
  as implemented, assigned by nobody; header claims registration-time asserts that do not exist;
  330/325/210-line functions; mode inferred from a byte field; 20 env vars overriding each other.
- M5 collect = D2H + `cudaStreamSynchronize` + serial memcpy on thread 0 while 47 threads wait
  at the next barrier (2-3 per layer); collect failure recomputes hit rows single-threaded.
Minors: heat counter never decays (after warm-up the policy is plain LRU); the cpu-overlap
auto-tuner times production decodes and never completes under MTP/-np>1; `cudaMalloc` + thread
spawn under the session mutex; toy-shaped tests (256x128, 64 experts, Q4_0) that would not
catch M1; release-build negative-id indexing hazard for our sentinel.

Scores (1-5): concurrency 3, memory/lifetime 4, error handling 3, numerics 4, design 2,
tests 3, perf design 3. Port cost into qwen4exp: drop-in files (moe-cache.cu/.cuh, API header,
test) + hand-merge ggml-backend.cpp (18 hunks, incl. meta unwrap), ggml-cpu.c (9 hunks, ~450
lines, our sentinel folds in), mmvq.cu (5 hunks), llama-context.cpp (6 hunks; decode-graph cache
needs a shared-session API or must be disabled), CLI (9 trivial); fit.cpp optional (ours 981
lines vs theirs 1601; on V100 `auto` is dormant anyway). **5-7 days without fit, 8-10 with, +2-3
to fix M1-M3, plus permanent ownership of 3.4k lines we did not design.**
Worth lifting regardless: the begin/plan/dispatch/collect/end contract with fallback-before-
barrier; `moe_cache_invalidate_session` + `active_sources`; the overflow discipline; gathering
slots with the stock `mul_mat_vec_q` + ids (no new kernel for the plain path); the CPU fused-
SwiGLU matcher + `swiglu_masked`; the `GGML_CUDA_MOE_CACHE_FAIL` stage injection harness; the
eligibility probe. Drop: the auto-tuner, per-scheduler sessions, undecayed heat, the default
session in `sched_new`, the first-provider fallback, 15 of 20 env vars, the common header.

### 11.B markldn/llama.cpp-qwen4exp-lru-async @ bf7cb1b (dual chain; ports upstream draft PR #27861)

**Verdict: solid prototype, not AAA.** The graph shape is right and validated by 6+ testers
(second mul_mat_id chain over slot tensors, get_rows remap, add of the two down outputs; the
sum is exact by construction on the mmvq path), the worker completion protocol is right
(pending -> todo -> batched tensor_set_async + one synchronize per backend -> done -> publish),
the llama-level module is readable. But:

Blockers for OUR use (file:line in their tree):
- B1 process singleton, no teardown: `g_cache`/`g_init_done` (llama-moe-expert-cache.cpp:94-96,
  :291-295), `by_up_src` keyed by raw model tensor pointers (:444), `llama_moe_cache_shutdown`
  has ZERO callers, pinned pages never unregistered. Our server frees and reloads models
  in-process (server-context.cpp:1493-1499) -> a reloaded model can alias the old keys, the
  worker reads a munmapped mapping -> SIGSEGV; VRAM + pinned pages leak per reload.
- B2 the load-bearing invariant "publish only after the copy, never while a graph runs" is NOT
  enforced: `llama_moe_cache_step()` runs right after `graph_compute` returns from the ASYNC
  sched call with `synchronize()` commented out (llama-context.cpp:2096-2102); the trailing
  GPU split (last cached layer's get_rows + chain) is still queued while `flush_table`
  rewrites `dev_table` on another stream -> a just-published expert computed on both chains,
  a just-evicted one on neither, for the last cached layer. Fix = synchronize before step().
Majors: the "async" worker is serialized behind `backend_mtx`, which is also held around the
whole graph compute (gpu_lock, .cpp:616-626; llama-context.cpp:2580-2582) - so fills sit on the
critical path (their own comment at llama-graph.cpp:2167-2169 retracts the HSA theory that
justified the lock); ABBA deadlock between the graph thread (backend_mtx -> mc->mtx via the
observer) and step() (mc->mtx -> backend_mtx) for two contexts on two threads; under `-sm layer`
the chain nodes carry no WEIGHTS-usage source, so the scheduler assigns them to the NEXT
layer's device at layer boundaries and copies the pool tensors across (inferred from sched
rules, not run); unconditional cudaHostRegisterMapped pinning of every offloaded expert tensor
(53 GB, no opt-out, device pointer unused = leftover of the first design; pinned file pages
are unreclaimable); cache VRAM allocated before compute buffers with no fit/capacity awareness;
duplicate dummy-slot ids are AVOIDED by the <= 5 token cap and a kernel patch, not fixed (mmq
loses duplicate lanes silently, mmf still breaks at first match; F16/BF16 experts ungated).
Minors: upload-bounds failure still publishes the slot; no expert-in-flight guard (the same
expert can be scheduled into two slots once the lock goes); observer keyed on a name strstr
(a second model or an MTP head pollutes the LRU); repack bufts unguarded; `-sm row` aborts in
the worker; "bit-identical" claim is token-level on low-entropy prompts (cached lanes = CUDA
q8_1 mmvq vs CPU Q8_K dot -> logits differ by construction); ~450 lines of DEAD first-design
code (moe-lru.cu, two ggml ops with CPU/CUDA/meta dispatch); no tests at all; comments cite a
private planning file.
Scores (1-5): concurrency 2, memory/lifetime 2, numerics 3, error handling 3, design 2,
tests 1, perf design 2. Port cost: faithful port + fixes **5-7 days**; building to the section 5
shape **6-8 days** - the parts we would keep were never the hard part.
Worth lifting: the graph shape (llama-graph.cpp:2207-2215, 2385-2415); init grouping by router
buft (:308-332); the worker protocol + lamport-clock LRU with in-flight exclusion (:448-489,
:523-576); whole-table flush per dirty layer (:190-200); upload_slice bounds check (fail the
job instead of publishing); pinning as opt-in whole-mapping ReadOnly with unregister; the PR
thread's measured LRU curve for Flash-Next (96 slots 87%, 128 -> 90%, 216 -> 95.6%, saturates
~384) and nasone32's residual-overhead profile (dummy rows, syncs, H2D count) as our target list.

### 11.C Route call (against the section 10.2 rule: AAA -> port, poor -> build)

Neither candidate is AAA; neither is plainly poor. A is production-hardened in the protocol
but structurally heavy and operationally brittle (M1-M3) and collides with our meta backend and
decode-graph cache; B has the right shape and two blockers for our server plus a false design
premise in the implementation. Port-with-fixes costs 5-7 (A, no fit, +2-3 for M1-M3) or 5-7 (B)
days; building the section 5 design costs 6-8 and leaves us owning ~1.5k lines we designed
instead of 3.4k (A) or a rewrite of 60% of 1k (B). **Recommendation: BUILD, on B's graph shape,
with A's hardening ideas**, i.e. section 5 plus these hard requirements learned from the reviews:
1. State per model (owned by `llama_model`, freed with it, pins unregistered), never a process
   singleton; keys never raw tensor pointers across reloads.
2. `ggml_backend_sched_synchronize` BEFORE `step()`; tables mutated only there; no lock around
   graph compute; worker owns its backend + stream; expert-in-flight guard + per-step byte cap.
3. Skip sentinel for both chains (no dummy slot, no duplicate ids); eligibility = quantized
   types on the mmvq path, batch <= 8; F16/BF16 experts off.
4. Pool buffers tagged `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` (or chain nodes pinned to the
   layer's backend) so `-sm layer` keeps the chain on the layer's device.
5. Pool allocation after `sched_reserve` (KV + compute first), budget re-probed on reserve,
   degrade per layer on failure, never abort, never permanently dead.
6. No global mutex on the CPU op path; the CPU node reads its host table via src[3] only.
7. Gates: KL battery vs `-ncmoe` (byte identity is impossible by construction) + greedy
   low-entropy read + coherence; failure-injection harness (lift A's FAIL stages); a
   test-backend-ops case for the sentinel dual chain.
**User confirmed 2026-09-07: BUILD.** Phase 0 measured the same evening (section 8): +33% cold /
+50% warm on Flash-Next, 1 V100 - the lane is worth building. Build log: section 12.

## 12. Build log

### 12.1 Increment 1 - module + dual-chain graph + static fill: GATE GREEN (2026-09-07)

Code: `src/llama-moe-cache.{h,cpp}` (per-context cache; per-layer pools `[ne0, ne1, n_slots+1]`
in the router's buffer type, tagged WEIGHTS; two I32 tables per layer: gpu = slot or SKIP,
cpu = expert id or SKIP), `build_moe_ffn` hook (ids through `get_rows` on both tables; the
CPU chain takes `ids_cpu`, a mirrored GPU chain over the slot tensors takes `ids_gpu`; the two
down outputs add before the gating weights), context wiring (cache created BEFORE
`sched_reserve` so reserved and decode graphs share topology; `step()` at decode entry after
`synchronize()`), CLI `--moe-cache off|N|auto` (`LLAMA_ARG_MOE_CACHE`), cparams/llama.h field.
Env instruments: `LLAMA_MOE_CACHE_DEBUG`, `LLAMA_MOE_CACHE_STATIC=K` (fill experts 0..K-1 per
layer at create), `LLAMA_MOE_CACHE_FORCE_CPU` (pools on the host buft = CPU wiring gate),
`LLAMA_MOE_CACHE_MAX_BATCH`. No CPU-op or CUDA changes: the existing negative-id skip sentinel
does the lane split on both backends.

Two traps found and fixed on the way: (1) the fork's `--fit` probe creates a context on a
no-alloc model (no tensor data) -> the static fill segfaulted; a layer without data is now
"not cacheable" (the probe measures only). (2) the CPU repack buffer (`CPU_REPACK`, 11.5 GB
of Q4_K experts on an AVX2 host) is not a host buffer and its bytes are not the canonical
layout -> only 1 of 40 layers qualified; `--moe-cache` now forces `--no-repack` like both
forks. Per-layer skip reasons print under DEBUG.

Gate (CPU build, host, Qwen3.6-35B-A3B UD-Q4_K_XL, `-ngl 0 -t 10`, 3 prompts x 32 greedy
tokens, sha over content): `--moe-cache off -nr` **6a3cf5192354** == `--moe-cache 2048` empty
tables (40 layers engaged, 2102 MiB) **6a3cf5192354** == `LLAMA_MOE_CACHE_STATIC=8`
**6a3cf5192354** (8 experts per layer computed on the second chain). The repacked baseline
also hashes 6a3cf5192354. `GGML_SCHED_DEBUG=2` shows the `ffn_moe_cache_*` nodes in the
executed graph. Decode 8.3 t/s in all arms (no measurable graph overhead at this scale).

### 12.2 Increment 2 - observation + LRU/admission + async fill worker: GATE GREEN (2026-09-07)

Code: the graph registers each engaged layer's flat routed ids on `llm_graph_result`
(`t_moe_cache_ids`, marked output); `decode()` copies them out asynchronously right after
each ubatch's compute (same path as the logits); `step()` at the next decode entry (after
`synchronize()`) publishes the worker's finished uploads (table write), consumes the
observations (hits refresh recency and heat, misses count demand), admits at the first miss
into a free slot or the second miss into a full pool, evicts LRU among slots not in flight and
not hit this step (cold slots with <= `hot_uses` hits first, heat halved every 64 steps),
clears the victim's table entry BEFORE any graph can run again, and hands `{layer, expert,
slot}` jobs to one worker thread that uploads with `ggml_backend_tensor_set_async` on its own
backend per device and synchronizes per batch. Caps: `inserts` per layer per step (2) and
`step_mib` per step (96). Counters + `LLAMA_MOE_CACHE_STATS=N` line.

Gate (CPU build, same vehicle, 3 prompts x 96 greedy tokens, cache 1024 MiB = 14 of 256
experts per layer, no static fill): baseline `-nr` **6a3cf5192354** == dynamic cache
**6a3cf5192354** over 292 steps with **14985 fills / 14482 evictions / 0 observations
dropped**, hit rate 35-46% (39% final). Every upload and eviction raced a running decode and
not one byte moved: the publish-after-sync protocol holds. Same-arithmetic (CPU) proof only;
the V100 gate compares text and PPL, not bytes.

### 12.3 V100 gate round 1 (X99, Flash-Next, 2026-09-07): correct, coherent, throttled

Dev sm_70 bins shipped to the X99 (its `devbins151` dir, run inside the launcher image
with `--entrypoint`), user's turboquant serve paused for the window and restarted after.
Config: `-ngl 99 -ncmoe 48 -c 4096 -fa on -t 24`, mmap default, 3 prompts x 96 greedy tokens
(chat, thinking off).

| Leg | tg t/s (p0 / p1 / p2) | notes |
|---|---|---|
| `--moe-cache off` (repack default) | 14.04 / 14.40 / 14.87 | 6.6 GiB VRAM |
| `--moe-cache off -nr` | 13.51 / 13.84 / 13.72 | the fair baseline (cache mode forces -nr) |
| `--moe-cache 12000`, cold | 14.31 / 15.35 / 15.18 | 48 layers x 85 slots (17% of experts), 18.7 GiB VRAM |
| same, warm (2nd pass) | 15.52 / 15.73 / 15.68 | hit 34-36% steady, resident 2563 of 4080 slots, in-flight 32 |
| same, warm (3rd pass) | 15.18 / 15.72 / 15.66 | |

Correctness: cache-cold and cache-warm outputs identical to each other (sha 82652eab8124);
vs the CPU baseline the texts diverge at a near-tie wording choice ("divide the total distance
by" vs "divide the distance by"), the class both forks document (GPU q8_1 mmvq vs CPU Q8_K dot);
the train answer, the code and the sky explanation are correct and coherent in every leg.

Diagnosis of the small gain (+10-15% vs -nr): the fill schedule was the limiter, not the
mechanism. `step_mib=96` capped uploads at ~32 slabs per step for ALL layers (the counters show
in-flight pinned at 32 every step) and the scheduling loop walked layers in order, so the byte
budget went to the first ~30 layers while the tail layers never filled (resident 2563 = 63% of
slots after 850 steps, evictions in the full front layers while free slots sat in the back).
Fix (12.4): round-robin one upload per layer per round, `inserts` 4, `step_mib` 384.

### 12.4 V100 gate round 2 - round-robin fill schedule: +20-25% (2026-09-07)

Same box, config and prompts as 12.3; scheduler now round-robin over layers, `inserts` 4,
`step_mib` 384 (CPU dynamic gate re-passed byte-identical first: sha 6a3cf5192354, hit 44%).

| Leg (cache 12000 MiB, 85 slots/layer = 17% of experts) | tg t/s (p0 / p1 / p2) |
|---|---|
| cold (first pass) | 15.68 / 16.94 / 16.97 |
| warm (2nd pass) | 16.46 / 17.06 / 16.98 |
| warm (3rd pass) | 16.02 / 17.29 / 17.00 |

Counters: hit **63.6% at step 150 -> 67.3% at step 750** (was 35%), resident 3946 of 4080 slots
(pools full within ~150 steps), fills 53k / evictions 49k over 750 steps, 0 observations
dropped. vs `-nr` baseline 13.5-13.8: **+20-25%**; vs the repack baseline 14.0-14.9: +12-18%.
Outputs coherent; warm pass 2 reproduced the round-1 warm text exactly (sha 82652eab8124), the
other passes differ at near-tie wording only.

### 12.5 V100 gate round 3 - 18000 MiB, same budget as the turboquant run (2026-09-07)

| Leg (cache 18000 MiB, 128 slots/layer = 25% of experts, 24.7 GiB VRAM) | tg t/s (p0 / p1 / p2) |
|---|---|
| cold | 16.67 / 18.30 / 18.60 |
| warm (2nd pass) | 18.52 / 18.97 / 18.65 |
| warm (3rd pass) | 17.81 / 19.00 / 18.62 |

Hit **69.3% at step 150 -> 74.6% at step 750**, pools full (5989 of 6144 slots), 0 dropped.
vs `-nr` baseline 13.5-13.8: **+30-38%**; vs repack baseline: +22-30%. The warm pass 2 output
hashed identical to the CPU baseline (a3ec3a2719c4). Same box, same budget, same prompts as
TheTom's build (18.4 cold / 21.0 warm, 83% hit): this design lands ~10% below it on day one.
Remaining gap, in order of expected value: (1) their fused gate+up+SwiGLU GPU path (one kernel
where we launch three), (2) their in-kernel overlap of CPU misses with GPU hits (we serialize
the CPU split and the GPU chain), (3) shared-shape pools let capacity migrate to hot layers
(75% vs 83% hit) - our answer is profile-weighted per-layer budgets from the #148 artifact,
(4) CUDA graphs for the GPU splits.

Status at the end of the build day: increments 1-2 complete and gated on CPU (byte-identical)
and V100 (coherent, +30-38%); `auto` budget now sized after the KV/compute reserve (12.6);
NOT done: PPL/KL quality battery vs `-ncmoe` (next, before any production roll), pinning of
mmap'd experts, profile-weighted budgets, fused GPU chain, image build/roll, env-gates
wizard exposure beyond the flag catalog. Tree uncommitted (user call).

### 12.6 Cache created after the reserve (2026-09-07)

`llama_context` now runs `sched_reserve()` first, creates the cache (so `auto` reads free VRAM
with the KV and compute buffers already allocated), then reserves once more so the graphs carry
the cache chain. CPU gate re-passed byte-identical (sha 6a3cf5192354). Cost: one extra reserve
at startup when the cache is on.

Reproduction kit (session scratchpad, not in the tree): `gate151.sh` (CPU byte gate: base
`-nr` vs cache legs on Qwen3.6-35B-A3B at `-ngl 0`), `x99_gate.sh` / `x99_gate2.sh` (V100 legs
with the dev bins in the X99's `devbins151` dir run via `--entrypoint` inside the
launcher image; pauses and restores the user's turboquant container), `x99_probe.py`.

### 12.7 V100 gate round 4 - cache + MTP drafter (2026-09-07)

`--moe-cache 18000 --spec-type draft-mtp --spec-draft-model mtp-Qwen3.8-Flash-Next-Mtp-Q8_0.gguf
-ngld 99 --spec-draft-n-max 3` (verify batches of 2-4 tokens go through the cache; max batch 8),
29.5 GiB VRAM:

| Pass | tg t/s (p0 / p1 / p2) |
|---|---|
| cold | 24.48 / 29.89 / 24.50 |
| warm | 32.39 / 32.48 / 27.50 |
| warm 2 | 30.14 / 34.23 / 26.70 |

Hit 66.5% at step 150 (observations 1830 per step = 480 lanes x ~3.8 tokens: the verify
batches are observed and cached), 0 dropped, coherent (cold and warm hashes equal the round-2
cold and warm-2 outputs). Reference: the launcher-managed `-ncmoe 48` + MTP serve of 2026-08-27
measured 25.2 t/s on counting and 21.8 on code without a cache.

### 12.8 Rolled to the X99 launcher (2026-09-07, user order)

Image `llamacpp-local-v100:58dacccec-151` (working tree at HEAD 58dacccec + the uncommitted
#151 changes; wizard flag catalog regenerated, 328 flags) built, pushed as
`127.0.0.1:5000/llamacpp-local-v100:58dacccec-151`, `:latest` re-pointed locally and in the
registry; local tags pruned to current + rollback `41506e100`. X99: pulled the exact tag, the
turboquant container `llama-tq-backend` stopped with its restart policy set to `no` (start it
again with `docker start llama-tq-backend` if wanted), `llama-launcher` recreated on the new
image (the launcher compose in the `llamacpp-v100` service dir, host network, :8399, cache volume
intact, 18 models scanned, health ok, wizard served). Serve loaded through
`POST /models/load` for `Qwen3.8-Flash-Next-UD-Q4_K_XL` with
`-ngl 99 -ncmoe 48 -c 32768 -fa on -t 40 --moe-cache 14000 --spec-type draft-mtp
--spec-draft-model /models/Qwen3.8-FLash-Next/mtp-Qwen3.8-Flash-Next-Mtp-Q8_0.gguf -ngld 99
--spec-draft-n-max 3 --cache-reuse 256`, env `LLAMA_MOE_CACHE_STATS=200`; 27.3 GiB VRAM.

Probe through the router (greedy, thinking off, 160 tokens): code 27.1 t/s cold -> 32.8 warm,
train problem 30.8 (correct), counting 37.0 at 100% draft acceptance; acceptance 82-100%;
all outputs read coherent. Reference for the same shape without the cache (2026-08-27 launcher
serve, `-ncmoe 48` + MTP n-max 3): 25.2 counting / 21.8 code.

### 12.9 Production shape on the X99 launcher + wizard config (2026-09-07, late)

Three findings from putting the serve into the wizard's shape:
1. **17000 MiB is over the edge at 32k + MTP**: at rest 30.3-30.4 GiB, but real requests grow
   the CUDA pools (a 6000-token generation of the user's reached 32.3 of 32.8 GiB) and a prompt
   graph then fails with `ggml_backend_sched_alloc_splits: failed to reserve graph buffers` ->
   HTTP 500 on that request (the serve survives). **15000 is the production budget** (28.3 GiB
   at rest, ~2 GiB growth under use, ~2.5 GiB left).
2. **The wizard's MTP-head template adds `LLAMA_SPEC_DRAFT_NO_PAD=1` and
   `GGML_CUDA_DISABLE_GRAPHS=1`**; with them the same serve ran 20-25 t/s instead of 27-40.
   docs/env-gates.md already calls NO_PAD a kill-switch (padding is the MTP win) and the user's
   proven 76 t/s Qwen config removes both -> the saved config turns both off (`gateOff`).
3. **The ncmoe template's `--load-mode none` costs ~20% decode on this dual-socket host**
   (pinned host memory lands on one NUMA node, 40 threads read from both): at 15000, no-mmap
   23.1 / 25.6 / 27.6 / 31.3 / 19.2 t/s vs mmap **28.2 / 31.8 / 33.7 / 38.6 / 23.4** on the same
   five prompts, and the load takes 79 s vs 9 s -> the saved config cuts `--load-mode`
   (`flagCut`). The cache's fill worker copies from pageable memory then; it keeps up (in-flight
   stays bounded), so pinning stays a v2 item measured on its own.

Saved wizard config: **id 277962759280 "Flash-Next ncmoe48 + MoE cache 15000 + MTP n3 (28-39
t/s, 1 V100, 32k, mmap)"** (mode ncmoe, spec mtph nmax 3, ctx 32768, kv f16, cachereuse,
flagSet `--n-cpu-moe 48 / --moe-cache 15000 / --threads 40`, gateOff NO_PAD + DISABLE_GRAPHS,
flagCut `--load-mode`). The serve running now is exactly that shape (loaded via the API).

Incident: an unload for one of these relaunches landed while the user had a ~6000-token
generation running and killed it. Rule from now on: check the child's `/slots` for
`is_processing` before any unload (the relaunch driver does).

Steady-state numbers seen on this serve: the user's long generation ran at 22.3 t/s (17000
budget, no-mmap, NO_PAD on); the corrected shape does 28-39 t/s on short prompts warm.

### 12.10 The 262k-context OOM (user's serve, 2026-09-07 17:05) and the budget rule

User's wizard serve: `-c 262000 -ctk q8_0 -ctv q8_0 --moe-cache 15000` + MTP head + p-min 0.8
loaded with **86 MiB free** (the local bench skipped for that reason), then the first request
died in `ggml_backend_cuda_buffer_type_alloc_buffer: allocating 861 MiB ... out of memory`
-> `Compute error` on every request. Measured with the same args at `--moe-cache 10000`:
27490 MiB at rest, 28636 MiB after two requests (24-26 t/s cold).

| Context (q8_0 KV, MTP head on) | Non-cache footprint | Safe `--moe-cache` (about 2.5 GiB left after pool growth) |
|---|---|---|
| 32768 | ~13.3 GiB | 15000 |
| 262000 | ~17.1 GiB (KV 3.2 + indexer/state + bigger compute buffers) | **11000** |

Rule: budget <= 32768 - footprint - ~3500 MiB. The cache allocates a fixed N blindly today;
the next code increment adds a clamp to (free VRAM - reserve) with a warning so a too-large N
degrades to a smaller cache instead of a dead serve (and `auto` must reserve the drafter's
VRAM, which loads after the cache is sized). Serve left running at 10000 for the user.

### 12.11 Budget clamp + drafter-aware reserve (2026-09-07, image 58dacccec-151b)

`create()` now reads the pools' device free VRAM after the scheduler reserve and clamps a fixed
`--moe-cache N` to free minus a reserve (WARN with the numbers), `auto` takes exactly that, and
a budget of zero after the reserve logs and stays off. The reserve is 3072 MiB, raised in
`common` by the draft model's file size + 512 MiB when a draft model is configured, because
the drafter loads after the cache sizes itself (the user's 262k OOM of 12.10 would have clamped
15000 to ~11000). `LLAMA_MOE_CACHE_RESERVE_MB` overrides; `llama_context_params.moe_cache_reserve_mib`
carries it. CPU byte gate unchanged (6a3cf5192354).

Rolled as image `58dacccec-151b` (2026-09-07 evening): on the user's 262k + MTP config with
`--moe-cache 15000` the log reads `MoE cache budget 15000 MiB exceeds free 19420 MiB - reserve
7528 MiB: clamped to 11892 MiB` (reserve = 3072 + 3944 draft file + 512); 29.3 GiB at rest,
30.5 GiB after requests, 24-34 t/s, coherent. Rollback image: 58dacccec-151.

### 12.12 Kepler (K80, sm_37) battery + the wide-node gather fix (2026-09-27)

Ask: a build for the coordinator's K80 and the quality check locally (X99 offline). Image
`llamacpp-local-v100:e117ee884-kepler` (CUDA 11.8.0 / ubuntu22.04 / gcc-11, arch 37,
PURGE_CUDA_COMPAT=1, 7 min). Vehicle Qwen3.6-35B-A3B-UD-Q4_K_XL (40 MoE layers x 256 experts
top-8, 1856 KiB per expert slab), one K80 die, `-ngl 99 -ncmoe 40 -fa on -t 16 -nr`, budget
6000 MiB = 82 slots/layer. Scripts and logs: `llama.cpp-work/151-kepler/`.

Level 3 (17 greedy probes, thinking off): cache off 17/17; cache on 17/17 on two passes at
83% hit; decode 16.0-16.9 t/s on vs 16.1-16.4 off = **speed-neutral on Kepler** (no dp4a).

Level 1 (wikitext-2, -c 512 x 16, base = same binary cache off), the K80 facts first:
`-ub 512` is NOT deterministic run-to-run (floors 0.000097 / 0.000187 mean KLD; the sched's
>=32-token expert offload runs the generic dequant+cuBLAS path), `-ub 16` is exactly 0.
Cache legs at -ub 16, static fill 64 experts/layer so the GPU chain carries ~25% of the lanes:

| leg (pre-fix) | mean KLD | same-top | verdict |
|---|---:|---:|---|
| width 4 (cap 16) vs cache-off width-4 control 0.006038 / 97.2% | 0.005651 | 97.5% | clean |
| width 8, DEFAULT cap 8 (hit 63%) vs control 0.005843 / 96.9% | 1.3914 | 57.4% | BROKEN |
| width 16, cap 16 (hit 63%) vs floor 0 | 1.7279 | 53.4% | BROKEN |
| no cache, 10 expert layers on the K80 at width 16 | 0.000405 | 99.3% | kernels sound |

Root cause: `ggml_cuda_mul_mat_id` generic path (ggml-cuda.cu ~2015-2120). Skipped lanes
(`expert_to_use < 0`) never set their `ids_from_sorted_host` entry, the vector was
zero-initialised, and the final `get_rows_cuda(dst_sorted, ids_from_sorted, dst)` gather
copied sorted row 0's OUTPUT into every skipped lane. mmvq (width 1) and mmvq-mmid (2-5) skip
the sentinel correctly; on pre-Turing (`get_mmvq_mmid_max_batch_pascal_older`: q4_K/q5_K = 5,
the V100 included [CORRECTED 2026-09-28: NOT the V100 - Volta takes the MMQ-id kernel below width 64, see 12.13]) widths 6-8 take the generic path. Exposure: cache nodes >= 6 wide = MTP
n-max >= 5, ngram drafts up to 8, MAX_BATCH >= 6 with wide verify batches. The X99 production
shape (MTP n-max 3 = 4-wide verify) escapes. Turing+ goes through MMQ (unverified here).

Fix (user pick "fix 1", 2026-09-27, `ggml/src/ggml-cuda/ggml-cuda.cu`): `dst_sorted` gets one
extra row, memset to zero, and `ids_from_sorted_host` is initialised to that row's index, so
skipped lanes gather zeros. 7 lines. Image `llamacpp-local-v100:e117ee884-widefix-kepler`
(5 min). Regate on the same K80 protocol:

| leg (post-fix) | mean KLD | same-top | PPL ratio |
|---|---:|---:|---:|
| cache off, width 16, vs the pre-fix base | 0.000000 | 100.0% | 1.0009 (stock path untouched) |
| width 8, DEFAULT cap 8 (control 0.005843 / 96.9%) | 0.005236 | 97.5% | 0.9985 |
| width 16, cap 16 | 0.004502 | 97.4% | 0.9988 |
| width 4 (mmvq path, untouched) | 0.005651 | 97.5% | 0.9982 (bit-identical to pre-fix) |
| decode arm, cache on, 17 probes | 17/17 | - | 11-30 t/s, unchanged |

Reading: widths 8 and 16 are now inside the width-controls band. The ~0.005 residual is the
class of "25% of expert lanes computed on the GPU in every layer" (the 0.000405 control put
whole layers on the GPU, 10 of 40); a CPU-only chain-logic control (pools forced onto the host
with `-ngl 0`, base recomputed on the CPU, width 16, static 64) measured **exactly 0.000000 mean
KLD (max 5.3e-5), 100% same-top**: the chain logic is bit-exact; the residual is arithmetic.

Also found: (1) `LLAMA_MOE_CACHE_MAX_BATCH` >= 32 aborts (`GGML_ASSERT(id >= 0 && id <
n_expert)` ggml-backend.cpp:1689, the sched's used-expert scan reads the sentinel) - clamp or
skip, not fixed [FIXED 2026-09-28 by the clamp, see 12.14]; (2) on the K80 the legacy CUDA pool grows across chunks, so the pools turn
that into an OOM: reserve 1024 died at chunk 2, the default 3072 at chunk 16, cache idle both
times - keep the default reserve on Kepler, growth not chased; (3) `LLAMA_MOE_CACHE_FORCE_CPU`
only applies when the router is on the host (with a GPU present the pools still land on it).

### 12.13 V100 regate of the wide-node fix (2026-09-28, image e117ee884-widefix)

Same protocol as 12.12 on the X99 (one V100 32 GB, Qwen3.6-35B-A3B Q4_K_XL staged in tmpfs,
-ncmoe 40, cache 6000 MiB, static 64, -t 40; artifacts `llama.cpp-work/151-v100/x99/`). Two
images: pre-fix 58dacccec-151b and the working-tree build e117ee884-widefix (fix 1 +
PURGE_CUDA_COMPAT + regenerated FLAGS catalog). Mean KLD / same-top vs the pre-fix cache-off
-ub 16 base (PPL 7.0950):

| leg | pre-fix (151b) | post-fix (widefix) | control (cache off, same width) |
|---|---:|---:|---:|
| width 4, cap 16 | 0.005333 / 97.1% | 0.005333 / 97.1% | 0.005450 / 96.9% |
| width 8, default cap 8 | 0.005364 / 97.2% | 0.005364 / 97.2% | 0.005598 / 97.2% |
| width 16, cap 16 | 0.006053 / 97.3% | 0.006053 / 97.3% | - |
| cache off, width 16, vs the pre-fix base | (base) | 0.000000 / 100% (max 5.1e-5) | - |
| CPU chain control (-ngl 0, FORCE_CPU pools, width 16, own CPU base) | - | 0.000000 / 100% | - |
| decode probes (17, greedy, 4k ctx) | - | 17/17 off, 17/17 on | - |

Decode t/s (min/mean/max): cache off 33.6/47.9/74.4, cache on 28.1/58.9/82.5 (+23% mean; hit
83% at step 1250, 6021 MiB of slots, 4 inserts/layer/step). 14/17 probe outputs byte-identical
across the arms; the 3 long-form ones (binary search, planets, story) share their openings and
diverge in wording only.

Reading: the V100 never had the bug. `ggml_cuda_should_use_mmq` (mmq.cu) on Volta = fp16 MMA
hardware without int8 MMA -> MMQ for `ne11 < MMQ_DP4A_MAX_BATCH_SIZE` (64), so mul_mat_id at
widths 6-63 runs the MMQ-id kernel, which maps the skip sentinel correctly; the generic gather
is reached only at width >= 64, which the cache chain cannot emit (MAX_BATCH >= 32 asserts,
12.12). The K80 (cc 3.7, below DP4A) has no MMQ at all and fell into the generic path right
above the mmvq-mmid limit. The 12.12 claim that the V100 shared the exposure at widths 6-8 was
wrong: the fix is a no-op on the V100 for MMQ-supported expert quants (every current target)
and matters on Kepler and on any generic-path case (unsupported quant type, f16/bf16 experts,
FORCE_CUBLAS builds). Consequence: on the V100 the cache was never unsafe under MTP n-max >= 5
or ngram drafts; that restriction was Kepler-only and is now fixed there too. The widefix
image is bit-identical to 151b on the stock path and digit-identical on the cache legs = safe
to roll; `:latest` was not re-pointed and no launcher was rolled (user call; the X99 launcher
container is currently absent).

### 12.14 MAX_BATCH clamp (2026-09-28)

The `LLAMA_MOE_CACHE_MAX_BATCH >= 32` abort from 12.12 is the sched's op-offload path: a
mul_mat_id node at least `GGML_OP_OFFLOAD_MIN_BATCH` tokens wide (default 32, env-parsed in
ggml-cuda.cu) has its host-resident expert weights offloaded to the GPU, and the sched's
used-expert scan (ggml-backend.cpp:1689) reads the node's ids to pick the experts to copy. The
CPU chain's ids carry the skip sentinel for cached lanes, so the scan hits a negative id and
asserts. Fix (src/llama-context.cpp): clamp `max_batch` to `GGML_OP_OFFLOAD_MIN_BATCH - 1`
with a WARN. Not chosen: skipping negatives in the scan - fleet-shared code, and the assert is
a legitimate guard; the chain simply must never own a node the sched offloads.

Gate on the X99 V100 (dev build-cuda75 bins inside the widefix image, Qwen3.6-35B-A3B, base =
image binary, cache off, -ub 64):

| leg | binary | result |
|---|---|---|
| A: MAX_BATCH=64, -ub 64, cache on | image (no clamp) | `max batch 64`, then the assert at ggml-backend.cpp:1689 (trap reproduced) |
| D: cache off, -ub 64 | dev | 0.000000 / 100% (dev libs == image on the stock path) |
| B: MAX_BATCH=64, -ub 64, cache on | dev | WARN `reaches the sched offload width 32, clamped to 31`, `max batch 31`, 0.000000 / 100% (64-wide nodes stay on the stock path) |
| C: MAX_BATCH=16, -ub 16, cache on | dev | 0.005717 / 97.6% vs the -ub 64 base (width band), hit 63.6% = chain still engages |

Artifacts `llama.cpp-work/151-v100/x99-mb/`. Not in the rolled images yet (env-only trap; rides
the next build).

### 12.15 First serve of the GSQ-RCO 3.5bit Flash-Next on the fork (2026-09-28, image 1c63c03a1-smt2)

The user replaced the UD-Q4_K_XL on the X99 with `Qwen3.8-Flash-Next-GSQ-RCO-3.5bit.gguf` (44.6 GiB,
file type Q3_K-S: 642 q8_0 attention/dense tensors, 72 q3_K + 62 q2_0 + 10 q2_K expert tensors, 97
bf16) plus the PLE table as a second shard (`Qwen3.8-Flash-Next-ngram-embeddings-Q4_0.gguf`, 27.5
GiB), joined by `qwen3.8-flash-next-0000N-of-00002.gguf` symlinks; the launcher lists the set as
`qwen3.8-flash-next`. The 12.9 production shape was transplanted as-is (ncmoe 48, cache 15000, MTP
head Q8_0 n-max 3 p-min 0.75, 32k f16, mmap, template envs off, `--model-draft` pinned to the
Flash-Next head because the wizard's `draftMatches` still cross-matches the 27B head, #149) and
launched through the launcher (`POST /models/load`, 1 V100, `-lv 4` + `LLAMA_MOE_CACHE_STATS=200`
for the read-out only).

Load 40 s cold page cache / 10 s warm. Placement: CPU_Mapped 44976 + 28110 MiB, CUDA0 model 4380
MiB, KV 768 + 288 MiB, RS 1801 MiB, compute 726 MiB; `MoE cache on: 48 layers, 15056 MiB of slots`
(no clamp); draft 3290 MiB. VRAM 26.9 GiB at rest, 28.6 GiB after requests (n-max 4: 29.1 GiB).
`--cache-reuse 256` is refused on this context (`cache_reuse is not supported by this context`) -
harmless no-op on qwen4exp.

| leg (Phase 0 prompts, 6 x 200 tokens, greedy, thinking off) | cold mean | warm mean | acceptance | cache hit |
|---|---|---|---|---|
| n-max 3 (saved shape) | 26.7 (20.0-31.7) | **27.6** (21.5-31.7) | 34-87 % | 81 -> 84 % |
| n-max 4 | 26.6 (21.0-32.0) | 27.3 (21.6-31.7) | 30-84 % | 81 -> 83 % |

n-max 4 is inside the noise band with lower acceptance and +0.5 GiB VRAM: n-max 3 stays. Quality:
the 12.12 exact-answer battery (17 greedy probes, `/home/anyei/151-v100/probes.py`) = 16/17; the
one miss is the keyword check (the story says "beacon" for the lighthouse; the text is on topic).
All six Phase 0 outputs read coherent and correct (LRU class, C binary search with the invariant,
train setup, planets). Greedy run-to-run: 1/6 byte-identical between the two n-max 3 passes, the
other five diverge at near-tie wording points 82-149 chars in ("hazardous rocks, shoals" vs "hidden
dangers") = the cached-lane GPU-vs-CPU arithmetic class of 12.13, but denser than the Q4_K_XL
battery's 3/17 - consistent with flatter logits at 3.5 bpw. A KL battery against the UD-Q4_K_XL
base (skill quality-battery) is the way to say whether the quant is "good enough"; it needs the
Q4_K_XL back on the X99 (user: "I will bring the UD Q4_K_XL later").

Saved wizard config: **id 55366197782 "GSQ-RCO 3.5bit: ncmoe48 + MoE cache 15000 + MTP head n3
(27.6 t/s warm avg, hit 84%, 16/17 exact, 1 V100, 32k f16, mmap) 2026-09-28"** on model
`qwen3.8-flash-next`; serve left UP in that shape. The 27B dense counterpart from 4.10 of the
mmsmt doc is id 53382003705 (`Qwen3.8-27B-UD-Q4_K_XL`, gpu1 + MTP n-max 4 + T1 route, 56.9 t/s).
Artifacts: X99 `/home/anyei/153-p0/probe-gsq-{rep1,rep2,n4-rep1,n4-rep2}.json`,
`probes-gsq-exact.json`.
