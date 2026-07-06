# Token-Generation Improvement Round — Task Tracker

Goal (from `do_your_best.md`): best possible generation quality (no regression), faster prefill/prompt caching, and above all faster token generation on 2x (future 4x) Tesla V100. Baselines and setup details: see memory `v100-inference-setup.md` and commit `d2b706d1b`.

Status legend: ✅ done · 🔄 in progress · ⬜ pending

## ✅ Done (verified, measured)

| # | Task | Where | Result |
|---|------|-------|--------|
| 1 | MTP draft padding to fixed `n_max` so batch shape stays stable and graph reuse works | `common/speculative.cpp` | Single-stream spec 33.6 → 62–65 t/s. Kill-switch: `LLAMA_SPEC_DRAFT_NO_PAD` |
| 2 | Make `llama_meta_device_get_split_state` regexes `static const` (were rebuilt ~29x per tensor per alloc) | `src/llama-model.cpp` | Graph alloc 38ms → 4–6ms; 4-conc spec aggregate 33 → 61 t/s |
| 3 | `GGML_CUDA_FORCE_GRAPHS` env override for the Volta (cc<8.0) CUDA-graphs gate | `ggml-cuda.cu` | Tested: no gain (launch overhead not the bottleneck). Kept as opt-in |
| 4 | One-shot P2P NVLink AllReduce for 2 GPUs (`GGML_CUDA_ALLREDUCE=p2p`, NCCL fallback, 4MB cap via `GGML_CUDA_AR_P2P_MAX_BYTES`) | `ggml-cuda/allreduce.cu(+cuh)`, `ggml-cuda.cu` | tg 48.8 → 49.4 (+1.2%), ppl identical. Committed `d2b706d1b` |
| 5 | Env-gated instrumentation: `LLAMA_SPEC_TIMING` (server spec phases), `LLAMA_DECODE_TIMING` (build/alloc/set-inputs/compute per ubatch) | server, `src/llama-context.cpp` | Used to find items 1–2 and 8 |
| 6 | Distributed-inference analysis doc (coordinator + N workers over RPC) | `docs/distributed-inference-plan.md` | Doc only; implementation deferred (item 12) |
| 7 | Exactness verification | — | temp-0 output identical spec vs nospec |
| 8 | Multi-shape decode graph cache: LRU cache of small-batch decode graphs, each entry with its own scheduler; meta backend compute-container ring 2 → N slots with on-demand shadow recreation and per-entry split-state invalidation. Env: `LLAMA_DECODE_GRAPH_CACHE` (entries, default 4, 0 disables), `LLAMA_DECODE_GRAPH_CACHE_TOKENS` (max cached ubatch size, default 64), `GGML_META_MAX_GRAPHS` (ring slots, default 8) | `src/llama-context.{cpp,h}`, `ggml/src/ggml-backend-meta.cpp` | Reuse 61–64/256 → 248–256/256 ubatches (steady state 100%, build+alloc 0ms); 4-conc spec aggregate 61.7 → 64.7 t/s (+5%); single-stream spec 66 t/s and nospec 48 t/s unchanged; temp-0 output identical cache on/off, spec and nospec. All output-extraction sites (logits/embd/nextn/sampling/layer-inputs) wired to `active_sched()`; `synchronize()` needs no change since all schedulers share the same backend objects |

## ⬜ Pending (approved roadmap, not started)

- [ ] **9. GPU sampling in tensor-split mode** — logits are vocab-split across GPUs; needs an allgather before sampler ops (`src/llama-context.cpp:1195`, meta backend `handle_per_row` assert).
- [ ] **10. TurboQuant 4-bit KV cache type** — best-in-market KV compression with no measurable quality loss vs mainstream (goal item 5 in `do_your_best.md`).
- [ ] **11. Close the remaining 4-concurrency gap** — remeasured after item 8: spec aggregate 64.7 vs 89.6 t/s nospec. Rebuild overhead is now zero (100% reuse), so the rest is compute: 4 slots × 4 tokens/step through the target model at ~64% draft acceptance. Profile whether per-step verify cost or draft-model calls dominate.
- [ ] **12. Distributed inference implementation** — build the coordinator/worker design from `docs/distributed-inference-plan.md` (explicitly deferred; targets the future 4x V100 setup).
