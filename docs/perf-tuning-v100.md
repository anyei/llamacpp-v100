# V100 Tensor-Parallel Performance Tuning — Reference

Optimization rounds of July 2026 on 2x Tesla V100-SXM2-32GB (NVLink, cc 7.0),
serving via `llama-server` in tensor-split mode.
Task-by-task history: `TASKS.md`. Image rebuild: `REBUILD-IMAGE.md`.
Serving profiles: `docker-compose.mtp.yml` / `docker-compose.nospec.yml`.

Models used throughout (mounted at `/models` inside the containers):

| Model file | Role |
|---|---|
| `Qwen3.6-27B-UD-Q4_K_XL-MTP.gguf` (16.4 GB) | primary serving + benchmark model — hybrid attention + DeltaNet with MTP head |
| `Qwen3-0.6B-BF16.gguf` | small test model for distributed/RPC experiments (fast iteration) |
| `GLM-4.7-Flash-REAP-23B-A3B-UD-Q4_K_XL.gguf` (~14 GB) | MoE + MLA (`deepseek2`) experiments, single-GPU serving profile |

## Headline results

| Metric | Before | After | Change |
|---|---|---|---|
| Generation, single stream, MTP spec | 33.6 t/s | 66 t/s | +96% |
| Generation, 4 concurrent, MTP spec (aggregate) | 32.5 t/s | 125.8 t/s | +287% |
| Generation, 4 concurrent, no spec (aggregate) | 89.6 t/s | 89.6 t/s | baseline |
| Prefill, long chat history (small messages) | 860 t/s | 1237 t/s | +44% |
| Prefill, clean 4096-token batches, depth 0 | ~1400 t/s | ~1400 t/s | unchanged |
| KV memory (q8_0, verified lossless) | 1x | 0.5x | −50% |

Exactness: all changes verified byte-identical temp-0 output vs the unmodified
baseline (fresh-server and cached-prompt paths).

## What was changed (chronological)

1. **MTP draft padding** (`common/speculative.cpp`) — drafts padded to fixed
   `n_max` length so the verify batch shape stays stable and graph reuse works.
   Made variable-confidence drafting (`--spec-draft-p-min 0.75`) viable.
   Kill switch: `LLAMA_SPEC_DRAFT_NO_PAD=1`.

2. **Static split-state regexes** (`src/llama-model.cpp`) — tensor-parallel
   graph alloc 38ms → 4-6ms.

3. **One-shot P2P AllReduce** (`ggml-cuda/allreduce.cu`) — direct NVLink
   exchange for small tensors instead of NCCL latency. Opt-in:
   `GGML_CUDA_ALLREDUCE=p2p` (cap: `GGML_CUDA_AR_P2P_MAX_BYTES`, default 4MB).

4. **Multi-shape decode graph cache** (`src/llama-context.{cpp,h}`,
   commit `9bac6eead`) — LRU cache of small-batch decode graphs, each entry
   owning its own scheduler, so workloads alternating between a few shapes
   (spec draft/verify) replay graphs instead of rebuilding. 100% steady-state
   reuse. Env: `LLAMA_DECODE_GRAPH_CACHE` (entries, default 4, 0 disables),
   `LLAMA_DECODE_GRAPH_CACHE_TOKENS` (max cached ubatch size, default 64).
   Meta backend side: compute-container ring (`GGML_META_MAX_GRAPHS`,
   default 8 — keep above the decode cache size) with on-demand shadow
   recreation, so an undersized ring degrades gracefully.

5. **KV cache quantization** (commit `73ec43626`) — verified in tensor mode
   with the in-tree Hadamard rotation (upstream PR #21038, auto-enabled for
   quantized KV): q8_0/q8_0 is lossless (ppl 6.2922 vs 6.2904 f16 @32k) at 2x
   compression; K q8_0/V q4_0 gives 2.67x at +0.14%; q4_0/q4_0 gives 4x at
   +0.30%. Mixed K/V types require `GGML_CUDA_FA_ALL_QUANTS=ON` (now in
   `.devops/cuda.Dockerfile`) — without it there is no FA kernel for K≠V and
   tensor mode aborts. TurboQuant evaluated and rejected (see TASKS.md item 10).

6. **Full-sequence equal splits for spec rollback** (`src/llama-batch.{h,cpp}`,
   commit `abf23cdd1`) — hybrid models (attention + recurrent DeltaNet) forced
   one sequence per ubatch whenever speculative rollback was active, so a
   4-slot verify ran as 4 sequential decodes. `split_equal(..., full_seqs=true)`
   groups consecutive equal-count sequences whole (preserving the
   rollback-snapshot invariant). 4-conc spec: 64.7 → 125 t/s.

7. **Prompt-batch defragmentation** (`tools/server/server-context.cpp`,
   commit `a81a4218c`) — the server broke prompt batches at *every* user
   message for checkpointing; now only where a checkpoint is actually created
   (`checkpoint_min_step` spacing, default 8192, or the last user message).
   Long chat-history prefill +44%.

8. **Prompt-cache checkpoint pruning** (`tools/server/server-task.cpp`,
   commit `d8a6fa636`) — cached conversations keep only their last 2
   checkpoints. On hybrid models each checkpoint is the full recurrent state
   (~150-230 MiB regardless of token count), so long chats were becoming
   multi-GiB cache entries. Compose profiles set `--cache-ram 12288
   --ctx-checkpoints 8`; the old defaults (32 checkpoints x ~150 MiB x slots +
   8 GiB cache) could OOM a 46 GiB host.

9. **Simplify pass** (commit `65286b53f`) — one graph-params construction per
   ubatch (was up to 6 copies of a ~27 KB struct), shared checkpoint-spacing
   predicate, range-constructed checkpoint tail, collapsed split branches.
   Also resolved the last temp-0 divergence.

10. **MMVQ Volta (sm70) table** (`ggml/src/ggml-cuda/mmvq.cu`, commit
    `b912d1b1e`) — Volta previously fell through to `MMVQ_PARAMETERS_GENERIC`,
    which lumps `ncols_dst` 1-4 at `nwarps=4`, untuned for sm70. Added a
    dedicated `MMVQ_PARAMETERS_VOLTA` that differs only at the batch-1 decode
    path: K-quants (Q2_K…Q6_K) now use `nwarps=2` (mirrors TURING). Measured
    sweep, single-GPU tg128, Qwen 27B Q4_K: nwarps 1=30.2, **2=31.6**, 4=31.1,
    8=27.8 → **+1.8%**. ppl identical to 4 decimals (5.9063). The change alters
    only the cross-warp reduction order, so it is *not* byte-exact (gated on
    ppl, not bytes). **Scope: batch-1 nospec decode only** — the MTP-spec prod
    path verifies at batch-4 (`ncols_dst=4`) and is unaffected (measured
    neutral). Tuning `ncols_dst` 2-4 helped pure batched decode (+3-4%) but was
    prod-neutral on the AllReduce-bound 2-GPU spec path — deliberately not
    shipped (see TASKS.md item 18 for the follow-up on the nospec `-np>1`
    profile). Pascal/older keep GENERIC.

## Volta-specific facts worth remembering

- CUDA graphs are hard-disabled for cc < 8.0; launch overhead is not the
  bottleneck (measured via `GGML_CUDA_FORCE_GRAPHS`, no gain).
- Quantized matmul dispatch: MMVQ vec kernels up to 8 tokens, dp4a MMQ from 9
  to 64, dequant+cuBLAS above. Step-time curve (27B, 2 GPU): 1 tok = 20.8ms,
  4 = 40.4, 8 = 69.6, 12 = 65.8, 16 = 74.5, 32 = 123 — note MMQ at 12 beats
  MMVQ at 8. The batch-1 MMVQ path for K-quants is tuned for sm70 as of
  `b912d1b1e` (`nwarps=2`, +1.8% nospec decode; see change #10).
- Prefill at depth is FA-bound physics: ~1400 t/s at depth 0 falls to ~720 t/s
  at 65k context in perfect batches. Mid-prefill log numbers are cumulative
  averages, so they always decay; look at deltas between lines for the
  instantaneous rate.
- **Long-context *decode* is FA compute-bound, not KV-bandwidth-bound**
  (measured, task 19). tg decays ~30% from depth 0 to 65k (Qwen 27B single-GPU:
  31.5 → 22.1 t/s, ~13.5 ms/token of FA at 65k; only full-attention layers
  decay, DeltaNet is depth-independent). The decode kernel is `fattn-vec`
  (D=256, stream-K KV-split already engaged). **Consequence: quantizing KV
  makes long-context decode *slower*, not faster** — it adds dequant work to an
  already compute-bound kernel (d0→d65k decay: f16 −29.8%, q8_0 −32.0%, q4_0
  −34.6%). Use quantized KV for *capacity* (more context/slots), never for
  long-context *speed*. The vec-kernel block size (`nthreads=128`) is already
  optimal on Volta (64 and 256 both regress); further speed needs sm70
  kernel-algorithm work (deferred).
- `GGML_CUDA_FORCE_CUBLAS` / `FORCE_MMQ` are compile-time flags; the env vars
  do nothing.
- Tensor mode requires flash attention and disables backend sampling — the
  latter costs nothing (CPU sampling fully overlaps GPU compute; measured
  89.6 vs 89.5 t/s with a full sampler chain).

## Diagnostic knobs (all env-gated, zero cost when unset)

| Env | Prints / does |
|---|---|
| `LLAMA_DECODE_TIMING=1` | per-256-ubatch build/alloc/set-inputs/compute + reuse counts |
| `LLAMA_SPEC_TIMING=1` | per-64-iteration draft/checkpoint/decode/accept phases |
| `LLAMA_BATCH_DEBUG=1/2` | ubatch composition per split (needs `-lv 5`) |
| `GGML_META_DEBUG=1\|2` | tensor-parallel split states per node; `2` adds per-src resolution (needs `-lv 5`) |
| `GGML_META_DEBUG_REDUCE=1` | where each AllReduce boundary lands (placement is device-count-independent — a 1-GPU run is authoritative) |
| `GGML_META_TIMING=1` | per-128-graph wall-time attribution, compute vs AllReduce-boundary (adds explicit syncs — diagnostic only, not for serving). Measured the reduce tax at ~10 ms/token that motivated task 17 |
| `LLAMA_META_DUP_DEVICE=N` | duplicate the tensor-split device list: one GPU runs a genuine n-way split (byte-exactness gates + split debugging without a multi-GPU box) |
| `LLAMA_DEBUG_DUMP_DIR` + `_FILTER` | full-tensor dumps from the eval callback (corner samples hide mid-tensor divergence) |
| `GGML_RPC_NO_W2W=1` | coordinator env: force coordinator-bridged copies instead of worker-to-worker pulls |

## Which split mode per architecture

| Architecture | Examples | `-sm tensor` | Recommendation |
|---|---|---|---|
| Dense / GQA | Qwen3.x dense, Kimi-Dev-72B, Llama-family | ✅ supported, tuned | tensor (2 GPU) |
| Hybrid recurrent | Qwen3.5/3.6 (DeltaNet layers) | ✅ supported, tuned | tensor (2 GPU) |
| MoE w/ GQA | Qwen3.x-A3B MoE | ✅ supported | single GPU if it fits (light active set), else tensor |
| **MLA (`deepseek2`)** | DeepSeek V2/V3/R1, Kimi K2, GLM-4.7-Flash | ✅ supported since task 13 closed (attention mirrored on every device, FFN/experts split; ppl statistically identical to single-GPU; temp-0 text can diverge — MoE-router-amplified reduction noise, quality-neutral) | **single GPU** still recommended when the model fits (as fast as TP for MoE-light models); tensor mode for models that don't |

MLA background: `deepseek2` models keep a single shared latent KV head
(e.g. GLM-4.7-Flash: `n_head_kv = 1`, 576-dim latent, ~1.1 KB/token) — there is
no head dimension to split, so the head-based tensor-split policy breaks.
Since `c45c70081` the combination fails at model create with an actionable
error (0.2 s, before weights load) instead of a GGML_ASSERT + core dump that
crash-looped `restart: always` containers. The proper scheme (mirror the
latent cache, split the query heads) is the remaining work in TASKS.md
item 13; it matters mainly for the 4x V100 future if a large MLA model
becomes a daily driver. Measured on GLM-4.7-Flash-REAP (23B-A3B, 14 GB Q4):
single V100 = 79 t/s coherent; the WIP tensor mode is no faster (79.8) and
incorrect. Related: task 14 — freeing a tensor-split model without a context
segfaults in cleanup, which is why the gate throws at model create.

## Agreed deployment plan for the future hardware

Target: new machine with a 4-slot SXM2 board (NVLink island) + a 2-slot SXM2
board (NVLink island, PCIe between boards); this machine keeps a PCIe V100.

1. **4-GPU island = flagship, one process**: main model, `-sm tensor
   -ts 0.25,0.25,0.25,0.25` + NCCL. Never TP across the PCIe gap to the
   other board. On assembly, run `nvidia-smi topo -m` and bench 6-way vs
   4-way tensor before trusting this assumption.
2. **2-GPU board = independent second server** (second model / draft /
   coder), not a pipeline stage — merge into the flagship only when a model
   needs > 128 GB.
3. **PCIe V100 = utility node** (small/MoE models solo, embeddings). Never
   in a TP group; never host spec-draft models across the network from
   their target.
4. **Cross-machine distribution = held in reserve** for (a) models that do
   not fit 192 GB or (b) evacuating tail layers to free island VRAM for
   giant contexts. Both use `-sm layer --rpc worker:50052`
   (`docker-compose.rpc-worker.yml`); measured protocol cost ~3%/token +
   2x network RTT.

## Distributed inference — measured results (2026-07-07)

All numbers on loopback RPC (worst case for latency; a wired LAN adds ~2x
RTT per token on top). Model: `Qwen3.6-27B-UD-Q4_K_XL-MTP.gguf` unless noted.
Full experiment history: `TASKS.md` item 12.

| Experiment | Result |
|---|---|
| 27B in-process `-sm tensor` (reference ceiling) | 47.8 t/s |
| 27B whole-model on a 2-GPU TP island | 33.3 t/s short ctx, 29.2 t/s @ 5.2k ctx |
| 27B pipeline (1 local GPU + 1 RPC GPU, `-sm layer`) | 32.3 t/s vs 33.2 in-process = ~3%/token |
| 0.6B whole-model on island (`Qwen3-0.6B-BF16.gguf`) | 148 t/s (free GPUs) |
| Island VRAM, 27B @ 8k ctx | 19.6 GB/GPU before the slice-packing fix → **9.2 GB/GPU** after |
| First model load over the socket | ~40 min (synchronous protocol, one-time) |
| Reload with worker weight cache (`-c`) | **40-50 s** |
| 27B slot save (501 MB, 5259 tokens) over RPC | 531 ms save / 635 ms restore, post-restore generation byte-identical |
| Hybrid context checkpoint over RPC (5212-token prefill) | 149.6 MiB state read, clean |

State-integrity methodology (caught two silent corruption bugs): the same
logical state is saved from three configs — island over RPC, in-process
tensor-parallel, single GPU — and compared byte-for-byte. Island and
in-process TP saves are byte-identical; TP vs single-GPU differ in ~40% of
bytes, which is low-mantissa AllReduce ordering noise (round trips are exact
and generated text is identical). Any structural deviation from this pattern
indicates a placement bug — this is the regression test for the meta
backend's raw-byte get/set paths.

Storage baseline (recorded for task 15, `--ssd-streaming`): the models NVMe
sustains 1.6 GB/s O_DIRECT sequential reads → a dense 27B Q4 fully streamed
from disk has a ~0.1 t/s decode ceiling; see TASKS.md item 15 for what that
implies.

Docker images: `llamacpp-local-v100:v100-spicy` = the pre-distributed
serving image kept as rollback; `llamacpp-local-v100:distributed-inference`
= built from `ddc1fd670` with the full distributed stack (islands,
split-state push, slice-packed allocation, state save/restore over RPC,
alias-lifetime hardening).

## Remaining roadmap

- Task 12 phase 1 (TP islands): **complete, measured, hardened** — worker
  `ggml-rpc-server --tensor-parallel` + automatic coordinator split-state
  push; 27B hybrid runs sharded (9.2 GB/GPU) and coherent on a 2-GPU island;
  state save/restore + prompt checkpoints verified byte-exact over RPC;
  adopted-alias lifetime closed. Usage guide:
  `docs/distributed-inference-guide.md`.
- Task 12 phases 2-3 (async RPC, cross-host NCCL): parked unless RDMA/25GbE
  or a concrete >192 GB model plan appears.
- Task 13: correct MLA tensor-parallelism (see above) — priority rises only
  if a DeepSeek-family model becomes the flagship.
- Task 15: `--ssd-streaming` three-tier weight placement — investigation
  phase; breakdown and storage baseline in TASKS.md.
- **Token-generation levers, closed 2026-07-08 (details in TASKS.md 17-20):**
  - Task 17 (overlap AllReduce with the next subgraph's independent prefix):
    **closed negative, reverted.** The residual stream makes every post-reduce
    node consume the reduced tensor immediately, so there is no independent
    prefix to hoist (dense/hybrid), and where one exists (MoE) the per-boundary
    event/launch cost on Volta exceeds the reduce it hides. The reduce tax is
    unrecoverable at the scheduler level; only a chunked-GEMM reduce-scatter
    epilogue (kernel work) could help.
  - Task 18 (sm70 MMVQ tuning): **partial win landed** (+1.8% batch-1 nospec
    decode, prod-spec neutral) — see change #10.
  - Task 19 (FA long-context decay): **sized, compute-bound, deferred** — the
    easy launch-config lever is exhausted; see Volta facts above.
  - Task 20 (adaptive speculative draft cap, `LLAMA_SPEC_ADAPTIVE`): **measured
    tg-neutral, ships off.** Cuts ~5% of draft passes but the p_min gate already
    captures the early-exit value and MTP drafting is cheap vs verify.
  - **Net: batch-1 tg headroom on Volta is now mostly exhausted at the
    orchestration/launch-config level. Remaining gains require sm70
    kernel-algorithm work (MMVQ small-batch, fattn-vec) — high effort, uncertain
    payoff.**
