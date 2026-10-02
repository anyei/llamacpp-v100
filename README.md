# llama.cpp — V100 Tensor-Parallel Fork

A performance-focused fork of [llama.cpp](https://github.com/ggml-org/llama.cpp)
tuned for **NVIDIA Tesla V100 (Volta, sm70, NVLink)** serving, with working
**tensor parallelism**, tuned **speculative decoding (MTP)**, and experimental
**distributed inference** (coordinator + workers, tensor-parallel "islands").

Reference hardware: Tesla V100-SXM2-32GB. The current numbers come from the X99
box (one V100 32 GB, 2x Xeon E5-2690 v4, 251 GB RAM); the older tables come from
the original 2x V100 NVLink box (see Reference hardware below). Models used for
tuning and benchmarks:

- `Qwen3.8-27B` (dense, MTP head built in) - current dense serving model
- `Qwen3.8-Flash-Next` (`qwen4exp` MoE, 177B total / 3B active, GSQ-RCO 3.5-bit,
  72 GiB) with its Q8_0 MTP head file - current MoE serving model: experts in
  host RAM behind a VRAM expert cache
- `Qwen3.6-27B-UD-Q4_K_XL-MTP.gguf` (16.4 GB) and `Qwen3.6-35B-A3B` - the
  earlier reference tables
- `GLM-4.7-Flash-REAP-23B-A3B-UD-Q4_K_XL.gguf` (~14 GB) — MoE / MLA
  experiments, single-GPU profile
- `Qwen3-0.6B-BF16.gguf` — small model for fast distributed/RPC iteration

## Latest results (2026-09-30): Qwen3.8 on one V100

One Tesla V100-SXM2-32GB (250 W) in the X99 box (2x Xeon E5-2690 v4, 251 GB RAM),
image `llamacpp-local-v100:063bdc2a0-db` (the #154 port, committed as 1300d2bef),
single stream, MTP self-speculation with up to 3 draft tokens per step
(`--spec-draft-n-max 3 --spec-draft-p-min 0.75`). Decode t/s is the server's own
`eval time` summary for the whole request.

**Qwen3.8-27B** (dense, `Qwen3.8-27B-AD-IQ3_S.gguf`, 13.8 GB, MTP head built in),
one long generation per context window:

| Context window | KV cache | Tokens generated | **Decode t/s** (whole request) | Draft acceptance |
|---|---|---:|---:|---:|
| 64k (`-c 65536`) | f16 | 65,343 (window full) | **52.5** | 91.9 % |
| 128k (`-c 131072`) | q8_0 | 76,280 | **47.2** | 88.0 % |

The rate follows how predictable the text is: per 8k-token stretch it ranged
46-66 t/s in the 64k run and 41-57 t/s in the 128k run. What keeps long contexts
fast on Volta (same-binary A/Bs on Qwen3.8-27B UD-Q4_K_XL, TASKS #153):

- **Small-width flash-attention kernel** (T3, default on): MTP decode at 56k depth
  +19 % (f16 KV) and +23 % (q8_0 KV) with identical text; the verify step at 128k
  takes 71 ms instead of 106 (f16) and 73 instead of 119 (q8_0), and the second
  kernel increment took another ~3 % off. q8_0 KV now runs within ~2 % of f16 at
  depth, so it halves KV memory at almost no speed cost.
- **Small-batch tensor-core matmul route** (T1, `mmsmt`, default on): MTP verify
  batches (3-8 rows) run on the tensor cores, 1.17-1.34x faster at widths 4-8;
  best MTP serve setting +8 % (52.8 -> 56.9 t/s).

```bash
llama-server -m Qwen3.8-27B-AD-IQ3_S.gguf -ngl 99 -c 131072 -fa on -ctk q8_0 -ctv q8_0 \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75 --cache-reuse 256
# the 64k run: -c 65536 with f16 KV (no -ctk/-ctv)
```

**Qwen3.8-Flash-Next** (`qwen4exp` MoE, 177B total / 3B active, 512 experts top-10,
GSQ-RCO 3.5-bit, 72 GiB) on the same box: dense weights on the GPU, routed experts
in host RAM (`-ncmoe 48`) behind a 15 GB VRAM expert cache, Q8_0 MTP head file,
`-t 40`. Before = production on 2026-09-29, after = production now; one session,
runs interleaved ([`docs/strata-port-plan.md`](docs/strata-port-plan.md) 9.1):

| | Before | **After** | Change |
|---|---:|---:|---|
| Decode, mean of 3 prompts (t/s) | 28.0 | **40.0** | **+43 %** |
| Time per MTP step (ms) | 94.6 | **64.0** | 1.48x faster |
| Prefill, 3.7k-token prompt (t/s) | 79.5 | **326.2** | **4.1x** |
| Prefill, 15.9k-token prompt (t/s) | 81.1 | **330.3** | **4.1x** |

On the production serve after the roll: 50.5 and 43.3 t/s on two 256-token test
prompts, and 48.9 t/s over a 6,274-token generation (81 % draft acceptance).
Where it came from (TASKS #154): an AVX2 Q2_0 CPU kernel for the missed experts
(-8.3 ms per step), CUDA graphs off for this serve (-3.7 ms), the MoE doorbell
(`LLAMA_MOE_DOORBELL=1`: each layer's cache misses go to a host executor while the
GPU computes the hits, instead of splitting the graph; -19.5 ms), and the prefill
stream ring (`GGML_SCHED_PREFILL_STREAM=1`) with `-ub 4096 -b 4096` (4x prefill).
Quality: KLD 0.020 against the CPU chain (the float-order class), coherent text.
Since 2026-10-01 (TASKS #156) the decode graph is reused instead of rebuilt every
step, which makes CUDA graphs a win again (they are back ON), and the sparse QSA
attention got cheaper at depth (one shared input for its 12 layers, an exact skip
of masked positions in the Volta attention kernel). Production config, same box:
**55.7 ms per MTP step (47.6 t/s, -13 %)** on short prompts, **71.8 ms at 28k and
107.8 ms at ~95k context (-19 %)**, prefill unchanged at ~334 t/s; every piece is
exact (KLD 0 against the code before it). Later the same day the QSA indexer
learned to rank whole blocks, as the reference model does (`LLAMA_QSA_BLOCK_TOPK=1`,
now on in production): **66.3 ms per step at 28k (-7.5 %) and 89.5 ms at ~95k
(-17 %, 24.8 -> 31.1 t/s)**, short prompts and prefill unchanged; this one changes
the selection at the cutoff (KLD 0.0046, perplexity equal at 8k / 32k / ~95k,
17/17 exact-answer probes).
The exact flags and envs are under Usage examples in
[`docs/env-gates.md`](docs/env-gates.md).

## Earlier benchmarks (Qwen3.6, 2x V100 NVLink box, July 2026) — single stream, max t/s per config

**2× V100-SXM2-32GB (NVLink)**, temp-0, `-n 128`, single stream, image built from this
tree. **MTP** = the model's built-in multi-token-prediction head used for self-speculation
(`--spec-type draft-mtp`). Numbers are decode (tg) / prefill (pp) t/s. Every change is
gated on byte-identical / PPL-neutral temp-0 output vs. baseline. Commands to reproduce
are below the tables.

**Qwen3.6-27B** — dense, MTP head (17.9 GB):

| Configuration | Prefill | **Decode** | | Configuration | Prefill | **Decode** |
|---|---:|---:|---|---|---:|---:|
| CPU (20t) | 10.1 | 1.4 | | 1 GPU + MTP | 106.6 | **51** ¹ |
| CPU + MTP | 9.7 | 3.4 | | 2 GPU layer + MTP | 100.9 | 52.6 |
| 1 GPU | 115.6 | 31.5 | | 2 GPU tensor | 100.5 | 48.8 |
| 2 GPU layer | 113.3 | 32.8 | | **2 GPU tensor + MTP** | 89.3 | **81** ¹ |

**Qwen3.6-35B-A3B** — MoE (3B active), MTP head:

| Configuration | Prefill | **Decode** | | Configuration | Prefill | **Decode** |
|---|---:|---:|---|---|---:|---:|
| CPU (20t) | 35.7 | 7.8 | | 1 GPU + MTP | 159.2 | **140** ¹ |
| CPU + MTP | 32.9 | **14** ¹ | | 2 GPU layer + MTP | 175.7 | 133.1 |
| 1 GPU | 203.3 | 100.9 | | 2 GPU tensor | 138.0 | 110.4 |
| 2 GPU layer | 200.3 | 100.1 | | **2 GPU tensor + MTP** | 122.8 | **165** ¹ |

**GLM-4.7-Flash-UD** — MoE / MLA (deepseek2 arch), no MTP head:

| Configuration | Prefill | **Decode** |
|---|---:|---:|
| CPU (20t) | 35.4 | 9.5 |
| 1 GPU | 185.0 | 89.5 |

Takeaways: MTP adds ~+40-50% (dense) up to ~+35% (MoE); tensor-split helps the dense 27B
(32.8 → 48.8) more than the already-fast MoE; the MoE + MTP stack holds **~166 t/s**
single stream, the dense 27B **~81 t/s** (server steady-state). Concurrency multiplies
throughput: 35B at **`-np 4` = ~263 t/s aggregate** (4 streams × ~69 t/s each, 24.2k
tokens / 92 s wall, 86-89% draft acceptance — measured on the same compose with `-np 4`
and four simultaneous code-gen requests).

¹ MTP rows marked ¹ are the **production server steady-state** (`docker-compose.mtp.yml`,
full config + env gates, long generation, ~88.6% MTP draft acceptance, mean accepted len
3.66), read from `docker logs`: **35B tensor ~166** (peak `tg_3s` 174), **35B 1-GPU ~140**
(peak 148), **35B CPU ~14** (vs cli 9.9), **27B tensor ~81** (peak 87), **27B 1-GPU ~51** (peak 53). Unmarked MTP rows are `llama-cli` short-run
figures that measure the cold→warm ramp (tg climbs for the first ~500 tokens), so they
read several t/s low vs. the warmed server steady-state.

**Reproduce with [`docker-compose.mtp.yml`](docker-compose.mtp.yml)** (the production
server config). The **MTP (¹) numbers must come from the server, not `llama-cli`** —
speculation has a warmup ramp (draft-acceptance EMA + graph cache) that short cli runs
never leave, so cli reads several t/s low; the server's steady-state `tg` is the real
figure. Non-MTP numbers match cli either way.

The compose defaults to **2 GPU tensor + MTP, 27B** (`tg` peaks at the table value).
Edit it per scenario, bring it up, send one long request, and read the held `tg`:

```bash
# 1) edit docker-compose.mtp.yml for the scenario:
#    model     : -m /models/Qwen3.6-27B-...   -> swap to the 35B-A3B gguf
#    CPU       : CUDA_VISIBLE_DEVICES=  (empty) ; command: -ngl 0  (drop -sm tensor / -ts)
#    1 GPU     : CUDA_VISIBLE_DEVICES=0        ; command: drop -sm tensor / -ts
#    2 GPU lyr : command: -sm layer            (instead of -sm tensor -ts 0.5,0.5)
#    no MTP    : delete the --spec-type / --spec-draft-* command lines
docker compose -f docker-compose.mtp.yml up -d

# 2) warm it up: one long generation (~1-2k tokens) so MTP + the graph cache settle
curl -s localhost:8097/v1/chat/completions -H 'content-type: application/json' -d \
  '{"messages":[{"role":"user","content":"Write a complete Flappy Bird clone in Python with pygame."}],"max_tokens":2000,"temperature":0}' >/dev/null

# 3) read the held steady-state (tg / tg_3s AFTER the first ~500-token ramp; and the
#    final `eval time = … tokens per second` + `draft acceptance` summary)
docker compose -f docker-compose.mtp.yml logs 2>&1 | grep -E "print_timing"
```

Other measured highlights: KV cache q8_0 = **0.5×** memory (lossless), 27B on a remote
2-GPU RPC TP island **33 t/s** (loopback), **real cross-host inference at ~2-4% network
tax** (whole model on a GPU-less worker box over the LAN — see Distributed inference),
and DeepSeek-V4 **81 GB streamed from SSD on one 32 GB V100** (SSD-streaming section).

## What this fork adds over upstream

### Backend-agnostic improvements (not V100-specific)

These live in backend-neutral code (`common/`, `src/llama-*`, `tools/server/`),
so they help speculative decoding, hybrid/recurrent models, and long-context
serving on **any** ggml backend — CPU, CUDA, Metal, ROCm, Vulkan, RPC — not
just CUDA/Volta. They were tuned and measured here on V100s, but nothing about
them is V100-specific.

- **Multi-shape decode graph cache** (`src/llama-context`) — an LRU of decode
  graphs keyed by ubatch shape, each with its own scheduler, so a workload that
  alternates between a few small shapes (speculative draft/verify, varying
  concurrency) replays cached graphs instead of rebuilding **and reallocating**
  every step. 100% steady-state reuse; build+alloc drops to ~0 ms. Env:
  `LLAMA_DECODE_GRAPH_CACHE` (entries, default 4), `LLAMA_DECODE_GRAPH_CACHE_TOKENS`
  (max cached ubatch size). (`GGML_META_MAX_GRAPHS` is the CUDA-tensor-split
  companion knob.)
- **Speculative draft padding** (`common/speculative`) — drafts padded to a
  fixed length so the verify batch shape stays constant across steps; this is
  what lets graph reuse and confidence-gated drafting (`--spec-draft-p-min`)
  actually pay off instead of forcing a rebuild every iteration. Kill switch:
  `LLAMA_SPEC_DRAFT_NO_PAD=1`.
- **Full-sequence equal ubatch splits** (`src/llama-batch`) — hybrid recurrent
  models (e.g. DeltaNet) no longer collapse to one-sequence-per-ubatch under
  speculative rollback; `split_equal(..., full_seqs=true)` groups whole
  sequences while preserving the rollback-snapshot invariant (the 64.7 → 125 t/s
  concurrent-spec fix).
- **Server prompt-batch defragmentation** (`tools/server`) — prompt processing
  breaks the batch only where a context checkpoint is actually created, not at
  every user message (large win for long chat-history prefill).
- **Prompt-cache checkpoint pruning + RAM bounds** (`tools/server`) — cached
  conversations keep only their newest checkpoints (hybrid-model checkpoints are
  ~150–230 MiB each regardless of token count), preventing multi-GiB cache
  entries and host OOM.
- **Adaptive speculative draft cap** (`common/speculative`, opt-in
  `LLAMA_SPEC_ADAPTIVE=1`) — caps each draft round near the measured acceptance
  EMA to skip cold draft passes. Measured tg-neutral on the MTP config here (the
  confidence gate already captures the value), so it ships **off by default**.
- **MoE expert cache** (`src/llama-moe-cache`, `--moe-cache off|N|auto`,
  TASKS #151) — for `-ncmoe` serves, the hottest CPU-resident routed experts
  are kept in per-layer VRAM slot pools; a second GPU `mul_mat_id` chain
  computes the cached lanes while the stock CPU chain computes the misses
  (skip sentinel), and a miss is never fetched on the current token's path.
  Async fill worker, LRU + heat eviction, budget clamped to free VRAM minus a
  reserve. Measured on Qwen3.8-Flash-Next, 1x V100: +23-38% decode at 12-18 GB
  of slots, quality inside the batch-width noise floor. Design + gate log:
  [`docs/moe-cache-plan.md`](docs/moe-cache-plan.md).
- **Qwen3.8-Flash-Next (`qwen4exp`) support** (`src/models/qwen4exp.cpp`,
  port of upstream PR 27742 audited against the transformers reference,
  TASKS #143) — GDN + QSA + per-layer embeddings (PLE) + hyper-connections +
  MRoPE; the converter's `--mtp` exports the MTP head as its own GGUF for
  `draft-mtp`, and `LLAMA_MMAP_RANDOM=1` advises the ~97 GiB PLE gather
  table for random access instead of pulling it in at load.
- **AVX2 Q2_0 dot kernel** (`ggml/src/ggml-cpu/arch/x86/quants.c`, TASKS #154
  item 1) - x86 had only the scalar Q2_0 kernel; the AVX2 one is ~6x faster on
  one thread and reaches the memory wall at 40 threads. Flash-Next keeps 43 % of
  its expert weights in Q2_0: -8.3 ms per MTP step. The float summation order
  differs from the scalar kernel (KLD 0.017, the noise class).
- **Robustness fixes** — clean failure on unreachable `--rpc` endpoints (was a
  silent CPU fallback), on failed context/lora init (was a null-pointer crash),
  and a lora-path double-free.
- **Env-gated diagnostics** (zero cost when unset): `LLAMA_DECODE_TIMING`
  (per-ubatch build/alloc/set/compute + reuse), `LLAMA_SPEC_TIMING`
  (draft/checkpoint/decode/accept phases).

### CUDA / Volta (V100) specific
- **One-shot P2P NVLink AllReduce** for 2-GPU tensor mode
  (`GGML_CUDA_ALLREDUCE=p2p`, NCCL fallback, cap `GGML_CUDA_AR_P2P_MAX_BYTES`).
- **MMVQ sm70 tuning** — a dedicated Volta parameter table for the quantized
  matrix-vector kernels; K-quant batch-1 decode uses `nwarps=2` (+1.8% nospec
  tg, perplexity-identical). Volta was previously served by the generic
  (untuned) path.
- **Small-batch tensor-core matmul route** (`ggml-cuda/mmsmt.cu`, TASKS #153 T1,
  default on, `GGML_CUDA_SMT=0` disables) - quantized weights times 3-8
  activation rows (MTP verify batches) on `mma.sync.m8n8k4` instead of MMVQ /
  dp4a MMQ: 1.17-1.34x at widths 4-8, quality-neutral (KL gate). Design:
  [`docs/mmsmt-implementation.md`](docs/mmsmt-implementation.md).
- **Small-width flash-attention kernel** (`ggml-cuda/fattn-mma-volta-small.cuh`,
  T3, default on, `GGML_CUDA_FA_NO_VOLTA_SMALL=1` disables) - decode and verify
  attention (up to 8 query rows, f16 or q8_0 KV) on the tensor cores at the HBM
  rate: decode at 128k depth +21 % (f16) / +60 % (q8_0), MTP decode at 56k
  +19-23 %. Plan + gates: [`docs/ninfer-t3-t4-plan.md`](docs/ninfer-t3-t4-plan.md).
- **MoE doorbell + prefill stream ring** (TASKS #154, env-gated, off by default)
  - for `-ncmoe` serves with `--moe-cache`: `LLAMA_MOE_DOORBELL=1` keeps the
  decode graph on the GPU and hands each layer's cache misses to a host
  executor through a pinned mailbox (Flash-Next, together with the AVX2 kernel
  and graphs off: 95 -> 64 ms per MTP step); `GGML_SCHED_PREFILL_STREAM=1`
  copies the host-resident expert weights of large prompt ubatches on a helper
  thread and its own stream (4.1x prefill with `-ub 4096`). Plan + gates:
  [`docs/strata-port-plan.md`](docs/strata-port-plan.md).
- **Quantized KV in tensor mode** — verified lossless at q8_0 (2x); mixed K/V
  types enabled via `GGML_CUDA_FA_ALL_QUANTS`. Note (measured): on Volta,
  quantized KV is for **capacity**, not *speed*. Before the small-width
  attention kernel (T3), quantizing KV made long contexts slower; since then
  q8_0 runs within ~2 % of f16 at depth.
- **MLA tensor mode** (`deepseek2` family: GLM-4.7-Flash, DeepSeek V2/V3/R1,
  Kimi K2) — attention runs mirrored, FFN/experts split; validated by
  perplexity (statistically identical to single-GPU). Temp-0 text can diverge
  from single-GPU runs (MoE-router-amplified reduction noise) without affecting
  quality. Single GPU remains fastest when the model fits.
- **Meta tensor-split backend diagnostics**: `GGML_META_DEBUG=1|2` (split-state
  + per-src resolution), `GGML_META_DEBUG_REDUCE` (AllReduce boundary
  placement), `GGML_META_TIMING` (compute-vs-reduce wall-time attribution),
  `LLAMA_DEBUG_DUMP_DIR`/`_FILTER` (full-tensor eval-callback dumps),
  `LLAMA_META_DUP_DEVICE` (genuine n-way tensor splits on one GPU).

### Distributed inference (experimental)

**Validated on real hardware (2026-07-09)** — first cross-host inference, a
GPU-less CPU worker box serving whole models over the LAN (0.15 ms RTT),
decode t/s:

| Model (whole model on the remote worker) | loopback | cross-network | tax |
|---|---:|---:|---|
| Qwen3-0.6B | 20.9 | 19.9-20.1 | ~4% |
| Qwen3.6-35B-A3B (MoE) | 7.7 | 7.4-7.6 | ~2-4% |

The model file lives only on the coordinator: the worker's share of weights
streams once (23 GB ≈ 178 s at GbE) and is hash-cached on the worker's disk —
warm reloads skip the transfer entirely; per-token traffic is just KB-scale
activations + the logits row. Full lifecycle (with diagram): guide §0.

- **Worker images for any box**: CUDA (`--build-arg CUDA_DOCKER_ARCH=<cc>`),
  **CPU-only** (`cpu.Dockerfile --target rpc-worker`, no CUDA anywhere), and
  **Vulkan** (Intel Arc/iGPU via `/dev/dri`). Compose profiles:
  `docker-compose.rpc-worker{,-cpu,-vulkan}.yml`.
- **TP islands**: `ggml-rpc-server --tensor-parallel` exposes all local GPUs
  as one tensor-parallel device over RPC; the coordinator automatically
  computes and uploads per-tensor split states (weights, KV, recurrent-state
  caches). A 27B hybrid model loads sharded (9.2 GB per island GPU,
  slice-packed allocation) and generates coherently, driven entirely over
  the network; reloads take ~40-50 s with the worker weight cache.
- **State integrity over islands**: prompt checkpoints and slot save/restore
  work across RPC (views are resolved to their root tensors on the wire; a
  501 MB / 5259-token 27B state round-trips with byte-identical generation).
- RPC protocol 4.2: split-state upload, device descriptions, buffer-usage
  mirroring, meta-aware (logical-address) sanitization, async command
  markers + events (scheduler pipeline parallelism engages across RPC
  devices), multi-connection workers with per-connection buffer reclaim,
  and fenced worker-to-worker activation transfers (`GGML_RPC_NO_W2W=1`
  to force the old coordinator-bridged copies).
- Pipeline over RPC measured at ~3%/token protocol cost — cross-machine
  `-sm layer` is practical on ordinary Ethernet.

### `--ssd-streaming` — run MoE models bigger than VRAM **+** RAM

> **Status: working (beta).** Landed and usable via CLI flags / env gates.
> **DeepSeek-V4-Flash 81 GB runs on one 32 GB V100 + 46 GB RAM** — coherent,
> byte-/PPL-neutral vs resident. Decode is IO-bound and depends on your NVMe and
> the expert-cache hit rate (this box's ~1.6–2.7 GB/s drive: ~1.5–2.6 t/s; a
> faster Gen-4/5 drive scales up). Full design + measured results in
> [`docs/ssd-streaming-plan.md §8`](docs/ssd-streaming-plan.md); every knob in
> [`docs/env-gates.md`](docs/env-gates.md).

**Run it** (DeepSeek-81GB on 1 GPU; experts stream to a 30 GB RAM cache + a 14 GB
VRAM slot cache, non-experts on the GPU):

```bash
docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES=0 -e GGML_SSD_STREAM_DEBUG=1 \
  -v /path/to/models:/models:ro --entrypoint /app/llama llamacpp-local-v100:latest cli \
  -m /models/DeepSeek-V4-Flash-...gguf -ngl 99 --no-mmap -c 4096 -n 96 --temp 0 -st -v \
  --ssd-streaming --ssd-stream-budget 30000 \
  --ssd-stream-gpu --ssd-stream-vram-budget 14000 \
  -p "Explain how a CPU pipeline works."
```

`--ssd-streaming` = RAM/SSD expert tier; `--ssd-stream-gpu` adds the VRAM slot cache
(GPU landing, single-GPU). On a fast NVMe, add `-e LLAMA_SSD_STREAM_READ_THREADS=4`
(parallel miss-path reads — a prefill/long-prompt win). `GGML_SSD_STREAM_DEBUG=1`
prints the per-tier hit rates and the miss-path read/H2D time split (needs `-v`).

Or use the **tuned showcase server** — [`docker-compose.ssd.yml`](docker-compose.ssd.yml)
(best-measured VRAM/RAM/read-thread values, all knobs overridable inline; watch
`docker compose logs` for the `GPU cache hit=` line):

```bash
MODELS_DIR=/path/to/models docker compose -f docker-compose.ssd.yml up
```

**The use case.** Today a model has to fit in VRAM, or in VRAM + system RAM
(spilled via `-ngl` / CPU offload). When it doesn't fit *even in VRAM + RAM
combined*, you're stuck — mmap demand-paging thrashes the disk and collapses
to a fraction of a token/sec. `--ssd-streaming` adds the **SSD as a managed
third tier**: the small always-needed weights live resident (GPU/RAM), and the
bulk of the model is held on NVMe and pulled in on demand. The SSD is the
holding hand that lets a model far larger than your memory actually *run* at a
usable speed.

**Why MoE is the sweet spot (any MoE architecture, not just ours).** A
Mixture-of-Experts model activates only a few experts per token, so per step
you only need to *read* that token's experts — not the whole model. That turns
"stream 80 GB per token" into "stream the few MB of experts this token
actually uses," backed by a hot-expert cache in RAM/VRAM. This is
architecture-agnostic: it targets **any** MoE GGUF (Qwen3-MoE, Mixtral,
GPT-OSS, GLM, DeepSeek, …). **GLM-4.7-Flash and DeepSeek-V4-Flash are our two
first-class targets** (and we're validating on DeepSeek first) — but nothing in
the design is specific to them.

**Measured feasibility (this box: 2× V100 = 64 GB VRAM, 46 GB RAM, one NVMe).**
Reference model: `DeepSeek-V4-Flash-IQ2XXS`, **81 GB** — larger than VRAM *and*
RAM, so it can only run this way. 77.9 GB of it is experts (256 per layer,
6 active/token); 8.8 GB is always-resident. Random O_DIRECT reads on the NVMe
sustain **~2.7 GB/s** (they *bypass* the page cache, avoiding the reclaim trap
that sinks naive prefetching). Projected decode, expert-IO-bound:

| Expert-cache hit rate | Projected t/s | vs. today (mmap thrash) |
|---|---|---|
| 0% (pure stream, no cache) | ~1.5 | ~1.1 |
| ~44% (RAM cache, uniform routing) | ~2.6 | — |
| ~80% (realistic routing skew) | ~7.4 | — |
| ~95% (strong skew) | ~30 → compute-bound | — |

Even a cold pure-stream already beats mmap thrash; a hot-expert cache and real
routing locality multiply from there. Design, numbers, and the reproducible
benchmark (`scripts/ssd-stream-bench-odirect.cpp`) are in
[`docs/ssd-streaming-plan.md`](docs/ssd-streaming-plan.md).

## Fork knobs (env gates)

Most of what this fork adds is **off by default** and gated by an env var (or CLI
flag). The exceptions are measured wins that ship on, each with an opt-out: the
Volta small-batch matmul route (`GGML_CUDA_SMT=0`), the Volta small-width
attention kernel (`GGML_CUDA_FA_NO_VOLTA_SMALL=1`), MTP draft padding
(`LLAMA_SPEC_DRAFT_NO_PAD=1`), the decode graph cache (`LLAMA_DECODE_GRAPH_CACHE=0`)
and tiled RPC weight uploads (`GGML_META_TILED_UPLOAD=0`); the MMVQ sm70 table and
the AVX2 Q2_0 kernel have no switch. The **complete, authoritative list**
— every gate with its type, default, measured effect, and usage examples, plus
every CLI flag the fork adds (section 10) — lives
in [`docs/env-gates.md`](docs/env-gates.md). The tables below are a curated
highlight of the most-used knobs (the full reference also covers the meta
expert-parallel gates, the fleet/discovery/auto-weight family, worker
`--score`/`--allow-shutdown`, surgical re-provision, and the CUDA
containment/fault-injection knobs):

**SSD streaming — CPU tier**

| Env gate | Default | What it does |
|---|---|---|
| `LLAMA_SSD_STREAM_BUFFER` | off | Master switch: stream MoE expert weights from the GGUF on demand (run models bigger than VRAM+RAM). |
| `LLAMA_SSD_STREAM_BUDGET` | 8192 MiB | RAM expert-cache byte budget (LRU-bounded). |
| `LLAMA_SSD_STREAM_READ_THREADS` | 1 | Parallelize SSD miss-path preads (prefill/long-prompt win; decode-neutral). |
| `LLAMA_SSD_STREAM_SLRU` | off | Segmented LRU for the RAM cache instead of plain LRU (measured no-win; opt-in). |
| `LLAMA_SSD_STREAM_PROTECTED_PCT` | 80 | Protected-segment size for the RAM SLRU. |
| `LLAMA_SSD_STREAM_SERIAL` | off | Node-at-a-time execution. **Required for `-ngl 0`** (pure CPU). |
| `LLAMA_SSD_STREAM_NO_ODIRECT` | off | Force buffered reads instead of O_DIRECT (kill-switch / A-B). |
| `GGML_SSD_STREAM_DEBUG` | off | Periodic hit/miss/evict/hit-rate + miss-path timing log. |
| `LLAMA_SSD_STREAMING` | off | Legacy advisory-mmap "residency director" (phase-1 negative result; prefer the streamed buffer). |

**SSD streaming — GPU landing** (VRAM expert-slot cache; single-GPU)

| Env gate | Default | What it does |
|---|---|---|
| `LLAMA_SSD_STREAM_GPU` | off | Compute streamed experts on the GPU via a VRAM slot cache + id→slot indirection. Needs `…_BUFFER=1`. |
| `LLAMA_SSD_STREAM_VRAM_BUDGET` | 4096 MiB | Total VRAM budget for the slot pools. |
| `LLAMA_SSD_STREAM_VRAM_POOLS` | auto | Override the expert-slice class count the budget splits across (auto-detected). |
| `LLAMA_SSD_STREAM_GPU_SLRU` | on | Segmented LRU (scan resistance) for the VRAM cache; `=0` = plain LRU. |
| `LLAMA_SSD_STREAM_GPU_PROTECTED_PCT` | 80 | Protected-segment size for the VRAM SLRU. |
| `LLAMA_SSD_STREAM_GPU_NO_RECLAIM` | off | Kill-switch for the `input_cpy` VRAM reclaim (default shrinks the dead-weight copy). |
| `GGML_OP_OFFLOAD_MIN_BATCH` | 32 | *(upstream)* Min tokens for a `MUL_MAT_ID` to offload; not needed with GPU landing (auto-offloads at batch-1). |

**MoE expert cache** (hot CPU-resident experts in spare VRAM; `-ncmoe` serves)

| Env gate / flag | Default | What it does |
|---|---|---|
| `--moe-cache off\|N\|auto` *(flag)* | off | VRAM budget (MiB) for the expert slot pools; forces `--no-repack`. Env alias `LLAMA_ARG_MOE_CACHE`. |
| `LLAMA_MOE_CACHE_RESERVE_MB` | 3072 (+ draft model) | VRAM kept free of the cache; an oversized fixed budget is clamped with a warning. |
| `LLAMA_MOE_CACHE_MAX_BATCH` | 8 | Widest node the cache chain owns (MTP/ngram verify batches); clamped below `GGML_OP_OFFLOAD_MIN_BATCH`. |
| `LLAMA_MOE_CACHE_STATS` | 0 | Log hit/fill/evict/resident counters every N steps (libllama INFO: `-v`, or `-lv 4` on the server). |
| `LLAMA_MOE_DOORBELL` | off | `=1`: decode/verify graphs hand each layer's cache misses to a host executor instead of splitting the graph (Flash-Next production). Needs `--moe-cache` on a CUDA device. |
| `GGML_SCHED_PREFILL_STREAM` | off | `=1` (3 slots): prompt ubatches of 1024+ tokens copy host-resident expert weights on a helper thread and stream; use with `-ub 4096 -b 4096`. |
| `LLAMA_QSA_BLOCK_TOPK` | off | `=1`: the qwen4exp QSA indexer ranks whole blocks (the reference selection) instead of every cell; Flash-Next -7.5 % per MTP step at 28k, -17 % at ~95k, short prompts equal, KLD 0.0046 vs off (TASKS #156 7.3). |

**Volta kernels** (on by default)

| Env gate | Default | What it does |
|---|---|---|
| `GGML_CUDA_SMT` | 1 | Small-batch tensor-core matmul route for 3-8 activation rows (MTP verify); `=0` falls back to MMVQ/MMQ. |
| `GGML_CUDA_FA_NO_VOLTA_SMALL` | off | `=1` routes decode/verify attention back to the tile/vec kernels instead of the small-width tensor-core kernel. |

**Speculative decoding / MTP**

| Env gate | Default | What it does |
|---|---|---|
| `LLAMA_SPEC_DRAFT_NO_PAD` | off | Kill-switch for MTP draft-length padding (padding keeps batch shape stable — the default win). |
| `LLAMA_SPEC_ADAPTIVE` | off | Cap each draft round by an acceptance EMA (measured tg-neutral). |
| `LLAMA_SPEC_TIMING` | off | Log server draft/verify/accept timings. |
| `--spec-draft-p-min` *(flag)* | — | Confidence gate: stop drafting below this probability. |

**Tensor-parallel AllReduce (2-GPU tensor split)**

| Env gate | Default | What it does |
|---|---|---|
| `GGML_CUDA_ALLREDUCE=p2p` | NCCL | One-shot P2P NVLink AllReduce for 2 GPUs (falls back to NCCL). |
| `GGML_CUDA_AR_P2P_MAX_BYTES` | 4 MB | Size cap above which P2P defers to NCCL. |
| `GGML_CUDA_FORCE_GRAPHS` | - | Removed (upstream enables Volta graphs by default since 601ad05a9); setting it does nothing. |
| `GGML_CUDA_DISABLE_GRAPHS` | off | Disable CUDA graphs — set in the MTP compose (~5% loss with spec shape churn); leave on for plain dense decode and for the Flash-Next serve (graphs ON since #156, -11.6 % per MTP step). |

**Meta tensor-split backend**

| Env gate | Default | What it does |
|---|---|---|
| `GGML_META_MAX_GRAPHS` | 8 | Shadow-container slots (raise if many decode graph shapes are cached). |
| `GGML_META_DEBUG` | off | Split-state diagnostics (`=2` also prints per-source resolution); with expert placement active, `>=1` runs the `EXPERT_AUDIT` ownership counters. |
| `GGML_META_DEBUG_REDUCE` | off | Print AllReduce boundary placement. |
| `GGML_META_TIMING` | off | Per-step compute-vs-reduce timing. |
| `LLAMA_META_DUP_DEVICE` | 1 | Duplicate the device list N× so one GPU runs a genuine N-way split (validation harness). |

**Expert-parallel / hot-expert placement**

| Env gate | Default | What it does |
|---|---|---|
| `LLAMA_META_EP_ONLY` | off | Expert-parallel split shape: segment only routed experts across members, mirror the rest. |
| `LLAMA_META_ATTN_OWNER` | -1 | Dedicate attention/KV to a member; accepts an owner group (`0,1`) with layers interleaved across owners (#70). |
| `LLAMA_EXPERT_PROFILE` | off | Per-layer router expert-frequency profiler to a JSON path (#74; ~1.5 t/s overhead while on). |
| `LLAMA_META_EXPERT_PLACEMENT` | off | Frequency-ranked whole-expert placement from an artifact JSON (#75; unset = uniform behavior). |

**Decode graph cache · Distributed/RPC · KV cache**

| Env gate | Default | What it does |
|---|---|---|
| `LLAMA_DECODE_GRAPH_CACHE` | 4 | Cached small-batch decode graphs for steady-state reuse (`=0` disables). |
| `LLAMA_DECODE_GRAPH_CACHE_TOKENS` | 64 | Max ubatch size eligible for the cache. |
| `GGML_ALLOC_EXACT_PLAN` | on | The graph allocator re-plans unless every tensor has exactly the planned size, so the same graph gets the same memory layout - and the same fused kernels and numbers - in every path (decode cache == main scheduler, bit for bit; TASKS #156 7.4). `=0` = the old "fits" rule. |
| `GGML_RPC_NO_W2W` | off | Disable direct worker-to-worker tensor pull (bridge through the coordinator). |
| `GGML_RDMA_DEV` / `GGML_RDMA_GID` | auto | RDMA device / GID selection for the RPC transport. |
| `LLAMA_ATTN_ROT_DISABLE` | off | Opt out of Hadamard-rotated KV quantization (auto-on when KV is quantized). |

**Diagnostics** (zero cost unless set): `LLAMA_DECODE_TIMING`, `LLAMA_BATCH_DEBUG`,
`LLAMA_DEBUG_DUMP_DIR`/`_FILTER`, `LLAMA_DSV4_COMPRESS_DEBUG`, `GGML_SCHED_DEBUG`, `GGML_RPC_DEBUG`.

## Documentation map

| Doc | Contents |
|---|---|
| [`docs/perf-tuning-v100.md`](docs/perf-tuning-v100.md) | consolidated results, per-change details, Volta facts, deployment plan |
| [`docs/distributed-inference-guide.md`](docs/distributed-inference-guide.md) | how to run coordinator/workers/TP islands |
| [`docs/distributed-inference-plan.md`](docs/distributed-inference-plan.md) | the distributed design rationale |
| [`docs/expert-parallel-plan.md`](docs/expert-parallel-plan.md) | expert-parallel fleet design + gated experiments (task 28) |
| [`docs/expert-profiling.md`](docs/expert-profiling.md) | router-frequency profiling → placement procedure (tasks 74/75) |
| [`docs/expert-placement-plan.md`](docs/expert-placement-plan.md) | hot-expert placement design + validation staircase (task 75) |
| [`docs/architecture-diagrams.md`](docs/architecture-diagrams.md) | mermaid diagrams: placement, splits, EP, caching, fleet |
| [`docs/v4-single-box-benchmark.md`](docs/v4-single-box-benchmark.md) | V4 single-box vs fleet measurements (#54/#65) |
| [`docs/research/`](docs/research/) | #67 research iterations: parallel decoding, horizontal scaling |
| [`docs/validation-playbook.md`](docs/validation-playbook.md) | test scenarios + exact commands used to validate all of this |
| [`docs/ssd-streaming-plan.md`](docs/ssd-streaming-plan.md) | SSD streaming: design, measured results, CPU + GPU-landing tiers (task 15) |
| [`docs/env-gates.md`](docs/env-gates.md) | every fork env gate + CLI flag, grouped, with usage examples |
| [`docs/dev-workflow.md`](docs/dev-workflow.md) | the dev image, how runs/tests are done, correctness gates, the iterative loop |
| [`docs/moe-cache-plan.md`](docs/moe-cache-plan.md) | MoE expert cache: fork survey, dual-chain design, build log + V100/Kepler gates (task 151) |
| [`docs/strata-port-plan.md`](docs/strata-port-plan.md) | Flash-Next decode + prefill on one V100 (task 154): AVX2 Q2_0, MoE doorbell, prefill stream ring, the closed items with numbers, baseline-vs-after tables |
| [`docs/mmsmt-implementation.md`](docs/mmsmt-implementation.md), [`volta-smallt-gemm-plan.md`](docs/volta-smallt-gemm-plan.md) | Volta small-batch tensor-core matmul route (task 153 T1): design, kernel versions, gates |
| [`docs/ninfer-t3-t4-plan.md`](docs/ninfer-t3-t4-plan.md) | Volta small-width flash-attention kernel (task 153 T3) and the launch-fusion census (T4) |
| [`docs/dual-drafters.md`](docs/dual-drafters.md) | speculative drafter roster, dual-drafter dispatch, measured sweet spots (#132-#142) |
| [`docs/launcher-wizard-plan.md`](docs/launcher-wizard-plan.md) | the launch wizard / router UI: design, gate + flag catalogs, increments |
| [`docs/parallel-decoding-plan.md`](docs/parallel-decoding-plan.md) | parallel decoding design (task 127) |
| [`docs/fill-the-bubble-plan.md`](docs/fill-the-bubble-plan.md), [`lp-pair-fusion-plan.md`](docs/lp-pair-fusion-plan.md), [`hot-expert-replication-plan.md`](docs/hot-expert-replication-plan.md) | the three task-71 boundary-cost escape lanes (closed, with the measured negatives) |
| [`research/`](research/) | code reviews, port audits, and engine surveys (qwen4exp port audit, ninfer-v100 survey) |
| [`REBUILD-IMAGE.md`](REBUILD-IMAGE.md) | building the production Docker image |
| [`TASKS.md`](TASKS.md) | full task history with measurements and open items |

## Quick start (single machine, 2x V100)

Build the image and serve with the tuned profile:

```bash
docker build -f .devops/cuda.Dockerfile --target server -t llamacpp-local-v100:latest .

# point the profiles at your GGUF directory (or create llama.cpp/.env with
# MODELS_DIR=... — the compose files default to ./models)
export MODELS_DIR=/path/to/your/ggufs

docker compose -f docker-compose.mtp.yml up -d      # MTP speculation: fastest
# or
docker compose -f docker-compose.nospec.yml up -d   # plain serving
# or
docker compose -f docker-compose-glm-4.7.nospec.yml up -d  # MoE on a single GPU
```

Key flags the profiles use (see the compose files for the full set):
`-sm tensor -ts 0.5,0.5 -fa on` (tensor parallel), `-ctk q8_0 -ctv q8_0`
(lossless half-size KV), `--spec-type draft-mtp --spec-draft-n-max 3
--spec-draft-n-min 1 --spec-draft-p-min 0.75` (speculation),
`--cache-ram 12288 --ctx-checkpoints 8` (bounded prompt-cache RAM).

For multi-machine setups see the
[distributed guide](docs/distributed-inference-guide.md) and
`docker-compose.rpc-worker.yml`.

## Pending items

Tracked in detail in [`TASKS.md`](TASKS.md). Open as of 2026-10-01:

- **Decode speed follow-ups** - from #156: a block-level top-k for the QSA indexer
  (design written, it changes which cells win ties at the cutoff, so it ships
  opt-in after a KLD + speed review), and a look at why the decode-graph cache is
  not bit-identical to the main scheduler. From #153: the next T3 lever (overlap
  the width-5 attention tiles with the memory stream) and T4 launch fusion
  (census + plan only).
- **New models** (#152) - MiMo-V2.6-Flash-RL and GLM-5.3-Flash GSQ-RCO GGUFs:
  analysis done, route decided, waiting for the go.
- **Speculation** - RPC-hosted drafters (#132, in progress), the fused MTP draft
  chain (#140; closed for `qwen4exp` in #154, no net win there), selective tree
  arming (#141), parallel decoding for the over-VRAM models (#127).
- **Distributed serving** - fault tolerance and the fleet UI are done; RPC
  auth/TLS still open (WireGuard covers it in practice); RDMA / fast NICs (#60,
  hardware).
- **SSD streaming** (task 15) - beta, usable via CLI flags; GPU landing is
  single-GPU; `-sm tensor` + DeepSeek crashes at load (parked).
- **Bug queue** - #130 (launcher generations break half-way client-side,
  backlogged), #144 (Flash-Next tensor-split mmproj OOM), #147 (fleet UI roster
  for `qwen4exp`), #149 (wizard drafter matching), and the older #53, #68, #72,
  #73.

## Reference hardware & platform

Developed, measured and validated on two GPU boxes:

| Component | Original box (2026-07/08: the Qwen3.6 tables) | X99 box (2026-09: the Qwen3.8 numbers) |
|---|---|---|
| GPUs | 2x Tesla V100-SXM2-32GB (Volta, cc 7.0) on an SXM2 carrier, NVLink NV2 between the pair (what makes `-sm tensor` + NCCL fast) | 1x Tesla V100-SXM2-32GB (PG500-216), 250 W |
| GPU driver | 580.x (CUDA 13 userspace; images build against CUDA 12.8) | 580.173.02 |
| CPU | Intel Core i9-10850K, 10c/20t @ 3.6 GHz | 2x Intel Xeon E5-2690 v4, 28c/56t (AVX2 + FMA, no AVX-512) |
| RAM | 46 GiB (+ zram swap) — the 27B CPU-side config needs ~11 GiB anonymous memory beyond weights | 251 GB |
| Storage | Models on a consumer NVMe (measured 1.6 GB/s O_DIRECT sequential); bulk HDDs for everything else | local disk + the models share |
| OS / runtime | Linux (LTS kernel 6.18), Docker 29 with the NVIDIA container runtime; serving and dev builds both containerized | same |

The original box now holds a Tesla K80 (the `:kepler` image variant) and runs the
image registry and the dev builds. Network: the distributed numbers include
loopback runs and real cross-host runs over GbE (since 2026-07-09); the original
box and the X99 share a 10 GbE link.

**Linux-only, and deliberately so.** This fork is developed and tested
exclusively on Linux inside Docker. Upstream llama.cpp supports Windows and
macOS, but nothing added here has ever been run there, and there are no
positive expectations that it works: the distributed transport, `O_DIRECT`
I/O paths, the cgroup-based memory-cap testing methodology, and the compose
deployment profiles are all Linux-specific, and the tuning targets
(NVLink SXM2 V100s) barely exist outside Linux servers anyway. If you try it
on Windows, you are on your own — report findings, but expect breakage.

## Provenance and caveats

- Forked from upstream llama.cpp (see git history for the merge base).
- Tuned specifically for Volta (cc 7.0): MMQ/MMVQ dispatch thresholds, the
  small-batch matmul and attention kernels, and the measured curves are
  V100-specific. CUDA graphs work on Volta since the upstream merge, but the MTP
  compose turns them off (measured net loss); the Flash-Next serve runs them since
  #156.
- The RPC protocol has **no authentication or TLS** — private networks only.
- Everything measured on the reference hardware above (the box is named with
  each table); your numbers will vary with interconnect, model, and
  quantization.
