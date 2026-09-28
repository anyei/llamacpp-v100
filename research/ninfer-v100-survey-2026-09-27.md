# NInfer on Tesla V100 (sm_70): what it does, what it measures, what transfers to ggml-cuda

Survey date: 2026-09-27. Read-only research; no GPU code was run. Every number below carries its
config; every claim carries a file path (relative to the clone named in brackets) or a URL.

Clones (shallow, HEAD as of 2026-09-27) under
`/tmp/claude-1000/-home-anyei-server-git-projects-llama-cpp/8734393e-9a79-4d59-967d-f394f47a281e/scratchpad/ninfer/`:

| Clone dir | Repo | HEAD | Role |
|---|---|---|---|
| `Neroued_ninfer` | github.com/Neroued/ninfer (2469 stars, Apache-2.0) | e31bc99 2026-09-26 | UPSTREAM. sm_120a (RTX 5090) only. |
| `geoffwatts_ninfer-v100` | github.com/geoffwatts/ninfer-v100 | b37d0dd 2026-09-14 | The V100 port. Base of every other V100 fork. |
| `dollarwong_ninfer-v100` | github.com/dollarwong/ninfer-v100 (31 stars) | f17d37f 2026-09-15 | geoffwatts b37d0dd + `deploy-v100/` (systemd, scripts, NInfer-vs-llama.cpp bench). Zero engine changes (`diff -rq` shows only `deploy-v100/`). |
| `huangserva_ninfer-v100-tpx` | github.com/huangserva/ninfer-v100-tpx | 6ffe1c7 2026-09-26 | geoffwatts b37d0dd + rewritten INT8 decode-attention kernel + prefill-attention retune + 2-GPU tensor parallel. |
| `ww485000_ninfer-windows-v100` | github.com/ww485000/ninfer-windows-v100 | 5a23a41 2026-09-26 | geoffwatts + MSVC/vcpkg Windows build, v3 artifact reader. |
| `JimmyMax_ninfer-v100` | github.com/JimmyMax/ninfer-v100 | 0cb6b6f 2026-09-27 | fork of liujun-7788/ninfer-v3-v100 (fork of geoffwatts): v2+v3 artifacts, `nvfp4full` (all-64-layer NVFP4 MLP) binder probe, Docker, real-HTTP benchmark tables. |

Also consulted (GitHub API / raw only, not cloned): geoffwatts PR #10, #11 and issue #13 by
user Flo5k5 (the most detailed V100 profiling write-up in the ecosystem), dnv2003/v100-skinny
(origin of the NVFP4 "QPN" kernel; MIT for `kernels/`), 1CatAI/1Cat-vLLM (Apache-2.0; origin of the
e2m1 shift decode), plus1998/NInfer-V100-Duo and ValerioDolci/ninfer-tp2 (TP references cited
by huangserva).

Important context: the fork lineage is geoffwatts -> everyone else, and geoffwatts forked an
OLDER upstream (it still has `src/targets/qwen3_6_27b/`; upstream HEAD moved to
`src/models/qwen3_5/`, v3 artifacts and a 5-model registry). `diff -rq Neroued_ninfer
geoffwatts_ninfer-v100` = 386 differing + 476 only-in-one-side files, so "fork vs upstream" diffs
mix Volta work with upstream drift; fork-vs-geoffwatts diffs are clean (see
`scratchpad/diff_geoff_vs_*.txt`).

---

## 1. Engine architecture

### 1.1 Language / runtime / product shape
- From-scratch C++20 + CUDA, CMake/Ninja, no Python at runtime (Python only in `tools/convert`,
  `tools/bench`, `eval/`). Vendored deps: cpp-httplib, nlohmann, spdlog, utf8proc, llama-jinja,
  and on the V100 port `third_party/llama_cpp_fattn/` [geoffwatts: `third_party/`].
- Two binaries: `ninfer` (CLI) and `ninfer-serve` (OpenAI Chat/Responses + Anthropic Messages,
  streaming, tools) [Neroued: `README.md`, `docs/serving.md`].
- Deliberately narrow: one GPU, one resident model, startup-fixed `--max-concurrency 1..8`,
  bounded FIFO, no preemption, no weight offload, no multi-GPU (upstream)
  [Neroued: `README.md` "Capabilities and limits"; `docs/maintainer/engine-architecture.md` s.1].
- Model shapes are compile-time constants per registered "target": `hidden=5120, layers=64,
  intermediate=17408, q_heads=24, kv_heads=4, head_dim=256, rotary_dim=64, gdn_key_heads=16,
  gdn_value_heads=48, gdn head dims 128, conv kernel 4, output_rows=248320 (padded vocab),
  mtp_layers=1`; 16 full-attention + 48 GDN layers (`static_assert`)
  [geoffwatts: `src/targets/qwen3_6_27b/impl/config.h:13-69`; Neroued:
  `docs/maintainer/qwen3_5-model.md` "Current official geometries"].
  There is NO generic graph interpreter: the layer schedule is hand-written C++
  (`run_layers` -> `attn_mix` / `gdn_mix` / `mlp_tail`) [geoffwatts:
  `src/targets/qwen3_6/impl/runtime/text_context_impl.h:818-1060`].

### 1.2 Model loading and the artifact format
- `.ninfer` container, v2 (magic `NInfer\0\2`) or v3 (magic `NINFER\x00\x03`, 32-byte header with
  UUID, model/weight decoupling; introduced upstream at f76e19c0) [JimmyMax: `README.md` head;
  Neroued: `docs/maintainer/artifact-container.md`]. geoffwatts b37d0dd reads v2 only; v3 support
  on Volta came from PR #10 (Flo5k5) and the liujun-7788 / JimmyMax and ww485000 forks
  (`src/artifact/reader.cpp`, `binder.h` differ vs geoffwatts).
- Registered persistent numeric formats (exactly nine): `bf16, fp32, int32`; grouped ints
  `q4_g64_fp16 (4.25 b/w), q5_g64_fp16, q6_g64_fp16, q8_g32_fp16`; block-scaled `nvfp4`
  (E2M1 codes, K16 groups, one E4M3FN scale/group, one FP32 global divisor per tensor);
  row-scaled `fp8_e4m3fn_row_bf16` [Neroued: `docs/maintainer/tensor-formats.md` s.1].
  Layouts: `contiguous_le_v1`, `row_split_k128_v1` (grouped ints), `block_scale_k16_m128x4_v1`
  (nvfp4, swizzled scale plane), `row_scale_v1` [`docs/maintainer/storage-layouts.md`].
- The official Qwen3.8-27B `nvfp4` artifact (22.09 GiB) is MIXED: NVFP4 for MLP of layers 0-55;
  row-scaled FP8 for token embedding, attention in/out projections, all GDN projections, the
  full LM head, and the MLP of layers 56-63 (112 NVFP4 tensors, 146 FP8 tensors); weights are
  imported from unsloth/Qwen3.8-27B-NVFP4 without requantization
  [geoffwatts: `model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md:117-141`].
  The `groupwise-int` artifact (19.03 GiB) is a Q4/Q5/Q6 mix for the body
  [`model-cards/Qwen3.8-27B-NInfer/README.md:113,121`]. "nvfp4full" community artifacts put NVFP4
  in all 64 MLPs; JimmyMax's `Binder::peek` probes per-tensor format so one binary loads both
  [JimmyMax: `README.md` head, `src/targets/qwen3_6_27b/impl/load/bindings.cpp`].
- Device-side load-time repack on Volta only: NVFP4 MLP gate/up/down and FP8 payloads are
  permuted IN PLACE on the GPU into "QPN fragment order" (`prepack_qpn_kernel`), same allocation
  size, scratch freed before inference, recorded as `VoltaQpnPrepacked`; the file is untouched
  [geoffwatts: `src/ops/linear/nvfp4/nvfp4_prepack_sm70.cu:18-70`; `docs/v100.md:20-32`;
  `README.md` "On Volta, loading an NVFP4 artifact repacks..."].

### 1.3 How 4-bit weights run on Volta (no FP4/INT4 tensor cores)
Software W4A16 on `mma.sync.m8n8k4.f16.f16.f32`, the only tensor-core instruction Volta has
(no ldmatrix, no m16n8k16, no cp.async, no bf16/int8 MMA) [huangserva: `README.md` "What changed";
geoffwatts: `src/ops/common/volta_mma.cuh:1-20`].

- e2m1 -> fp16 by bit shifts (sign into bit 15, the 3 E/M bits into fp16 lanes; the value comes
  out scaled by 2^-14 and is rebiased with one `__hmul2` by a hoisted 2^14 constant; the 2^14
  cannot be folded into the group scale because `gscale*2^14` overflows fp16 on real outliers).
  e4m3 group scale -> fp16 by shift (x2^-8, folded into the tensor's global divisor). Decode from
  TurboMind / 1Cat-vLLM (Apache-2.0) [geoffwatts: `src/ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh:1-90`].
  No lookup tables.
- "QPN" (quadpair-split-N) mapping: m8n8k4 is issued per quadpair (4 independent 8x8x4 MMAs
  per warp instruction). The four quadpairs split N (each owns 8 output rows) and share one 8xK
  activation tile, so the warp tile is 8 tokens x 32 output rows. A-row utilisation at T=4/8:
  50%/100% instead of 12%/25% for the naive 32x8 mapping; activation traffic per weight byte
  drops 4x. The 4 (or SPLITK=4/8/16) warps of a CTA split K, no shared memory or barrier in the
  main loop (weights stream global->register; activations are L1/L2 resident), one
  `__syncthreads()` cross-warp K reduction at the end (replaces split-K workspace/atomics)
  [`src/ops/linear/q4/q4_volta_qpn_gemm.cuh:1-60`; `nvfp4_volta_qpn_gemm.cuh:98-140`].
  Siblings for Q4, W8, FP8, NVFP4 (`*_volta_qpn_gemm.cuh`). Q4's stored nibble order already IS
  the B-fragment order (free); NVFP4's shift decoder emits (k,k+4) pairs, fixed by applying the
  same permutation to the activations with two `__byte_perm` (register-only) instead of
  repacking weights [`nvfp4_volta_qpn_gemm.cuh:29-45`].
- Group-scale cadence: one fp16 scale register per 16 MMA slices (Q4 K64) or per 4 slices (NVFP4
  K16, reading four consecutive scales with one aligned 4-byte load from the swizzled plane).
- Activations: bf16 in the base port -> per-lane bf16->fp16 conversion inside the loop (4x
  redundant across quadpair siblings); this was measured as a limiter (Volta's F2F pipe is
  quarter rate) and fixed by fp16 staging / fp16 producers in huangserva and in Flo5k5's PR #11
  [huangserva: `CHANGES.md` commit 53f65504; PR #11 body "fp16 activations end to end"].
- Reported kernel efficiency (v100-skinny origin, not re-measured here): "QPN2 holds 71% of read
  roofline at M=8"; NACC/SPLITK sweep "441 -> 637 GB/s at M=8"
  [nvfp4_volta_qpn_gemm.cuh:100-105; https://github.com/dnv2003/v100-skinny README].
- Dispatch by token width T (Volta): T <= `kNvfp4VoltaQpnMaxTokens` -> QPN (chunked); wider ->
  `nvfp4_volta_mma_gemm` (dequant into SHARED memory then m8n8k4, split-K with fp32 workspace;
  T=2048 gate_up 45.5 -> 31.3 ms vs chunked QPN) or the CUTLASS sm70 routes
  (`*_cutlass_sm70.cu`) which dequantise the whole weight to an FP16 GLOBAL scratch per call
  [`src/ops/linear/nvfp4/nvfp4_dispatch.cpp:45-95`; `q4_volta_mma_gemm.cuh:1-40`].

### 1.4 KV cache
- Paged, 64-token pages, typed homogeneous pools, block tables, Device/Host replicas, prefix
  checkpoints hold KV + full GDN continuation state [Neroued: `docs/maintainer/paged-kv-cache.md`
  s.1-2; `README.md` "Resource-aware long-context reuse"].
- Dtypes on Volta: `bf16` (1024 B/token/kv-head, K+V), `int8` group-64 (528 B), `fp8` (516 B,
  measured SLOWER than int8 on V100 under real load); `nvfp4`/`k8v4` KV are rejected at planning
  on Volta [JimmyMax: `docs/V100-BUILD.md` "KV dtype guide"; geoffwatts: `docs/v100.md:34-35`].
- INT8 codec: per-token, per-64-group absmax int8 with fp16 scale, head dim 256 -> 4 scales;
  a normalized D256 Sylvester-Hadamard rotation is applied to K at append and to Q at query
  time (V not rotated) to flatten outliers before quantisation
  [`src/ops/kv_cache/int8_g64_codec.cuh:1-50`; `hadamard_d256.cuh`; `docs/v100.md:22-24`;
  `prompt_volta.cuh:55-70`].
- Size for Qwen3.8-27B: only 16 of 64 layers have KV: 16 x 4 heads x 528 B = 33.8 KB/token ->
  262,144 tokens = 8.5 GB int8. GDN state per sequence: 48 layers x 48 heads x 128x128 fp32 =
  151 MB (+ tiny conv state). This is why 240k-262k contexts fit in 32 GB next to ~20 GB of
  weights [config.h above; JimmyMax launch log "KV 215,040 tokens, int8 ... runtime 9.33 GiB"].

### 1.5 Attention kernels for sm_70 (and what huangserva changed)
Base port [geoffwatts]:
- Decode/verify ("small-T", T <= 8 rows per kv head incl. GQA group): split-KV partial kernels
  on m8n8k4 with fragment maps transcribed from ggml-cuda/mma.cuh: `small_t_bf16_volta.cuh`,
  `small_t_i8_volta.cuh` (int8 dequantised through registers into an fp16 smem tile; bf16 TC
  beat int8 SIMT by 1.46x "while reading twice the bytes", so the TC path was rebuilt for int8),
  shared reduce kernel for the split partials, ~480 keys/split heuristic for long windows
  [`src/ops/softmax_attention/dense/causal_cache/small_t_i8_volta.cuh:1-40`;
  `small_t.cuh:80-95`].
- Prefill: the VENDORED llama.cpp `fattn-mma-f16.cuh` (pinned at llama.cpp commit 62bf73d25,
  2026-08-10, MIT; `mma.cuh` and `cp-async.cuh` byte-for-byte, `fattn-stream-k.cuh` extracted
  from fattn-common.cuh, a 280-line `common.cuh` shim), driven by
  `src/ops/launcher/gqa_attention_volta_flash.cu`: per layer it gathers the visible key range
  from the paged bf16/int8 cache into a contiguous FP16 buffer, converts Q to FP32, runs the
  kernel with ncols2=2 (GQA ratio 6), ncols1=16, converts FP32 out to BF16
  [`third_party/llama_cpp_fattn/README.md`; `gqa_attention_volta_flash.cu:1-90`].
- `prompt_volta.cuh`: correctness-first SIMT fallback (one CTA per token-head).
huangserva rewrite [`src/ops/softmax_attention/dense/causal_cache/small_t_i8_volta_v2.cuh:1-60`,
`README.md`, `CHANGES.md`]:
- Profiling at 186K keys, T=4 showed the original at ~140 GB/s (393 MB of int8 K+V per layer
  call in 2.79 ms); the cost was the per-tile pipeline (all 4 warps recomputed the full QK^T
  and exp for the same 16 keys, 2 barriers per 16 keys, small loads), not one resource.
- v2: 8 warps, Bc=64 keys per smem tile, warp w does QK^T only for its own 8 keys, row maxima
  and P exchanged through smem, each warp does PV over all 64 keys for its own D/8=32 columns
  (half the accumulator registers), 3 barriers per 64 keys, next K/V tile prefetched to
  registers, int8->fp16 via PRMT byte-permute (Volta's cvt is slow), PV in fp32 via
  `m8n8k4.f32.f16.f16.f32` (removes 8 cvt + 8 FADD per group, ~10% at 186K). Result 2.79 ->
  1.29 ms per call (~305 GB/s). Env `NINFER_SM70_ATTN_V2=0` restores the original.
- Prefill flash kernel retuned: ncols1 16 -> 32 (64-column Q tile), 131K context: 1024-token
  chunk 119 -> 88.4 ms [`CHANGES.md` 11339e9a; diff of `gqa_attention_volta_flash.cu`].
- Split-merge kernel 29 -> 13 us; `NINFER_SM70_LONG_SPLIT_KEYS` 480 -> 1920 cuts the 186K/T=4
  kernel 1402 -> 1130 us (single GPU; not default) [`README.md` runtime switches].
- Remaining wall at 186K: attention 10.5 ms of a ~33 ms round; "Volta can only keep one thread
  block per SM resident" for this kernel; several further rewrites only gave 3-14%.
Independent rewrite by Flo5k5 (PR #11, not merged into geoffwatts master as of b37d0dd; the PR
is closed): split-K QK^T across dim-split warps, wave-aligned split count (80 SMs x 2 CTAs),
magic-bias int8 dequant, fp32 PV with lazy rescale, then a key-major kernel; int8 small-T at
26k context: W=1 264 -> 93 us, W=5 290 -> 128 us, W=8 554 -> 207 us per layer call
[https://github.com/geoffwatts/ninfer-v100/pull/11].

### 1.6 Decode loop
- Exact-batch CUDA graphs: one graph per legal batch topology (B rows, spec window); request
  identity and page IDs are graph INPUTS, not keys [Neroued: `docs/maintainer/engine-architecture.md`
  s.8; `src/core/decode_graph.cpp` (stream capture, ThreadLocal mode)].
- The whole MTP round is inside one graph: target verify at width K+1, K sequential draft steps
  through the MTP layer, on-GPU acceptance (`speculative_round.cuh`, one warp per request),
  on-GPU sampling (`sampling.cuh`: exact argmax, temperature/top-k 20/top-p/min-p/penalties),
  and `mtp_prepare_next_round_kernel` for the next round; ninfer_bench reports
  `decode_path=mtp_cuda_graph graph_prime=<2K+3> outputs`. No host sync inside a round
  [dollarwong: `deploy-v100/bench/results/nvfp4-official-sweep.txt`; geoffwatts:
  `src/ops/kernel/{sampling,speculative_round,mtp_round}.cuh`].
- Per-layer op sequence (attention layer): rmsnorm -> `attn_input_proj` (ONE fused
  [Q6144|K1024|gate6144|V1024] x 5120 GEMM) -> q-rmsnorm, k-rmsnorm, rope -> attention
  (append + split partials + reduce) -> sigmoid_mul -> `linear_add` (o_proj + residual fused) ->
  rmsnorm -> `linear_swiglu` (gate/up + SiLU*mul in the epilogue) -> `linear_add` (down +
  residual). GDN layer: `gdn_norm_gating_proj` (rmsnorm + a/b gating proj fused) ->
  `gdn_input_proj_conv_snapshot` ([q2048|k2048|v6144|z6144] x 5120 GEMM + causal conv1d + SiLU +
  conv-state update fused) -> `gated_delta_net_batch_update` (one launch, state in registers,
  sequential over the verify width) -> gated_rmsnorm -> `linear_add` -> mlp (3)
  [`text_context_impl.h:818-1010`; `src/targets/qwen3_6_27b/impl/variant.cpp:171-331`;
  huangserva `docs/tp2-design-notes.md`].
- Launch count: about 1255 kernel launches per MTP step (K=4) on the base port, 726 after PR
  #11's fusions (one-pass 5120-wide RMSNorm, fused GDN norm+gating, q/k-RMSNorm+RoPE fused,
  SwiGLU/residual epilogues) [PR #11 body]. PDL is a no-op on sm_70 (`src/core/pdl.cuh`).
- Ordinary (no-spec) decode is a separate graph; on V100 it measures 29.3-30.7 tok/s
  (pp2048+tg256, see 2.1).

### 1.7 Prefill path
- Chunked, eager (not graph-captured), one prefill at a time (concurrent requests queue; 503
  after `--pending-timeout-ms`), `--prefill-chunk` default 1024, 2048 recommended on V100
  (8K prefill 1078.6 -> 1137.0 tok/s, +5.4%) [`docs/v100.md` last section; JimmyMax
  `docs/V100-BUILD.md` "Prefill is serialized"].
- GEMMs: CUTLASS sm70 HMMA after dequant to FP16 global scratch, or the smem-dequant
  `*_volta_mma_gemm` kernels (1.3-1.5x better than chunked QPN at T=2048) [1.3].
- GDN prefill on Volta = the sequential recurrent kernel for the whole width (`T_full = 0`): the
  chunked/matmul GDN formulation needs Ampere+ mma and is trap-stubbed on sm_70. The comment
  notes llama.cpp's own ggml-cuda GDN kernel also has no chunked path, and 1Cat-vLLM profiling
  puts GDN at single-digit % of SM70 prefill [`src/ops/linear_attention/gated_delta_net/gated_delta_net.cpp:249-266`].
- Attention prefill = vendored llama.cpp MMA kernel (1.5).
- Vision encode on sm_70 accelerated in b37d0dd (last geoffwatts commit).

### 1.8 Speculative decoding / MTP
- Qwen3.8-27B's native single MTP layer is bound as `mtp_layers=1` (fc [2*hidden -> hidden],
  own attention + MLP) and used by `--spec mtp --draft-tokens K` (K 1..5 upstream; 1..7 on Volta
  after the "width-6+ verify" fix) [config.h; `src/product/speculative_options.h`].
- Drafting = K sequential AR steps through the MTP layer (`mtp_forward_ar_step`); each step
  ends in an argmax over either the full head or, with `--lm-head-draft`, the "optimized
  proposal head" = an indexed vocabulary-subset head with a row->token-id remap (verification
  always uses the full head) [Neroued: `docs/maintainer/qwen3_5-model.md:252-254`;
  `docs/serving.md:49`]. PR #11 measured the Q4 proposal head at 795 GB/s and added a
  register-streamed W8 GEMV at T=1 for the draft path.
- Context-lookup MTP (a host-side n-gram fast path, credited to syv-ai/qwen38-27b-rtx3090): if
  the last 16 generated tokens exactly match an earlier span, the tokens that followed it are
  proposed (up to 15) and verified in one round. This is why the headline 219/228/236 tok/s
  numbers exist: the bench corpus is a "deterministic continuation" and the sweep logs show 54
  rounds for 256 tokens (4.7 tok/round) at K=1 [`docs/v100.md` "Context-lookup MTP";
  dollarwong `nvfp4-official-sweep.txt` "spec round/fb 54/0"]. The authors say to treat those
  as a synthetic ceiling.
- DFlash2 (z-lab 5-layer masked-block drafter shipped in the artifact) is also ported to Volta
  (`--spec dflash2`, K 1..15); MTP beats it on Volta except at 2K and 32K on the varied corpus
  [`docs/v100.md` "Varied-context DFlash2 sweep"].
- Real-text acceptance at K=3: 70% median (JimmyMax HTTP sweep), 35-68% (dollarwong serve log),
  46-77% (huangserva code vs Chinese) -- i.e. ~2.1-3.3 committed tokens per round.

### 1.9 Batching / concurrency
Startup-fixed 1..8 lanes, exact-batch decode graphs, one shared KV pool with reservations,
Device/Host prefix checkpoints (`--device-state-slots`, `--host-state-slots`, `--host-kv-mib`),
no continuous-batching preemption. On a 32 GB V100 the practical configs are C=1 or C=2
[README quick starts; JimmyMax `docs/V100-BUILD.md`].

### 1.10 Multi-GPU (huangserva TP2 only)
- Two `ninfer-serve` processes, one per GPU; the artifact is sharded OFFLINE
  (`tools/tp2/shard_qwen38_27b.py`: q/k/v/gate/up rows split, o_proj/GDN-out/down columns
  split, embedding/LM head/draft head/norms/vision replicated, `--verify` bit-checks the
  reassembly); per-rank ~12.4 GB files, ~20.4 GB device each at 262K context + vision
  [`README.md`; `deploy/README.md`].
- All-reduce after o_proj, GDN out and MLP down (3 per layer, 192 per step) captured INSIDE the
  decode CUDA graph via NCCL (needs an NCCL with sm_70 kernels: the pip wheel
  `nvidia-nccl-cu12==2.21.5`; Ubuntu 24.04's NCCL 2.31 has none). Prefill all-reduces run eager
  (>=160 KB fails inside a graph). Rank 0 adds its half into the residual, rank 1 into a zeroed
  buffer, then one all-reduce [`docs/tp2-design-notes.md`; `src/targets/qwen3_6_27b_tp2/impl/tp2_comm.cpp`].
- Messages < 128 KB bypass NCCL: a pinned-host "mailbox" on /dev/shm (`cudaHostRegister`; must be
  a RAM filesystem), one flag per block, fixed-order add so both ranks round identically; 40 KB
  all-reduce 26 -> 16 us (NCCL in-graph measured 15 us @10 KB, 24 us @40 KB)
  [`tp2_mailbox.cu:1-50`; design notes]. LM head and draft head optionally vocab-sharded.
- Comm is 2.2 ms of a 33 ms 2-GPU round at 186K (PCIe, `NCCL_P2P_DISABLE=1`, no NVLink; they
  estimate NVLink would help decode <10%). Lockstep: after every prefill chunk / decode round the
  ranks exchange token count + hash + cancel flag through a shared file; mismatch -> both restart
  (fired once in a 300-request run; 23 s recovery) [`README.md` "Known issues"; `deploy/README.md`].

---

## 2. Measured performance (exact configs)

Hardware note: V100-PCIe-32GB and V100-SXM2-32GB both have 900 GB/s HBM2; SXM2 clocks ~10%
higher and the authors measured it ~7% faster per decode round ("HBM-bound") [`docs/v100.md`].

### 2.1 geoffwatts engine benchmark (synthetic corpus; treat decode as a ceiling)
`ninfer_bench`, Tesla V100-PCIe-32GB, CUDA 12.8, INT8 g64 KV, CUDA graphs, optimized proposal
head, `bench/fixtures/bench_corpus.ids`, prefill = isolated pp2048, decode = pp2048+tg256,
1 warmup + 3 reps, single request [`README.md` Performance; `docs/v100.md`]:

| Artifact / backend | K | Prefill tok/s | Decode tok/s | Acceptance |
|---|---:|---:|---:|---:|
| Qwen3.8-27B nvfp4, MTP | 1 | 1,102.5 | 218.98 | 99.2% |
| Qwen3.8-27B nvfp4, MTP | 3 | 1,094.9 | 209.24 | 97.5% |
| Qwen3.8-27B nvfp4, MTP | 5 | 1,100.3 | 199.58 | 97.1% |
| Qwen3.8-27B nvfp4, MTP | 7 | 1,087.2 | 180.74 | 93.3% |
| Qwen3.8-27B groupwise-int, MTP | 5 | 1,083.9 | 130.96 | 97.1% |
| Qwen3.8-27B nvfp4, DFlash2 | 7 | 1,059.0 | 126.32 | 100% |
| Qwen3.8-27B groupwise-int, DFlash2 | 7 | 1,044.2 | 77.84 | 100% |
| Qwen3.8-27B nvfp4, NO spec (PR #10 table, v3 artifact) | - | 1,144.8 | 29.3 | n/a |
| Qwen3.6-35B-A3B groupwise-int, DFlash | 4 | 686.2 | 139.58 | 90.9% |

Varied non-repetitive corpus (37,758 tokens), nvfp4, int8 KV, 128 generated tokens, best static K
[`docs/v100.md` "Varied-context DFlash2 sweep"]:

| Context | No-spec tok/s | Best MTP (K) tok/s / acc | Best DFlash2 (K) tok/s / acc |
|---:|---:|---|---|
| 2,048 | 29.28 | 67.66 (K=3) / 59.9% | 68.33 (K=7) / 35.3% |
| 8,192 | 28.02 | 92.74 (K=3) / 78.6% | 46.92 (K=3) / 50.7% |
| 16,384 | 26.70 | 49.80 (K=3) / 49.0% | 39.95 (K=3) / 40.5% |
| 32,768 | 22.13 | 55.04 (K=4) / 57.8% | 55.43 (K=7) / 38.6% |
| 150,000 | 15.38 | 43.63 (K=2) / 79.2% | 41.09 (K=7) / 53.7% |

Context-lookup copy test (172-token verbatim prompt): nvfp4 201.0 tok/s at 12.91 tokens/round.

### 2.2 dollarwong: same-methodology SXM2 sweep and the llama.cpp head-to-head
Official-methodology rerun on V100-SXM2-32GB, self-converted abliterated nvfp4 artifact
(23,719,496,192 bytes; device weights 19.73 GiB incl. MTP), CUDA runtime 12.8 / driver 13.0,
int8 KV, `--max-ctx 8192`, prefill_chunk 1024 [`deploy-v100/README.md`; raw:
`deploy-v100/bench/results/nvfp4-official-sweep.txt`; script `scripts/official-sweep.sh`]:
- pp2048 prefill 1,167.48 tok/s; decode pp2048+tg256 MTP K=1/2/3/4/5 = 236.62 / 235.71 / 235.03
  / 233.26 / 231.93 tok/s, acceptance 99.17% at every K, 54 rounds per 256 tokens; no-spec
  30.66 tok/s ("MTP = 7.7x" on this corpus).
Cross-engine HTTP benchmark (`bench/bench_matrix.py`: OpenAI chat, temperature 0,
reasoning_effort none, max_tokens 120-200, `timings` from the server; corpus = random local
Chinese markdown, ~1.66 chars/token). NInfer: `ninfer-serve --max-context 131072 --kv-capacity
auto --prefill-chunk 2048 --kv-dtype int8 --spec mtp --draft-tokens 3 --lm-head-draft
--preserve-thinking --vision --max-concurrency 1` (`scripts/ninfer-serve.service`).
llama.cpp: NOT FOUND. The `llama-server` systemd unit is not committed; the only facts are
"Qwen3.8-27B-Uncensored GGUF" and "llama takes 22 GB" [`scripts/gpu-mode:5`, `scripts/bench-v100.sh:5`].
Build, quant, `-ngl`, FA, KV type, ubatch and whether any speculative/MTP path was on are
unknown. Note llama-server's `predicted_per_second` counts accepted draft tokens when spec is
on, and 52.3 tok/s on a ~19-20 GB GGUF would exceed the 900 GB/s floor for plain decode, so
the llama.cpp side was probably NOT target-only; unverifiable.

| Case | Engine | prompt tok | out tok | client tok/s | server decode tok/s | server prefill tok/s |
|---|---|---:|---:|---:|---:|---:|
| short (19 tok) | ninfer | 19 | 28 | 60.1 | 78.3 | 59.3 |
| short | llama.cpp | 19 | 28 | 37.8 | 52.3 | 18.0 |
| mid (~10K) | ninfer | 9,989 | 82 | 8.1 | 62.2 | 1,137.5 |
| mid | llama.cpp | 9,989 | 108 | 7.7 | 43.2 | 863.8 |
| long (~89K) | ninfer | 89,734 | 94 | 0.68 | 44.2 | 657.9 |
| long | llama.cpp | 89,734 | 95 | 0.53 | 20.1 | 467.3 |
| xl (~123K) | ninfer | 122,658 | 101 | 0.43 | 32.4 | 534.3 |
| xl | llama.cpp | 122,658 | 141 | 1.18 | 21.3 | 300.5 |

TTFT on the 19-token prompt (cold / 2nd / 3rd): ninfer 299 / 120 / 125 ms; llama.cpp 3171 /
483 / 221 ms [`bench/results/{ninfer,llama}.json`]. Serve logs: MTP acceptance 35-68%, decode
53-69 tok/s depending on context/thinking.

### 2.3 JimmyMax: real-prompt HTTP numbers on V100-PCIe-32GB (most representative)
`ninfer-serve` official v3 nvfp4 artifact, `--max-context 230000 --kv-capacity auto
--max-concurrency 2 --kv-dtype int8 --device-state-slots 2 --host-state-slots 8 --host-kv-mib
8192 --spec mtp --draft-tokens 3 --lm-head-draft --vision`, driver 580, CUDA 12.8, SM clock
locked at 1380 MHz (`nvidia-smi -lgc 1380,1380`; unlocked it drifted to ~1245 MHz = -10%)
[`docs/V100-BUILD.md`]:

| ctx | 2K | 8K | 16K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|---:|---:|
| prefill tok/s | 1043 | 1070 | 1012 | 907 | 737 | 526 |
| decode tok/s (128-token completions, 3 reps) | 96 | 105 | 104 | 69 | 73 | 45 |

K sweep, same recipe (avg over 2K/8K/32K/64K/128K): K=2 78.2, K=3 84.2 (2K 123.0, 8K 115.4,
32K 78.7, 64K 56.4, 128K 47.5; median acceptance 70.0%), K=4 61.7 (acc 42.9%), K=5 77.4.
Stock-artifact table (230k, vision on, C=1) [`docs/QWEN38-27B-EFFICIENTTHINK-K3.md`]:
prompt 1K/2K/4K/8K/16K/32K/64K/128K -> output 138.5/78.2/71.9/73.3/68.1/99.1/52.4/38.6 tok/s,
prefill 996/1035/1042/1026/976/882/721/518 tok/s, ITL 43-81 ms. EfficientThink mixed artifact
(+9.3% weight bytes) decoded 68 vs 73.4 tok/s (-7.4%) "roughly proportional to bytes: decode
is weight-bandwidth bound".

### 2.4 huangserva: single-GPU kernel rewrite and TP2 (Qwen3.8-27B nvfp4, MTP K=3, int8 KV)
2x Tesla V100 32G PCIe, no NVLink; 186K/193K rows = mean of 4 seeds x 1024 tokens; 8K = single
256-token run [`README.md`]:

| Scenario | 1x V100 upstream kernels | 1x V100 this repo | 2x V100 TP2 |
|---|---:|---:|---:|
| 186K code prompt, decode | 36 | 56 | 101.9 |
| 193K Chinese prompt, decode | 26 | 43 | 73.4 |
| 8K code / Chinese, decode | 79 / 58 | 97 / 73 | 140 / 106 |
| 186K first-token latency | 455 s | 379 s | 250 s |
| 262,144-token prompt | does not fit | does not fit | code 91-95, Chinese 68-71, TTFT ~393 s |

Round time 32.4 ms on 2 GPUs at 186K regardless of language (Chinese is slower only via
acceptance 0.46 vs 0.77). Versus one RTX 4090 48G running stock NInfer groupwise-int: 2xV100
decode on par or slightly faster (256K code 94.7 vs 80.9; 8K 140.2 vs 108.6), 4090 prefill
~1.85x faster (256K TTFT 210 s vs 393 s). Quality: 237/300 on their core-300 set vs 245
(llama.cpp Q4_K_M) and 238 (Q8_0) on a 4090, McNemar p=0.20 / 1.0 (within noise).

### 2.5 Flo5k5 (geoffwatts PR #11 / issue #13): 26k-context decode
V100-PCIE-32GB at application clocks 877/1380, CUDA 12.8.1, nvfp4 v3, int8 KV, `--spec mtp
--draft-tokens 4 --lm-head-draft`, serve-level, ms/step = predicted_ms/(predicted_n -
draft_n_accepted):

| Build | ms/step @26k | tok/step | tok/s @26k |
|---|---:|---:|---:|
| v3 port, K=5, full head | 70.9 | ~3.0 | ~45 |
| + `--lm-head-draft` | 66.1 | ~3.0 | 46.5 |
| + attention split-K, NVFP4 batching, fp16 staging, Q4 pipeline, K=4 | 43.5 | 2.96 | 68 |
| + MTP draft-path GEMVs | 41.9 | 2.96 | 70.6 |
| + verify-path fusions | 38.9 | 3.01 | 77.4 |
| + key-major int8 attention | n/a | n/a | 82.6 |

Short-context decode on the same build: 100-112 tok/s. Launches per step 1255 -> 726.

### 2.6 ww485000 (Windows) and the "228 tok/s" claim
Tesla PG503-216 32GB (V100 OEM), TCC, driver 576.57, CUDA 12.9, Windows 10, `ninfer_bench`
pp2048 / pp2048+tg256, int8 KV, CUDA graphs, MTP K=3, optimized proposal head, prefill chunk
2048, 1 warmup + 3 reps: official v3 nvfp4 1,135.88 / 228.14 tok/s, acceptance 99.17%
[`README.md`; `docs/V100-BUILD.md`]. It is the SAME synthetic bench corpus as 2.1/2.2, single
stream, single request, MTP + context-lookup; the authors state it "is not a promise for
arbitrary natural-language prompts". No real-prompt numbers are published in that repo.

### 2.7 Reference points
- Upstream RTX 5090 (1792 GB/s), nvfp4, no spec: 71.2 tok/s decode / 8,340 tok/s prefill at
  7,680 tokens; MTP3 by category (groupwise-int): Code 200.3, Story 130.4, Translation 198.1,
  Structured 224.4 tok/s; acceptance 38-90% [Neroued: `docs/performance/qwen3.8-27b.md`].
- v100-skinny (1Cat-vLLM + QPN kernels, 4x V100-SXM2-16GB, k=7): 219.1 tok/s on AIME-2026 P1,
  5.89 tok/round, 26.9 ms round; "+0.383 ms per extra draft row vs +0.817 ms per sequential
  drafter step" [dnv2003/v100-skinny README].
- Your fork (from the task statement): ~34 tok/s target-only and ~45 with MTP on 1x V100
  Q4_K_XL; 78-103 on 2x V100 with MTP/ngram drafts.

Bandwidth sanity check (mine): the nvfp4 artifact reads ~18.5-19 GB per target step (56 NVFP4
MLPs 7.9 GB + 8 FP8 MLPs 2.1 GB + FP8 GDN projections 5.5 GB + FP8 attention 1.7 GB + FP8
head 1.27 GB). At 900 GB/s that is a 20.5-21 ms floor. NInfer no-spec = 29.3-30.7 tok/s =
33-34 ms/step (~62% of floor); your llama.cpp Q4_K_XL target-only 34 tok/s on ~16-17 GB is the
same ~60% efficiency. NInfer's advantage is NOT the T=1 GEMV. It is the MTP round: 27-42 ms
for verify width 4-5 plus 3-4 draft steps (2.5, 2.6), against ~69 ms implied by 45 tok/s at
~3.1 tok/round in the fork. (I could not reconcile why PR #11's short-context round, ~30 ms,
is below the 34 ms no-spec step; different measurement boundaries are the likely cause.)

---

## 3. Reasons the authors give for being fast on Volta

Quoted or paraphrased, with source:
1. "Decode is predominantly HBM-bound, so PCIe bandwidth and host performance have little
   effect" -- weights are streamed once per round at 4-bit density through tensor cores
   [geoffwatts `README.md`; `docs/v100.md`].
2. Software NVFP4/FP8 on `mma.sync.m8n8k4` with the quadpair-split-N mapping: "activation
   traffic per weight byte drops 4x", "no shared memory and no barrier in the main loop",
   "one fp16 group scale is held in a register across exactly its group's 16 MMA slices"
   [`q4_volta_qpn_gemm.cuh:1-60`]. The SIMT route "is ALU-bound at roughly 22 ops per weight
   byte and re-reads the whole weight matrix once per 8 output columns"; the CUTLASS route
   "dequantises the entire weight into an FP16 scratch buffer in global memory on every call"
   [`q4_volta_mma_gemm.cuh:1-25`] -- the QPN design avoids both.
3. Verification width is nearly free, so native-depth MTP pays: "QPN makes the wider k=7
   verification batch cheap enough for the model's native MTP depth to pay"; "+0.383 ms per
   extra draft row vs +0.817 ms per sequential drafter step" [v100-skinny README, the kernel's
   origin]. NInfer's own long-context plan: draft steps should be single-column GEMVs at full
   bandwidth (795 GB/s proposal head) [PR #11].
4. Whole-round CUDA graph with GPU-side sampling/acceptance, exact-batch topologies, no host
   sync [engine-architecture s.8; ninfer_bench `decode_path=mtp_cuda_graph`].
5. INT8 group-64 KV with Hadamard rotation halves attention bytes at long context without
   losing the tensor-core path ("bf16 beat int8 by 1.46x while reading twice the bytes, purely
   because it had tensor cores"; the int8 TC kernel "takes the tensor-core math and keeps
   int8's halved traffic") [`small_t_i8_volta.cuh:10-15`].
6. Fused ops and fewer launches: fused QKV|gate, fused GDN qkvz+conv, SwiGLU and residual in
   GEMM epilogues, fused norm+RoPE; "about 1255 kernel launches per step, many of them small
   norm/activation kernels" was one of the four named causes of the long-context collapse
   [issue #13; PR #11].
7. Volta-specific micro-facts the authors hit and fixed: bf16->fp16 conversions on the
   quarter-rate F2F pipe (fp16 staging), PRMT for int8->fp16 instead of cvt, wave-aligned
   split-KV counts for 80 SMs, "Volta can only keep one thread block per SM resident" for the
   attention kernel [issue #13; huangserva README; PR #11].
8. The prefill attention kernel is llama.cpp's; they claim nothing over llama.cpp there. The
   prefill edge in 2.2 comes from bigger chunks (2048), fused projections and fewer dequant
   passes, not from the attention kernel.
9. Model-level: only 16/64 layers carry KV and GDN state is O(1) in context, so long-context
   decode degrades slowly (5090: 71 -> 53 tok/s from 8K to 260K no-spec) -- an architecture
   property that llama.cpp shares once its kernels are as efficient.

---

## 4. Transferability to a ggml/llama.cpp-based engine

Facts about the local tree checked for this section (`ggml/src/ggml-cuda/` at fork HEAD
e117ee884): `volta_mma_available()` / `fp16_mma_hardware_available()` (common.cuh:316-346);
`fattn.cu` has a Volta branch of the MMA kernel (line 69: "On Volta the GQA optimizations
aren't as impactful vs. minimizing wasted compute") -- this is the kernel NInfer vendors;
MMQ on Volta: `ggml_cuda_should_use_mmq` returns MMQ (dp4a tiles) only when
`ne11 < MMQ_DP4A_MAX_BATCH_SIZE (64)`; above that Volta falls to dequant + cuBLAS FP16
(mmq.cu:267-330, mmq.cuh:8); `MMVQ_MAX_BATCH_SIZE 8` (mmvq.cuh:3); `gated_delta_net.cu` exists
with "TODO: Add chunked kernel for even faster pre-fill" (line 180); CUDA graphs and
`mul_mat_id` fusion paths exist (ggml-cuda.cu:1741, 2616-2663, 3087).

### (a) Already present in llama.cpp (no port needed)
- Volta MMA flash attention for prefill (identical kernel; NInfer even pins llama.cpp
  62bf73d25). huangserva's retune (ncols1 32, 64-column Q tile at D=256/GQA-6) is just a
  template parameter choice in that kernel: cheap to try in `fattn.cu`'s Volta branch.
- CUDA graphs for decode; fused mul_mat_id + GLU paths; quantized KV cache (q8_0/q4_0 via the
  vec kernels); dequant + cuBLAS HGEMM prefill for large batches (same approach as NInfer's
  CUTLASS-sm70 route); MMVQ dp4a for T<=8.
- Your fork already has MTP, n-gram drafts, DFlash2 and fused ops (memory notes), i.e. the
  round-level features. What differs is the per-round cost on Volta, not the feature list.

### (b) Absent and portable to ggml-cuda for dense models
1. QPN-style small-T tensor-core GEMM for Q4_K/Q8_0/NVFP4 on sm_70 (T=2..32). Today T<=8 goes
   to MMVQ (dp4a SIMT, quantised activations) and T>8 to MMQ dp4a tiles. The NInfer/v100-skinny
   kernels show the m8n8k4 quadpair mapping keeps T=4..8 at 60-71% of read roofline, which is
   what makes an MTP/DFlash round cost ~1.1-1.3 target steps instead of ~2.3. Port shape: a new
   `mul_mat` route gated on `cc == 700 && 2 <= ne11 <= 32` for Q4_K (nibble order needs a
   per-format fragment map; Q4_0/Q8_0 are simplest), dequant-in-register to fp16, fp16
   activations (convert once, not per lane). Gain: per-round time at MTP width; expected
   +40-80% decode with MTP on 1x V100 if your round is really ~2.3 steps. Effort: 1-2 weeks per
   format family; correctness oracle exists (test-backend-ops). Licence-clean sources: the
   fragment maps are ggml's own (`mma.cuh`), v100-skinny `kernels/` is MIT.
2. Whole-round GPU residency for speculative decode: NInfer keeps verify + K draft steps +
   acceptance + sampling in one CUDA graph with no host sync. In llama.cpp the sampler and the
   accept/rollback loop run on the host and each draft step is its own graph/eval. Moving
   argmax drafting and acceptance for greedy/low-temperature paths onto the GPU (a small
   `speculative_round`-like kernel plus graph-captured draft steps) removes K+1 host round
   trips per round (each is a launch gap + D2H of one token). Gain: several ms per round on
   Volta (PR #11 shows launch count alone was worth 66.1 -> 38.9 ms together with fusions).
   Effort: engine-level, 1-2 weeks, must respect llama.cpp's sampler contract (only exact
   argmax / fixed-seed paths can be graph-captured).
3. INT8 group-64 KV with Hadamard-rotated K/Q and an int8 tensor-core decode-attention kernel
   for sm_70. llama.cpp's q8_0 KV goes through the vec (SIMT) FA kernels on Volta; NInfer
   measured SIMT int8 at 1.86 TFLOP/s vs 2.71 for the bf16 MMA kernel. A "Q8 KV + m8n8k4"
   variant of the fattn-vec/mma path with in-register dequant is the item; huangserva's
   8-warp/Bc=64 layout (1.29 ms per 393 MB call, ~305 GB/s) is the reference point. Gain: only
   at long context (>=32K); huangserva 186K: 36 -> 56 tok/s single GPU. Effort: 2-3 weeks,
   kernel-only. The Hadamard rotation changes the KV bytes format (not GGUF-visible, cache
   only).
4. Fused projections: one GEMM for [Q|K|gate|V] and for GDN [q|k|v|z] (+ conv1d + SiLU in the
   epilogue), SwiGLU and residual-add in GEMM epilogues, fused q/k-RMSNorm+RoPE. In ggml this is
   partly graph-level (concatenate weights at load into one tensor -> one mul_mat) and partly
   backend fusion (ggml-cuda already fuses some GLU/add patterns). Gain: launch count (NInfer
   went 1255 -> 726 per step) and fewer activation round-trips; on Volta small kernels are
   relatively more expensive because SM count is low (80) and clocks are low. Effort: 1 week
   for weight concatenation at load + epilogue fusions in existing fused paths.
5. fp16 activation staging for tensor-core paths on Volta (avoid F2F conversions inside loops):
   relevant only to any new HMMA kernels you add (MMQ/MMVQ use int8 activations, unaffected).
6. Prefill chunk / ubatch: NInfer prefill runs 2048-token chunks; on Volta llama.cpp's
   cuBLAS route re-dequantises every weight to fp16 per ubatch, so `-ub 2048` (memory
   permitting) cuts dequant passes 4x versus 512. Config-only; verify on your box.
7. Volta-specific tuning constants: split-KV count aligned to 80 SMs x 2 CTAs; longer key
   splits at very long context (480 -> 1920 keys, 1402 -> 1130 us at 186K/T=4); SM clock lock
   (`nvidia-smi -lgc 1380,1380`, -10% drift otherwise).

### (c) Portable to MoE (mul_mat_id / expert kernels)
- The QPN small-T kernel is the same building block for experts: with top-8 routing at T=4..8
  each expert sees 1-8 rows, exactly the regime where an 8-row m8n8k4 tile is efficient. The
  port is a `mul_mat_id` variant that iterates (expert, row-subset) pairs with the QPN inner
  loop. Same effort class as (b)1 plus routing plumbing; your expert-cache work (#151) is
  orthogonal and composes (cache decides WHICH expert bytes are resident; QPN decides how fast
  resident bytes are consumed).
- NInfer's own Volta MoE path is NOT a strong reference: its decode kernels are SIMT
  (`dot_bf16_eight`, warp-level dot products that re-read an expert's weights per token), its
  prefill is "weight-stationary grouped expert GEMMs" via SIMT at ~425 GB/s of a 728 GB/s
  ceiling, and shapes are hard-coded (`kHidden=2048, kExperts=256, kTopK=8, kIntermediate=512`)
  [`src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu:24-30`;
  `sparse_moe_prefill_kernels.cu:1103-1115`]. The 35B-A3B V100 result (DFlash K=3: 125.9 tok/s,
  30 ms per round for ~3 B active parameters) is far from bandwidth-bound; upstream on 5090 gets
  642 tok/s. Do not port the MoE kernels; port the dense QPN idea into mul_mat_id.
- The GDN handling is the same for dense and MoE Qwen3.5-family models and is already the
  sequential kernel in both engines; nothing to port (NInfer's chunked GDN is Ampere+ only).

### (d) Tied to the artifact / fixed-model design, not portable
- The `.ninfer` container, v2/v3 binder, compile-time target registry, `constexpr` shapes,
  hand-written layer schedule, workspace recipes, exact-batch graph profiles per topology.
- Load-time in-place QPN repack of NVFP4/FP8 payloads (ggml would instead need the fragment
  order handled at kernel level, or a repack-on-load like the CPU backend's `repack` types; the
  latter is feasible but changes tensor bytes in memory).
- Mixed FP8-row + NVFP4 recipe with per-tensor global divisors (ggml has NVFP4 and MXFP4 types
  but no fp8-row-scaled weight type; your Q4_K_XL mix plays the same role).
- ReplaySSM (raw-input replay of GDN state for speculative rollback) -- relevant if your MTP
  rollback for GDN layers currently snapshots the 151 MB state per verify; NInfer records the
  per-token raw transition inputs and re-folds only the accepted prefix
  [Neroued: `docs/maintainer/replayssm-gdn.md`]. Portable in principle, but it is engine logic
  tied to their transaction model.
- Prefix-reuse checkpoints (Device/Host state + KV), lockstep TP2 with a Python proxy and
  offline-sharded artifacts: not applicable to ggml's in-process split modes.

---

## 5. Correctness / quality trade-offs and licensing

- NVFP4 vs groupwise-int (same model, same eval harness, one sample per problem; EvalScope,
  thinking on, MTP3): Qwen3.8-27B groupwise-int AIME25/26 96.67/96.67, GPQA-D 87.37, ERQA
  66.25, RealWorldQA 82.22; NVFP4 96.67/96.67, 90.40, 66.25, 83.53 -- within single-sample
  noise [Neroued `README.md` Evaluation; model cards]. huangserva's 300-question set: NVFP4
  TP2 237 vs llama.cpp Q4_K_M 245 / Q8_0 238 on a 4090, McNemar p=0.20/1.0.
- KV precision: INT8 g64 + Hadamard is the recommended Volta profile; FP8 KV fits +14.4% tokens
  but decodes slower on Volta; NVFP4/K8V4 KV unavailable on Volta. Upstream's published tables
  use INT8 g64 KV as well. No perplexity numbers for int8-vs-bf16 KV were found in the V100
  forks ("not found").
- Numerics of the fast paths: fp16 operands / fp32 accumulate (CUTLASS sm70 "0.17%" and the
  Q4 MMA kernel "0.223%" L2 relative error vs fp32 host reference) [`q4_volta_mma_gemm.cuh`];
  PR #11's changes are explicitly "not bit-exact with the previous kernels" (vec8 RMSNorm,
  epilogue rounding, fp16 activations) and shipped without a perplexity run. The sm_70 MTP
  width>=5 regression that "drifted off the greedy argmax" was traced to a dropped Volta verify
  kernel and fixed (bit-exact vs `--spec none` for K<=7) [`speculative_options.h`].
- Context-lookup MTP is lossless (verification is authoritative) but inflates benchmark
  numbers on repetitive corpora; JimmyMax notes a v2-loader "cache replay turns alternate
  garbage" bug that v3 does not reproduce; huangserva TP2 had one rank divergence in 300
  requests (caught by lockstep, root cause unknown).
- EfficientThink mixed artifact: keeping sensitive layers in FP8 costs 7.4% decode for a
  quality trade [JimmyMax `docs/QWEN38-27B-EFFICIENTTHINK-K3.md`].
- Licensing: NInfer and all forks Apache-2.0 (huangserva adds NOTICE + CHANGES.md per s.4(b));
  vendored llama.cpp attention MIT; `volta_mma.cuh` is transcribed from ggml-cuda/mma.cuh (MIT);
  the QPN NVFP4 kernel is adapted from dnv2003/v100-skinny whose `kernels/` are MIT (GitHub
  reports NOASSERTION because the repo is dual-licensed by directory; `fork_patches/` is
  Apache-2.0 from 1Cat-vLLM); the e2m1 shift decode is from TurboMind via 1Cat-vLLM
  (Apache-2.0). Model artifacts: Apache-2.0 (Qwen, unsloth NVFP4). All compatible with an MIT
  llama.cpp fork provided notices are kept.

---

## Top 5 transferable ideas, ranked by expected gain on V100

1. Small-T (2..32 rows) tensor-core GEMM for quantized weights on sm_70 with the quadpair-split-N
   m8n8k4 mapping, in-register dequant, fp16 activations, split-K warps, no smem in the main
   loop. This is THE reason NInfer's MTP round costs ~1 target step on Volta. Applies to dense
   `mul_mat` and to `mul_mat_id` (experts see 1-8 rows). Expected: biggest single win for every
   spec-decode mode you run (MTP, n-gram, DFlash2) on both 1x and 2x V100.
2. Whole-round GPU speculative loop (draft steps + acceptance + argmax sampling captured in the
   decode graph; no per-draft host round trip). Second-largest term in PR #11's 66 -> 39 ms.
3. INT8 g64 KV + Hadamard with an int8 m8n8k4 decode-attention kernel (huangserva/Flo5k5 layouts).
   Long-context only, but it is where your fleet serves (128K-262K): 36 -> 56 tok/s at 186K
   single GPU in their data.
4. Launch-count reduction through weight concatenation at load ([Q|K|gate|V], GDN [q|k|v|z]) and
   epilogue fusion (SwiGLU, residual add, q/k-norm+RoPE). Volta's 80 SMs at 1.38 GHz make each
   small kernel proportionally costlier; 1255 -> 726 launches/step in PR #11.
5. Volta prefill housekeeping: 2048-token chunks / `-ub 2048` (fewer dequant passes on the
   cuBLAS route), the 64-column Q tile in the Volta FA kernel (119 -> 88 ms per 1024-token chunk
   at 131K), SM clock lock. Cheap, config/constant-level, measurable in an afternoon.

Not recommended to port: NInfer's MoE kernels (SIMT, hard-coded shapes, far from roofline on
Volta), the artifact/binder stack, TP2's process-pair + proxy design.

## Things I could not verify ("not found")
- The llama.cpp build/flags/quant behind dollarwong's head-to-head (unit file not committed).
- Any NInfer-vs-llama.cpp comparison by huangserva other than quality (their speed comparison
  is against NInfer on a 4090).
- Per-kernel GB/s of the base geoffwatts QPN kernels on V100 (only the v100-skinny origin
  numbers and PR #11's 795 GB/s proposal-head figure are stated).
- Perplexity deltas for int8-KV vs bf16-KV on Volta.
- Whether Flo5k5's PR #11 kernels were merged anywhere (PR closed; not in geoffwatts b37d0dd,
  not in the JimmyMax/ww485000/huangserva clones by file list).
