# qwen4exp (Qwen3.8-Flash-Next) port - upstream PR 27742 + transformers oracle audit

Branch `qwen4exp` (off master, 2026-08-27). Port of upstream llama.cpp PR 27742
(open, unmerged upstream as of fe235f434). Correctness oracle = HF transformers
`src/transformers/models/qwen4_exp/` (merged on transformers master, Qwen-team
authored: modular_qwen4_exp.py 1186 lines + configuration 334 lines), fetched
2026-08-27 into scratchpad qwen4exp-ref/. Real-checkpoint facts cross-checked
against Qwen/Qwen3.8-Flash-Next config.json.

## What the arch is

176B total / ~6B active MoE multimodal, early Qwen4 preview:
- 48 layers: 3-of-4 GDN (gated delta net, Qwen3.5 kernel, SIGMOID output gate)
  + every 4th layer full GQA attention (16 q / 2 kv heads, head_dim 256).
- QSA sparse attention on the full-attn layers: 4-head indexer (dim 128, one
  raw-cached key head), mean-pooled blocks of ratio 4, budget 2048 tokens.
- Hyper-connections: 4 residual streams, low-rank (320) gated input mixer per
  block + per-stream injection; final mixer doubles as output norm.
- PLE: one layer (layer 1, 0-based) reads a ~97 GiB n-gram hash table
  (16 heads = 2 orders x 8, head_dim 128), signed-sqrt sigmoid gate + dilated
  depthwise conv (K=4, dilation 3).
- IMRoPE interleaved, partial rotary 64/256, sections [11,11,10], theta 1e7.
- MoE: 512 experts top-10 softmax+renorm, gated shared expert (512).
- Vision: unmodified Qwen3-VL ViT (existing clip path).
- In-model 1-layer MTP head EXISTS in the checkpoint but the PR/conversion
  DROPS it (vLLM does too). SGLang serves it (IndexShare MTP) - future lane.

## Port mechanics

- 3-way apply of the PR diff (29 files, merge-base c5fc7e348): 21 clean,
  8 files / 15 hunks hand-woven. Fork-side choices:
  - kept fork MAX_EXPERTS 512 (qwen4exp needs exactly 512), added PLE bounds
  - did NOT adopt base-drift arches (QWEN3TTS, MINIMAX_01) or the dsv4_state
    buft pattern (unused in fork), DEEPSEEK4 test config left as fork had it
  - pulled true base drift the PR depends on: llama-kv-cells.h ext.tok field +
    for_each_token_in + value-init reset (upstream version taken wholesale)
  - re-added declarations the conflicted hunks dropped: kv_cache has_cell_ext /
    get_prev_tokens / get_cells + context forwarder
- Upstream now carries the meta backend (llama_meta_device_get_split_state in
  upstream master); the PR itself classifies the new caches for split serving:
  idx cache and PLE conv cache = MIRRORED. Landed with the port.

## Gates (2026-08-27)

- build-cpu clean; test-llama-archs qwen4exp CPU: OK (0.00e+00) + session OK
  (exercises GDN+QSA+PLE+HC synthetic model incl. state save/restore).
- Canonical loopback byte gate: off-leg sha c80261ff reproduced, stable 6/6 -
  the shared kv-cache/loader changes are inert for existing arches.
- build-cuda75: see TASKS #143 status.
- Real-GGUF smoke (X99, unsloth UD-Q4_K_XL 4 shards): pending.

## Transformers-oracle audit (component by component)

Verified equivalent (line-level):
- HC mix: grouped RMSNorm over one stream w/ folded (1+w) gamma; silu(down/hc);
  sigmoid(up); mean collapse over streams; inject raw at mix time,
  2*sigmoid(inject/hc) at combine; wide residual = raw pre-norm input;
  final mixer without inject = output norm; res_hc init = hc copies. All match
  modular Qwen4ExpTextGatedResidual + decoder wiring exactly.
- PLE: hash = (t0*m0) ^ (ts*ms) over uint64 - reference torch.long products
  cannot overflow (multipliers bounded by 2^63/vocab), xor preserves sign bit 0,
  so uint64 % == torch.remainder; EOS window cut matches _shift_right_ignore_eos
  (own-token EOS does not cut its own window); missing predecessor = EOS; head
  layout head-major per token == flatten(-2); signed sqrt clamp 1e-6 gate;
  value broadcast per stream; conv = sum of shifted taps, tap k reads
  (K-1-k)*dilation back, zero history on fresh state == reference pad.
  Predecessor tokens come from attention-cache cell ext.tok instead of a third
  conv state - equivalent information, exercised by the session-restore test.
- GDN: qkv/z/beta/alpha projections, sigmoid(beta), softplus(alpha+dt_bias) *
  (-exp(A_log)) with the transform baked by the inherited qwen35 converter,
  l2-normed q/k post-conv, z-gated RMSNorm output - inherits the fork's proven
  qwen35 delta-net base; the ONE numerical difference (sigmoid output gate) is
  hardcoded and matches the checkpoint's output_gate_type=sigmoid.
- Full attention: per-head [q|gate] interleave in wq, q/k RMSNorm (folded),
  partial IMRoPE on first 64 dims, 1/sqrt(256) scale, sigmoid output gate
  before wo. Matches Qwen3_5Attention lineage.
- QSA: raw (pre-norm pre-rope) keys cached; pooling in f32; pooled key normed
  then roped at block-start position; relu per head before head-sum; the
  reference's 1/sqrt(d) score scale is omitted - ranking-invariant (selection
  only, uniform positive scale, -inf bias unaffected); per-block bias mode
  reuses the attention mask for the per-cell half; tail (query's own partial
  block) forced in via finite 1e9 bias; incomplete blocks excluded; top-k
  unmask combined with the original causal mask so over-selection can never
  unmask a causally-invisible cell.
- MoE: softmax-then-topk with renorm (norm_topk_prob=True), shared expert with
  per-token sigmoid gate - existing fork machinery.
- Conversion: PLE layer ids converted 1-based -> 0-based; hash constants read
  as exact int64 from checkpoint buffers (bypassing the f32 cast); zero-centred
  gammas +1-folded by the inherited endswith("norm.weight") rule which also
  correctly EXCLUDES linear_attn.norm.weight (plain-weight RMSNormGated), plus
  explicit folds for the ple/indexer norms the suffix rule misses; indexer qk
  proj split; PLE shards streamed via memmap; MTP head dropped; vision tower =
  Qwen3VLVisionModel unchanged.

Divergences found (documented, deliberately NOT "fixed" - all are in the PR
itself; changing them would fork us off the PR we track):
1. QSA top-k width: C++ always selects budget + ratio - 1 cells; the reference
   selects block_topk whole blocks + the true tail (0..ratio-1 cells). When the
   tail is shorter than ratio-1, up to ratio-1-tail extra cells of the
   next-ranked block get unmasked (<= 3 tokens of ~2051 at ratio 4). Bounded,
   deterministic, absorbed in the PR's own 98% top-1 vs vLLM validation.
2. Unified-cache multi-seq QSA pollution: with --kv-unified and multiple
   sequences, block tiles are built over cells of ALL sequences (positions
   collide across seqs -> blk_cells slot collisions, filled[] overcounts, the
   pooled key mixes sequences). Masks still drop foreign cells so output stays
   causal-correct, but selection quality degrades. Serve qwen4exp with
   per-seq streams (default, no --kv-unified) or -np 1.
3. Quantized KV also quantizes the RAW indexer keys (type_k is forwarded to
   the idx cache): -ctk q4_0 adds selector noise the reference never has.
   First serves should run f16 KV; A/B q4_0 later against a quality gate.
   Also: the idx cache allocates an unused V tensor per QSA layer
   (n_embd_head_v=256/cell) - dead weight at huge ctx, cosmetic otherwise.
4. eos_token_id robustness nit: conversion uses eos[-1] where the reference
   uses eos[0] for the PLE hash reset when eos is a LIST. Real checkpoint has
   a single eos (248044), so moot for Flash-Next; would matter for a future
   multi-eos qwen4exp checkpoint.
5. output_gate_type is not recorded in GGUF; C++ hardcodes sigmoid. Correct
   for Flash-Next (config says sigmoid); a hypothetical silu-gate qwen4exp
   checkpoint would be silently wrong. Latent only.
6. mrope block positions: pooled-key rope writes the block-start position into
   all 4 mrope sections - exact for text, approximate for image positions
   (PR's own note). Multimodal QSA selection is approximate.
7. ple_key tensor shape assumes ple_embed_dim == hidden_size (true for
   Flash-Next: 2048); a checkpoint with a different ple_embed_dim would fail
   tensor-shape checks at load rather than run wrong.

## Serving guidance (first fleet contact)

- GGUF metadata (unsloth vintage) loads only if it carries the PR's final key
  set - verify PLE keys are present with a metadata dump before debugging.
- Start config: f16 KV (finding 3), no --kv-unified (finding 2), modest ctx;
  the PLE table wants --no-mmap OFF (it is gathered from the mapping;
  LLAMA_MMAP_RANDOM* machinery ships in the port for exactly this) and the X99
  251GB box.
- Expect the PR's known GPU-arch sensitivity (#27763 SM110 garbage >8 layers,
  #27780 SM121 aborts) to be untested territory on V100/SM70: gate coherence
  (read the output) before any t/s number.
