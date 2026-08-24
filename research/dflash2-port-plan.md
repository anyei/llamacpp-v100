# DFlash2 port plan (#138) - study before implementation

**Status: STUDY (2026-08-24). No code written yet - user chose "study first,
then implement".** This doc maps upstream PR 27342's DFlash2 support onto this
fork so the design is owned before any graft. Source: upstream commits
`9731ad3f2..1deefcca3` (`5ecbe1ac1 support DFlash2`, `1deefcca3 Add p_min`) on
the fetched branch `pr-27342-dflash2`.

## 0. Strategy verdict: MANUAL GRAFT, not merge/cherry-pick

- `git merge-base dflash2 pr-27342-dflash2` is empty and `9731ad3f2` is NOT an
  ancestor of `dflash2` - the two histories are effectively **unrelated**. A
  `git merge`/`cherry-pick` would drag the whole upstream tree the fork never
  had. So DFlash2 is hand-applied from the `9731ad3f2..1deefcca3` diff onto the
  fork's diverged files, one scoped piece at a time, gated like a review fix.
- The fork's `dflash.cpp` has diverged hard from upstream's: the fork added the
  **DSV4 backbone** (`build_dsv4`, base class `llm_graph_context_dsv4_mla`), the
  **DSpark markov + conf head**, spec-tree `process_rows`, PEARL, and the
  review #9/#17 consumer logic. DFlash2 (conv + selector) modifies only the
  **dense `graph<false>` decoder**, which the fork kept intact - so the graft
  lands in the dense path and leaves the fork-only dsv4/dspark paths alone.

## 1. What DFlash2 IS (the two mechanisms)

1. **Dynamic depthwise conv mixing** (`build_dflash2_conv`): a causal,
   grouped 1-D convolution whose per-tap coefficients are INPUT-dependent
   (`conv_proj` projects the normed hidden to coefficients) plus a static
   per-tap `conv_base`. Applied at FOUR points in each dense layer:
   pre-attn (side 0 on `noise_norm`) + post-attn (side 1 on the attn output),
   and pre-ffn (side 0) + post-ffn (side 1). Kernel is causal within each draft
   block (`conv_kernel_size` taps, `conv_group_size` channel groups). This is
   the "flash" temporal mixing between draft positions.
2. **Learned selector head** (`build_post_sampling`, virtual): after logits,
   for each draft position it takes the top-k candidates and scores
   predecessor->successor transitions using low-rank `selector_prev`/
   `selector_next` embeddings (rank `selector_rank`) conditioned on
   `selector_hidden @ t_embd`, adds the unary logit, and packs
   `[candidate_ids ; pairwise transition scores]` into `res->t_h_nextn` (the
   "dflash2_lattice"). The draft-time consumer runs a beam/Viterbi over this
   lattice to pick the block. This REPLACES the dspark markov+conf head's role
   for a DFlash2 vehicle.
3. Minor: `logit_scale`, `final_logit_softcapping`, `embedding_scale` added to
   the dense decoder (Qwen3.8 uses these).

## 2. Footprint (files to touch), upstream +689/-83 over 20 files

Additive, low-conflict (enums/fields):
- `src/llama-arch.h` / `.cpp`: +5 `LLM_KV_DFLASH_*` keys, +7 `LLM_TENSOR_DFLASH_*`
  enums, their name strings (`blk.%d.attn_conv_base` etc., `selector_*`), and
  `LLM_TENSOR_INFOS` op rows (conv_base=MUL, conv_proj=MUL_MAT,
  selector_hidden=MUL_MAT, selector_prev/next=GET_ROWS). Purely additive.
- `src/llama-hparams.h`: +5 `dflash_*` uint32 fields.
- `src/llama-model.h`: +4 per-layer conv tensor ptrs (`dflash_attn_conv_base/
  proj`, `dflash_ffn_conv_base/proj`) on `llama_layer`; +3 model-level selector
  ptrs (`dflash_selector_prev/next/hidden`).
- `src/llama-graph.h`: +1 `virtual void build_post_sampling() const {}` on
  `llm_graph_context`. Inherited through `llm_graph_context_dsv4_mla` -> the
  fork's dflash graph can override it.

Graft into diverged files (the real work):
- `src/models/models.h`: the fork's `llama_model_dflash::graph` has ZERO data
  members. Add `const llama_model & model;` member + `void build_post_sampling()
  const override;`. NOTE: the fork threads `model` as an explicit param today;
  adding the reference member means the ctor must init it (both `graph<true>`
  and `graph<false>` ctors).
- `src/models/dflash.cpp`:
  - `load_arch_hparams`: read the 5 `LLM_KV_DFLASH_*` keys into hparams
    (+ optional logit_scale/softcap/embedding_scale). COEXISTENCE: the fork
    reads `dflash.block_size` via raw `model.gguf_kv` at graph time
    (`:227-230`); upstream reads it into `hparams.dflash_block_size` (same key
    `"%s.block_size"`). Keep BOTH or migrate the raw read to the hparam - the
    selector needs `hparams.dflash_block_size`.
  - `load_arch_tensors`: add the `selector_hidden.weight`-gated block (throws on
    missing conv/selector metadata; divisibility + lattice-size guards) and the
    per-layer conv tensor loads. Must sit ALONGSIDE the fork's dspark-head block
    (`:80-92`) - both are presence-gated, a vehicle has one or the other.
  - `graph<false>::graph` dense loop: insert `build_dflash2_conv` at the four
    cb points the fork already has (`:591` noise_norm, `:620-622` attn out,
    `:627` ffn_norm, `:630-636` ffn out); set `res->t_inp_tokens = inp->tokens`
    (the selector needs it - fork doesn't set it today); add the logit/embd
    scaling. The DSV4 path (`build_dsv4`) is NOT touched (DFlash2 is a dense
    Qwen-class drafter).
  - Add `build_dflash2_conv` (static) and `build_post_sampling` (member) - port
    verbatim from the diff; all ggml ops they use (`ggml_fill`, `ggml_top_k`,
    `ggml_pad`, `ggml_cast`, `ggml_repeat_4d`, `ggml_tanh`, views/reshapes)
    are CONFIRMED present in the fork's ggml.
- Invocation of `build_post_sampling()`: one line in the fork's
  `llama_model::build_graph` after `build_sampling()` + the graph_max_nodes
  bump in llama-context.cpp. [details in section 4]

Convert side (may already be satisfied):
- NOT convert_hf_to_gguf.py - the changes live in the `conversion/` package:
  `conversion/__init__.py` maps `"DFlash2DraftModel": "qwen"` and
  `conversion/qwen.py` extends the existing DFlashModel class
  (`@register("DFlashDraftModel", "DFlash2DraftModel")`), emitting the 4 new
  keys when `conv_kernel_size` is in the dflash config, the selector codebook
  renames, and optional logit/softcap/embedding scales. Arch string STAYS
  "dflash" (v2 = v1 + extras, distinguished by selector presence).
  `gguf-py`: 4 keys + 7 tensor enums/names + 4 writer methods.
- The user's existing `Qwen3.8-27B-DFlash2-{Q4_K_M,Q8_0}.gguf` (X99, built
  2026-08-20, #138 notes) already carry `general.architecture=dflash`, 81
  tensors incl. the 4x5 conv + 3 selector tensors and the conv/selector
  metadata keys - the vehicles EXIST and match upstream's names, so the
  convert graft is only needed for future HF rebuilds, not to load these.
  (Re-confirm when X99 returns - it is dark now. Also +16 lines in
  examples/speculative-simple for the dists overload - port for parity.)

## 3. THE central design question: consumer dispatch (t_h_nextn)

Both heads write `res->t_h_nextn`: the fork's dspark conf head (a per-position
confidence vector, `dflash.cpp:298`) and DFlash2's selector (the packed
`[ids ; scores]` lattice). The fork's `common_speculative_impl_draft_dflash`
(`common/speculative.cpp:1136-1594`) reads `t_h_nextn` via
`llama_get_embeddings_nextn` and TODAY interprets it as dspark conf (gated by
review #9's `has_conf`). For a DFlash2 vehicle it must instead run the lattice
beam-select.

The dispatch key (UPSTREAM'S OWN DESIGN, no new API needed): the impl ctor
reads meta `dflash.selector_top_k`; `is_dflash2 = selector_top_k > 0`. A
DFlash2 vehicle then: takes `llama_set_embeddings_nextn(ctx_dft, true,
/*masked*/ false)`, sets `logits = false` on its noise-block rows (it NEVER
reads raw logits at draft time), and disables backend-sampling chains. This
slots beside the fork's `is_dspark` as a sibling mode.

## 4. Consumer algorithm (recon complete)

- **Invocation**: `build_post_sampling()` is called once in
  `llama_model::build_graph` right after `build_sampling()` (upstream
  llama-model.cpp:2462) - a 1-line addition at the fork's equivalent spot.
  Plus a `graph_max_nodes` bump in llama-context.cpp
  (`+32 * min(n_tokens, block_size * n_seq_max)` when selector_rank > 0).
- **Lattice layout** (per draft position row, packed into n_embd floats via
  t_h_nextn): `[top_k candidate token ids as F32 | top_k x top_k transition
  scores | zero-pad]`. Loader guard `n_embd >= top_k*(top_k+1)` enforces fit.
- **draft() walk**: for each block position i>=1, `row = lattice + (beg+i) *
  n_embd_dec`; `scores = row + top_k + predecessor*top_k` (the previously
  chosen candidate INDEX selects the score column). Greedy (temp<=0):
  `predecessor = argmax(scores)`, with `p_min` early-stop via
  softmax-at-argmax; token = `(llama_token) row[predecessor]`. Stochastic
  (temp>0): softmax(scores/temperature) sampled with a per-seq mt19937
  (seeded `dp.seed ^ 0x85ebca6b`, reset on begin/accept-0), truncate when
  sampled prob < p_min, and RECORD the distribution.
- **Maximal-coupling verification (the piece my first draft missed)**: temp>0
  DFlash2 drafting is only lossless with residual rejection sampling. Upstream
  adds `common_speculative_token_dist {ids, probs}`, dp fields
  `dists/temperature/seed`, a new `common_sampler_sample_and_accept_n(...,
  dists)` overload (+100 lines sampling.cpp: accept draft i iff
  `u*q <= p`, else sample clamped residual `max(0, p-q)` and stop; dedicated
  rng `sampler_seed ^ 0x9e3779b9` cloned with the sampler), and server wiring
  (pass `&slot.spec_dists` + temp + seed into draft params; use the overload
  at verify when `can_rollback && temp>0 && dists.size()==draft.size()`;
  clear dists wherever spec_draft clears). This is REQUIRED for temp>0
  serving; greedy-only serving works without it.
- **process() refactor caution**: upstream also refactors dflash `process()`
  to flat batch-order chunking + `llama_synchronize(ctx_dft)` after inject
  decodes + `selector_reset` on begin(). The fork's process() carries the #10
  rows_tgt extraction map - DO NOT copy upstream's process() wholesale (it
  would clobber #10). Port only: the synchronize call, selector_reset
  bookkeeping, and the `common_base_params_to_speculative` n_batch/n_ubatch
  raise to n_outputs_max.

## 5. Interactions with landed review fixes (must not regress)

- **#9 (has_conf)**: DFlash2 has no conf head; the `has_conf` gate stays false
  for it. The new `is_dflash2` branch is a sibling of the conf branch.
- **#17 (uniform blocks)**: the selector, like the markov head, reads the decode
  as uniform strided blocks (`n_blocks = ubatch.n_seqs_unq`, asserts
  `n_tokens % n_blocks == 0`). The #17 uniform `n_draft_uni` sizing MUST apply
  to DFlash2 too - extend the `is_dspark` guard in `draft()` (`:1460`) to
  `is_dspark || is_dflash2`.
- **#10/#51 (tree rows_tgt)**: if DFlash2 ever runs in the spec tree, its
  `process()`/extraction reads must honor the `rows_tgt` map like dflash/mtp.
  The selector reads `res->t_embd` + input tokens per block - same misindex
  risk. Gate DFlash2 out of the tree initially (roster), enable + fix later.
- **conv mixing is causal WITHIN a block**: verify against the fork's block
  layout (anchor-first for dspark; DFlash2 uses its own). The `build_dflash2_conv`
  zero-pads the first `tap` positions per block - matches a fresh draft block.

## 6. Gate plan (when implemented)

1. Build clean (CPU + CUDA75).
2. Loader gate: load `Qwen3.8-27B-DFlash2-Q8_0.gguf` as `--spec-type
   draft-dflash` for the Qwen3.8-27B target; expect the tensor count to reconcile
   (81 created, the #138 `expected 81 got 58` error GONE) and the DFlash2 log
   lines (conv kernel/group, selector rank/top-k).
3. Coherence: temp-0 greedy decode reads clean; drafting ACTIVE (draft_n > 0);
   acceptance rate sane (this is the whole point vs the MTP head).
4. Byte-identity is NOT expected vs any existing drafter (different weights);
   the gate is coherence + acceptance + no asserts, plus off-vs-on
   (spec vs no-spec) target-output identity at temp 0 (the drafter must not
   change the verified output).
5. Regate the canonical 71 byte gate (c80261ff) to prove the shared dense-path
   changes didn't perturb non-DFlash2 vehicles.
6. Fleet A/B (step 3, needs X99): DFlash2 vs MTP acceptance + t/s on Qwen3.8;
   then dual drafters DFlash2 + MTP chasing 80-100 t/s. BLOCKED until the fleet
   returns.

## 7. Proposed increments (one commit each, gated between)

1. **Model side**: arch/hparams/tensor enums + loader (v2 detection by
   selector tensor, guards) + `build_dflash2_conv` at the 4 dense-loop sites +
   `build_post_sampling` selector graph + `t_inp_tokens` populate + scales +
   build_graph call + graph_max_nodes bump. Gate: DFlash2 GGUF LOADS (81
   tensors reconcile - the #138 "expected 81, got 58" error gone), v1 dflash
   vehicle still loads, 71 byte gate green (shared paths untouched for
   non-dflash2).
2. **Greedy consumer**: `is_dflash2` mode in the dflash impl (nextn unmasked,
   logits=false rows, lattice walk with p_min early-stop, #17 uniformity
   extended to is_dflash2, selector_reset + synchronize + n_batch raise).
   Gate: temp-0 serve on Qwen3.8 target + DFlash2 drafter - coherent, drafting
   active, spec-vs-nospec target bytes identical, acceptance measured.
3. **Stochastic + maximal coupling**: token dists + sampling.cpp overload +
   server dists wiring (+ speculative-simple parity). Gate: temp-1 serve
   lossless-property spot-checks + no assert; temp-0 unchanged.
4. **Conversion parity** (optional, later): conversion/ + gguf-py graft for
   future HF rebuilds.

## 8. Open questions for the user

- **Q1 conv perf on V100**: `build_dflash2_conv` runs kernel_size taps x 4
  sites x n_layers of view/repeat/mul/add - the drafter graph gets notably
  heavier than plain dense. Assess at increment-2 gate (t/s + acceptance
  before the fleet A/B).
- **Q2 increment boundary**: OK to defer stochastic/temp>0 support to
  increment 3? Production serves run temp>0 - until increment 3 lands, a
  DFlash2 drafter is only correct for greedy serving (increment 2 would
  REFUSE temp>0 + dflash2 or fall back to no-dists acceptance = NOT lossless;
  refusing is safer).
- **Q3 tree**: gate DFlash2 out of LLAMA_SPEC_TREE for v1 (roster exclusion,
  safest - its lattice reads are not rows_tgt-aware), wire later if the tree
  lane reopens.
- **Q4 upstream refactor of process()**: NOT ported (would clobber the #10
  rows_tgt fix); only sync/reset/n_batch pieces come over. Flag if you want
  the full upstream process() shape instead - it would need re-applying #10
  on top.
