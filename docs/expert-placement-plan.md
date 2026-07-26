# Hot-Expert Placement Plan (TASKS #75)

Replace the meta backend's uniform feature-dim expert split with a
frequency-ranked, whole-expert, expert-dim split so the dual-role owners'
VRAM holds the HOT experts. Profile evidence and GO decision: #74
(hy3 coverage@25.5% = 0.913 vs 0.255 uniform; docs/expert-profiling.md).
Projected: member-RAM expert traffic -8.5x, decode 5.06-5.22 -> ~8-10 t/s.

Design agreed 2026-07-24: permutation + contiguous split; config artifact
`LLAMA_META_EXPERT_PLACEMENT=<json>`; branch `expert-placement`; truncated-hy3
CPU-loopback gates before any fleet A/B.

**STATUS 2026-07-26.** Shipped and gated: split policy, permuted upload, remap +
mask tables, and the mul_mat_id SKIP SENTINEL that replaced v1's dummy-slot-0
encoding (that encoding produced duplicate ids per token and crashed CUDA - see
section 5). Gates: off/identity/skew byte-identical on the CPU loopback, CUDA
PPL-neutral, no regression on the MTP serving path. NOT yet measured: the fleet
A/B (gate 5) - it needs all workers on an image carrying the sentinel, and the
projected numbers in the line above are still PROJECTIONS, not results.

## 0. Why the feature-dim split cannot bank coverage

Today `ffn_{gate,up,gate_up}_exps.weight` split AXIS_1 and `ffn_down_exps.weight`
AXIS_0: every member holds a row-slice of EVERY expert, so each member reads its
slice regardless of which expert fires - VRAM hit fraction == byte fraction by
construction. Hot-placement needs whole-expert ownership = a dim-2 (expert axis)
split with a per-layer permutation so each member's expert set is contiguous in
permuted order.

## 1. Artifact chain

1. `LLAMA_EXPERT_PROFILE=<path>` (#74, shipped) -> per-layer router histogram.
2. `scripts/expert-placement.py profile.json --ts <expert shares> -o placement.json`
   (shipped, this branch): per-layer hottest-first permutation `perm[il][k]` =
   original expert id at permuted position k; per-member counts by
   largest-remainder rounding of the shares; coverage report. v2 =
   entropy-weighted per-layer counts, generator-only change (schema already
   per-layer).
3. `LLAMA_META_EXPERT_PLACEMENT=placement.json` on the coordinator enables the
   runtime placement. Unset = bit-for-bit today's behavior (workflow rule:
   default-off env gate).

Measured on hy3-full profile at the record split (21,21,46,50,27): owners
(members 0+1) hold 49/192 experts (25.5%) capturing **91.3%** of routed
selections; +bucketA merge 91.2%.

## 2. Runtime mechanism

Three cooperating pieces, all keyed off the placement being loaded:

### 2a. Split policy (llama-model.cpp, get_tensor_config)

`_exps` weights get `GGML_BACKEND_SPLIT_AXIS_2` with per-member `ne` =
`counts_per_layer[il]` (instead of AXIS_1/AXIS_0 feature cuts). Expert tensors
are `[hidden, n_ff, n_expert]`; dim-2 chunks are whole experts and
byte-contiguous, so quant granularity is never cut (any block size works - a
whole expert is always block-aligned).

### 2b. Permuted upload (llama-model-loader, NOT the meta backend)

The permutation lives at the SOURCE: `load_all_data` permutes the dim-2 expert
chunks of a placed `_exps` tensor in the staging buffer before the normal
`ggml_backend_tensor_set` (each chunk = one expert = `nb[2]` contiguous bytes;
a one-pass chunk copy). The logical tensor the meta buffer sees is already in
permuted (hottest-first) order, so the EXISTING AXIS_2 contiguous chunk-splice
path lands member j's experts with no meta-backend upload changes at all.
Placed tensors bypass the mmap-alias and chunked-upload fast paths (they need
a mutable staging copy).

Consequences, recorded: (a) #44 slice provenance cannot serve permuted bytes
(hash mismatch vs the file) - placed tensors cold-stream once, then the worker
cache holds the permuted slices and warm reloads hit as usual; (b) any
debug/read-back of a placed `_exps` weight sees permuted expert order.

### 2c. Routed compute: id remap + weight mask (graph side, build_moe_ffn)

Members can only compute the (token, k) pairs whose expert they own. Rather
than per-member subgraphs (the meta backend builds ONE structural graph), two
constant per-layer lookup tables turn ownership into data:

- `exp_remap[il]`: `[n_expert]` per member - global expert id -> member-LOCAL
  slot for owned experts, the SKIP SENTINEL `-1` for non-owned
  (`LLAMA_EXPERT_SLOT_SKIP`). Split state MIRRORED with per-member CONTENTS
  (uploaded via a member-targeted set API) - a benign state lie: the contents
  differ per member, and the state only has to be consistent.
  **v1 used local slot 0 for non-owned lanes and that was WRONG** (2026-07-26):
  mul_mat_id requires a token's ids to be DISTINCT, so collapsing lanes onto one
  slot corrupted the CUDA id helper's index arithmetic - see section 5. The
  sentinel also means a member reads NO weights for lanes it does not own, which
  is what makes the projected traffic reduction reachable at all.
- `exp_mask[il]`: F32 `[n_expert]` per member - 1.0 owned, 0.0 non-owned.
  Registered MIRRORED with per-member CONTENTS (same member-targeted upload as
  the remap); the member masks SUM to the all-ones vector, which is what makes
  the name-tagged PARTIAL step below exact.

Graph (env-gated branch in build_moe_ffn):

- `ids_local = get_rows(exp_remap, selected_experts)` feeds mul_mat_id/add_id.
  Non-owned lanes carry `-1`; the backend skips them and their dst rows stay
  zeroed (CUDA zeroes dst up front, the CPU binning loop does the same).
- `weights = mul(weights, get_rows(exp_mask, selected_experts))` AFTER weight
  normalization (norm_w must see the mirrored un-masked weights or member-local
  sums would diverge), zeroing the gating weight of non-owned pairs. With the
  sentinel their expert output is already zero, so the mask is now belt-and-braces
  for the value - but it is still LOAD-BEARING for the split derivation below,
  which is why it stays.

**Derivation chain (the part the first draft of this plan got wrong):** with
only masking, every intermediate would derive MIRRORED and no reduce boundary
would ever fire - members' differing values would never reconcile. The PARTIAL
tag on the masked expert product fixes the derivation:

```text
exp_mask                          MIRRORED  (per-member contents; masks sum to ones)
get_rows(exp_mask, ids)        -> MIRRORED  (state; the contents differ per member)
mul(weights MIR, mask rows)    -> MIRRORED  (state; only owned lanes stay nonzero)
mul(experts, w_masked)         -> PARTIAL   (name-tagged 'ffn_moe_weighted_placed'
                                             rule in handle_bin_bcast - each member's
                                             nonzero lanes are exactly its owned
                                             pairs, so the member sum is the logical
                                             value)
expert-sum ADD chain           -> PARTIAL   (existing rule - today's expert-sum pattern)
```

The delayed AllReduce fires at the SAME boundary as the uniform split -
**boundary count unchanged**, the latency law (#49) untouched; only
member-local bytes-read shrink. Gate/up/down mul_mat_id outputs ride as
MIRRORED with identical shapes on every member (ids count is fixed); their
non-owned lanes differ across members but nothing consumes them before the
mask zeroes their contribution.

The tables are static per-layer leaves created at load in a small model-owned
meta-buffer context (not in the GGUF), named `blk.<il>.exp_remap` /
`blk.<il>.exp_mask` (the split policy name-matches them). Because their
CONTENTS differ per member while plain meta set_tensor writes all members
identically (MIRRORED) or divides (PARTIAL), they are written through a new
`ggml_backend_meta_tensor_set_member(tensor, j, ...)` API that targets one
member's shadow directly (an RPC member's shadow streams to the worker through
the normal RPC buffer path). exp_remap is I32 (get_rows returns I32 for I32
src; CPU+CUDA both ship I32 get_rows kernels). Ids are gathered FLAT
(`[k*n_tokens]`, table `[1, n_expert]`) to satisfy get_rows' dim-2 matching,
then reshaped back.

Additional v1 guards (abort loudly at load): models with routed-expert biases
(`ffn_*_exps.bias` - both the PARTIAL down-bias trick and add_id gate/up
biases) and `weight_before_ffn` arches (mask ordering differs); every member
must own >= 1 expert on every placed layer (a zero-count member's remap has no
valid local slot).

### 2d. Shared/dense/bias interactions

- Shared experts, router, norms: untouched (mirrored today, stay mirrored).
- `ffn_down_exps.bias` (PARTIAL /n_bufs trick): incompatible with ownership
  masking - if the model HAS routed-expert biases, ABORT loudly at load with
  placement on (hy3/GLM: none; revisit if a target model needs it).
- `add_id` gate/up biases: same rule.
- MTP/nextn expert tensors (#71 stage 1 localizes draft): out of scope v1;
  placement applies to the main model's `_exps` only.

## 3. Consistency check (loud, load-time)

On load with placement set: n_expert and n_layer must match the model; every
`perm[il]` must be a bijection over [0, n_expert); `counts_per_layer[il]` must
sum to n_expert and have one entry per meta member; member count must equal the
meta device's member count. Any mismatch = hard error naming the field.
Log one INFO line per model: aggregate owner coverage implied by the profile
(from the JSON metadata) so the serve's expected hit fraction is on record.

## 4. Validation staircase (gates before any fleet touch)

Vehicle: `/mnt/files/hy3-trunc5-mtp.gguf` + CPU loopback workers (dev-container
build, docs/dev-workflow.md §1c). CPU meta paths are deterministic - byte gates
are valid here (NOT on the GPU fleet - gotcha #4).

**RE-RUN 2026-07-26 against the skip-sentinel build** (gates 1-3 below were
first measured against the v1 dummy-slot encoding, which is gone). Vehicle:
trunc-hy3, 2 loopback RPC members, temp 0, seed 1234, 24 tokens, sha over the
generated answer only - the loading spinner and the t/s banner are timing
dependent and must be excluded. All four hashed identically (`f61930b5`):
placement off on the pre-fix build, placement off on the sentinel build,
identity artifact, hot-first under `GGML_META_NO_DELAY=1`, and hot-first with
the delayed reduce. Placement was confirmed ACTIVE in the placed runs via
`EXPERT_AUDIT` (owned+skipped == computed every graph, 0 violations over 2376
pairs) - identical shas alone cannot distinguish a pass from a silent
artifact-load fallback, so always check that.
Harness notes: this fork's CLI needs `-st` or it parks in
`console::readline_advanced`; `-DLLAMA_UI_GZIP=OFF` is required when `/src` is
mounted read-only.

1. **Off-gate**: placement env unset on this branch == master build,
   byte-identical (feature is a no-op). Since the sentinel touches shared
   mul_mat_id code, this gate now also covers every NON-placement MoE model.
2. **Identity gate**: placement JSON with identity permutation + uniform counts
   == placement-off, byte-identical (upload path + remap/mask plumbing exact).
3. **Skew gate**: real hottest-first placement == placement-off,
   byte-identical (permutation must not change math - each expert's arithmetic
   is identical, the weighted sum is per-pair and order-independent).
   **MEASURED 2026-07-24: exact under GGML_META_NO_DELAY=1 (all five variants
   byte-identical to OFF); the delayed-reduce path diverges for strongly
   skewed counts (32/160, 160/32) by fp summation GROUPING only -
   deterministic, boundaries identical (GGML_META_DEBUG_REDUCE diff empty).
   PPL gate settles it: trunc-hy3 8-chunk wikitext - OFF 119683.40 +/- 3655,
   identity 119673.89, skew-160/32 119694.74 - spread 0.009%, ~300000x inside
   the error bar. Quality-neutral rounding class (dev-workflow §5).**
4. **Ownership audit**: GGML_META_DEBUG counters - per member, mul_mat_id rows
   computed vs owned; a member computing a non-owned pair (beyond the masked
   slot-0 dummies) is a bug even if output matches.
   **MEASURED 2026-07-24: PASS. Implemented as (a) a load-time static audit of
   the built tables (true partition, owned slots within counts, non-owned -> 0;
   loud named-field error, section 3 style) and (b) runtime counters in the meta
   backend reading back the member shadows of the member-local ids and mask rows
   after each graph (GGML_META_DEBUG>=1; `EXPERT_AUDIT:` per-graph + FINAL
   lines). Gotcha that cost a day: an end-of-graph readback of the flat gathers
   is STALE by default - the ggml allocator recycles their cell once consumed
   (both gathers even alias one address), so the audit saw a recycled mirrored
   F32 tensor identical on all members. Fix: under the debug env, build_moe_ffn
   pins the two flat gathers with ggml_set_output (never freed / never
   overwritten); value-neutral (text sha identical pinned vs not). CPU loopback,
   2 members, trunc-hy3, 24 tok temp-0: uniform [96,96] - member 0 owned 2751 /
   dummy 633, member 1 owned 633 / dummy 2751, computed 3384 each, owned sums to
   computed, violations 0; skew [32,160] - 1506/1878, violations 0; placement
   off - zero audit output. Negative control: GGML_META_DEBUG=2 selftests fire
   (injected bad table caught at load; injected non-owned non-dummy pair warns),
   and a hand-corrupted artifact hard-errors at load naming the field.**
4b. **Ownership audit ON CUDA** (added 2026-07-26, was the hole that let a fatal
   bug through gates 1-4): gates 1-4 all ran on CPU loopback, where an
   out-of-range expert id is a silent garbage read that the mask usually zeroes.
   The same graph on CUDA faults. Vehicle - 2 minutes, no fleet:

   ```bash
   docker run -d --name llama-ep-trunc --gpus all \
     -v "$BUILD:/srcbin:ro" -v "$ART_DIR:/place:ro" -v /mnt/files:/models:ro --network host \
     -e LD_LIBRARY_PATH=/srcbin/bin -e GGML_META_DEBUG=1 \
     -e LLAMA_META_EP_ONLY=1 -e LLAMA_META_ATTN_OWNER=0,1 -e LLAMA_META_ALLOW_MULTI_LOCAL=1 \
     -e CUDA_VISIBLE_DEVICES=0,1 -e LLAMA_META_EXPERT_PLACEMENT=/place/trunc-place-4.json \
     --entrypoint /srcbin/bin/llama-server nvidia/cuda:12.8.1-devel-ubuntu24.04 \
     -m /models/hy3-trunc5-mtp.gguf --override-kv hy_v3.block_count=int:5 \
     --device CUDA0,CUDA1 -sm tensor -ts 1,1 -ngl 99 --no-mmap \
     -c 4096 -ub 256 -b 256 --host 0.0.0.0 --port 8099 -np 1 -fit off
   ```

   The artifact is a 4-layer/2-member placement (the model keeps
   `block_count=81` in its metadata, so the override is what makes n_layer=4;
   generate from a truncated copy of `profiles/hy3-full.json`). PASS = zero
   `EXPERT_AUDIT: WARN` lines across at least two DIFFERENT graph shapes
   (prefill then decode) - one clean graph proves nothing, the failure only
   appears on the first execution of a NEWLY BUILT shape. Run the identity
   permutation too: it is the cheapest way to separate a mechanism bug from a
   ranking bug.
5. **Fleet A/B** (only after 1-4b): record hy3 EP config, uniform vs placement,
   same image, coherence-read + t/s; optionally PPL spot-check. Expectation:
   owner VRAM hit fraction ~91% vs 25.5%, decode toward 8-10 t/s.
   **BLOCKED 2026-07-26** by the CUDA fault above - see TASKS.md #75 for the
   evidence chain and the meta-backend shadow-ring root-cause hypothesis.

## 5. Open risks

- mul_mat_id on a dim-2-sliced shadow must accept local ids in [0, cnt_j) -
  true for all backends (ne[2] is the expert count it sees). **THIS RISK WAS
  UNDER-STATED AND IT IS WHAT BROKE GATE 5 (2026-07-26).** In-range is not
  enough: ggml's CUDA mul_mat_id also requires a token's ids to be DISTINCT,
  because top-k normally selects distinct experts. `mm_ids_helper` (mmid.cu:44)
  keeps ONE lane per (token, expert) in `iex_used` while `nex_prev` counts every
  lane below that expert index, and the host sorted path does the same with a
  `break` after the first match (ggml-cuda.cu:1997). Section 2c maps EVERY
  non-owned lane onto local slot 0, so one token lands up to n_expert_used lanes
  on the same slot; `expert_bounds` and the compact row space then disagree and
  `quantize_mmq_q8_1` writes outside its staging buffer (compute-sanitizer:
  `Invalid __global__ write of size 4 bytes`). The CPU backend loops lanes
  independently and is immune - which is exactly why gates 1-4 passed.
  The skip-sentinel fix below is therefore MANDATORY, not a v2 nicety.
- get_rows on I32 src (remap table): verify backend support; fall back to
  f32 table + cast if not.
- Prefill: a big ubatch routes to most experts - masked pairs still ride the
  mul_mat_id call. **CORRECTED 2026-07-26:** the bound holds for the pair COUNT
  only, not for the work. Today every member computes every pair on a ROW SLICE
  (dst derives split), while placement makes the mul_mat_id dst MIRRORED, so each
  member does FULL-width GEMMs for every pair: per-member expert FLOPs rise ~n_members
  (5x on the record roster) and the MoE intermediates grow the same way. Harmless
  on the V100 owners, but the CPU/RAM workers do 5x their previous expert math -
  budget a prefill regression and watch whether a worker becomes the new critical path.
- **The dummy slot is a traffic floor** (2026-07-26): a member reads its owned
  selected experts PLUS local slot 0 for every non-owned lane, i.e. one full expert
  (11.3 MB on hy3) per layer even when it owns nothing in that token. For a cold
  member (~3% of selections) that is 0.24 + 1.0 experts/layer against 2.23 in the
  uniform split - a ~1.8x traffic cut, not the projected 8.5x. Slot 0 is the member's
  HOTTEST expert by construction (perm is hottest-first, ranges are contiguous), which
  is why the owners barely feel it and the cold members pay it in full. v2 fix: a skip
  sentinel in mul_mat_id ids (write zero rows, read no weights) makes non-owned pairs
  actually free and retires the mask.
- Graph build cache: tables are static leaves (upload-time registration), no
  per-token rebuild; uid/content keys unaffected.
- `-ts` for non-expert tensors is unchanged; EXPERT shares come from the
  placement JSON when set (the JSON's member_shares should match the serve's
  expert -ts; the consistency check warns on mismatch). **NOT IMPLEMENTED as of
  2026-07-26** - `member_shares` is parsed and never compared, so whole-expert
  rounding shifts bytes silently: `[25,24,54,58,31]` against `-ts 21,21,46,50,27`
  puts ~0.5 GB MORE expert weight on CUDA0 than the control leg (measured 25485
  vs 24853 MiB used). With `-fit off` and `LLAMA_FLEET_CAPACITY_CHECK=0` nothing
  catches it, and the fleet gate is a POOLED check that would not catch a
  per-device shortfall anyway.
- Load-time gates cover routed-expert biases but NOT `ffn_*_exps.scale` /
  `input_scale` (absent on hy3/GLM Q4_K): those would take a feature-axis split
  against expert-axis weights. Add the same style of gate before a model that
  carries them meets placement.

## 6. Regenerating the artifact for a serve roster (fleet A/B prerequisite)

The placement artifact is ROSTER-SPECIFIC, not just model-specific: its
`counts_per_layer` rows have one entry per META MEMBER, in meta device order,
and the runtime consistency check hard-errors when that count differs from the
serve's member count. Any change to the roster or to the expert `-ts`
invalidates the artifact - regenerate it, do not reuse.

Rules:

1. **Member order = meta device order = the order of `--device`/EP_DEVICES**
   (e.g. the hy3 record serve CUDA0,CUDA1,RPC0,RPC1,RPC2 = local V100 owners
   first, then local cpu worker, .11, .15). `--ts` for the generator must be
   given in that same order.
2. **Shares = the serve's EXPERT `-ts`, not layer shares.** For the record
   dual-role config that is `--ts 21,21,46,50,27` (owners hold 21 GB of
   experts each). Zero-share members are not representable in v1 (every
   member must own >= 1 expert per placed layer - the table-builder guard);
   a pure attention owner with 0% experts would need the v1 guard relaxed
   (give it a minimal share instead).
3. **Hot-first ordering does the placement**: the generator ranks experts
   hottest-first per layer, and members receive contiguous runs in `--ts`
   order - so the members listed FIRST get the hottest experts. Owners
   (VRAM) must therefore come first in the roster, which they already do in
   the dual-role EP configs.
4. **Interaction with LLAMA_META_EP_ONLY / ATTN_OWNER**: placement only
   overrides the `_exps` weights of placed layers; the dedicated-attention
   and mirror rules are untouched. The placement JSON's shares must be
   regenerated whenever the EP `-ts` is retuned (e.g. the 18->21 GB owner
   bump was a different artifact-shares config than the earlier record).

Example (hy3 record roster, both-domain profile merge):

```bash
scripts/expert-placement.py profiles/hy3-full.json \
    --profile-b profiles/hy3-bucketA.json \
    --ts 21,21,46,50,27 -o placements/hy3-record-21-21-46-50-27.json
```

Name the artifact after model+shares (as above) so a stale artifact is
visible at a glance; keep them under `placements/` next to `profiles/`.
The generator's coverage report prints the owners' expected hit fraction -
record it with the serve (hy3-full at this split: members 0+1 = 25.5% of
experts covering 91.3% of routed selections).
