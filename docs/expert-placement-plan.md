# Hot-Expert Placement Plan (TASKS #75)

Replace the meta backend's uniform feature-dim expert split with a
frequency-ranked, whole-expert, expert-dim split so the dual-role owners'
VRAM holds the HOT experts. Profile evidence and GO decision: #74
(hy3 coverage@25.5% = 0.913 vs 0.255 uniform; docs/expert-profiling.md).
Projected: member-RAM expert traffic -8.5x, decode 5.06-5.22 -> ~8-10 t/s.

Design agreed 2026-07-24: permutation + contiguous split; config artifact
`LLAMA_META_EXPERT_PLACEMENT=<json>`; branch `expert-placement`; truncated-hy3
CPU-loopback gates before any fleet A/B.

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
  slot for owned experts, 0 (any valid local slot) for non-owned. Split state
  MIRRORED with per-member CONTENTS (uploaded via a member-targeted set API) -
  a benign state lie: the remap only chooses which garbage lane non-owned
  pairs compute, and those lanes are zeroed by the mask below.
- `exp_mask[il]`: F32 `[n_expert]` per member - 1.0 owned, 0.0 non-owned.
  Split state **PARTIAL - and honestly so**: each expert is owned by exactly
  one member, so the member masks SUM to the all-ones vector, which is
  precisely PARTIAL's contract (shadows sum to the logical value).

Graph (env-gated branch in build_moe_ffn):

- `ids_local = get_rows(exp_remap, selected_experts)` feeds mul_mat_id/add_id.
- `weights = mul(weights, get_rows(exp_mask, selected_experts))` AFTER weight
  normalization (norm_w must see the mirrored un-masked weights or member-local
  sums would diverge), zeroing the gating weight of non-owned pairs. The
  non-owned mul_mat_id output (computed against the harmless local slot 0) is
  finite garbage that contributes exactly 0 to the weighted sum.

**Derivation chain (the part the first draft of this plan got wrong):** with
only masking, every intermediate would derive MIRRORED and no reduce boundary
would ever fire - members' differing values would never reconcile. The PARTIAL
mask fixes the derivation:

```text
exp_mask                          PARTIAL   (honest: masks sum to ones)
get_rows(exp_mask, ids)        -> PARTIAL   (gather of summands; new rule in handle_get_rows)
mul(weights MIR, mask PARTIAL) -> PARTIAL   (mul by a mirrored factor distributes
                                             over the member sum; new rule in handle_bin_bcast)
mul(experts MIR, w_masked)     -> PARTIAL   (same rule)
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

1. **Off-gate**: placement env unset on this branch == master build,
   byte-identical (feature is a no-op).
2. **Identity gate**: placement JSON with identity permutation + uniform counts
   == placement-off, byte-identical (upload path + remap/mask plumbing exact).
3. **Skew gate**: real hottest-first placement == placement-off,
   byte-identical (permutation must not change math - each expert's arithmetic
   is identical, the weighted sum is per-pair and order-independent).
4. **Ownership audit**: GGML_META_DEBUG counters - per member, mul_mat_id rows
   computed vs owned; a member computing a non-owned pair (beyond the masked
   slot-0 dummies) is a bug even if output matches.
5. **Fleet A/B** (only after 1-4): record hy3 EP config, uniform vs placement,
   same image, coherence-read + t/s; optionally PPL spot-check. Expectation:
   owner VRAM hit fraction ~91% vs 25.5%, decode toward 8-10 t/s.

## 5. Open risks

- mul_mat_id on a dim-2-sliced shadow must accept local ids in [0, cnt_j) -
  true for all backends (ne[2] is the expert count it sees).
- get_rows on I32 src (remap table): verify backend support; fall back to
  f32 table + cast if not.
- Prefill: a big ubatch routes to most experts - masked pairs still ride the
  mul_mat_id call. Cost is bounded by today's behavior (every member already
  computes every pair today, on row slices). Decode is the win target.
- Graph build cache: tables are static leaves (upload-time registration), no
  per-token rebuild; uid/content keys unaffected.
- `-ts` for non-expert tensors is unchanged; EXPERT shares come from the
  placement JSON when set (the JSON's member_shares should match the serve's
  expert -ts; the consistency check warns on mismatch).

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
