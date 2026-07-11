# Expert-Parallel Distributed Workers — Design & Feasibility (TASKS.md #28)

Phase 0 deliverable (2026-07-11): design, measured-fact projections, and gated
increments for distributing MoE inference with the **expert** as the unit, across
the real 4-worker fleet (see `distributed-inference-guide.md` §3b for the fleet).

## 1. Why the expert is the right unit — the composition law

Task #31 measured the fundamental limit of layer pipelines: per-token time is the
**SUM of stage times plus hops**, so every box — including the slowest — is on the
critical path of every token (35B across 5 machines: 1.9 t/s vs 108 on the V100s
alone). Expert parallelism changes the composition law:

```
pipeline:         t_token = Σ_stages t_stage           (slow box taxes EVERY token)
expert-parallel:  t_token = t_attn + Σ_moe_layers max_{contacted workers} t_expert
                            (a worker is contacted only if the router picked
                             one of ITS experts for this token)
```

Consequences (the "smallest computable chunk" principle):
- A slow box owning **few, cold** (rarely-routed) experts contributes capacity
  while almost never gating a token. Its cost when contacted is bounded by ONE
  expert FFN + one activation round trip — not a model slab.
- The tunables are *how many* and *which* experts each box owns. Placement is a
  continuous dial between "contributes nothing, costs nothing" and "full member."
- What matters for a slow box is its **latency and link**, not its FLOPS: the
  per-contact payload is tiny (activations), the compute is one small FFN.

## 2. What crosses the network (sizes from the real models)

Per MoE layer, per token, per contacted worker:
- coordinator → worker: the token's hidden state, `n_embd × 2-4 B` ≈ **8-16 KB**
  (Qwen3.6-35B n_embd=4096 f16 ⇒ 8 KB; batch B tokens ⇒ B× that, shared per worker)
- worker → coordinator: `(experts owned ∩ routed) × n_embd × 4 B` ≈ **16-48 KB**
- merge on the coordinator: weighted add of k vectors — trivial.

Per token totals (Qwen3.6-35B-A3B: 40-ish MoE layers, k=8; DeepSeek-V4-Flash:
43 layers, k=6+1): ~0.7–2 MB/token spread over all contacted workers — GbE does
~120 MB/s ⇒ **wire time ~6–16 ms/token**, and the *latency* floor is
`layers × RTT` ≈ 43 × 0.3 ms ≈ **13 ms/token** on this LAN (sequential layers).
Compare the targets below: this tax is decisive for fast models, minor for the
big-model capacity play.

## 3. The prize (projections from measured numbers)

Reference vehicle: **DeepSeek-V4-Flash 81 GB** (77.9 GB experts, 8.8 GB
non-expert; measured layout in `ssd-streaming-plan.md` §8). It fits NOWHERE in
the fleet today except SSD-streamed on one V100 box at **2.7-3.1 t/s** (73% RAM
hit @30 GB budget).

Expert-parallel across the fleet's pooled RAM (64+64+32+16 = **176 GB — the
whole expert set becomes RAM-resident**):
- per-token expert bytes touched: ~1.83 GB (cold) spread across owners in
  parallel: `t_compute ≈ max_i(bytes_i / bw_i)`; balanced by bandwidth
  (~34+21+60 GB/s CPU boxes + V100 coordinator share) ⇒ **~15-30 ms**
- + activation wire/latency ≈ 15-25 ms ⇒ **~25-55 ms/token ⇒ 18-40 t/s**
  vs 2.7 t/s today — a **7-15× runnability win** on hardware already owned.
That is the case that justifies the build. For models that fit one fast box
(35B on V100s), EP is still a loss — same conclusion as #31.

## 4. Design

Coordinator (V100 box) runs everything except routed-expert FFNs: attention,
norms, router, shared experts, KV. Workers own **expert subsets** per layer and
serve `expert_ffn(x, expert_id)` calls.

Key insight for the implementation route: **the meta tensor-split backend
already implements intra-layer expert segmentation** (`ffn_*_exps` split
AXIS_1-segmented across devices, `mul_mat_id` PARTIAL + delayed AllReduce —
validated on Qwen A3B and GLM in tensor mode). The missing piece is running the
meta device over **RPC member devices** instead of local CUDA devices. The RPC
backend now has async + events (proto 4.2) and per-endpoint containment (29b).
The per-layer "AllReduce" for a segmented expert sum is just the sum of each
member's partial expert-output vector — **k × n_embd × 4 B per layer**, NOT a
row-split GEMM reduce; Ethernet handles that fine (this is why #17/#31's
"Ethernet AllReduce is dead" verdict does not apply: that verdict was for
per-GEMM row splits).

Placement policy (the user-guidance dial, informed by #31):
1. **Static bandwidth-weighted** (increment 1): expert count per worker ∝
   measured per-box memory bandwidth; attention/shared experts mirrored on the
   coordinator (never cross the wire).
2. **Hot/cold by routing stats** (increment 2): the ssd-streaming work already
   proved routing skew is measurable online (LRU hit rates by expert). Hot
   experts → coordinator VRAM/fast boxes; the cold tail → slow boxes (the i5
   holds experts it serves a few times a minute — capacity, ~zero gating).
3. **Dynamic migration** (increment 3, stretch): move experts when observed
   contact latency × frequency says so; the beacon (29d) carries free_mib and
   has room for a bandwidth score.

## 5. Gated increments

- **0. This document** — design + projections from measured facts. ✅
- **1. Meta-over-RPC spike**: build a coordinator-side meta device whose
  members are RPC devices (CPU workers); run Qwen3.6-35B-A3B with experts
  segmented across 2 workers, attention local. Gates: coherent temp-0 output;
  PPL == single-box reference; per-token wire bytes within 2× of §2 estimates.
  Known risks: meta assumes member backends support the reduce path it uses
  (generic fallback is coordinator-bridged get/sum/set — correct, one extra
  hop; optimize later via W2W); graph-build cost per layer.
- **2. DeepSeek-81GB across the fleet**: the prize run. Gates: loads with
  experts RAM-resident across ≥3 workers (--model-dir makes this cheap);
  decode ≥ 3× the 2.7 t/s SSD baseline; coherent output; PPL sane.
- **3. Placement policy**: bandwidth-weighted then hot/cold (routing-stat
  export from the router — the ssd-stream debug counters are the template).
  Gate: measurable tail-latency reduction vs uniform placement.
- **4. Batching/multi-stream**: EP naturally batches per-worker (all tokens'
  hits on a worker's experts in one call) — revisit the #31 -np finding here;
  EP is the path where MoE batch throughput CAN scale.

## 6. Open questions (tracked, not blocking increment 1)

- Router on coordinator needs each token's router logits before expert calls —
  already local (router weights are non-expert, mirrored).
- KV/prefill: prefill ships B×8 KB per layer per worker — fine (batch-amortized).
- Failure: a dead expert-owner mid-token → 29b containment fails the request;
  re-placement of its experts = restart with rediscovery (29b loop) — experts
  reload from --model-dir locally.
- Quantized expert compute on old CPUs (i5-3210M lacks AVX2 VNNI etc.): its
  per-expert FFN cost is the "contacted-worker max" — measure in increment 1.
