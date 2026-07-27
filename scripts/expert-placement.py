#!/usr/bin/env python3
"""TASKS #75: generate an expert-placement artifact from a #74 profile.

Turns a LLAMA_EXPERT_PROFILE histogram into the JSON consumed by
LLAMA_META_EXPERT_PLACEMENT: per MoE layer, a frequency-ranked PERMUTATION of
expert IDs plus per-member expert counts. The runtime places experts
contiguously in permuted order (hottest first), so the members listed first
(the VRAM owners) hold the hottest experts — the permutation+contiguous-split
design agreed for #75.

Placement is greedy-by-frequency per layer (optimal for a per-layer budget —
SlimCaching/Prism). v1 uses a uniform per-layer budget: every layer splits its
n_expert experts across members by the same fractions. v2 (entropy-weighted
per-layer budgets) only changes this generator — the JSON schema already
carries explicit per-layer counts.

Usage:
  expert-placement.py profile.json --ts 21,21,46,50,27 -o placement.json
  # --ts: expert-share fractions per meta member, aligned with the serve's
  #       expert -ts (same order, same meaning: bytes of experts per member).
  #       Members are listed in META DEVICE ORDER; hot experts go to the
  #       EARLIEST members with nonzero share, so put VRAM owners first
  #       (they already are, in the dual-role EP configs).

The runtime consistency check (loud, load-time) validates n_expert/n_layer
against the model and that each layer's permutation is a bijection.
"""
import argparse
import hashlib
import json
import math
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profile", help="LLAMA_EXPERT_PROFILE json (from #74)")
    ap.add_argument("--ts", required=True,
                    help="comma list of per-member expert shares (fractions or weights, "
                         "normalized; order = meta member order = expert -ts order)")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--profile-b", help="optional second-domain profile: counts are summed "
                                        "so the hot ranking reflects both traffics")
    args = ap.parse_args()

    def load_json(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            sys.exit(f"cannot read profile '{path}': {e}")

    prof = load_json(args.profile)
    n_layer  = prof["n_layer"]
    n_expert = prof["n_expert"]
    counts   = [list(c) for c in prof["counts"]]

    if args.profile_b:
        prof_b = load_json(args.profile_b)
        assert prof_b["n_layer"] == n_layer and prof_b["n_expert"] == n_expert, \
            "profile-b shape mismatch"
        # normalize each profile to equal total weight before summing so a longer
        # run does not dominate the ranking
        tot_a = sum(sum(c) for c in counts) or 1
        tot_b = sum(sum(c) for c in prof_b["counts"]) or 1
        for il in range(n_layer):
            counts[il] = [a / tot_a + b / tot_b
                          for a, b in zip(counts[il], prof_b["counts"][il])]

    try:
        shares = [float(x) for x in args.ts.split(",")]
    except ValueError as e:
        sys.exit(f"bad --ts '{args.ts}': {e}")
    if not all(math.isfinite(s) and s >= 0 for s in shares):
        sys.exit(f"--ts entries must be finite and non-negative: {shares}")
    if not shares or all(s == 0 for s in shares):
        sys.exit("--ts must contain at least one nonzero share")
    total = sum(shares)
    fracs = [s / total for s in shares]

    # per-member expert counts (uniform per-layer budget, v1): largest-remainder
    # rounding so counts sum exactly to n_expert
    base = [f * n_expert for f in fracs]
    cnt = [math.floor(b) for b in base]  # shares validated finite/non-negative above
    rem = n_expert - sum(cnt)
    order = sorted(range(len(fracs)), key=lambda j: -(base[j] - cnt[j]))
    for j in order[:rem]:
        cnt[j] += 1
    assert sum(cnt) == n_expert

    layers = []
    n_moe = 0
    for il in range(n_layer):
        c = counts[il]
        if sum(c) == 0:
            layers.append(None)   # dense / never-routed layer: no placement
            continue
        n_moe += 1
        # hottest-first permutation; ties broken by expert id for determinism
        perm = sorted(range(n_expert), key=lambda e: (-c[e], e))
        layers.append(perm)

    try:
        with open(args.profile, "rb") as f:
            prof_sha = hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError as e:
        sys.exit(f"cannot hash profile '{args.profile}': {e}")

    out = {
        "model":            prof.get("model"),
        "n_layer":          n_layer,
        "n_expert":         n_expert,
        "source_profile":   {"path": args.profile, "sha256_16": prof_sha,
                             "tokens": prof.get("tokens_profiled")},
        "member_shares":    fracs,
        # per-layer expert counts per member; identical rows in v1, but the
        # schema is per-layer so entropy-weighted budgets (v2) need no runtime change
        "counts_per_layer": [cnt if p is not None else None for p in layers],
        # perm[il][k] = ORIGINAL expert id placed at permuted position k;
        # member j owns permuted positions [sum(cnt[:j]), sum(cnt[:j+1]))
        "perm":             layers,
    }
    try:
        with open(args.output, "w") as f:
            json.dump(out, f)
    except OSError as e:
        sys.exit(f"cannot write '{args.output}': {e}")

    # report: what fraction of routed selections the first member(s) capture
    print(f"model={out['model']} n_expert={n_expert} moe_layers={n_moe} members={len(fracs)}")
    csum = 0
    for j, k in enumerate(cnt):
        csum += k
        cov = []
        for il in range(n_layer):
            if layers[il] is None:
                continue
            tot = sum(counts[il]) or 1
            cov.append(sum(counts[il][e] for e in layers[il][:csum]) / tot)
        mean_cov = sum(cov) / len(cov) if cov else 0.0
        print(f"  member {j}: {k:4d} experts/layer ({fracs[j]:6.1%})  "
              f"cumulative coverage {mean_cov:6.1%} (uniform {csum/n_expert:6.1%})")


if __name__ == "__main__":
    main()
