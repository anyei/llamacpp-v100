#!/usr/bin/env python3
"""TASKS #74/#75: analyze an LLAMA_EXPERT_PROFILE histogram.

For each MoE layer, sort experts by selection count and report what fraction of
routed activations the top-K% hottest experts capture ("coverage"). Uniform
routing would give coverage == K; the gap is the prize hot-expert placement
(#75) can bank by keeping the hot set in owner VRAM instead of a uniform slice.

Usage: expert-coverage.py profile.json [profile-b.json]
With a second profile, also cross-evaluates: coverage on B's traffic of the
top-K set chosen from A (hot-set stability across domains).
"""
import json
import sys


def load(path):
    with open(path) as f:
        j = json.load(f)
    layers = [(il, c) for il, c in enumerate(j["counts"]) if sum(c) > 0]
    return j, layers


def coverage(counts, frac):
    total = sum(counts)
    if total == 0:
        return 0.0
    k = max(1, round(len(counts) * frac))
    return sum(sorted(counts, reverse=True)[:k]) / total


def cross_coverage(counts_a, counts_b, frac):
    total = sum(counts_b)
    if total == 0:
        return 0.0
    k = max(1, round(len(counts_a) * frac))
    hot_a = sorted(range(len(counts_a)), key=lambda e: -counts_a[e])[:k]
    return sum(counts_b[e] for e in hot_a) / total


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    j, layers = load(sys.argv[1])
    fracs = [0.125, 0.25, 0.255, 0.375, 0.50]

    print(f"model={j.get('model')}  n_expert={j['n_expert']}  n_expert_used={j['n_expert_used']}  "
          f"moe_layers={len(layers)}  tokens={j.get('tokens_profiled')}")
    print(f"{'layer':>5} " + " ".join(f"cov@{f:g}" for f in fracs))
    aggs = {f: [] for f in fracs}
    for il, c in layers:
        row = [coverage(c, f) for f in fracs]
        for f, v in zip(fracs, row):
            aggs[f].append(v)
        print(f"{il:>5} " + " ".join(f"{v:7.3f}" for v in row))
    print(f"{'MEAN':>5} " + " ".join(f"{sum(aggs[f])/len(aggs[f]):7.3f}" for f in fracs))
    print(f"{'FLAT':>5} " + " ".join(f"{f:7.3f}" for f in fracs))

    f255 = sum(aggs[0.255]) / len(aggs[0.255])
    lift = f255 - 0.255
    verdict = "GO" if lift >= 0.10 else ("MARGINAL" if lift >= 0.05 else "NO-GO")
    print(f"\ncoverage@25.5% = {f255:.3f} vs uniform 0.255  ->  lift {lift:+.3f}  ->  {verdict} for #75")

    if len(sys.argv) > 2:
        jb, layers_b = load(sys.argv[2])
        cb = dict(layers_b)
        xs = []
        for il, ca in layers:
            if il in cb:
                xs.append(cross_coverage(ca, cb[il], 0.255))
        if xs:
            x = sum(xs) / len(xs)
            print(f"cross-domain: top-25.5% set from A covers {x:.3f} of B's traffic "
                  f"({'stable' if x >= 0.9 * f255 else 'DOMAIN-SENSITIVE'})")


if __name__ == "__main__":
    main()
