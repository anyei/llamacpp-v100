#!/usr/bin/env python3
"""Derive bucket B = final - snapshotA from two cumulative LLAMA_EXPERT_PROFILE dumps.
Usage: profile-diff.py final.json snapshotA.json out-B.json
The profiler accumulates; a snapshot taken between traffic buckets lets the
second bucket be recovered by subtraction (boundary smears by <500 tokens,
the dump cadence)."""
import json
import sys

final, snap, out = sys.argv[1:4]
F = json.load(open(final))
A = json.load(open(snap))
assert F["n_layer"] == A["n_layer"] and F["n_expert"] == A["n_expert"]
B = dict(F)
B["tokens_profiled"] = F["tokens_profiled"] - A["tokens_profiled"]
B["rows_per_layer"] = [f - a for f, a in zip(F["rows_per_layer"], A["rows_per_layer"])]
B["counts"] = [[f - a for f, a in zip(fl, al)] for fl, al in zip(F["counts"], A["counts"])]
assert min(min(r) for r in B["counts"]) >= 0
json.dump(B, open(out, "w"))
print(f"A tokens {A['tokens_profiled']}  final {F['tokens_profiled']}  B {B['tokens_profiled']}")
