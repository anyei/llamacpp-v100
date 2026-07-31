#!/usr/bin/env python3
# TASKS #84: offline expert-union curve from a LLAMA_EXPERT_PROFILE_IDS dump.
# Record format (int32): il, k, n_tok, ids[k*n_tok] lane-major (position-major
# order: ids[j*k : (j+1)*k] = position j's top-k global expert ids).
# For each width w, slide a w-position window over each record's positions and
# count the distinct-expert union - the member-read multiplier a w-lane verify
# batch (adjacent positions) would pay. mult(w) = mean_union(w) / mean_union(1).
import sys
import struct
from collections import defaultdict

path = sys.argv[1]
W_MAX = int(sys.argv[2]) if len(sys.argv) > 2 else 8

# per (w): [sum_union, n_windows]; per (layer, w) for the depth breakdown
tot = defaultdict(lambda: [0, 0])
per_layer = defaultdict(lambda: defaultdict(lambda: [0, 0]))
n_rec = 0
positions = 0

with open(path, "rb") as f:
    while True:
        hdr = f.read(12)
        if len(hdr) < 12:
            break
        il, k, n_tok = struct.unpack("<3i", hdr)
        data = f.read(4 * k * n_tok)
        if len(data) < 4 * k * n_tok:
            sys.stderr.write(f"truncated record at #{n_rec} (il {il})\n")
            break
        ids = struct.unpack(f"<{k*n_tok}i", data)
        n_rec += 1
        positions += n_tok
        rows = [ids[j*k:(j+1)*k] for j in range(n_tok)]
        for w in range(1, W_MAX + 1):
            if n_tok < w:
                continue
            for j in range(n_tok - w + 1):
                u = len(set().union(*[set(r) for r in rows[j:j+w]]))
                tot[w][0] += u
                tot[w][1] += 1
                pl = per_layer[il][w]
                pl[0] += u
                pl[1] += 1

if not tot:
    sys.exit("no records")

print(f"records {n_rec}, positions {positions}, layers {len(per_layer)}")
u1 = tot[1][0] / tot[1][1]
print(f"{'w':>3} {'windows':>10} {'uniq/window':>12} {'mult':>6} {'linear':>7}")
for w in sorted(tot):
    s, n = tot[w]
    uw = s / n
    print(f"{w:>3} {n:>10} {uw:>12.2f} {uw/u1:>6.2f} {w:>7}")

# depth breakdown at the widest measured w: min/max layers
w = max(tot)
layer_mult = {il: d[w][0]/d[w][1]/u1 for il, d in per_layer.items() if d[w][1] > 0}
if layer_mult:
    lo = min(layer_mult, key=layer_mult.get)
    hi = max(layer_mult, key=layer_mult.get)
    print(f"w{w} per-layer mult: min layer {lo} {layer_mult[lo]:.2f}, "
          f"max layer {hi} {layer_mult[hi]:.2f}")
