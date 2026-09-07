#!/usr/bin/env python3
# #132 gate 0, v3: segment dumps into requests (pos0 resets), align request-by-request.
import sys
from difflib import SequenceMatcher

def load_requests(path):
    reqs = [[]]
    last = -1
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            p, d, a = line.split(';')
            rec = (int(p), [int(x) for x in d.split(',') if x], [int(x) for x in a.split(',') if x])
        except ValueError:
            continue
        if rec[0] < last - 50:  # position reset = new request (restores only step back a few)
            reqs.append([])
        last = rec[0]
        reqs[-1].append(rec)
    return [r for r in reqs if r]

def build_stream(recs):
    st = {}
    for pos0, _d, acc in recs:
        for i, t in enumerate(acc):
            st[pos0 + 1 + i] = t
    lo, hi = min(st), max(st)
    return [st.get(p, -1) for p in range(lo, hi + 1)], lo

RA = load_requests(sys.argv[1])
RB = load_requests(sys.argv[2])
print(f"requests: A {len(RA)}, B {len(RB)}")

tot = dict(rej=0, joined=0, rescue=0, agree_rej=0, pos=0, agree=0, aligned=0, alen=0)
for ra, rb in zip(RA, RB):
    sa, base_a = build_stream(ra)
    sb, base_b = build_stream(rb)
    sm = SequenceMatcher(None, sa, sb, autojunk=False)
    a2b = {}
    m = 0
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            a2b[base_a + blk.a + k] = base_b + blk.b + k
        m += blk.size
    tot['aligned'] += m
    tot['alen'] += len(sa)
    bpred = {}
    for pos0, draft, _acc in rb:
        for i, t in enumerate(draft):
            bpred[pos0 + 1 + i] = t
    for pos0, draft, acc in ra:
        for i, t in enumerate(draft):
            pb = a2b.get(pos0 + 1 + i)
            if pb is not None and pb in bpred:
                tot['pos'] += 1
                tot['agree'] += (bpred[pb] == t)
        n_acc = len(acc) - 1
        if n_acc >= len(draft):
            continue
        tot['rej'] += 1
        pb = a2b.get(pos0 + 1 + n_acc)
        if pb is None or pb not in bpred:
            continue
        tot['joined'] += 1
        if bpred[pb] == acc[-1]:
            tot['rescue'] += 1
        if bpred[pb] == draft[n_acc]:
            tot['agree_rej'] += 1

print(f"alignment coverage: {tot['aligned']}/{tot['alen']} = {tot['aligned']/max(1,tot['alen']):.2f}")
print(f"A rejections {tot['rej']}, joined {tot['joined']}")
if tot['joined']:
    print(f"B rescues A's rejection: {tot['rescue']}/{tot['joined']} = {tot['rescue']/tot['joined']:.3f}")
    print(f"B agrees with A's wrong token: {tot['agree_rej']}/{tot['joined']} = {tot['agree_rej']/tot['joined']:.3f}")
if tot['pos']:
    print(f"A/B agreement on aligned drafted positions: {tot['agree']}/{tot['pos']} = {tot['agree']/tot['pos']:.3f}")
