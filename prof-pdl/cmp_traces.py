#!/usr/bin/env python3
"""Per-kernel median/step comparison between two profiler traces (A=control, B=PDL).
Usage: cmp_traces.py <A.json.gz> <B.json.gz> [steps=20]"""
import gzip, json, sys, collections, statistics

def load(p, steps):
    ev = json.load(gzip.open(p, "rt"))["traceEvents"]
    g = collections.defaultdict(list)
    for e in ev:
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memset", "gpu_memcpy"):
            g[e["name"][:60]].append(e["dur"])
    return {k: (len(v) / steps, statistics.median(v), sum(v) / steps) for k, v in g.items()}

steps = int(sys.argv[3]) if len(sys.argv) > 3 else 20
A, B = load(sys.argv[1], steps), load(sys.argv[2], steps)
keys = sorted(set(A) | set(B), key=lambda k: -(B.get(k, (0, 0, 0))[2]))
print(f"{'kernel':60s} {'n/step':>7} {'medA':>8} {'medB':>8} {'dmed':>8} "
      f"{'sumA/st':>9} {'sumB/st':>9} {'dsum':>9}")
tA = tB = 0.0
for k in keys:
    a = A.get(k, (0.0, 0.0, 0.0)); b = B.get(k, (0.0, 0.0, 0.0))
    tA += a[2]; tB += b[2]
    if max(a[2], b[2]) < 5:  # us/step
        continue
    print(f"{k:60s} {b[0]:7.1f} {a[1]:8.2f} {b[1]:8.2f} {b[1]-a[1]:+8.2f} "
          f"{a[2]:9.1f} {b[2]:9.1f} {b[2]-a[2]:+9.1f}")
print(f"{'TOTAL':60s} {'':7s} {'':8s} {'':8s} {'':8s} {tA:9.1f} {tB:9.1f} {tB-tA:+9.1f}")
