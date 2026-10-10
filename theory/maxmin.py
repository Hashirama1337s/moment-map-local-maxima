"""How many simultaneous local minima can one spectrum have? Hill-climb on the spectrum (lower bound on the max) vs the
realizable-SYT bound (OEIS A211400 / 2). Run from probe/: py ../theory/maxmin.py <m> <n> <iters>"""
import sys, os, numpy as np, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import t1
m, n, iters = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); Y = t1.syt_all(m, n); half = 2 if m == n else 1
def count(lam):
    E = lam[Y]; L, _ = t1.fit(E); return int((t1.xmin(E, L) > 0).sum()) // half
rng = np.random.default_rng(11); best = (-1, None)
for start in range(8):
    lam = np.sort(rng.standard_normal(m * n)); c = count(lam)
    for it in range(iters):
        prop = np.sort(lam + rng.standard_normal(m * n) * 0.05 * rng.uniform(0.1, 1)); cp = count(prop)
        if cp >= c: lam, c = prop, cp
    if c > best[0]: best = (c, lam.tolist())
    print("start", start, "count", c, flush=True)
json.dump(dict(shape=f"{m}x{n}", best_count=best[0], spectrum=best[1]), open(os.path.join(t1.OUT, f"maxmin_{m}x{n}.json"), "w"))
print("BEST", m, n, best[0])
