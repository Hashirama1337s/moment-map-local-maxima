"""Validation of impl_b_E.bb_secant against exhaustive enumeration, started from a deliberately BAD upper bound
(the worst structured partition) so the B&B itself must find the optimum (the earlier test started from the
descent optimum and could not detect an invalid bound)."""
import os
import sys
import itertools
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import impl_b as B  # noqa: E402
import impl_b_E as E  # noqa: E402


def brute(lam, m, G):
    perms = np.array(list(itertools.permutations(range(len(lam)))))
    X = lam[perms].reshape(-1, m, G)
    R = X - X.mean(2, keepdims=True) - X.mean(1, keepdims=True) + X.mean((1, 2), keepdims=True)
    return float((R ** 2).sum((1, 2)).min())


def bad_start(P):
    """Partition into consecutive blocks (adjacent values per group) -- typically far from optimal."""
    gs = np.array([np.where((P.groups == np.arange(P.m) + P.m * b).all(1))[0][0] for b in range(P.G)])
    return P.rss(gs), gs


cases = [(s, (3, 3)) for s in range(1, 6)] + [(s, (2, 4)) for s in range(6, 9)] + [(s, (2, 3)) for s in range(9, 11)] \
    + [(s, (4, 2)) for s in range(11, 13)]
worst = 0.0
for seed, (m, G) in cases:
    rng = np.random.default_rng(seed)
    lam = np.sort(rng.standard_normal(m * G) * 3)
    P = E.Problem(lam, m)
    ub0, g0 = bad_start(P)
    t = time.time()
    ub, best, info = E.bb_secant(P, ub0, g0)
    bf = brute(lam, m, G)
    rel = (ub - bf) / bf
    worst = max(worst, abs(rel))
    print(f"{m}x{G} seed={seed}: start={ub0:.6g} secant={ub:.15g} brute={bf:.15g} rel={rel:+.1e} "
          f"{info['status']} boxes={info['boxes']} t={time.time() - t:.1f}s", flush=True)
print(f"max |rel| = {worst:.2e}")
# V1 chain 4 (64 values, 1-D s) from the bad start, vs the certified dist-B&B value
lam = np.linalg.eigvalsh(B.build_H_plant(6, 1004))
P = E.Problem(lam, 2)
ub0, g0 = bad_start(P)
t = time.time()
ub, best, info = E.bb_secant(P, ub0, g0)
print(f"V1 ch4 secant from bad start: {ub / np.sum(lam ** 2):.12g} (dist-B&B: 3.83867774643e-05) "
      f"{info['status']} boxes={info['boxes']} t={time.time() - t:.0f}s", flush=True)
