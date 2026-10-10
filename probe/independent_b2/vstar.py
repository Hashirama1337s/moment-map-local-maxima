"""V3: exact C* from the complete list of chamber regions (regions.py).

Why exact:
  cost = (sum lam^2 - (sum lam)^2/N - F/N) / sum lam^2 with F(placement) = |v|^2, v_j = z_j . E.
  F = max_{|u|=1} (u . v)^2, so  max_placement F = max_u max_placement (sum_s E_s u.z(s))^2
  and for fixed u the inner max is attained (rearrangement inequality) by the placement that sorts the
  levels in the same order as the values u.z(s).  Hence the optimum is attained by a placement whose
  ordering is induced by a generic linear functional, i.e. by one of the regions of the arrangement
  {u : u.(z(s)-z(t)) = 0}.  F is invariant under bit flips and qubit permutations (the hyperoctahedral group
  B_n), which acts simply transitively on Weyl chambers, so it suffices to take the max over the regions in
  the chamber u_1 >= ... >= u_n >= 0 -- all of which regions.py enumerates with a completeness certificate.
  (Proof that every region's own placement is >= G(u)^2 for every u in it: F(pi_R) = max_w (w . v_R)^2 >= (u.v_R)^2
   = G(u)^2 for u in R.)

Usage: python vstar.py n k0 k1   (chains k0..k1-1)
"""
import json
import os
import sys
import time

import numpy as np

from common import spectrum, zmat, bits, cost, cost_lstsq, ranks_from_order, is_order_isomorphic

HERE = os.path.dirname(os.path.abspath(__file__))


def load_orders(n):
    z = np.load(os.path.join(HERE, f"regions_n{n}.npz"))
    return z["orders"].astype(np.int64)


def analyse(n, lam, orders, chunk=20000):
    N = 2 ** n
    Z = zmat(n)
    S = bits(n)
    R = len(orders)
    F = np.empty(R)
    fixed = np.zeros(R, bool)
    fixmargin = np.empty(R)
    for a in range(0, R, chunk):
        O = orders[a:a + chunk]
        V = np.einsum("i,rij->rj", lam, Z[O].astype(float))  # v_R = sum_i lam_i z(order_i)
        F[a:a + chunk] = np.sum(V * V, 1)
        vals = V @ S.T  # (r, N): v . s
        # value at the region's own ordering positions
        vo = np.take_along_axis(vals, O, axis=1)
        d = np.diff(vo, axis=1)
        fixed[a:a + chunk] = (d > 0).all(1)
        fixmargin[a:a + chunk] = d.min(1) / np.abs(vals).max(1)
    lam2 = float(np.dot(lam, lam))
    tot = float(np.sum(lam))
    C = (lam2 - tot * tot / N - F / N) / lam2
    return C, F, fixed, fixmargin


def run(n, ks):
    orders = load_orders(n)
    Z = zmat(n)
    import math
    group = (2 ** n) * math.factorial(n)
    rows = []
    for k in ks:
        t0 = time.time()
        lam = spectrum(n, k)
        C, F, fixed, margin = analyse(n, lam, orders)
        ib = int(np.argmin(C))
        srt = np.sort(C)
        r = ranks_from_order(orders[ib])
        c1 = cost(r, lam, Z)
        c2 = cost_lstsq(r, lam, Z)
        iso, gap = is_order_isomorphic(r, lam, Z)
        # near-tie margins of fixed points
        fm = margin[fixed]
        rows.append(dict(n=n, k=k, seed_H=1000 + k, n_regions_chamber=int(len(orders)), Cstar=float(C[ib]),
                         Cstar_recheck=c1, Cstar_lstsq=c2, second_best=float(srt[1]),
                         gap_to_second=float(srt[1] - srt[0]),
                         n_fixed_chamber=int(fixed.sum()), n_fixed_total=int(fixed.sum()) * group,
                         best_is_fixed=bool(fixed[ib]), best_is_order_isomorphic=iso,
                         min_fixed_margin=float(fm.min()) if len(fm) else None,
                         min_nonfixed_violation=float(margin[~fixed].max()) if (~fixed).any() else None,
                         ranks=r.tolist(), fit_weights=(lam[r] @ Z / 2 ** n).tolist(),
                         cost_max_region=float(C.max()), time=time.time() - t0,
                         trace=float(np.sum(lam)), lam2=float(np.dot(lam, lam))))
        print(f"n={n} k={k}: C* = {C[ib]:.12g}  (recheck {c1:.12g}, lstsq {c2:.12g})  2nd = {srt[1]:.12g}  "
              f"fixed pts (chamber) = {int(fixed.sum())}  best fixed={bool(fixed[ib])} iso={iso}  "
              f"[{time.time() - t0:.1f}s]", flush=True)
    return rows


if __name__ == "__main__":
    n = int(sys.argv[1])
    k0, k1 = int(sys.argv[2]), int(sys.argv[3])
    rows = run(n, range(k0, k1))
    with open(os.path.join(HERE, f"vstar_n{n}.json"), "w") as f:
        json.dump(rows, f, indent=1)
