"""Independent numerical cross-checks of V3 (cannot prove optimality, can only falsify it):
  (a) random directions u in all of R^n -> sorted placement -> cost; min must be >= C*;
  (b) alternating local search from random placements (fit -> re-sort until the ordering is stable);
      min must be >= C*; every stable point reached, mapped to the chamber by bit flips + qubit permutation,
      must be one of the enumerated regions AND be flagged as an order-isomorphic fixed point.
Usage: python crosscheck.py n k0 k1 n_dirs n_starts
"""
import json
import os
import sys
import time

import numpy as np

from common import spectrum, zmat, bits, cost_of_E, ranks_from_order
from vstar import analyse, load_orders

HERE = os.path.dirname(os.path.abspath(__file__))


def canonical_order(r, lam, Z, n):
    """Map a fixed-point placement to its chamber representative; return its ordering (bitstrings by rank)."""
    N = 2 ** n
    E = lam[r]
    v = E @ Z
    s = np.arange(N)
    perm_s = s.copy()
    # bit flips: new placement r'(s) = r(s xor mask) with mask flipping bits where v_j < 0
    mask = 0
    for j in range(n):
        if v[j] < 0:
            mask |= 1 << (n - 1 - j)
    r1 = r[s ^ mask]
    vabs = np.abs(v)
    q = np.argsort(-vabs, kind="stable")  # new qubit a = old qubit q[a]
    B = bits(n)
    # new bitstring for old bitstring t: bits B[t, q]
    newidx = B[:, q] @ (1 << (n - 1 - np.arange(n)))
    r2 = np.empty(N, dtype=np.int64)
    r2[newidx] = r1
    order = np.argsort(r2)
    return order


def main(n, ks, n_dirs, n_starts):
    rng = np.random.default_rng(12345 + n)
    Z = zmat(n)
    S = bits(n)
    N = 2 ** n
    orders = load_orders(n)
    vt = np.dtype((np.void, N))
    okeys = np.ascontiguousarray(orders.astype(np.uint8)).view(vt).ravel()
    idx_of = {bytes(k): i for i, k in enumerate(okeys)}
    vs = {row["k"]: row for row in json.load(open(os.path.join(HERE, f"vstar_n{n}.json")))}
    out = []
    for k in ks:
        t0 = time.time()
        lam = spectrum(n, k)
        Cr, F, fixed, margin = analyse(n, lam, orders)
        cstar = vs[k]["Cstar"]
        # (a) random directions
        U = rng.standard_normal((n_dirs, n))
        vals = U @ S.T
        Rk = np.argsort(np.argsort(vals, axis=1), axis=1)
        Ca = cost_of_E(lam[Rk], lam, Z)
        # (b) alternating local search
        best_b = np.inf
        reached = set()
        bad = 0
        for _ in range(n_starts):
            r = rng.permutation(N)
            for it in range(1000):
                v = lam[r] @ Z
                rn = np.argsort(np.argsort(S @ v, kind="stable"), kind="stable")
                if np.array_equal(rn, r):
                    break
                r = rn
            c = float(cost_of_E(lam[r], lam, Z))
            best_b = min(best_b, c)
            o = canonical_order(r, lam, Z, n)
            key = o.astype(np.uint8).tobytes()
            i = idx_of.get(key)
            if i is None or not fixed[i]:
                bad += 1
            else:
                reached.add(i)
        row = dict(n=n, k=k, Cstar=cstar, rand_dir_min=float(Ca.min()), rand_dir_hits=int(np.sum(Ca <= cstar + 1e-12)),
                   n_dirs=n_dirs, local_search_min=best_b, n_starts=n_starts,
                   distinct_fixed_points_reached=len(reached), reached_not_in_enumeration=bad,
                   ok=bool(Ca.min() >= cstar - 1e-12 and best_b >= cstar - 1e-12 and bad == 0), time=time.time() - t0)
        out.append(row)
        print(f"n={n} k={k}: C*={cstar:.12g}  rand-dir min={Ca.min():.12g} (hits {row['rand_dir_hits']}/{n_dirs})  "
              f"local-search min={best_b:.12g}  fixed pts reached={len(reached)}  not-in-enumeration={bad}  "
              f"{'OK' if row['ok'] else 'FAIL'} [{time.time() - t0:.1f}s]", flush=True)
    with open(os.path.join(HERE, f"crosscheck_n{n}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    n = int(sys.argv[1])
    main(n, range(int(sys.argv[2]), int(sys.argv[3])), int(sys.argv[4]), int(sys.argv[5]))
