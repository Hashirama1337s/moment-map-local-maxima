"""Independent re-enumeration of the chamber orderings by LP (n <= 5), compared with regions.py.

DFS over prefixes of the ordering of the 2^n subsets.  A next element t is admissible only if no remaining
element t' is forced below it by the chamber (all prefix sums of t - t' >= 0), and the prefix + t + "t below
every remaining element" is strictly realisable: LP feasibility of
   u . (s_{j+1} - s_j) >= 1 (prefix),  u . (t' - t) >= 1 (all remaining t'),  u_i - u_{i+1} >= 1,  u_n >= 1.
"""
import os
import sys
import time

import numpy as np
from scipy.optimize import linprog

from regions import enumerate_regions

HERE = os.path.dirname(os.path.abspath(__file__))


def lp_enumerate(n):
    N = 2 ** n
    S = ((np.arange(N)[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1).astype(np.int64)
    pref = np.cumsum(S, axis=1)  # prefix sums of bits for dominance
    walls = []
    for i in range(n - 1):
        d = np.zeros(n)
        d[i], d[i + 1] = 1, -1
        walls.append(d)
    d = np.zeros(n)
    d[n - 1] = 1
    walls.append(d)
    walls = np.array(walls)
    nlp = [0]

    def feasible(rows):
        A = np.vstack([walls, rows])
        nlp[0] += 1
        res = linprog(np.zeros(n), A_ub=-A, b_ub=-np.ones(len(A)), bounds=[(None, None)] * n, method="highs")
        return res.status == 0

    out = []

    def rec(prefix, remaining, prows):
        if not remaining:
            out.append(tuple(prefix))
            return
        rem = np.array(sorted(remaining))
        for t in rem:
            others = rem[rem != t]
            if len(others):
                dom = (pref[t][None, :] - pref[others] >= 0).all(1)  # t' forced below t
                if dom.any():
                    continue
            rows = list(prows)
            if prefix:
                rows.append(S[t] - S[prefix[-1]])
            rows_t = rows + [S[o] - S[t] for o in others]
            if len(rows_t) == 0 or feasible(np.array(rows_t, dtype=float)):
                rec(prefix + [t], remaining - {t}, rows)

    rec([], set(range(N)), [])
    return out, nlp[0]


if __name__ == "__main__":
    for n in [int(x) for x in sys.argv[1:]]:
        t0 = time.time()
        lp_orders, nlp = lp_enumerate(n)
        dd = enumerate_regions(n, verbose=False, certify=False)["orders"]
        A = set(lp_orders)
        Bs = set(tuple(int(x) for x in o) for o in dd)
        print(f"n={n}: LP-DFS orderings = {len(A)} ({nlp} LPs, {time.time() - t0:.1f}s); DD orderings = {len(Bs)}; "
              f"identical sets: {A == Bs}")
