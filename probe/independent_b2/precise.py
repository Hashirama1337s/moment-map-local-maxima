"""High-precision recheck of the reported C* (and greedy cost) values.
Eigenvalues of the float64 matrix H = impl_b.build_H_plant(n, 1000+k) are refined to ~30 digits by mpmath Rayleigh
quotients of the LAPACK eigenvectors (error <= ||r||^2 / gap, printed), then the cost of the reported placement is
recomputed in mpmath (dps 40) by an explicit residual vector E - c0 - sum_j c_j z_j.
Usage: python precise.py
"""
import json
import os

import mpmath as mp
import numpy as np

from common import B, zmat

HERE = os.path.dirname(os.path.abspath(__file__))
mp.mp.dps = 40


def hp_eigs(n, k):
    H = B.build_H_plant(n, 1000 + k)
    w, X = np.linalg.eigh(H)
    N = H.shape[0]
    Hm = mp.matrix(N, N)
    for i in range(N):
        for j in range(N):
            Hm[i, j] = mp.mpc(float(H[i, j].real), float(H[i, j].imag))
    lam = []
    rmax = mp.mpf(0)
    for a in range(N):
        x = mp.matrix([mp.mpc(float(X[i, a].real), float(X[i, a].imag)) for i in range(N)])
        Hx = Hm * x
        num = sum((mp.conj(x[i]) * Hx[i] for i in range(N)), mp.mpc(0))
        den = sum((mp.conj(x[i]) * x[i] for i in range(N)), mp.mpc(0))
        l = mp.re(num / den)
        r = mp.sqrt(sum((abs(Hx[i] - l * x[i]) ** 2 for i in range(N)), mp.mpf(0)) / mp.re(den))
        rmax = max(rmax, r)
        lam.append(l)
    lam.sort()
    gaps = [lam[i + 1] - lam[i] for i in range(N - 1)]
    return lam, rmax, min(gaps), w


def hp_cost(lam, ranks, n):
    N = 2 ** n
    Z = zmat(n)
    E = [lam[r] for r in ranks]
    c0 = sum(E, mp.mpf(0)) / N
    c = [sum((E[s] * int(Z[s, j]) for s in range(N)), mp.mpf(0)) / N for j in range(n)]
    res = mp.mpf(0)
    for s in range(N):
        L = c0 + sum((c[j] * int(Z[s, j]) for j in range(n)), mp.mpf(0))
        res += (E[s] - L) ** 2
    return res / sum((l * l for l in lam), mp.mpf(0))


if __name__ == "__main__":
    out = []
    for n in (5, 6):
        vs = json.load(open(os.path.join(HERE, f"vstar_n{n}.json")))
        gr = {}
        for fn in os.listdir(HERE):
            if fn.startswith(f"greedy_n{n}_") and fn.endswith(".json"):
                for row in json.load(open(os.path.join(HERE, fn))):
                    gr[row["k"]] = row
        for row in vs:
            k = row["k"]
            lam, rmax, mingap, w = hp_eigs(n, k)
            maxdiff = max(abs(float(lam[i]) - w[i]) for i in range(len(w)))
            cs = hp_cost(lam, row["ranks"], n)
            rec = dict(n=n, k=k, Cstar_hp=mp.nstr(cs, 20), Cstar_float=row["Cstar_lstsq"],
                       rel_diff=float(abs(cs - row["Cstar_lstsq"]) / cs), rq_resid_max=float(rmax),
                       min_level_gap=float(mingap), max_eig_shift=maxdiff, trace=float(sum(lam)))
            if k in gr:
                cg = hp_cost(lam, gr[k]["ranks"], n)
                rec.update(greedy_hp=mp.nstr(cg, 20), ratio_hp=mp.nstr(cg / cs, 15))
            out.append(rec)
            print(rec, flush=True)
    with open(os.path.join(HERE, "precise.json"), "w") as f:
        json.dump(out, f, indent=1)
