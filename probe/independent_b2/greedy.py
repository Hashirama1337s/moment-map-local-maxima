"""V4: greedy peeling with an EXACT minimum-variance perfect matching at every step.

Exactness of one step (2m levels x_1 < ... < x_2m, all arithmetic in exact rationals = the float64 inputs):
  For a perfect matching M with gaps g_b (upper - lower), S_M = sum g, Q_M = sum g^2 and
      2 * residual = sum (g - mean g)^2 = Q_M - S_M^2 / m = min_t sum_b (g_b - t)^2 .
  So  min_M residual = min_t h(t)/2,  h(t) = min_M sum (g_b - t)^2 = m t^2 + phi(t),  phi(t) = min_M (Q_M - 2 S_M t),
  phi is the lower envelope of finitely many lines.  An optimal M* is optimal for the fixed-t problem at
  t* = S_{M*}/m in [0, x_2m - x_1], and any M' on the envelope at t* has Q' - S'^2/m <= h(t*) = optimum.
  The envelope on [0, span] is computed exactly by Eisner-Severance bisection: query the fixed-t optimum at
  the intersection of the two bracketing lines; recurse iff the returned line is strictly below there.
  The fixed-t problem is a min-weight perfect matching on the complete graph, solved with networkx's
  Edmonds blossom algorithm (max_weight_matching, maxcardinality=True) on INTEGER weights obtained by
  scaling the exact rational weights (g - t)^2 by their common denominator (networkx uses only integer
  arithmetic when all weights are integers).  All comparisons are exact Fractions.

Usage: python greedy.py n k0 k1
"""
import json
import math
import os
import sys
import time
from fractions import Fraction

import networkx as nx
import numpy as np

from common import spectrum, zmat, cost, cost_lstsq

HERE = os.path.dirname(os.path.abspath(__file__))


def fixed_t_matching(x, t):
    """Exact min sum (x_j - x_i - t)^2 perfect matching. x: list of Fractions (sorted), t: Fraction."""
    L = len(x)
    w = {}
    for i in range(L):
        for j in range(i + 1, L):
            d = x[j] - x[i] - t
            w[(i, j)] = d * d
    den = 1
    for v in w.values():
        den = math.lcm(den, v.denominator)
    iw = {e: v.numerator * (den // v.denominator) for e, v in w.items()}
    big = max(iw.values()) + 1
    G = nx.Graph()
    for (i, j), v in iw.items():
        G.add_edge(i, j, weight=big - v)  # Python ints -> integer-only blossom
    M = nx.max_weight_matching(G, maxcardinality=True)
    pairs = sorted(tuple(sorted(e)) for e in M)
    assert len(pairs) == L // 2
    return tuple(pairs)


def line(x, M):
    g = [x[j] - x[i] for i, j in M]
    return sum(g, Fraction(0)), sum((a * a for a in g), Fraction(0))  # S, Q


def min_variance_matching(x):
    m = len(x) // 2
    tlo, thi = Fraction(0), x[-1] - x[0]
    found = {}
    calls = [0]

    def oracle(t):
        calls[0] += 1
        M = fixed_t_matching(x, t)
        found[M] = line(x, M)
        return M

    def ell(M, t):
        S, Q = found[M]
        return Q - 2 * S * t

    def rec(tL, ML, tR, MR):
        SL, QL = found[ML]
        SR, QR = found[MR]
        if (SL, QL) == (SR, QR):
            return
        assert SL != SR, "parallel distinct envelope lines"
        tx = (QL - QR) / (2 * (SL - SR))
        assert tL <= tx <= tR
        Mx = oracle(tx)
        if ell(Mx, tx) < ell(ML, tx):
            rec(tL, ML, tx, Mx)
            rec(tx, Mx, tR, MR)

    ML = oracle(tlo)
    MR = oracle(thi)
    rec(tlo, ML, thi, MR)
    vals = {M: (Q - S * S / m) / 2 for M, (S, Q) in found.items()}
    best = min(vals.values())
    winners = [M for M, v in vals.items() if v == best]
    return winners[0], best, len(found), calls[0], len(winners)


def brute_min_variance(x):
    """Brute force over all perfect matchings (for validation, len(x) <= 16)."""
    m = len(x) // 2
    best = [None, None]

    def rec(rem, S, Q, M):
        if not rem:
            v = (Q - S * S / m) / 2
            if best[0] is None or v < best[0]:
                best[0], best[1] = v, tuple(sorted(M))
            return
        i = rem[0]
        for k in range(1, len(rem)):
            j = rem[k]
            g = x[j] - x[i]
            rec(rem[1:k] + rem[k + 1:], S + g, Q + g * g, M + [(i, j)])

    xf = x
    x = [float(v) for v in xf]
    rec(list(range(len(x))), 0.0, 0.0, [])
    return best[1], best[0]


def greedy_placement(lam, verify_brute_upto=0):
    n = int(round(math.log2(len(lam))))
    groups = [[i] for i in range(len(lam))]
    x = [Fraction(float(v)) for v in lam]  # exact copies of the float64 levels (sorted ascending)
    bitsof = np.zeros((len(lam), n), dtype=np.int64)
    steps = []
    for q in range(n):
        t0 = time.time()
        M, res, nlines, ncalls, nwin = min_variance_matching(x)
        st = dict(step=q + 1, levels=len(x), residual=float(res), envelope_lines=nlines, oracle_calls=ncalls,
                  n_optimal_on_envelope=nwin, pairs=[list(p) for p in M])
        if verify_brute_upto and len(x) <= verify_brute_upto:
            Mb, rb = brute_min_variance(x)
            st["brute_residual"] = float(rb)
            st["brute_equal"] = bool(abs(rb - float(res)) <= 1e-12 * max(1.0, abs(rb)))
            st["brute_same_matching"] = bool(Mb == M)
        newx, newg = [], []
        for i, j in M:  # x[i] < x[j]: j is the upper member -> bit 1
            for lvl in groups[j]:
                bitsof[lvl, q] = 1
            newx.append((x[i] + x[j]) / 2)
            newg.append(groups[i] + groups[j])
        order = sorted(range(len(newx)), key=lambda a: newx[a])
        x = [newx[a] for a in order]
        groups = [newg[a] for a in order]
        st["time"] = time.time() - t0
        steps.append(st)
    # placement: bitstring s (qubit 1 = MSB = step 1) -> level rank
    svals = bitsof @ (1 << (n - 1 - np.arange(n)))
    assert len(set(svals.tolist())) == len(lam)
    r = np.empty(len(lam), dtype=np.int64)
    r[svals] = np.arange(len(lam))
    return r, steps


if __name__ == "__main__":
    n = int(sys.argv[1])
    k0, k1 = int(sys.argv[2]), int(sys.argv[3])
    vs = {row["k"]: row for row in json.load(open(os.path.join(HERE, f"vstar_n{n}.json")))}
    Z = zmat(n)
    out = []
    for k in range(k0, k1):
        t0 = time.time()
        lam = spectrum(n, k)
        r, steps = greedy_placement(lam, verify_brute_upto=16)
        cg = cost(r, lam, Z)
        cg2 = cost_lstsq(r, lam, Z)
        cs = vs[k]["Cstar_lstsq"]
        row = dict(n=n, k=k, greedy_cost=cg, greedy_cost_lstsq=cg2, Cstar=cs, ratio=cg2 / cs, ranks=r.tolist(),
                   steps=steps, time=time.time() - t0)
        out.append(row)
        print(f"n={n} k={k}: greedy cost = {cg2:.12g}  C* = {cs:.12g}  ratio = {cg2 / cs:.10g}  "
              f"steps: " + "; ".join(f"{s['levels']}lv res={s['residual']:.6g} lines={s['envelope_lines']}"
                                     + (f" brute={'=' if s.get('brute_equal') else 'X'}" if 'brute_equal' in s else "")
                                     for s in steps) + f"  [{time.time() - t0:.1f}s]", flush=True)
    with open(os.path.join(HERE, f"greedy_n{n}_k{k0}-{k1 - 1}.json"), "w") as f:
        json.dump(out, f, indent=1)
