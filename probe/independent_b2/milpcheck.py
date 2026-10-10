"""Independent check of each greedy step with a different exact solver (HiGHS MILP via scipy.optimize.milp):
at t* = mean gap of the chosen matching M*, solve  min sum_e (g_e - t*)^2 x_e  over perfect matchings
(binary x, degree = 1 at every level).  Necessary condition for M* to be the minimum-variance matching:
the MILP optimum equals sum_{M*}(g - t*)^2.  Also checks the min-variance value  against the MILP matching's
own variance.  Replays the greedy level lists from the stored pairs."""
import json
import os

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

from common import spectrum

HERE = os.path.dirname(os.path.abspath(__file__))


def milp_match(x, t):
    L = len(x)
    edges = [(i, j) for i in range(L) for j in range(i + 1, L)]
    c = np.array([(x[j] - x[i] - t) ** 2 for i, j in edges])
    A = np.zeros((L, len(edges)))
    for e, (i, j) in enumerate(edges):
        A[i, e] = 1
        A[j, e] = 1
    res = milp(c, constraints=LinearConstraint(A, 1, 1), integrality=np.ones(len(edges)), bounds=Bounds(0, 1),
               options=dict(mip_rel_gap=0, time_limit=600))
    sel = [edges[e] for e in np.flatnonzero(res.x > 0.5)]
    return res.fun, sel, res.status


for n in (5, 6):
    rows = []
    for fn in sorted(os.listdir(HERE)):
        if fn.startswith(f"greedy_n{n}_") and fn.endswith(".json"):
            rows += json.load(open(os.path.join(HERE, fn)))
    for row in sorted(rows, key=lambda r: r["k"]):
        x = list(spectrum(n, row["k"]))
        msgs = []
        for st in row["steps"]:
            M = [tuple(p) for p in st["pairs"]]
            if len(x) >= 8:
                g = np.array([x[j] - x[i] for i, j in M])
                t = g.mean()
                ours = float(np.sum((g - t) ** 2))
                val, sel, status = milp_match(x, t)
                gm = np.array([x[j] - x[i] for i, j in sel])
                var_milp = 0.5 * float(np.sum((gm - gm.mean()) ** 2))
                ok = (val >= ours - 1e-9 * max(1, ours)) and (var_milp >= st["residual"] - 1e-9)
                msgs.append(f"{len(x)}lv: ours={ours:.10g} milp={val:.10g} {'OK' if ok else 'FAIL'}")
            newx = [(x[i] + x[j]) / 2 for i, j in M]
            x = sorted(newx)
        print(f"n={n} k={row['k']}: " + "; ".join(msgs), flush=True)
