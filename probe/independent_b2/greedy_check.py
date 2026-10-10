"""Consistency check of the greedy output: for a hierarchical placement, the Walsh components of E of order >= 2
are partitioned by their lowest qubit, so  ||E - L||^2 = sum_q 2^(q-1) * residual_q  should hold exactly
(residual_q = (1/2) sum_b (g_b - mean g)^2 at step q, levels at step q being means of 2^(q-1) original levels).
Also re-derives each step's matching from the placement and re-scores it in float."""
import json
import os
import sys

import numpy as np

from common import spectrum, zmat

HERE = os.path.dirname(os.path.abspath(__file__))
for n in (5, 6):
    rows = []
    for fn in sorted(os.listdir(HERE)):
        if fn.startswith(f"greedy_n{n}_") and fn.endswith(".json"):
            rows += json.load(open(os.path.join(HERE, fn)))
    for row in sorted(rows, key=lambda r: r["k"]):
        lam = spectrum(n, row["k"])
        lam2 = float(lam @ lam)
        s = sum(2 ** (st["step"] - 1) * st["residual"] for st in row["steps"])
        direct = row["greedy_cost_lstsq"] * lam2
        print(f"n={n} k={row['k']}: sum_q 2^(q-1) res_q = {s:.12g}   ||E-L||^2 = {direct:.12g}   rel diff = {abs(s - direct) / direct:.2e}"
              f"   ties(>1 optimal on envelope) at steps: {[st['levels'] for st in row['steps'] if st['n_optimal_on_envelope'] > 1]}")
