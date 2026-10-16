"""BAR-D5LP verdict: reads results/d5lp/{controls,joint,greedy,upper}.json, re-verifies the 4x4 / qubits5 certificates and the
greedy spectrum in exact integer arithmetic (independent of the run), prints HOLDS / FAILED per sealed item.
Run: py -3.11 theory/d5lp_verdict.py"""
import os, sys, json, math
from fractions import Fraction
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d5lp as L
D = L.OUT; J = lambda f: json.load(open(os.path.join(D, f)))
ctl, jt = J("controls.json"), J("joint.json")
print("CONTROLS", "PASS" if ctl["ALL_PASS"] else "FAIL (run VOID)")


def recheck(label, ranks, F):
    R, d, nr = L.build(ranks, F); A = np.vstack([L.gap_rows(d), R]); r = jt[label]
    ok = L.verify_certificate(A, r["cert_rows"], r["cert_mult"])
    return ok, r["verdict"], r["support"], r["n_cert_realms"]


rk, _ = L.bip_realms(4, 4); ok1, v1, s1, n1 = recheck("4x4", rk, L.bip_fitmat(4, 4))
print(f"P1 4x4 joint LP ({len(rk)} realms): {v1}, certificate re-verified exactly: {ok1} (support {s1} rows, {n1} realms) ->",
      "HOLDS" if (v1 == "INFEASIBLE" and ok1) else "FAILED")
ok2, v2, s2, n2 = recheck("qubits5", L.qubit_realms(5), L.qubit_fitmat(5))
print(f"P2 5-qubit joint LP (516 orders): {v2}, certificate re-verified exactly: {ok2} (support {s2} rows, {n2} orders) ->",
      "HOLDS" if (v2 == "INFEASIBLE" and ok2) else "FAILED")
g = J("greedy.json"); R, d, nr = L.build(rk, L.bip_fitmat(4, 4))
lam = [Fraction(x) for x in g["spectrum"]]; den = 1
for f in lam: den = den * f.denominator // math.gcd(den, f.denominator)
z = np.array([int(f * den) for f in lam], dtype=object); vals = (R.astype(object) @ z).reshape(nr, d - 1)
cnt = sum(1 for row in vals if min(row) > 0); gaps = all(z[k + 1] > z[k] for k in range(d - 1))
print(f"P3 LP-greedy from the 1114 hill-climb realms: exact recount {cnt} realms (run reported {g['exact_count_all']}), strictly "
      f"sorted {gaps}, greedy completed {g['completed']} ->", "HOLDS" if (cnt > 1114 and gaps) else "FAILED")
if os.path.exists(os.path.join(D, "upper.json")):
    u = J("upper.json")
    print(f"DESCRIPTIVE upper bound: {u['n_cores']} pairwise disjoint infeasible realm sets (sizes {min(u['sizes'])}-{max(u['sizes'])}, "
          f"all exact {u['all_exact']}, completed {u['completed']}) -> max <= {u['upper_bound']}")
print("DESCRIPTIVE joint LP:", {k: v["verdict"] for k, v in jt.items()})
