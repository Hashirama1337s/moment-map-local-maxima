"""Door 5 (Addendum C), lower-bound search: big-M MILP, 180 s time limit (descriptive; not part of any proof).
max sum_T z_T  s.t.  A_T lam - M z_T >= eps - M  (z_T = 1 forces every row of realm T >= eps),  gaps >= eps,  0 = lam_0 <= lam <= SC.
The incumbent spectrum is re-counted exactly (all realms it satisfies, Fraction arithmetic) and written to results/d5lp/milp_probe.json.
MILP solvers are not deterministic across machines: the shipped certificate (certificates/lower_1209.json) is the proof object.
Run (from probe/): python ../theory/d5lp_milp_probe.py"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d5lp as L
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix, hstack, vstack

eps = 1.0; SC = 1e4
ranks, _ = L.bip_realms(4, 4); R, d, nr = L.build(ranks, L.bip_fitmat(4, 4)); G = L.gap_rows(d)
M = float(np.abs(R).sum(1).max()) * SC + eps; k = R.shape[0]; owner = np.repeat(np.arange(nr), d - 1)
Zc = csr_matrix((-M * np.ones(k), (np.arange(k), owner)), shape=(k, nr))
A = vstack([hstack([csr_matrix(R.astype(float)), Zc]), hstack([csr_matrix(G.astype(float)), csr_matrix((d - 1, nr))])]).tocsr()
lb = np.r_[(eps - M) * np.ones(k), eps * np.ones(d - 1)]
c = np.r_[np.zeros(d), -np.ones(nr)]; integ = np.r_[np.zeros(d), np.ones(nr)]
lo = np.r_[0, np.zeros(d - 1), np.zeros(nr)]; hi = np.r_[0, SC * np.ones(d - 1), np.ones(nr)]
t0 = time.time()
r = milp(c, constraints=LinearConstraint(A, lb, np.inf), integrality=integ, bounds=Bounds(lo, hi), options=dict(time_limit=180, disp=False))
out = dict(status=r.status, message=r.message, incumbent=None if r.x is None else int(round(-r.fun)),
           dual_bound=float(-r.mip_dual_bound) if getattr(r, "mip_dual_bound", None) is not None else None,
           gap=getattr(r, "mip_gap", None), secs=round(time.time() - t0, 1), M=M, eps=eps)
if r.x is not None:                                    # exact recount of the incumbent spectrum
    lamF = L.exact_lam(r.x[:d] / SC, 10 ** 15); z = np.array([int(f * 10 ** 15) for f in lamF], dtype=object)
    vals = (R.astype(object) @ z).reshape(nr, d - 1)
    out.update(exact_count=int(sum(1 for row in vals if min(row) > 0)), gaps_strict=all(z[j + 1] > z[j] for j in range(d - 1)),
               spectrum=[str(f) for f in lamF])
json.dump(out, open(os.path.join(L.OUT, "milp_probe.json"), "w"))
print(json.dumps({k2: v for k2, v in out.items() if k2 != "spectrum"}))
