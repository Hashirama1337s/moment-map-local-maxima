"""K1 near-miss test: order-isomorphic fixpoint assignments with one adjacent-rank transposition (borderline cases).
Checks (closed-form strict local min mod gauge+torus) <=> (order-isomorphic), and closed form == analytic Hessian."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core_k import *
from k1_check import predicted, order_iso, fixpoint
rng = np.random.default_rng(161803); rows = []
for dims in [(2, 2, 2), (2, 2, 3), (2, 3, 3)]:
    d = int(np.prod(dims)); basis = herm_basis(d); kern = d + sum(x * (x - 1) for x in dims)
    for hs in range(4):
        G = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)); H = (G + G.conj().T) / (2 * np.sqrt(d))
        lam, V = np.linalg.eigh(H); scale = np.linalg.norm(H) ** 2
        for t in range(6):
            base = fixpoint(lam, dims, rng)
            for r in range(d - 1):
                perm = base.copy()
                ci = int(np.where(np.argsort(lam)[r] == perm)[0][0]); cj = int(np.where(np.argsort(lam)[r + 1] == perm)[0][0])
                perm[ci], perm[cj] = perm[cj], perm[ci]
                U = V[:, perm]; Hp, L = parts(U, H, dims); E = np.real(np.diag(Hp)); Lv = np.real(np.diag(L))
                pr = predicted(E, Lv, dims); tol = 1e-9 * scale
                strict = bool(np.sum(np.abs(pr) <= tol) == kern and pr[0] > -tol); oi = order_iso(E, Lv) > 0
                diff = None
                if r % 4 == 0:
                    diff = float(np.max(np.abs(np.linalg.eigvalsh(hess_analytic(Hp, L, dims, basis)) - pr)))
                rows.append(dict(dims=dims, strict=strict, orderiso=bool(oi), diff=diff))
agree = sum(r['strict'] == r['orderiso'] for r in rows)
print(f"near-miss assignments: {len(rows)}; agreement {agree}/{len(rows)}; strict {sum(r['strict'] for r in rows)}; "
      f"max closed-vs-analytic {max(r['diff'] for r in rows if r['diff'] is not None):.1e}")
json.dump(dict(seed=161803, rows=rows), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'k1_nearmiss.json'), 'w'))
