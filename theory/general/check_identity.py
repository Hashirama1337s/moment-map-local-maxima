"""Check (CHECKED NUMERICALLY) the two identities used in the proof of Theorem B at random T-fixed D:
 (W) alpha(ell) := ell_{c+alpha} - ell_c equals the weighted mean sum_p |a_p|^2 Delta_p / sum_p |a_p|^2  (a = E_alpha entries)
 (V) z^dag M z = (1/2W) sum_{p,q in supp} w_p w_q Delta_p Delta_q |y_p - y_q|^2 + alpha(ell) sum_{p not in supp} Delta_p |z_p|^2,
     z_p = a_p y_p on supp, with M = alpha(ell) diag(Delta) - v v^dag / W, v = a * Delta."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_core import *
from run_case import CASES

rng = np.random.default_rng(4242); out = {}
for case in ['u22', 'u23', 'u222', 'ferm42', 'ferm52', 'bos32', 'bos42', 'spin2', 'spin_h1', 'coll2', 'coll3', 'su3adj']:
    G = CASES[case](); d = G.d; eW = 0.0; eV = 0.0; nr = 0
    for trial in range(30):
        E = rng.normal(size=d); ell = G.Pdiag @ E
        for r in G.roots:
            pairs = r['pairs']; a = np.array([r['E'][cp, c] for (cp, c) in pairs]); w = np.abs(a) ** 2; Wt = w.sum()
            Dl = np.array([E[cp] - E[c] for (cp, c) in pairs]); al = ell[pairs[0][0]] - ell[pairs[0][1]]
            eW = max(eW, abs(al - np.sum(w * Dl) / Wt)); nr += 1
            v = a * Dl; M = al * np.diag(Dl) - np.outer(v, v.conj()) / Wt
            z = rng.normal(size=len(a)) + 1j * rng.normal(size=len(a)); lhs = np.real(z.conj() @ M @ z)
            s = np.abs(a) > 1e-12; y = np.zeros(len(a), complex); y[s] = z[s] / a[s]
            ws, Ds, ys = w[s], Dl[s], y[s]
            rhs = np.sum(np.outer(ws * Ds, ws * Ds) * np.abs(ys[:, None] - ys[None, :]) ** 2) / (2 * Wt) + al * np.sum(Dl[~s] * np.abs(z[~s]) ** 2)
            eV = max(eV, abs(lhs - rhs))
    out[case] = dict(root_blocks_checked=nr, weighted_mean_err=float(eW), variance_identity_err=float(eV),
                     support_sizes=sorted({int(np.sum(np.abs([r['E'][cp, c] for (cp, c) in r['pairs']]) > 1e-12)) for r in G.roots}),
                     block_sizes=sorted({len(r['pairs']) for r in G.roots}))
    print(case, out[case], flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'identities.json'), 'w'), indent=1)
