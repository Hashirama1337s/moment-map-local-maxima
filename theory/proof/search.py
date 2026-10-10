"""Break attempt: for FIXED generic (GUE) H and an adversarial equally-spaced H, find critical points of f of any
index by Levenberg-Marquardt on r = [L,H'] = 0 (exact Jacobian), from starts designed near degenerate local parts:
  S1: polygon saddle (constructed.family_B) + noise eps, then LM
  S2: Haar-random U, then LM
  S4: LM on the AUGMENTED system  [L,H'] = 0  AND  traceless(Tr_B H') = 0   (forces alpha fully degenerate)
      (and the transposed version for Tr_A when chosen)
  S3: polygon saddle + noise, then gradient DESCENT, then LM polish  (where do descents from the gap region land?)
Every converged point (||grad|| < 1e-12) is classified: Hessian index, gauge-invariant assignment test,
local-part gaps, N1 commutant-form minimum, P5 order-isomorphism (for assignments).
Usage: python search.py <m> <n> <seed> <kind: gue|eqsp> <budget_seconds>
"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import *
from constructed import family_B
from scipy.stats import unitary_group

OUT = os.path.dirname(os.path.abspath(__file__))

def lm_aug(U, H, m, n, basis, which, w=1.0, tol=1e-13, maxit=300):
    """LM on [ coords(i[L,H']) ; w*coords(traceless part of Tr_B H' (which='A') or Tr_A H' (which='B')) ]."""
    k = m if which == 'A' else n
    bk = herm_basis(k)
    def resfun(Hp, L):
        Tk = ptrB(Hp, m, n) if which == 'A' else ptrA(Hp, m, n)
        Tk = Tk - np.trace(Tk) / k * np.eye(k)
        return np.concatenate([coords(1j * comm(L, Hp), basis), w * coords(Tk, bk)])
    def jac(Hp, L):
        J1 = jac_resid(Hp, L, m, n, basis)
        cols = []
        for K in basis:
            dH = 1j * comm(Hp, K)
            Tk = ptrB(dH, m, n) if which == 'A' else ptrA(dH, m, n)
            Tk = Tk - np.trace(Tk) / k * np.eye(k)
            cols.append(w * coords(Tk, bk))
        return np.vstack([J1, np.array(cols).T])
    mu = 1e-3
    Hp, L = parts(U, H, m, n); r = resfun(Hp, L); nr = np.linalg.norm(r)
    for it in range(maxit):
        if nr < tol: break
        J = jac(Hp, L); JTJ = J.T @ J; g = J.T @ r
        while True:
            xi = -np.linalg.lstsq(JTJ + mu * np.eye(len(basis)), g, rcond=None)[0]
            Un = U @ expm(1j * from_coords(xi, basis))
            Hpn, Ln = parts(Un, H, m, n); rn = resfun(Hpn, Ln); nrn = np.linalg.norm(rn)
            if nrn < nr:
                U, Hp, L, r, nr = Un, Hpn, Ln, rn, nrn; mu = max(mu / 5, 1e-12 * max(1.0, float(np.trace(JTJ)) / len(basis))); break
            mu *= 4
            if mu > 1e12: return U, nr
    return U, nr

def classify(U, H, m, n, basis):
    Hp, L = parts(U, H, m, n)
    g = 2 * np.linalg.norm(comm(L, Hp))
    scale = np.linalg.norm(H) ** 2
    Hs = hess_analytic(Hp, L, m, n, basis); ev = np.linalg.eigvalsh(Hs)
    tol = 1e-8 * scale
    asg, s2, ov = is_assignment(Hp, m, n)
    ga, gb = local_gaps(Hp, m, n)
    q, _, _ = commutant_form(Hp, L, m, n)
    ordiso = None
    if asg:
        Hg = gauge_to_product(Hp, m, n)
        E = np.real(np.diag(Hg)); Lv = np.real(np.diag(P(np.diag(E).astype(complex), m, n)))
        dE = E[:, None] - E[None, :]; dL = Lv[:, None] - Lv[None, :]
        off = ~np.eye(m * n, dtype=bool)
        ordiso = float(np.min((dE * dL)[off]))  # > 0  <=> order-isomorphic (P5 local min)
    return dict(grad=float(g), f=cost(U, H, m, n), hess_min=float(ev[0]), n_neg=int(np.sum(ev < -tol)),
                n_zero=int(np.sum(np.abs(ev) <= tol)), assignment=bool(asg), s2=s2, ovdef=ov, alpha_gap=ga,
                beta_gap=gb, N1_min=q, ordiso_min=ordiso,
                local_min_candidate=bool(ev[0] >= -tol))

if __name__ == '__main__':
    m, n, seed, kind, budget = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], float(sys.argv[5])
    d = m * n; rng = np.random.default_rng(seed); basis = herm_basis(d)
    if kind == 'gue':
        G = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)); H = (G + G.conj().T) / (2 * np.sqrt(d))
    else:
        Q = unitary_group.rvs(d, random_state=seed); H = Q @ np.diag(np.arange(d, dtype=float)) @ Q.conj().T
    lam, V = np.linalg.eigh(H)
    t0 = time.time(); recs = []
    k = 0
    while time.time() - t0 < budget:
        mode = ['S1', 'S2', 'S4', 'S3'][k % 4]; k += 1
        if mode in ('S1', 'S3'):
            res = family_B(lam, m, n, rng) if m >= 2 else None
            if res is None: continue
            Hp0, _ = res; _, W = np.linalg.eigh(Hp0); U0 = V @ W.conj().T
            eps = [1e-3, 1e-2, 1e-1, 3e-1][rng.integers(4)]
            X = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)); X = (X + X.conj().T) / 2
            U0 = U0 @ expm(1j * eps * X / np.linalg.norm(X))
        else:
            U0 = unitary_group.rvs(d, random_state=int(rng.integers(1 << 31))); eps = None
        ts = time.time()
        if mode == 'S3':
            U1, f1, g1 = descend(U0, H, m, n, iters=3000, tol=1e-6)
            U1, nr, _ = lm_critical(U1, H, m, n, basis, tol=1e-14)
        elif mode == 'S4':
            which = 'A' if (rng.random() < 0.5 or n < 2) else 'B'
            U1, nr_aug = lm_aug(U0, H, m, n, basis, which)
            U1, nr, _ = lm_critical(U1, H, m, n, basis, tol=1e-14)
            eps = which
        else:
            U1, nr, _ = lm_critical(U0, H, m, n, basis, tol=1e-14)
        c = classify(U1, H, m, n, basis); c.update(mode=mode, eps=eps, secs=time.time() - ts)
        recs.append(c)
        if c['grad'] < 1e-12:
            print(f"{mode} eps={eps} grad={c['grad']:.1e} f={c['f']:.6f} Hmin={c['hess_min']:+.3e} nneg={c['n_neg']:2d} "
                  f"asg={c['assignment']} aG={c['alpha_gap']:.1e} bG={c['beta_gap']:.1e} N1={c['N1_min']:+.2e} "
                  f"ordiso={c['ordiso_min']} cand={c['local_min_candidate']}", flush=True)
        else:
            print(f"{mode} eps={eps} NOT CONVERGED grad={c['grad']:.1e}", flush=True)
    conv = [r for r in recs if r['grad'] < 1e-12]
    degen = [r for r in conv if min(r['alpha_gap'], r['beta_gap']) < 1e-6]
    nonasg = [r for r in conv if not r['assignment']]
    cands = [r for r in conv if r['local_min_candidate']]
    bad = [r for r in cands if (not r['assignment']) or min(r['alpha_gap'], r['beta_gap']) < 1e-6 or r['N1_min'] < -1e-8]
    bad_p5 = [r for r in cands if r['assignment'] and (r['ordiso_min'] is None or r['ordiso_min'] <= 0)]
    summ = dict(m=m, n=n, seed=seed, kind=kind, runs=len(recs), converged=len(conv), degenerate_local=len(degen),
                non_assignment=len(nonasg), local_min_candidates=len(cands), theorem_violations=len(bad),
                p5_violations=len(bad_p5),
                degenerate_with_negative_N1=sum(r['N1_min'] < -1e-8 for r in degen),
                nonasg_with_negative_N1=sum(r['N1_min'] < -1e-8 for r in nonasg),
                max_N1_over_degenerate=max([r['N1_min'] for r in degen], default=None),
                max_hessmin_over_nonassignment=max([r['hess_min'] for r in nonasg], default=None))
    print("SUMMARY", json.dumps(summ))
    json.dump(dict(summary=summ, records=recs), open(os.path.join(OUT, f'search_{m}x{n}_{kind}_{seed}.json'), 'w'), indent=0)
