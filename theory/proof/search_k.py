"""k-partite break attempt (Theorem G-k): critical points of fixed H of any index, from starts near degenerate local parts.
  S1: family-B-k saddle (rows 0,1 of a random factor j tied; polygon coherences; exact critical point of the given H) + noise, LM
  S2: Haar-random U, LM
  S4: LM on [L,H'] = 0 AND traceless(Tr_{not j} H') = 0 (factor j fully degenerate), then plain LM
  S3: S1 start, gradient descent, LM polish
Classification: Hessian index, k-partite assignment test, local gaps per factor, N1 commutant form, order-isomorphism,
and the D2-k re-placement check (||D-L|| = ||H'-L||, f(D) <= f(U)).
Usage: python search_k.py <dims e.g. 2,2,3> <seed> <gue|eqsp> <budget_s>"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core_k import *
from constructed import family_B
from scipy.stats import unitary_group

def family_B_k(lam, dims, j, rng):
    dj = dims[j]; others = [x for i, x in enumerate(dims) if i != j]; n = int(np.prod(others))
    res = family_B(lam, dj, n, rng)
    if res is None: return None
    Hq = res[0]  # ordering (j, others...)
    k = len(dims); order = [j] + [i for i in range(k) if i != j]
    T = Hq.reshape(tuple([dims[i] for i in order]) * 2)
    inv = np.argsort(order)
    T = T.transpose(list(inv) + list(k + inv))
    return T.reshape(int(np.prod(dims)), -1)

def lm_aug(U, H, dims, basis, j, tol=1e-13, maxit=300):
    dj = dims[j]; bk = herm_basis(dj)
    def res(Hp, L):
        T = ptr_keep(Hp, dims, j); T = T - np.trace(T) / dj * np.eye(dj)
        return np.concatenate([coords(1j * comm(L, Hp), basis), coords(T, bk)])
    def jac(Hp, L):
        J1 = jac_resid(Hp, L, dims, basis); cols = []
        for K in basis:
            T = ptr_keep(1j * comm(Hp, K), dims, j); T = T - np.trace(T) / dj * np.eye(dj); cols.append(coords(T, bk))
        return np.vstack([J1, np.array(cols).T])
    mu = 1e-3; Hp, L = parts(U, H, dims); r = res(Hp, L); nr = np.linalg.norm(r)
    for it in range(maxit):
        if nr < tol: break
        J = jac(Hp, L); JTJ = J.T @ J; g = J.T @ r
        while True:
            xi = -np.linalg.lstsq(JTJ + mu * np.eye(len(basis)), g, rcond=None)[0]
            Un = U @ expm(1j * from_coords(xi, basis)); Hpn, Ln = parts(Un, H, dims); rn = res(Hpn, Ln); nrn = np.linalg.norm(rn)
            if nrn < nr:
                U, Hp, L, r, nr = Un, Hpn, Ln, rn, nrn; mu = max(mu / 5, 1e-12 * max(1.0, float(np.trace(JTJ)) / len(basis))); break
            mu *= 4
            if mu > 1e12: return U
    return U

def classify(U, H, dims, basis):
    Hp, L = parts(U, H, dims); g = 2 * np.linalg.norm(comm(L, Hp)); scale = np.linalg.norm(H) ** 2; tol = 1e-8 * scale
    ev = np.linalg.eigvalsh(hess_analytic(Hp, L, dims, basis))
    asg, imp, ov = is_assignment(Hp, dims); gaps = local_gaps(Hp, dims); q, gs = commutant_form(Hp, L, dims)
    oi = None
    if asg:
        T = gauge(Hp, dims); H2 = T.conj().T @ Hp @ T; E = np.real(np.diag(H2)); Lv = np.real(np.diag(P1(np.diag(E).astype(complex), dims)))
        dE = E[:, None] - E[None, :]; dL = Lv[:, None] - Lv[None, :]; off = ~np.eye(len(E), dtype=bool)
        oi = float(np.min((dE * dL)[off]))
    a, b, fD, leak, Loff = d2_reassign(Hp, dims)
    return dict(grad=float(g), f=cost(U, H, dims), hess_min=float(ev[0]), n_neg=int(np.sum(ev < -tol)), n_zero=int(np.sum(np.abs(ev) <= tol)),
                assignment=asg, impurity=imp, ovdef=ov, local_gaps=gaps, N1_min=q, ordiso_min=oi,
                cand=bool(ev[0] >= -tol), d2_fU=a, d2_DL=b, d2_fD=fD, d2_leak=leak, d2_Loff=Loff)

if __name__ == '__main__':
    dims = tuple(int(x) for x in sys.argv[1].split(',')); seed = int(sys.argv[2]); kind = sys.argv[3]; budget = float(sys.argv[4])
    d = int(np.prod(dims)); k = len(dims); rng = np.random.default_rng(seed); basis = herm_basis(d)
    if kind == 'gue':
        G = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)); H = (G + G.conj().T) / (2 * np.sqrt(d))
    else:
        Q = unitary_group.rvs(d, random_state=seed); H = Q @ np.diag(np.arange(d, dtype=float)) @ Q.conj().T
    lam, V = np.linalg.eigh(H); kern = d + sum(x * (x - 1) for x in dims)
    t0 = time.time(); recs = []; it = 0
    while time.time() - t0 < budget:
        mode = ['S1', 'S2', 'S4', 'S3'][it % 4]; it += 1; info = None
        if mode in ('S1', 'S3'):
            j = int(rng.integers(k)); Hq = family_B_k(lam, dims, j, rng)
            if Hq is None: continue
            _, W = np.linalg.eigh(Hq); U0 = V @ W.conj().T
            eps = [1e-3, 1e-2, 1e-1, 3e-1][rng.integers(4)]
            X = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)); X = (X + X.conj().T) / 2
            U0 = U0 @ expm(1j * eps * X / np.linalg.norm(X)); info = dict(j=j, eps=eps)
        else:
            U0 = unitary_group.rvs(d, random_state=int(rng.integers(1 << 31)))
        ts = time.time()
        if mode == 'S3':
            U1 = descend(U0, H, dims); U1, nr = lm_critical(U1, H, dims, basis)
        elif mode == 'S4':
            j = int(rng.integers(k)); info = dict(j=j); U1 = lm_aug(U0, H, dims, basis, j); U1, nr = lm_critical(U1, H, dims, basis)
        else:
            U1, nr = lm_critical(U0, H, dims, basis)
        c = classify(U1, H, dims, basis); c.update(mode=mode, info=info, secs=time.time() - ts); recs.append(c)
        if c['grad'] < 1e-12:
            print(f"{mode} {info} grad={c['grad']:.1e} f={c['f']:.5f} Hmin={c['hess_min']:+.3e} nneg={c['n_neg']} nzero={c['n_zero']} "
                  f"asg={c['assignment']} gaps={[f'{x:.1e}' for x in c['local_gaps']]} N1={c['N1_min']:+.2e} oi={c['ordiso_min']} "
                  f"D2: fU={c['d2_fU']:.5f} |D-L|^2={c['d2_DL']:.5f} fD={c['d2_fD']:.5f}", flush=True)
        else:
            print(f"{mode} NOT CONVERGED grad={c['grad']:.1e}", flush=True)
    conv = [r for r in recs if r['grad'] < 1e-12]
    degen = [r for r in conv if min(r['local_gaps']) < 1e-6]
    nonasg = [r for r in conv if not r['assignment']]
    cands = [r for r in conv if r['cand']]
    viol = [r for r in cands if (not r['assignment']) or min(r['local_gaps']) < 1e-6 or r['N1_min'] < -1e-8 or (r['ordiso_min'] or -1) <= 0
            or r['n_zero'] != kern]
    gapviol = [r for r in conv if (not r['assignment'] or min(r['local_gaps']) < 1e-6) and not (r['n_neg'] > 0 and r['N1_min'] < -1e-8)]
    scale = np.linalg.norm(H) ** 2
    d2viol = [r for r in conv if abs(r['d2_fU'] - r['d2_DL']) > 1e-9 * scale or r['d2_fD'] > r['d2_fU'] + 1e-9 * scale or r['d2_leak'] > 1e-7]
    summ = dict(dims=dims, seed=seed, kind=kind, runs=len(recs), converged=len(conv), degenerate_local=len(degen), non_assignment=len(nonasg),
                candidates=len(cands), kernel_expected=kern, candidate_violations=len(viol), gapclass_without_negative_direction=len(gapviol),
                d2_violations=len(d2viol), max_grad=max([r['grad'] for r in conv], default=None),
                worst_N1_degenerate=max([r['N1_min'] for r in degen], default=None),
                worst_hessmin_nonassignment=max([r['hess_min'] for r in nonasg], default=None),
                max_fD_over_fU=max([r['d2_fD'] / r['d2_fU'] for r in conv if r['d2_fU'] > 1e-12], default=None))
    print("SUMMARY", json.dumps(summ))
    json.dump(dict(summary=summ, records=recs), open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
              f"searchk_{'x'.join(map(str, dims))}_{kind}_{seed}.json"), 'w'), indent=0)
