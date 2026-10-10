"""K1 check (k-partite Hessian at an assignment) and the exact Givens tie formula.
Closed form (unnormalised f; divide by N = ||H||^2 for C1):
  - d zeros (diagonal K: gauge-diagonal + torus);
  - each pair of cells at Hamming distance >= 2: 2 (L_c - L_c')(E_c - E_c'), multiplicity 2;
  - each (j, {a,a'}): with Delta_r = E[a at j, r] - E[a' at j, r] over the N_j = d/d_j other-coordinate tuples r,
    2 * eig( mean(Delta) diag(Delta) - Delta Delta^T / N_j ), multiplicity 2.
Claim: strict local min mod gauge+torus  <=>  L order-isomorphic to E.
Givens: for a tie L_c = L_c' at Hamming distance >= 2,  f(theta) = f(0) - sin^4(theta) (E_c-E_c')^2 ||P1(e_c - e_c')||^2."""
import sys, os, json, time, numpy as np
from itertools import product, combinations
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core_k import *

def predicted(E, Lv, dims):
    cells = list(product(*[range(x) for x in dims])); d = len(cells); pos = {c: i for i, c in enumerate(cells)}
    ev = [0.0] * d
    for i, j in combinations(range(d), 2):
        if sum(a != b for a, b in zip(cells[i], cells[j])) >= 2:
            v = 2 * (Lv[i] - Lv[j]) * (E[i] - E[j]); ev += [v, v]
    for jj, dj in enumerate(dims):
        others = list(product(*[range(x) for t, x in enumerate(dims) if t != jj])); Nj = len(others)
        for a, a2 in combinations(range(dj), 2):
            D = []
            for r in others:
                c = list(r); c.insert(jj, a); c2 = list(r); c2.insert(jj, a2)
                D.append(E[pos[tuple(c)]] - E[pos[tuple(c2)]])
            D = np.array(D); M = D.mean() * np.diag(D) - np.outer(D, D) / Nj
            for v in np.linalg.eigvalsh(M): ev += [2 * v, 2 * v]
    assert len(ev) == d * d
    return np.sort(ev)

def order_iso(E, Lv):
    dE = E[:, None] - E[None, :]; dL = Lv[:, None] - Lv[None, :]; off = ~np.eye(len(E), dtype=bool)
    return float(np.min((dE * dL)[off]))

def fixpoint(lam, dims, rng):
    d = len(lam); perm = rng.permutation(d)
    for _ in range(200):
        E = lam[perm]; Lv = np.real(np.diag(P1(np.diag(E).astype(complex), dims)))
        new = np.empty(d, int); new[np.argsort(Lv, kind='stable')] = np.argsort(lam)
        if np.all(new == perm): break
        perm = new
    return perm

if __name__ == '__main__':
    rng = np.random.default_rng(271828); out = []; t0 = time.time()
    for dims in [(2, 2, 2), (2, 2, 3), (2, 3, 3), (2, 2, 2, 2)]:
        d = int(np.prod(dims)); k = len(dims); basis = herm_basis(d)
        kern = d + sum(dj * (dj - 1) for dj in dims)   # = gauge (sum d_j^2 - k + 1) + torus (d - sum d_j + k - 1)
        for hs in range(3):
            G = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)); H = (G + G.conj().T) / (2 * np.sqrt(d))
            lam, V = np.linalg.eigh(H); scale = np.linalg.norm(H) ** 2
            nA = 40 if d <= 12 else 12
            for t in range(nA):
                perm = fixpoint(lam, dims, rng) if t % 2 else rng.permutation(d)
                U = V[:, perm]; Hp, L = parts(U, H, dims); E = np.real(np.diag(Hp)); Lv = np.real(np.diag(L))
                pr = predicted(E, Lv, dims)
                ha = np.linalg.eigvalsh(hess_analytic(Hp, L, dims, basis))
                diff = float(np.max(np.abs(ha - pr)))
                agd = None
                if t < 3 and d <= 12:
                    hg = hess_autograd(U, H, dims, basis); agd = float(np.max(np.abs(hg - hess_analytic(Hp, L, dims, basis))))
                tol = 1e-9 * scale
                strict = bool(np.sum(np.abs(pr) <= tol) == kern and pr[0] > -tol)
                oi = order_iso(E, Lv)
                out.append(dict(dims=dims, hseed=hs, fixpoint=bool(t % 2), closed_vs_analytic=diff, autograd_vs_analytic=agd,
                                strict=strict, orderiso=bool(oi > 0), oi_min=oi, grad=float(2 * np.linalg.norm(comm(L, Hp)))))
        print(dims, "done", time.time() - t0, flush=True)
    agree = sum(r['strict'] == r['orderiso'] for r in out)
    print(f"assignments tested: {len(out)}; strict<=>order-iso agreement: {agree}/{len(out)}; strict count: "
          f"{sum(r['strict'] for r in out)}; max |closed - analytic|: {max(r['closed_vs_analytic'] for r in out):.2e}; "
          f"max |autograd - analytic|: {max(r['autograd_vs_analytic'] or 0 for r in out):.2e}; max grad: {max(r['grad'] for r in out):.1e}")
    # ---- Givens tie test ----
    giv = []
    for dims in [(2, 2, 2), (2, 2, 3), (2, 2, 2, 2)]:
        d = int(np.prod(dims)); cells = list(product(*[range(x) for x in dims]))
        for rep in range(5):
            E = rng.normal(size=d)
            while True:
                i, j = rng.choice(d, 2, replace=False)
                if sum(a != b for a, b in zip(cells[i], cells[j])) >= 2: break
            w = np.zeros(d); w[i], w[j] = 1, -1
            Pw = np.real(np.diag(P1(np.diag(w).astype(complex), dims)))
            Lv = np.real(np.diag(P1(np.diag(E).astype(complex), dims)))
            E = E - (Lv[i] - Lv[j]) / (Pw @ Pw) * Pw          # enforce L_i = L_j
            H = np.diag(E).astype(complex); Lv = np.real(np.diag(P1(H, dims)))
            f0 = cost(np.eye(d, dtype=complex), H, dims); errs = []
            for th in [0.05, 0.3, 0.9, 1.4]:
                Kg = np.zeros((d, d), complex); Kg[i, j] = -1j; Kg[j, i] = 1j
                ft = cost(expm(1j * th * Kg), H, dims)
                pred = -np.sin(th) ** 4 * (E[i] - E[j]) ** 2 * (Pw @ Pw)
                errs.append(abs((ft - f0) - pred))
            giv.append(dict(dims=dims, hamming=int(sum(a != b for a, b in zip(cells[i], cells[j]))), tie=float(abs(Lv[i] - Lv[j])),
                            dE=float(E[i] - E[j]), max_abs_err=float(max(errs)), spec_gap=float(np.min(np.diff(np.sort(E))))))
    print("Givens exact-formula test:", len(giv), "cases; max abs error", max(g['max_abs_err'] for g in giv),
          "; min spectral gap", min(g['spec_gap'] for g in giv))
    json.dump(dict(seed=271828, assignments=out, givens=giv), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'k1_check.json'), 'w'), indent=0)
