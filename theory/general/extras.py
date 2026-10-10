"""Extra checks.  Usage: python extras.py <part> [args]
  lemma3 <case> <seed> <nlm>   : at LM critical points that are not T-fixed, check the explicit descent bound
                                 Q(K*)/||K*||^2 <= -2||[X,H']||^2/||X||^2, K* = i[X,H'], X in z_k(L) (centraliser of L in k+R1)
  ties <case>                  : equally spaced spectrum; classify T-fixed assignments with exact L-ties; Hessian + Givens
  coherent <n> <N> <seed>      : fermions; coherent orders of N-subset sums vs self-consistent (order-isomorphic) assignments
"""
import sys, os, json, time, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_core import *
from run_case import CASES

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

def lemma3(case, seed, nlm):
    G = CASES[case](); d = G.d; H = gue(d, seed); hb = herm_basis(d); rng = np.random.default_rng(seed + 7)
    nH2 = float(np.real(np.vdot(H, H))); res = dict(case=case, seed=seed, n=0, nonT=0, worst_ratio=None, max_comm_KL=0.0, line_err=0.0)
    ratios = []
    for i in range(nlm):
        U, nr = lm_critical(G, haar(d, rng), H, hb)
        if nr > 1e-11 * nH2: continue
        res['n'] += 1; Hp, L = parts(G, U, H); Z = centralizer_in_k(G, L)
        M = np.array([comm(X, Hp).reshape(-1) for X in Z]).T
        u, s, vt = np.linalg.svd(M, full_matrices=False)
        if s[0] < 1e-6 * np.sqrt(nH2): continue          # T-fixed
        res['nonT'] += 1
        X = np.einsum('k,kij->ij', vt[0], Z); C = comm(X, Hp); Ks = 1j * C; Ks = (Ks + Ks.conj().T) / 2
        res['max_comm_KL'] = max(res['max_comm_KL'], float(np.linalg.norm(comm(Ks, L)) / np.linalg.norm(Ks)))
        PZ = G.P(1j * comm(Hp, Ks)); Q = -2 * np.real(np.vdot(PZ, PZ)) / np.real(np.vdot(Ks, Ks))
        bound = -2 * np.real(np.vdot(C, C)) / np.real(np.vdot(X, X))
        ratios.append(Q / bound)
        # line test: f(U exp(isK*)) - f(U) ~ Q s^2/2  (K* normalised)
        Kn = Ks / np.linalg.norm(Ks); f0 = cost(G, U, H)
        for sstep in (1e-2, 3e-3):
            df = cost(G, U @ expm(1j * sstep * Kn), H) - f0
            res['line_err'] = max(res['line_err'], abs(df / (0.5 * Q * sstep ** 2) - 1))
    res['worst_ratio'] = float(min(ratios)) if ratios else None   # must be >= 1 (Q at least as negative as the bound)
    print(json.dumps(res)); return res

def eqsp_H(d, seed):
    from scipy.stats import unitary_group
    Q = unitary_group.rvs(d, random_state=seed); return Q @ np.diag(np.arange(d, dtype=float)) @ Q.conj().T

def ties(case, seed=5):
    G = CASES[case](); d = G.d; H = eqsp_H(d, seed); lam = np.sort(np.linalg.eigvalsh(H)); hb = herm_basis(d)
    nH2 = float(np.sum(lam ** 2)); tol = 1e-8 * nH2
    rootpair = np.zeros((d, d), bool)
    for r in G.roots:
        for (cp, c) in r['pairs']: rootpair[cp, c] = rootpair[c, cp] = True
    out = dict(case=case, strict=0, strict_ok=0, tied=0, tied_rootpair=0, tied_rootpair_neg=0, tied_nonroot_only=0,
               tied_nonroot_psd=0, tied_nonroot_givens_drop=0, tied_nonroot_neg=0, violating=0, violating_neg=0, worst_drop=None)
    drops = []
    seen = set()
    for perm in itertools.permutations(range(d)):
        if not G.multfree:
            k = tuple(G.label[list(perm)])
            if k in seen: continue
            seen.add(k)
        E = D_from_perm(G, lam, perm); ell = G.Pdiag @ E
        off = G.label[:, None] != G.label[None, :]
        prod = (ell[:, None] - ell[None, :]) * (E[:, None] - E[None, :]); sc = 1e-9 * nH2
        if np.any(prod[off] < -sc):
            out['violating'] += 1
            if out['violating'] <= 300:
                U = U_from_E(H, E); Hp, L = parts(G, U, H); ev = np.linalg.eigvalsh(hess_analytic(G, Hp, L, hb))
                out['violating_neg'] += int(ev[0] < -tol)
            continue
        tie = off & (np.abs(prod) <= sc) & (np.arange(d)[:, None] < np.arange(d)[None, :])
        U = U_from_E(H, E); Hp, L = parts(G, U, H); ev = np.linalg.eigvalsh(hess_analytic(G, Hp, L, hb))
        if not tie.any():
            out['strict'] += 1; out['strict_ok'] += int(ev[0] >= -tol); continue
        out['tied'] += 1
        pairs = list(zip(*np.where(tie)))
        if any(rootpair[c, cp] for c, cp in pairs):
            out['tied_rootpair'] += 1; out['tied_rootpair_neg'] += int(ev[0] < -tol); continue
        out['tied_nonroot_only'] += 1
        if ev[0] < -tol: out['tied_nonroot_neg'] += 1; continue
        out['tied_nonroot_psd'] += 1
        c, cp = pairs[0]; f0 = cost(G, U, H); th = 0.2
        Rg = np.eye(d, dtype=complex); Rg[c, c] = Rg[cp, cp] = np.cos(th); Rg[c, cp] = -np.sin(th); Rg[cp, c] = np.sin(th)
        df = cost(G, U @ Rg, H) - f0; e = np.zeros(d); e[c], e[cp] = 1, -1
        pred = -np.sin(th) ** 4 * (E[c] - E[cp]) ** 2 * np.linalg.norm(G.Pdiag @ e) ** 2
        drops.append(abs(df - pred)); out['tied_nonroot_givens_drop'] += int(df < -1e-12)
    out['givens_formula_err'] = float(max(drops)) if drops else None
    print(json.dumps(out)); return out

def coherent(n, N, seed, nsamp=200000):
    """Coherent orders: orders of {sum_{i in S} eps_i} for sorted eps (mod nothing: eps sorted fixes S_n). Each coherent
    order sigma gives one T-fixed assignment (sorted spectrum placed along sigma); compare with order-isomorphism."""
    G = g_fermions(n, N); d = G.d; H = gue(d, seed); lam = np.sort(np.linalg.eigvalsh(H))
    subsets = list(itertools.combinations(range(n), N))
    # map subsets to weight-basis index: basis ordered by JW bit pattern; recover from n_i diagonals
    occ = np.array([np.real(np.diag(t)) for t in G.tgens]).T.round().astype(int)    # d x n occupation
    idx = {tuple(np.where(occ[c])[0]): c for c in range(d)}
    rng = np.random.default_rng(seed); orders = set()
    for _ in range(nsamp // 1000):
        eps = np.sort(rng.normal(size=(1000, n)) * rng.exponential(size=(1000, 1)) ** rng.uniform(0, 3, size=(1000, 1)), axis=1)
        sums = np.array([eps[:, list(S)].sum(1) for S in subsets]).T
        for row in np.argsort(sums, axis=1): orders.add(tuple(row))
    res = dict(n=n, N=N, seed=seed, coherent_orders=len(orders), self_consistent=0, f=[])
    for o in orders:
        E = np.zeros(d)
        for rank, si in enumerate(o): E[idx[subsets[si]]] = lam[rank]
        ell = G.Pdiag @ E; prod = (ell[:, None] - ell[None, :]) * (E[:, None] - E[None, :]); off = ~np.eye(d, dtype=bool)
        if np.min(prod[off]) > 1e-12: res['self_consistent'] += 1; res['f'].append(float(np.sum((E - ell) ** 2)))
    res['f'] = sorted(res['f'])
    print(json.dumps(res)); return res

if __name__ == '__main__':
    part = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
    if part == 'lemma3': r = lemma3(sys.argv[2], int(sys.argv[3]), int(sys.argv[4])); tag = f'lemma3_{sys.argv[2]}_{sys.argv[3]}'
    elif part == 'ties': r = ties(sys.argv[2]); tag = f'ties_{sys.argv[2]}'
    elif part == 'coherent': r = coherent(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])); tag = f'coherent_{sys.argv[2]}_{sys.argv[3]}_{sys.argv[4]}'
    json.dump(r, open(os.path.join(OUT, tag + '.json'), 'w'), indent=1)
