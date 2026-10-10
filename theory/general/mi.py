"""H3 test: mutual information I(x) = S(Tr_B x) + S(Tr_A x) - S(x) on the unitary orbit of a full-rank density matrix.
Prediction (Theorems A^F, B^F in GENERAL.md, F = -S(rho_A) - S(rho_B)): local minima = product-diagonal x whose eigenvalues
p_ab are order-isomorphic to Lambda_ab = log r_a + log c_b (r, c = its own marginal spectra), i.e. to the outer product r (x) c.
Usage:
  python mi.py local <dA> <dB> <seed> <nstarts>   : descent + autograd Hessians + enumeration + block formula
  python mi.py census <dA> <dB> <nspec> <seed>     : classes of GLOBAL minima over random spectra (cubicle candidates)
"""
import sys, os, json, time, itertools, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_core import g_product, herm_basis, haar, comm
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
c128 = torch.complex128

def spectrum(d, rng):
    conc = 10 ** rng.uniform(-0.7, 0.7); p = rng.dirichlet(np.full(d, conc))
    u = rng.uniform(size=d); p = 0.97 * p + 0.03 * u / u.sum(); return p / p.sum()   # full rank, a.s. simple

def marg_np(x, dA, dB):
    X = x.reshape(dA, dB, dA, dB); return np.einsum('abcb->ac', X), np.einsum('abad->bd', X)

def ent(m):
    e = np.linalg.eigvalsh(m); e = e[e > 1e-300]; return float(-np.sum(e * np.log(e)))

def MI_np(x, dA, dB):
    rA, rB = marg_np(x, dA, dB); return ent(rA) + ent(rB) - ent(x)

def Lam_np(x, dA, dB):
    rA, rB = marg_np(x, dA, dB); ea, va = np.linalg.eigh(rA); eb, vb = np.linalg.eigh(rB)
    LA = va @ np.diag(np.log(ea)) @ va.conj().T; LB = vb @ np.diag(np.log(eb)) @ vb.conj().T
    return np.kron(LA, np.eye(dB)) + np.kron(np.eye(dA), LB)

def descend(rho, dA, dB, U0, iters=6000, dev='cuda'):
    dev = torch.device(dev if torch.cuda.is_available() else 'cpu'); d = dA * dB
    R = torch.tensor(rho, dtype=c128, device=dev); U = torch.tensor(np.array(U0), dtype=c128, device=dev); n = U.shape[0]
    IA = torch.eye(dA, dtype=c128, device=dev); IB = torch.eye(dB, dtype=c128, device=dev)
    def ev(U):
        x = U.mH @ R @ U; x = (x + x.mH) / 2; X = x.reshape(n, dA, dB, dA, dB)
        rA = torch.einsum('nabcb->nac', X); rB = torch.einsum('nabad->nbd', X)
        ea, va = torch.linalg.eigh(rA); eb, vb = torch.linalg.eigh(rB)
        ea = ea.clamp(min=1e-300); eb = eb.clamp(min=1e-300)
        val = -(ea * torch.log(ea)).sum(-1) - (eb * torch.log(eb)).sum(-1)
        LA = va @ torch.diag_embed(torch.log(ea).to(c128)) @ va.mH; LB = vb @ torch.diag_embed(torch.log(eb).to(c128)) @ vb.mH
        Lam = torch.einsum('nac,bd->nabcd', LA, IB).reshape(n, d, d) + torch.einsum('ac,nbd->nabcd', IA, LB).reshape(n, d, d)
        return val, x, Lam
    f, x, Lam = ev(U); t = torch.full((n,), 0.5, dtype=torch.float64, device=dev)
    for it in range(iters):
        G = -1j * (Lam @ x - x @ Lam); G = (G + G.mH) / 2; g2 = torch.real((G.conj() * G).sum((1, 2)))
        if float(torch.sqrt(g2).max()) < 1e-11: break
        Un = U @ torch.linalg.matrix_exp(-1j * t[:, None, None].to(c128) * G); fn, xn, Ln = ev(Un)
        ok = fn <= f - 0.3 * t * g2
        U = torch.where(ok[:, None, None], Un, U); f = torch.where(ok, fn, f)
        x = torch.where(ok[:, None, None], xn, x); Lam = torch.where(ok[:, None, None], Ln, Lam)
        t = torch.where(ok, torch.clamp(t * 1.5, max=50.0), (t * 0.5).clamp(min=1e-14))
    G = -1j * (Lam @ x - x @ Lam); gn = torch.sqrt(torch.real((G.conj() * G).sum((1, 2))))
    return U.cpu().numpy(), f.cpu().numpy(), gn.cpu().numpy(), it

def hess_autograd_mi(U, rho, dA, dB, basis):
    Ut = torch.tensor(U, dtype=c128); Rt = torch.tensor(rho, dtype=c128); Bt = torch.tensor(np.array(basis), dtype=c128)
    def f(xv):
        K = torch.einsum('p,pij->ij', xv.to(c128), Bt); V = Ut @ torch.linalg.matrix_exp(1j * K); x = V.conj().T @ Rt @ V
        X = x.reshape(dA, dB, dA, dB); rA = torch.einsum('abcb->ac', X); rB = torch.einsum('abad->bd', X)
        ea = torch.linalg.eigvalsh(rA); eb = torch.linalg.eigvalsh(rB)
        return -(ea * torch.log(ea)).sum() - (eb * torch.log(eb)).sum()
    return torch.autograd.functional.hessian(f, torch.zeros(len(basis), dtype=torch.float64)).numpy()

def _mi_fn(U, rho, dA, dB, basis):
    Ut = torch.tensor(U, dtype=c128); Rt = torch.tensor(rho, dtype=c128); Bt = torch.tensor(np.array(basis), dtype=c128)
    def f(xv):
        K = torch.einsum('p,pij->ij', xv.to(c128), Bt); V = Ut @ torch.linalg.matrix_exp(1j * K); x = V.conj().T @ Rt @ V
        X = x.reshape(dA, dB, dA, dB); ea = torch.linalg.eigvalsh(torch.einsum('abcb->ac', X)); eb = torch.linalg.eigvalsh(torch.einsum('abad->bd', X))
        return -(ea * torch.log(ea)).sum() - (eb * torch.log(eb)).sum()
    return f

def newton_polish(U, rho, dA, dB, basis, iters=8):
    """Newton steps (pseudo-inverse, kernel dropped) with autograd gradient/Hessian; returns U, ||grad||."""
    from scipy.linalg import expm as sexpm
    gn = None
    for it in range(iters + 1):
        f = _mi_fn(U, rho, dA, dB, basis); x0 = torch.zeros(len(basis), dtype=torch.float64, requires_grad=True)
        val = f(x0); g = torch.autograd.grad(val, x0)[0].numpy(); gn = float(np.linalg.norm(g))
        if gn < 1e-12 or it == iters: break
        Hs = torch.autograd.functional.hessian(f, torch.zeros(len(basis), dtype=torch.float64)).numpy(); Hs = (Hs + Hs.T) / 2
        w, V = np.linalg.eigh(Hs); keep = np.abs(w) > 1e-9 * max(1.0, np.abs(w).max())
        step = -(V[:, keep] @ ((V[:, keep].T @ g) / w[keep]))
        U = U @ sexpm(1j * np.einsum('p,pij->ij', step, basis))
    return U, gn

def block_spectrum_mi(G, E, dA, dB):
    """Closed form Hessian spectrum of I at product-diagonal D = diag(E) (theorem B^F): root blocks are
    alpha(Lam) diag(Delta) - kappa v v^dag / W with kappa = alpha(Lam)/alpha(ell); other pairs (Lam_c - Lam_c')(E_c - E_c')."""
    d = G.d; P = E.reshape(dA, dB); r = P.sum(1); c = P.sum(0)
    Lam = (np.log(r)[:, None] + np.log(c)[None, :]).reshape(-1); ell = G.Pdiag @ E
    ev = [0.0] * d; done = np.zeros((d, d), bool)
    for rt in G.roots:
        pairs = rt['pairs']; Ea = rt['E']; W = float(np.sum(np.abs(Ea) ** 2))
        a = np.array([Ea[cp, cc] for (cp, cc) in pairs]); Dl = np.array([E[cp] - E[cc] for (cp, cc) in pairs])
        aL = Lam[pairs[0][0]] - Lam[pairs[0][1]]; al = ell[pairs[0][0]] - ell[pairs[0][1]]; kap = aL / al
        v = a * Dl; M = aL * np.diag(Dl) - kap * np.outer(v, v.conj()) / W
        for x in np.linalg.eigvalsh(M): ev += [x, x]
        for (cp, cc) in pairs: done[cp, cc] = done[cc, cp] = True
    for i in range(d):
        for j in range(i + 1, d):
            if done[i, j]: continue
            x = (Lam[i] - Lam[j]) * (E[i] - E[j]); ev += [x, x]
    return np.sort(np.array(ev))

def U_from_E(rho, E):
    lam, V = np.linalg.eigh(rho); rank = {i: k for k, i in enumerate(np.argsort(E))}
    return np.array([V[:, rank[c]] for c in range(len(E))]).T

def selfcons(E, dA, dB, sc=1e-12):
    P = E.reshape(dA, dB); Lam = (np.log(P.sum(1))[:, None] + np.log(P.sum(0))[None, :]).reshape(-1)
    pr = (Lam[:, None] - Lam[None, :]) * (E[:, None] - E[None, :]); off = ~np.eye(len(E), dtype=bool)
    return float(np.min(pr[off]))

def tableau_key(E, dA, dB, transpose_sym):
    P = E.reshape(dA, dB); P = P[np.argsort(P.sum(1))][:, np.argsort(P.sum(0))]
    T = np.argsort(np.argsort(P.reshape(-1))).reshape(dA, dB)
    k = tuple(T.reshape(-1))
    if transpose_sym and dA == dB: k = min(k, tuple(T.T.reshape(-1)))
    return k

def local(dA, dB, seed, nstarts):
    t0 = time.time(); rng = np.random.default_rng(seed); d = dA * dB; G = g_product([dA, dB]); hb = herm_basis(d)
    u = rng.uniform(size=d); p = 0.6 * rng.dirichlet(np.ones(d)) + 0.4 * u / u.sum()   # mild spectrum (GD conditioning)
    Q = haar(d, rng); rho = Q @ np.diag(p) @ Q.conj().T; lam = np.sort(p)
    kern = d + dA * (dA - 1) + dB * (dB - 1); res = dict(dA=dA, dB=dB, seed=seed, spec=lam.tolist(), kern_pred=kern)
    # enumeration of T-fixed assignments: self-consistent (order-iso to r(x)c) ones = predicted local minima
    pred = {}; n_sc = 0; nonsc = []
    for perm in itertools.permutations(range(d)):
        E = np.zeros(d); E[list(perm)] = lam; s = selfcons(E, dA, dB)
        if s > 1e-12:
            n_sc += 1; k = tableau_key(E, dA, dB, False); pred.setdefault(k, E.copy())
        elif len(nonsc) < 200000: nonsc.append(perm)
    W = np.prod([np.math.factorial(x) for x in (dA, dB)]) if hasattr(np, 'math') else None
    import math; W = math.factorial(dA) * math.factorial(dB)
    res.update(n_selfcons=n_sc, orbits_pred=n_sc / W, classes_pred=len(pred))
    # Hessians at T-fixed points: autograd vs closed form; PSD+kernel iff self-consistent
    hk = dict(oi_ok=0, oi_bad=0, non_neg=0, non_bad=0, block_err=0.0)
    sample = list(pred.values()) + [None] * min(60, len(nonsc))
    for i, E in enumerate(sample):
        if E is None:
            perm = nonsc[rng.integers(len(nonsc))]; E = np.zeros(d); E[list(perm)] = lam
        U = U_from_E(rho, E); Hs = hess_autograd_mi(U, rho, dA, dB, hb); ev = np.linalg.eigvalsh((Hs + Hs.T) / 2)
        hk['block_err'] = max(hk['block_err'], float(np.abs(ev - block_spectrum_mi(G, E, dA, dB)).max()))
        tol = 1e-9; sc = selfcons(E, dA, dB) > 1e-12
        if sc: hk['oi_ok' if (ev[0] > -tol and np.sum(np.abs(ev) < tol) == kern) else 'oi_bad'] += 1
        else: hk['non_neg' if ev[0] < -tol else 'non_bad'] += 1
    res['tfixed_hessian'] = hk
    # descent from Haar starts
    Uf, ff, gn, its = descend(rho, dA, dB, [haar(d, rng) for _ in range(nstarts)], iters=6000)
    st = dict(n=nstarts, conv=0, minima=0, product_diag=0, selfcons=0, kern_ok=0, in_pred=0, viol=[]); found = {}
    for i in range(nstarts):
        U, g2 = newton_polish(Uf[i], rho, dA, dB, hb)
        if g2 > 1e-10: continue
        st['conv'] += 1; x = U.conj().T @ rho @ U
        Hs = hess_autograd_mi(U, rho, dA, dB, hb); ev = np.linalg.eigvalsh((Hs + Hs.T) / 2)
        if ev[0] < -1e-6: continue
        st['minima'] += 1
        rA, rB = marg_np(x, dA, dB); _, va = np.linalg.eigh(rA); _, vb = np.linalg.eigh(rB); T = np.kron(va, vb)
        y = T.conj().T @ x @ T; off = np.linalg.norm(y - np.diag(np.diag(y))) / np.linalg.norm(y)
        pd = off < 1e-4; st['product_diag'] += int(pd); st['max_off'] = max(st.get('max_off', 0.0), float(off))
        if pd:
            E = np.real(np.diag(y)); sc = selfcons(E, dA, dB) > 1e-10; st['selfcons'] += int(sc)
            st['kern_ok'] += int(np.sum(np.abs(ev) < 1e-6) == kern); pos = ev[np.abs(ev) >= 1e-6]; st['min_pos_eig'] = min(st.get('min_pos_eig', 9.0), float(pos.min()) if len(pos) else 9.0)
            k = tableau_key(E, dA, dB, False); st['in_pred'] += int(k in pred); found[k] = found.get(k, 0) + 1
            if not sc and len(st['viol']) < 3: st['viol'].append(dict(off=off, sc=selfcons(E, dA, dB)))
        elif len(st['viol']) < 3: st['viol'].append(dict(off=float(off), hmin=float(ev[0])))
    st['classes_found'] = len(found); st['its'] = its; res['descent'] = st
    res['MI_pred'] = sorted(MI_np(np.diag(E), dA, dB) for E in pred.values())
    res['MI_found'] = sorted(MI_np(np.diag(pred[k]), dA, dB) for k in found)
    res['seconds'] = time.time() - t0
    print(json.dumps({k: v for k, v in res.items() if k not in ('spec',)}), flush=True)
    json.dump(res, open(os.path.join(OUT, f'mi_local_{dA}x{dB}_{seed}.json'), 'w'), indent=1, default=float)

def cubicles(dA, dB, rng, nsamp=400000):
    """Orders of a_i + b_j with a, b ascending (Klyachko cubicles), by random sampling (a lower bound on the set)."""
    out = set()
    for _ in range(nsamp // 2000):
        s = rng.exponential(size=(2000, 1)) ** rng.uniform(0, 3, size=(2000, 1))
        a = np.sort(rng.normal(size=(2000, dA)) * s, 1); b = np.sort(rng.normal(size=(2000, dB)), 1)
        v = (a[:, :, None] + b[:, None, :]).reshape(2000, -1)
        for row in np.argsort(v, 1): out.add(tuple(row))
    return sorted(out)

def census(dA, dB, nspec, seed):
    rng = np.random.default_rng(seed); d = dA * dB; cubs = cubicles(dA, dB, rng)
    res = dict(dA=dA, dB=dB, nspec=nspec, seed=seed, n_cubicles=len(cubs)); hits = {}; nsc_hist = []; bad = 0
    for s in range(nspec):
        lam = np.sort(spectrum(d, rng)); best = None; nsc = 0
        for o in cubs:
            E = np.zeros(d); E[list(o)] = lam
            sc = selfcons(E, dA, dB) > 1e-12; nsc += sc
            P = E.reshape(dA, dB); v = -np.sum(P.sum(1) * np.log(P.sum(1))) - np.sum(P.sum(0) * np.log(P.sum(0)))
            if best is None or v < best[0]: best = (v, E, sc)
        if not best[2]:
            bad += 1
            if len(res.setdefault('bad_examples', [])) < 3: res['bad_examples'].append(dict(min_gap=float(np.min(np.diff(lam))), margin=selfcons(best[1], dA, dB)))
        nsc_hist.append(nsc)
        k = tableau_key(best[1], dA, dB, True); hits[k] = hits.get(k, 0) + 1
    res.update(global_min_classes_mod_transpose=len(hits), argmin_not_selfconsistent=bad,
               cubicles_mod_transpose=len({min(tuple(np.argsort(o)), tuple(np.argsort(o).reshape(dA, dB).T.reshape(-1))) if dA == dB else tuple(np.argsort(o)) for o in cubs}),
               local_minima_per_spectrum=dict(min=int(min(nsc_hist)), max=int(max(nsc_hist)), mean=float(np.mean(nsc_hist))),
               class_counts=sorted(hits.values(), reverse=True))
    print(json.dumps(res), flush=True)
    json.dump(res, open(os.path.join(OUT, f'mi_census_{dA}x{dB}_{seed}.json'), 'w'), indent=1)

if __name__ == '__main__':
    if sys.argv[1] == 'local': local(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    else: census(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
