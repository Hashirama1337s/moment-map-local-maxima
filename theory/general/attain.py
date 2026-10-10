"""Attainment: every realizable dA x dB tableau T is the strict global-minimiser class (C1 and mutual information) on an
open set of spectra.  CPU only, light.
  python attain.py tableaux <dA> <dB>          : SYT enumeration, LP realizability, exact Farkas certificates for non-realizable
  python attain.py radius <dA> <dB> <seed>     : generic witness per realizable T; C1 and MI separation; certified radii
  python attain.py tie                         : non-generic counterexample (2x4, spectrum 0..7): two C1 zero-residual classes
  python attain.py direct <seed>               : 2x3 direct orbit descent inside certified balls (C1 and MI)
"""
import sys, os, json, itertools, numpy as np
from fractions import Fraction
from scipy.optimize import linprog
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

def syt(dA, dB):
    """All standard Young tableaux of rectangular shape dA x dB, as dA x dB arrays of ranks 0..d-1."""
    d = dA * dB; out = []
    def rec(T, k, rowlen):
        if k == d: out.append(T.copy()); return
        for a in range(dA):
            b = rowlen[a]
            if b < dB and (a == 0 or rowlen[a - 1] > b):
                T[a, b] = k; rowlen[a] += 1; rec(T, k + 1, rowlen); rowlen[a] -= 1; T[a, b] = -1
    rec(-np.ones((dA, dB), int), 0, [0] * dA); return out

def cell_vec(dA, dB, a, b):
    v = np.zeros(dA + dB); v[a] = 1; v[dA + b] = 1; return v

def realize(T):
    """LP: alpha_a + beta_b strictly ordered as T, margin 1 between consecutive ranks. Returns witness or Farkas certificate."""
    dA, dB = T.shape; d = dA * dB; pos = {T[a, b]: (a, b) for a in range(dA) for b in range(dB)}
    G = np.array([cell_vec(dA, dB, *pos[k]) - cell_vec(dA, dB, *pos[k + 1]) for k in range(d - 1)])   # G x <= -1
    r = linprog(np.zeros(dA + dB), A_ub=G, b_ub=-np.ones(d - 1), bounds=[(None, None)] * (dA + dB), method='highs')
    if r.status == 0: return 'feasible', r.x
    # Farkas: y >= 0, G^T y = 0, sum y = 1  (then 0 = y^T G x <= -sum y < 0, contradiction)
    f = linprog(np.zeros(d - 1), A_eq=np.vstack([G.T, np.ones((1, d - 1))]), b_eq=np.r_[np.zeros(dA + dB), 1],
                bounds=[(0, None)] * (d - 1), method='highs')
    y = [Fraction(v).limit_denominator(1000) for v in f.x]
    Gq = [[Fraction(int(round(g))) for g in row] for row in G]
    ok = all(v >= 0 for v in y) and sum(y) > 0 and all(sum(y[k] * Gq[k][j] for k in range(d - 1)) == 0 for j in range(dA + dB))
    return 'infeasible', dict(certificate=[str(v) for v in y], exact_ok=ok)

def tableaux(dA, dB):
    Ts = syt(dA, dB); feas, infe = [], []
    for T in Ts:
        s, w = realize(T)
        (feas if s == 'feasible' else infe).append((T, w))
    res = dict(dA=dA, dB=dB, n_syt=len(Ts), n_realizable=len(feas), n_nonrealizable=len(infe),
               all_certificates_exact=all(w['exact_ok'] for _, w in infe),
               nonrealizable=[T.tolist() for T, _ in infe])
    if dA == dB: res['self_transpose_realizable'] = sum(int(np.array_equal(T, T.T)) for T, _ in feas)
    print(json.dumps(res)); return res, feas

def reps(dA, dB):
    """Placements modulo row x column permutations: rank array R (cells -> rank of value) with rank 0 at (0,0),
    row 0 and column 0 increasing.  Number = d!/(dA! dB!)."""
    d = dA * dB; out = []
    it = itertools.permutations(range(d))
    while True:
        blk = np.array(list(itertools.islice(it, 400000)))
        if len(blk) == 0: break
        R = blk.reshape(-1, dA, dB)
        ok = (R[:, 0, 0] == 0) & np.all(np.diff(R[:, 0, :], axis=1) > 0, axis=1) & np.all(np.diff(R[:, :, 0], axis=1) > 0, axis=1)
        out.append(blk[ok])
    return np.concatenate(out)

def additive_proj(dA, dB):
    A = np.array([cell_vec(dA, dB, a, b) for a in range(dA) for b in range(dB)])   # d x (dA+dB)
    Q, s, _ = np.linalg.svd(A, full_matrices=False); Q = Q[:, s > 1e-10]; return Q @ Q.T

def H(x):
    x = x[x > 0]; return float(-np.sum(x * np.log(x)))

def h2(t): return 0.0 if t <= 0 or t >= 1 else float(-t * np.log(t) - (1 - t) * np.log(1 - t))

def radius(dA, dB, seed):
    rng = np.random.default_rng(seed); d = dA * dB; _, feas = tableaux(dA, dB)
    Rp = reps(dA, dB); Pd = additive_proj(dA, dB); square = dA == dB
    tr = None
    if square:   # index of transpose of each rep (canonicalised)
        key = {tuple(r): i for i, r in enumerate(Rp)}
        def canon(M):
            M = M.copy(); a0, b0 = np.argwhere(M == 0)[0]
            M = M[np.r_[a0, [i for i in range(dA) if i != a0]]][:, np.r_[b0, [j for j in range(dB) if j != b0]]]
            M = M[np.r_[0, 1 + np.argsort(M[1:, 0])]][:, np.r_[0, 1 + np.argsort(M[0, 1:])]]; return tuple(M.reshape(-1))
        tr = np.array([key[canon(r.reshape(dA, dB).T)] for r in Rp])
    out = []
    for T, w in feas:
        idxT = int(np.where(np.all(Rp == T.reshape(-1), axis=1))[0][0])
        excl = {idxT} | ({int(tr[idxT])} if square else set())
        # ---- C1: generic witness (alpha, beta) in the open realizing cone; lambda* = sorted sums
        x = w + rng.uniform(-0.2, 0.2, size=dA + dB)        # margin 1 => order preserved
        vals = np.array([x[a] + x[dA + b] for a in range(dA) for b in range(dB)])
        assert np.all(np.argsort(np.argsort(vals)) == T.reshape(-1))
        lam = np.sort(vals); E = lam[Rp]                       # rows = placements, E[cell]
        dist = np.linalg.norm(E - E @ Pd.T, axis=1)
        zero = np.where(dist < 1e-9 * np.linalg.norm(lam))[0]
        others = np.array([i for i in range(len(Rp)) if i not in excl])
        delta = float(dist[others].min()); gap = float(np.min(np.diff(lam)))
        r_c1 = min(delta / 2, gap / 2)
        # ---- MI: product spectrum p_ab = r_a c_b with log r + log c realising T (same generic witness)
        u = x[:dA]; v = x[dA:]; s = 1.5 / max(1.0, np.ptp(vals))            # rescale logs for moderate spectra
        P = np.exp(s * (u[:, None] + v[None, :])); P /= P.sum(); p = np.sort(P.reshape(-1))
        assert np.all(np.argsort(np.argsort(P.reshape(-1))) == T.reshape(-1))
        Ep = p[Rp].reshape(-1, dA, dB)
        I = np.array([H(e.sum(1)) + H(e.sum(0)) - H(p) for e in Ep])
        zero_mi = np.where(I < 1e-12)[0]; dmi = float(I[others].min()); gp = float(np.min(np.diff(p)))
        # certified L1-radius tau via Fannes-Audenaert: gap of I between classes shrinks by at most omega(tau)
        def omega(t): return 2 * (t * np.log(max(dA - 1, 1)) + h2(t)) + 2 * (t * np.log(max(dB - 1, 1)) + h2(t))
        lo, hi = 0.0, 0.5
        for _ in range(80):
            mid = (lo + hi) / 2; lo, hi = (mid, hi) if omega(mid) < dmi else (lo, mid)
        tau = min(lo, gp / 4, float(p.min()) / 2)
        out.append(dict(lam_star=lam.tolist(), p_star=p.tolist(), T=T.tolist(), C1_zero_set=sorted(map(int, zero)), C1_expected=sorted(excl), C1_delta=delta, C1_gap=gap,
                        C1_radius_L2=r_c1, MI_zero_set=sorted(map(int, zero_mi)), MI_delta=dmi, MI_radius_L1=2 * tau))
    summ = dict(dA=dA, dB=dB, seed=seed, n_T=len(out), n_reps=len(Rp),
                C1_zero_set_is_class=all(o['C1_zero_set'] == o['C1_expected'] for o in out),
                MI_zero_set_is_class=all(o['MI_zero_set'] == o['C1_expected'] for o in out),
                C1_min_radius=min(o['C1_radius_L2'] for o in out), MI_min_radius_L1=min(o['MI_radius_L1'] for o in out),
                C1_min_delta=min(o['C1_delta'] for o in out), MI_min_delta=min(o['MI_delta'] for o in out))
    print(json.dumps(summ))
    json.dump(dict(summary=summ, per_T=out), open(os.path.join(OUT, f'attain_{dA}x{dB}_{seed}.json'), 'w'), indent=1)
    return summ, out

def tie():
    dA, dB = 2, 4; Rp = reps(dA, dB); Pd = additive_proj(dA, dB); lam = np.arange(8.0)
    E = lam[Rp]; dist = np.linalg.norm(E - E @ Pd.T, axis=1); z = np.where(dist < 1e-9)[0]
    res = dict(spectrum=lam.tolist(), zero_residual_classes=[Rp[i].reshape(dA, dB).tolist() for i in z])
    print(json.dumps(res)); json.dump(res, open(os.path.join(OUT, 'attain_tie_2x4.json'), 'w'), indent=1)

def direct(seed):
    """2x3: perturb the witness spectrum inside the certified ball and minimise over the unitary orbit directly (CPU)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch
    from gen_core import g_product, descend_batch, haar, lm_critical, herm_basis, cost
    import mi as M
    rng = np.random.default_rng(seed); dA, dB = 2, 3; d = 6; G = g_product([dA, dB]); hb = herm_basis(d)
    summ, per = radius(dA, dB, seed); Rp = reps(dA, dB); Pd = additive_proj(dA, dB); res = []
    for o in per:
        T = np.array(o['T']); iT = int(np.where(np.all(Rp == T.reshape(-1), axis=1))[0][0])
        # C1: lambda = lambda* + perturbation of L2 norm 0.9 * certified radius
        lam0 = np.array(o['lam_star']); xi = rng.normal(size=d); xi /= np.linalg.norm(xi); lam = lam0 + 0.9 * o['C1_radius_L2'] * xi
        Q = haar(d, rng); Hm = Q @ np.diag(lam) @ Q.conj().T
        Uf, ff, gn, _ = descend_batch(G, Hm, [haar(d, rng) for _ in range(32)], iters=2500, device='cpu')
        fbest = float(np.min(ff)); E = np.sort(lam)[Rp]; fcls = np.sum((E - E @ Pd.T) ** 2, axis=1); srt = np.sort(fcls)
        r = dict(T=o['T'], C1_orbit_min=fbest, C1_class_T=float(fcls[iT]), C1_second=float(srt[1]),
                 C1_match=bool(abs(fbest - fcls[iT]) < 1e-7 and fcls[iT] < srt[1]))
        # MI: p = p* + perturbation (sum 0) of L1 norm 0.9 * certified radius
        p0 = np.array(o['p_star']); xi = rng.normal(size=d); xi -= xi.mean(); xi /= np.abs(xi).sum(); p = p0 + 0.9 * o['MI_radius_L1'] * xi
        Q = haar(d, rng); rho = Q @ np.diag(p) @ Q.conj().T
        Uf, ff, gn, _ = M.descend(rho, dA, dB, [haar(d, rng) for _ in range(32)], iters=3000, dev='cpu')
        Ibest = float(np.min(ff)) - H(p); Ep = np.sort(p)[Rp].reshape(-1, dA, dB)
        Icls = np.array([H(e.sum(1)) + H(e.sum(0)) - H(p) for e in Ep]); srt = np.sort(Icls)
        r.update(MI_orbit_min=Ibest, MI_class_T=float(Icls[iT]), MI_second=float(srt[1]),
                 MI_match=bool(abs(Ibest - Icls[iT]) < 1e-8 and Icls[iT] < srt[1]))
        res.append(r)
    out = dict(seed=seed, n=len(res), all_match_C1=all(r['C1_match'] for r in res), all_match_MI=all(r['MI_match'] for r in res), cases=res)
    print(json.dumps(out)); json.dump(out, open(os.path.join(OUT, f'attain_direct_2x3_{seed}.json'), 'w'), indent=1)

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True); cmd = sys.argv[1]
    if cmd == 'tableaux':
        r, _ = tableaux(int(sys.argv[2]), int(sys.argv[3])); json.dump(r, open(os.path.join(OUT, f'attain_syt_{sys.argv[2]}x{sys.argv[3]}.json'), 'w'), indent=1)
    elif cmd == 'radius': radius(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == 'tie': tie()
    elif cmd == 'direct': direct(int(sys.argv[2]))
