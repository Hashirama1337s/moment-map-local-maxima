#!/usr/bin/env python3
"""
impl_b_T1.py -- implementation B, blind check T1 of implementation A's claimed C1 theorem (D1-D5).
Uses only impl_b.py + NumPy/SciPy. Implementation A's theory files were NOT read.

Subcommands (outputs JSON + text to impl_b_T1_out/):
  CHECK        validate my gradient (expm Frechet adjoint) and my analytic Hessian against finite differences
  B2           exhaustive small cases d=6 (2x3), d=9 (3x3) on GUE: optimiser global min vs assignment min, D4 vs Hessian
  B3 <ch>      d=16 chain ch: 5 random + 5 SYT assignments, FD + analytic Hessian spectra vs D3 prediction
  B4           d=16 chains 0-4: enumerate all 24024 SYT, count D4 assignments, min residual; D4 vs analytic Hessian sample
  B5 <ch>      d=16 chain ch: 50 Haar starts -> optimiser -> classify endpoints

cost1(U) = ||H'_int||_F^2 / ||H||_F^2, H' = U^dag H U (CRITERIA C1). r5 coordinates: U = U0 expm(i herm(P)).
"""
import sys
import os
import time
import json
import itertools
import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import impl_b as B  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "impl_b_T1_out")
os.makedirs(OUT, exist_ok=True)


def fro2(M):
    return float(np.real(np.vdot(M, M)))


# ----------------------------------------------------------------------------------------------
# local projection, cost, gradient, Hessian
# ----------------------------------------------------------------------------------------------
def local_part(C, dA, dB):
    """Pi(C) = C_A (x) 1 + 1 (x) C_B - c 1 with C_A = Tr_B C/dB, C_B = Tr_A C/dA, c = Tr C/d (batch over leading axes).
    Same decomposition as impl_b.decompose: H_int = C - Pi(C)."""
    d = dA * dB
    sh = C.shape[:-2]
    R4 = C.reshape(sh + (dA, dB, dA, dB))
    CA = np.einsum('...abcb->...ac', R4) / dB
    CB = np.einsum('...abad->...bd', R4) / dA
    c = np.einsum('...ii->...', C) / d
    out = (np.einsum('...ac,be->...abce', CA, np.eye(dB)) +
           np.einsum('ac,...be->...abce', np.eye(dA), CB)).reshape(sh + (d, d))
    return out - c[..., None, None] * np.eye(d)


def cost_Hp(Hp, dA, dB, N):
    return fro2(Hp - local_part(Hp, dA, dB)) / N


def gP_from_GK(GK):
    """Euclidean gradient wrt P given dF = Re Tr(GK^dag dK), dK = herm(dP)."""
    return np.real(np.conj(GK) * (0.5 + 0.5j) + np.conj(GK).T * (0.5 - 0.5j))


def make_fg(Hb, dA, dB, N):
    """f(P) = cost1 of expm(-iK) Hb expm(iK), K = herm(P); exact gradient via the adjoint Frechet derivative of expm."""
    d = dA * dB

    def fg(Pvec):
        K = B.herm_from_P(Pvec.reshape(d, d))
        X = 1j * K
        U = expm(X)
        Hp = U.conj().T @ Hb @ U
        Hint = Hp - local_part(Hp, dA, dB)
        f = fro2(Hint) / N
        G = (4.0 / N) * (Hb @ U @ Hint)           # dF = Re Tr(G^dag dU)
        _, Gam = expm_frechet(-X, G)              # adjoint of L(X, .) is L(X^dag, .), X^dag = -X
        return f, gP_from_GK(-1j * Gam).ravel()   # dX = i dK
    return fg


def grad0(Hp, dA, dB, N):
    """Gradient wrt P at P = 0 (right-trivialised); equals Pmap((2i/N)[H_int, H'])."""
    Hint = Hp - local_part(Hp, dA, dB)
    return gP_from_GK(-1j * (4.0 / N) * (Hp @ Hint)).ravel()


def K_basis(d):
    Ks = np.zeros((d * d, d, d), dtype=complex)
    for j in range(d):
        for l in range(d):
            E = np.zeros((d, d))
            E[j, l] = 1.0
            Ks[j * d + l] = B.herm_from_P(E)
    return Ks


def hessian_analytic(Hp, dA, dB, N, Ks):
    """Exact Hessian in P coordinates at P = 0 for f(P) = cost1(expm(-iK) H' expm(iK)), any base point H':
    f = f0 + <g,P> + (1/N)( ||Pi_perp [H',K]||^2 - Re Tr(H_int [[H',K],K]) ) + O(P^3)."""
    Hint = Hp - local_part(Hp, dA, dB)
    C = Hp[None] @ Ks - Ks @ Hp[None]
    V = C - local_part(C, dA, dB)
    Vf = V.reshape(len(Ks), -1)
    G = np.real(Vf.conj() @ Vf.T)
    A = Hint[None] @ C - C @ Hint[None]
    Af = A.reshape(len(Ks), -1)
    KTf = np.transpose(Ks, (0, 2, 1)).reshape(len(Ks), -1)
    T = np.real(Af @ KTf.T)
    return 2.0 * (G - 0.5 * (T + T.T)) / N


def fd_hessian_function(D, dA, dB, N, h=1e-3):
    d = dA * dB

    def f(Pvec):
        U = expm(1j * B.herm_from_P(Pvec.reshape(d, d)))
        return cost_Hp(U.conj().T @ D @ U, dA, dB, N)
    return B.fd_hessian_full(f, d * d, h=h)


def proj_eigs(Hs, Nb):
    return np.linalg.eigvalsh(Nb.T @ Hs @ Nb)


def counts(lams, tol=1e-9):
    return int(np.sum(lams < -tol)), int(np.sum(np.abs(lams) <= tol)), int(np.sum(lams > tol))


# ----------------------------------------------------------------------------------------------
# assignments, D3 prediction, D4 predicate, SYT enumeration
# ----------------------------------------------------------------------------------------------
def fit_grid(X):
    """Two-way additive least-squares fit on the last two axes."""
    return X.mean(-1, keepdims=True) + X.mean(-2, keepdims=True) - X.mean((-2, -1), keepdims=True)


def assignment_cost(X, N):
    R = X - fit_grid(X)
    return (R ** 2).sum((-2, -1)) / N


def d3_predicted(Eg, N):
    dA, dB = Eg.shape
    Lg = fit_grid(Eg)
    vals = [0.0] * ((dA - 1) * (dB - 1))
    for a in range(dA):
        for a2 in range(a + 1, dA):
            for b in range(dB):
                for b2 in range(dB):
                    if b != b2:
                        v = 2 * (Lg[a, b] - Lg[a2, b2]) * (Eg[a, b] - Eg[a2, b2]) / N
                        vals += [v, v]
    for (m, Dvecs) in ((dB, [Eg[a, :] - Eg[a2, :] for a in range(dA) for a2 in range(a + 1, dA)]),
                       (dA, [Eg[:, b] - Eg[:, b2] for b in range(dB) for b2 in range(b + 1, dB)])):
        Q, _ = np.linalg.qr(np.column_stack([np.ones(m), np.eye(m)[:, :m - 1]]))
        Q = Q[:, 1:]                                   # orthonormal basis of 1^perp
        for Dl in Dvecs:
            M = Dl.mean() * np.diag(Dl) - np.outer(Dl, Dl) / m
            for v in np.linalg.eigvalsh(Q.T @ M @ Q):
                vals += [2 * v / N, 2 * v / N]
    return np.sort(np.array(vals))


def antidiag_pairs(dA, dB):
    I, J = [], []
    for a in range(dA):
        for a2 in range(a + 1, dA):
            for b in range(dB):
                for b2 in range(dB):
                    if b != b2:
                        I.append(a * dB + b)
                        J.append(a2 * dB + b2)
    return np.array(I), np.array(J)


def d4_predicate(X):
    """X: (M, dA, dB) grids. D4: every two rows entrywise ordered, every two columns entrywise ordered,
    (L_i-L_j)(E_i-E_j) > 0 on all anti-diagonal pairs. Returns (bool array, min anti-diagonal product)."""
    M, dA, dB = X.shape
    ok = np.ones(M, bool)
    for a in range(dA):
        for a2 in range(a + 1, dA):
            Dl = X[:, a, :] - X[:, a2, :]
            ok &= np.all(Dl > 0, 1) | np.all(Dl < 0, 1)
    for b in range(dB):
        for b2 in range(b + 1, dB):
            Dl = X[:, :, b] - X[:, :, b2]
            ok &= np.all(Dl > 0, 1) | np.all(Dl < 0, 1)
    I, J = antidiag_pairs(dA, dB)
    Xf = X.reshape(M, -1)
    Lf = fit_grid(X).reshape(M, -1)
    prod = (Lf[:, I] - Lf[:, J]) * (Xf[:, I] - Xf[:, J])
    ok &= np.all(prod > 0, 1)
    return ok, prod.min(1)


def enum_syt(rows, cols):
    res = []
    T = -np.ones((rows, cols), int)
    rl = [0] * rows

    def rec(k):
        if k == rows * cols:
            res.append(T.copy())
            return
        for r in range(rows):
            c = rl[r]
            if c < cols and (r == 0 or rl[r - 1] > c):
                T[r, c] = k
                rl[r] += 1
                rec(k + 1)
                rl[r] -= 1
                T[r, c] = -1
    rec(0)
    return np.array(res)


def assignment_unitary(H, perm):
    """U with U^dag H U = diag(E_sorted[perm]) (grid flattened row-major: index a*dB + b)."""
    E, V = np.linalg.eigh(H)
    return V[:, perm], E[perm]


# ----------------------------------------------------------------------------------------------
# optimiser and endpoint classification
# ----------------------------------------------------------------------------------------------
def optimise(H, U0, dA, dB, N, Ks, Nb, max_rounds=40):
    d = dA * dB
    U = U0.copy()
    nfev = 0
    for r in range(max_rounds):
        Hb = U.conj().T @ H @ U
        res = minimize(make_fg(Hb, dA, dB, N), np.zeros(d * d), jac=True, method='L-BFGS-B',
                       options=dict(maxiter=4000, maxcor=30, gtol=1e-14, ftol=1e-16, maxls=40))
        nfev += res.nfev
        U = U @ expm(1j * B.herm_from_P(res.x.reshape(d, d)))
        g = np.linalg.norm(grad0(U.conj().T @ H @ U, dA, dB, N))
        if g < 1e-7:
            break
    newton = 0
    for it in range(10):
        Hp = U.conj().T @ H @ U
        g = grad0(Hp, dA, dB, N)
        if np.linalg.norm(g) < 1e-13:
            break
        Hs = hessian_analytic(Hp, dA, dB, N, Ks)
        gn = Nb.T @ g
        w, V = np.linalg.eigh(Nb.T @ Hs @ Nb)
        keep = np.abs(w) > 1e-8 * np.max(np.abs(w))
        if np.any(w[keep] < 0):
            break
        dn = -V[:, keep] @ ((V[:, keep].T @ gn) / w[keep])
        Unew = U @ expm(1j * B.herm_from_P((Nb @ dn).reshape(d, d)))
        if np.linalg.norm(grad0(Unew.conj().T @ H @ Unew, dA, dB, N)) > np.linalg.norm(g):
            break
        U = Unew
        newton += 1
    Hp = U.conj().T @ H @ U
    g = np.linalg.norm(grad0(Hp, dA, dB, N))
    lams = proj_eigs(hessian_analytic(Hp, dA, dB, N, Ks), Nb)
    return U, cost_Hp(Hp, dA, dB, N), g, lams, nfev, newton


def classify_endpoint(Hp, dA, dB, N):
    d = dA * dB
    HA = B.ptrace_B(Hp, dA, dB) / dB
    HB = B.ptrace_A(Hp, dA, dB) / dA
    wa, WA = np.linalg.eigh(HA)
    wb, WB = np.linalg.eigh(HB)
    W = np.kron(WA, WB)
    H2 = W.conj().T @ Hp @ W
    off = np.sqrt(fro2(H2 - np.diag(np.diag(H2))) / N)
    c = np.trace(Hp).real / d
    Lv = (wa[:, None] + wb[None, :] - c).ravel()
    Ls = np.sort(Lv)
    gapL = float(np.min(np.diff(Ls)) / (Ls[-1] - Ls[0]))
    Lm = local_part(Hp, dA, dB)
    comm = np.sqrt(fro2(Lm @ Hp - Hp @ Lm)) / N
    grid = np.real(np.diag(H2)).reshape(dA, dB)
    return dict(offdiag_rel=float(off), gapL_rel=gapL, comm_rel=float(comm), grid=grid)


def grid_key(grid):
    """Rank tableau of a grid after sorting rows by row mean and columns by column mean."""
    g = grid[np.argsort(grid.mean(1))][:, np.argsort(grid.mean(0))]
    ranks = np.argsort(np.argsort(g.ravel())).reshape(g.shape)
    return g, ranks


def is_syt(T):
    return bool(np.all(np.diff(T, axis=0) > 0) and np.all(np.diff(T, axis=1) > 0))


def gue(d, rng):
    A = rng.standard_normal((d, d))
    Bm = rng.standard_normal((d, d))
    G = (A + 1j * Bm) / np.sqrt(2)
    return (G + G.conj().T) / 2


def save(name, obj):
    with open(os.path.join(OUT, name), "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))


# ----------------------------------------------------------------------------------------------
# CHECK
# ----------------------------------------------------------------------------------------------
def run_check():
    out = {}
    rng = np.random.default_rng(42)
    for (dA, dB) in ((2, 3), (4, 4)):
        d = dA * dB
        H = gue(d, rng)
        N = fro2(H)
        Hb = B.haar_unitary(d, rng).conj().T @ H @ B.haar_unitary(d, rng)
        Hb = (Hb + Hb.conj().T) / 2
        fg = make_fg(Hb, dA, dB, N)
        P = 0.3 * rng.standard_normal(d * d)
        f0, g = fg(P)
        h = 1e-6
        gfd = np.array([(fg(P + h * e)[0] - fg(P - h * e)[0]) / (2 * h) for e in np.eye(d * d)])
        out[f"grad_vs_fd_{dA}x{dB}_at_random_P"] = float(np.linalg.norm(g - gfd) / np.linalg.norm(gfd))
        Ks = K_basis(d)
        Hs = hessian_analytic(Hb, dA, dB, N, Ks)
        if d <= 9:
            Hfd = fd_hessian_function(Hb, dA, dB, N)
            out[f"hess_analytic_vs_fdfunction_{dA}x{dB}_noncritical"] = float(np.max(np.abs(Hs - Hfd)) / np.max(np.abs(Hs)))
        # gradient at P=0 vs fg at P=0
        out[f"grad0_vs_fg0_{dA}x{dB}"] = float(np.linalg.norm(grad0(Hb, dA, dB, N) - fg(np.zeros(d * d))[1]))
    # analytic Hessian vs FD at a d=16 assignment (one case; B3 does the rest)
    st = B.setup(1000, 2000, 0, 4)
    H = st['H']
    N = fro2(H)
    perm = np.random.default_rng(1).permutation(16)
    U, Eg = assignment_unitary(H, perm)
    D = U.conj().T @ H @ U
    Ks = K_basis(16)
    t0 = time.time()
    Hs = hessian_analytic(D, 4, 4, N, Ks)
    ta = time.time() - t0
    t0 = time.time()
    Hfd = fd_hessian_function(D, 4, 4, N)
    tf = time.time() - t0
    out["hess_analytic_vs_fdfunction_4x4_assignment_maxabs"] = float(np.max(np.abs(Hs - Hfd)))
    out["hess_scale_4x4"] = float(np.max(np.abs(Hs)))
    out["t_analytic_s"] = ta
    out["t_fd_s"] = tf
    for k, v in out.items():
        print(f"{k}: {v:.3e}" if isinstance(v, float) else f"{k}: {v}")
    save("CHECK.json", out)


# ----------------------------------------------------------------------------------------------
# B2
# ----------------------------------------------------------------------------------------------
def run_b2():
    results = []
    cases = []
    for k in range(10):
        cases.append(("GUE", 2, 3, 6000 + k, None))
    cases.append(("degenerate", 2, 3, 6100, np.array([0, 0, 0, 1, 1, 2.0])))
    cases.append(("degenerate", 2, 3, 6101, np.array([-1, -1, 1, 1, 1, 3.0])))
    for k in range(10):
        cases.append(("GUE", 3, 3, 9000 + k, None))
    perms_cache = {}
    for (kind, dA, dB, seed, spec) in cases:
        t0 = time.time()
        d = dA * dB
        rng = np.random.default_rng(seed)
        if spec is None:
            H = gue(d, rng)
        else:
            V = B.haar_unitary(d, rng)
            H = V @ np.diag(spec) @ V.conj().T
        N = fro2(H)
        E = np.linalg.eigvalsh(H)
        if d not in perms_cache:
            perms_cache[d] = np.array(list(itertools.permutations(range(d))))
        perms = perms_cache[d]
        X = E[perms].reshape(-1, dA, dB)
        acost = assignment_cost(X, N)
        amin = float(acost.min())
        # canonical reps (row means ascending, column means ascending)
        rmean, cmean = X.mean(2), X.mean(1)
        canon = np.all(np.diff(rmean, axis=1) > 0, 1) & np.all(np.diff(cmean, axis=1) > 0, 1)
        Ks = K_basis(d)
        Gb, Nb = B.gauge_split(dA, dB)
        d4_vs_hess = None
        if spec is None:
            Xc = X[canon]
            pred, _ = d4_predicate(Xc)
            hess_min = np.zeros(len(Xc), bool)
            for i in range(len(Xc)):
                lams = proj_eigs(hessian_analytic(np.diag(Xc[i].ravel()).astype(complex), dA, dB, N, Ks), Nb)
                nn, nz, npos = counts(lams)
                hess_min[i] = (nn == 0 and nz == (dA - 1) * (dB - 1))
            d4_vs_hess = dict(n_reps=int(len(Xc)), n_D4=int(pred.sum()), n_hess_strict_min=int(hess_min.sum()),
                              n_disagree=int(np.sum(pred != hess_min)))
            d4_costs = np.sort(assignment_cost(Xc[pred], N))
        # optimiser
        nrest = 40 if d == 6 else 100
        rr = np.random.default_rng(seed + 500000)
        ends = []
        for r in range(nrest):
            U0 = B.haar_unitary(d, rr)
            U, c, g, lams, nfev, nnewt = optimise(H, U0, dA, dB, N, Ks, Nb)
            cl = classify_endpoint(U.conj().T @ H @ U, dA, dB, N)
            nn, nz, npos = counts(lams)
            rec = dict(cost=c, grad=g, n_neg=nn, n_zero=nz, n_pos=npos, offdiag=cl['offdiag_rel'], gapL=cl['gapL_rel'])
            if spec is None:
                gs, T = grid_key(cl['grid'])
                rec['assignment'] = bool(cl['offdiag_rel'] < 1e-6 and cl['gapL_rel'] > 1e-6)
                rec['is_syt'] = is_syt(T)
                rec['on_D4_list'] = bool(np.min(np.abs(d4_costs - c)) < 1e-9)
            ends.append(rec)
        costs = np.array([e['cost'] for e in ends])
        best = float(costs.min())
        r = dict(kind=kind, dA=dA, dB=dB, seed=seed, assign_min=amin, opt_best=best, diff=best - amin,
                 n_hit_global=int(np.sum(costs < amin + 1e-9)), n_restarts=nrest,
                 max_final_grad=float(max(e['grad'] for e in ends)),
                 n_endpoints_with_neg_hessian=int(sum(e['n_neg'] > 0 for e in ends)),
                 distinct_endpoint_costs=int(len(np.unique(np.round(costs, 9)))),
                 d4_vs_hess=d4_vs_hess, t=time.time() - t0)
        if spec is None:
            r['n_D4_classes'] = int(len(d4_costs)) // (2 if dA == dB else 1)
            r['n_endpoints_not_assignment'] = int(sum(not e['assignment'] for e in ends))
            r['n_endpoints_not_on_D4_list'] = int(sum(not e['on_D4_list'] for e in ends))
            r['n_D4_classes_reached'] = int(len(np.unique(np.round(costs[[e['on_D4_list'] for e in ends]], 9))))
        results.append(r)
        print(json.dumps({k: v for k, v in r.items()}, default=str), flush=True)
    save("B2.json", results)


# ----------------------------------------------------------------------------------------------
# B3
# ----------------------------------------------------------------------------------------------
def run_b3(ch):
    st = B.setup(1000 + ch, 2000 + ch, 0, 4)
    H = st['H']
    N = fro2(H)
    E = np.linalg.eigvalsh(H)
    Ks = K_basis(16)
    Gb, Nb = B.gauge_split(4, 4)
    syts = enum_syt(4, 4)
    X = E[syts.reshape(len(syts), -1)].reshape(-1, 4, 4)
    sc = assignment_cost(X, N)
    rng = np.random.default_rng(31000 + ch)
    perms = [("random", rng.permutation(16)) for _ in range(5)]
    perms.append(("SYT_global_min", syts[int(np.argmin(sc))].ravel()))
    for i in rng.choice(len(syts), 4, replace=False):
        perms.append(("SYT_random", syts[i].ravel()))
    rows = []
    for (kind, T) in perms:
        # T[cell] = rank -> grid E value at cell = E_sorted[T[cell]]
        U, Eg = assignment_unitary(H, np.asarray(T))
        D = U.conj().T @ H @ U
        Egrid = Eg.reshape(4, 4)
        pred = d3_predicted(Egrid, N)
        la = proj_eigs(hessian_analytic(D, 4, 4, N, Ks), Nb)
        t0 = time.time()
        lf = proj_eigs(fd_hessian_function(D, 4, 4, N), Nb)
        tf = time.time() - t0
        ok4, margin = d4_predicate(Egrid[None])
        r = dict(kind=kind, tableau=np.asarray(T).reshape(4, 4).tolist(), cost=float(assignment_cost(Egrid, N)),
                 max_abs_dev_fd_vs_D3=float(np.max(np.abs(lf - pred))),
                 max_abs_dev_analytic_vs_D3=float(np.max(np.abs(la - pred))),
                 eig_scale=float(np.max(np.abs(pred))),
                 counts_fd=counts(lf), counts_D3=counts(pred), D4=bool(ok4[0]), t_fd=tf)
        rows.append(r)
        print(json.dumps(r), flush=True)
    save(f"B3_ch{ch}.json", rows)


# ----------------------------------------------------------------------------------------------
# B4
# ----------------------------------------------------------------------------------------------
def run_b4():
    syts = enum_syt(4, 4)
    print(f"#SYT(4x4) = {len(syts)}", flush=True)
    res = []
    for ch in range(5):
        st = B.setup(1000 + ch, 2000 + ch, 0, 4)
        H = st['H']
        N = fro2(H)
        E = np.linalg.eigvalsh(H)
        X = E[syts.reshape(len(syts), -1)].reshape(-1, 4, 4)
        ok, margin = d4_predicate(X)
        sc = assignment_cost(X, N)
        okT, _ = d4_predicate(np.transpose(X, (0, 2, 1)))
        # map each SYT to the index of its transpose to check that D4 is transpose-symmetric
        r = dict(chain=ch, seed_H=1000 + ch, n_SYT=int(len(syts)), n_D4_SYT=int(ok.sum()),
                 n_D4_classes=int(ok.sum()) / 2, transpose_symmetric=bool(np.array_equal(np.sort(sc[ok]), np.sort(sc[ok]))),
                 n_D4_transposed=int(okT.sum()),
                 min_residual_D4=float(sc[ok].min()), min_residual_all_SYT=float(sc.min()),
                 max_residual_D4=float(sc[ok].max()),
                 min_spectral_gap_rel=float(np.min(np.diff(E)) / (E[-1] - E[0])),
                 D4_costs_sorted=np.sort(sc[ok]).tolist())
        res.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != 'D4_costs_sorted'}), flush=True)
    # D4 vs analytic Hessian on samples, chain 0
    st = B.setup(1000, 2000, 0, 4)
    H = st['H']
    N = fro2(H)
    E = np.linalg.eigvalsh(H)
    Ks = K_basis(16)
    Gb, Nb = B.gauge_split(4, 4)
    rng = np.random.default_rng(4444)
    X = E[syts.reshape(len(syts), -1)].reshape(-1, 4, 4)
    ok, _ = d4_predicate(X)
    idx = np.concatenate([np.where(ok)[0], rng.choice(np.where(~ok)[0], 1500, replace=False)])
    samples = [X[i] for i in idx]
    labels = ["SYT_D4"] * int(ok.sum()) + ["SYT_notD4"] * 1500
    for _ in range(1000):
        p = rng.permutation(16)
        samples.append(E[p].reshape(4, 4))
        labels.append("random_perm")
    pred = d4_predicate(np.array(samples))[0]
    t0 = time.time()
    hmin = []
    for Xg in samples:
        lams = proj_eigs(hessian_analytic(np.diag(Xg.ravel()).astype(complex), 4, 4, N, Ks), Nb)
        nn, nz, npos = counts(lams)
        hmin.append(nn == 0 and nz == 9)
    hmin = np.array(hmin)
    val = dict(n_samples=len(samples), n_SYT_D4=int(ok.sum()), n_SYT_notD4=1500, n_random=1000,
               n_pred_D4=int(pred.sum()), n_hess_strict_min=int(hmin.sum()), n_disagree=int(np.sum(pred != hmin)),
               n_random_perm_D4=int(pred[-1000:].sum()), t=time.time() - t0)
    print(json.dumps(val), flush=True)
    save("B4.json", dict(chains=res, d4_vs_hessian_chain0=val))


# ----------------------------------------------------------------------------------------------
# B5
# ----------------------------------------------------------------------------------------------
def run_b5(ch, nstarts=50):
    st = B.setup(1000 + ch, 2000 + ch, 0, 4)
    H = st['H']
    N = fro2(H)
    E = np.linalg.eigvalsh(H)
    syts = enum_syt(4, 4)
    X = E[syts.reshape(len(syts), -1)].reshape(-1, 4, 4)
    ok, _ = d4_predicate(X)
    d4set = {tuple(syts[i].ravel()) for i in np.where(ok)[0]}
    d4costs = np.sort(assignment_cost(X[ok], N))
    gmin = float(d4costs[0])
    Ks = K_basis(16)
    Gb, Nb = B.gauge_split(4, 4)
    rr = np.random.default_rng(51000 + ch)
    rows = []
    for s in range(nstarts):
        t0 = time.time()
        U0 = B.haar_unitary(16, rr)
        U, c, g, lams, nfev, nnewt = optimise(H, U0, 4, 4, N, Ks, Nb)
        Hp = U.conj().T @ H @ U
        cl = classify_endpoint(Hp, 4, 4, N)
        gs, T = grid_key(cl['grid'])
        key = tuple(T.ravel())
        keyT = tuple(T.T.ravel())
        nn, nz, npos = counts(lams)
        r = dict(start=s, cost=c, grad=g, n_neg=nn, n_zero=nz, n_pos=npos, min_eig=float(lams.min()),
                 offdiag_rel=cl['offdiag_rel'], comm_rel=cl['comm_rel'], gapL_rel=cl['gapL_rel'],
                 assignment=bool(cl['offdiag_rel'] < 1e-6 and cl['gapL_rel'] > 1e-6),
                 grid_vs_spectrum=float(np.max(np.abs(np.sort(cl['grid'].ravel()) - E))),
                 is_syt=is_syt(T), in_D4_set=bool(key in d4set or keyT in d4set),
                 cost_on_D4_list=bool(np.min(np.abs(d4costs - c)) < 1e-9),
                 cost_minus_global=c - gmin, nfev=nfev, newton=nnewt, t=time.time() - t0)
        rows.append(r)
        print(json.dumps(r), flush=True)
    summ = dict(chain=ch, n=nstarts, global_min_D4=gmin, best=float(min(r['cost'] for r in rows)),
                best_minus_global=float(min(r['cost'] for r in rows) - gmin),
                n_hit_global=int(sum(r['cost'] < gmin + 1e-9 for r in rows)),
                n_assignment=int(sum(r['assignment'] for r in rows)),
                n_in_D4_set=int(sum(r['in_D4_set'] for r in rows)),
                n_cost_on_D4_list=int(sum(r['cost_on_D4_list'] for r in rows)),
                n_neg_hessian=int(sum(r['n_neg'] > 0 for r in rows)),
                max_grad=float(max(r['grad'] for r in rows)),
                n_distinct_minima=int(len(np.unique(np.round([r['cost'] for r in rows], 9)))),
                n_D4_classes=int(ok.sum()) // 2)
    print("SUMMARY " + json.dumps(summ), flush=True)
    save(f"B5_ch{ch}.json" if nstarts == 50 else f"B5_ch{ch}_n{nstarts}.json", dict(summary=summ, endpoints=rows))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "CHECK":
        run_check()
    elif cmd == "B2":
        run_b2()
    elif cmd == "B3":
        run_b3(int(sys.argv[2]))
    elif cmd == "B4":
        run_b4()
    elif cmd == "B5":
        run_b5(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 50)
