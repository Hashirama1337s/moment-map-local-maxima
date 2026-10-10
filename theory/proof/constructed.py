"""Construct critical points with DEGENERATE local parts and test them.
Family A ('affine'): prescribe alpha, beta with ties (and optional extra additive L coincidences), draw a random
  H' block-diagonal on the eigenspaces of L_t = diag(alpha_a + beta_b), project (min-norm) onto P(H') = L_t.
  Then U = 1 is a critical point of f for H := H' (random, non-degenerate spectrum checked).
Family B ('polygon'): a GIVEN GUE Hamiltonian H.  Rows {0,1} tied: H' = sum_b [ (t_b/2 + r_b.sigma) (+) e_b ] (x) |b><b|,
  sum_b r_b = 0 (closed polygon, side lengths = half-gaps of the eigenvalue pairs placed in column b).
  Exact critical point of the given H with alpha_0 = alpha_1 and genuine same-column coherences.
For every point: spectrum gap, ||grad||, gauge-invariant assignment test, local-part gaps, analytic Hessian
min eigenvalue (and autograd cross-check for d <= 9), N1 commutant-form min eigenvalue, explicit descent line test.
"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import *
from scipy.optimize import minimize

OUT = os.path.dirname(os.path.abspath(__file__))
SX = np.array([[0, 1], [1, 0]], complex); SY = np.array([[0, -1j], [1j, 0]]); SZ = np.array([[1, 0], [0, -1]], complex)

def groups_of(Lval, tol=1e-9):
    order = np.argsort(Lval); groups, cur = [], [order[0]]
    for i in order[1:]:
        if abs(Lval[i] - Lval[cur[-1]]) < tol: cur.append(i)
        else: groups.append(cur); cur = [i]
    groups.append(cur); return groups

def family_A(m, n, atie, btie, extra, rng, coh=1.0):
    """atie/btie: lists of index groups to tie; extra: list of ((a,b),(a2,b2)) additive coincidences to force
    (implemented by solving for one beta)."""
    d = m * n
    al = np.sort(rng.normal(size=m)) * 2; be = np.sort(rng.normal(size=n)) * 2
    for g in atie: al[g] = al[g[0]]
    for g in btie: be[g] = be[g[0]]
    for (a, b), (a2, b2) in extra:  # force al[a]+be[b] = al[a2]+be[b2] by moving be[b2]
        be[b2] = al[a] + be[b] - al[a2]
    Lval = (al[:, None] + be[None, :]).ravel()
    Lt = np.diag(Lval).astype(complex)
    groups = groups_of(Lval)
    # basis of block-diagonal Hermitian subspace V
    Vb = []
    s = 1 / np.sqrt(2)
    for g in groups:
        for j in g:
            E = np.zeros((d, d), complex); E[j, j] = 1; Vb.append(E)
        for x in range(len(g)):
            for y in range(x + 1, len(g)):
                j, k = g[x], g[y]
                E = np.zeros((d, d), complex); E[j, k] = E[k, j] = s; Vb.append(E)
                E = np.zeros((d, d), complex); E[j, k] = 1j * s; E[k, j] = -1j * s; Vb.append(E)
    Vb = np.array(Vb)
    x0 = rng.normal(size=len(Vb)) * coh
    H0 = from_coords(x0, Vb) + Lt
    full = herm_basis(d)
    # constraint: coords of P(Delta) = coords of (Lt - P(H0))
    A = np.array([coords(P(E, m, n), full) for E in Vb]).T
    rhs = coords(Lt - P(H0, m, n), full)
    dx, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    Hp = H0 + from_coords(dx, Vb); Hp = (Hp + Hp.conj().T) / 2
    return Hp, al, be, groups

def closed_polygon(lengths, rng, tries=20000):
    """Exact closed polygon in R^3 with the given side lengths: random directions for sides 0..k-3, then the last two
    sides close the triangle on s = -(r_0+...+r_{k-3}) (law of cosines, random plane). Returns (R, 0.0) or (None, inf)."""
    k = len(lengths)
    for _ in range(tries):
        R = np.zeros((k, 3))
        for b in range(k - 2):
            u = rng.normal(size=3); R[b] = lengths[b] * u / np.linalg.norm(u)
        s = -R[:k - 2].sum(0); ns = np.linalg.norm(s)
        la, lb = lengths[k - 2], lengths[k - 1]
        if not (abs(la - lb) + 1e-6 < ns < la + lb - 1e-6): continue
        sh = s / ns; e = rng.normal(size=3); e -= (e @ sh) * sh; e /= np.linalg.norm(e)
        x = (ns ** 2 + la ** 2 - lb ** 2) / (2 * ns); y = np.sqrt(max(la ** 2 - x ** 2, 0.0))
        R[k - 2] = x * sh + y * e; R[k - 1] = s - R[k - 2]
        return R, 0.0
    return None, np.inf

def family_B(lam, m, n, rng, tries=2000):
    """Rows 0,1 tied (alpha_0 = alpha_1); rows 2.. (if m > 2) diagonal. Returns H' with spectrum exactly lam
    (up to the polygon-closure tolerance), or None."""
    d = m * n
    for _ in range(tries):
        perm = rng.permutation(d)
        pairs = perm[:2 * n].reshape(n, 2); rest = perm[2 * n:]
        lo, hi = lam[pairs].min(1), lam[pairs].max(1); half = (hi - lo) / 2
        if half.max() >= half.sum() - half.max() - 1e-3: continue  # strict polygon inequality
        R, res = closed_polygon(half, rng)
        if R is None: continue
        Hp = np.zeros((d, d), complex)
        for b in range(n):
            h = (lo[b] + hi[b]) / 2 * np.eye(2) + R[b, 0] * SX + R[b, 1] * SY + R[b, 2] * SZ
            for i in range(2):
                for j in range(2):
                    Hp[i * n + b, j * n + b] = h[i, j]
        for a in range(2, m):
            for b in range(n):
                Hp[a * n + b, a * n + b] = lam[rest[(a - 2) * n + b]]
        return Hp, dict(pairs=pairs.tolist(), rest=rest.tolist(), polygon=R.tolist())
    return None

def analyse(U, H, m, n, tag, do_autograd, out):
    d = m * n
    Hp, L = parts(U, H, m, n)
    lam = np.linalg.eigvalsh(H); sgap = float(np.min(np.diff(lam)))
    g = 2 * np.linalg.norm(comm(L, Hp))
    asg, s2, ov = is_assignment(Hp, m, n)
    ga, gb = local_gaps(Hp, m, n)
    basis = herm_basis(d)
    Hs = hess_analytic(Hp, L, m, n, basis)
    ev = np.linalg.eigvalsh(Hs)
    agd = None
    if do_autograd:
        Ha = hess_autograd(U, H, m, n, basis)
        agd = float(np.max(np.abs(Ha - Hs)))
    q, K, gsz = commutant_form(Hp, L, m, n)
    nK = np.sqrt(ip(K, K)); K = K / nK
    pred = ip(P(1j * comm(Hp, K), m, n), P(1j * comm(Hp, K), m, n))
    f0 = cost(U, H, m, n)
    line = []
    for s in [1e-1, 1e-2, 1e-3]:
        Us = U @ expm(1j * s * K)
        line.append([s, cost(Us, H, m, n) - f0, -s * s * pred])
    rec = dict(tag=tag, d=d, m=m, n=n, spec_gap=sgap, grad=float(g), f=f0, assignment=bool(asg), schmidt_s2=s2,
               overlap_defect=ov, alpha_gap=ga, beta_gap=gb, L_block_sizes=sorted(gsz, reverse=True)[:4],
               hess_min=float(ev[0]), hess_max=float(ev[-1]), n_neg=int(np.sum(ev < -1e-9 * max(1, abs(ev[-1])))),
               autograd_vs_analytic=agd, N1_min=q, N1_over_gap2=q / sgap ** 2, line=line)
    out.append(rec)
    print(f"{tag:28s} d={d:2d} gap={sgap:.3g} grad={g:.1e} asg={asg} s2={s2:.2e} aGap={ga:.1e} bGap={gb:.1e} "
          f"Hmin={ev[0]:+.3e} nneg={rec['n_neg']} AG={agd if agd is None else f'{agd:.1e}'} N1={q:+.3e} "
          f"line(1e-3)={line[2][1]:+.3e}/{line[2][2]:+.3e}", flush=True)
    return rec

if __name__ == '__main__':
    t0 = time.time(); out = []
    rng = np.random.default_rng(91009)
    patterns = [
        (2, 2, [[0, 1]], [], []),
        (2, 3, [[0, 1]], [], []),
        (2, 3, [], [[0, 1]], []),
        (3, 2, [[0, 1]], [], []),
        (2, 4, [[0, 1]], [], []),
        (2, 4, [], [[0, 1], [2, 3]], []),
        (2, 4, [[0, 1]], [[1, 2]], []),
        (4, 2, [[1, 2]], [[0, 1]], []),
        (3, 3, [[0, 1]], [], []),
        (3, 3, [[0, 1, 2]], [], []),
        (3, 3, [[0, 1]], [[0, 1]], []),
        (3, 3, [[0, 1]], [[1, 2]], []),
        (3, 3, [[1, 2]], [], [((0, 0), (1, 1))]),   # same-column tie + doubly-distinct tie in one L-eigenspace
        (3, 3, [[0, 1]], [], [((2, 0), (0, 1))]),
        (4, 4, [[0, 1]], [[2, 3]], []),
    ]
    print("=== Family A (prescribed tied L, random coherent blocks) ===")
    for (m, n, at, bt, ex) in (patterns if os.environ.get('SKIP_A') != '1' else []):
        for rep in range(3):
            Hp, al, be, groups = family_A(m, n, at, bt, ex, rng)
            analyse(np.eye(m * n, dtype=complex), Hp, m, n, f"A {m}x{n} at={at} bt={bt} ex={len(ex)} r{rep}",
                    m * n <= 9, out)
    print("=== Family B (given GUE H, rows 0/1 tied, polygon coherences) ===")
    for (m, n) in [(2, 3), (2, 4), (3, 3), (2, 5), (4, 3)]:
        d = m * n
        for seed in range(4):
            r2 = np.random.default_rng(1000 * d + seed)
            G = r2.normal(size=(d, d)) + 1j * r2.normal(size=(d, d)); H = (G + G.conj().T) / (2 * np.sqrt(d))
            lam, V = np.linalg.eigh(H)
            res = family_B(lam, m, n, r2)
            if res is None: print("no polygon", m, n, seed); continue
            Hp, info = res
            mu_, W = np.linalg.eigh(Hp)
            U = V @ W.conj().T
            U2, nr, it = lm_critical(U, H, m, n, herm_basis(d), tol=1e-14)
            print(f"   polish: |r| {np.linalg.norm(comm(*parts(U,H,m,n)[::-1])):.1e} -> {nr:.1e}, moved {np.linalg.norm(U2-U):.1e}")
            analyse(U2, H, m, n, f"B {m}x{n} GUE seed={1000*d+seed}", d <= 9, out)
    print("=== Family B on an equally spaced spectrum (max additive coincidences) ===")
    for (m, n) in [(2, 3), (2, 4), (3, 3)]:
        d = m * n
        for seed in range(3):
            r2 = np.random.default_rng(7000 + 10 * d + seed)
            lam = np.arange(d, dtype=float); Q, _ = np.linalg.qr(r2.normal(size=(d, d)) + 1j * r2.normal(size=(d, d)))
            H = Q @ np.diag(lam) @ Q.conj().T
            res = family_B(lam, m, n, r2)
            if res is None: print("no polygon", m, n, seed); continue
            Hp, info = res
            mu_, W = np.linalg.eigh(Hp)
            U = Q @ W.conj().T
            U2, nr, it = lm_critical(U, H, m, n, herm_basis(d), tol=1e-14)
            print(f"   polish: moved {np.linalg.norm(U2-U):.1e}, |r|={nr:.1e}")
            analyse(U2, H, m, n, f"B {m}x{n} EQSP seed={7000+10*d+seed}", d <= 9, out)
    print(f"wall {time.time()-t0:.1f}s")
    viol = [r for r in out if r['grad'] < 1e-11 and not (r['n_neg'] > 0 and r['N1_min'] < 0)]
    print("records:", len(out), " critical (grad<1e-11):", sum(r['grad'] < 1e-11 for r in out),
          " counterexample candidates (critical, degenerate/non-assignment, no negative direction):", len(viol))
    json.dump(out, open(os.path.join(OUT, 'constructed.json'), 'w'), indent=1)
