"""Check of Lemma 3 (explicit descent direction).  At a critical point, in a product basis diagonalising L,
for an L-eigenspace with projector Pi and block H_lam, and X Hermitian on the A factor:
   X_lam = Pi (X(x)1) Pi,   K* = i[X_lam, H_lam]   (Hermitian, [L,K*] = 0)
   Q(K*) / ||K*||^2  <=  -2 ||[X_lam,H_lam]||^2 / (n ||X||^2)      (and the 1(x)Y version with m)
Also checks the second derivative of f along U expm(i s K*) by finite differences."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import *
from constructed import family_A, family_B

def check_point(Hp, m, n):
    d = m * n
    _, V = np.linalg.eigh(ptrB(Hp, m, n)); _, W = np.linalg.eigh(ptrA(Hp, m, n)); T = np.kron(V, W)
    H2 = T.conj().T @ Hp @ T; H2 = (H2 + H2.conj().T) / 2
    L2 = P(H2, m, n)
    off = np.linalg.norm(L2 - np.diag(np.diag(L2)))
    Lv = np.real(np.diag(L2)); scale = max(1, np.max(np.abs(Lv)))
    used = np.zeros(d, bool); worst_ratio = np.inf; best_curv = 0.0; fd_err = 0.0; nK = 0
    for i in range(d):
        if used[i]: continue
        S = [j for j in range(d) if abs(Lv[j] - Lv[i]) < 1e-8 * scale]; used[S] = True
        Pi = np.zeros((d, d)); Pi[S, S] = 1
        for side, k, other in (('A', m, n), ('B', n, m)):
            for Xb in herm_basis(k):
                Xf = np.kron(Xb, np.eye(n)) if side == 'A' else np.kron(np.eye(m), Xb)
                Xl = Pi @ Xf @ Pi; Hl = Pi @ H2 @ Pi
                C = comm(Xl, Hl); c2 = ip(C, C)
                if c2 < 1e-20: continue
                K = 1j * C; K = (K + K.conj().T) / 2
                assert np.linalg.norm(comm(L2, K)) < 1e-9 * scale * np.linalg.norm(K)
                Q = -2 * ip(P(1j * comm(H2, K), m, n), P(1j * comm(H2, K), m, n))
                bound = -2 * c2 / (other * ip(Xb, Xb)) * ip(K, K)
                worst_ratio = min(worst_ratio, Q / bound)   # must be >= 1  (Q <= bound < 0)
                best_curv = min(best_curv, Q / ip(K, K))
                # finite-difference second derivative along expm(i s K) for H := H2, U = 1
                s = 1e-4 / np.sqrt(ip(K, K))
                fp = cost(expm(1j * s * K), H2, m, n); fm = cost(expm(-1j * s * K), H2, m, n); f0 = cost(np.eye(d), H2, m, n)
                fd = (fp + fm - 2 * f0) / s ** 2
                fd_err = max(fd_err, abs(fd - Q) / abs(Q)); nK += 1
    return dict(L_offdiag=float(off), min_Q_over_bound=float(worst_ratio) if nK else None, nK=nK,
                most_negative_curv=float(best_curv), fd_rel_err=float(fd_err))

if __name__ == '__main__':
    rng = np.random.default_rng(424242); out = []
    pats = [(2, 3, [[0, 1]], [], []), (2, 4, [[0, 1]], [[1, 2]], []), (3, 3, [[0, 1]], [[0, 1]], []),
            (3, 3, [[0, 1, 2]], [], []), (3, 3, [[1, 2]], [], [((0, 0), (1, 1))]), (4, 4, [[0, 1]], [[2, 3]], [])]
    for (m, n, at, bt, ex) in pats:
        for rep in range(4):
            Hp, *_ = family_A(m, n, at, bt, ex, rng)
            r = check_point(Hp, m, n); r.update(fam='A', m=m, n=n, at=at, bt=bt, ex=len(ex)); out.append(r); print(r, flush=True)
    for (m, n) in [(2, 3), (2, 4), (3, 3), (2, 5)]:
        for rep in range(4):
            d = m * n; G = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)); H = (G + G.conj().T) / (2 * np.sqrt(d))
            res = family_B(np.linalg.eigvalsh(H), m, n, rng)
            if res is None: continue
            r = check_point(res[0], m, n); r.update(fam='B', m=m, n=n); out.append(r); print(r, flush=True)
    ok = all(r['nK'] > 0 and r['min_Q_over_bound'] >= 1 - 1e-9 for r in out)
    print("ALL: explicit K* found and Q(K*) <= bound in every point:", ok, " points:", len(out),
          " max fd rel err:", max(r['fd_rel_err'] for r in out))
    json.dump(dict(seed=424242, ok=ok, rows=out), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'explicit_dir.json'), 'w'), indent=1)
