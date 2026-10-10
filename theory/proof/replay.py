"""Replay one search.py record (same RNG stream, expensive steps skipped for other records), save U, and verify its
saddle status by explicit line tests.  Usage: python replay.py m n seed kind index"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import *
from constructed import family_B
from search import lm_aug, classify
from scipy.stats import unitary_group

m, n, seed, kind, target = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
d = m * n; rng = np.random.default_rng(seed); basis = herm_basis(d)
if kind == 'gue':
    G = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)); H = (G + G.conj().T) / (2 * np.sqrt(d))
else:
    Q = unitary_group.rvs(d, random_state=seed); H = Q @ np.diag(np.arange(d, dtype=float)) @ Q.conj().T
lam, V = np.linalg.eigh(H)
k = 0; idx = -1
while True:
    mode = ['S1', 'S2', 'S4', 'S3'][k % 4]; k += 1
    if mode in ('S1', 'S3'):
        res = family_B(lam, m, n, rng)
        if res is None: continue
        Hp0, _ = res; _, W = np.linalg.eigh(Hp0); U0 = V @ W.conj().T
        eps = [1e-3, 1e-2, 1e-1, 3e-1][rng.integers(4)]
        X = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)); X = (X + X.conj().T) / 2
        U0 = U0 @ expm(1j * eps * X / np.linalg.norm(X))
    else:
        U0 = unitary_group.rvs(d, random_state=int(rng.integers(1 << 31)))
    if mode == 'S4':
        which = 'A' if (rng.random() < 0.5 or n < 2) else 'B'
    idx += 1
    if idx < target: continue
    if mode == 'S3':
        U1, f1, g1 = descend(U0, H, m, n, iters=3000, tol=1e-6); U1, nr, _ = lm_critical(U1, H, m, n, basis, tol=1e-14)
    elif mode == 'S4':
        U1, _ = lm_aug(U0, H, m, n, basis, which); U1, nr, _ = lm_critical(U1, H, m, n, basis, tol=1e-14)
    else:
        U1, nr, _ = lm_critical(U0, H, m, n, basis, tol=1e-14)
    break
c = classify(U1, H, m, n, basis); print("replayed record", target, mode, {k_: c[k_] for k_ in ['grad', 'f', 'hess_min', 'n_neg', 'assignment', 's2', 'alpha_gap', 'beta_gap', 'N1_min']})
Hp, L = parts(U1, H, m, n)
ev, evec = np.linalg.eigh(hess_analytic(Hp, L, m, n, basis))
Kh = from_coords(evec[:, 0], basis)
q, Kn, gs = commutant_form(Hp, L, m, n); Kn = Kn / np.sqrt(ip(Kn, Kn))
f0 = cost(U1, H, m, n)
for name, K, curv in (('Hessian-min eigvec', Kh, ev[0]), ('N1 minimiser', Kn, q)):
    print(name, "predicted curvature", curv)
    for s in [3e-2, 1e-2, 3e-3]:
        df = cost(U1 @ expm(1j * s * K), H, m, n) - f0
        print(f"   s={s:.0e}  f(s)-f(0)={df:+.4e}  0.5*curv*s^2={0.5*curv*s*s:+.4e}")
print("L level-set sizes (largest first):", sorted(gs, reverse=True)[:4])
np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'replay_{m}x{n}_{kind}_{seed}_{target}_U.npy'), U1)
np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'replay_{m}x{n}_{kind}_{seed}_{target}_H.npy'), H)
