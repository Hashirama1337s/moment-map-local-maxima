"""Replay one search_k.py record (same RNG stream; expensive steps skipped for other records) and verify it by line tests
along the Hessian-minimum eigenvector.  Also reports the L-level-set structure and Hamming distances inside level sets.
Usage: python replay_k.py <dims> <seed> <kind> <record index>"""
import sys, os, numpy as np
from itertools import product
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core_k import *
from search_k import family_B_k, lm_aug, classify
from scipy.stats import unitary_group

dims = tuple(int(x) for x in sys.argv[1].split(',')); seed = int(sys.argv[2]); kind = sys.argv[3]; target = int(sys.argv[4])
d = int(np.prod(dims)); k = len(dims); rng = np.random.default_rng(seed); basis = herm_basis(d)
if kind == 'gue':
    G = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)); H = (G + G.conj().T) / (2 * np.sqrt(d))
else:
    Q = unitary_group.rvs(d, random_state=seed); H = Q @ np.diag(np.arange(d, dtype=float)) @ Q.conj().T
lam, V = np.linalg.eigh(H)
it = 0; idx = -1
while True:
    mode = ['S1', 'S2', 'S4', 'S3'][it % 4]; it += 1
    if mode in ('S1', 'S3'):
        j = int(rng.integers(k)); Hq = family_B_k(lam, dims, j, rng)
        if Hq is None: continue
        _, W = np.linalg.eigh(Hq); U0 = V @ W.conj().T
        eps = [1e-3, 1e-2, 1e-1, 3e-1][rng.integers(4)]
        X = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)); X = (X + X.conj().T) / 2
        U0 = U0 @ expm(1j * eps * X / np.linalg.norm(X))
    else:
        U0 = unitary_group.rvs(d, random_state=int(rng.integers(1 << 31)))
    if mode == 'S4': j = int(rng.integers(k))
    idx += 1
    if idx < target: continue
    if mode == 'S3': U1 = descend(U0, H, dims); U1, nr = lm_critical(U1, H, dims, basis)
    elif mode == 'S4': U1 = lm_aug(U0, H, dims, basis, j); U1, nr = lm_critical(U1, H, dims, basis)
    else: U1, nr = lm_critical(U0, H, dims, basis)
    break
c = classify(U1, H, dims, basis)
print("replayed", target, mode, {x: c[x] for x in ['grad', 'f', 'hess_min', 'n_neg', 'assignment', 'impurity', 'N1_min']})
Hp, L = parts(U1, H, dims); ev, evec = np.linalg.eigh(hess_analytic(Hp, L, dims, basis)); K = from_coords(evec[:, 0], basis)
f0 = cost(U1, H, dims)
for s in [3e-2, 1e-2, 3e-3, 1e-3]:
    print(f"  s={s:.0e}  f(s)-f(0)={cost(U1 @ expm(1j * s * K), H, dims) - f0:+.4e}   0.5*Hmin*s^2={0.5 * ev[0] * s * s:+.4e}")
T = gauge(Hp, dims); H2 = T.conj().T @ Hp @ T; Lv = np.real(np.diag(P1(H2, dims))); cells = list(product(*[range(x) for x in dims]))
used = np.zeros(d, bool)
for i in range(d):
    if used[i]: continue
    S = [j for j in range(d) if abs(Lv[j] - Lv[i]) < 1e-8 * max(1, np.abs(Lv).max())]; used[S] = True
    if len(S) > 1:
        blk = H2[np.ix_(S, S)]
        print("  level set", [cells[s] for s in S], "max offdiag |H'|", float(np.max(np.abs(blk - np.diag(np.diag(blk))))),
              "Hamming", [sum(a != b for a, b in zip(cells[S[x]], cells[S[y]])) for x in range(len(S)) for y in range(x + 1, len(S))])
