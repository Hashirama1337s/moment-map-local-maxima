"""Check of the Commutant Lemma (PROOF-GAP.md, Lemma 2).
For a cell set S in an m x n grid, Pi = projector on span{|a,b>: (a,b) in S}.
Linear map  Phi: H_S -> ( K -> (Tr_A [H_S,K], Tr_B [H_S,K]) ),  H_S, K operators on Pi.
Lemma: ker Phi = span{ Pi_C : C a connected component of S in the rook graph }.
We compute ker Phi numerically (complex-linear, all K in a basis of M(S)) and compare."""
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit('\\', 1)[0] if '\\' in __file__ else '.')
from core import ptrA, ptrB

def components(S):
    S = list(S); idx = {c: i for i, c in enumerate(S)}; parent = list(range(len(S)))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, (a, b) in enumerate(S):
        for j, (a2, b2) in enumerate(S):
            if j > i and (a == a2 or b == b2): parent[find(i)] = find(j)
    comp = {}
    for i in range(len(S)): comp.setdefault(find(i), []).append(S[i])
    return list(comp.values())

def check(m, n, S):
    d = m * n; S = sorted(S); k = len(S)
    cell = [a * n + b for a, b in S]
    E = lambda i, j: np.outer(np.eye(d)[cell[i]], np.eye(d)[cell[j]]).astype(complex)
    units = [(i, j) for i in range(k) for j in range(k)]
    # rows: for each H-unit (i,j), stack Phi applied with every K-unit
    cols = []
    for (i, j) in units:
        Hm = E(i, j); out = []
        for (p, q) in units:
            Km = E(p, q); C = Hm @ Km - Km @ Hm
            out.append(ptrA(C, m, n).ravel()); out.append(ptrB(C, m, n).ravel())
        cols.append(np.concatenate(out))
    M = np.array(cols).T  # complex linear map from k^2 coefficients
    s = np.linalg.svd(M, compute_uv=False)
    rank = int(np.sum(s > 1e-10 * s[0])) if s[0] > 0 else 0
    kerdim = k * k - rank
    comps = components(S)
    # verify predicted kernel vectors lie in kernel
    worst = 0.0
    for C in comps:
        v = np.zeros(k * k, complex)
        for c in C:
            i = S.index(c); v[i * k + i] = 1
        worst = max(worst, np.linalg.norm(M @ v))
    return kerdim, len(comps), worst

if __name__ == '__main__':
    rng = np.random.default_rng(20261009)
    res = []; bad = 0; ntest = 0
    for (m, n) in [(2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (3, 4), (4, 4)]:
        cells = [(a, b) for a in range(m) for b in range(n)]
        for t in range(60):
            k = int(rng.integers(1, min(len(cells), 7) + 1))
            S = [cells[i] for i in rng.choice(len(cells), k, replace=False)]
            kd, nc, w = check(m, n, S)
            ntest += 1
            ok = (kd == nc) and w < 1e-12
            bad += (not ok)
            res.append(dict(m=m, n=n, S=S, kerdim=kd, ncomp=nc, worst=w, ok=ok))
    print(f"commutant lemma: {ntest} random cell sets, mismatches = {bad}")
    import collections
    print("kerdim==#components in all:", all(r['ok'] for r in res))
    print("examples:", [(r['m'], r['n'], len(r['S']), r['kerdim'], r['ncomp']) for r in res[:8]])
    json.dump(dict(seed=20261009, ntest=ntest, mismatches=bad,
                   rows=[{**r, 'S': [list(map(int, c)) for c in r['S']]} for r in res]),
              open(__file__.replace('lemma_check.py', 'lemma_check.json'), 'w'), indent=0)
