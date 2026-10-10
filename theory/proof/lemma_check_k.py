"""Lemma 2-k check: for a cell set S in a d1 x ... x dk grid, the kernel of
   H_S -> ( K -> (Tr_{not j}[H_S, K])_{j=1..k} ),  H_S, K operators on span{|c> : c in S},
has dimension = number of Hamming-1-connected components of S (cells adjacent iff they differ in exactly one coordinate),
and contains every component projector Pi_C."""
import sys, os, json, numpy as np
from itertools import product
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core_k import ptr_keep

def components(S):
    par = list(range(len(S)))
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            if sum(a != b for a, b in zip(S[i], S[j])) == 1: par[find(i)] = find(j)
    comp = {}
    for i in range(len(S)): comp.setdefault(find(i), []).append(i)
    return list(comp.values())

def check(dims, S):
    d = int(np.prod(dims)); strides = [int(np.prod(dims[i + 1:])) for i in range(len(dims))]
    idx = [sum(c[i] * strides[i] for i in range(len(dims))) for c in S]; k = len(S)
    def E(i, j):
        M = np.zeros((d, d), complex); M[idx[i], idx[j]] = 1; return M
    units = [(i, j) for i in range(k) for j in range(k)]
    cols = []
    for (i, j) in units:
        Hm = E(i, j); out = []
        for (p, q) in units:
            Km = E(p, q); C = Hm @ Km - Km @ Hm
            for jj in range(len(dims)): out.append(ptr_keep(C, dims, jj).ravel())
        cols.append(np.concatenate(out))
    M = np.array(cols).T
    s = np.linalg.svd(M, compute_uv=False); rank = int(np.sum(s > 1e-10 * s[0])) if s[0] > 0 else 0
    comps = components(S); worst = 0.0
    for C in comps:
        v = np.zeros(k * k, complex)
        for i in C: v[i * k + i] = 1
        worst = max(worst, float(np.linalg.norm(M @ v)))
    return k * k - rank, len(comps), worst

if __name__ == '__main__':
    rng = np.random.default_rng(31415); rows = []; bad = 0
    for dims in [(2, 2, 2), (2, 2, 3), (2, 3, 3), (3, 3, 3), (2, 2, 2, 2)]:
        cells = list(product(*[range(x) for x in dims]))
        for t in range(80):
            kk = int(rng.integers(1, min(len(cells), 7) + 1))
            S = [cells[i] for i in rng.choice(len(cells), kk, replace=False)]
            kd, nc, w = check(dims, S); ok = (kd == nc and w < 1e-12); bad += not ok
            rows.append(dict(dims=dims, S=[list(map(int, c)) for c in S], kerdim=kd, ncomp=nc, ok=ok))
    import collections
    print("Lemma 2-k:", len(rows), "random cell sets; mismatches =", bad)
    print("component-count histogram:", dict(collections.Counter(r['ncomp'] for r in rows)))
    json.dump(dict(seed=31415, mismatches=bad, rows=rows), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lemma_check_k.json'), 'w'))
