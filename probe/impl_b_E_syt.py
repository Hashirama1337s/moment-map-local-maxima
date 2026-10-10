#!/usr/bin/env python3
"""
impl_b_E_syt.py -- implementation B exact minimum of the two-way additive residual over a dA x dB grid by depth-first
branch-and-bound over standard Young tableaux, built COLUMN BY COLUMN (own method, blind).

Why the search space is complete: at a global minimum with distinct values every two rows and every two columns
are entrywise ordered (T1 lemma: otherwise the D3 row/column Hessian block has a negative eigenvalue), so after
row/column permutations (which leave the residual unchanged) the rank grid is a standard Young tableau (SYT).
Every SYT is generated exactly once: column b = {smallest remaining value} + (dA-1) other remaining values, sorted,
required to be entrywise above column b-1 (this is exactly "rows and columns increasing").

Bound: RSS(grid) = sum_b ||z_b - zbar||^2 (z_b = centred column). For chosen columns C (any subset),
sum_{b in C} ||z_b - mean_C z||^2 <= sum_{b in C} ||z_b - zbar||^2 <= RSS, so the variance of the columns chosen so
far is a valid, monotone lower bound. Lookahead (also valid): each still-to-be-built column must contain the smallest
value remaining when it is built, etc. -- not used; plain variance bound only.
Prune when LB >= UB (1 - TOL). Exact (to TOL) once the tree is exhausted.

Usage: py -3.11 impl_b_E_syt.py <n> <dA> <k> [ub_hint]
"""
import os
import sys
import json
import time
import itertools
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import impl_b as B  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "impl_b_E_out")
TOL = 1e-12


def rss_grid(X):
    R = X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + X.mean()
    return float(np.sum(R ** 2))


def descent(lam, m, G, rng, n_starts=300):
    n = len(lam)
    I, J = np.triu_indices(n, 1)
    idx = np.arange(len(I))

    def batch(F):
        X = F.reshape(-1, m, G)
        R = X - X.mean(2, keepdims=True) - X.mean(1, keepdims=True) + X.mean((1, 2), keepdims=True)
        return (R ** 2).sum((1, 2))
    best, bp = np.inf, None
    for _ in range(n_starts):
        p = rng.permutation(n)
        cur = float(batch(lam[p][None])[0])
        while True:
            F = np.tile(lam[p], (len(I), 1))
            F[idx, I], F[idx, J] = F[idx, J], F[idx, I].copy()
            v = batch(F)
            k = int(np.argmin(v))
            if v[k] >= cur * (1 - 1e-14):
                break
            p = p.copy()
            p[I[k]], p[J[k]] = p[J[k]], p[I[k]]
            cur = float(v[k])
        if cur < best:
            best, bp = cur, p.copy()
    return best, bp.reshape(m, G)


class Search:
    def __init__(self, lam, m, G, ub, ub_grid, time_limit=1100):
        self.lam = lam
        self.m, self.G = m, G
        self.ub, self.best = ub, ub_grid
        self.nodes = 0
        self.t0 = time.time()
        self.time_limit = time_limit
        self.timed_out = False
        self.combo = {}

    def combos(self, r):
        if r not in self.combo:
            self.combo[r] = np.array(list(itertools.combinations(range(r), self.m - 1)), dtype=np.int64).reshape(-1, self.m - 1)
        return self.combo[r]

    def run(self):
        n = len(self.lam)
        self.dfs(np.arange(n), None, np.zeros(self.m), 0.0, 0, [])
        return self.ub, self.best

    def dfs(self, remaining, prev, S, Q, b, cols):
        if self.timed_out:
            return
        self.nodes += 1
        if self.nodes % 200000 == 0:
            print(f"   [syt] nodes={self.nodes} ub={self.ub:.12g} t={time.time() - self.t0:.0f}s", flush=True)
            if time.time() - self.t0 > self.time_limit:
                self.timed_out = True
                return
        m, G, lam = self.m, self.G, self.lam
        if b == G:
            rss = Q - float(S @ S) / G
            if rss < self.ub:
                self.ub = rss
                self.best = np.array(cols).T.copy()
            return
        first = remaining[0]
        rest = remaining[1:]
        cmb = self.combos(len(rest))
        C = np.column_stack([np.full(len(cmb), first), rest[cmb]])          # (nc, m) sorted rank columns
        if prev is not None:
            C = C[np.all(C > prev[None, :], axis=1)]                            # entrywise above previous column
            if len(C) == 0:
                return
        if b == G - 1:
            pass
        Z = lam[C]
        Z = Z - Z.mean(1, keepdims=True)
        Q2 = Q + np.sum(Z ** 2, axis=1)
        S2 = S[None, :] + Z
        LB = Q2 - np.sum(S2 ** 2, axis=1) / (b + 1)
        cut = self.ub * (1 - TOL)
        ok = np.where(LB < cut)[0]
        if len(ok) == 0:
            return
        ok = ok[np.argsort(LB[ok])]
        for i in ok:
            if LB[i] >= self.ub * (1 - TOL):
                break
            c = C[i]
            mask = np.ones(len(remaining), bool)
            mask[np.searchsorted(remaining, c)] = False
            self.dfs(remaining[mask], c, S2[i], float(Q2[i]), b + 1, cols + [c])


def main():
    n, m, k = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    lam = np.sort(np.linalg.eigvalsh(B.build_H_plant(n, 1000 + k)))
    G = len(lam) // m
    norm = float(np.sum(lam ** 2))
    gap = float(np.min(np.diff(lam)))
    t0 = time.time()
    ub, grid = descent(lam, m, G, np.random.default_rng(888000 + k))
    # descent grid -> SYT form (sort rows and columns by means, then sort within columns)
    ub_heur = ub
    s = Search(lam, m, G, ub * (1 + 1e-12), None)
    ub_s, best = s.run()
    status = "TIME_LIMIT" if s.timed_out else "OPTIMAL"
    if best is None:
        best_grid = None
        val = ub_heur
    else:
        best_grid = best
        val = rss_grid(lam[best])
    out = dict(case=f"{m}x{G}", n=n, k=k, seed_H=1000 + k, status=status, value=val / norm, rss=val, norm=norm,
               heuristic_value=ub_heur / norm, rank_grid=None if best_grid is None else best_grid.tolist(),
               nodes=s.nodes, t=time.time() - t0, min_gap=gap,
               note="best is None means the search found nothing below the heuristic UB*(1+1e-12): heuristic is optimal")
    if best_grid is None:
        # the heuristic grid is optimal; report it in SYT form
        g = grid[np.argsort(lam[grid].mean(1))][:, np.argsort(lam[grid].mean(0))]
        g = np.sort(g, axis=0)
        g = g[:, np.argsort(g[0])]
        out['rank_grid'] = g.tolist()
        out['value'] = rss_grid(lam[g]) / norm
        out['rss'] = rss_grid(lam[g])
    G_ = np.array(out['rank_grid'])
    out['syt'] = bool(np.all(np.diff(G_, axis=0) > 0) and np.all(np.diff(G_, axis=1) > 0))
    with open(os.path.join(OUT, f"V_{m}x{G}_ch{k}_syt.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"{m}x{G} ch{k}: status={status} value={out['value']:.11e} heur={ub_heur / norm:.11e} nodes={s.nodes} "
          f"SYT={out['syt']} t={out['t']:.0f}s", flush=True)
    print("  rank grid:", out['rank_grid'], flush=True)


if __name__ == "__main__":
    main()
