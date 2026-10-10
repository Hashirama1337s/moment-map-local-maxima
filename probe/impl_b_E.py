#!/usr/bin/env python3
"""
impl_b_E.py -- implementation B exact global minimum of cost1 for whole-spectrum splits (blind; own code only).

Theorem used (accepted): min over U(d) of cost1 = min over grid assignments X of the two-way additive-fit residual
RSS(X) / ||lam||^2.

Reformulation (mine): for a dA x dB grid, RSS(X) = sum_b ||z_b - zbar||^2, z_b = column b centred over rows.
WLOG (row-permutation gauge) the common row profile rho is sorted and every column is sorted ascending, so
    min_X RSS = min_{partition P of the spectrum into dB groups of dA}  min_rho  sum_{g in P} ||p_g - rho||^2,
with p_g = sorted, centred group profile, expressed in an orthonormal (Helmert) basis of the sum-zero subspace
(dimension dA - 1: 1 for V1, 3 for V2).

Exact method: certified branch-and-bound over rho-boxes.
  LB(box)  = min over partitions (excluding a global no-good list) of sum_g dist(p_g, box)^2
             -> set-partitioning ILP solved to optimality by HiGHS (scipy.optimize.milp, mip_rel_gap = 0),
                using the solver's DUAL bound;
  UB       = exact RSS of every partition the ILPs return;
  no-goods = every evaluated partition (each has RSS >= the incumbent, so it can never be strictly better);
  prune when LB >= UB (1 - TOL); otherwise re-solve with the new no-good, then split the box.
Any partition Q with RSS(Q) < UB(1 - TOL) has its mean profile inside some leaf box and is not on the no-good list,
so that leaf's LB <= RSS(Q) < UB(1-TOL), and the leaf could not have been pruned. Hence exact to TOL.

Usage: py -3.11 impl_b_E.py V1 <k>   (n=6, grid 2 x 32)
       py -3.11 impl_b_E.py V2 <k>   (n=5, grid 4 x 8)
"""
import os
import sys
import json
import time
import heapq
import itertools
import numpy as np
from scipy.optimize import milp, linprog, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import impl_b as B  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "impl_b_E_out")
os.makedirs(OUT, exist_ok=True)
TOL = 1e-10


def helmert(m):
    rows = []
    for i in range(1, m):
        v = np.zeros(m)
        v[:i] = -1.0
        v[i] = i
        rows.append(v / np.linalg.norm(v))
    return np.array(rows)


class Problem:
    def __init__(self, lam, m):
        self.lam = np.sort(np.asarray(lam, float))
        self.N = len(self.lam)
        self.m = m                       # group size = number of grid rows (dA)
        self.G = self.N // m             # number of groups = number of grid columns (dB)
        self.groups = np.array(list(itertools.combinations(range(self.N), m)), dtype=np.int64)
        vals = self.lam[self.groups]
        prof = vals - vals.mean(1, keepdims=True)
        self.coords = prof @ helmert(m).T          # (ngroups, m-1)
        self.lo0 = self.coords.min(0)
        self.hi0 = self.coords.max(0)
        rows = self.groups.ravel()
        cols = np.repeat(np.arange(len(self.groups)), m)
        self.A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(self.N, len(self.groups)))
        self.nogood = []                 # list of sorted group-index arrays
        self.nogood_set = set()
        self.n_ilp = 0
        self.n_lp = 0
        self.t_ilp = 0.0

    def rss(self, gidx):
        c = self.coords[gidx]
        return float(np.sum((c - c.mean(0)) ** 2))

    def add_nogood(self, gidx):
        key = tuple(sorted(int(i) for i in gidx))
        if key not in self.nogood_set:
            self.nogood_set.add(key)
            self.nogood.append(np.array(key, dtype=np.int64))

    def box_cost(self, lo, hi):
        c = self.coords
        dl = np.clip(lo - c, 0, None)
        dh = np.clip(c - hi, 0, None)
        return np.sum(dl ** 2 + dh ** 2, axis=1)

    def _rows(self, keep):
        """Equality matrix and no-good matrix restricted to the kept columns."""
        A = self.A[:, keep]
        pos = {int(g): i for i, g in enumerate(keep)}
        rows_ng = [ng for ng in self.nogood if all(int(g) in pos for g in ng)]
        E = None
        if rows_ng:
            er = np.repeat(np.arange(len(rows_ng)), self.G)
            ec = np.array([pos[int(g)] for ng in rows_ng for g in ng])
            E = csr_matrix((np.ones(len(ec)), (er, ec)), shape=(len(rows_ng), len(keep)))
        return A, E

    def solve(self, cost, ub):
        """min sum cost over partitions avoiding no-goods, exact.
        Step 1: drop groups with cost >= ub (they cannot appear in a partition with total < ub).
        Step 2: LP relaxation; if LP >= ub -> prune; if LP solution integral -> it is optimal.
        Step 3: reduced-cost fixing (any integer x containing g has cost >= dual_obj + rc_g + sum of negative rc).
        Step 4: MILP (HiGHS, mip_rel_gap = 0) on the surviving columns; its dual bound is the LB.
        Returns (lower bound valid for all partitions with cost < ub, chosen group indices or None).
        A returned LB >= ub(1-TOL) means: no partition with cost < ub(1-TOL)."""
        cut = ub * (1 - TOL)
        keep = np.where(cost < cut)[0] if np.isfinite(ub) else np.arange(len(cost))
        if len(keep) < self.G:
            return np.inf, None
        A, E = self._rows(keep)
        if np.any(np.asarray(A.sum(1)).ravel() == 0):
            return np.inf, None
        t0 = time.time()
        lp = linprog(cost[keep], A_eq=A, b_eq=np.ones(self.N), A_ub=E,
                     b_ub=None if E is None else np.full(E.shape[0], self.G - 1.0), bounds=(0, 1), method='highs')
        self.n_lp += 1
        if lp.status == 2:
            self.t_ilp += time.time() - t0
            return np.inf, None
        if lp.status != 0:
            raise RuntimeError(f"linprog status {lp.status}: {lp.message}")
        y = lp.eqlin.marginals
        z = lp.ineqlin.marginals if E is not None else np.zeros(0)
        rc = cost[keep] - A.T @ y - (E.T @ z if E is not None else 0.0)
        dual_obj = float(y.sum() + (z.sum() * (self.G - 1) if E is not None else 0.0))
        lp_lb = min(float(lp.fun), dual_obj + float(np.sum(np.sort(np.minimum(rc, 0))[:self.G])))
        if lp_lb >= cut:
            self.t_ilp += time.time() - t0
            return lp_lb, None
        xi = lp.x
        if np.all(np.minimum(np.abs(xi), np.abs(1 - xi)) < 1e-9):
            gidx = keep[np.where(xi > 0.5)[0]]
            if len(gidx) == self.G and np.all(np.asarray(self.A[:, gidx].sum(1)).ravel() == 1):
                self.t_ilp += time.time() - t0
                return min(lp_lb, float(np.sum(cost[gidx]))), gidx
        negsum = float(np.sum(np.sort(np.minimum(rc, 0))[:self.G - 1]))
        keep2 = keep[dual_obj + rc + negsum < cut]
        if len(keep2) < self.G:
            self.t_ilp += time.time() - t0
            return cut, None
        A2, E2 = self._rows(keep2)
        if np.any(np.asarray(A2.sum(1)).ravel() == 0):
            self.t_ilp += time.time() - t0
            return cut, None
        scale = 1e6 / ub if np.isfinite(ub) and ub > 0 else 1.0
        cons = [LinearConstraint(A2, 1, 1)]
        if E2 is not None:
            cons.append(LinearConstraint(E2, -np.inf, self.G - 1))
        res = milp(c=cost[keep2] * scale, constraints=cons, integrality=np.ones(len(keep2)),
                   bounds=Bounds(0, 1), options=dict(disp=False, mip_rel_gap=0.0))
        self.t_ilp += time.time() - t0
        self.n_ilp += 1
        if res.status == 2 or res.x is None:
            return cut, None
        if res.status != 0:
            raise RuntimeError(f"milp status {res.status}: {res.message}")
        gidx = keep2[np.where(res.x > 0.5)[0]]
        assert len(gidx) == self.G
        lb = max(lp_lb, res.mip_dual_bound / scale)
        return min(lb, float(np.sum(cost[gidx]))), gidx


def heuristic_ub(P, rng, n_random=20):
    """Alternating minimisation (ILP at a point rho, then rho <- mean profile), from structured + random starts."""
    best, bestg = np.inf, None
    starts = []
    m, G = P.m, P.G
    # structured: consecutive blocks as grid rows (column b = lam[b], lam[G+b], ...), and adjacent groups
    for gs in ([np.where((P.groups == np.arange(m) * G + b).all(1))[0][0] for b in range(G)],
               [np.where((P.groups == np.arange(m) + m * b).all(1))[0][0] for b in range(G)]):
        gs = np.array(gs)
        r = P.rss(gs)
        if r < best:
            best, bestg = r, gs
        starts.append(P.coords[gs].mean(0))
    for _ in range(n_random):
        starts.append(P.lo0 + rng.random(len(P.lo0)) * (P.hi0 - P.lo0))
    for rho in starts:
        prev = None
        for _ in range(30):
            cost = np.sum((P.coords - rho) ** 2, axis=1)
            lb, g = P.solve(cost, 1.5 * best)
            if g is None:
                break
            r = P.rss(g)
            if r < best:
                best, bestg = r, g
            key = tuple(sorted(g))
            if key == prev:
                break
            prev = key
            rho = P.coords[g].mean(0)
    return best, bestg


def solve_lin(P, cost, cut, l, u, lp_only=False):
    """Exact min of sum_g cost_g over partitions avoiding no-goods with l <= sum_g x_g p_g <= u (p_g = coords).
    Returns (lower bound on that min [>= cut means: nothing below cut], group indices of an optimal or integral-LP
    solution or None). LP -> reduced-cost fixing -> MILP (HiGHS, mip_rel_gap 0) on surviving columns."""
    C = P.coords
    ng = len(P.nogood)
    allcols = np.arange(len(cost))
    A, E = P._rows(allcols)
    rows = [] if E is None else [E]
    rows += [csr_matrix(C.T), csr_matrix(-C.T)]
    from scipy.sparse import vstack
    Aub = vstack(rows).tocsr()
    bub = np.concatenate([np.full(0 if E is None else E.shape[0], P.G - 1.0), u, -l])
    t0 = time.time()
    lp = linprog(cost, A_eq=A, b_eq=np.ones(P.N), A_ub=Aub, b_ub=bub, bounds=(0, 1), method='highs')
    P.n_lp += 1
    if lp.status == 2:
        P.t_ilp += time.time() - t0
        return np.inf, None
    if lp.status != 0:
        raise RuntimeError(f"linprog status {lp.status}: {lp.message}")
    y = lp.eqlin.marginals
    z = lp.ineqlin.marginals
    rc = cost - A.T @ y - Aub.T @ z
    dual_obj = float(y.sum() + z @ bub)
    lp_lb = min(float(lp.fun), dual_obj + float(np.sum(np.sort(np.minimum(rc, 0))[:P.G])))
    xi = lp.x
    integral = bool(np.all(np.minimum(np.abs(xi), np.abs(1 - xi)) < 1e-9))
    if integral:
        g = np.where(xi > 0.5)[0]
        P.t_ilp += time.time() - t0
        if len(g) == P.G and np.all(np.asarray(P.A[:, g].sum(1)).ravel() == 1):
            return min(lp_lb, float(np.sum(cost[g]))), g
    if lp_lb >= cut or lp_only:
        P.t_ilp += time.time() - t0
        return lp_lb, None
    negsum = float(np.sum(np.sort(np.minimum(rc, 0))[:P.G - 1]))
    keep2 = np.where(dual_obj + rc + negsum < cut)[0]
    if len(keep2) < P.G:
        P.t_ilp += time.time() - t0
        return cut, None
    A2, E2 = P._rows(keep2)
    if np.any(np.asarray(A2.sum(1)).ravel() == 0):
        P.t_ilp += time.time() - t0
        return cut, None
    scale = 1e6 / max(abs(cut), 1e-12)
    cons = [LinearConstraint(A2, 1, 1), LinearConstraint(csr_matrix(C[keep2].T), l, u)]
    if E2 is not None:
        cons.append(LinearConstraint(E2, -np.inf, P.G - 1))
    res = milp(c=cost[keep2] * scale, constraints=cons, integrality=np.ones(len(keep2)),
               bounds=Bounds(0, 1), options=dict(disp=False, mip_rel_gap=0.0))
    P.t_ilp += time.time() - t0
    P.n_ilp += 1
    if res.status == 2 or res.x is None:
        return cut, None
    if res.status != 0:
        raise RuntimeError(f"milp status {res.status}: {res.message}")
    g = keep2[np.where(res.x > 0.5)[0]]
    lb = max(lp_lb, res.mip_dual_bound / scale)
    return min(lb, float(np.sum(cost[g]))), g


def obbt(P, ub, l, u, rounds=20, tol=1e-9, verbose=False):
    """Optimality-based bound tightening of the s-box (valid: any partition with RSS <= ub satisfies
    sum_g [q_g - p_g.(l+u)/G] x_g + sum_i l_i u_i / G <= ub because the secant underestimates -||s||^2/G on [l,u]).
    For each coordinate, min/max s_i over the LP relaxation of {partition, l <= s <= u, that inequality}.
    Returns tightened (l, u), or (None, None) if the LP is infeasible (no partition can beat ub in this box)."""
    C = P.coords
    G = float(P.G)
    q = np.sum(C ** 2, axis=1)
    A, E = P._rows(np.arange(len(q)))
    from scipy.sparse import vstack
    l, u = l.copy(), u.copy()
    for r in range(rounds):
        lin = q - C @ ((l + u) / G)
        const = float(np.sum(l * u) / G)
        rows = ([] if E is None else [E]) + [csr_matrix(lin[None, :])]
        Aub = vstack(rows).tocsr()
        bub = np.concatenate([np.full(0 if E is None else E.shape[0], P.G - 1.0), [ub - const]])
        changed = 0.0
        for i in range(C.shape[1]):
            for sense in (1.0, -1.0):
                lp = linprog(sense * C[:, i], A_eq=A, b_eq=np.ones(P.N), A_ub=vstack([Aub, csr_matrix(C.T), csr_matrix(-C.T)]).tocsr(),
                             b_ub=np.concatenate([bub, u, -l]), bounds=(0, 1), method='highs')
                P.n_lp += 1
                if lp.status == 2:
                    return None, None
                if lp.status != 0:
                    raise RuntimeError(lp.message)
                val = sense * lp.fun
                if sense > 0 and val > l[i]:
                    changed = max(changed, val - l[i])
                    l[i] = val
                if sense < 0 and val < u[i]:
                    changed = max(changed, u[i] - val)
                    u[i] = val
        if verbose:
            print(f"   [obbt] round {r}: widths {np.round(u - l, 6)} max change {changed:.3g}", flush=True)
        if changed < tol * max(1.0, float(np.max(u - l))):
            break
    return l, u


def bb_secant(P, ub, best, time_limit=1100, max_resolve=6, split_frac=0.01):
    """Certified spatial B&B on s = sum_g p_g (RSS = sum q_g - ||s||^2/G, concave in s; G = number of groups).
    In a box [l,u]: -s_i^2/G >= secant  -(l_i+u_i) s_i/G + l_i u_i/G, so
    LB = min_{partition, l<=s<=u} sum_g [q_g - p_g.(l+u)/G] + sum_i l_i u_i/G (exact ILP/LP).
    Secant error <= sum_i (u_i-l_i)^2/(4G) -> 0 quadratically."""
    t0 = time.time()
    C = P.coords
    G = float(P.G)                    # RSS = sum q_g - ||s||^2 / G  (G = number of groups; 8 for V2)
    q = np.sum(C ** 2, axis=1)
    P.add_nogood(best)
    heap = [(-np.inf, 0, G * C.min(0), G * C.max(0))]
    cnt, n_boxes, min_pruned = 1, 0, np.inf
    while heap:
        if time.time() - t0 > time_limit:
            return ub, best, dict(status="TIME_LIMIT", open_boxes=len(heap), min_open_lb=heap[0][0],
                                  boxes=n_boxes, min_pruned_lb=min_pruned, time=time.time() - t0)
        plb, _, l, u = heapq.heappop(heap)
        if plb >= ub * (1 - TOL):
            min_pruned = min(min_pruned, plb)
            continue
        n_boxes += 1
        if n_boxes % 100 == 0:
            print(f"   [bb] boxes={n_boxes} open={len(heap)} lb_front={plb:.10g} ub={ub:.12g} "
                  f"nogoods={len(P.nogood)} LPs={P.n_lp} MILPs={P.n_ilp} t={time.time() - t0:.0f}s", flush=True)
        cost = q - C @ ((l + u) / G)
        const = float(np.sum(l * u) / G)
        err = float(np.sum((u - l) ** 2) / (4 * G))
        big = err > split_frac * ub
        LB = None
        resolved = False
        for it in range(1 if big else max_resolve):
            cut = ub * (1 - TOL) - const
            lb, g = solve_lin(P, cost, cut, l, u, lp_only=big)
            LB = lb + const
            if g is not None:
                r = P.rss(g)
                if r < ub:
                    ub, best = r, g
                P.add_nogood(g)
            if LB >= ub * (1 - TOL):
                min_pruned = min(min_pruned, LB)
                resolved = True
                break
            if g is None and not big:
                break
        if resolved:
            continue
        j = int(np.argmax(u - l))
        mid = 0.5 * (l[j] + u[j])
        for (a, b) in ((l[j], mid), (mid, u[j])):
            l2, u2 = l.copy(), u.copy()
            l2[j], u2[j] = a, b
            heapq.heappush(heap, (LB, cnt, l2, u2))
            cnt += 1
    return ub, best, dict(status="OPTIMAL", boxes=n_boxes, min_pruned_lb=min_pruned, time=time.time() - t0)


def descent_ub(P, rng, n_starts=300):
    """Upper bound by steepest descent over single transpositions of grid entries, from random grids.
    Returns (RSS of the best partition, group indices). Columns are then sorted, which can only lower the RSS
    (||sort x - sort y|| <= ||x - y||)."""
    lam, m, G, n = P.lam, P.m, P.G, P.N
    I, J = np.triu_indices(n, 1)
    idx = np.arange(len(I))
    lookup = {tuple(int(v) for v in g): i for i, g in enumerate(P.groups)}

    def rss_batch(F):
        X = F.reshape(-1, m, G)
        R = X - X.mean(2, keepdims=True) - X.mean(1, keepdims=True) + X.mean((1, 2), keepdims=True)
        return (R ** 2).sum((1, 2))

    best, bestp = np.inf, None
    for _ in range(n_starts):
        p = rng.permutation(n)
        cur = float(rss_batch(lam[p][None])[0])
        while True:
            F = np.tile(lam[p], (len(I), 1))
            F[idx, I], F[idx, J] = F[idx, J], F[idx, I].copy()
            v = rss_batch(F)
            k = int(np.argmin(v))
            if v[k] >= cur * (1 - 1e-14):
                break
            p = p.copy()
            p[I[k]], p[J[k]] = p[J[k]], p[I[k]]
            cur = float(v[k])
        if cur < best:
            best, bestp = cur, p.copy()
    cols = np.sort(bestp.reshape(m, G), axis=0).T
    gidx = np.array([lookup[tuple(int(v) for v in c)] for c in cols])
    return P.rss(gidx), gidx


def branch_and_bound(P, ub, best, time_limit=1100, max_resolve=6):
    t0 = time.time()
    heap = [(0.0, 0, P.lo0.copy(), P.hi0.copy())]
    cnt = 1
    n_boxes = 0
    min_pruned_lb = np.inf
    P.add_nogood(best)
    while heap:
        if time.time() - t0 > time_limit:
            return ub, best, dict(status="TIME_LIMIT", open_boxes=len(heap), min_open_lb=heap[0][0],
                                  boxes=n_boxes, min_pruned_lb=min_pruned_lb)
        plb, _, lo, hi = heapq.heappop(heap)
        if plb >= ub * (1 - TOL):
            min_pruned_lb = min(min_pruned_lb, plb)
            continue
        n_boxes += 1
        if n_boxes % 100 == 0:
            print(f"   [bb] boxes={n_boxes} open={len(heap)} lb_front={plb:.10g} ub={ub:.12g} "
                  f"nogoods={len(P.nogood)} LPs={P.n_lp} MILPs={P.n_ilp} t={time.time() - t0:.0f}s", flush=True)
        cost = P.box_cost(lo, hi)
        resolved = False
        lb = None
        for _ in range(max_resolve):
            lb, g = P.solve(cost, ub)
            if g is None or lb >= ub * (1 - TOL):
                min_pruned_lb = min(min_pruned_lb, lb)
                resolved = True
                break
            r = P.rss(g)
            if r < ub:
                ub, best = r, g
            P.add_nogood(g)
        if resolved:
            continue
        w = (hi - lo)
        j = int(np.argmax(w))
        mid = 0.5 * (lo[j] + hi[j])
        for (a, b) in ((lo[j], mid), (mid, hi[j])):
            lo2, hi2 = lo.copy(), hi.copy()
            lo2[j], hi2[j] = a, b
            heapq.heappush(heap, (lb, cnt, lo2, hi2))
            cnt += 1
    return ub, best, dict(status="OPTIMAL", boxes=n_boxes, min_pruned_lb=min_pruned_lb, time=time.time() - t0)


def rank_grid(P, gidx):
    """Grid rows = rank within column (sorted ascending), columns ordered by their row-0 value; 0-based ranks."""
    cols = P.groups[gidx]
    cols = cols[np.argsort(cols[:, 0])]
    return cols.T          # (m, G): entry = index into the sorted spectrum = rank


def d4_check(grid_vals):
    X = grid_vals
    rows_ok = all(np.all(X[a] < X[a2]) or np.all(X[a] > X[a2]) for a in range(X.shape[0]) for a2 in range(a + 1, X.shape[0]))
    cols_ok = all(np.all(X[:, b] < X[:, b2]) or np.all(X[:, b] > X[:, b2]) for b in range(X.shape[1]) for b2 in range(b + 1, X.shape[1]))
    L = X.mean(1, keepdims=True) + X.mean(0, keepdims=True) - X.mean()
    Xf, Lf = X.ravel(), L.ravel()
    I, J = np.triu_indices(len(Xf), 1)
    anti = bool(np.all((Lf[I] - Lf[J]) * (Xf[I] - Xf[J]) > 0))
    return rows_ok, cols_ok, anti


def run(case, k):
    t0 = time.time()
    if case == "V1":
        n, m = 6, 2
    else:
        n, m = 5, 4
    lam = np.linalg.eigvalsh(B.build_H_plant(n, 1000 + k))
    P = Problem(lam, m)
    norm = float(np.sum(P.lam ** 2))
    rng = np.random.default_rng(777000 + k)
    if HEURISTIC == "descent":
        ub, best = descent_ub(P, rng)
    else:
        ub, best = heuristic_ub(P, rng)
    ub_heur = ub
    t_heur = time.time() - t0
    if BOUND == "secant":
        ub, best, info = bb_secant(P, ub, best)
    else:
        ub, best, info = branch_and_bound(P, ub, best)
    info['bound'] = BOUND
    info['heuristic'] = HEURISTIC
    grid = rank_grid(P, best)
    gv = P.lam[grid]
    # direct recomputation of the two-way additive residual on the grid (independent of the profile algebra)
    R = gv - gv.mean(1, keepdims=True) - gv.mean(0, keepdims=True) + gv.mean()
    rss_direct = float(np.sum(R ** 2))
    rows_ok, cols_ok, anti = d4_check(gv)
    out = dict(case=case, k=k, n=n, grid_shape=[m, P.G], seed_H=1000 + k, status=info['status'],
               value=ub / norm, value_direct=rss_direct / norm, rss=ub, norm=norm, heuristic_value=ub_heur / norm,
               rank_grid=grid.tolist(), syt=bool(np.all(np.diff(grid, axis=0) > 0) and np.all(np.diff(grid, axis=1) > 0)),
               d4_rows_ordered=rows_ok, d4_cols_ordered=cols_ok, d4_additive_order=anti,
               n_ilp=P.n_ilp, t_ilp=P.t_ilp, n_nogood=len(P.nogood), bb=info, t_heur=t_heur, t=time.time() - t0)
    tag = "_secant" if (case == "V1" and BOUND == "secant") else ""
    with open(os.path.join(OUT, f"{case}_ch{k}{tag}.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"{case} ch{k}: status={info['status']} value={ub / norm:.12g} (direct {rss_direct / norm:.12g}) "
          f"heur={ub_heur / norm:.12g} SYT={out['syt']} D4={rows_ok and cols_ok and anti} ILPs={P.n_ilp} "
          f"boxes={info.get('boxes')} t={out['t']:.0f}s", flush=True)
    print("  rank grid:", grid.tolist(), flush=True)


HEURISTIC = "alternating"   # V1 runs (chains 0-9) used this; V2 uses "descent" (pass --descent)
BOUND = "dist"              # V1 runs used the dist^2 box bound; V2 uses the secant bound (pass --secant)

if __name__ == "__main__":
    if "--descent" in sys.argv:
        HEURISTIC = "descent"
    if "--secant" in sys.argv:
        BOUND = "secant"
    run(sys.argv[1], int(sys.argv[2]))
