"""Exact enumeration of the orderings of {0,1}^n induced by generic linear functionals u
(regions of the arrangement {d.u = 0 : d in {-1,0,1}^n \ 0}) inside the Weyl chamber
u_1 >= u_2 >= ... >= u_n >= 0.

Method: depth-first incremental double description, integer arithmetic only.
  * a region is a pointed polyhedral cone kept as its list of extreme rays (primitive integer vectors);
  * a region is split by the first hyperplane that has extreme rays strictly on both sides; the two
    children get the old rays of their side, the rays on the hyperplane, and one new ray
    (d.p) q - (d.q) p for every ADJACENT pair (p on + side, q on - side);
  * adjacency = combinatorial DD test: p, q adjacent  iff  no other extreme ray r has
    Z(r) >= Z(p) & Z(q), with Z(.) the set of already-processed hyperplanes (all valid for the cone) through the ray;
  * a leaf (no remaining hyperplane cuts it) is a region of the full arrangement; its interior point is the sum
    of its extreme rays, and the ordering of the 2^n subset sums at that point is the region's ordering.

Completeness certificate (run on every leaf): for every facet of every region that is not a chamber wall,
the ordering just across the facet is computed exactly (sort by (u_f . s, -side * d . s)) and must be an
enumerated ordering. The region graph inside the (convex) chamber is connected, so closure under facet
crossing implies every region was found.
Facets are found combinatorially (maximal ray sets among hyperplane faces) AND by an SVD rank test; the two
must agree.

Usage: python regions.py N   -> writes regions_n{N}.npz
"""
import itertools
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def hyperplanes(n):
    walls = []
    for i in range(n - 1):
        d = [0] * n
        d[i], d[i + 1] = 1, -1
        walls.append(tuple(d))
    d = [0] * n
    d[n - 1] = 1
    walls.append(tuple(d))
    rest = []
    for d in itertools.product((-1, 0, 1), repeat=n):
        if not any(d):
            continue
        first = next(x for x in d if x != 0)
        if first < 0 or d in walls:
            continue
        rest.append(d)
    return np.array(walls + rest, dtype=np.int64), len(walls)


def chamber_rays(n):
    return np.array([[1] * k + [0] * (n - k) for k in range(1, n + 1)], dtype=np.int64)


def enumerate_regions(n, verbose=True, certify=True):
    Dall, nw = hyperplanes(n)
    R0 = chamber_rays(n)
    P0 = R0 @ Dall.T
    cuts = (P0 > 0).any(0) & (P0 < 0).any(0)
    keep = np.zeros(len(Dall), bool)
    keep[:nw] = True
    keep |= cuts
    D = Dall[keep]
    H = len(D)
    DT = D.T.copy()
    S = ((np.arange(2 ** n)[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1).astype(np.int64)  # (N, n) bits

    stack = [(R0, P0[:, keep], 0)]
    orders = []
    interiors = []
    nrays = []
    nfacets = []
    neigh = []  # list of uint8 arrays (k, N)
    n_nodes = 0
    t0 = time.time()
    svd_disagree = 0
    while stack:
        rays, P, k = stack.pop()
        n_nodes += 1
        sub = P[:, k:]
        both = (sub > 0).any(0) & (sub < 0).any(0)
        idx = np.flatnonzero(both)
        if len(idx) == 0:
            # ---- leaf: a region of the full arrangement
            u = rays.sum(0)
            vals = S @ u
            order = np.argsort(vals, kind="stable")
            sv = vals[order]
            assert np.all(np.diff(sv) > 0), "tie at interior point"
            assert np.all(u[:-1] > u[1:]) and u[-1] > 0, "interior point not in open chamber"
            orders.append(order.astype(np.uint8))
            interiors.append(u)
            nrays.append(len(rays))
            if certify:
                Z = (P == 0)  # m x H
                cnt = Z.sum(0)
                cand = np.flatnonzero(cnt >= n - 1)
                Zc = Z[:, cand].T.astype(np.int64)  # c x m
                # rays in h but not in h'
                notin = Zc @ (1 - Zc).T  # c x c ; notin[a,b] = #rays in a not in b
                incl = (notin == 0)  # rays_a subset of rays_b
                eq = incl & incl.T
                strict = incl & ~eq
                maximal = ~strict.any(1)
                fac_comb = cand[maximal]
                # SVD rank check
                fac_svd = []
                for h in cand:
                    A = rays[Z[:, h]].astype(float)
                    sv_ = np.linalg.svd(A, compute_uv=False)
                    r = int((sv_ > 1e-9 * sv_[0]).sum())
                    gap_ok = (sv_[sv_ > 1e-9 * sv_[0]].min() > 1e-6 * sv_[0]) if r > 0 else True
                    if not gap_ok:
                        svd_disagree += 1
                    if r == n - 1:
                        fac_svd.append(h)
                if sorted(fac_svd) != sorted(fac_comb.tolist()):
                    svd_disagree += 1
                nfacets.append(len(fac_comb))
                side = np.sign(P.sum(0))  # sign of d . u on this region
                rows = []
                for h in fac_comb:
                    if h < nw:
                        continue  # chamber wall
                    uf = rays[Z[:, h]].sum(0)
                    d = D[h]
                    key1 = S @ uf
                    key2 = -side[h] * (S @ d)
                    o2 = np.lexsort((key2, key1))
                    k1 = key1[o2]
                    k2 = key2[o2]
                    same = (np.diff(k1) == 0) & (np.diff(k2) == 0)
                    assert not same.any(), "tie across facet"
                    rows.append(o2.astype(np.uint8))
                if rows:
                    neigh.append(np.array(rows))
            if verbose and len(orders) % 20000 == 0:
                print(f"  n={n}: {len(orders)} regions, {n_nodes} nodes, {time.time() - t0:.1f}s", flush=True)
            continue
        h = k + int(idx[0])
        s = P[:, h]
        pos = np.flatnonzero(s > 0)
        neg = np.flatnonzero(s < 0)
        zer = np.flatnonzero(s == 0)
        Z = (P[:, :h] == 0)
        Zp = Z[pos]
        Zn = Z[neg]
        common = Zp[:, None, :] & Zn[None, :, :]
        cnt = common.sum(-1)
        ca, cb = np.nonzero(cnt >= n - 2)
        new = []
        if len(ca):
            C = common[ca, cb].astype(np.int64)  # K x h
            M = C @ (1 - Z.astype(np.int64)).T  # K x m ; 0 => ray contains common set
            nz = (M == 0).sum(1)  # includes p and q themselves
            adj = np.flatnonzero(nz == 2)
            for t in adj:
                p = rays[pos[ca[t]]]
                q = rays[neg[cb[t]]]
                w = s[pos[ca[t]]] * q - s[neg[cb[t]]] * p
                g = np.gcd.reduce(np.abs(w))
                new.append(w // g)
        if new:
            W = np.array(new, dtype=np.int64)
            PW = W @ DT
            assert np.all(PW[:, h] == 0)
        else:
            W = np.zeros((0, n), np.int64)
            PW = np.zeros((0, H), np.int64)
        ip = np.concatenate([pos, zer])
        im = np.concatenate([neg, zer])
        stack.append((np.vstack([rays[ip], W]), np.vstack([P[ip], PW]), h + 1))
        stack.append((np.vstack([rays[im], W]), np.vstack([P[im], PW]), h + 1))
    orders = np.array(orders)
    out = dict(orders=orders, interiors=np.array(interiors), nrays=np.array(nrays), n_nodes=n_nodes,
               n_hyperplanes_cutting=int(cuts.sum()), n_hyperplanes_total=len(Dall), time=time.time() - t0)
    if certify:
        allneigh = np.vstack(neigh) if neigh else np.zeros((0, 2 ** n), np.uint8)
        vt = np.dtype((np.void, 2 ** n))
        oset = np.ascontiguousarray(orders).view(vt).ravel()
        nset = np.ascontiguousarray(allneigh).view(vt).ravel()
        uniq = np.unique(oset)
        found = np.isin(nset, uniq)
        out.update(n_unique_orders=len(uniq), n_neighbor_checks=len(nset), n_neighbor_missing=int((~found).sum()),
                   nfacets=np.array(nfacets), svd_disagree=svd_disagree)
    return out


if __name__ == "__main__":
    n = int(sys.argv[1])
    res = enumerate_regions(n)
    print(f"n={n}: regions in chamber = {len(res['orders'])}  unique = {res.get('n_unique_orders')}  "
          f"DFS nodes = {res['n_nodes']}  cutting hyperplanes = {res['n_hyperplanes_cutting']}/{res['n_hyperplanes_total']}  "
          f"time = {res['time']:.1f}s")
    if 'n_neighbor_checks' in res:
        print(f"  completeness: neighbor orderings checked = {res['n_neighbor_checks']}, missing = {res['n_neighbor_missing']}, "
              f"facet test comb-vs-SVD disagreements = {res['svd_disagree']}")
        print(f"  rays per region: min {res['nrays'].min()} max {res['nrays'].max()}; "
              f"facets per region: min {res['nfacets'].min()} max {res['nfacets'].max()}")
    np.savez_compressed(os.path.join(HERE, f"regions_n{n}.npz"), orders=res['orders'], interiors=res['interiors'],
                        nrays=res['nrays'])
