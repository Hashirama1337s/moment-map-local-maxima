"""BAR-D4 T1/T2/T4: k-partite Hessian at assignments vs prediction; local-min classification; exact full 4-qubit minimum vs
optimiser on U(16).  Run from probe/:  py ../theory/door4_hess.py T1 | T2 | T4"""
import sys, os, json, time, math, itertools, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "probe"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import impl_a as A
from torch.autograd.functional import hessian
DEV, CT, RT = A.DEV, A.CT, A.RT
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def P1(Hp, dims):
    """Projection onto 1-local operators for factors dims (k-partite)."""
    d = Hp.shape[0]; k = len(dims); c = torch.trace(Hp) / d; T = Hp.reshape(*dims, *dims); out = c * A.eye(d)
    letters = "abcdefgh"; up = letters[:k]; lo = "ijklmnop"[:k]
    for j in range(k):
        # partial trace over all factors except j
        s_in = "".join(up[q] if q == j else "z" + str(q) for q in range(k))  # placeholder, built below
        idx_in = [up[q] for q in range(k)] + [lo[q] if q == j else up[q] for q in range(k)]
        red = torch.einsum("".join(idx_in) + "->" + up[j] + lo[j], T) / (d / dims[j])
        red = red - c * torch.eye(dims[j], dtype=CT, device=DEV)
        mats = [torch.eye(dq, dtype=CT, device=DEV) for dq in dims]; mats[j] = red
        M = mats[0]
        for q in range(1, k): M = torch.kron(M, mats[q])
        out = out + M
    return out


def P1b(Hp, dims):
    """Batched P1: Hp (B, d, d)."""
    B, d, _ = Hp.shape; k = len(dims); c = torch.diagonal(Hp, dim1=-2, dim2=-1).sum(-1) / d
    T = Hp.reshape(B, *dims, *dims); out = c[:, None, None] * A.eye(d)[None]
    up = "abcdefgh"[:k]; lo = "ijklmnop"[:k]
    for j in range(k):
        idx_in = "z" + "".join(up) + "".join(lo[q] if q == j else up[q] for q in range(k))
        red = torch.einsum(idx_in + "->z" + up[j] + lo[j], T) / (d / dims[j]) - c[:, None, None] * torch.eye(dims[j], dtype=CT, device=DEV)[None]
        left = int(np.prod(dims[:j])); right = int(np.prod(dims[j + 1:]))
        M = torch.kron(torch.eye(left, dtype=CT, device=DEV), torch.ones(1, 1, dtype=CT, device=DEV))
        full = torch.einsum("pq,zab,rs->zparqbs", torch.eye(left, dtype=CT, device=DEV), red, torch.eye(right, dtype=CT, device=DEV)).reshape(B, d, d)
        out = out + full
    return out


def costb(Hp, dims, N):
    return ((Hp - P1b(Hp, dims)).abs() ** 2).sum((-1, -2)) / N


def cost(Hp, dims, N):
    return (torch.linalg.norm(Hp - P1(Hp, dims)) ** 2 / N).real


def gauge_basis(dims):
    d = int(np.prod(dims)); vecs = []
    for j, dj in enumerate(dims):
        for Lg in A.ggm(dj):
            mats = [A.eye(dq) for dq in dims]; mats[j] = Lg; M = mats[0]
            for q in range(1, len(dims)): M = torch.kron(M, mats[q])
            vecs.append((M.real + M.imag).reshape(-1))
    vecs.append(A.eye(d).real.reshape(-1))
    Q, _ = torch.linalg.qr(torch.stack(vecs).to(RT).T); return Q


def fit_k(E):
    """1-local additive fit of a k-dim grid E: mu + sum_j alpha_j."""
    k = E.ndim; mu = E.mean(); L = np.full(E.shape, mu)
    for j in range(k):
        ax = tuple(q for q in range(k) if q != j); m = E.mean(axis=ax) - mu
        shape = [1] * k; shape[j] = E.shape[j]; L = L + m.reshape(shape)
    return L


def predict_k(E, N):
    dims = E.shape; k = len(dims); d = E.size; L = fit_k(E); cells = list(itertools.product(*[range(x) for x in dims]))
    ev = [0.0] * (d - (sum(x - 1 for x in dims) + 1))
    for i in range(d):
        for j in range(i + 1, d):
            ci, cj = cells[i], cells[j]
            if sum(a != b for a, b in zip(ci, cj)) >= 2:
                v = 2 * (L[ci] - L[cj]) * (E[ci] - E[cj]) / N; ev += [v, v]
    for j in range(k):
        others = [range(x) for q, x in enumerate(dims) if q != j]
        for a in range(dims[j]):
            for a2 in range(a + 1, dims[j]):
                Delta = []
                for rest in itertools.product(*others):
                    c1 = list(rest); c1.insert(j, a); c2 = list(rest); c2.insert(j, a2); Delta.append(E[tuple(c1)] - E[tuple(c2)])
                D = np.array(Delta); m = len(D); M = D.mean() * np.diag(D) - np.outer(D, D) / m
                Qc = np.linalg.qr(np.c_[np.ones(m), np.eye(m)[:, :m - 1]])[0][:, 1:]
                for v in 2 * np.linalg.eigvalsh(Qc.T @ M @ Qc) / N: ev += [v, v]
    return np.sort(np.array(ev))


def autograd_at(H, U0, dims, N, Q, C):
    d = U0.shape[0]
    def f(P):
        U = U0 @ A.expiK(A.herm(P)); return cost(U.conj().T @ H @ U, dims, N)
    P0 = torch.zeros(d, d, dtype=RT, device=DEV, requires_grad=True); val = f(P0); g = torch.autograd.grad(val, P0)[0].reshape(-1)
    Hs = hessian(f, torch.zeros(d, d, dtype=RT, device=DEV), vectorize=True).reshape(d * d, d * d)
    w = torch.linalg.eigvalsh(C.T @ ((Hs + Hs.T) / 2) @ C).cpu().numpy()
    return val.item(), (g - Q @ (Q.T @ g)).norm().item(), np.sort(w)


def gue(d, seed):
    rng = np.random.default_rng(seed); G = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)); H = (G + G.conj().T) / 2
    w, V = np.linalg.eigh(H); return A.T(H), w, A.T(V)


def setup_dims(dims):
    Q = gauge_basis(dims); d = int(np.prod(dims)); Qf, _ = torch.linalg.qr(Q, mode="complete"); return Q, Qf[:, Q.shape[1]:]


if __name__ == "__main__":
    mode = sys.argv[1]; rng = np.random.default_rng(77)
    if mode == "T1":
        worst = 0; maxg = 0; rows = []
        for dims in ((2, 2, 2), (2, 2, 3)):
            d = int(np.prod(dims)); Q, C = setup_dims(dims)
            for s in range(5):
                H, w, V = gue(d, 500 + s); N = float((w ** 2).sum())
                for r in range(4):
                    perm = rng.permutation(d) if r < 2 else np.argsort(np.argsort(fit_k(rng.standard_normal(dims)).reshape(-1)))  # 2 random + 2 'coherent-like'
                    U0 = V[:, torch.as_tensor(perm, device=DEV)]; E = w[perm].reshape(dims)
                    val, g, wa = autograd_at(H, U0, dims, N, Q, C); p = predict_k(E, N)
                    rel = np.abs(wa - p).max() / np.abs(p).max(); worst = max(worst, rel); maxg = max(maxg, g)
                    rows.append(dict(dims=dims, seed=s, r=r, rel=rel, grad=g, nneg=int((wa < -1e-10).sum()), nneg_pred=int((p < -1e-10).sum())))
        print(f"T1: max rel Hessian diff {worst:.2e}, max grad {maxg:.2e} -> {'PASS' if worst < 1e-8 and maxg < 1e-9 else 'FAIL'}")
        json.dump(rows, open(os.path.join(OUT, "D4_T1.json"), "w"), indent=0, default=str)
    elif mode == "T2":
        dims = (2, 2, 2); d = 8; Q, C = setup_dims(dims); syms = []
        for perm3 in itertools.permutations(range(3)):
            for flips in itertools.product((0, 1), repeat=3): syms.append((perm3, flips))
        def canon(p):  # p: eigen-index per cell (cell = bits b0 b1 b2) -> canonical under axis perms and flips
            keys = []
            for perm3, flips in syms:
                q = np.empty(8, int)
                for m in range(8):
                    bits = [(m >> j) & 1 for j in range(3)]; nb = [bits[perm3[j]] ^ flips[j] for j in range(3)]
                    q[sum(b << j for j, b in enumerate(nb))] = p[m]
                keys.append(tuple(q))
            return min(keys)
        classes = sorted({canon(np.array(p)) for p in itertools.permutations(range(8))}); print("2x2x2 assignment classes:", len(classes), flush=True)
        agree = tot = 0; nmin = []
        for s in range(20):
            H, w, V = gue(d, 900 + s); N = float((w ** 2).sum()); cnt = 0
            for p in classes:
                p = np.array(p); U0 = V[:, torch.as_tensor(p, device=DEV)]; E = w[p].reshape(2, 2, 2, order="F")
                # cell index m = b0 + 2 b1 + 4 b2 ; reshape with Fortran order so E[b0,b1,b2] = w[p[m]]
                _, _, wa = autograd_at(H, U0, dims, N, Q, C)
                nz = np.sort(np.abs(wa))[d - 4:]
                is_min = bool((wa > -1e-10).all() and nz.min() > 1e-10)
                L = fit_k(E); iso = bool(np.array_equal(np.argsort(E.reshape(-1)), np.argsort(L.reshape(-1))))
                agree += (is_min == iso); tot += 1; cnt += is_min
            nmin.append(cnt); print(f"  spectrum {s}: local minima {cnt}, agreement so far {agree}/{tot}", flush=True)
        print(f"T2: autograd local-min == order-isomorphism in {agree}/{tot} -> {'PASS' if agree == tot else 'FAIL'}; local minima per spectrum {nmin} (bound: A009997(3) = 2)")
    elif mode == "T4":
        import door4 as D4
        ranks = np.load(os.path.join(OUT, "coherent_n4.npy")); dims = (2, 2, 2, 2); rows = []; worst = None
        for kind, k in [("chain", c) for c in range(20)] + [("GUE", s) for s in range(100)]:
            if kind == "chain":
                st = A.setup(1000 + k, 2000 + k, 4); H = st["H"]; w = st["lam"].cpu().numpy().real; V = st["V"]
            else:
                H, w, V = gue(16, 3000 + k)
            N = float((w ** 2).sum()); cmin, rank, call, iso = D4.exact_full_qubit(w, ranks, 4)
            # instrument check of the exact value: place eigenvalues by rank (cell m = bitmask, factor j = bit j)
            perm = np.argsort(np.argsort(np.zeros(16)))  # placeholder
            order = np.empty(16, int); order[:] = rank.astype(int)                 # rank[mask] = eigen index (sorted)
            # torch kron ordering: factor 0 is the most significant -> cell index c = sum bit_j * 2^(3-j)
            cells = [sum(((m >> j) & 1) << (3 - j) for j in range(4)) for m in range(16)]
            colidx = np.empty(16, int)
            for m in range(16): colidx[cells[m]] = order[m]
            U0 = V[:, torch.as_tensor(colidx, device=DEV)]
            c_inst = cost(U0.conj().T @ H @ U0, dims, N).item()
            row = dict(kind=kind, k=k, exact=float(cmin), instrument=c_inst, n_local_minima=int(iso.sum()))
            if kind == "chain" or k < 20:                                           # optimiser comparison (100 starts)
                g = torch.Generator(device=DEV).manual_seed(k)
                P = (0.5 * torch.randn(100, 16, 16, generator=g, device=DEV, dtype=RT)).requires_grad_(True)
                herm_b = lambda P: (P + P.transpose(-1, -2)) / 2 + 1j * (P - P.transpose(-1, -2)) / 2
                Ub = lambda P: torch.linalg.matrix_exp(1j * herm_b(P).to(CT))
                opt = torch.optim.Adam([P], lr=0.05)
                for _ in range(600):
                    opt.zero_grad(); U = Ub(P); f = costb(U.conj().transpose(-1, -2) @ H @ U, dims, N).sum(); f.backward(); opt.step()
                lb = torch.optim.LBFGS([P], lr=1, max_iter=500, tolerance_grad=1e-13, tolerance_change=1e-18, line_search_fn="strong_wolfe")
                def cl():
                    lb.zero_grad(); U = Ub(P); f = costb(U.conj().transpose(-1, -2) @ H @ U, dims, N).sum(); f.backward(); return f
                lb.step(cl)
                with torch.no_grad(): U = Ub(P); vals = costb(U.conj().transpose(-1, -2) @ H @ U, dims, N)
                best = vals.min().item(); row["optimiser_hits_exact"] = int((vals - cmin).abs().le(1e-9).sum())
                row["optimiser_best"] = best; gap = best - cmin; worst = gap if worst is None else min(worst, gap)
            rows.append(row); print(json.dumps(row), flush=True)
        print(f"T4: min over runs of (optimiser - exact) = {worst:.3e} -> {'PASS' if worst > -1e-9 else 'FAIL (K3 killed)'}")
        json.dump(rows, open(os.path.join(OUT, "D4_T4.json"), "w"), indent=0)
