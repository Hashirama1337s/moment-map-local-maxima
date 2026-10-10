"""BAR-T2 E2: exact minimal-interaction split on an m x n grid by GPU branch-and-bound over partial Young tableaux.
Node = order ideal (row lengths r) filled with the k smallest eigenvalues. Every cell gets an interval [lo, hi] for its
final eigenvalue: filled -> exact; unfilled (a,b) -> rank in [k + u_ab - 1, d - (m-a)(n-b)] (u_ab = unfilled cells in the
rectangle above-left, inclusive). Bound = min over v in box of ||(I-P) v||^2 (P = additive projection), certified by the
dual: for w = (I-P) v_hat, LB = max(0, S)^2 / (4 ||w||^2), S = sum_c 2 min(w_c lo_c, w_c hi_c). Valid for ANY v_hat.
v_hat from alternating projections v <- clip(P v, lo, hi), warm-started from the parent.  Incumbent: multistart
Loizeau-Sels alternation (fixed points = D4 local minima).   Run from probe/:  py ../theory/bb.py <mode> ..."""
import sys, os, json, time, math, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "probe"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu"); F = torch.float64


def Pproj(x):  # additive projection on the last two dims
    return x.mean(-1, keepdim=True) + x.mean(-2, keepdim=True) - x.mean((-1, -2), keepdim=True)


def alternation(lam, m, n, starts=20000, iters=300, seed=0):
    """Multistart sort-fit alternation. Returns best residual (unnormalised), its rank grid, and #distinct fixed points."""
    g = torch.Generator(device=DEV).manual_seed(seed); d = m * n
    lamt = torch.as_tensor(np.sort(lam), dtype=F, device=DEV); var = lamt.var()
    f = torch.rand(starts, 1, generator=g, device=DEV, dtype=F) * 0.98 + 0.01
    al = torch.sort(torch.randn(starts, m, generator=g, device=DEV, dtype=F), 1)[0] * (f * var).sqrt()
    be = torch.sort(torch.randn(starts, n, generator=g, device=DEV, dtype=F), 1)[0] * ((1 - f) * var).sqrt()
    s = al[:, :, None] + be[:, None, :] + lamt.mean()
    prev = None
    for it in range(iters):
        order = torch.argsort(s.reshape(starts, d), 1)                 # cell order by fitted value
        rank = torch.empty_like(order); rank.scatter_(1, order, torch.arange(d, device=DEV).expand(starts, d))
        E = lamt[rank].reshape(starts, m, n); s = Pproj(E)
        if prev is not None and torch.equal(rank, prev): break
        prev = rank
    res = ((E - s) ** 2).sum((1, 2)); k = int(torch.argmin(res))
    grids = rank.reshape(starts, m, n).cpu().numpy()
    return res[k].item(), grids[k], res.cpu().numpy(), grids


def intervals(lamt, R, G, m, n):
    """R: (B, m) row lengths, G: (B, m, n) ranks (-1 unfilled). Returns lo, hi (B, m, n)."""
    d = m * n; B = R.shape[0]; k = R.sum(1)                            # (B,)
    bidx = torch.arange(n, device=DEV)
    unf_row = (bidx[None, None, :] + 1 - R[:, :, None]).clamp(min=0)   # unfilled cells in row a' with column <= b
    u = torch.cumsum(unf_row, 1)                                       # (B, m, n): rectangle [0..a] x [0..b]
    aidx = torch.arange(m, device=DEV)
    lo_rank = (k[:, None, None] + u - 1).clamp(0, d - 1)
    hi_rank = (d - (m - aidx)[:, None] * (n - bidx)[None, :]).expand(B, m, n)
    filled = G >= 0
    lo_rank = torch.where(filled, G, lo_rank); hi_rank = torch.where(filled, G, hi_rank)
    return lamt[lo_rank], lamt[hi_rank]


def bound(lo, hi, v0=None, T=60):
    v = ((lo + hi) / 2) if v0 is None else torch.minimum(torch.maximum(v0, lo), hi)
    for _ in range(T):
        v = torch.minimum(torch.maximum(Pproj(v), lo), hi)
    w = v - Pproj(v); w = w - Pproj(w); w = w - Pproj(w)              # re-project: w must lie in range(I-P)
    S = (2 * torch.minimum(w * lo, w * hi)).sum((1, 2)); W = (w ** 2).sum((1, 2))
    # floating-point safety: subtract a generous rounding margin from S; drop the bound when ||w|| is at noise level
    scale = torch.maximum(lo.abs(), hi.abs()).amax((1, 2))
    errS = 1e-11 * w.abs().sum((1, 2)) * scale + 1e-11 * (Pproj(w).abs().sum((1, 2))) * scale * 1e3
    Ssafe = (S - errS).clamp(min=0)
    good = W > 1e-20 * (scale ** 2) * lo.shape[1] * lo.shape[2]
    lb = torch.where(good, Ssafe ** 2 / (4 * W.clamp(min=1e-300)), torch.zeros_like(W))
    lb = torch.minimum(lb, W)                                          # a lower bound can never exceed the primal value
    return lb, v, W                                                    # W = primal value at v (upper estimate)


def branch_and_bound(lam, m, n, UB, grid_UB=None, batch=4096, T=60, time_cap=3600.0, eps_rel=1e-12, log_every=30.0):
    d = m * n; lamt = torch.as_tensor(np.sort(lam), dtype=F, device=DEV); N = float((lamt ** 2).sum()); eps = eps_rel * N
    # root: cell (0,0) = rank 0; if square, break transpose symmetry: rank 1 at (0,1)
    R0 = torch.zeros(1, m, dtype=torch.long, device=DEV); G0 = -torch.ones(1, m, n, dtype=torch.long, device=DEV)
    R0[0, 0] = 1; G0[0, 0, 0] = 0
    if m == n: R0[0, 0] = 2; G0[0, 0, 1] = 1
    lo, hi = intervals(lamt, R0, G0, m, n); lb, v, _ = bound(lo, hi, None, T * 5)
    stack = [(R0, G0, v, lb)]; nodes = 0; t0 = time.time(); tl = t0; best = (UB, grid_UB); open_min = []
    while stack:
        R, G, V, LB = stack.pop()
        keep = LB < best[0] - eps
        R, G, V, LB = R[keep], G[keep], V[keep], LB[keep]
        if R.shape[0] == 0: continue
        k = R.sum(1)                                                   # all nodes in a batch share k (same depth)
        # children: place rank k in row a at column R[:, a] if allowed
        ch_R, ch_G, ch_V, ch_P = [], [], [], []
        for a in range(m):
            ok = R[:, a] < n
            if a > 0: ok &= R[:, a - 1] > R[:, a]
            if not ok.any(): continue
            Rc = R[ok].clone(); Gc = G[ok].clone(); col = Rc[:, a]
            Gc[torch.arange(Gc.shape[0], device=DEV), a, col] = k[ok]; Rc[:, a] += 1
            ch_R.append(Rc); ch_G.append(Gc); ch_V.append(V[ok]); ch_P.append(LB[ok])
        Rc = torch.cat(ch_R); Gc = torch.cat(ch_G); Vc = torch.cat(ch_V); Pc = torch.cat(ch_P)
        nodes += Rc.shape[0]
        lo, hi = intervals(lamt, Rc, Gc, m, n)
        if int(k[0]) + 1 == d:                                         # leaves: exact residual
            E = lo; res = ((E - Pproj(E)) ** 2).sum((1, 2)); j = int(torch.argmin(res))
            if res[j].item() < best[0]: best = (res[j].item(), Gc[j].cpu().numpy())
            continue
        lbc, vc, _ = bound(lo, hi, Vc, T); lbc = torch.maximum(lbc, Pc)
        keep = lbc < best[0] - eps
        Rc, Gc, vc, lbc = Rc[keep], Gc[keep], vc[keep], lbc[keep]
        if Rc.shape[0] == 0: continue
        order = torch.argsort(lbc, descending=True)                   # push worst first -> best popped first
        Rc, Gc, vc, lbc = Rc[order], Gc[order], vc[order], lbc[order]
        for s0 in range(0, Rc.shape[0], batch):
            stack.append((Rc[s0:s0 + batch], Gc[s0:s0 + batch], vc[s0:s0 + batch], lbc[s0:s0 + batch]))
        now = time.time()
        if now - tl > log_every:
            tl = now; glb = min(float(x[3].min()) for x in stack) if stack else best[0]
            print(f"  t={now - t0:6.0f}s nodes={nodes:.3e} stack={len(stack)} depth={int(k[0])} UB={best[0] / N:.6e} "
                  f"open-LB={min(glb, best[0]) / N:.6e}", flush=True)
        if now - t0 > time_cap:
            glb = min(float(x[3].min()) for x in stack)
            return dict(status="CAP", UB=best[0] / N, LB=min(glb, best[0]) / N, nodes=nodes, secs=now - t0, grid=best[1])
    return dict(status="EXACT", UB=best[0] / N, LB=best[0] / N, nodes=nodes, secs=time.time() - t0, grid=best[1])


if __name__ == "__main__" and sys.argv[1] not in ("V2", "RUN2"):
    import impl_a as A, t1
    mode = sys.argv[1]
    if mode == "E2a":                                                  # 4x4 validation vs enumeration, chains 0-19
        Y = t1.syt_all(4, 4); worst = 0; rows = []
        for ch in range(20):
            lam = A.setup(1000 + ch, 2000 + ch, 4)["lam"].cpu().numpy().real; N = (lam ** 2).sum()
            ub, gub, _, _ = alternation(lam, 4, 4, 2000)
            r = branch_and_bound(lam, 4, 4, ub, gub)
            enum = t1.syt_min(np.sort(lam), 4, 4, Y)[0]; diff = abs(r["UB"] - enum); worst = max(worst, diff)
            rows.append(dict(chain=ch, enum=enum, bb=r["UB"], status=r["status"], nodes=r["nodes"], alt=ub / N, secs=round(r["secs"], 2)))
            print(json.dumps(rows[-1]), flush=True)
        print(f"E2a: max |B&B - enumeration| = {worst:.2e} -> {'PASS' if worst <= 1e-12 else 'FAIL'}")
        json.dump(rows, open(os.path.join(t1.OUT, "E2a.json"), "w"), indent=0)
    elif mode == "RUN":                                                # m n chains cap [starts]
        m, n, chains, cap = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], float(sys.argv[5])
        starts = int(sys.argv[6]) if len(sys.argv) > 6 else 50000; nq = int(round(math.log2(m * n)))
        lo_c, hi_c = map(int, chains.split("-")); out = []
        for ch in range(lo_c, hi_c + 1):
            lam = A.setup(1000 + ch, 2000 + ch, nq)["lam"].cpu().numpy().real; N = (lam ** 2).sum(); t0 = time.time()
            ub, gub, allres, _ = alternation(lam, m, n, starts)
            nfix = len(np.unique(np.round(allres / N, 12)))
            hits = int((np.abs(allres - ub) <= 1e-12 * N).sum())
            print(f"chain {ch} {m}x{n}: alternation best {ub / N:.6e} (hit by {hits}/{starts} starts; {nfix} distinct fixed-point values) {time.time() - t0:.1f}s", flush=True)
            r = branch_and_bound(lam, m, n, ub, gub, time_cap=cap)
            r.update(chain=ch, shape=f"{m}x{n}", alternation_best=ub / N, alternation_hits=hits, alternation_starts=starts,
                     distinct_fixed_points=nfix); r["grid"] = None if r["grid"] is None else np.asarray(r["grid"]).tolist()
            print(json.dumps({k: v for k, v in r.items() if k != "grid"}), flush=True); out.append(r)
            with open(os.path.join(t1.OUT, f"E2_{m}x{n}.jsonl"), "a") as fh: fh.write(json.dumps(r) + "\n")


# ---------------- two-ended variant: fill smallest ranks from the top-left AND largest from the bottom-right ----------------
def intervals2(lamt, R, S, G, m, n):
    d = m * n; B = R.shape[0]; k1 = R.sum(1); k2 = S.sum(1)
    b = torch.arange(n, device=DEV)[None, None, :]
    unf = ((b >= R[:, :, None]) & (b < n - S[:, :, None])).to(torch.long)      # (B, m, n)
    u = unf.cumsum(1).cumsum(2)                                                  # unfilled in [0..a] x [0..b]
    w = unf.flip(1).flip(2).cumsum(1).cumsum(2).flip(1).flip(2)                 # unfilled in [a..] x [b..]
    lo_rank = (k1[:, None, None] + u - 1).clamp(0, d - 1)
    hi_rank = (d - k2[:, None, None] - w).clamp(0, d - 1)
    filled = G >= 0
    return lamt[torch.where(filled, G, lo_rank)], lamt[torch.where(filled, G, hi_rank)]


def branch_and_bound2(lam, m, n, UB, grid_UB=None, batch=16384, T=40, time_cap=3600.0, eps_rel=1e-12, log_every=60.0, sumc=False):
    d = m * n; lamt = torch.as_tensor(np.sort(lam), dtype=F, device=DEV); N = float((lamt ** 2).sum()); eps = eps_rel * N
    gaps = (lamt[1:] - lamt[:-1]).cpu().numpy()
    R0 = torch.zeros(1, m, dtype=torch.long, device=DEV); S0 = torch.zeros_like(R0)
    G0 = -torch.ones(1, m, n, dtype=torch.long, device=DEV)
    R0[0, 0] = 1; G0[0, 0, 0] = 0; S0[0, m - 1] = 1; G0[0, m - 1, n - 1] = d - 1
    if m == n: R0[0, 0] = 2; G0[0, 0, 1] = 1
    lo, hi = intervals2(lamt, R0, S0, G0, m, n); lb, v, _ = bound(lo, hi, None, T * 5)
    W0 = torch.ones(1, dtype=F, device=DEV)
    stack = [(R0, S0, G0, v, lb, W0)]; nodes = 0; t0 = time.time(); tl = t0; best = (UB, grid_UB); done_w = 0.0
    while stack:
        R, S, G, V, LB, Wt = stack.pop()
        keep = LB < best[0] - eps
        done_w += float(Wt[~keep].sum())
        R, S, G, V, LB, Wt = R[keep], S[keep], G[keep], V[keep], LB[keep], Wt[keep]
        if R.shape[0] == 0: continue
        k1, k2 = int(R[0].sum()), int(S[0].sum())
        if k1 + k2 == d:                                              # (cannot happen: leaves handled below)
            continue
        low = gaps[k1 - 1] >= gaps[d - k2 - 1]                         # side with the larger next gap goes first
        chR, chS, chG, chV, chP, chW = [], [], [], [], [], []
        nch = torch.zeros(R.shape[0], dtype=F, device=DEV)
        for a in range(m):
            free = R[:, a] + S[:, a] < n
            okc = free & (((R[:, a - 1] > R[:, a]) if a > 0 else torch.ones_like(free)) if low else ((S[:, a + 1] > S[:, a]) if a < m - 1 else torch.ones_like(free)))
            nch += okc.to(F)
        for a in range(m):
            free = R[:, a] + S[:, a] < n
            if low:
                ok = free & ((R[:, a - 1] > R[:, a]) if a > 0 else torch.ones_like(free))
            else:
                ok = free & ((S[:, a + 1] > S[:, a]) if a < m - 1 else torch.ones_like(free))
            if not ok.any(): continue
            Rc, Sc, Gc = R[ok].clone(), S[ok].clone(), G[ok].clone(); idx = torch.arange(Rc.shape[0], device=DEV)
            if low: Gc[idx, a, Rc[:, a]] = k1; Rc[:, a] += 1
            else: Gc[idx, a, n - 1 - Sc[:, a]] = d - 1 - k2; Sc[:, a] += 1
            chR.append(Rc); chS.append(Sc); chG.append(Gc); chV.append(V[ok]); chP.append(LB[ok]); chW.append(Wt[ok] / nch[ok])
        Rc, Sc, Gc, Vc, Pc, Wc = map(torch.cat, (chR, chS, chG, chV, chP, chW)); nodes += Rc.shape[0]
        lo, hi = intervals2(lamt, Rc, Sc, Gc, m, n)
        if k1 + k2 + 1 == d:
            E = lo; res = ((E - Pproj(E)) ** 2).sum((1, 2)); j = int(torch.argmin(res))
            if res[j].item() < best[0]: best = (res[j].item(), Gc[j].cpu().numpy())
            done_w += float(Wc.sum()); continue
        if sumc:
            Um = (Gc < 0).to(F); SU = lamt.sum() - torch.where(Gc >= 0, lamt[Gc.clamp(min=0)], torch.zeros_like(lo)).sum((1, 2))
            lbc, vc, _ = bound_sum(lo, hi, Um, SU, Vc, T)
        else:
            lbc, vc, _ = bound(lo, hi, Vc, T)
        lbc = torch.maximum(lbc, Pc)
        keep = lbc < best[0] - eps
        done_w += float(Wc[~keep].sum())
        Rc, Sc, Gc, vc, lbc, Wc = Rc[keep], Sc[keep], Gc[keep], vc[keep], lbc[keep], Wc[keep]
        if Rc.shape[0] == 0: continue
        order = torch.argsort(lbc, descending=True)
        Rc, Sc, Gc, vc, lbc, Wc = Rc[order], Sc[order], Gc[order], vc[order], lbc[order], Wc[order]
        for s0 in range(0, Rc.shape[0], batch):
            stack.append((Rc[s0:s0 + batch], Sc[s0:s0 + batch], Gc[s0:s0 + batch], vc[s0:s0 + batch], lbc[s0:s0 + batch], Wc[s0:s0 + batch]))
        now = time.time()
        if now - tl > log_every:
            tl = now; glb = min(float(x[4].min()) for x in stack)
            print(f"  t={now - t0:6.0f}s nodes={nodes:.3e} stack={len(stack)} depth={k1 + k2} UB={best[0] / N:.6e} open-LB={min(glb, best[0]) / N:.6e} done~{100 * done_w:.4f}%", flush=True)
        if now - t0 > time_cap:
            glb = min(float(x[4].min()) for x in stack)
            return dict(status="CAP", UB=best[0] / N, LB=min(glb, best[0]) / N, nodes=nodes, secs=now - t0, grid=best[1], done_weight=done_w)
    return dict(status="EXACT", UB=best[0] / N, LB=best[0] / N, nodes=nodes, secs=time.time() - t0, grid=best[1])


if __name__ == "__main__" and sys.argv[1] in ("V2", "RUN2"):
    import impl_a as A, t1
    if sys.argv[1] == "V2":                                            # validate two-ended B&B: 4x4 vs enumeration, 4x8 vs one-ended
        Y = t1.syt_all(4, 4); worst = 0
        for ch in range(20):
            lam = A.setup(1000 + ch, 2000 + ch, 4)["lam"].cpu().numpy().real
            ub, gub, _, _ = alternation(lam, 4, 4, 200)                # weak incumbent on purpose
            r = branch_and_bound2(lam, 4, 4, ub, gub); e = t1.syt_min(np.sort(lam), 4, 4, Y)[0]; worst = max(worst, abs(r["UB"] - e))
        print(f"V2 4x4: max |B&B2 - enumeration| = {worst:.2e} -> {'PASS' if worst <= 1e-12 else 'FAIL'}", flush=True)
        prev = [json.loads(l) for l in open(os.path.join(t1.OUT, "E2_4x8.jsonl"))]; worst = 0
        for p in prev:
            lam = A.setup(1000 + p["chain"], 2000 + p["chain"], 5)["lam"].cpu().numpy().real
            ub, gub, _, _ = alternation(lam, 4, 8, 2000); r = branch_and_bound2(lam, 4, 8, ub, gub)
            worst = max(worst, abs(r["UB"] - p["UB"])); print(f"  4x8 chain {p['chain']}: one-ended nodes {p['nodes']} two-ended nodes {r['nodes']} |diff| {abs(r['UB'] - p['UB']):.1e}", flush=True)
        print(f"V2 4x8: max |two-ended - one-ended| = {worst:.2e} -> {'PASS' if worst <= 1e-12 else 'FAIL'}")
    else:                                                              # RUN2 m n chains cap starts
        m, n, chains, cap = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], float(sys.argv[5])
        starts = int(sys.argv[6]) if len(sys.argv) > 6 else 200000; nq = int(round(math.log2(m * n))); lo_c, hi_c = map(int, chains.split("-"))
        for ch in range(lo_c, hi_c + 1):
            lam = A.setup(1000 + ch, 2000 + ch, nq)["lam"].cpu().numpy().real; N = (lam ** 2).sum()
            ub, gub, allres, _ = alternation(lam, m, n, starts); hits = int((np.abs(allres - ub) <= 1e-12 * N).sum())
            print(f"chain {ch} {m}x{n}: alternation best {ub / N:.6e} (hit {hits}/{starts})", flush=True)
            r = branch_and_bound2(lam, m, n, ub, gub, time_cap=cap)
            r.update(chain=ch, shape=f"{m}x{n}", method="two-ended", alternation_best=ub / N, alternation_hits=hits, alternation_starts=starts)
            r["grid"] = None if r["grid"] is None else np.asarray(r["grid"]).tolist()
            print(json.dumps({k: v for k, v in r.items() if k != "grid"}), flush=True)
            with open(os.path.join(t1.OUT, f"E2_{m}x{n}_2end.jsonl"), "a") as fh: fh.write(json.dumps(r) + "\n")


# ---------------- stronger bound: box + fixed sum over the unfilled region ----------------
def proj_box_sum(x, lo, hi, U, SU, iters=60):
    """Project x onto {lo <= v <= hi, sum_U v = SU} (cells outside U have lo = hi). Bisection on the shift nu. Returns v, nu."""
    big = (hi - lo).amax((1, 2)) + (x.abs()).amax((1, 2)) + 1.0
    a = -big * 2; b = big * 2
    for _ in range(iters):
        nu = (a + b) / 2; v = torch.minimum(torch.maximum(x - nu[:, None, None] * U, lo), hi)
        s = (v * U).sum((1, 2)); hi_side = s > SU
        a = torch.where(hi_side, nu, a); b = torch.where(hi_side, b, nu)
    nu = (a + b) / 2; v = torch.minimum(torch.maximum(x - nu[:, None, None] * U, lo), hi)
    return v, nu


def bound_sum(lo, hi, U, SU, v0=None, T=60):
    v = ((lo + hi) / 2) if v0 is None else torch.minimum(torch.maximum(v0, lo), hi)
    for _ in range(T):
        v, nu = proj_box_sum(Pproj(v), lo, hi, U, SU, 40)
    w = v - Pproj(v); w = w - Pproj(w); w = w - Pproj(w)
    W = (w ** 2).sum((1, 2)); scale = torch.maximum(lo.abs(), hi.abs()).amax((1, 2))
    def m_of(nu):  # Lagrangian lower bound on min_{v in box, sum_U v = SU} <w, v>
        c = w - nu[:, None, None] * U
        return torch.minimum(c * lo, c * hi).sum((1, 2)) + nu * SU
    m = torch.maximum(m_of(nu), m_of(torch.zeros_like(nu) * 0 + nu * 0)) if False else m_of(nu)
    # refine nu by a few golden-section steps around the primal multiplier (any nu is valid)
    lo_n, hi_n = nu - scale, nu + scale
    for _ in range(30):
        n1 = lo_n + 0.382 * (hi_n - lo_n); n2 = lo_n + 0.618 * (hi_n - lo_n); f1, f2 = m_of(n1), m_of(n2)
        lo_n = torch.where(f1 < f2, n1, lo_n); hi_n = torch.where(f1 < f2, hi_n, n2)
    m = torch.maximum(m, m_of((lo_n + hi_n) / 2))
    errm = 1e-11 * (w.abs().sum((1, 2)) + U.sum((1, 2)) * scale) * scale * 1e3
    msafe = (m - errm).clamp(min=0)
    good = W > 1e-20 * (scale ** 2) * lo.shape[1] * lo.shape[2]
    lb = torch.where(good, msafe ** 2 / W.clamp(min=1e-300), torch.zeros_like(W))
    return torch.minimum(lb, W), v, W


def knuth_estimate(lam, m, n, UB, dives=4000, T=60, seed=0):
    """Knuth's unbiased estimator of the two-ended B&B tree size for a fixed incumbent UB (random dives)."""
    d = m * n; lamt = torch.as_tensor(np.sort(lam), dtype=F, device=DEV); N = float((lamt ** 2).sum()); eps = 1e-12 * N
    gaps = (lamt[1:] - lamt[:-1]).cpu().numpy(); g = torch.Generator(device=DEV).manual_seed(seed)
    R = torch.zeros(dives, m, dtype=torch.long, device=DEV); S = torch.zeros_like(R); G = -torch.ones(dives, m, n, dtype=torch.long, device=DEV)
    R[:, 0] = 1; G[:, 0, 0] = 0; S[:, m - 1] = 1; G[:, m - 1, n - 1] = d - 1
    if m == n: R[:, 0] = 2; G[:, 0, 1] = 1
    alive = torch.ones(dives, dtype=torch.bool, device=DEV); weight = torch.ones(dives, dtype=F, device=DEV)
    total = torch.ones(dives, dtype=F, device=DEV); V = None; per_depth = []
    k1, k2 = int(R[0].sum()), int(S[0].sum())
    while k1 + k2 < d and alive.any():
        low = gaps[k1 - 1] >= gaps[d - k2 - 1]
        cands = []
        for a in range(m):
            free = R[:, a] + S[:, a] < n
            ok = free & (((R[:, a - 1] > R[:, a]) if a > 0 else torch.ones_like(free)) if low else ((S[:, a + 1] > S[:, a]) if a < m - 1 else torch.ones_like(free)))
            cands.append(ok & alive)
        C = torch.stack(cands, 1)                                         # (dives, m) candidate rows
        # evaluate every candidate child's bound
        lbs = torch.full((dives, m), float("inf"), dtype=F, device=DEV)
        for a in range(m):
            ok = C[:, a]
            if not ok.any(): continue
            Rc, Sc, Gc = R[ok].clone(), S[ok].clone(), G[ok].clone(); idx = torch.arange(Rc.shape[0], device=DEV)
            if low: Gc[idx, a, Rc[:, a]] = k1; Rc[:, a] += 1
            else: Gc[idx, a, n - 1 - Sc[:, a]] = d - 1 - k2; Sc[:, a] += 1
            lo, hi = intervals2(lamt, Rc, Sc, Gc, m, n)
            if k1 + k2 + 1 == d: lb = ((lo - Pproj(lo)) ** 2).sum((1, 2))
            else: lb, _, _ = bound(lo, hi, None, T)
            lbs[ok, a] = lb
        surv = lbs < UB - eps; ns = surv.sum(1)
        weight = weight * ns.to(F); total = total + torch.where(alive, weight, torch.zeros_like(weight))
        alive = alive & (ns > 0)
        # pick a random surviving child
        rnd = torch.rand(dives, m, generator=g, device=DEV, dtype=F) * surv.to(F)
        pick = torch.argmax(rnd, 1); idx = torch.arange(dives, device=DEV)
        sel = alive
        if low:
            G[idx[sel], pick[sel], R[idx[sel], pick[sel]]] = k1; R[idx[sel], pick[sel]] += 1; k1 += 1
        else:
            G[idx[sel], pick[sel], n - 1 - S[idx[sel], pick[sel]]] = d - 1 - k2; S[idx[sel], pick[sel]] += 1; k2 += 1
        per_depth.append(float(weight[alive].mean()) if alive.any() else 0.0)
    est = total.mean().item(); se = total.std().item() / math.sqrt(dives)
    return est, se, per_depth
