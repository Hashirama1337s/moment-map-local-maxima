"""Post-bar analysis (labelled as such): (a) is the I3b plant's own labeling the exact C1 global min, and does the plant lie on
that minimum's phase torus? (b) how far apart (TPS distance D) are TPSs on one torus?  Run from probe/."""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "probe"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import impl_a as A, probe as PR, t1
DEV, CT, RT = A.DEV, A.CT, A.RT


def canon(grid):
    """Canonical form of a rank grid up to row/column permutation and transpose: sort rows by first entry, columns likewise."""
    g = grid[np.argsort(grid[:, 0])]; g = g[:, np.argsort(g[0])]
    t = grid.T; t = t[np.argsort(t[:, 0])]; t = t[:, np.argsort(t[0])]
    return min(g.tobytes(), t.tobytes())


def D_torus(U, V, dA, dB, theta):
    Ph = torch.diag(torch.exp(1j * theta.to(CT)))
    return A.tps_distance(U @ Ph, V, dA, dB)


def dist_torch(U, V, dA, dB):
    """Differentiable version of the closed-form D (A5), A-side vs A-side and A-side vs B-side, min."""
    X = A.alg_basis(U, dA, dB, "A")
    def dist(Y, m):
        s = ((X.conj() @ Y.T).abs() ** 2).sum().real
        return torch.sqrt(torch.clamp(2.0 * (1.0 - s / m), min=0.0))
    return torch.minimum(dist(A.alg_basis(V, dA, dB, "A"), dA * dA), dist(A.alg_basis(V, dA, dB, "B"), dA * dA))


def opt_theta(U, V, dA, dB, sign=+1, starts=8, steps=400):
    best = None; d = U.shape[0]
    for s in range(starts):
        th = torch.tensor(np.random.default_rng(s).uniform(0, 2 * np.pi, d), dtype=RT, device=DEV, requires_grad=True)
        opt = torch.optim.Adam([th], lr=0.05)
        for _ in range(steps):
            opt.zero_grad(); val = sign * dist_torch(U @ torch.diag(torch.exp(1j * th.to(CT))), V, dA, dB); val.backward(); opt.step()
        v = dist_torch(U @ torch.diag(torch.exp(1j * th.detach().to(CT))), V, dA, dB).item()
        if best is None or sign * v < sign * best: best = v
    return best


if __name__ == "__main__":
    Y = t1.syt_all(4, 4); rows = []
    for ch, eps in ((0, 0.05), (0, 1e-3), (1, 1e-3), (2, 1e-3)):
        st = PR.setup_decoupled(7000 + ch, 4, eps); lam = st["lam"].cpu().numpy().real
        cmin, grid, nloc, c_all, is_min = t1.syt_min(lam, 4, 4, Y)
        # plant labeling: product eigenbasis of the plant's local part, matched to H's eigenvalues by perturbation order
        Hp = st["S"].conj().T @ st["H"] @ st["S"]; HA, HB, _ = A.h_parts(Hp, 4, 4)
        wA, VA = torch.linalg.eigh(HA); wB, VB = torch.linalg.eigh(HB)
        Uloc = st["S"] @ torch.kron(VA, VB)                     # plant TPS, gauge-rotated so its local part is diagonal
        ov = (st["V"].conj().T @ Uloc).abs() ** 2               # eigenvector k vs product state g
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-ov.cpu().numpy()); plant_grid = np.empty(16, int); plant_grid[c] = r
        plant_grid = plant_grid.reshape(4, 4)
        E = lam[plant_grid]; _, R = t1.fit(E); c_plant_label = (R ** 2).sum() / (lam ** 2).sum()
        same = canon(plant_grid) == canon(grid)
        Umin = t1.U_of(st, plant_grid)
        D_raw = A.tps_distance(Umin, Uloc, 4, 4); D_best = opt_theta(Umin, Uloc, 4, 4, +1)
        D_span = opt_theta(Umin, Umin, 4, 4, -1)                # how far a torus point can be from its own base point
        rank = int((c_all < c_plant_label - 1e-15).sum())
        rows.append(dict(chain=ch, eps=eps, syt_min=float(cmin), plant_label_cost=float(c_plant_label), plant_label_is_global_min=bool(same),
                         labels_below_plant_label=rank, D_plant_raw=D_raw, D_plant_best_on_torus=D_best, torus_max_D=D_span,
                         min_overlap=float(ov.cpu().numpy()[r, c].min())))
        print(json.dumps(rows[-1]), flush=True)
    json.dump(rows, open(os.path.join(t1.OUT, "torus.json"), "w"), indent=0)
