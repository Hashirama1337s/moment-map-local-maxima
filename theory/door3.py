"""Door 3 (exact Cao et al. procedure vs exact global full-qubit factorization).
Greedy: split off one qubit exactly (min-variance matching, qubit_exact), recurse on the rest spectrum (pair means),
criterion per step = C1 or the Cao ratio. Global: exact best coherent order (door4). Run from probe/: py ../theory/door3.py n chains"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "probe"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import impl_a as A, qubit_exact as Q, door4 as D4
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def greedy(lam, n, crit="C1"):
    """Returns rank-of-bitstring array: placement[mask] = index into sorted lam."""
    lam = np.sort(lam); idx = [np.array([i]) for i in range(len(lam))]  # each 'level' carries the eigen-indices it stands for
    levels = lam.copy(); groups = [[i] for i in range(len(lam))]          # group of original indices with their bit labels
    labels = {i: 0 for i in range(len(lam))}
    for q in range(n):
        if len(levels) == 1: break
        cost, best, ratio, nv, nq = Q.exact_qubit(levels)
        pairs = best[2] if crit == "C1" else None
        if crit == "ratio":
            pairs = ratio[2]                                                # exact ratio-optimal matching (hull vertex)
        new_levels, new_groups = [], []
        for (i, j) in pairs:
            lo, hi = (i, j) if levels[i] <= levels[j] else (j, i)
            for o in groups[hi]: labels[o] |= (1 << q)                    # upper member of the pair gets bit q = 1
            new_levels.append((levels[i] + levels[j]) / 2); new_groups.append(groups[lo] + groups[hi])
        order = np.argsort(new_levels); levels = np.array(new_levels)[order]; groups = [new_groups[k] for k in order]
    placement = np.empty(2 ** n, dtype=np.int64)
    for o, m in labels.items(): placement[m] = o
    return placement


def residual(lam, placement, n):
    lam = np.sort(lam); X = D4.design(n); Pm = X @ np.linalg.pinv(X); E = lam[placement]; R = E - Pm @ E
    return (R ** 2).sum() / (lam ** 2).sum()


if __name__ == "__main__":
    n, chains = int(sys.argv[1]), int(sys.argv[2]); ranks = (lambda p: np.load(p) if os.path.exists(p) else np.load(p[:-4] + ".npz")["ranks"])(os.path.join(OUT, f"coherent_n{n}.npy")); rows = []
    for ch in range(chains):
        lam = A.setup(1000 + ch, 2000 + ch, n)["lam"].cpu().numpy().real
        g_c1 = residual(lam, greedy(lam, n, "C1"), n); g_ra = residual(lam, greedy(lam, n, "ratio"), n)
        ex, rank, call, iso = D4.exact_full_qubit(lam, ranks, n)
        rank_of_greedy = int((call < g_c1 - 1e-15).sum())
        r = dict(n=n, chain=ch, exact_global=float(ex), greedy_C1=g_c1, greedy_ratio=g_ra, greedy_C1_over_exact=g_c1 / ex,
                 coherent_orders_better_than_greedy=rank_of_greedy, n_local_minima=None if iso is None else int(iso.sum()))
        rows.append(r); print(json.dumps(r), flush=True)
    json.dump(rows, open(os.path.join(OUT, f"door3_n{n}.json"), "w"), indent=0)
