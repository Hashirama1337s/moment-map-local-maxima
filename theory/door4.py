"""BAR-D4 (sealed theory/BAR-D4.md): minimal interaction for full qubit factorizations.
Coherent Boolean term orders = orderings of subset sums of weights 0 < w1 < ... < wn (OEIS A009997).
Run from probe/:  py ../theory/door4.py ENUM n | T12 | T4 | T5"""
import sys, os, json, time, math, itertools, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "probe"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.optimize import linprog
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def popsum(mask, n):
    return np.array([(mask >> j) & 1 for j in range(n)], float)


def enumerate_coherent(n, log_every=20.0):
    """All coherent orders of the 2^n subset sums with 0 < w1 < ... < wn, as rank arrays (rank[mask]).
    DFS on the lower half (sums < W/2), LP feasibility with witness reuse."""
    N = 2 ** n; half = N // 2; full = N - 1
    vec = [popsum(m, n) for m in range(N)]
    # dominance: T is 'obviously smaller' than S if T = S minus an element, or S with one element replaced by a smaller one
    def preds(S):
        out = []
        for j in range(n):
            if (S >> j) & 1:
                out.append(S & ~(1 << j))
                for i in range(j):
                    if not (S >> i) & 1: out.append((S & ~(1 << j)) | (1 << i))
        return out
    P = [preds(S) for S in range(N)]
    base_A = []; base_b = []
    base_A.append(-np.eye(n)[0]); base_b.append(-1.0)                       # w1 >= 1
    for j in range(n - 1):
        r = np.zeros(n); r[j] = 1; r[j + 1] = -1; base_A.append(r); base_b.append(-1.0)   # w_{j+1} - w_j >= 1
    allw = np.ones(n)
    results = []; t0 = time.time(); tl = t0; nodes = [0]; lps = [0]
    def feasible(A, b, w):
        if w is not None and np.all(np.array(A) @ w <= np.array(b) + 1e-9): return w
        lps[0] += 1
        r = linprog(np.zeros(n), A_ub=np.array(A), b_ub=np.array(b), bounds=[(None, None)] * n, method="highs")
        return r.x if r.status == 0 else None
    def dfs(prefix, placed, A, b, w):
        nonlocal tl
        nodes[0] += 1
        if len(prefix) == half:
            rank = np.empty(N, dtype=np.int16)
            for k, S in enumerate(prefix): rank[S] = k; rank[full ^ S] = N - 1 - k
            results.append(rank); return
        cands = [S for S in range(N) if not placed[S] and all(placed[T] for T in P[S])]
        for S in cands:
            if placed[full ^ S]: continue
            A2 = list(A); b2 = list(b)
            if prefix:                                                     # sum(S) - sum(prev) >= 1
                A2.append(vec[prefix[-1]] - vec[S]); b2.append(-1.0)
            for C in cands:                                                # S is the smallest remaining candidate
                if C != S: A2.append(vec[S] - vec[C]); b2.append(-1.0)
            A2.append(2 * vec[S] - allw); b2.append(-1.0)                  # 2 sum(S) <= W - 1 (lower half)
            w2 = feasible(A2, b2, w)
            if w2 is None: continue
            # keep only chain constraints downstream (candidate-min constraints are re-imposed at the next level)
            A3 = list(A); b3 = list(b)
            if prefix: A3.append(vec[prefix[-1]] - vec[S]); b3.append(-1.0)
            A3.append(2 * vec[S] - allw); b3.append(-1.0)
            placed[S] = True; dfs(prefix + [S], placed, A3, b3, w2); placed[S] = False
            if time.time() - tl > log_every:
                tl = time.time(); print(f"  n={n} t={tl - t0:.0f}s orders={len(results)} nodes={nodes[0]} LPs={lps[0]}", flush=True)
    placed = [False] * N; placed[0] = True
    dfs([0], placed, base_A, base_b, None)
    return np.array(results), dict(nodes=nodes[0], LPs=lps[0], secs=time.time() - t0)


def verify_orders(ranks, n, samples=200000, seed=0):
    """Independent check: random sorted positive weights -> induced order must be in the enumerated set."""
    rng = np.random.default_rng(seed); N = 2 ** n
    V = np.array([popsum(m, n) for m in range(N)])
    known = {r.tobytes() for r in ranks}; seen = set(); missing = 0
    for k in range(samples):
        w = np.sort(rng.exponential(1.0, n)) if k % 2 == 0 else np.sort(rng.uniform(0, 1, n))
        s = V @ w; r = np.empty(N, dtype=np.int16); r[np.argsort(s)] = np.arange(N)
        key = r.tobytes(); seen.add(key); missing += key not in known
    return len(seen), missing


def design(n):
    N = 2 ** n; Z = np.array([[1 - 2 * ((m >> j) & 1) for j in range(n)] for m in range(N)], float)
    return np.c_[np.ones(N), Z]


def exact_full_qubit(lam, ranks, n):
    """Exact global min of full n-qubit minimal interaction (K3): best coherent placement; also #local minima (order-iso)."""
    lam = np.sort(lam); X = design(n); Pm = X @ np.linalg.pinv(X)            # projection onto 1-local diagonal
    E = lam[ranks.astype(np.int64)]                                         # (orders, N)
    L = E @ Pm.T; R = E - L; cost = (R ** 2).sum(1) / (lam ** 2).sum()
    iso = np.array([np.array_equal(np.argsort(E[i]), np.argsort(L[i])) for i in range(len(E))]) if len(E) < 200000 else None
    k = int(np.argmin(cost)); return cost[k], ranks[k], cost, iso


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "ENUM":
        n = int(sys.argv[2]); ranks, info = enumerate_coherent(n)
        np.save(os.path.join(OUT, f"coherent_n{n}.npy"), ranks)
        nseen, missing = verify_orders(ranks, n, 200000 if n <= 5 else 2000000)
        print(json.dumps(dict(n=n, coherent_orders=len(ranks), **info, sampled_distinct=nseen, sampled_not_in_enumeration=missing)), flush=True)
