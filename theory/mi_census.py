"""Implementation A independent mutual-information census (McConnell-Jennings counts). JJR reduction: global MI minimum over the
unitary orbit is product-diagonal (Jevtic-Jennings-Rudolph); argmin over SYT placements of S(row sums) + S(col sums)."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import t1
from scipy.optimize import linprog
def realizable(T):
    m, n = T.shape; cells = np.argsort(T.reshape(-1)); A = []
    for k in range(m * n - 1):
        i, j = divmod(cells[k], n); i2, j2 = divmod(cells[k + 1], n); r = np.zeros(m + n); r[i] += 1; r[m + j] += 1; r[i2] -= 1; r[m + j2] -= 1; A.append(r)
    return linprog(np.zeros(m + n), A_ub=np.array(A), b_ub=-np.ones(len(A)), bounds=[(None, None)] * (m + n), method="highs").status == 0
def H(x): x = x[x > 0]; return -(x * np.log(x)).sum()
rng = np.random.default_rng(31415); out = {}
for (m, n) in ((2, 2), (2, 3), (2, 4), (2, 5), (3, 3)):
    Y = t1.syt_all(m, n); real = np.array([realizable(T) for T in Y]); wins = set(); nonreal = 0
    for k in range(20000):
        p = np.sort(rng.dirichlet(np.ones(m * n) * rng.choice([0.3, 1, 3])))[::-1]; P = p[Y]
        I = np.array([H(Pi.sum(1)) + H(Pi.sum(0)) for Pi in P]); j = int(np.argmin(I))
        wins.add(min(Y[j].tobytes(), Y[j].T.tobytes()) if m == n else Y[j].tobytes()); nonreal += (not real[j])
    out[f"{m}x{n}"] = dict(classes=len(wins), nonrealizable_argmins=int(nonreal), spectra=20000, realizable_classes=int(real.sum() // (2 if m == n else 1)), SYT=len(Y))
    print(f"{m}x{n}", out[f"{m}x{n}"], flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "mi_census.json"), "w"), indent=1)
