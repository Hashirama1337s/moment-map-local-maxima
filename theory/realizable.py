"""(a) check: D4 <=> the additive fit L orders all cells exactly as E (order isomorphism); (b) count realizable SYT
(orderings of outer sums u_a + v_b) by LP feasibility.  Run from probe/."""
import sys, os, numpy as np, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import t1
from scipy.optimize import linprog
def realizable(T):
    m, n = T.shape; cells = np.argsort(T.reshape(-1)); A = []; nv = m + n
    for k in range(m * n - 1):
        i, j = divmod(cells[k], n); i2, j2 = divmod(cells[k + 1], n); row = np.zeros(nv)
        row[i] += 1; row[m + j] += 1; row[i2] -= 1; row[m + j2] -= 1; A.append(row)   # u_i+v_j - u_i2 - v_j2 <= -1
    r = linprog(np.zeros(nv), A_ub=np.array(A), b_ub=-np.ones(len(A)), bounds=[(None, None)] * nv, method="highs")
    return r.status == 0
out = {}
for (m, n) in ((2, 4), (3, 3), (2, 8), (3, 4), (4, 4)):
    Y = t1.syt_all(m, n); real = np.array([realizable(T) for T in Y]); out[f"{m}x{n}"] = dict(SYT=len(Y), realizable=int(real.sum()))
    print(m, n, out[f"{m}x{n}"], flush=True)
    if (m, n) == (4, 4):
        np.save(os.path.join(t1.OUT, "realizable_4x4.npy"), real)
        # (a) order-isomorphism check + are local minima always realizable?
        rng = np.random.default_rng(3); bad = 0; nonreal_min = 0; tot = 0
        for k in range(30):
            G = rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16)); lam = np.linalg.eigvalsh((G + G.conj().T) / 2)
            E = lam[Y]; L, _ = t1.fit(E); d4 = (t1.xmin(E, L) > 0)
            flatE = E.reshape(len(Y), -1); flatL = L.reshape(len(Y), -1)
            iso = np.array([np.array_equal(np.argsort(flatE[q]), np.argsort(flatL[q])) for q in range(len(Y))])
            bad += int((iso != d4).sum()); nonreal_min += int((d4 & ~real).sum()); tot += len(Y)
        print("D4 vs order-isomorphism mismatches:", bad, "of", tot, "| local minima that are NOT realizable:", nonreal_min)
json.dump(out, open(os.path.join(t1.OUT, "realizable.json"), "w"))
