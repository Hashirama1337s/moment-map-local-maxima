"""BAR-D13C (sealed theory/BAR-D13C.md): door 13 cold-limit law, exact high-precision low-temperature optima.
MI(class) = ln Z - (sum_a R_a ln R_a + sum_b C_b ln C_b - sum_k q_k ln q_k) / Z with q_k = exp(-beta e_k), R/C = row/col sums of q.
Run:  py -3.11 theory/d13_cold.py            (writes results/d13/cold.json)"""
import sys, os, json, math, time, numpy as np
import mpmath as mp
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "d13")


class BM:                                              # verbatim copies (lab modules bmargin.syt / bmargin.ising_spectrum)
    @staticmethod
    def syt(m, n):
        out = []; grid = -np.ones((m, n), int)
        def rec(k):
            if k == m * n: out.append(grid.copy()); return
            for a in range(m):
                for b in range(n):
                    if grid[a, b] < 0 and (a == 0 or grid[a - 1, b] >= 0) and (b == 0 or grid[a, b - 1] >= 0):
                        grid[a, b] = k; rec(k + 1); grid[a, b] = -1
        rec(0); return np.array(out)

    @staticmethod
    def ising_spectrum(nsites, seed):
        """4-site mixed-field Ising chain: sum Z_i Z_{i+1} + sum (0.9045 X_i + (0.809 + U(-0.1, 0.1)) Z_i)."""
        rng = np.random.default_rng(seed); X = np.array([[0, 1], [1, 0]], float); Z = np.diag([1.0, -1.0]); I = np.eye(2)
        def op(o, i):
            M = np.array([[1.0]])
            for j in range(nsites): M = np.kron(M, o if j == i else I)
            return M
        H = sum(op(Z, i) @ op(Z, i + 1) for i in range(nsites - 1)) + sum(0.9045 * op(X, i) + (0.809 + rng.uniform(-0.1, 0.1)) * op(Z, i) for i in range(nsites))
        return np.linalg.eigvalsh(H)


class D:                                               # verbatim copy (lab module d8_exact.gue_spectrum)
    @staticmethod
    def gue_spectrum(d, seed):
        """GUE with <|H_ij|^2> = 1/d, traceless; numpy default_rng(seed) stream."""
        rng = np.random.default_rng(seed)
        X = (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) / np.sqrt(d); H = 0.5 * (X + X.conj().T)
        H -= np.trace(H).real / d * np.eye(d)
        return np.linalg.eigvalsh(H)

def classes(m, n):                                     # identical to d13_thermal.py
    T = BM.syt(m, n); idx = T.reshape(len(T), -1)
    if m == n:
        seen = {}; keep = []
        for i, t in enumerate(T):
            k = min(tuple(t.ravel()), tuple(t.T.ravel()))
            if k not in seen: seen[k] = i; keep.append(i)
        idx = idx[keep]
    return idx


def spectrum(fam, d, s):
    E = np.sort(D.gue_spectrum(d, 70000 + s)) if fam == "GUE" else np.sort(BM.ising_spectrum(int(np.log2(d)), 80000 + s))
    return (E - E.mean()) / E.std()


def exact_mi(E, idx, m, n, beta, extra=0):
    e = E - E.min(); mp.mp.dps = 40 + math.ceil(beta * (E.max() - E.min()) / math.log(10)) + extra
    q = [mp.exp(-mp.mpf(beta) * mp.mpf(float(v))) for v in e]; Z = mp.fsum(q); lnZ = mp.log(Z)
    qlq = mp.fsum(x * mp.log(x) for x in q); memo = {}
    def xlx(ranks):
        key = frozenset(ranks)
        if key not in memo: s = mp.fsum(q[r] for r in ranks); memo[key] = s * mp.log(s)
        return memo[key]
    out = []
    for row in idx:
        g = row.reshape(m, n); S = mp.fsum(xlx(g[a]) for a in range(m)) + mp.fsum(xlx(g[:, b]) for b in range(n))
        out.append(lnZ - (S - qlq) / Z)
    lead = []; mb = mp.mpf(beta); em = [mp.mpf(float(v)) for v in e]; tmemo = {}
    def term(k, ka, kb):                               # leading-order term for cell ranks (ab, a0, 0b)
        key = (k, ka, kb)
        if key not in tmemo:
            pab, pa0, p0b = q[k] / Z, q[ka] / Z, q[kb] / Z
            tmemo[key] = pab * (-mb * (em[k] - em[ka] - em[kb]) + lnZ) - pab + pa0 * p0b
        return tmemo[key]
    for row in idx:
        g = row.reshape(m, n)
        lead.append(mp.fsum(term(g[a, b], g[a, 0], g[0, b]) for a in range(1, m) for b in range(1, n)))
    return out, lead


def float_mi(E, idx, m, n, beta):                     # door-13 float formula (d13_thermal.py)
    ME = E[idx].reshape(-1, m, n); w = np.exp(-beta * (ME - E.min())); w /= w.sum((1, 2), keepdims=True)
    H = lambda p, ax: -(p * np.log(np.clip(p, 1e-300, None))).sum(axis=ax)
    return H(w.sum(2), 1) + H(w.sum(1), 1) - H(w.reshape(len(w), -1), 1)


def Xval(E, idx, m, n):
    e = E - E.min(); g = idx.reshape(-1, m, n)
    return np.minimum(e[g[:, 1, 1]], e[g[:, 1, 0]] + e[g[:, 0, 1]]), e[g[:, 1, 1]] < e[g[:, 1, 0]] + e[g[:, 0, 1]]


def run_set(m, n, fam, cnt, betas=(50, 100, 200)):
    idx = classes(m, n); rows = []; t0 = time.time(); c1_worst = 0.0
    for s in range(cnt):
        E = spectrum(fam, m * n, s); X, binds = Xval(E, idx, m, n)
        aX = set(np.where(X >= X.max() - 1e-12)[0].tolist()); rec = dict(seed=s, nX=len(aX), classes=len(idx))
        if s < 5:                                       # C1
            ex, _ = exact_mi(E, idx, m, n, 1.0); fl = float_mi(E, idx, m, n, 1.0)
            c1_worst = max(c1_worst, max(abs(float(a) - b) / max(abs(float(a)), 1e-300) for a, b in zip(ex, fl)))
        for beta in betas:
            ex, ld = exact_mi(E, idx, m, n, beta); order = sorted(range(len(idx)), key=lambda i: ex[i]); w, r2 = order[0], order[1]
            wl = min(range(len(idx)), key=lambda i: ld[i])
            rec[f"win_{beta}"] = int(w); rec[f"lead_win_{beta}"] = int(wl)
            if beta == 200:
                ex2, _ = exact_mi(E, idx[[w, r2]], m, n, beta, extra=60)          # C2
                gap = (ex[r2] - ex[w]) / ex[w]; dps = 40 + math.ceil(beta * (E.max() - E.min()) / math.log(10))
                tie = (not ex2[0] < ex2[1]) or gap < mp.mpf(10) ** (-(dps - 20))      # Amendment 1 tie rule
                rec.update(c2_same=bool(ex2[0] < ex2[1]), c2_relgap=float(gap), log10_gap=float(mp.log10(gap)) if gap > 0 else None,
                           dps=dps, tie=bool(tie),
                           inX=bool(w in aX), lead_ok=bool(wl == w), ratio=float(ex[w] / ld[w]), mi=float(ex[w]),
                           e11_binds=bool(binds[w]), X_win=float(X[w]), X_max=float(X.max()))
        rec["float_cold_100"] = int(np.argmin(float_mi(E, idx, m, n, 100.0)))
        rows.append(rec)
    nt = [r for r in rows if not r["tie"]]
    summ = dict(set=f"{m}x{n}_{fam}", count=cnt, classes=len(idx), ties=len(rows) - len(nt), C1_worst_rel=c1_worst,
                C2_all_same=all(r["c2_same"] for r in rows),
                P1_frac=float(np.mean([r["inX"] for r in nt])), P2_frac=float(np.mean([r["lead_ok"] for r in nt])),
                P3_frac=float(np.mean([abs(r["ratio"] - 1) < 0.1 for r in rows])),
                chance_baseline=float(np.mean([r["nX"] / r["classes"] for r in rows])),
                e11_binds_frac=float(np.mean([r["e11_binds"] for r in rows])),
                stable_50_100_200=float(np.mean([r["win_50"] == r["win_100"] == r["win_200"] for r in rows])),
                float_cold100_agrees=float(np.mean([r["float_cold_100"] == r["win_200"] for r in rows])),
                secs=round(time.time() - t0, 1))
    print(json.dumps(summ), flush=True); return summ, rows


if __name__ == "__main__":
    sets = [(2, 4, "GUE", 200), (3, 3, "GUE", 200), (4, 4, "GUE", 50), (4, 4, "ISING", 50)]
    if len(sys.argv) > 1 and sys.argv[1] == "PROBE": sets = [(2, 4, "GUE", 5), (4, 4, "GUE", 1)]
    # C3: inputs equal the stored blind inputs
    c3 = {}
    if sets[-1][2] == "ISING":
        for fam in ("GUE", "ISING"):
            B = np.load(os.path.join(OUT, f"blind_inputs_4x4_{fam}.npy"))
            c3[fam] = float(max(np.abs(spectrum(fam, 16, s) - B[s]).max() for s in range(50)))
        print("C3", c3, flush=True)
    res = {}
    for m, n, fam, cnt in sets:
        summ, rows = run_set(m, n, fam, cnt); res[summ["set"]] = dict(summary=summ, rows=rows)
    if sets[-1][2] == "ISING":
        json.dump(dict(C3=c3, sets=res), open(os.path.join(OUT, "cold.json"), "w"), indent=1)
