"""BAR-D5LP (sealed theory/BAR-D5LP.md): door 5 as an exact LP. Can ONE sorted spectrum realise every realm at once?
Realm T holds iff the additive fit L of E = lam[T] is strictly increasing along T's rank order (published theorem / D4).
Each realm = (mn - 1) integer-scaled homogeneous strict inequalities on lam. Joint LP: max margin t; feasible iff t* > 0.
Infeasible -> Gordan certificate y >= 0, y != 0, A^T y = 0, verified in exact integers.
Floating LP (scipy HiGHS) only FINDS objects; every accepted object is re-verified with Python ints / Fractions.
Run:  py -3.11 theory/d5lp.py CONTROLS | JOINT | GREEDY | UPPER         (CPU, light)"""
import os, sys, json, time, math
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack as svstack, hstack as shstack
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "results", "d5lp"); os.makedirs(OUT, exist_ok=True)
RES = os.path.join(HERE, "results")


def syt_all(m, n):                                   # verbatim t1.syt_all (t1 imports torch; keep this light)
    out = []; T = -np.ones((m, n), int); rowlen = [0] * m
    def rec(k):
        if k == m * n: out.append(T.copy()); return
        for a in range(m):
            b = rowlen[a]
            if b < n and (a == 0 or rowlen[a - 1] > b):
                T[a, b] = k; rowlen[a] += 1; rec(k + 1); rowlen[a] -= 1; T[a, b] = -1
    rec(0); return np.array(out)


def realizable(T):                                   # as realizable.py: x_a + y_b strictly ordered as T (margin 1)
    m, n = T.shape; cells = np.argsort(T.reshape(-1)); A = []
    for k in range(m * n - 1):
        i, j = divmod(cells[k], n); i2, j2 = divmod(cells[k + 1], n); row = np.zeros(m + n)
        row[i] += 1; row[m + j] += 1; row[i2] -= 1; row[m + j2] -= 1; A.append(row)
    r = linprog(np.zeros(m + n), A_ub=np.array(A), b_ub=-np.ones(len(A)), bounds=[(None, None)] * (m + n), method="highs")
    return r.status == 0


# ------------------------------------------------------------------ realm inequality rows (integers)
def bip_fitmat(m, n):
    """Integer matrix F (mn x mn): mn * L_cell = sum_c' F[cell, c'] E_c' (cells flattened a*n + b)."""
    d = m * n; F = np.zeros((d, d), np.int64)
    for a in range(m):
        for b in range(n):
            c = a * n + b
            for b2 in range(n): F[c, a * n + b2] += m          # mn * row mean = m * row sum
            for a2 in range(m): F[c, a2 * n + b] += n          # mn * col mean = n * col sum
            F[c, :] -= 1                                        # - grand sum
    return F


def qubit_fitmat(nq):
    """Integer matrix N * Pm = X X^T with X = [1, Z_1..Z_n] (door4.design)."""
    N = 2 ** nq; X = np.array([[1] + [1 - 2 * ((mm >> j) & 1) for j in range(nq)] for mm in range(N)], np.int64)
    return X @ X.T


def realm_rows(rank, F):
    """rank: flat array cell -> rank. Returns integer rows R (d-1 x d) over lam with R lam > 0 <=> realm holds."""
    d = len(rank); cells = np.argsort(rank); P = np.zeros((d, d), np.int64); P[np.arange(d), rank] = 1   # E = P lam
    G = F @ P                                                                                             # scaled L = G lam
    return G[cells[1:]] - G[cells[:-1]]


def gap_rows(d):
    R = np.zeros((d - 1, d), np.int64)
    for k in range(d - 1): R[k, k + 1] = 1; R[k, k] = -1
    return R


def bip_realms(m, n):
    Y = syt_all(m, n); keep = []
    for T in Y:
        if m == n and T[0, 1] != 1: continue           # transpose classes: canonical rep has rank 1 at (0, 1)
        if realizable(T): keep.append(T.reshape(-1))
    return np.array(keep), len(Y)


def qubit_realms(nq):
    return np.load(os.path.join(RES, f"coherent_n{nq}.npy")).astype(np.int64)


def build(ranks, F):
    d = ranks.shape[1]; blocks = [realm_rows(r, F) for r in ranks]
    return np.vstack(blocks), d, ranks.shape[0]


# ------------------------------------------------------------------ LP + exact checks
def max_margin(A, d):
    """max t s.t. A lam >= t (all rows), lam_0 = 0, lam_{d-1} <= 1, t <= 1. Returns (t, lam, result)."""
    k = A.shape[0]; As = csr_matrix(A.astype(float))
    Aub = shstack([-As, csr_matrix(np.ones((k, 1)))]).tocsr()                # -A lam + t <= 0
    c = np.zeros(d + 1); c[-1] = -1
    bounds = [(0, 0)] + [(None, 1)] * (d - 1) + [(None, 1)]
    r = linprog(c, A_ub=Aub, b_ub=np.zeros(k), bounds=bounds, method="highs")
    if r.status != 0: raise RuntimeError(r.message)
    return r.x[-1], r.x[:d], r


def exact_lam(lam, scale=10 ** 12):
    return [Fraction(int(round(v * scale)), scale) for v in lam]


def exact_rows_positive(A, lamF):
    """Exact: every row of integer A times rational lam strictly > 0. Returns (all_ok, n_bad, min_value)."""
    den = 1
    for f in lamF: den = den * f.denominator // math.gcd(den, f.denominator)
    z = [int(f * den) for f in lamF]
    vals = [sum(int(a) * zz for a, zz in zip(row, z) if a) for row in A.tolist()]
    bad = sum(v <= 0 for v in vals); return bad == 0, bad, Fraction(min(vals), den)


def gordan_certificate(A):
    """Find y >= 0, sum y = 1, A^T y = 0 (float LP, sparse vertex), then exact integer y on that support via exact null space."""
    k, d = A.shape; As = csr_matrix(A.astype(float))
    Aeq = svstack([As.T, csr_matrix(np.ones((1, k)))]).tocsr(); beq = np.r_[np.zeros(d), 1.0]
    r = linprog(np.zeros(k), A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * k, method="highs-ds")
    if r.status != 0: return None, r.message
    S = np.where(r.x > 1e-12)[0]
    y = exact_null_positive(A[S])
    if y is None: return None, f"support {len(S)}: no exact positive null vector"
    return (S, y), "ok"


def exact_null_positive(B):
    """B: integer rows. Exact rational basis of {c : c^T B = 0}; return a strictly positive integer c if the basis is 1-dim."""
    k, d = B.shape; M = [[Fraction(int(B[i, j])) for i in range(k)] for j in range(d)]   # d x k, solve M c = 0
    piv = []; rr = 0
    for col in range(k):
        p = next((i for i in range(rr, d) if M[i][col] != 0), None)
        if p is None: continue
        M[rr], M[p] = M[p], M[rr]; pv = M[rr][col]; M[rr] = [v / pv for v in M[rr]]
        for i in range(d):
            if i != rr and M[i][col] != 0:
                f = M[i][col]; M[i] = [a - f * b for a, b in zip(M[i], M[rr])]
        piv.append(col); rr += 1
        if rr == d: break
    free = [c for c in range(k) if c not in piv]
    if len(free) != 1: return None
    v = [Fraction(0)] * k; v[free[0]] = Fraction(1)
    for i, pc in enumerate(piv): v[pc] = -M[i][free[0]]
    L = 1
    for f in v: L = L * f.denominator // math.gcd(L, f.denominator)
    z = [int(f * L) for f in v]
    if all(t < 0 for t in z): z = [-t for t in z]
    if not all(t > 0 for t in z): return None
    g = 0
    for t in z: g = math.gcd(g, t)
    return [t // g for t in z]


def verify_certificate(A, S, y):
    """Exact: y positive ints on rows S of integer A; sum_i y_i A[S_i] == 0 identically."""
    if not (len(S) == len(y) > 0 and all(isinstance(t, int) and t > 0 for t in y)): return False
    tot = [0] * A.shape[1]
    for i, t in zip(S, y):
        for j, a in enumerate(A[i].tolist()): tot[j] += t * a
    return all(v == 0 for v in tot)


def joint(ranks, F, label):
    """Joint LP over all realms + strict ordering of lam. Returns a record with exact verification."""
    t0 = time.time(); R, d, nr = build(ranks, F); A = np.vstack([gap_rows(d), R])
    t, lam, _ = max_margin(A, d); rec = dict(label=label, realms=int(nr), rows=int(A.shape[0]), d=int(d), margin=float(t))
    if t > 1e-9:
        lamF = exact_lam(lam); ok, bad, mn = exact_rows_positive(A, lamF)
        rec.update(verdict="FEASIBLE" if ok else "FEASIBLE_FLOAT_ONLY", exact_ok=ok, exact_bad_rows=bad,
                   exact_min_row=str(mn), spectrum=[str(f) for f in lamF])
    else:
        cert, msg = gordan_certificate(A)
        if cert is None: rec.update(verdict="UNRESOLVED", msg=msg)
        else:
            S, y = cert; ok = verify_certificate(A, S, y)
            nd = d - 1; realms_in = sorted({int((i - nd) // (d - 1)) for i in S if i >= nd})
            rec.update(verdict="INFEASIBLE" if ok else "CERT_FAILED", exact_ok=ok, support=len(S), cert_rows=[int(i) for i in S],
                       cert_mult=y, cert_realms=realms_in, n_cert_realms=len(realms_in), max_mult=max(y))
    rec["secs"] = round(time.time() - t0, 2); return rec, A


def count_realms(ranks, F, lam):
    """Float count: realms whose rows are all > 0 on lam (lam sorted)."""
    R, d, nr = build(ranks, F); v = (R @ np.asarray(lam, float)).reshape(nr, d - 1)
    return int((v.min(1) > 0).sum()), v.min(1)


# ------------------------------------------------------------------ modes
def controls():
    out = {}
    # C1: encoding vs instrument on the stored hill-climb spectrum
    sys.path.insert(0, HERE)
    lam = np.array(json.load(open(os.path.join(RES, "maxmin_4x4.json")))["spectrum"])
    ranks, nsyt = bip_realms(4, 4); F = bip_fitmat(4, 4)
    mine, _ = count_realms(ranks, F, lam)
    Y = syt_all(4, 4); E = lam[Y]; mu = E.mean((-1, -2), keepdims=True); L = E.mean(-1, keepdims=True) + E.mean(-2, keepdims=True) - mu
    idx = [(a, b, a2, b2) for a in range(4) for a2 in range(a + 1, 4) for b in range(4) for b2 in range(b)]
    a, b, a2, b2 = map(np.array, zip(*idx))
    xm = ((L[:, a, b] - L[:, a2, b2]) * (E[:, a, b] - E[:, a2, b2])).min(-1)        # verbatim t1.xmin
    inst = int((xm > 0).sum()) // 2
    out["C1"] = dict(mine=mine, instrument=inst, stored=1114, realizable_classes=len(ranks), syt=nsyt,
                     PASS=bool(mine == inst == 1114 and len(ranks) == 3330))
    print("C1", out["C1"], flush=True)
    # C2: 2x2, 3x3 feasible; qubits 3, 4 feasible
    c2 = []
    for (m, n) in ((2, 2), (3, 3)):
        rk, _ = bip_realms(m, n); rec, _ = joint(rk, bip_fitmat(m, n), f"{m}x{n}"); c2.append(rec)
    for nq in (3, 4):
        rec, _ = joint(qubit_realms(nq), qubit_fitmat(nq), f"qubits{nq}"); c2.append(rec)
    want = {"2x2": 1, "3x3": 18, "qubits3": 2, "qubits4": 14}
    out["C2"] = dict(cases=[{k: r[k] for k in ("label", "realms", "verdict", "margin", "exact_ok")} for r in c2],
                     PASS=all(r["verdict"] == "FEASIBLE" and r["exact_ok"] and r["realms"] == want[r["label"]] for r in c2))
    print("C2", out["C2"], flush=True)
    # C3: null, one non-realizable 3x3 tableau alone
    Y = syt_all(3, 3); nonreal = [T for T in Y if not realizable(T)]
    T = nonreal[0]; rec, _ = joint(T.reshape(1, -1), bip_fitmat(3, 3), "3x3-nonrealizable")
    out["C3"] = dict(n_nonrealizable_3x3=len(nonreal), tableau=T.tolist(), verdict=rec["verdict"], exact_ok=rec.get("exact_ok"),
                     support=rec.get("support"), PASS=bool(len(nonreal) == 6 and rec["verdict"] == "INFEASIBLE" and rec["exact_ok"]))
    print("C3", out["C3"], flush=True)
    out["ALL_PASS"] = all(out[k]["PASS"] for k in ("C1", "C2", "C3"))
    json.dump(out, open(os.path.join(OUT, "controls.json"), "w"), indent=1); print("CONTROLS", "PASS" if out["ALL_PASS"] else "FAIL")


def joint_mode():
    out = {}
    for (m, n) in ((4, 4), (2, 3), (2, 4), (2, 5), (2, 6), (2, 8), (3, 4)):
        rk, nsyt = bip_realms(m, n); rec, _ = joint(rk, bip_fitmat(m, n), f"{m}x{n}"); rec["syt"] = nsyt; out[rec["label"]] = rec
        print({k: v for k, v in rec.items() if k not in ("cert_rows", "cert_mult", "spectrum", "cert_realms")}, flush=True)
    for nq in (5,):
        rec, _ = joint(qubit_realms(nq), qubit_fitmat(nq), f"qubits{nq}"); out[rec["label"]] = rec
        print({k: v for k, v in rec.items() if k not in ("cert_rows", "cert_mult", "spectrum", "cert_realms")}, flush=True)
    json.dump(out, open(os.path.join(OUT, "joint.json"), "w"), indent=1)


def realm_block(R, d, idx):
    return R[np.concatenate([np.arange(i * (d - 1), (i + 1) * (d - 1)) for i in idx])] if len(idx) else np.zeros((0, d), np.int64)


def greedy_mode(cap_s=float(os.environ.get("CAP_S", 1700))):
    """P3: start from the realms of the stored hill-climb spectrum; add realms in a seeded random order; keep each whose
    addition stays feasible (max margin > 1e-9). Final spectrum re-checked exactly; all realms it satisfies are counted."""
    t0 = time.time(); ranks, _ = bip_realms(4, 4); F = bip_fitmat(4, 4); R, d, nr = build(ranks, F); G = gap_rows(d)
    lam0 = np.array(json.load(open(os.path.join(RES, "maxmin_4x4.json")))["spectrum"]); lam0 = (lam0 - lam0[0]) / (lam0[-1] - lam0[0])
    vmin = (R @ lam0).reshape(nr, d - 1).min(1); S = [int(i) for i in np.where(vmin > 0)[0]]
    t, lam, _ = max_margin(np.vstack([G, realm_block(R, d, S)]), d); start = len(S)
    order = np.random.default_rng(5).permutation(nr); n_lp = 0; log = open(os.path.join(OUT, "greedy.log"), "w")
    for k, i in enumerate(order):
        if i in S: continue
        rows = R[i * (d - 1):(i + 1) * (d - 1)]
        if (rows @ lam).min() > 1e-9: S.append(int(i)); continue        # current witness already realises it
        t2, lam2, _ = max_margin(np.vstack([G, realm_block(R, d, S + [int(i)])]), d); n_lp += 1
        if t2 > 1e-9: S.append(int(i)); t, lam = t2, lam2
        if n_lp % 50 == 0:
            print(f"k={k} kept={len(S)} lps={n_lp} margin={t:.3e} secs={time.time() - t0:.0f}", file=log, flush=True)
            json.dump(dict(k=int(k), S=sorted(S), lam=lam.tolist(), margin=float(t)), open(os.path.join(OUT, "greedy_ckpt.json"), "w"))
        if time.time() - t0 > cap_s: print("CAP reached", file=log, flush=True); break
    lamF = exact_lam(lam, 10 ** 15); A = np.vstack([G, realm_block(R, d, S)]); ok, bad, mn = exact_rows_positive(A, lamF)
    # count every realm the final spectrum satisfies, exactly
    den = 10 ** 15; z = [int(f * den) for f in lamF]; vals = (R.astype(object) @ np.array(z, dtype=object)).reshape(nr, d - 1)
    exact_count = int(sum(1 for r in vals if min(r) > 0)); gaps_ok = all(z[j + 1] > z[j] for j in range(d - 1))
    out = dict(start=start, kept=len(S), exact_set_ok=ok, exact_bad=bad, final_margin=float(t), exact_count_all=exact_count,
               gaps_strict=gaps_ok, lps=n_lp, completed=bool(k == nr - 1), secs=round(time.time() - t0, 1),
               spectrum=[str(f) for f in lamF], realms=sorted(S))
    json.dump(out, open(os.path.join(OUT, "greedy.json"), "w"))
    print({k: v for k, v in out.items() if k not in ("spectrum", "realms")}, flush=True)


def infeasible_core(R, d, G, idx):
    """Deletion filter: shrink an infeasible realm set to a minimal infeasible one (each test = max-margin LP)."""
    core = list(idx)
    for i in list(core):
        trial = [j for j in core if j != i]
        t, _, _ = max_margin(np.vstack([G, realm_block(R, d, trial)]), d)
        if t <= 1e-9: core = trial
    return core


def upper_mode(cap_s=float(os.environ.get("CAP_S", 1700))):
    """Descriptive upper bound: repeatedly extract a certificate from the remaining realms, shrink it to a minimal infeasible
    realm set, verify exactly, remove it. k disjoint infeasible sets => max <= 3330 - k."""
    t0 = time.time(); ranks, _ = bip_realms(4, 4); F = bip_fitmat(4, 4); R, d, nr = build(ranks, F); G = gap_rows(d)
    alive = list(range(nr)); cores = []; log = open(os.path.join(OUT, "upper.log"), "w"); feasible_left = False
    while True:
        A = np.vstack([G, realm_block(R, d, alive)]); t, lam, _ = max_margin(A, d)
        if t > 1e-9: feasible_left = True; break
        cert, msg = gordan_certificate(A)
        if cert is None: print("UNRESOLVED", msg, file=log, flush=True); break
        S, y = cert; realms = sorted({alive[(i - (d - 1)) // (d - 1)] for i in S if i >= d - 1})
        core = infeasible_core(R, d, G, realms)
        Ac = np.vstack([G, realm_block(R, d, core)]); c2, _ = gordan_certificate(Ac)
        ok = c2 is not None and verify_certificate(Ac, *c2)
        cores.append(dict(realms=core, size=len(core), exact_ok=ok)); sc = set(core); alive = [j for j in alive if j not in sc]
        print(f"core {len(cores)} size {len(core)} ok {ok} alive {len(alive)} secs {time.time() - t0:.0f}", file=log, flush=True)
        json.dump(dict(cores=cores, alive=len(alive)), open(os.path.join(OUT, "upper_ckpt.json"), "w"))
        if time.time() - t0 > cap_s: print("CAP reached", file=log, flush=True); break
    out = dict(n_cores=len(cores), all_exact=all(c["exact_ok"] for c in cores), sizes=[c["size"] for c in cores],
               remaining_feasible=feasible_left, alive=len(alive), upper_bound=nr - sum(c["exact_ok"] for c in cores),
               completed=feasible_left, secs=round(time.time() - t0, 1), cores=cores)
    json.dump(out, open(os.path.join(OUT, "upper.json"), "w"))
    print({k: v for k, v in out.items() if k != "cores"}, flush=True)


def hinge_mode(eps=float(os.environ.get("EPS", 1e-3)), rounds=int(os.environ.get("ROUNDS", 8))):
    """Descriptive lower bound (not sealed): iteratively reweighted hinge LP. min sum_T w_T s_T, s_T >= eps - row (every row of
    realm T), s_T >= 0, gaps >= eps, lam in [0, 1]. Count realms with all rows > 0 exactly; reweight w_T = 1 / (s_T + delta)."""
    t0 = time.time(); ranks, _ = bip_realms(4, 4); F = bip_fitmat(4, 4); R, d, nr = build(ranks, F); G = gap_rows(d)
    k = R.shape[0]; owner = np.repeat(np.arange(nr), d - 1)
    S = csr_matrix((-np.ones(k), (np.arange(k), owner)), shape=(k, nr))
    Aub = svstack([shstack([csr_matrix(-R.astype(float)), S]), shstack([csr_matrix(-G.astype(float)), csr_matrix((d - 1, nr))])]).tocsr()
    bub = np.r_[-eps * np.ones(k), -eps * np.ones(d - 1)]
    bounds = [(0, 0)] + [(0, 1)] * (d - 1) + [(0, None)] * nr; w = np.ones(nr); best = (-1, None); hist = []
    for it in range(rounds):
        r = linprog(np.r_[np.zeros(d), w], A_ub=Aub, b_ub=bub, bounds=bounds, method="highs")
        lam = r.x[:d]; s = r.x[d:]; lamF = exact_lam(lam, 10 ** 15); den = 10 ** 15
        z = np.array([int(f * den) for f in lamF], dtype=object); vals = (R.astype(object) @ z).reshape(nr, d - 1)
        cnt = int(sum(1 for row in vals if min(row) > 0)); gaps = all(z[j + 1] > z[j] for j in range(d - 1)); hist.append(cnt)
        if gaps and cnt > best[0]: best = (cnt, [str(f) for f in lamF])
        print(f"round {it} exact count {cnt} sorted {gaps} secs {time.time() - t0:.0f}", flush=True)
        w = 1.0 / (s + 1e-4 * eps)
    out = dict(eps=eps, rounds=rounds, history=hist, best=best[0], spectrum=best[1], secs=round(time.time() - t0, 1))
    json.dump(out, open(os.path.join(OUT, f"hinge_eps{eps:g}.json"), "w")); print({k: v for k, v in out.items() if k != "spectrum"})


if __name__ == "__main__":
    if sys.argv[1] == "HINGE": hinge_mode(); sys.exit()
    {"CONTROLS": controls, "JOINT": joint_mode, "GREEDY": greedy_mode, "UPPER": upper_mode}[sys.argv[1]]()
