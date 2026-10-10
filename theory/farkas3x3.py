"""Exact realizability of rectangular standard Young tableaux (SYT) by additive matrices, with Farkas certificates.

A SYT T of shape m x n (convention of t1.py / realizable.py: T[a, b] = rank of the cell (a, b), 0 = smallest) is
REALIZABLE if there are reals x_a (rows), y_b (columns) such that L_ab = x_a + y_b orders the cells strictly as T does.

For every non-realizable T this script finds a FARKAS CERTIFICATE: nonnegative integer multipliers mu_(p,q) on strict
inequalities "L_p < L_q" (p before q in T's order), not all zero, whose combination is identically zero as a linear form
in (x, y).  Summing the inequalities would then give 0 < 0, so the strict system is infeasible.  Every certificate is
checked in exact integer arithmetic, in two forms:
  * PAIR form: a minimal-support combination of pair inequalities L_p < L_q (any p before q);
  * CONSECUTIVE form: the same certificate rewritten on the m*n - 1 consecutive inequalities L_(rank k) < L_(rank k+1)
    (the constraint set used by realizable.py); a pair (p, q) is the sum of the consecutive ones between their ranks.
For every realizable T it records an exact integer witness (x, y) and checks the strict order exactly.

Floating-point LP (scipy HiGHS) is used only to FIND candidates; every accepted object is re-verified with Python ints /
fractions.Fraction.  Shapes: 3x3 (the deliverable) and 2x2..2x5 (all realizable; used for the counts 1, 5, 14, 42).

Run (CPU, a few seconds):  python theory/farkas3x3.py
Output: theory/results/farkas3x3.json
"""
import os, sys, json, itertools
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def syt_all(m, n):
    """Verbatim copy of t1.syt_all (t1.py imports torch and the probe instrument; this script must stay light).
    All standard Young tableaux of rectangular shape m x n as rank grids (cell -> rank of eigenvalue, 0 = smallest)."""
    out = []; T = -np.ones((m, n), int); rowlen = [0] * m
    def rec(k):
        if k == m * n: out.append(T.copy()); return
        for a in range(m):
            b = rowlen[a]
            if b < n and (a == 0 or rowlen[a - 1] > b):
                T[a, b] = k; rowlen[a] += 1; rec(k + 1); rowlen[a] -= 1; T[a, b] = -1
    rec(0); return np.array(out)


def cell_of_rank(T):
    m, n = T.shape
    return {int(T[a, b]): (a, b) for a in range(m) for b in range(n)}


def vec(m, n, cell):
    """Coefficient vector of L_cell = x_a + y_b in the variables (x_0..x_{m-1}, y_0..y_{n-1})."""
    v = [0] * (m + n); v[cell[0]] += 1; v[m + cell[1]] += 1; return v


# ---------------------------------------------------------------- exact checks
def check_witness(T, x, y):
    """Exact: L_ab = x_a + y_b is strictly increasing along T's rank order."""
    pos = cell_of_rank(T); vals = [x[a] + y[b] for (a, b) in (pos[k] for k in range(T.size))]
    return all(vals[k] < vals[k + 1] for k in range(len(vals) - 1))


def check_pair_cert(T, pairs):
    """Exact: pairs = [(p, q, mu)], p before q in T, mu positive int; sum mu (vec(p) - vec(q)) == 0 identically."""
    m, n = T.shape; r = {c: k for k, c in cell_of_rank(T).items()}; tot = [0] * (m + n)
    for p, q, mu in pairs:
        if not (isinstance(mu, int) and mu > 0 and r[p] < r[q]): return False
        vp, vq = vec(m, n, p), vec(m, n, q)
        for j in range(m + n): tot[j] += mu * (vp[j] - vq[j])
    return len(pairs) > 0 and all(t == 0 for t in tot)


def check_consecutive_cert(T, mu):
    """Exact: mu[k] >= 0 int on consecutive inequalities L_(rank k) - L_(rank k+1) < 0; not all zero; combination == 0."""
    m, n = T.shape; pos = cell_of_rank(T); tot = [0] * (m + n)
    if not (all(isinstance(u, int) and u >= 0 for u in mu) and any(u > 0 for u in mu)): return False
    for k, u in enumerate(mu):
        vp, vq = vec(m, n, pos[k]), vec(m, n, pos[k + 1])
        for j in range(m + n): tot[j] += u * (vp[j] - vq[j])
    return all(t == 0 for t in tot)


# ---------------------------------------------------------------- search (float LP, then exact)
def consecutive_rows(T):
    m, n = T.shape; pos = cell_of_rank(T)
    return np.array([np.array(vec(m, n, pos[k])) - np.array(vec(m, n, pos[k + 1])) for k in range(T.size - 1)], float)


def find_witness(T):
    """LP with margin 1 (as realizable.py); round the vertex to rationals, scale to integers, verify exactly."""
    m, n = T.shape; G = consecutive_rows(T)
    r = linprog(np.zeros(m + n), A_ub=G, b_ub=-np.ones(len(G)), bounds=[(0, None)] * (m + n), method="highs")
    if r.status != 0: return None
    for den in (1, 2, 6, 12, 60, 1000):
        fr = [Fraction(v).limit_denominator(den) for v in r.x]
        L = 1
        for f in fr: L = L * f.denominator // np.gcd(L, f.denominator)
        z = [int(f * L) for f in fr]; x, y = z[:m], z[m:]
        if check_witness(T, x, y): return x, y
    return None


def nullspace_exact(rows, ncols):
    """Exact rational null space of the matrix whose COLUMNS are the given row vectors (i.e. solve sum_i c_i rows_i = 0).
    Returns a basis as lists of Fractions (one entry per row vector)."""
    k = len(rows); A = [[Fraction(rows[i][j]) for i in range(k)] for j in range(ncols)]   # ncols x k
    piv = []; R = [r[:] for r in A]; rr = 0
    for c in range(k):
        p = next((i for i in range(rr, len(R)) if R[i][c] != 0), None)
        if p is None: continue
        R[rr], R[p] = R[p], R[rr]; pv = R[rr][c]; R[rr] = [v / pv for v in R[rr]]
        for i in range(len(R)):
            if i != rr and R[i][c] != 0:
                f = R[i][c]; R[i] = [a - f * b for a, b in zip(R[i], R[rr])]
        piv.append(c); rr += 1
    free = [c for c in range(k) if c not in piv]; basis = []
    for fcol in free:
        v = [Fraction(0)] * k; v[fcol] = Fraction(1)
        for i, pc in enumerate(piv): v[pc] = -R[i][fcol]
        basis.append(v)
    return basis


def to_int_vector(v):
    L = 1
    for f in v: L = L * f.denominator // np.gcd(L, f.denominator)
    z = [int(f * L) for f in v]; g = 0
    for t in z: g = int(np.gcd(g, abs(t)))
    return [t // g for t in z]


def find_pair_certificate(T):
    """Minimal-support Farkas certificate on pair inequalities.  Exhaustive over supports of size 2, 3, 4 (in that order):
    for each support S, the exact null space of {vec(p) - vec(q)}_{(p,q) in S}; accept a strictly positive null vector."""
    m, n = T.shape; pos = cell_of_rank(T); d = T.size
    pairs = [(pos[i], pos[j]) for i in range(d) for j in range(i + 1, d)]
    rows = [[a - b for a, b in zip(vec(m, n, p), vec(m, n, q))] for p, q in pairs]
    for size in (2, 3, 4):
        for S in itertools.combinations(range(len(pairs)), size):
            B = nullspace_exact([rows[i] for i in S], m + n)
            if len(B) != 1: continue
            z = to_int_vector(B[0])
            if all(t < 0 for t in z): z = [-t for t in z]
            if all(t > 0 for t in z):
                cert = [(pairs[i][0], pairs[i][1], int(t)) for i, t in zip(S, z)]
                if check_pair_cert(T, cert): return cert
    return None


def pair_to_consecutive(T, cert):
    r = {c: k for k, c in cell_of_rank(T).items()}; mu = [0] * (T.size - 1)
    for p, q, u in cert:
        for k in range(r[p], r[q]): mu[k] += u
    return mu


def tableau_str(T):
    return "/".join("".join(str(int(v) + 1) for v in row) for row in T)        # 1-based entries, rows separated by '/'


def run_shape(m, n, want_certs):
    Ys = syt_all(m, n); real, nonreal = [], []
    for T in Ys:
        w = find_witness(T)
        if w is not None:
            real.append(dict(T=T.tolist(), x=w[0], y=w[1], exact_ok=check_witness(T, *w)))
            continue
        cert = find_pair_certificate(T) if want_certs else None
        rec = dict(T=T.tolist(), T_1based=tableau_str(T))
        if cert is not None:
            mu = pair_to_consecutive(T, cert)
            rec.update(pair_certificate=[dict(smaller=list(p), larger=list(q), multiplier=u) for p, q, u in cert],
                       pair_certificate_exact_ok=check_pair_cert(T, cert),
                       consecutive_multipliers=mu, consecutive_certificate_exact_ok=check_consecutive_cert(T, mu))
        nonreal.append(rec)
    return Ys, real, nonreal


def main():
    os.makedirs(OUT, exist_ok=True); res = {}
    # ---- 3x3: the deliverable
    Ys, real, nonreal = run_shape(3, 3, True)
    all_real_ok = all(r["exact_ok"] for r in real)
    all_cert_ok = all(r.get("pair_certificate_exact_ok") and r.get("consecutive_certificate_exact_ok") for r in nonreal)
    # transpose structure of the non-realizable set
    keys = {tuple(map(tuple, r["T"])) for r in nonreal}
    closed = all(tuple(map(tuple, np.array(r["T"]).T)) in keys for r in nonreal)
    # agreement with the earlier run (theory/general/results/attain_syt_3x3.json), if present
    prev = os.path.join(os.path.dirname(os.path.abspath(__file__)), "general", "results", "attain_syt_3x3.json")
    agree = None
    if os.path.exists(prev):
        P = json.load(open(prev)); agree = sorted(map(str, P["nonrealizable"])) == sorted(str(r["T"]) for r in nonreal)
    res["3x3"] = dict(n_syt=len(Ys), n_realizable=len(real), n_nonrealizable=len(nonreal),
                      all_witnesses_exact=all_real_ok, all_certificates_exact=all_cert_ok,
                      nonrealizable_closed_under_transpose=closed, n_transpose_classes_realizable=len(real) // 2,
                      matches_attain_syt_3x3=agree, nonrealizable=nonreal, realizable=real)
    print(f"3x3: SYT {len(Ys)}, realizable {len(real)} (exact witnesses ok: {all_real_ok}), non-realizable {len(nonreal)} "
          f"(exact certificates ok: {all_cert_ok}); transpose-closed: {closed}; matches attain.py: {agree}")
    for r in nonreal:
        c = "; ".join(f"{p['multiplier']}*(L{tuple(p['smaller'])} < L{tuple(p['larger'])})" for p in r["pair_certificate"])
        print(f"  {r['T_1based']}:  {c}   consecutive: {r['consecutive_multipliers']}")
    # ---- 2 x n, n = 2..5: every SYT realizable, exact witnesses
    for nn in (2, 3, 4, 5):
        Ys, real, nonreal = run_shape(2, nn, False)
        res[f"2x{nn}"] = dict(n_syt=len(Ys), n_realizable=len(real), n_nonrealizable=len(nonreal),
                              all_witnesses_exact=all(r["exact_ok"] for r in real), realizable=real)
        print(f"2x{nn}: SYT {len(Ys)}, realizable {len(real)} (exact: {res[f'2x{nn}']['all_witnesses_exact']}), "
              f"non-realizable {len(nonreal)}")
    json.dump(res, open(os.path.join(OUT, "farkas3x3.json"), "w"), indent=1)
    ok = (res["3x3"]["n_syt"] == 42 and res["3x3"]["n_realizable"] == 36 and res["3x3"]["n_nonrealizable"] == 6
          and all_real_ok and all_cert_ok and all(res[f"2x{k}"]["n_nonrealizable"] == 0 and res[f"2x{k}"]["all_witnesses_exact"]
                                                  for k in (2, 3, 4, 5)))
    print("ALL EXACT CHECKS PASS" if ok else "SOME CHECK FAILED"); return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
