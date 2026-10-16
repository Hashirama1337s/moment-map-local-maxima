"""Stand-alone exact checker for Addendum C (door 5). Pure Python (json, fractions, itertools); no LP, no NumPy.
It rebuilds every inequality from the definitions in the note and checks, in exact integer / rational arithmetic:
  * FEASIBLE shapes: the spectrum is strictly increasing and realises the full count of realms
    (bipartite: canonical standard Young tableaux whose additive fit orders the cells as the tableau; count = A211400 / 2
    for square shapes, A211400 otherwise; qubits: every listed coherent order, each with an integer weight witness).
  * INFEASIBLE shapes, the pair and the 485 cores: every tableau / order used is a valid SYT / coherent order with an exact
    witness of realizability; the multipliers are positive integers; the weighted sum of the rows is identically zero.
  * the 485 cores are pairwise disjoint as realm classes (tableau modulo transpose);
  * the lower-bound spectra realise the claimed number of realms.
Run from anywhere:  python check_d5.py [certificate folder]   (default: results/d5lp/certificates next to this file)"""
import os, sys, json
from fractions import Fraction
from math import gcd

HERE = os.path.dirname(os.path.abspath(__file__))
CERT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "d5lp", "certificates")
FAIL = []


def check(cond, msg):
    if not cond: FAIL.append(msg)
    return cond


# ---------------------------------------------------------------- objects
def syt_list(m, n):
    out = []; T = [[-1] * n for _ in range(m)]; rowlen = [0] * m
    def rec(k):
        if k == m * n: out.append([r[:] for r in T]); return
        for a in range(m):
            b = rowlen[a]
            if b < n and (a == 0 or rowlen[a - 1] > b):
                T[a][b] = k; rowlen[a] += 1; rec(k + 1); rowlen[a] -= 1; T[a][b] = -1
    rec(0); return out


def is_syt(T):
    m, n = len(T), len(T[0]); flat = sorted(v for r in T for v in r)
    return flat == list(range(m * n)) and all(T[a][b] < T[a][b + 1] for a in range(m) for b in range(n - 1)) \
        and all(T[a][b] < T[a + 1][b] for a in range(m - 1) for b in range(n))


def bip_rows(T):
    """Integer rows (over lam_0..lam_{d-1}) of d * (l_{c_{k+1}} - l_{c_k}), l = row mean + column mean - grand mean of lam[T]."""
    m, n = len(T), len(T[0]); d = m * n
    def dl(a, b):                                   # coefficient vector of d * l_ab
        v = [0] * d
        for b2 in range(n): v[T[a][b2]] += m
        for a2 in range(m): v[T[a2][b]] += n
        for k in range(d): v[k] -= 1
        return v
    pos = {T[a][b]: (a, b) for a in range(m) for b in range(n)}
    return [[x - y for x, y in zip(dl(*pos[k + 1]), dl(*pos[k]))] for k in range(d - 1)]


def q_rows(rank):
    """Integer rows of N * (l_{c_{k+1}} - l_{c_k}), l = projection of lam[rank] onto span{1, Z_1..Z_n}."""
    N = len(rank); nq = N.bit_length() - 1
    z = [[1] + [1 - 2 * ((c >> j) & 1) for j in range(nq)] for c in range(N)]
    def dl(c):
        v = [0] * N
        for c2 in range(N): v[rank[c2]] += sum(x * y for x, y in zip(z[c], z[c2]))
        return v
    cell = {rank[c]: c for c in range(N)}
    return [[x - y for x, y in zip(dl(cell[k + 1]), dl(cell[k]))] for k in range(N - 1)]


def gap_row(d, k):
    v = [0] * d; v[k + 1] = 1; v[k] = -1; return v


def realizable_ok(T, wit):
    m, n = len(T), len(T[0]); x, y = wit["x"], wit["y"]
    vals = {T[a][b]: x[a] + y[b] for a in range(m) for b in range(n)}
    return all(vals[k] < vals[k + 1] for k in range(m * n - 1))


def coherent_ok(rank, wit):
    w = wit["w"]; N = len(rank); nq = N.bit_length() - 1
    if not (w[0] > 0 and all(w[j] < w[j + 1] for j in range(nq - 1))): return False
    s = {rank[c]: sum(w[j] for j in range(nq) if (c >> j) & 1) for c in range(N)}
    return all(s[k] < s[k + 1] for k in range(N - 1))


def as_int_vector(spec):
    lam = [Fraction(x) for x in spec]; den = 1
    for f in lam: den = den * f.denominator // gcd(den, f.denominator)
    return [int(f * den) for f in lam]


def dot(r, z): return sum(a * b for a, b in zip(r, z) if a)


def certificate_ok(entries, d, label):
    tot = [0] * d
    for e in entries:
        mu = e["multiplier"]
        if not check(isinstance(mu, int) and mu > 0, f"{label}: multiplier not a positive int"): return False
        if e["kind"] == "gap": row = gap_row(d, e["k"])
        elif "tableau" in e:
            T = e["tableau"]
            check(is_syt(T), f"{label}: not a SYT"); check(realizable_ok(T, e["outer_sum_witness"]), f"{label}: realizability witness fails")
            row = bip_rows(T)[e["k"]]
        else:
            rk = e["order_rank"]; check(coherent_ok(rk, e["weight_witness"]), f"{label}: coherent witness fails")
            row = q_rows(rk)[e["k"]]
        for j in range(d): tot[j] += mu * row[j]
    return check(len(entries) > 0 and all(t == 0 for t in tot), f"{label}: weighted sum is not zero")


def count_bip(m, n, z):
    """Exact number of realm classes (canonical representatives) realised by integer-scaled sorted spectrum z."""
    cnt = 0
    for T in syt_list(m, n):
        if m == n and T[0][1] != 1: continue
        if all(dot(r, z) > 0 for r in bip_rows(T)): cnt += 1
    return cnt


def canon(T):
    t = tuple(tuple(r) for r in T); tt = tuple(zip(*t)); return min(t, tt)


# ---------------------------------------------------------------- checks
REAL = {"2x2": 1, "2x3": 5, "2x4": 14, "2x5": 42, "3x3": 18}          # A211400 (halved for square shapes)
COH = {"qubits3": 2, "qubits4": 14}                                    # A009997
summary = []
for f in sorted(os.listdir(CERT)):
    if not f.startswith("joint_"): continue
    J = json.load(open(os.path.join(CERT, f))); lab = J["shape"]
    if J["verdict"] == "FEASIBLE":
        z = as_int_vector(J["spectrum"]); d = len(z); check(all(z[k] < z[k + 1] for k in range(d - 1)), f"{lab}: not strictly sorted")
        if J["kind"] == "bipartite":
            m, n = map(int, lab.split("x")); c = count_bip(m, n, z); ok = check(c == REAL[lab], f"{lab}: realises {c}, need {REAL[lab]}")
        else:
            orders = J["orders"]; keys = {tuple(o["order_rank"]) for o in orders}
            ok = check(len(keys) == len(orders) == COH[lab], f"{lab}: order count") and \
                all(check(coherent_ok(o["order_rank"], o["weight_witness"]), f"{lab}: witness") and
                    check(all(dot(r, z) > 0 for r in q_rows(o["order_rank"])), f"{lab}: an order fails") for o in orders)
        summary.append(f"{lab}: all {J['realms']} realms at once: FEASIBLE, exact spectrum {'OK' if ok else 'FAIL'}")
    else:
        d = (int(lab[6:]) and 2 ** int(lab[6:])) if J["kind"] == "qubits" else int(lab.split("x")[0]) * int(lab.split("x")[1])
        ok = certificate_ok(J["certificate"], d, lab)
        summary.append(f"{lab}: all {J['realms']} realms at once: INFEASIBLE, certificate ({len(J['certificate'])} rows) {'OK' if ok else 'FAIL'}")
P = json.load(open(os.path.join(CERT, "pair_4x4.json")))
summary.append(f"pair_4x4: two realms never coexist, certificate ({len(P['certificate'])} rows) {'OK' if certificate_ok(P['certificate'], 16, 'pair') else 'FAIL'}")
U = json.load(open(os.path.join(CERT, "upper_cores.json"))); seen = set(); disjoint = True; allok = True
for i, c in enumerate(U["cores"]):
    for r in c["realms"]:
        k = canon(r["tableau"]); disjoint &= k not in seen; seen.add(k)
        allok &= check(is_syt(r["tableau"]) and realizable_ok(r["tableau"], r["outer_sum_witness"]), f"core {i}: realm not realizable")
    used = {canon(e["tableau"]) for e in c["certificate"] if e["kind"] == "realm"}
    allok &= check(used <= {canon(r["tableau"]) for r in c["realms"]}, f"core {i}: certificate uses a realm outside the core")
    allok &= certificate_ok(c["certificate"], 16, f"core {i}")
check(disjoint, "cores not disjoint")
summary.append(f"upper: {len(U['cores'])} pairwise-disjoint infeasible realm sets ({'disjoint' if disjoint else 'NOT disjoint'}, certificates "
               f"{'OK' if allok else 'FAIL'}) -> max <= {3330 - len(U['cores'])}")
for f in ("lower_1209.json", "lower_1164_greedy.json"):
    Lw = json.load(open(os.path.join(CERT, f))); z = as_int_vector(Lw["spectrum"])
    srt = check(all(z[k] < z[k + 1] for k in range(15)), f"{f}: not sorted"); c = count_bip(4, 4, z)
    check(c == Lw["claimed_realms"], f"{f}: counts {c}, claimed {Lw['claimed_realms']}")
    summary.append(f"{f}: strictly sorted {srt}, exact realm count {c} (claimed {Lw['claimed_realms']})")
print("\n".join(summary))
print("ALL CHECKS PASS" if not FAIL else "FAILURES:\n  " + "\n  ".join(FAIL[:20]))
sys.exit(1 if FAIL else 0)
