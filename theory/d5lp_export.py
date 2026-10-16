"""Export the door-5 exact objects to theory/results/d5lp/certificates/ in a solver-free, self-describing JSON form:
  joint_<shape>.json   : all-realms verdict; FEASIBLE -> rational spectrum; INFEASIBLE -> certificate rows as
                          (kind, realm tableau or order, step k, multiplier), kind = "gap" (lam_{k+1} - lam_k) or "realm"
  pair_4x4.json        : the incompatible pair with its certificate
  lower_1209.json      : the MILP-probe spectrum (exact rationals) and its realm count
  upper_cores.json     : 485 disjoint infeasible realm sets, each with its own exact certificate
Run: py -3.11 theory/d5lp_export.py   (recomputes certificates on the stored supports; every one verified exactly)"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d5lp as L
C = os.path.join(L.OUT, "certificates"); os.makedirs(C, exist_ok=True)


def describe(A_rows_idx, mult, d, reps, label_of):
    out = []
    for i, y in zip(A_rows_idx, mult):
        i = int(i)
        if i < d - 1: out.append(dict(kind="gap", k=i, multiplier=int(y)))
        else:
            j, k = divmod(i - (d - 1), d - 1); out.append(dict(kind="realm", **label_of(reps[j]), k=int(k), multiplier=int(y)))
    return out


from fractions import Fraction
from scipy.optimize import linprog


def int_witness(G, nv, lo_bounds):
    """LP G v <= -1 (margin 1), round to rationals, scale to integers; caller verifies exactly."""
    r = linprog(np.zeros(nv), A_ub=G, b_ub=-np.ones(len(G)), bounds=lo_bounds, method="highs")
    assert r.status == 0
    for den in (1, 2, 6, 12, 60, 1000, 10 ** 6):
        fr = [Fraction(v).limit_denominator(den) for v in r.x]; Lc = 1
        for f in fr: Lc = Lc * f.denominator // np.gcd(Lc, f.denominator)
        z = [int(f * Lc) for f in fr]
        if all(sum(int(g) * zz for g, zz in zip(row, z)) < 0 for row in G.astype(int).tolist()): return z
    raise AssertionError("no integer witness")


def bip_witness(T):
    m, n = T.shape; cells = np.argsort(T.reshape(-1)); G = []
    for k in range(m * n - 1):
        i, j = divmod(cells[k], n); i2, j2 = divmod(cells[k + 1], n); row = np.zeros(m + n)
        row[i] += 1; row[m + j] += 1; row[i2] -= 1; row[m + j2] -= 1; G.append(row)
    z = int_witness(np.array(G), m + n, [(None, None)] * (m + n)); return dict(x=z[:m], y=z[m:])


def q_witness(rank):
    N = len(rank); nq = N.bit_length() - 1; cells = np.argsort(rank); G = []
    bits = lambda c: np.array([(c >> j) & 1 for j in range(nq)], float)
    for k in range(N - 1): G.append(bits(cells[k]) - bits(cells[k + 1]))
    for j in range(nq - 1): r = np.zeros(nq); r[j] = 1; r[j + 1] = -1; G.append(r)
    r = np.zeros(nq); r[0] = -1; G.append(r)
    return dict(w=int_witness(np.array(G), nq, [(None, None)] * nq))


def bip_label(m, n):
    return lambda r: dict(tableau=np.asarray(r).reshape(m, n).tolist(), outer_sum_witness=bip_witness(np.asarray(r).reshape(m, n)))


def q_label(r):
    return dict(order_rank=np.asarray(r).tolist(), weight_witness=q_witness(np.asarray(r)))


def cert_for(ranks, F, label_of, d):
    R, _, _ = L.build(ranks, F); A = np.vstack([L.gap_rows(d), R]); (S, y), msg = L.gordan_certificate(A)
    assert L.verify_certificate(A, S, y), msg
    return describe(S, y, d, ranks, label_of)


def main():
    J = json.load(open(os.path.join(L.OUT, "joint.json"))); C2 = json.load(open(os.path.join(L.OUT, "controls.json")))
    shapes = {"2x2": (2, 2), "2x3": (2, 3), "2x4": (2, 4), "2x5": (2, 5), "2x6": (2, 6), "2x8": (2, 8), "3x3": (3, 3), "3x4": (3, 4), "4x4": (4, 4)}
    for lab, (m, n) in shapes.items():
        rk, _ = L.bip_realms(m, n); F = L.bip_fitmat(m, n); rec, A = L.joint(rk, F, lab)
        out = dict(shape=lab, kind="bipartite", realms=int(len(rk)), verdict=rec["verdict"])
        if rec["verdict"] == "FEASIBLE": out["spectrum"] = rec["spectrum"]
        else:
            d = m * n; S, y = rec["cert_rows"], rec["cert_mult"]; assert L.verify_certificate(A, S, y)
            out["certificate"] = describe(S, y, d, rk, bip_label(m, n))
        json.dump(out, open(os.path.join(C, f"joint_{lab}.json"), "w"), indent=0); print(lab, out["verdict"], len(rk), flush=True)
    for nq in (3, 4, 5):
        rk = L.qubit_realms(nq); F = L.qubit_fitmat(nq); rec, A = L.joint(rk, F, f"qubits{nq}")
        out = dict(shape=f"qubits{nq}", kind="qubits", realms=int(len(rk)), verdict=rec["verdict"])
        if rec["verdict"] == "FEASIBLE": out["spectrum"] = rec["spectrum"]; out["orders"] = [q_label(r) for r in rk]
        else: out["certificate"] = describe(rec["cert_rows"], rec["cert_mult"], 2 ** nq, rk, q_label)
        json.dump(out, open(os.path.join(C, f"joint_qubits{nq}.json"), "w"), indent=0); print(nq, out["verdict"], flush=True)
    rk, _ = L.bip_realms(4, 4); F = L.bip_fitmat(4, 4); U = json.load(open(os.path.join(L.OUT, "upper.json")))
    cores = []
    for c in U["cores"]:
        sub = rk[c["realms"]]; cores.append(dict(realms=[bip_label(4, 4)(r) for r in sub],
                                                 certificate=cert_for(sub, F, bip_label(4, 4), 16)))
    json.dump(dict(shape="4x4", n_cores=len(cores), total_realms=3330, upper_bound=3330 - len(cores), cores=cores),
              open(os.path.join(C, "upper_cores.json"), "w"))
    pair = [c for c in cores if len(c["realms"]) == 2][0]; json.dump(dict(shape="4x4", **pair), open(os.path.join(C, "pair_4x4.json"), "w"), indent=1)
    M = json.load(open(os.path.join(L.OUT, "milp_probe.json")))
    json.dump(dict(shape="4x4", claimed_realms=M["exact_count"], spectrum=M["spectrum"]), open(os.path.join(C, "lower_1209.json"), "w"), indent=0)
    G = json.load(open(os.path.join(L.OUT, "greedy.json")))
    json.dump(dict(shape="4x4", claimed_realms=G["exact_count_all"], spectrum=G["spectrum"]), open(os.path.join(C, "lower_1164_greedy.json"), "w"), indent=0)
    print("cores", len(cores), "pair", [len(c["realms"]) for c in cores].count(2))


if __name__ == "__main__":
    main()
