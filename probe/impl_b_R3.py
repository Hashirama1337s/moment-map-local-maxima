#!/usr/bin/env python3
"""
impl_b_R3.py -- implementation B re-evaluation of the R3 witnesses (two-implementation rule), impl_b only.
Inputs read: D:\\_LIVE\\QMEREO\\theory\\results\\witness_U\\U_n{n}_ch{k}.npy ONLY (allow_pickle=False).

Usage: py -3.11 impl_b_R3.py <n> <k> [<k> ...]   -> impl_b_R3_out/R3_n{n}_ch{k}.json
       py -3.11 impl_b_R3.py SUMMARY             -> rule counts over all JSON files

Per (n, k), at U* and at the plant S (H = S H_plant(1000+k) S^dag, S = haar(2000+k)):
  cost1; cost2 with the CRITERIA A3 solver (replicated per start so the tie-break spread is visible, selection
  identical to impl_b.cpo_solve); cost2 at U* with an exact non-degenerate diagonal CPO; cost3 (plant: A4 inner
  optimiser; U*: basis product state alpha=e_0, beta=e_0, plus the inner optimiser); D(U*, S).
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import impl_b as B  # noqa: E402

WIT = r"<repo>\theory\results\witness_U"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "impl_b_R3_out")
os.makedirs(OUT, exist_ok=True)


def fro2(M):
    return float(np.real(np.vdot(M, M)))


def cpo_per_start(Hint, LA, LB, dA, dB, n_starts=8, max_iter=200, rtol=1e-12):
    """Exact replica of impl_b.cpo_solve, returning every start's final (f, a, b) and the index it would select."""
    d = dA * dB
    nA, nB = LA.shape[0], LB.shape[0]

    def f_val(a, b):
        T = np.kron(np.tensordot(a, LA, axes=(0, 0)), np.tensordot(b, LB, axes=(0, 0)))
        return fro2(Hint @ T - T @ Hint)

    res = []
    best_i = None
    for s in range(n_starts):
        rng = np.random.default_rng(s)
        a = rng.standard_normal(nA)
        b = rng.standard_normal(nB)
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        f_old = f_val(a, b)
        for _ in range(max_iter):
            OB = np.tensordot(b, LB, axes=(0, 0))
            TA = np.einsum('jac,bd->jabcd', LA, OB).reshape(nA, d, d)
            w, V = np.linalg.eigh(B._gram(Hint, TA))
            a = V[:, 0].copy()
            OA = np.tensordot(a, LA, axes=(0, 0))
            TB = np.einsum('ac,kbd->kabcd', OA, LB).reshape(nB, d, d)
            w, V = np.linalg.eigh(B._gram(Hint, TB))
            b = V[:, 0].copy()
            f_new = float(w[0])
            if abs(f_new - f_old) < rtol * max(abs(f_old), 1e-300):
                break
            f_old = f_new
        f_fin = f_val(a, b)
        res.append((f_fin, a.copy(), b.copy()))
        if best_i is None or f_fin < res[best_i][0]:
            best_i = s
    return res, best_i


def cost2_from_cpo(Hp, OA, OB, dA, dB, Hop):
    S_lin, S_pt = B.schwinger_terms(Hp, OA, OB, dA, dB)
    return float(np.mean(np.maximum(S_lin, S_pt))) / Hop ** 2


def c2_block(Hp, Hint, Hop, dA, dB):
    LA, LB = B.ggm_basis(dA), B.ggm_basis(dB)
    res, bi = cpo_per_start(Hint, LA, LB, dA, dB)
    per = []
    for (f, a, b) in res:
        OA = np.tensordot(a, LA, axes=(0, 0))
        OB = np.tensordot(b, LB, axes=(0, 0))
        gap = float(np.min(np.diff(np.linalg.eigvalsh(OA))))
        per.append(dict(f=f, cost2=cost2_from_cpo(Hp, OA, OB, dA, dB, Hop), gapOA=gap))
    return dict(cost2=per[bi]['cost2'], f=per[bi]['f'], gapOA=per[bi]['gapOA'], selected_start=bi, per_start=per,
                cost2_min_over_starts=min(p['cost2'] for p in per), cost2_max_over_starts=max(p['cost2'] for p in per),
                n_starts_f_below_1e_8=int(sum(p['f'] < 1e-8 for p in per)))


def exact_diag_cpo(Hp, Hint, Hop, dA, dB):
    """Non-degenerate traceless diagonal CPO: O = diag(0,1,..,m-1) - mean, unit Frobenius norm."""
    def mk(m):
        v = np.arange(m, dtype=float) - (m - 1) / 2
        return np.diag(v / np.linalg.norm(v)).astype(complex)
    OA, OB = mk(dA), mk(dB)
    T = np.kron(OA, OB)
    f = fro2(Hint @ T - T @ Hint)
    return dict(cost2=cost2_from_cpo(Hp, OA, OB, dA, dB, Hop), f=f)


def c3_basis(Hp, Hop, dA, dB):
    th = np.zeros(2 * dA + 2 * dB)
    th[0] = 1.0           # Re alpha_0
    th[2 * dA] = 1.0      # Re beta_0
    c, _ = B.make_cost3_fn(Hp, Hop, dA, dB)(th)
    return float(c)


def eval_chain(n, k):
    t0 = time.time()
    st = B.setup(1000 + k, 2000 + k, 0, n)
    dA, dB, d, H, S = st['dA'], st['dB'], st['d'], st['H'], st['S']
    Hop = B.op_norm(H)
    Us = np.load(os.path.join(WIT, f"U_n{n}_ch{k}.npy"), allow_pickle=False)
    assert Us.shape == (d, d), Us.shape
    unit_err = float(np.sqrt(fro2(Us.conj().T @ Us - np.eye(d))))
    out = dict(n=n, k=k, seed_H=1000 + k, seed_S=2000 + k, unitarity_err=unit_err, Hop=Hop, HF2=fro2(H))
    for tag, U in (("Ustar", Us), ("S", S)):
        Hp = U.conj().T @ H @ U
        Hp = 0.5 * (Hp + Hp.conj().T)
        HA, HB, c, Hint = B.decompose(Hp, dA, dB)
        r = dict(cost1=B.cost1(H, Hint),
                 offdiag_rel=float(np.sqrt(fro2(Hp - np.diag(np.diag(Hp))) / fro2(H))))
        r['c2'] = c2_block(Hp, Hint, Hop, dA, dB)
        c3, det3 = B.cost3(Hp, Hop, dA, dB, return_details=True)
        r['cost3_inner'] = c3
        r['cost3_inner_finals'] = det3['finals']
        if tag == "Ustar":
            r['c2_exact_diag_cpo'] = exact_diag_cpo(Hp, Hint, Hop, dA, dB)
            r['cost3_basis_e0e0'] = c3_basis(Hp, Hop, dA, dB)
        out[tag] = r
    out['D_Ustar_S'] = B.distance(Us, S, dA, dB)
    out['D_Ustar_1'] = B.distance(Us, np.eye(d), dA, dB)
    out['t'] = time.time() - t0
    with open(os.path.join(OUT, f"R3_n{n}_ch{k}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    u, s = out['Ustar'], out['S']
    print(f"n={n} k={k} c1 {u['cost1']:.8g}/{s['cost1']:.8g}  c2 {u['c2']['cost2']:.8g}/{s['c2']['cost2']:.8g} "
          f"(exactCPO@U* {u['c2_exact_diag_cpo']['cost2']:.3g})  c3 basis {u['cost3_basis_e0e0']:.3g} inner "
          f"{u['cost3_inner']:.3g}/{s['cost3_inner']:.8g}  D {out['D_Ustar_S']:.6g}  t={out['t']:.0f}s", flush=True)


def summary():
    rows = []
    for fn in sorted(os.listdir(OUT)):
        if fn.startswith("R3_") and fn.endswith(".json"):
            with open(os.path.join(OUT, fn)) as fh:
                rows.append(json.load(fh))
    rows.sort(key=lambda r: (r['n'], r['k']))
    readings = {
        "R3_primary (C2: CRITERIA solver at both; C3: U* basis e0e0 vs S inner)":
            lambda u, s: (u['c2']['cost2'], s['c2']['cost2'], u['cost3_basis_e0e0'], s['cost3_inner']),
        "R3_exactCPO (C2: exact diag CPO at U* vs CRITERIA solver at S; C3 as primary)":
            lambda u, s: (u['c2_exact_diag_cpo']['cost2'], s['c2']['cost2'], u['cost3_basis_e0e0'], s['cost3_inner']),
        "R3_inner (C2: CRITERIA solver at both; C3: inner optimiser at both)":
            lambda u, s: (u['c2']['cost2'], s['c2']['cost2'], u['cost3_inner'], s['cost3_inner']),
        "R3_robust (C2: max over 8 starts at U* < min over 8 starts at S; C3 as primary)":
            lambda u, s: (u['c2']['cost2_max_over_starts'], s['c2']['cost2_min_over_starts'], u['cost3_basis_e0e0'], s['cost3_inner']),
    }
    res = {}
    for name, fn in readings.items():
        cnt = {4: 0, 6: 0}
        tot = {4: 0, 6: 0}
        fails = []
        for r in rows:
            u, s = r['Ustar'], r['S']
            c2u, c2s, c3u, c3s = fn(u, s)
            ok = (u['cost1'] < s['cost1']) and (c2u < c2s) and (c3u < c3s) and (r['D_Ustar_S'] >= 0.3)
            tot[r['n']] += 1
            cnt[r['n']] += int(ok)
            if not ok:
                fails.append(f"n{r['n']}ch{r['k']}")
        res[name] = dict(count_n4=cnt[4], total_n4=tot[4], count_n6=cnt[6], total_n6=tot[6], failing=fails)
        print(f"{name}: n=4 {cnt[4]}/{tot[4]}, n=6 {cnt[6]}/{tot[6]}; failing: {fails}")
    print()
    print("n k | cost1 U*/S | cost2 U* sel [min,max over starts] / S sel [min,max] | exactCPO U* | c3 U* basis / inner | c3 S | D(U*,S) | offdiag U* | unit_err | gapOA U* sel / S sel")
    for r in rows:
        u, s = r['Ustar'], r['S']
        print(f"{r['n']} {r['k']:2d} | {u['cost1']:.8g} / {s['cost1']:.8g} | "
              f"{u['c2']['cost2']:.6g} [{u['c2']['cost2_min_over_starts']:.3g},{u['c2']['cost2_max_over_starts']:.3g}] / "
              f"{s['c2']['cost2']:.6g} [{s['c2']['cost2_min_over_starts']:.3g},{s['c2']['cost2_max_over_starts']:.3g}] | "
              f"{u['c2_exact_diag_cpo']['cost2']:.2e} (f {u['c2_exact_diag_cpo']['f']:.1e}) | "
              f"{u['cost3_basis_e0e0']:.2e} / {u['cost3_inner']:.2e} | {s['cost3_inner']:.6g} | {r['D_Ustar_S']:.6g} | "
              f"{u['offdiag_rel']:.1e} | {r['unitarity_err']:.1e} | {u['c2']['gapOA']:.1e} / {s['c2']['gapOA']:.1e}")
    with open(os.path.join(OUT, "SUMMARY.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "SUMMARY":
        summary()
    else:
        n = int(sys.argv[1])
        for k in sys.argv[2:]:
            eval_chain(n, int(k))
