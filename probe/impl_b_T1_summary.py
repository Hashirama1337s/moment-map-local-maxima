"""Aggregate impl_b_T1_out/*.json into the numbers quoted in impl_b_T1_REPORT.md."""
import json
import os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "impl_b_T1_out")


def load(n):
    with open(os.path.join(OUT, n)) as fh:
        return json.load(fh)


print("== CHECK")
for k, v in load("CHECK.json").items():
    print(f"  {k}: {v:.3e}")

print("== B2")
b2 = load("B2.json")
for r in b2:
    dv = r['d4_vs_hess']
    extra = "" if dv is None else (f" D4={dv['n_D4']} hessStrictMin={dv['n_hess_strict_min']} disagree={dv['n_disagree']}/{dv['n_reps']}"
                                   f" classes={r['n_D4_classes']} reached={r['n_D4_classes_reached']}"
                                   f" notAssign={r['n_endpoints_not_assignment']} notOnD4={r['n_endpoints_not_on_D4_list']}")
    print(f"  {r['kind']:10s} {r['dA']}x{r['dB']} seed={r['seed']} assign_min={r['assign_min']:.12g} opt_best={r['opt_best']:.12g} "
          f"diff={r['diff']:+.2e} hits={r['n_hit_global']}/{r['n_restarts']} maxgrad={r['max_final_grad']:.1e} "
          f"negHess={r['n_endpoints_with_neg_hessian']}{extra}")
gue = [r for r in b2 if r['kind'] == 'GUE']
print(f"  max(opt_best - assign_min) over all {len(b2)} cases: {max(r['diff'] for r in b2):+.3e}; "
      f"min: {min(r['diff'] for r in b2):+.3e}")
print(f"  max final grad over all restarts: {max(r['max_final_grad'] for r in b2):.2e}")
print(f"  D4-vs-Hessian disagreements (GUE, all canonical reps): {sum(r['d4_vs_hess']['n_disagree'] for r in gue)} "
      f"of {sum(r['d4_vs_hess']['n_reps'] for r in gue)}")
print(f"  endpoints not assignment: {sum(r['n_endpoints_not_assignment'] for r in gue)}; not on D4 list: "
      f"{sum(r['n_endpoints_not_on_D4_list'] for r in gue)}; with negative Hessian eig: "
      f"{sum(r['n_endpoints_with_neg_hessian'] for r in b2)}")

print("== B3")
allfd, allan = [], []
for ch in range(3):
    rows = load(f"B3_ch{ch}.json")
    for r in rows:
        allfd.append(r['max_abs_dev_fd_vs_D3'])
        allan.append(r['max_abs_dev_analytic_vs_D3'])
        print(f"  ch{ch} {r['kind']:15s} cost={r['cost']:.6g} D4={r['D4']!s:5s} counts_fd={r['counts_fd']} counts_D3={r['counts_D3']} "
              f"dev_fd={r['max_abs_dev_fd_vs_D3']:.2e} dev_an={r['max_abs_dev_analytic_vs_D3']:.2e} scale={r['eig_scale']:.3g}")
print(f"  MAX |FD Hessian spectrum - D3| = {max(allfd):.3e};  MAX |analytic - D3| = {max(allan):.3e}")

print("== B4")
b4 = load("B4.json")
for r in b4['chains']:
    print(f"  chain {r['chain']} (seed_H {r['seed_H']}): D4 SYT {r['n_D4_SYT']} / {r['n_SYT']} -> classes {r['n_D4_classes']:.0f}; "
          f"transposed-D4 {r['n_D4_transposed']}; min residual {r['min_residual_D4']:.10g} (= min over all SYT {r['min_residual_all_SYT']:.10g}); "
          f"max D4 residual {r['max_residual_D4']:.6g}")
print(f"  D4 vs analytic Hessian, chain 0: {b4['d4_vs_hessian_chain0']}")

print("== B5")
for tag in ("", "_n500"):
    for ch in range(3):
        fn = f"B5_ch{ch}{tag}.json"
        if not os.path.exists(os.path.join(OUT, fn)):
            continue
        s = load(fn)['summary']
        e = load(fn)['endpoints']
        print(f"  {fn}: n={s['n']} assignment={s['n_assignment']} inD4set={s['n_in_D4_set']} costOnD4={s['n_cost_on_D4_list']} "
              f"negHess={s['n_neg_hessian']} hitGlobal={s['n_hit_global']} best-global={s['best_minus_global']:+.2e} "
              f"distinct={s['n_distinct_minima']}/{s['n_D4_classes']} maxgrad={s['max_grad']:.1e} "
              f"min gapL={min(x['gapL_rel'] for x in e):.2e} max offdiag={max(x['offdiag_rel'] for x in e):.1e}")
