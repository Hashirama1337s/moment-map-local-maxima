"""Print the markdown tables for REPORT.md from the JSON outputs."""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
prec = {(r["n"], r["k"]): r for r in json.load(open(os.path.join(HERE, "precise.json")))}

print("### V3 table\n")
print("| n | chain k (seed) | C* (mpmath, 12 s.f.) | float64 C* | 2nd-best region | rel. gap | fixed pts / chamber | fixed pts total | optimum is order-isomorphic |")
print("|---|---|---|---|---|---|---|---|---|")
for n in (5, 6):
    for r in json.load(open(os.path.join(HERE, f"vstar_n{n}.json"))):
        p = prec[(n, r["k"])]
        cs = float(p["Cstar_hp"])
        print(f"| {n} | {r['k']} ({r['seed_H']}) | {cs:.12g} | {r['Cstar_lstsq']:.12g} | {r['second_best']:.12g} | "
              f"{(r['second_best'] - cs) / cs:.2e} | {r['n_fixed_chamber']} / {r['n_regions_chamber']} | {r['n_fixed_total']} | "
              f"{r['best_is_order_isomorphic']} |")
print()
print("### Optimal placements (chamber representative)\n")
for n in (5, 6):
    for r in json.load(open(os.path.join(HERE, f"vstar_n{n}.json"))):
        w = ", ".join(f"{x:.6f}" for x in r["fit_weights"])
        print(f"- n={n}, k={r['k']}: ranks r[s] for s = 0..{2 ** n - 1}: `{' '.join(str(x) for x in r['ranks'])}`  "
              f"(fit c_1..c_{n} = {w})")
print()
print("### V4 table\n")
print("| n | chain k | greedy cost (mpmath) | C* | greedy / C* | step residuals (levels: res) | envelope lines per step |")
print("|---|---|---|---|---|---|---|")
for n in (5, 6):
    rows = []
    for fn in sorted(os.listdir(HERE)):
        if fn.startswith(f"greedy_n{n}_") and fn.endswith(".json"):
            rows += json.load(open(os.path.join(HERE, fn)))
    for r in sorted(rows, key=lambda x: x["k"]):
        p = prec[(n, r["k"])]
        steps = "; ".join(f"{s['levels']}: {s['residual']:.6g}" for s in r["steps"])
        lines = "/".join(str(s["envelope_lines"]) for s in r["steps"])
        print(f"| {n} | {r['k']} | {float(p['greedy_hp']):.12g} | {float(p['Cstar_hp']):.12g} | {float(p['ratio_hp']):.10g} | {steps} | {lines} |")
print()
for n in (5, 6):
    rows = []
    for fn in sorted(os.listdir(HERE)):
        if fn.startswith(f"greedy_n{n}_") and fn.endswith(".json"):
            rows += json.load(open(os.path.join(HERE, fn)))
    for r in sorted(rows, key=lambda x: x["k"]):
        print(f"- greedy n={n}, k={r['k']}: ranks r[s]: `{' '.join(str(x) for x in r['ranks'])}`")
