"""BAR-D13C verdict (with Amendment 1 tie rule): reads results/d13/cold.json, prints controls and HOLDS / FAILED per item.
Run: py -3.11 theory/d13_cold_verdict.py"""
import os, json
R = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "d13", "cold.json")))
c3 = R["C3"]; print("C3 inputs = stored blind inputs:", c3, "PASS" if max(c3.values()) < 1e-12 else "FAIL")
c1 = max(s["summary"]["C1_worst_rel"] for s in R["sets"].values()); c2 = all(s["summary"]["C2_all_same"] for s in R["sets"].values())
print(f"C1 mpmath vs float MI at beta=1, worst relative {c1:.2e}:", "PASS" if c1 < 1e-9 else "FAIL")
print("C2 winner order reproduced at dps+60:", "PASS" if c2 else "FAIL")
for k, s in R["sets"].items():
    m = s["summary"]; nt = m["count"] - m["ties"]
    print(f"{k}: n={m['count']} ties={m['ties']} classes={m['classes']} chance={m['chance_baseline']:.3f} | "
          f"P1 {m['P1_frac']:.3f} {'HOLDS' if m['P1_frac'] >= 0.95 else 'FAILED'} | "
          f"P2 {m['P2_frac']:.3f} {'HOLDS' if m['P2_frac'] >= 0.95 else 'FAILED'} | "
          f"P3 {m['P3_frac']:.3f} {'HOLDS' if m['P3_frac'] >= 0.95 else 'FAILED'} | e11 binds {m['e11_binds_frac']:.2f}, "
          f"stable 50/100/200 {m['stable_50_100_200']:.2f}, double-precision beta=100 winner = exact beta=100 winner "
          f"{sum(r['float_cold_100'] == r['win_100'] for r in s['rows']) / len(s['rows']):.3f}")
    miss = [r["seed"] for r in s["rows"] if not r["tie"] and not r["inX"]]
    if miss: print("   P1 misses (seeds):", miss)
