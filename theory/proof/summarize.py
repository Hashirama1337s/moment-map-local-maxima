"""Aggregate search_*.json: per run, converged critical points, degenerate-local / non-assignment counts, theorem and
P5 violations, worst (least negative) N1 and Hessian minimum over the gap class, and the kernel dimension at every
local-minimum candidate vs gauge + torus = (m^2+n^2-1) + (m-1)(n-1)."""
import glob, json, os, collections
D = os.path.dirname(os.path.abspath(__file__))
tot = collections.Counter(); lines = []
for fn in sorted(glob.glob(os.path.join(D, 'search_*.json'))):
    J = json.load(open(fn)); s = J['summary']; R = J['records']; m, n = s['m'], s['n']
    conv = [r for r in R if r['grad'] < 1e-12]
    gap = [r for r in conv if (not r['assignment']) or min(r['alpha_gap'], r['beta_gap']) < 1e-6]
    cand = [r for r in conv if r['local_min_candidate']]
    kz = m * m + n * n - 1 + (m - 1) * (n - 1)
    kbad = sum(r['n_zero'] != kz for r in cand)
    bymode = collections.Counter(r['mode'] for r in gap)
    maxg = max([r['grad'] for r in conv], default=None)
    worstN1 = max([r['N1_min'] for r in gap], default=None); worstH = max([r['hess_min'] for r in gap], default=None)
    lines.append(f"{os.path.basename(fn):28s} runs={len(R):4d} conv={len(conv):4d} maxgrad={maxg:.1e} gapclass={len(gap):4d} "
                 f"(by mode {dict(bymode)}) cand={len(cand):3d} viol={s['theorem_violations']} p5viol={s['p5_violations']} "
                 f"kernel!={kz}:{kbad} worstN1={worstN1:+.3f} worstHmin={worstH:+.3f}")
    tot.update(runs=len(R), conv=len(conv), gap=len(gap), cand=len(cand), viol=s['theorem_violations'],
               p5=s['p5_violations'], kbad=kbad)
print("\n".join(lines)); print("TOTAL", dict(tot))
open(os.path.join(D, 'search_summary.txt'), 'w').write("\n".join(lines) + "\nTOTAL " + json.dumps(dict(tot)) + "\n")
