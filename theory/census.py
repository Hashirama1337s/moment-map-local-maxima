"""Census of C1 local minima ('realms') = SYT passing (X), per spectrum ensemble. Pure combinatorics on spectra (BAR-T1 D4).
Run from probe/:  py ../theory/census.py <samples>"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "probe"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t1


def spectra(kind, m, n, k):
    d = m * n; rng = np.random.default_rng(1234 + k)
    if kind == "chain":
        import impl_a as A
        nq = int(np.log2(d)); return A.setup(1000 + k, 2000 + k, nq)["lam"].cpu().numpy().real
    if kind == "GUE":
        G = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)); return np.linalg.eigvalsh((G + G.conj().T) / 2)
    if kind == "Poisson":
        return np.sort(rng.uniform(-1, 1, d))
    if kind == "free-fermion":                           # 4 modes, exactly additive under any 2|2 mode split (control)
        eps = rng.uniform(0.2, 1.0, 4); s = np.array([[(i >> j) & 1 for j in range(4)] for i in range(16)]) - 0.5
        return np.sort(s @ eps)


def stats(lam, m, n, Y):
    cmin, grid, nloc, c, is_min = t1.syt_min(lam, m, n, Y)
    cs = np.sort(c[is_min]); return dict(n_local_minima=int(nloc), global_min=float(cmin),
                                       second_min_ratio=float(cs[2] / cs[0]) if (m == n and len(cs) > 2) else (float(cs[1] / cs[0]) if len(cs) > 1 else None),
                                       max_local_min=float(cs[-1]) if len(cs) else None)


if __name__ == "__main__":
    S = int(sys.argv[1]); out = {}
    for (m, n) in ((2, 2), (3, 3), (2, 8), (4, 4)):
        Y = t1.syt_all(m, n); kinds = ["GUE", "Poisson"] + (["chain"] if m * n in (4, 16) else []) + (["free-fermion"] if (m, n) == (4, 4) else [])
        for kind in kinds:
            rows = [stats(spectra(kind, m, n, k), m, n, Y) for k in range(S if kind != "free-fermion" else 5)]
            nl = np.array([r["n_local_minima"] for r in rows]); gm = np.array([r["global_min"] for r in rows])
            key = f"{m}x{n} {kind}"
            out[key] = dict(SYT_classes=len(Y) // (2 if m == n else 1), samples=len(rows), local_minima_median=float(np.median(nl)),
                            local_minima_min=int(nl.min()), local_minima_max=int(nl.max()), global_min_median=float(np.median(gm)),
                            global_min_max=float(gm.max()))
            print(key, json.dumps(out[key]), flush=True)
    json.dump(out, open(os.path.join(t1.OUT, "census.json"), "w"), indent=1)
