"""Independent falsification checks of impl_b_E results (different algorithms from the certified B&B).

V1: at tau* = mean pair coordinate of the certified optimum M*, compute the exact min-cost perfect matching with cost
    (d/sqrt2 - tau*)^2 by Edmonds' blossom algorithm (networkx). If some matching M had cost(M, tau*) < RSS(M*),
    then RSS(M) <= cost(M, tau*) < RSS(M*), contradicting optimality. So blossom must return cost == RSS(M*).
V1/V2: brute-force local search over the certified grid (all transpositions of two entries) must not find a lower residual.
"""
import os
import sys
import json
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import impl_b as B  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "impl_b_E_out")


def rss_grid(X):
    R = X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + X.mean()
    return float(np.sum(R ** 2))


def swap_search(X):
    """Best residual over all single transpositions of two grid entries (must be >= RSS(X) at a local min)."""
    flat = X.ravel().copy()
    base = rss_grid(X)
    best = np.inf
    n = len(flat)
    for i in range(n):
        for j in range(i + 1, n):
            f = flat.copy()
            f[i], f[j] = f[j], f[i]
            best = min(best, rss_grid(f.reshape(X.shape)))
    return base, best


def main(case, chains):
    rows = []
    for k in chains:
        r = json.load(open(os.path.join(OUT, f"{case}_ch{k}.json")))
        n = r['n']
        lam = np.sort(np.linalg.eigvalsh(B.build_H_plant(n, 1000 + k)))
        grid = np.array(r['rank_grid'])
        X = lam[grid]
        rss = rss_grid(X)
        out = dict(case=case, k=k, rss_grid=rss, rss_reported=r['rss'])
        if case == "V1":
            d = (X[1] - X[0]) / np.sqrt(2)
            tau = d.mean()
            G = nx.Graph()
            for i in range(len(lam)):
                for j in range(i + 1, len(lam)):
                    G.add_edge(i, j, weight=-((lam[j] - lam[i]) / np.sqrt(2) - tau) ** 2)
            M = nx.max_weight_matching(G, maxcardinality=True)
            cost = sum(((lam[max(a, b)] - lam[min(a, b)]) / np.sqrt(2) - tau) ** 2 for a, b in M)
            out['blossom_cost_at_tau'] = float(cost)
            out['blossom_minus_rss_rel'] = float((cost - rss) / rss)
        base, best_swap = swap_search(X)
        out['best_single_swap_rss'] = best_swap
        out['swap_minus_rss_rel'] = float((best_swap - base) / base)
        rows.append(out)
        print(json.dumps(out), flush=True)
    json.dump(rows, open(os.path.join(OUT, f"CHECK_{case}.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2:]])
