"""BAR-D4 T5: exact full 6-qubit factorization (min over the 124,187 coherent orders) for chains 0-9, cross-checked against a
multistart 1-local sort-fit alternation (any lower value would kill K3)."""
import sys, os, json, time, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "probe")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import impl_a as A, door4 as D4
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"); n = 6; N = 64
ranks = np.load(os.path.join(OUT, "coherent_n6.npz"))["ranks"]; Xd = D4.design(n); Pm = Xd @ np.linalg.pinv(Xd)
Z = torch.as_tensor(Xd[:, 1:], device=A.DEV, dtype=torch.float64)                     # (64, 6) bit signs
def alternation(lam, starts=200000, iters=200, seed=0):
    g = torch.Generator(device=A.DEV).manual_seed(seed); lamt = torch.as_tensor(np.sort(lam), device=A.DEV, dtype=torch.float64)
    w = torch.randn(starts, n, generator=g, device=A.DEV, dtype=torch.float64) * lamt.std() / 2
    mu = lamt.mean(); prev = None
    for _ in range(iters):
        s = mu + w @ Z.T                                                             # (starts, 64) fitted values per cell
        order = torch.argsort(s, 1); rank = torch.empty_like(order); rank.scatter_(1, order, torch.arange(N, device=A.DEV).expand(starts, N))
        E = lamt[rank]; w = (E - E.mean(1, keepdim=True)) @ Z / N; mu = E.mean(1, keepdim=True)
        if prev is not None and torch.equal(rank, prev): break
        prev = rank
    R = E - (mu + w @ Z.T); res = (R ** 2).sum(1) / (lamt ** 2).sum(); return res.min().item(), len(torch.unique(torch.round(res * 1e12)))
rows = []
for ch in range(10):
    lam = A.setup(1000 + ch, 2000 + ch, n)["lam"].cpu().numpy().real; t0 = time.time()
    cmin, rank, call, iso = D4.exact_full_qubit(lam, ranks, n)
    alt, ndistinct = alternation(lam)
    r = dict(chain=ch, exact_full_qubit=float(cmin), n_local_minima=int(iso.sum()), alternation_best=alt, alternation_distinct_minima=ndistinct,
             gap=alt - float(cmin), secs=round(time.time() - t0, 1))
    rows.append(r); print(json.dumps(r), flush=True)
worst = min(r["gap"] for r in rows)
print(f"T5: min(alternation - exact) = {worst:.3e} -> {'PASS' if worst > -1e-12 else 'FAIL (K3 killed)'}; local minima per chain {[r['n_local_minima'] for r in rows]} (bound 124187)")
json.dump(rows, open(os.path.join(OUT, "D4_T5.json"), "w"), indent=0)
