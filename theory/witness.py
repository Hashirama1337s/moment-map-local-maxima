"""Exact witnesses for the pre-registered readout R3 (PROBE-BAR): a factorization U* far from the lattice plant (U = S) with
strictly lower C1, C2, C3. U* = exact C1 global-min assignment (n = 4, SYT enumeration) or the spectral_init assignment (n = 6).
C3 is evaluated at a basis-state (alpha, beta), an upper bound on min over (alpha, beta). Saves U* for implementation B.
Run from probe/:  py ../theory/witness.py <n> <chains>"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "probe"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import impl_a as A, t1, probe as PR
DEV, CT, RT = A.DEV, A.CT, A.RT

n, chains = int(sys.argv[1]), int(sys.argv[2]); rows = []
Y = t1.syt_all(4, 4) if n == 4 else None
os.makedirs(os.path.join(t1.OUT, "witness_U"), exist_ok=True)
for ch in range(chains):
    st = A.setup(1000 + ch, 2000 + ch, n); dA = dB = st["dA"]; lam = st["lam"].cpu().numpy().real
    if n == 4:
        cmin, grid, _, _, _ = t1.syt_min(lam, 4, 4, Y); U = t1.U_of(st, grid)
    else:
        U, _ = PR.spectral_init(st, 0)
    np.save(os.path.join(t1.OUT, "witness_U", f"U_n{n}_ch{ch}.npy"), U.cpu().numpy())
    out = {}
    for name, W in (("plant", st["S"]), ("witness", U)):
        Hp = W.conj().T @ st["H"] @ W
        with torch.no_grad():
            c1 = A.cost1(Hp, st["H_fro2"], dA, dB).item(); c2 = A.cost2(Hp, dA, dB, st["H_op2"]).item()
        if name == "plant":
            c3 = A.cost3_inner((st["lam"], st["V"]), W, dA, dB, st["K"], st["dt"])
        else:
            a = torch.zeros(dA, dtype=CT, device=DEV); a[0] = 1; b = torch.zeros(dB, dtype=CT, device=DEV); b[0] = 1
            with torch.no_grad(): c3 = A.cost3_at((st["lam"], st["V"]), W, a, b, dA, dB, st["K"], st["dt"]).item()
        out[name] = (c1, c2, c3)
    D = A.tps_distance(U, st["S"], dA, dB)
    r = dict(n=n, chain=ch, plant_c1=out["plant"][0], plant_c2=out["plant"][1], plant_c3=out["plant"][2],
             wit_c1=out["witness"][0], wit_c2=out["witness"][1], wit_c3=out["witness"][2], D_witness_plant=D,
             all_strictly_lower=bool(all(out["witness"][k] < out["plant"][k] for k in range(3))), far=bool(D >= 0.3))
    rows.append(r); print(json.dumps(r), flush=True)
k = sum(r["all_strictly_lower"] and r["far"] for r in rows)
print(f"R3 witnesses n={n}: {k}/{len(rows)} chains have a factorization >= 0.3 from the lattice with strictly lower C1, C2 AND C3")
json.dump(rows, open(os.path.join(t1.OUT, f"witness_n{n}.json"), "w"), indent=0)
