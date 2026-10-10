"""Quick D5 spot-check with impl_b's own C2 (Amendment 2 reading) and C3 (A4 inner optimiser) at d=16 assignments."""
import os
import sys
import json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import impl_b as B  # noqa: E402
import impl_b_T1 as T  # noqa: E402

rows = []
syts = T.enum_syt(4, 4)
for ch in range(3):
    st = B.setup(1000 + ch, 2000 + ch, 0, 4)
    H = st['H']
    N = T.fro2(H)
    E = np.linalg.eigvalsh(H)
    X = E[syts.reshape(len(syts), -1)].reshape(-1, 4, 4)
    sc = T.assignment_cost(X, N)
    rng = np.random.default_rng(55000 + ch)
    for kind, perm in (("SYT_global_min", syts[int(np.argmin(sc))].ravel()), ("random", rng.permutation(16))):
        U, Eg = T.assignment_unitary(H, perm)
        Hp = U.conj().T @ H @ U
        HA, HB, c, Hint = B.decompose(Hp, 4, 4)
        Hop = B.op_norm(H)
        c2, det = B.cost2(Hp, Hint, Hop, 4, 4, return_details=True)
        c3 = B.cost3(Hp, Hop, 4, 4)
        OA, OB = det['OA'], det['OB']
        offA = np.sqrt(T.fro2(OA - np.diag(np.diag(OA))))
        offB = np.sqrt(T.fro2(OB - np.diag(np.diag(OB))))
        gapA = float(np.min(np.diff(np.linalg.eigvalsh(OA))))
        # C3 at a basis product state (alpha = e_0, beta = e_0), exact evaluation of Eq. 12
        cg = B.make_cost3_fn(Hp, Hop, 4, 4)
        th = np.zeros(16)
        th[0] = 1.0
        th[8] = 1.0
        c3_basis, _ = cg(th)
        r = dict(chain=ch, kind=kind, cost1=float(T.cost_Hp(Hp, 4, 4, N)), cpo_f=float(det['f']),
                 cost2=float(c2), CPO_offdiag_OA=float(offA), CPO_offdiag_OB=float(offB), min_gap_OA=gapA,
                 cost3_inner_opt=float(c3), cost3_at_basis_state=float(c3_basis))
        rows.append(r)
        print(json.dumps(r), flush=True)
T.save("D5.json", rows)
