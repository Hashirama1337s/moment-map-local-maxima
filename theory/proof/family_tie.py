"""Explicit critical family through a tied assignment (equally spaced spectrum 0..7 on 2x2x2).
Pairs c1=(0,0,1)/c1'=(1,0,0) and c2=(0,1,1)/c2'=(1,1,0) are at Hamming distance 2 and P1(e_c1 - e_c1') = P1(e_c2 - e_c2').
If L ties both pairs at an assignment E and delta1 = E_c1' - E_c1, delta2 = E_c2' - E_c2 have opposite signs, then Givens
rotations by theta1, theta2 with sin^2(theta1) delta1 + sin^2(theta2) delta2 = 0 leave L unchanged, keep H' block-diagonal on
the L-level sets, hence give critical points (non-assignments) with the same f.  We check grad = 0 along the family and watch
the Hessian minimum as theta1 -> 0."""
import sys, os, json, numpy as np
from itertools import permutations, product
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core_k import *
from k1_check import predicted, order_iso
dims = (2, 2, 2); d = 8; cells = list(product(range(2), range(2), range(2))); pos = {c: i for i, c in enumerate(cells)}
c1, c1p, c2, c2p = pos[(0, 0, 1)], pos[(1, 0, 0)], pos[(0, 1, 1)], pos[(1, 1, 0)]
w1 = np.zeros(d); w1[c1], w1[c1p] = 1, -1; w2 = np.zeros(d); w2[c2], w2[c2p] = 1, -1
p1 = np.real(np.diag(P1(np.diag(w1).astype(complex), dims))); p2 = np.real(np.diag(P1(np.diag(w2).astype(complex), dims)))
print("||P1 w1 - P1 w2|| =", np.linalg.norm(p1 - p2))
basis = herm_basis(d); found = None; best = None
for perm in permutations(range(d)):
    E = np.array(perm, float); Lv = np.real(np.diag(P1(np.diag(E).astype(complex), dims)))
    if abs(Lv[c1] - Lv[c1p]) > 1e-12 or abs(Lv[c2] - Lv[c2p]) > 1e-12: continue
    dl1, dl2 = E[c1p] - E[c1], E[c2p] - E[c2]
    if dl1 * dl2 >= 0: continue
    pr = predicted(E, Lv, dims)
    if pr[0] < -1e-9: continue           # want an assignment that is 2nd-order PSD (only the tie zero modes)
    found = (E, Lv, dl1, dl2); break
E, Lv, dl1, dl2 = found
print("assignment E (cells 000..111):", E, " f =", cost(np.eye(d, dtype=complex), np.diag(E).astype(complex), dims),
      " deltas", dl1, dl2, " min predicted Hessian eig", predicted(E, Lv, dims)[0])
H = np.diag(E).astype(complex); out = []
for th1 in [0.3, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3]:
    s2 = -np.sin(th1) ** 2 * dl1 / dl2
    if s2 > 1: continue
    th2 = np.arcsin(np.sqrt(s2))
    K = np.zeros((d, d), complex)
    K[c1, c1p] = -1j * th1; K[c1p, c1] = 1j * th1; K[c2, c2p] = -1j * th2; K[c2p, c2] = 1j * th2
    U = expm(1j * K); Hp, L = parts(U, H, dims)
    g = 2 * np.linalg.norm(comm(L, Hp)); ev = np.linalg.eigvalsh(hess_analytic(Hp, L, dims, basis))
    asg = is_assignment(Hp, dims)[0]; q = commutant_form(Hp, L, dims)[0]
    out.append(dict(theta1=th1, theta2=float(th2), grad=float(g), f=cost(U, H, dims), assignment=asg, hess_min=float(ev[0]), N1=q))
    print(f"theta1={th1:.0e} theta2={th2:.3e} grad={g:.1e} f={cost(U, H, dims):.6f} assignment={asg} Hmin={ev[0]:+.3e} N1={q:+.3e}")
json.dump(dict(E=E.tolist(), rows=out), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'family_tie.json'), 'w'), indent=1)
