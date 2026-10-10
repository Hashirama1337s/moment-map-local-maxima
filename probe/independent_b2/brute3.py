"""Validation: n = 3 brute force over all 8! = 40320 placements vs the region method, chains k = 0..9.
Also counts, over ALL placements, those with L order-isomorphic to E, and compares with
(#fixed chamber regions) x 2^n n!."""
import itertools
import math

import numpy as np

from common import spectrum, zmat, bits, cost_of_E
from vstar import analyse, load_orders

n = 3
N = 2 ** n
Z = zmat(n)
S = bits(n)
P = np.array(list(itertools.permutations(range(N))), dtype=np.int64)  # P[p, s] = rank of bitstring s
orders = load_orders(n)
group = 2 ** n * math.factorial(n)
allok = True
for k in range(10):
    lam = spectrum(n, k)
    E = lam[P]
    C = cost_of_E(E, lam, Z)
    V = E @ Z
    L = V @ S.T  # ordering of L == ordering of v.s
    # order-isomorphic: argsort of L equals the rank pattern
    Lr = np.argsort(np.argsort(L, axis=1, kind="stable"), axis=1)
    srt = np.sort(L, axis=1)
    iso = (Lr == P).all(1) & (np.diff(srt, axis=1).min(1) > 1e-12)
    Cr, F, fixed, margin = analyse(n, lam, orders)
    bf_min = C.min()
    reg_min = Cr.min()
    n_bf_min = int(np.sum(C <= bf_min + 1e-13))
    ok = abs(bf_min - reg_min) < 1e-13 and int(iso.sum()) == int(fixed.sum()) * group
    allok &= ok
    print(f"k={k}: brute-force min = {bf_min:.12g}  region min = {reg_min:.12g}  #argmin placements = {n_bf_min} "
          f"(expect {group} x #optimal chamber regions)  order-iso placements: brute = {int(iso.sum())}, "
          f"region = {int(fixed.sum())} x {group} = {int(fixed.sum()) * group}  {'OK' if ok else 'MISMATCH'}")
print("ALL OK" if allok else "SOME MISMATCH")
