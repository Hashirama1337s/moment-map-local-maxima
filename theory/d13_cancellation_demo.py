"""Addendum D, Section 3: why double precision cannot resolve the cold end. On the 4x4 GUE instance seed 0 at beta = 100,
compare the double-precision mutual information I = H(A) + H(B) - H(AB) with the exact multi-precision value, over all
12,012 tableau classes. Prints the ranges. Run (from probe/): python ../theory/d13_cancellation_demo.py   (about 1 minute)"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d13_cold as C

idx = C.classes(4, 4); E = C.spectrum("GUE", 16, 0); beta = 100.0
fl = C.float_mi(E, idx, 4, 4, beta); ex, _ = C.exact_mi(E, idx, 4, 4, beta); ex = np.array([float(v) for v in ex])
w = np.exp(-beta * (E - E.min())); w /= w.sum(); hab = -(w * np.log(w)).sum()
print(f"4x4 GUE seed 0, beta = 100, {len(idx)} classes")
print(f"  joint entropy H(AB)            = {hab:.2e}")
print(f"  exact I        : min {ex.min():.2e}  max {ex.max():.2e}")
print(f"  double-prec. I : min {fl.min():.2e}  max {fl.max():.2e}")
print(f"  exact argmin = {int(np.argmin(ex))}, double-precision argmin = {int(np.argmin(fl))}")
