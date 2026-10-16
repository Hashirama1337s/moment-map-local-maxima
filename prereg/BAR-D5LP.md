# BAR-D5LP (sealed before the run). Door 5 as an exact LP: can one spectrum realise every realm at once?

**In plain words.** A "realm" is one stable way of splitting a 16-level quantum system into two 4-level parts. Our published
theorem (Zenodo 10.5281/zenodo.23275890) says each realm is a stable minimum exactly when the additive fit of the placed
spectrum orders the cells like the placement does. That is a list of strict linear inequalities on the sorted energies. So
"can one spectrum have all 3,330 realms at once?" is one linear program: feasible = an exact rational spectrum we can check by
hand-rule arithmetic; infeasible = a Farkas (Gordan) certificate, a nonnegative integer combination of the inequalities that
sums to zero, also checked exactly. Hill-climbing (theory/maxmin.py) reached 1,114.

## Object (fixed)
- Shape m x n, sorted spectrum lam_0 < ... < lam_{mn-1}. Placement = realizable standard Young tableau T (rank grid, t1.syt_all
  convention). E = lam[T]; L = row mean + column mean - grand mean (t1.fit). Realm T holds iff L is strictly increasing along T's
  rank order (mn - 1 consecutive strict inequalities); this is the D4 / order-isomorphism criterion (realizable.py check 0
  mismatches; T3 1000/1000 autograd). Rows scaled by mn to integers.
- Square grids: T and its transpose give the same realm (class count = realizable / 2). All realizable T are included.
- Qubit version (n qubits): placement = coherent order (theory/results/coherent_n{n}.npy), L = projection on span{1, Z_j}
  (door4.design), same consecutive inequalities, rows scaled to integers.
- Joint system: all realm inequalities + lam strictly increasing, all homogeneous. LP: maximise margin t subject to every row
  >= t, lam_{mn-1} - lam_0 <= 1. Feasible iff t* > 0. Gordan: infeasible iff a nonzero y >= 0 with A^T y = 0 exists.

## Controls (must pass before the predictions are scored; if any fails, the run is VOID, not FAILED)
- C1 (encoding vs instrument): on the stored hill-climb spectrum (results/maxmin_4x4.json, 1114) my inequality evaluator
  counts exactly the same number of realms as t1.fit + t1.xmin (count // 2), both = 1114.
- C2 (known attained case): 3x3 joint LP over all 18 classes is FEASIBLE and the exact rational witness passes every realm
  exactly (FINDINGS 6.5: 18/18 attained). Also 2x2 (1 class) feasible. Qubits n = 3 (2) and n = 4 (14) FEASIBLE (LOG door 5).
- C3 (null): a single non-realizable 3x3 tableau gives an INFEASIBLE LP with an exactly verified Gordan certificate.
- C4 (exactness): every reported spectrum is re-checked in Fraction arithmetic; every certificate is an integer vector,
  nonnegative, nonzero, with A^T y = 0 exactly.

## Predictions (scored)
- P1: the 4x4 joint LP (3,330 classes) is INFEASIBLE, with an exactly verified certificate. FAILED if feasible.
- P2: the 5-qubit joint LP (516 orders) is INFEASIBLE, exactly verified. FAILED if feasible.
- P3: an LP-greedy extension starting from the 1,114 realms of the hill-climb spectrum (add realms in a fixed seeded random
  order, keep each one whose addition stays feasible with margin > 1e-9, then exact recheck of the final spectrum) reaches a
  spectrum with MORE than 1,114 realms, exactly verified. FAILED if <= 1,114.

## Descriptive (reported, not scored)
- Joint LP verdicts for 2x3, 2x4, 2x5, 2x6, 2x8, 3x4 (bipartite) and certificate support sizes.
- Upper bound on the true maximum: 3,330 minus the number of pairwise disjoint infeasible subsets found (each must lose one).
- MILP for the exact maximum only if a probe projects under 30 minutes; otherwise report the bracket [lower, upper].

Seeds: greedy order rng = numpy default_rng(5). Interpreter: py -3.11 (scipy 1.17.1, HiGHS). Code: theory/d5lp.py.
