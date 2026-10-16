# Addendum C. How many realms can one spectrum hold? The single-spectrum bound is not attained at 4x4 or five qubits

**Moki & Julio (AI-assisted)** · October 2026 · addendum to "Local maxima of the moment-map norm on a unitary orbit"
(version 1.1; DOI 10.5281/zenodo.23275890)

## In plain words

Split a 16-level quantum system into two 4-level parts. There are 3,330 essentially different ways of doing it that can be
stable (local optima of minimal interaction). The paper shows that each of them is stable for some list of 16 energies. It
leaves one question open: can a single list of energies make all of them stable at once? The answer is no. By the paper's
order test, each stable way is a short list of linear inequalities on the sorted energies. "All of them at once" is then one
linear program, and its answer comes with a proof anyone can check by integer arithmetic. The proof is a list of nonnegative
whole-number weights; summing the inequalities with those weights gives 0 > 0. The same holds for five qubits (516 stable
ways). It also holds for the 2x6, 2x8 and 3x4 grids. At 2x3, 2x4, 2x5 and 3x3, and for three and four qubits, one list does
make all of them stable at once. For 4x4 the true maximum lies between 1,209 and 2,845. Two specific stable ways can never
coexist at all.

## 1. Setting

Notation is the paper's (Sections 2 and 7). Fix shape m x n, d = mn, and a sorted simple spectrum lam_0 < lam_1 < ... <
lam_{d-1}. A standard Young tableau T of shape m x n assigns rank T(a,b) in {0, ..., d-1} to cell (a,b). The placement is
E_ab = lam_{T(a,b)}, and its additive fit is l_ab = (row mean) + (column mean) - (grand mean) of E. The two-party case of
Theorem 1(b) is Corollary 6. The class of T is a local minimum of minimal interaction (a "realm") iff l orders all d cells
strictly as E does. Because E increases along T's order, this is the chain of d - 1 strict inequalities

    l_{c_{k+1}} - l_{c_k} > 0,   k = 0, ..., d-2,     where c_k is the cell of rank k.                       (R_T)

Each l_c is a linear form in lam with coefficients in (1/d)Z. Multiplying by d gives integer rows, so (R_T) reads A_T lam > 0
for an integer (d-1) x d matrix A_T. Only realizable T can satisfy (R_T): l is an outer sum, and l ordering the cells as T
does is the definition of realizability. For m = n, T and its transpose give the same realm (the SWAP). We keep one
representative per class: rank 1 sits at cell (0,1). Qubits (Corollary 7) work the same way. A realm is a coherent Boolean
term order O of {0,1}^n; the fit is the projection onto span{1, Z_1, ..., Z_n}, and 2^n times that projection is an integer
matrix. Representatives are the 2, 14, 516 coherent orders of n = 3, 4, 5 with weights 0 < w_1 < ... < w_n (the paper's cone
w_1 > ... > w_n > 0 with the qubits relabelled).

**Simultaneous realisation.** A spectrum realises the set S of realms iff

    G lam > 0  and  A_T lam > 0 for all T in S,                                                              (J_S)

where G lam > 0 says lam_{k+1} - lam_k > 0. All rows are homogeneous. So (J_S) has a solution iff the LP "maximise t subject
to every row >= t, lam_0 = 0, lam_{d-1} <= 1" has optimum t* > 0. By Gordan's theorem (J_S) has no solution iff there are
integers y >= 0, not all zero, with y^T [G; A_S] = 0. Such a y is a certificate. Adding the rows with weights y gives a form
that is identically zero; yet if (J_S) held, that form would be strictly positive at lam.

**What is computed and what is proved.** A floating-point LP (HiGHS via SciPy) is used only to find candidates. Every
accepted object is re-verified in exact integer or rational arithmetic, independently of the solver. A feasibility claim
comes with an exact rational spectrum that satisfies every row of (J_S). An infeasibility claim comes with an integer
certificate y on an explicit support. The integer identity y^T [G; A_S] = 0 is the proof. A stand-alone checker
(`check_d5.py`) rebuilds every row from the definitions above, without the LP code.

## 2. Results

**Proposition C1 (all realms at once).** Let S be the set of all realms of the shape. The rational spectrum or the
certificate is in `certificates/`.

| shape | realms (= cubicle bound) | all at once? | evidence (exact) |
|---|---|---|---|
| 2x2 | 1 | yes | rational spectrum, margin 0.333 |
| 2x3 | 5 | yes | rational spectrum, margin 1/5 (exact) |
| 2x4 | 14 | yes | rational spectrum, margin 0.143 |
| 2x5 | 42 | yes | rational spectrum, margin 0.0863 |
| 3x3 | 18 | yes | rational spectrum, margin 1/8 (exact) |
| 3 qubits | 2 | yes | rational spectrum, margin 0.143 |
| 4 qubits | 14 | yes | rational spectrum, margin 0.0606 |
| 2x6 | 132 | **no** | certificate on 12 rows, touching 10 realms |
| 2x8 | 1,430 | **no** | certificate on 16 rows, touching 15 realms |
| 3x4 | 295 | **no** | certificate on 12 rows, touching 11 realms |
| **4x4** | **3,330** | **no** | certificate on 16 rows, touching 15 realms; largest weight 2,612,252 |
| **5 qubits** | **516** | **no** | certificate on 18 rows, touching 16 orders |

Margins are in the normalisation lam_0 = 0, lam_{d-1} <= 1, with rows scaled by d. They are floating-point LP optima; the
shipped rational spectra attain them to within 1e-11 (exactly for 2x3 and 3x3). Feasibility rests only on the rational spectra. So the cubicle bound of Corollary 6 is
attained at 2x2-2x5 and 3x3. The coherent-order bound of Corollary 7 is attained at three and four qubits. Neither is
attained at 2x6, 2x8, 3x4, 4x4 or five qubits. Six qubits (124,187 orders) was not run. We do not know whether infeasibility at
five qubits implies it at six.

**Proposition C2 (an incompatible pair).** The two 4x4 realms below (ranks 1-based, rows = first factor) are never local
minima at the same spectrum. Write row (k) for the inequality comparing the cells of 1-based ranks k and k + 1. The
certificate uses 4 gap rows (lam_6 - lam_5, lam_8 - lam_7, lam_9 - lam_8, lam_13 - lam_12 in 1-based level labels) with
weight 16 each. It also uses rows (3), (6), (8) of the first realm with weights 3, 3, 1, and rows (2), (4), (7), (9), (11) of
the second with weights 1, 3, 2, 3, 3. These rows sum to the zero form; they are listed in `certificates/pair_4x4.json`.

```
 1  2  3 11        1  2  4 10
 4  7  8 14        3  5  8 11
 5  9 12 15        6  7  9 14
 6 10 13 16       12 13 15 16
```

**Proposition C3 (bracket at 4x4).** Let N(lam) be the number of realms of a simple spectrum lam. Then

    1,209 <= max over lam of N(lam) <= 2,845.

- Lower bound: an explicit rational spectrum with N = 1,209, checked exactly, in `certificates/lower_1209.json`.
- Upper bound: 485 pairwise disjoint sets of realms (sizes 2-14), each with an exact certificate that its members cannot all
  hold at once. Each set loses at least one realm, so N <= 3,330 - 485. The certificates are in
  `certificates/upper_cores.json`.

## 3. How the numbers were found (not part of the proofs)

- **Lower bounds.**
  - A hill-climb on the spectrum (paper, Corollary 6) gave 1,114.
  - An LP-greedy started from those 1,114 realms. It visits the other realms in a fixed random order (seed 5) and keeps each
    one whose addition leaves (J_S) feasible. It reached 1,164 after 2,173 LPs.
  - A mixed-integer program with a big-M switch per realm, stopped after 180 s, returned an incumbent of objective 1,199.
    Its spectrum realises 1,209 realms on exact recount, because some realms whose switch was off hold anyway. The dual bound
    at the stop was 3,329. The exact maximum would need a much stronger formulation; it was not run.
- **Upper bound.** Solve (J_S) for the remaining realms and read the certificate's realms. Shrink them to a minimal
  infeasible set by deletion. Verify that set exactly, remove it, and repeat. After 485 sets, the remaining 356 realms are
  jointly feasible (floating-point LP; not re-verified exactly, and not used in any bound).
- **Pre-registration.** The predictions (4x4 and 5 qubits infeasible; LP-greedy above 1,114) and the controls were written and
  hashed before the run (`prereg/BAR-D5LP.md`, sha256 68ff3761...). All three predictions held.
- **Controls.**
  - The row construction reproduces the paper's census instrument on its stored 1,114-realm spectrum: 1,114 realms on both.
  - It counts 3,330 realizable classes among 24,024 tableaux.
  - The known attained cases (2x2, 3x3, three and four qubits) come out feasible.
  - A single non-realizable 3x3 tableau comes out infeasible, with a 3-row certificate.

## 4. Scope

- **Proved here** (by exact certificates):
  - Proposition C1 for the shapes listed.
  - Proposition C2.
  - Both bounds of Proposition C3.
- **Not settled:**
  - The exact maximum at 4x4.
  - Any maximum or bracket at five qubits beyond "less than 516" (the best spectrum found has 233).
  - All shapes not listed, including six qubits.
- **Depends on:** Theorem 1(b) / Corollaries 6-7 of the paper, which identify realms with the order chains (R_T).
- **Degenerate spectra:** not treated, as in the paper.

## References

- The paper (version 1.0 of this record): Theorem 1(b), Corollaries 6 and 7, and the counts OEIS A211400 and A009997 cited
  there.
- Gordan's theorem of the alternative is classical. The certificates are checked directly, so no step depends on its proof.

## Reproduce

From `probe/`:

- `python ../theory/d5lp.py CONTROLS`, then `python ../theory/d5lp.py JOINT`: the table of C1. Takes about 1 minute.
- `python ../theory/d5lp.py GREEDY`: about 16 minutes.
- `python ../theory/d5lp.py UPPER`: about 15 minutes.
- `python ../theory/d5lp_milp_probe.py`: 180 s. MILP solvers are not deterministic across machines, so the exact lower
  bound is the shipped spectrum, not this run.
- `python ../theory/d5lp_export.py`: writes `certificates/`.
- `python ../theory/check_d5.py`: pure-Python exact check of every certificate and spectrum, independent of the LP code.
