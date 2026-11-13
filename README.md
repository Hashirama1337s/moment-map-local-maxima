# Splitting a quantum system into parts: an exact test for the stable splits

**A quantum system does not come with built-in parts; for a simple spectrum (no two energy levels equal) we prove an exact test for which ways of splitting it are stable, and it explains counts McConnell and Jennings had found by computer.**

[![DOI 10.5281/zenodo.23275890](brand/doi.svg)](https://doi.org/10.5281/zenodo.23275890) ![Licence: text and data CC BY 4.0, code MIT](brand/licence.svg) [![Moki & Julio lab map](brand/lab.svg)](https://github.com/Hashirama1337s)

Moki & Julio · 2026 · version 1.1

**Read** [PAPER.md](PAPER.md) · **Numbers** [SUPPLEMENT.md](SUPPLEMENT.md) · **New in v1.1** [ADDENDUM-C.md](ADDENDUM-C.md),
[ADDENDUM-D.md](ADDENDUM-D.md)

## In plain words

A quantum system is described by its **Hamiltonian**: the list of its energy levels (its **spectrum**) and how they are
arranged. Nothing in that description says "this is part A and that is part B". **Quantum mereology** asks whether the
Hamiltonian alone singles out a split into subsystems. A widely used rule is **minimal interaction**: choose the split
whose parts interact the least.

That rule has many local winners: splits that cannot be improved by any small change. We call them **stable** here; the
paper says "local optima". In symplectic geometry (a branch of geometry whose language the paper uses), the interaction
cost is smallest exactly where a related quantity, the squared size of a "moment map", is largest (both terms are in
"Words you will meet"). So the question becomes: where are the local maxima of that quantity?

A picture that helps (it is only a picture): split a 6-level system into a 2-level part and a 3-level part, and think of a
2 x 3 grid of seats, one row per state of the first part and one column per state of the second. Seat each energy level
in one cell. Where the picture breaks: quantum mechanics allows far more than seatings, because the states of the
system can be mixed together in any (unitary) way, not just reseated. A known result of Biliotti, with a one-line
corollary recorded in the paper, shows that every stable split is a seating of this kind once the states inside each
part are suitably mixed among themselves (a "change of basis" inside each part).

What we found, **for a simple spectrum**: an exact test that says which seatings are stable (Theorems 1 and 2). For two
parts, a test of the same kind classifies the splits that leave the parts least correlated (mutual information).
Combined with the classical result of Jevtic, Jennings and Rudolph that the least-correlated state is always a seating,
it explains the counts 1, 5, 14, 42, 18 for grids 2x2, 2x3, 2x4, 2x5 and 3x3, which McConnell and Jennings had computed
numerically. Degenerate spectra (two or more equal energies) stay open. The proofs are complete and in the paper.

## At a glance

*The exact claims, for specialists. Terms are explained in "How it works" and "Words you will meet".*

Setting: H Hermitian with simple spectrum; K a compact connected subgroup of U(d) acting on the unitary orbit by
conjugation; P the orthogonal projection onto the Hermitian part of Lie(K) plus the scalars; D a point of the orbit.

- **Theorem 1 (any such K, F = ||.||²).** Credited: critical points satisfy [P(D), D] = 0 (Kirwan, Ness); at a local
  maximum the K-orbit is symplectic and the stabilisers agree (Biliotti); recorded here as a one-line corollary, a local
  maximum is fixed by a maximal torus T.
  Proved here: a T-fixed point D is a local maximum **if and only if** P(D) orders the eigenvalues of D strictly as D does
  across distinct T-weight spaces; it is then a strict local maximum modulo an explicit set of critical points; a tie
  across weight spaces is never a local maximum.
- **Theorem 2.** The same test, with P(D) replaced by the gradient of F, for every C² K-invariant F(P(H')) with
  positive-definite second derivative.
- **Theorem 3 (two parties, mutual information).** On the unitary orbit of a full-rank bipartite state with simple
  spectrum, the local minima of mutual information are the classical states whose probabilities are ordered like the
  product of their own marginals: Klyachko's cubicles.
- **Theorem 4 and Corollary 5.** Every cubicle is the unique global-minimiser class on a nonempty open set of spectra,
  for both functionals; with Jevtic-Jennings-Rudolph this identifies the counts 1, 5, 14, 42, 18 with cubicle counts.
  Appendix A gives exact Farkas certificates for the six non-realizable 3x3 tableaux; Appendix B exact integer witnesses.
- **v1.1, Addendum C.** One spectrum cannot make all 3,330 realms of 4x4 stable at once, nor all 516 of five qubits; it
  can at 2x3, 2x4, 2x5, 3x3 and three and four qubits. The 4x4 maximum lies in [1209, 2845]. Every answer has an
  exact integer certificate.
- **v1.1, Addendum D.** Near absolute zero, the least-correlated split of a thermal state is the one that keeps the first
  level exciting both parts as high as possible: the lowest levels fill one row and one column (Theorem D1,
  Corollary D2, for spectra off finitely many hyperplanes).

Credited results and locators: Section 3 of [PAPER.md](PAPER.md). Line-by-line comparison with Georgoulas-Robbin-Salamon's
Hessian lemma: Section 8.

Full technical record: [TECHNICAL.md](TECHNICAL.md).

## What we do not claim

- **Degenerate spectra.** The proofs use distinct energies; nothing is claimed when two or more are equal.
- The cubicle counts bound the number of stable classes at a *single* spectrum only from above (3330 at 4x4, 124,187 for
  six qubits). Addendum C shows the bound is not attained at 4x4, 2x6, 2x8, 3x4 or five qubits; the exact 4x4 maximum is
  only bracketed, in [1209, 2845].
- General Hamiltonian K-manifolds are not treated.
- The equivalence of this landscape with the sorted-eigenvalue landscape minimised by released code is an observation,
  not a theorem.
- Addendum D: a finer leading-order tie-break formula (Conjecture D3) is tested, not proved.
- Numerics (Hessian cross-checks, censuses, timings, branch-and-bound, greedy ratios) are not used in any proof.
- The paper was read adversarially by independent reviewers before release; this is not journal peer review.

## How it works

### The idea

For two parts, the test reads like this. Seat the energies in the grid. Now build the "additive fit" of that seating:
for each cell, take its row's average plus its column's average minus the average of everything. The seating is stable
**exactly when the fit puts the cells in the same order as the energies do**.

So the only orders that can ever be stable are orders that some "row number plus column number" can produce. Those
orders have a name, **Klyachko's cubicles** (also called realizable standard Young tableaux), and each one is the winning
pattern for some list of energies. Counting them gives the McConnell-Jennings numbers.

```mermaid
flowchart LR
  A[A list of energies<br/>with no two equal] --> B[Seat them in a grid:<br/>rows = part A, columns = part B]
  B --> C[Additive fit: row average<br/>+ column average - overall average]
  C --> D{Does the fit order the<br/>cells like the energies?}
  D -->|yes| E[Stable split]
  D -->|no| F[Not stable: a small<br/>change improves it]
```

### A worked example

Write the ranks 1 to 9 of the energies (1 = lowest) into a 3 x 3 grid so that every row and every column increases.
There are 42 ways to do that. Which of them can come from "row number + column number"?

**One that can.** The grid 123 / 456 / 789 (rows separated by "/"). Take row numbers x = (0, 3, 6) and column numbers
y = (0, 1, 2). The sums x_a + y_b are 0, 1, 2 / 3, 4, 5 / 6, 7, 8: they rise in exactly the order of the grid.

**One that cannot.** The grid 126 / 347 / 589. Write L(a, b) = x_a + y_b, with rows and columns numbered 0 to 2. Reading
the grid gives three inequalities that would have to hold:

    x0 + y1 < x1 + y0,    x2 + y0 < x0 + y2,    x1 + y2 < x2 + y1.

Add them up. Both sides become x0 + x1 + x2 + y0 + y1 + y2, so the sum says "something < itself", which is impossible. No
choice of numbers works. Such an impossibility proof is called a **Farkas certificate**.

Exactly 6 of the 42 grids fail like this; 36 pass. A grid and its mirror image (swap rows and columns) describe the same
split when both parts have 3 levels, so the 36 pair up into **18** patterns: 18 kinds of seating, each of which is the
best split for some list of energies. That is the McConnell-Jennings count for 3 x 3. It is not the number of stable
splits of one system: a single list of energies has at most 18 stable classes. For 2 x 2 there are 2 grids, one pair,
so the count is **1**. In fact, for 2 x 2 there is exactly one stable split class for every simple spectrum.

## Reproduce and reuse

Questions this lets you answer:

- Which orders of a grid can ever be a stable split, and which are ruled out (with a proof you can check)?
- Can one list of energies make every realm of a given shape stable at once?
- How many least-correlated classes does a grid shape have?

**Re-run the core checks** (Python 3.11 with NumPy, SciPy and networkx; `mi_census.py` also imports torch, CPU is
enough). Run from `probe/`:

```
git clone https://github.com/Hashirama1337s/moment-map-local-maxima
cd moment-map-local-maxima/probe
python ../theory/check_d5.py        # exact check of every Addendum C proof object, pure Python, seconds
python ../theory/farkas3x3.py       # Appendix A certificates and Appendix B witnesses, seconds
python ../theory/mi_census.py       # McConnell-Jennings counts, about half a minute
```

`check_d5.py` ends with `ALL CHECKS PASS`, `farkas3x3.py` with `ALL EXACT CHECKS PASS`, and `mi_census.py` prints the
class counts:

```
2x2 {'classes': 1, 'nonrealizable_argmins': 0, 'spectra': 20000, 'realizable_classes': 1, 'SYT': 2}
2x3 {'classes': 5, 'nonrealizable_argmins': 0, 'spectra': 20000, 'realizable_classes': 5, 'SYT': 5}
2x4 {'classes': 14, 'nonrealizable_argmins': 0, 'spectra': 20000, 'realizable_classes': 14, 'SYT': 14}
2x5 {'classes': 42, 'nonrealizable_argmins': 0, 'spectra': 20000, 'realizable_classes': 42, 'SYT': 42}
3x3 {'classes': 18, 'nonrealizable_argmins': 0, 'spectra': 20000, 'realizable_classes': 18, 'SYT': 42}
```

The census is numerical (20,000 random spectra per shape) and is not used in any proof. Every other command, with its
expected output, is in the Reproduce table of [TECHNICAL.md](TECHNICAL.md).

**Reuse: test your own grid.** Save this as `probe/mygrid.py` and run `python mygrid.py` from `probe/`. Grids are written
with ranks 0 to 8 (0 = lowest energy); the two below are the examples of the worked example above.

```python
import sys, numpy as np
sys.path.insert(0, '../theory')
import farkas3x3 as F

good = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])   # tableau 123/456/789, written with ranks 0..8
bad  = np.array([[0, 1, 5], [2, 3, 6], [4, 7, 8]])   # tableau 126/347/589
print(F.find_witness(good))            # row numbers x and column numbers y whose sums x_a + y_b rise in this order
print(F.find_witness(bad))             # None: no such numbers exist
print(F.find_pair_certificate(bad))    # three inequalities that add up to "something < itself"
```

It prints `([0, 3, 6], [0, 1, 2])`, then `None`, then the three inequalities of the worked example as cell pairs:
`[((0, 1), (1, 0), 1), ((2, 0), (0, 2), 1), ((1, 2), (2, 1), 1)]`. `find_witness` takes any m x n grid and verifies its
answer in exact integer arithmetic before returning it; `find_pair_certificate` searches certificates of up to four
inequalities.

## Words you will meet

- **Hamiltonian** - the description of a quantum system's energy; its eigenvalues are the energy levels.
- **Spectrum; simple spectrum** - the list of energy levels; "simple" means no two are equal ("degenerate" if some are).
- **Split (tensor-product structure)** - a way of viewing one system as two or more parts.
- **Minimal interaction** - the rule "choose the split whose parts interact least".
- **Stable split (local optimum, realm)** - a split that no small change improves.
- **Symplectic geometry** - a branch of geometry; the paper states the problem in its language.
- **Moment map** - a tool from symplectic geometry; here, the projection P onto what the parts can do on their own.
- **Unitary change; change of basis** - the ways quantum states may be mixed with each other; inside one part, a
  relabelling and mixing of that part's own states.
- **Mutual information** - how much knowing one part tells you about the other.
- **Additive fit** - row average plus column average minus overall average.
- **Cubicle (realizable standard Young tableau)** - a grid order that some "row number + column number" produces.
- **Farkas certificate** - inequalities that add up to an impossibility, proving that no numbers fit.

## How to cite

```
Moki & Julio (2026). Local maxima of the moment-map norm on a unitary orbit: a root-block sign test, mutual information, and Klyachko's cubicles. Version 1.1. Zenodo. https://doi.org/10.5281/zenodo.23275890
```

The DOI always resolves to the newest archived version. Version 1.0 is 10.5281/zenodo.23275891. Machine-readable: [CITATION.cff](CITATION.cff).
Text and figures: CC BY 4.0 ([LICENSE](LICENSE)); code: MIT ([LICENSE-CODE](LICENSE-CODE)).

## Related projects

- [certified-qubit-peel](https://github.com/Hashirama1337s/certified-qubit-peel): a published qubit-building search, checked against certified best splits
- All projects: [the Moki & Julio lab map](https://github.com/Hashirama1337s)

---

*Prepared with AI assistance; what was checked, and by what, is stated above.*
