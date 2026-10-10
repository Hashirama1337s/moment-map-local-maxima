# Local maxima of the moment-map norm on a unitary orbit: a root-block sign test, mutual information, and Klyachko's cubicles

[![DOI](https://zenodo.org/badge/DOI/{{DOI}}.svg)](https://doi.org/{{DOI}})

**Moki & Julio · 2026** · DOI [{{DOI}}](https://doi.org/{{DOI}}) · paper: [PAPER.md](PAPER.md) · numerical ledger: [SUPPLEMENT.md](SUPPLEMENT.md)

## In plain words

A quantum system's description does not come with a built-in division into parts. Quantum mereology (paper references
[CS, CPR, St, SFD]) asks whether a Hamiltonian alone singles out a split of Hilbert space into subsystems. A widely used criterion is minimal interaction [LS, Zanardi,
Cao]: choose the split whose parts interact the least. In the language of symplectic geometry, that cost is minimised
exactly where the squared norm of a moment map is maximised, and the question becomes: where are the local maxima of that
function on a unitary orbit?

For a simple spectrum, the paper proves which torus-fixed splits are local optima: a torus-fixed point is a local maximum
if and only if its projection orders the eigenvalues strictly as the point itself does, across distinct weight spaces. For
two parties it also proves which classes are the global minimisers, both for minimal interaction and for mutual
information, and identifies a set of class counts that McConnell and Jennings had computed numerically without an
explanation. Degenerate spectra and the sharpness of the single-spectrum bounds stay open. Everything needed to check
the proofs and re-run the numbers is here.

## What the paper proves

Setting: H Hermitian with simple spectrum; K a compact connected subgroup of U(d) acting on the unitary orbit by
conjugation; P the orthogonal projection onto the Hermitian part of Lie(K) plus the scalars; D a point of the orbit.

- **Theorem 1 (any such K, F = ||.||^2).** Credited: critical points satisfy [P(D), D] = 0 (Kirwan, Ness); at a local
  maximum the K-orbit is symplectic and the stabiliser of the point equals the stabiliser of its moment-map image
  (Biliotti). Recorded here as a one-line corollary: a local maximum is fixed by a maximal torus T. Proved here: a T-fixed
  point D is a local maximum **if and only if** P(D) orders the eigenvalues of D strictly as D does across distinct
  T-weight spaces; it is then a strict local maximum modulo an explicit set of critical points (the K-orbit of D when the
  weights are multiplicity-free); a tie across weight spaces is never a local maximum.
- **Theorem 2 (any such K, general F).** The same test, with P(D) replaced by the gradient of F, holds for every C^2
  K-invariant functional F(P(H')) with positive-definite second derivative.
- **Theorem 3 (two parties, mutual information).** On the unitary orbit of a full-rank bipartite state with simple
  spectrum, the local minima of mutual information are the classical states whose probabilities are ordered like the
  product of their own marginals. Those orders are Klyachko's cubicles (realizable standard Young tableaux).
- **Theorem 4 (two parties, attainment).** Every cubicle is the unique global-minimiser class on a nonempty open set of
  spectra, for both functionals.
- **Corollary 5.** Combined with the classical global minimum of mutual information (Jevtic, Jennings, Rudolph), Theorem 4
  identifies the class counts 1, 5, 14, 42, 18 (shapes 2x2, 2x3, 2x4, 2x5, 3x3), which McConnell and Jennings had computed
  numerically and left unexplained, with cubicle counts. Appendix A gives exact Farkas certificates for the six
  non-realizable 3x3 tableaux; Appendix B lists exact integer witnesses for the 36 realizable 3x3 tableaux and for all 2xn
  tableaux, n = 2..5.

Limits, stated in the paper (Section 9): the spectrum must be simple; the cubicle counts bound the number of local minima
at a *single* spectrum only from above (3330 at 4x4, 124,187 for six qubits; the largest number found at one 4x4 spectrum
is 1114, so sharpness there is open); general Hamiltonian K-manifolds are not treated.

## What is in the repository

| Path | Contents |
|---|---|
| [PAPER.md](PAPER.md) | The theorem paper: definitions, credited known results (Section 3), Theorems 1-4 with complete proofs, corollaries, a line-by-line comparison with Georgoulas-Robbin-Salamon's Hessian lemma (Section 8), scope and open problems (Section 9), Appendices A-B. |
| [SUPPLEMENT.md](SUPPLEMENT.md) | The numerical ledger: exact combinatorics, attainment numerics, Hessian cross-checks, exact algorithms, censuses, the greedy-versus-exact comparison, material cut from the main text, and the recorded failures and corrections (S8). None of it is used in a proof. |
| `theory/` | Software: tableau enumeration and Hessian classification (`t1.py`), realizable-tableau counts (`realizable.py`), exact qubit split by parametric perfect matching (`qubit_exact.py`), exact grid branch-and-bound (`bb.py`), coherent-order enumeration for qubits (`door4*.py`), mutual-information census (`mi_census.py`), Farkas certificates and witnesses (`farkas3x3.py`), censuses (`census.py`, `maxmin.py`, `witness.py`, `torus.py`), greedy peeling (`door3.py`). |
| `theory/general/` | Checks of the general-K statements (`gen_core.py`, `run_case.py`, `mi.py`, `attain.py`) and their results. |
| `theory/proof/` | The degenerate-local-part search (`core.py`, `core_k.py`, `search.py`) and its results. |
| `theory/results/` | Every result file the supplement cites. The n = 6 coherent-order enumeration is shipped compressed as `coherent_n6.npz` (key `ranks`). |
| `probe/` | Implementation A (`impl_a.py`, GPU-capable) and implementation B (`impl_b*.py`), written separately from a shared specification, with their evaluation scripts. |
| `probe/independent_b2/` | Implementation B2: an independent exact qubit-split and coherent-order enumeration (integer double description with a completeness certificate), with its logs. |
| [ENV.txt](ENV.txt) | The software versions used. |

## Reproduce

Python 3.11 with NumPy, SciPy and networkx; `torch` with CUDA only for `impl_a.py`, `t1.py T12` and `bb.py` (CPU fallback
works, slower). Run the `../theory` commands from `probe/`; `regions.py` runs from `probe/independent_b2/`. The commands
are written with `python`; the supplement's ledger records the same invocations with the Windows launcher `py`. The
"expected" column is program output as recorded in the supplement. The class-count and single-spectrum figures in the
paper's abstract and Section 9 are 1, 5, 14, 42, 18, 3330, 124,187 and 1114; Appendix A states six non-realizable 3x3
tableaux and Appendix B states 36 realizable 3x3 witnesses.

| Result | Command | Output | Expected |
|---|---|---|---|
| Hessian formula vs autograd (cross-check of the block formula; SUPPLEMENT S3) | `python ../theory/t1.py T12 4 5 5` | `theory/results/T12_n4.json` | relative difference < 1e-8 |
| Local minimum <=> order test, d = 16, 1000 checks (SUPPLEMENT S3) | `python ../theory/t1.py T3 5 100` | `T3.json` | 1000/1000 |
| Realizable tableau counts (OEIS A211400; SUPPLEMENT S1) | `python ../theory/realizable.py` | `realizable.json` | 36 / 295 / 6660 |
| Mutual-information census (Corollary 5; SUPPLEMENT S1) | `python ../theory/mi_census.py` | `mi_census.json` | 1 / 5 / 14 / 42 / 18 |
| Farkas certificates and witnesses (Appendices A-B) | `python ../theory/farkas3x3.py` | `farkas3x3.json` | all exact checks pass |
| Attainment numerics (illustration of Theorem 4; SUPPLEMENT S2) | `python ../theory/general/attain.py tableaux 2 5`; `python ../theory/general/attain.py radius 3 3 7` | `theory/general/results/attain_syt_2x5.json`, `attain_3x3_7.json` | see SUPPLEMENT S2 |
| Exact qubit split = enumeration (SUPPLEMENT S4) | `python ../theory/qubit_exact.py E1a` | `E1a.json` | max difference <= 1e-12 |
| Exact grid branch-and-bound = enumeration (SUPPLEMENT S4) | `python ../theory/bb.py E2a` | `E2a.json` | EXACT |
| Coherent Boolean term orders (OEIS A009997; SUPPLEMENT S5) | `python ../theory/door4.py ENUM n` (n = 3..6) | `coherent_n*.npy` | 2 / 14 / 516 / 124,187 |
| Independent enumeration with completeness certificate (SUPPLEMENT S4) | `python regions.py 6` (from `probe/independent_b2/`) | `regions_n6.log` | 124,187 regions, 0 missing |
| Single-spectrum census (SUPPLEMENT S5) | `python ../theory/census.py 100` | `census.json` | table in SUPPLEMENT S5 |

## What is new and what is credited

Credited in Section 3 of the paper, each with a locator, and not claimed here: the critical-point condition (Kirwan-Ness;
matrix form Chu-Driessel Lemma 3.1 and Woodward Lemma 7.1.1; gradient form Georgoulas-Robbin-Salamon Lemma 3.1), the
symplectic-orbit and stabiliser results at a local maximum (Biliotti), the global minimum at product-diagonal states
(Tegmark, with the Soulas critique and a convexity proof; Jevtic-Jennings-Rudolph for mutual information), the one-factor
case (Brockett, via Helmke-Moore), cubicles and sweep polytopes (Klyachko;
Castillo-Labbe), realizable tableaux (Mallows-Vanderbei; OEIS A211400), coherent Boolean term orders (Maclagan; Fine-Gill;
OEIS A009997), and the McConnell-Jennings counts themselves. Section 8 compares the Hessian used here, line by line, with
Georgoulas-Robbin-Salamon's Lemma 3.5.

New in the paper: Theorems 1-2 (b)-(d) (the root-block sign test, the variance identity, the exact tie formula and the
slice argument), Lemma 5.1 (the second-order torus statement with an explicit ascent direction), Theorems 3 and 4, and
the Farkas certificates of Appendix A.

Software in this repository, not part of the mathematical claims: the exact algorithms (parametric perfect matching for
qubit splits, branch-and-bound over partial tableaux, LP-pruned coherent-order enumeration) and the three implementations.

## Honesty record

The supplement keeps the failures and corrections made during this work (S8), the cross-check tolerances actually
observed (S3-S4), and the material cut from the main text with the reason (S7). The paper was read adversarially by
independent reviewers before release; this is not journal peer review.

This work was done with AI assistance. Nothing in it should be relied on without reading the proofs; they are complete and
in the paper.

## Cite

```
Moki & Julio (2026). Local maxima of the moment-map norm on a unitary orbit: a root-block sign test, mutual
information, and Klyachko's cubicles. Version 1.0. DOI {{DOI}}.
```

See [CITATION.cff](CITATION.cff). Text and figures: CC BY 4.0 ([LICENSE](LICENSE)); code: MIT ([LICENSE-CODE](LICENSE-CODE)).
