# Supplement to PAPER v0.4: numerical verification ledger

DRAFT v0.4 (2026-10-10; the v0.3 supplement with the additions requested by the second referee report, in S1 and S4). INTERNAL. This supplement holds the numerical material that v0.2 carried in its main text. **No
number below is used in a proof in PAPER.md.** The exact combinatorial checks in S1 are the one exception: they are
exact integer computations, cited by Corollary 5 and Appendix A.

**Sources.** Every number below is copied from paper/PAPER.md (v0.2), whose numbers were traced to result files in
paper/NUMBERCHECK.md, or was produced in this revision (S1, marked NEW). Where NUMBERCHECK found that the support is text
only, this is said.

**Implementations.**
- Implementation A is the primary code, in theory/ and probe/impl_a.py.
- Implementation B is an independent, blind NumPy implementation (probe/impl_b*).
- Implementation C is an independent, blind, exact-arithmetic implementation (independent implementation B2, directory
  probe/independent_b2/).
- The proof-search code is in theory/proof/. The general-group code is in theory/general/.

**Environment.** ENV.txt: Python 3.11.9, torch 2.11 + CUDA 12.8, NumPy 2.4.6, SciPy 1.17.1, networkx 3.6.1.
- Interpreter: python.
- README.md commands are run from probe/ (written there as `py`).
- Integrity hashes: MANIFEST.sha256 (`sha256sum -c MANIFEST.sha256`). It predates this revision and does not cover
  theory/farkas3x3.py.

## S1. Exact combinatorics (used by Corollary 5 and Appendix A)

| Check | Command | Output | Result |
|---|---|---|---|
| NEW: exact witnesses and Farkas certificates | `python theory/farkas3x3.py` (from the project root; ~2.5 s, CPU) | theory/results/farkas3x3.json | 3x3: 42 SYT; 36 with exact integer witnesses; 6 with exact Farkas certificates. 2x2, 2x3, 2x4, 2x5: 2, 5, 14, 42 SYT, all with exact integer witnesses. "ALL EXACT CHECKS PASS". v0.4: the JSON now also stores the 2xn witnesses; they are printed with the 3x3 ones in Appendix B of the paper |
| Realizable SYT counts (floating LP, HiGHS) | `py ../theory/realizable.py` | theory/results/realizable.json | 2x4 14/14; 3x3 36/42; 2x8 1430/1430; 3x4 295/462; 4x4 6660/24024. These equal OEIS A211400 |
| Non-realizable 3x3 list, independent code path | `python attain.py tableaux 3 3` (in theory/general/) | theory/general/results/attain_syt_3x3.json | the same 6 tableaux (farkas3x3.py checks the match: True) |
| Genericity counterexample | `python attain.py tie` | attain_tie_2x4.json | lambda = (0..7), 2x4: three zero-residual classes |

**How the NEW script works.**
- The floating LP is used only to find candidates.
- Witnesses are rounded to integers and checked exactly.
- Certificates are searched exhaustively over supports of size 2, 3 and 4, among the pair inequalities L(p) < L(q). Each is
  solved exactly with fractions.Fraction, scaled to coprime integers, and checked exactly in two forms: pair and consecutive.
- All six minimal certificates have support 3 with multipliers 1.
- The six tableaux form three transpose pairs.

## S2. Attainment numerics (illustration of Theorem 4; not used in its proof)

Command: `python attain.py radius <dA> <dB> 7` for the five shapes, and `python attain.py direct 7`, in theory/general/.
Outputs: theory/general/results/attain_*_7.json.
- **Witness spectra.** There are 99 (2 + 5 + 14 + 42 + 36). In every case the zero set of C1 at lambda*, and of I at p*, is
  exactly the class of T (with its transpose for square shapes).
- **Certified radii.**
  - C1: the L2 radius is at least 0.043 in every shape.
  - MI: the computed L1 radius is positive. Its size (2.8e-7) is an artefact of a loose continuity bound and carries no
    meaning beyond positivity.
- **Direct orbit minimisation, 2x3.** 32 Haar starts per case, at spectra perturbed to 0.9 x the certified radius. The orbit
  minimum equals the class-T value and lies below the second-best class in C1 5/5 and MI 5/5 cases.

## S3. Hessian and classification cross-checks (Theorems 1-3 are proved without them)

| Check | Command | Output | Result |
|---|---|---|---|
| Closed-form Hessian vs autograd, bipartite | `py ../theory/t1.py T12 4 5 5`; `py ../theory/t1.py T12 6 3 1` | theory/results/T12_n4.json, T12_n6.json | d = 16: <= 2.9e-15. d = 64: <= 6.8e-15 (6 instances) |
| Independent analytic Hessian (implementation B) | probe/impl_b_T1_REPORT.md | | d = 16, 30 assignments: agrees with A's closed form to 6.3e-15 |
| Local minimum <=> order test, d = 16 | `py ../theory/t1.py T3 5 100` | T3.json | 1000/1000 |
| Order test vs full Hessian test | LOG/FINDINGS text only | (no result file) | 720,720/720,720. Support is text only (NUMBERCHECK C1.8) |
| General compact K (16 groups) | theory/general/ (gen_core.py, run_case.py, mi.py) | theory/general/results/ | Block formula vs analytic Hessian <= 6.6e-14. Analytic vs autograd <= 1.1e-14. MI closed form vs autograd <= 1.8e-14 (221 T-fixed points). Toral and variance identities <= 3.2e-14 on 1230 root blocks. 7,078 minima (seeds 11 and 12), 0 violations |
| Ties, equally spaced spectra (10 groups) | theory/general/extras.py | ties_*.json | 872/872 root-pair ties are saddles. 192/192 non-root ties drop along the Givens curve (formula error 6e-15) |
| Critical points of all indices | theory/proof/ (core.py, search.py, run_search.sh; search_k.py) | theory/proof/*.json | 13,084 bipartite and 1,967 k-partite converged critical points, 0 violations |
| Ascent bound of Lemma 5.1 | theory/general/, theory/proof/explicit_dir.py | lemma3_*.json | Never violated (460 points; and 40 points with smallest ratio 1.0000014). Tightness is NOT claimed |

## S4. Exact algorithms built on Corollary 7 and on the bipartite case

| Item | Command | Output | Result |
|---|---|---|---|
| Coherent orders, LP-pruned DFS (A) | `py ../theory/door4.py ENUM n` | coherent_n*.npy, enum_n6.log | 2, 14, 516, 124,187 for n = 3..6; n = 6 took 1,686 s, once |
| Coherent orders, exact double description (C) | `python regions.py 6` (in probe/independent_b2/; likewise `python regions.py 5`) | regions_n*.npz, regions_n6.log | 1, 2, 14, 516, 124,187 for n = 2..6. Facet-closure certificate: every non-wall facet crossing stays in the set (n = 5: 2,654 crossings; n = 6: 872,018), 0 missing |
| Chamber-closure check arithmetic (Corollary 7), n = 6 | same command | regions_n6.log | Arrangement: hyperplanes sum_{j in S} w_j = sum_{j in S'} w_j inside the cone w_1 > ... > w_n > 0; 98 of the 364 hyperplanes cut the cone. Depth-first exact-integer double description: 248,373 DFS nodes, 124,187 leaves (regions), 124,187 distinct orderings, 91.4 s. Each region has 6-24 extreme rays and 6-13 facets. Facets found two ways (combinatorial and SVD rank test): 0 disagreements. For every facet that is not a chamber wall, the ordering just across it was computed exactly and looked up: 872,018 neighbour orderings checked, 0 missing. Closure under facet crossing plus connectivity of the chamber graph in the convex cone (Corollary 7) gives completeness |
| Exact six-qubit optimum per spectrum | `py ../theory/door4_T5.py` | D4_T5.json | 1.4-1.8 s per spectrum (10 chains), after the one-time enumeration |
| A vs C optimum | probe/independent_b2/ (vstar.py) | probe/independent_b2/*.json | Agreement <= 9.6e-15 relative. n = 6: chains 0-4 only; n = 5: chains 0-9 |
| Local-minimum counts, n = 6 | as above | D4_T5.json | Chains 0-4 (A and C): 5,538, 5,319, 5,598, 4,414, 2,273. Chains 5-9 (A only): up to 6,149. C* ranges over 6.8e-4 to 3.5e-3 |
| Qubit-versus-rest split | `py ../theory/qubit_exact.py E1a`; `py ../theory/qubit_exact.py E1b` | E1a.json, E1b_bb.json (E1b_fullhull.json = full hull) | Minimum-variance perfect matching. Equals the 2x8 enumeration to 1.7e-16 (120 spectra). n = 6: 5.4-12.5 s per chain, 44-89 matchings. Implementation B (MILP branch-and-bound) agrees to 12 digits. Polynomial time is NOT claimed |
| Tableau branch-and-bound | `py ../theory/bb.py E2a`; `py ../theory/bb.py V2`; `py ../theory/bb.py RUN2 4 8 0-4 900` | E2a.json, E2_4x8*.jsonl | 4x4: equals enumeration, 20/20. 4x8: exact, 0.6-8.5 s, 5/5. B certified 4/5 with identical grids. 8x8 and 4x16 (chain 0): open, best 4.575e-4 and 4.286e-4 against 200k-start heuristics 4.942e-4 and 4.338e-4, uncertified |

## S5. Census and attainment of the single-spectrum bounds (Corollaries 6-7)

| Item | Command | Output | Result |
|---|---|---|---|
| 4x4 census, 100 spectra per ensemble | `py ../theory/census.py 100` | census.json | Local-minimum classes per spectrum: NN chains median 434.5 (181-792); GUE median 639 (301-870); Poisson median 143 (16-474) |
| Largest count found | `py ../theory/maxmin.py 4 4 400` (bipartite); qubits: inline script, logged | maxmin_3x3.json, maxmin_4x4.json, door5_qubits.json | 3x3: 18 (bound 18; the spectrum is stored). 4x4: 1114 (bound 3330). Qubits: n = 3: 2/2; n = 4: 14/14; n = 5: 233/516 |
| MI census, 20,000 spectra per shape | `py ../theory/mi_census.py` | theory/results/mi_census.json | Classes 1, 5, 14, 42, 18; 0 non-realizable minimisers, including 3x3 |
| MI census over cubicle candidates | theory/general/mi.py | mi_census_*_6.json | Every cubicle class occurs. By design this census cannot detect non-realizable minimisers |

## S6. Greedy one-qubit-at-a-time peeling versus the exact optimum

**Definitions.**
- **Model ("NN chain").** H = sum_{i=1}^{n-1} (Jx_i X_i X_{i+1} + Jy_i Y_i Y_{i+1} + Jz_i Z_i Z_{i+1}) + sum_i (hx_i X_i + hy_i Y_i + hz_i Z_i).
  The chain is open, all couplings are i.i.d. N(0,1), and chain k uses seed 1000 + k.
- **Greedy.** At each step, split one qubit off the current level list exactly: a minimum-variance perfect matching of the
  levels into pairs, with the upper member of each pair getting bit 1. Continue on the sorted pair means.
  - "Plain C1" selects the matching by C1.
  - "Ratio rule" selects by the ratio Tr(H_int^2)/Tr(H_q^2) of Cao et al.
- **Ratio.** C1(greedy placement) / C*.
- **Baseline.** C* is the exact global minimum of C1 over all unitaries, computed by coherent-order enumeration (Corollary 7).
- **N.** N = 5 chains at n = 4, and 10 chains at n = 5 and n = 6.
- **Command.** `py ../theory/door3.py 4 5`; `... 5 10`; `... 6 10`. Outputs: door3_n*.json.

| Rule | n = 4 (N = 5) | n = 5 (N = 10) | n = 6 (N = 10) |
|---|---|---|---|
| Plain C1, median ratio | 2.04 | 3.78 | 9.74 |
| Ratio rule, median (range) | not computed | 3.77 (1.31-5.68) | 5.73 (2.97-11.11) |

- The overall range of the plain-C1 ratios is 1.21-20.07.
- Implementation C reproduced the plain-C1 ratios for chains 0-4 at n = 5 and n = 6 (10 values).

## S7. Material cut from the main text, with the reason

| Item (v0.2) | Reason for removal |
|---|---|
| "Three criteria reject the lattice cut" (n = 4: 20/20; n = 6: C1, C3 10/10; C2 not run at n = 6) | Non-criticality is immediate from (Crit). The C2 verdict is tie-break dependent (13/20 under the worst tie-break at n = 4). Kept as a one-line remark |
| Sorted-eigenvalue BFGS (GOE 20, NN chains 10, free fermions 5 spectra) | It was an emulation, not the authors' code |
| State selection on the torus: D(t) = 2 R_rms t; t_half R_rms ~ 0.263 | 0.263 restates D(t) = 2 R_rms t once "half" is a fixed distance. The drift law itself is analytic |
| Approximate-locality eps-scan (n = 7) and the 73/16384 comparison | The sealed rule was not met (3/5 chains completed to eps = 0.5). The dimension-count match is post hoc |
| Realm count vs chaos (Spearman 0.24, 0.38; bar 0.8) | Negative result, unrelated to the theorems |
| Carroll-Singh recipe notes (Eq. 27 phase; set-valued criterion) | Not about the theorems |
| Fermion self-consistent counts (12/10 at (5,2); 65/168 and 144/324 sampled) | Sampled lower bounds; the Hartree-Fock analogy is not a result. Kept as a one-line remark |
| "The released sorted-fit code has interior minima exactly at our points" | Observation without proof. Listed as open in Sec. 9 |

## S8. Recorded failures and corrections (kept on purpose)

- E2a run 1: a branch-and-bound dual-bound bug, fixed before any claim.
- I3 as first written: failed 0/20 (control premise wrong). I3b failed by landscape. In I3 run 1 the optimiser left U(d).
- Door 6 at n = 6 was ill-posed. Door-6 P1 tolerance failed (2.2e-16 against 1e-20). Door-6b P2 had a denominator bug.
- "Polynomial time" for the qubit split was withdrawn.
- Proof corrections:
  - Lemma 1 complexification;
  - the sign case split in the k-partite block, replaced by the variance identity;
  - general-F necessity at alpha(l) = 0, open in v0.2 and closed in v0.3 by a convexity sign argument (Section 5);
  - Attainment Step 1, replaced by the column-difference argument;
  - the transpose-class argument.
- A conjectured uniform curvature bound for non-assignment critical points was refuted (a critical family reaches curvature
  0).
- E1b.json was overwritten by a full-hull run. The branch-and-bound timings are now in E1b_bb.json.
- A line-by-line number check of the v0.1 draft found 40 statements to fix (13 mismatches, 10 unsupported statements, 17
  overclaims); all were fixed in v0.2.
