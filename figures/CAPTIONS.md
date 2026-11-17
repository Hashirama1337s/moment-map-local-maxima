# Captions - moment-map-local-maxima

Run: `python figures/make_figures.py .` from the repository root (Python 3.11, matplotlib; standard library otherwise;
a few seconds). The script re-checks in exact integer arithmetic every witness (row and column numbers) and every
certificate it draws, checks that `farkas3x3.json` lists each of the 42 grids exactly once (by enumerating them itself),
and checks 36 / 2 = 18 against `mi_census.json`. It stops if any check fails.

## 1. `figures/grids_3x3.svg` (hero)

**Point.** The exact test sorts every possible seating of a 3 x 3 split: 36 grids can be stable, 6 never can, and mirror
pairs turn the 36 into the 18 that McConnell and Jennings had counted by computer.

**Caption.** All 42 ways to rank the energies of a 9-level system in a 3 x 3 grid (1 = lowest; every row and column
increases). By the paper's test, a ranking can be a stable split only if some "row number + column number" puts the
cells in the same order. That works for 36 grids (blue) and fails for 6 (orange), each failure proved by a short
certificate. A grid and its mirror image describe the same split, so the 36 give 18 kinds of stable split: the count
McConnell and Jennings had found numerically. This holds for spectra with no two energies equal.

**Alt text.** Forty-two small 3 x 3 number grids: 36 outlined in blue under "36 possible", 6 outlined in orange under
"6 impossible", with a note that mirror pairs give 18 kinds of stable split.

**Sources.**
| item | file | field |
|---|---|---|
| the 36 grids and their witnesses | `theory/results/farkas3x3.json` | `3x3.realizable[*].T`, `.x`, `.y` |
| the 6 grids and their certificates | same | `3x3.nonrealizable[*].T`, `.pair_certificate` |
| 42 / 36 / 6 / 18 | same | `n_syt`, `n_realizable`, `n_nonrealizable`, `n_transpose_classes_realizable` |
| 18 found numerically | `theory/results/mi_census.json` | `3x3.classes` = 18 |
| "simple spectrum" scope | `README.md` "In plain words" | |

## 2. `figures/worked_example.svg`

**Point.** The test is checkable by hand: a grid passes when row and column numbers reproduce its order, and fails when
three of its own demands add up to "something < itself".

**Caption.** The README's worked example. Left: the grid 123 / 456 / 789 passes. With row numbers x = 0, 3, 6 and column
numbers y = 0, 1, 2, the sums x + y (small grey numbers) rise in exactly the order of the ranks. Right: the grid
126 / 347 / 589 fails. Each arrow joins a lower rank to a higher one, so "row + column" must be smaller where the arrow
starts; adding the three demands gives the same total on both sides, "something < itself", so no row and column
numbers exist. This kind of proof is a Farkas certificate.

**Alt text.** Two 3 x 3 grids. The blue one, ranks 1 to 9 in reading order, has x = 0, 3, 6 beside its rows and y = 0,
1, 2 above its columns, with each cell's sum. The orange one, 126 / 347 / 589, has three green arrows: 2 to 3, 5 to 6,
and 7 to 8.

**Sources.**
| item | file | field |
|---|---|---|
| witness x = (0, 3, 6), y = (0, 1, 2) | `theory/results/farkas3x3.json` | `3x3.realizable[]` with `T` = [[0,1,2],[3,4,5],[6,7,8]] |
| three pairs (cells (0,1)->(1,0), (2,0)->(0,2), (1,2)->(2,1)), weights 1 | same | `3x3.nonrealizable[]` with `T_1based` = "126/347/589", `pair_certificate` |
| the three inequalities in words | `README.md` "A worked example" | |

## 3. `figures/bracket_4x4.svg`

**Point.** For a 16-level system split 4 x 4, no list of energies makes all 3,330 kinds of stable split stable at once;
the true maximum is proved to lie between 1,209 and 2,845, drawn as an interval.

**Caption.** Split a 16-level system into two 4-level parts. There are 3,330 kinds of stable split, but no single list
of 16 energies (no two equal) makes all of them stable at once. The largest number one list can achieve lies in the
shaded interval: at least 1,209, because an explicit list of energies reaches 1,209 (checked exactly), and at most
2,845, because 485 separate groups of splits each contain one that must fail (3,330 - 485). The exact maximum is open.

**Alt text.** A number line from 0 to 3,500 with a shaded interval from 1,209 to 2,845 labelled "true maximum lies
here" and a separate mark at 3,330 for all kinds of stable split.

**Sources.**
| item | file | field |
|---|---|---|
| 1,209 | `theory/results/d5lp/certificates/lower_1209.json` | `claimed_realms` (checked exactly by `theory/check_d5.py`) |
| 2,845, 485, 3,330 | `theory/results/d5lp/certificates/upper_cores.json` | `upper_bound`, `n_cores` (= length of `cores`), `total_realms` |
| "no single list makes all stable" | `ADDENDUM-C.md` Proposition C1 (4x4: no); `theory/results/d5lp/joint.json` `4x4.verdict` = INFEASIBLE | |
| "exact maximum is open" | `ADDENDUM-C.md` Section 4 | |
