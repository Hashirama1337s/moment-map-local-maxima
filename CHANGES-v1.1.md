# Changes in version 1.1

**In plain words.** Version 1.1 adds two short addenda. It does not change any theorem, proof or number of version 1.0.

Added:
- `ADDENDUM-C.md`: can one spectrum make every realm a local minimum at once? An exact LP answer with integer certificates:
  - no at 4x4, 2x6, 2x8, 3x4 and five qubits;
  - yes at 2x2-2x5, 3x3 and three or four qubits;
  - the 4x4 maximum lies in [1209, 2845];
  - one explicit pair of 4x4 realms never coexists.
- `ADDENDUM-D.md`: the cold limit of the least-correlated split of a thermal state (Theorem D1, Corollary D2), with an exact
  multi-precision test and a tested, unproved tie-break (Conjecture D3).
- Code:
  - `theory/d5lp.py`, `theory/d5lp_verdict.py`, `theory/d5lp_export.py`, `theory/d5lp_milp_probe.py`;
  - `theory/check_d5.py`, a stand-alone exact checker;
  - `theory/d13_cold.py`, `theory/d13_cold_verdict.py`, `theory/d13_cancellation_demo.py`.
- Data:
  - `theory/results/d5lp/`, including `certificates/` with SHA256SUMS;
  - `theory/results/d13/`: cold.json and the two stored 4x4 input sets (blind_inputs_4x4_GUE.npy, blind_inputs_4x4_ISING.npy).
- `prereg/`: the predictions and pass/fail bars of both addenda as written before the runs, byte-exact, with SHA256SUMS.

Edited, each marked "v1.1" in place:
- `PAPER.md`: the version line; the abstract's last sentence; the last bullets of Corollaries 6 and 7; Section 9; and a credit line after the
  proof of Corollary 5 ([JJR, Sec. VI E] state that every Young tableau is optimal for some spectrum; true through 2x5, false from 3x3). The edits
  point to Addendum C, which settles the sharpness question at 4x4 and five qubits.
- `README.md`: the summary, the repository table, the reproduction table, and the citation version.
- `CITATION.cff`: the version and abstract.
