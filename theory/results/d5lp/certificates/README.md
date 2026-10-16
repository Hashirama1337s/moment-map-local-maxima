# Certificates for Addendum C: file format and conventions

**In plain words.** Each file is a proof object that `theory/check_d5.py` verifies with whole numbers alone.

- Spectra are lists of exact rationals ("p/q" strings), sorted ascending: lam_0 < ... < lam_{d-1}.
- A tableau is a rank grid (0-based): tableau[a][b] = rank of the level placed in cell (a, b).
- An order_rank (qubits) is a list over cells c = 0..2^n - 1 (bitmask), giving the rank of each cell.
- **Row convention (important for the integer identity).** A certificate entry {kind, k, multiplier} stands for one row:
  - kind "gap": lam_{k+1} - lam_k;
  - kind "realm" (bipartite): d * (l_{c_{k+1}} - l_{c_k}), d = mn, l = row mean + column mean - grand mean of lam[tableau],
    c_k = cell of rank k;
  - kind "realm" (qubits): 2^n * (l_{c_{k+1}} - l_{c_k}), l = least-squares projection of lam[order] onto span{1, Z_1, ..., Z_n}.
  - The scaled rows have integer coefficients. The certificate claims that sum(multiplier * row) is identically zero as a
    linear form in lam. Read without the scale factors, the identity does not hold.
- outer_sum_witness {x, y}: integers with x_a + y_b strictly increasing along the tableau's rank order (realizability).
- weight_witness {w}: integers 0 < w_1 < ... < w_n whose subset sums are strictly increasing along the order (coherence).
- upper_cores.json: 485 sets of realms, pairwise disjoint as classes (tableau modulo transpose), each with its own
  certificate that uses only its own realms.
- SHA256SUMS: hashes of the JSON files.
