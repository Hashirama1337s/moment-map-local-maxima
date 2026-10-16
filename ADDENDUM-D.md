# Addendum D. The cold limit of the least-correlated split of a thermal state

**Moki & Julio (AI-assisted)** · October 2026 · addendum to "Local maxima of the moment-map norm on a unitary orbit"
(version 1.1; DOI 10.5281/zenodo.23275890)

## In plain words

Take a quantum system in thermal equilibrium and ask which way of splitting it into two parts leaves the parts least
correlated (lowest mutual information). At high temperature the answer is the split with the least interaction energy. As
the system cools, the answer changes. Here we prove what it becomes near absolute zero. Only the lowest few energy levels
matter. Correlation dies off like exp(-X / temperature), and the winning split is the one that makes X largest. X is the
smaller of two energies: the first level that excites both parts at once, and the sum of the two single excitations. Both
are counted from the ground state. In practice: put the lowest levels along one row (the excitations of one part alone),
the next ones down one column (the other part alone), and keep the first level that excites both parts as high as
possible. A finer leading-order formula also breaks the ties left by this rule. It is not proved, but it picked the exact
winner in all 500 test cases, computed with 277-386-digit arithmetic.

## 1. Setting

H has energies E_0 < E_1 < ... < E_{d-1} on C^m (x) C^n, d = mn, m, n >= 2. The Gibbs state is rho_beta = exp(-beta H)/Z. By
[JJR] and Theorem 3 of the paper, the minimum of the mutual information I over the unitary orbit of rho_beta is attained at
classical states. Up to local unitaries and SWAP (m = n), these are diagonal states whose probability grid p_ab ∝
exp(-beta E_{T(a,b)}) is arranged by a standard Young tableau T. Rank 0 (the ground level) sits at cell (0,0), and energies
increase along rows and columns. Theorem 3 says a local minimum is ordered like the product of its marginals, so after
permuting rows and columns it is sorted along both. Write e_k = E_k - E_0 for the excitation energies and e_ab = e_{T(a,b)}.
Write I_T(beta) for the mutual information of the diagonal state with grid p_ab ∝ exp(-beta E_{T(a,b)}). Because rho_beta
is full rank with simple spectrum, Theorem 3 applies, and every global minimiser is such a state. Define

    X_T = min( e_11 , e_10 + e_01 ).

So X_T is the smaller of the first doubly excited cell and the sum of the two lowest singly excited cells.

## 2. Results

**Theorem D1 (decay rate).** For every standard Young tableau T and every beta > 0,

    I_T(beta) <= (m-1)(n-1) (1 + beta Delta_T) exp(-beta X_T),    Delta_T = max_{a,b>=1} |e_ab - e_a0 - e_0b|.

If moreover e_11 != e_10 + e_01, there is beta_T such that for beta >= beta_T

    I_T(beta) >= exp(-beta X_T) / (8 d^2).

So, under this condition, -(1/beta) ln I_T(beta) -> X_T.

**Corollary D2 (cold winner).** Assume e_11 != e_10 + e_01 for every T (true off finitely many hyperplanes of spectra).
Then there is beta_0 such that for every beta > beta_0, every global minimiser of I on the unitary orbit of rho_beta is, up
to local unitaries and SWAP, a tableau T with X_T = X* := max_T X_T. Moreover

    X* = min( e_{m+n-1} , e_1 + e_{max(m,n)} ),

attained by the "hook" placement. For n >= m: levels 0, ..., n-1 along row 0, levels n, ..., n+m-2 down column 0 below
(0,0), level n+m-1 at (1,1).

*Proof of D1, upper bound.* I(p) = min over product distributions r (x) s of D(p || r (x) s); the minimum is at the
marginals. Take q_ab = exp(-beta e_ab) (so q_00 = 1), Z = sum q_ab, and the product distribution
t_ab = q_a0 q_0b / N, N = (sum_a q_a0)(sum_b q_0b). Cells with a = 0 or b = 0 have q_ab = q_a0 q_0b. Hence

    D(p || t) = ln(N/Z) - beta sum_{a,b>=1} p_ab (e_ab - e_a0 - e_0b),   N - Z = sum_{a,b>=1} (q_a0 q_0b - q_ab).

With ln x <= x - 1 and Z >= 1:
D(p || t) <= sum_{a,b>=1} [ q_a0 q_0b - p_ab + beta p_ab |e_ab - e_a0 - e_0b| ]. In a standard Young tableau, a, b >= 1
implies e_a0 >= e_10, e_0b >= e_01 and e_ab >= e_11. So each term is at most
exp(-beta (e_10 + e_01)) + beta Delta_T exp(-beta e_11) <= (1 + beta Delta_T) exp(-beta X_T).

*Proof of D1, lower bound.* Mutual information does not increase under local maps (data processing). Coarse-grain A to the
indicator [a >= 1] and B to [b >= 1]. Then compare the joint and product laws on the single event S = {a >= 1, b >= 1}:

    I_T >= d(u || v),   u = P(S) = sum_{a,b>=1} p_ab,   v = P(a >= 1) P(b >= 1),

where d is the binary relative entropy. For fixed v, g(u) = d(u || v) has g(v) = g'(v) = 0 and g''(t) = 1/(t(1-t)) >= 1/t.
Taylor's formula with integral remainder gives d(u || v) >= (u - v)^2 / (2 max(u, v)). Since 1 <= Z <= d and the tableau is
sorted,

    exp(-beta e_11)/d <= u <= d exp(-beta e_11),   exp(-beta(e_10 + e_01))/d^2 <= v <= d^2 exp(-beta(e_10 + e_01)).

If e_11 < e_10 + e_01, then u >= 2v for large beta, so d(u || v) >= u/8 >= exp(-beta X_T)/(8d). If e_11 > e_10 + e_01, then
v >= 2u for large beta, and d(u || v) >= v/8 >= exp(-beta X_T)/(8 d^2). □

*Proof of D2.* The global minimisers are classical tableau states (Section 1), and there are finitely many tableaux. If
X_T < X*, Theorem D1 gives I_T(beta) >= exp(-beta X_T)/(8d^2) > (m-1)(n-1)(1 + beta Delta_{T*}) exp(-beta X*) >= I_{T*}(beta)
for large beta. Formula for X*: rank 1 sits at (0,1) or (1,0). Call the other of these two cells c, let r be its rank, and let s be the
rank of (1,1). If rank 1 is at (0,1), only row-0 cells precede (1,0), so r <= n. If it is at (1,0), r <= m. So
e_10 + e_01 = e_1 + e_r <= e_1 + e_{max(m,n)}. Before (1,1) only row-0 and column-0
cells can come, so s <= m + n - 1. Both bounds are reached together by the hook placement. Since e_k increases with k,
X_T = min(e_s, e_1 + e_r) <= min(e_{m+n-1}, e_1 + e_{max(m,n)}), with equality for the hook. □

**Remark.** The argmax-X set can contain many tableaux. At 4x4 it holds 234 of 12,012 classes on average for GUE spectra
and 462 for the Ising family below. D2 places the cold winner in this set; it does not pick one member.

**Conjecture D3 (leading-order tie-break; tested, not proved).** For generic spectra and large beta, the minimiser is the
argmin over tableaux of

    I_lead = sum_{a,b>=1} [ p_ab ln(p_ab / (p_a0 p_0b)) - p_ab + p_a0 p_0b ],

and I_T / I_lead -> 1. This formula is the leading term of a formal expansion of I around the ground cell. We have not bounded
the remainder uniformly over the near-ties it is meant to break.

## 3. Exact test (pre-registered)

- **Instances.** The spectra were normalised to zero mean and unit variance:
  - 200 GUE spectra at 2x4 (14 tableau classes);
  - 200 GUE spectra at 3x3 (21 classes);
  - 50 GUE spectra at 4x4 (12,012 classes);
  - 50 spectra of a 4-site mixed-field Ising chain at 4x4.
- **Exact computation.** For each instance, I was computed exactly for every class at beta = 50, 100 and 200 in
  arbitrary-precision arithmetic, with 277-386 significant digits at beta = 200. The winner is the exact argmin.
- **Controls.**
  - At beta = 1, on the first 5 instances of each set, the multi-precision I agrees with double precision to 1.5e-12
    relative.
  - Winner and runner-up at beta = 200, recomputed with 60 more digits, keep their order. Winner-runner-up gaps go down to
    10^-152 relative.
  - The 4x4 inputs match the stored instances bit for bit.
- **Pre-registration.** Predictions and bars were written and hashed before the run (`prereg/BAR-D13C.md`, sha256
  b679402d...). One amendment, also hashed before the scored run, replaced a tie threshold that was too coarse
  (`prereg/BAR-D13C-AMEND1.md`, 3ff7b1bb...). The amendment
  matters. Under the original threshold, 7, 199, 50 and 50 instances would have been excluded as ties, which would have left P1
  and P2 unscoreable at 4x4. The new rule counts a tie only when the order is not resolved at the working precision.

| set | classes | winner in argmax-X set (P1) | chance level | winner = argmin I_lead (P2) | max abs(I/I_lead - 1) (P3) |
|---|---|---|---|---|---|
| 2x4 GUE (200) | 14 | 199/200 | 0.090 | 200/200 | 3.7e-7 |
| 3x3 GUE (200) | 21 | 200/200 | 0.154 | 200/200 | 1.8e-5 |
| 4x4 GUE (50) | 12,012 | 50/50 | 0.019 | 50/50 | 4.7e-8 |
| 4x4 Ising (50) | 12,012 | 50/50 | 0.038 | 50/50 | < 1e-15 |

- **The one P1 miss** (2x4, GUE seed 126) is consistent with D2. That instance's best X is only 0.0037 above the winner's, so
  beta (X* - X_T) = 0.73 at beta = 200; the asymptotic regime starts later. I_lead still picks the exact winner there.
- **Hook formula.** The closed form for X* in D2 matches the enumerated maximum in all 500 instances.
- **Finite temperature.** beta = 200 is not "cold enough" for every spectrum. An independent re-implementation, written blind
  from the statement, found the exact beta = 200 winner outside the argmax-X set in 3 of 300 random 2x4 spectra and 3 of 100
  2x5 spectra (0 of 300 at 3x3). In those cases the winner's X was 0.003-0.015 below X*, and all six moved into the set by
  beta = 4000. The same re-implementation agreed with the test above at 3x3 and 4x4 (40 of 40 winners in the argmax-X set and
  equal to the I_lead argmin).
- **Earlier double-precision cold winners.** Computed at beta = 100, they agree with the exact beta = 100 winners in only
  79.5% / 19.5% / 2% / 0% of the four sets. The cause is cancellation, not underflow. I is a difference of entropies that agree
  beyond double precision. Example: 4x4 GUE seed 0 at beta = 100 (`theory/d13_cancellation_demo.py`). The exact I of the 12,012
  classes ranges from 2e-76 to 5e-41, while the joint entropy is 3e-24. The double-precision I is rounding noise between
  -3.7e-40 and +3.7e-40, and its argmin is unrelated to the exact one. This is why the cold end needed exact arithmetic.

## 4. Scope

- **Proved:** Theorem D1 and Corollary D2 (generic spectra, beta -> infinity).
- **Tested, not proved:** Conjecture D3.
- **No explicit beta_0.** None is given: near-ties in X (as in seed 126) push it up.
- **Hot limit.** The hot limit (minimum interaction, [Cao, App. D]) and the switches in between are not part of this addendum.

## Reproduce

From `probe/`:

- `python ../theory/d13_cold.py`: writes `theory/results/d13/cold.json`. About 5 minutes on one core.
- `python ../theory/d13_cancellation_demo.py`: the double-precision example of Section 3. Under a minute.
- `python ../theory/d13_cold_verdict.py`: prints the table.

## References

Labels as in the paper: OPENED = the cited part was read for this addendum.
- [JJR] Jevtic, Jennings, Rudolph, PRL 108, 110403 (2012), arXiv:1112.3372. OPENED: the classical global minimum (as cited
  in the paper) and Sec. VI E (the Young-tableau minimisers; no temperature or beta -> infinity analysis appears).
- [Cao] arXiv:2606.10004, Appendix D, Eq. (76): mutual information at leading order in the modular Hamiltonian is governed
  by Tr(H_int^2), the hot-limit counterpart. OPENED.
- Paper (version 1.0 of this record): Theorem 3.
