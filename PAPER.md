# Local maxima of the moment-map norm on a unitary orbit: a root-block sign test, mutual information, and Klyachko's cubicles

**Moki & Julio** · October 2026 · version 1.0 · repository https://github.com/Hashirama1337s/moment-map-local-maxima

The numerical ledger is in [SUPPLEMENT.md](SUPPLEMENT.md); the code and data are in `theory/` and `probe/` (see [README.md](README.md)).

## Abstract

Let H be a Hermitian d x d matrix with simple spectrum, K a compact connected subgroup of U(d) acting on the unitary orbit
O_H by conjugation, and P the orthogonal projection onto the Hermitian part of Lie(K) plus the scalars. Minimising the
"minimal interaction" cost ||H' - P(H')||^2 over O_H is the same as maximising ||P(H')||^2, the squared norm of the moment map.
Critical points satisfy [P(H'), H'] = 0 (Kirwan, Ness). At a local maximum the K-orbit is symplectic and the stabiliser of
the point equals the stabiliser of its moment-map image (Biliotti); the one-line consequence that the point is fixed by a
maximal torus T of K is added here. We decide which T-fixed points are local maxima. A T-fixed point D is a local maximum if and only if P(D) orders the
eigenvalues of D exactly as D does, across distinct T-weight spaces. It is then a strict local maximum modulo an explicit
set of critical points (the K-orbit of D when the weights are multiplicity-free). The proof analyses the Hessian root by root. The Hessian has the general form of
Georgoulas-Robbin-Salamon's Lemma 3.5. The new ingredients are a variance identity, an exact formula at ties, and a slice
argument. The same test, with P(D) replaced by the gradient of F, holds for every C^2 K-invariant functional F(P(H')) with
positive-definite second derivative. This includes minus the sum of the marginal entropies. So the local minima of mutual
information on the unitary orbit of a full-rank state with simple spectrum are the classical states whose probabilities are
ordered like the product of their own marginals. For two parties these orders are Klyachko's cubicles (realizable standard
Young tableaux). We prove that every cubicle is the unique global-minimiser class on a nonempty open set of spectra, for both
functionals. Combined with the classical global minimum of mutual information (Jevtic-Jennings-Rudolph), this identifies
McConnell and Jennings's numerical class counts 1, 5, 14, 42, 18 (shapes 2x2, 2x3, 2x4, 2x5, 3x3) with cubicle counts. The
six non-realizable 3x3 tableaux carry exact Farkas certificates. For the number of local minima at a single spectrum, the
cubicle counts are only upper bounds: 3330 at 4x4 and 124,187 for six qubits. At 4x4 the largest number we found for one
spectrum is 1114, so sharpness there is open.

## 1. Introduction

Quantum mereology asks whether a Hamiltonian alone singles out a split of Hilbert space into subsystems [CS, CPR, St, SFD].
A widely used criterion is minimal interaction [LS, Zanardi, Cao]: choose the tensor-product structure U that minimises
C1(U) = ||H' - P(H')||^2 / ||H||^2, with H' = U^dag H U and P the projection onto local operators. Adil et al. [Adil] ask for
"some measure on the space of these realms", and Cao et al. [Cao] observe many local minima. McConnell and Jennings [MJ]
counted, numerically, the classes of eigenvalue arrangements that minimise the classical mutual information: 1, 5, 14, 42, 18
for shapes 2x2, 2x3, 2x4, 2x5, 3x3. They gave no explanation.

This paper classifies the local optima of both functionals on a unitary orbit with simple spectrum, and proves the
identification of those counts. Section 2 fixes notation. Section 3 lists what is known and whom we credit. Section 4 states
the results. Sections 5-6 contain complete proofs, and Section 7 the corollaries. Section 8 compares our Hessian with
Georgoulas-Robbin-Salamon [GRS] line by line. Section 9 lists what is not proved.

What is ours, given Section 3: the root-block sign test and its proof (Theorems 1-2, parts (b)-(d)); the second-order form
of torus-fixedness with an explicit ascent direction (Lemma 5.1); the mutual-information classification (Theorem 3); and
attainment (Theorem 4) with the Farkas certificates (Appendix A). The torus-fixedness of true local maxima of ||mu||^2 is
Biliotti's [Biliotti]. The identification of local and global optima with cubicles uses Klyachko's index set [Klyachko]; we
do not construct it.

## 2. Setting and notation

**2.1 Orbit, group, functional.**
- H is Hermitian on C^d with eigenvalues lambda_1 < ... < lambda_d (simple spectrum). O_H = {U^dag H U : U in U(d)}. The
  inner product is <X, Y> = Re Tr(X^dag Y), with norm ||.||.
- K is a compact connected subgroup of U(d), acting on O_H by x -> k x k^dag. Scalars act trivially, so we replace K by the
  compact connected group K.U(1) without changing any orbit. Then 1 lies in k_H := {iX : X in Lie(K)}, the Hermitian
  generators, and P is the orthogonal projection onto k_H. This is the "Lie(K) + scalars" projection.
- P is K-equivariant. Since 1 is in k_H, Tr P(Z) = <1, P(Z)> = Tr Z.
- F is a real function on an open K-invariant subset of k_H that contains P(O_H). It is C^2 and K-invariant. Put
  Phi(x) = F(P(x)) on O_H and Lambda(x) = grad F(P(x)), which lies in k_H. Equivariance gives
  Lambda(k x k^dag) = k Lambda(x) k^dag.
- (SC) means: D^2F(l)[Y, Y] > 0 for every nonzero traceless Y in k_H, at every l in P(O_H).
- The main case is F = ||.||^2. Then Phi(x) = ||P x||^2, Lambda = 2 P(x), (SC) holds, and
  ||H||^2 C1 = ||x - P x||^2 = ||H||^2 - Phi(x). The map U -> U^dag H U from U(d) onto O_H is continuous and open. So U is a
  local minimum of C1 if and only if U^dag H U is a local maximum of Phi.

**2.2 Torus data.**
- T is a maximal torus of K and t_H its Hermitian generators; 1 lies in t_H. C^d splits into weight spaces V_mu: each X in t_H
  acts on V_mu as the scalar mu(X). The weights are distinct linear functionals on t_H.
- R is the set of roots of Lie(K) tensor C with respect to T. For each root alpha there is a root vector E_alpha in
  Lie(K) tensor C, unique up to scale, with [X, E_alpha] = alpha(X) E_alpha for X in t_H. We take E_{-alpha} = E_alpha^dag.
  E_alpha maps V_mu into V_{mu+alpha}. Put W_alpha = ||E_alpha||^2 > 0.
- k_alpha := {y E_alpha - conj(y) E_alpha^dag : y in C} is a real 2-plane of skew-Hermitian matrices, and V_alpha := i k_alpha.
  Because K is connected, the centraliser of T in Lie(K) is Lie(T). This gives the orthogonal decomposition
  k_H = t_H (+) sum over positive roots of V_alpha.
- A point x in O_H is **T-fixed** if it commutes with t_H, i.e. preserves every V_mu. Such a D has an orthonormal eigenbasis
  {e_c} (the "cells" c) with each e_c in a weight space V_{mu(c)} and D e_c = E_c e_c. The E_c are distinct.
- l := P(D) is T-invariant, so it lies in t_H, and l e_c = l_c e_c with l_c = mu(c)(l). Likewise Lambda := grad F(l) lies in
  t_H, with Lambda_c = mu(c)(Lambda).
- A pair of cells {c, c'} is a **same-weight pair** if mu(c) = mu(c'), a **root pair** if mu(c') - mu(c) is in R, and a
  **non-root pair** otherwise.
- For a root alpha, an alpha-pair is an ordered pair p = (c -> c') with mu(c') - mu(c) = alpha. Put a_p = <e_c', E_alpha e_c>,
  w_p = |a_p|^2 and Delta_p = E_c' - E_c. Then E_alpha = sum_p a_p e_c' e_c^dag and W_alpha = sum_p w_p.
- G_T := prod_mu U(V_mu), the unitaries commuting with T, and N_D := {k g D g^dag k^dag : k in K, g in G_T}. When every weight
  space is one-dimensional ("multiplicity-free": products of unitary groups, U(n) on wedge^N or Sym^N, irreducible SU(2)),
  G_T fixes D and N_D = K.D.

**2.3 Second variation (no term dropped).** For A in u(d) put x(s) = e^{sA} x e^{-sA}. Then x' = [A, x] and
x'' = [A, [A, x]] at s = 0. Since P is linear and Lambda lies in k_H,

  Phi'(0) = <Lambda, [A, x]> = Tr([x, Lambda] A),
  Phi''(0) = D^2F(P x', P x') + DF(P x''),  with  DF(P x'') = <Lambda, [A, [A, x]]> = -<[A, Lambda], [A, x]>.      (2.1)

The last step uses Tr(Lambda [A, Y]) = -Tr([A, Lambda] Y) for skew-Hermitian A.
- **(Crit)** x is critical if and only if [Lambda(x), x] = 0.
- At a critical point, Phi''_x(A) := D^2F(P[A, x], P[A, x]) - <[A, Lambda], [A, x]> is a quadratic form on u(d). It vanishes
  on the commutant of x and does not depend on the curve. This is the Hessian.
- **Metric.** We use the Frobenius norm on u(d). At a point D with simple spectrum, T_D O_H is identified with u(d)_off, the
  skew-Hermitian matrices with zero diagonal in the basis {e_c}, through A -> [A, D].
- For F = ||.||^2: Phi''_x(A) = 2||P[A, x]||^2 - 2<[A, l], [A, x]>, with l = P(x).
- What the second summand DF(P x'') contributes: it vanishes identically on directions with [A, Lambda] = 0 (Lemma 5.1).
  At a T-fixed point it is the diagonal term -2 beta(Lambda) sum_p Delta_p |z_p|^2 of each weight-difference block
  (Proposition 5.2). It is never dropped.

## 3. Known results (credited)

- **K1. Moment map.** On O_H with its Kirillov-Kostant-Souriau form, P is the moment map of K. So minimal interaction
  maximises ||mu_K||^2 [Woodward; CDKW for the density-matrix setting].
- **K2. Criticality.** (Crit) for ||mu||^2 is the Kirwan-Ness criterion [Kirwan; Ness]. GRS Lemma 3.1 writes the gradient of
  (1/2)||mu||^2 as J L_x mu(x). Chu-Driessel [Chu-Driessel, Lemma 3.1] state the matrix form [P(H'), H'] = 0 [also Woodward,
  Lemma 7.1.1].
- **K3. Torus-fixedness (Biliotti).** Let G be a compact connected Lie group acting in a Hamiltonian fashion, and let x be a
  local maximum of ||mu||^2. Then G.x is symplectic [Biliotti, Prop. 1.2], and G_x = G_{mu(x)} [Biliotti, Prop. 2.1]. The
  centraliser of mu(x) contains a maximal torus, so x is fixed by a conjugate of T. That last step is a one-line corollary;
  Biliotti does not state it.
- **K4. Global optima.** The global minimum of C1 is diagonal in a product basis [Tegmark]; see [Soulas, fn. 11] for a
  critique of that proof. It also follows from convexity (Atiyah-Guillemin-Sternberg, via [Woodward]). The minimum of mutual
  information on a unitary orbit lies on a classical, product-diagonal state [JJR].
- **K5. Cubicles.** Orders of outer sums x_a + y_b are the realizable standard Young tableaux [Mallows-Vanderbei; Araujo;
  OEIS A211400]. They are the vertices of the sweep polytope [Castillo-Labbe] and Klyachko's cubicles [Klyachko]. For qubits
  they are the coherent Boolean term orders [Maclagan; Fine-Gill; OEIS A009997].
- **K6. Related.** The one-factor case is Brockett's "similarly ordered" fixed points [Helmke-Moore]. The local-minimum
  structure of qubit-versus-rest splits is [Cao, Thm 1]. Global-extremum bounds for Schur-convex functions of partial traces
  are in [Costa Rico-Shteyner]. The torus of equal-cost TPSs is [Tuma], with the dimension count due to [St].
- **K7. Hessian formula.** GRS Lemma 3.5 gives the covariant Hessian of (1/2)||mu||^2 on a closed Kahler manifold. See
  Section 8.

## 4. Results

**Hypotheses (H).** H has simple spectrum. K is a compact connected subgroup of U(d) acting by conjugation. P is the
orthogonal projection onto Lie(K)_H + R1. F is C^2 and K-invariant and satisfies (SC). Phi = F o P on O_H.

**Theorem 1 (F = ||.||^2).** Assume (H) with Phi(x) = ||P x||^2.
- (a) Every local maximum of Phi is K-conjugate to a T-fixed point [Biliotti; second-order form in Lemma 5.1].
- (b) A T-fixed point D is a local maximum of Phi if and only if

      (l_c - l_c')(E_c - E_c') > 0   for all cells c, c' with mu(c) != mu(c'),                     (O)

  where l = P(D). In words, l orders the eigenvalues of D exactly as D does, across distinct weight spaces.
- (c) If (O) holds, the Hessian at D is negative semidefinite. Its kernel is exactly
  T_D N_D := {[B, D] : B in Lie(K) + Lie(G_T)}. There is a neighbourhood U of D with Phi(x) <= Phi(D) on U, and equality
  exactly on N_D intersect U. In the multiplicity-free case N_D = K.D, so D is a strict local maximum modulo K.
- (d) If (O) fails, one of two things happens. Either some A has Phi''_D(A) > 0, or a non-root pair has l_c = l_c'. In the
  second case, with u = P(e_c e_c^dag - e_c' e_c'^dag) != 0, the Givens rotation by theta in span{e_c, e_c'} gives exactly

      Phi(D_theta) = Phi(D) + sin^4(theta) (E_c - E_c')^2 ||u||^2.

**Theorem 2 (strictly convex invariant F).** Assume (H). Then (a)-(c) of Theorem 1 hold with (O) replaced by

      (Lambda_c - Lambda_c')(E_c - E_c') > 0   for all cells c, c' with mu(c) != mu(c'),          (O_F)

where Lambda = grad F(P(D)). Part (d) holds with Phi(D_theta) - Phi(D) = (1/2) sin^4(theta) (E_c - E_c')^2 D^2F(u, u) + o(sin^4 theta),
which is > 0 for small theta != 0, at a non-root Lambda-tie. The proof does not assume any relation between Lambda and l.

**Theorem 3 (mutual information).** Let rho be a full-rank density matrix on C^dA (x) C^dB with simple spectrum, and let
K = U(dA) x U(dB). The local minima of I(rho') = S(rho'_A) + S(rho'_B) - S(rho') on the unitary orbit of rho are exactly the
states K-conjugate to a product-diagonal D = sum p_ab |ab><ab| with

      p_ab order-isomorphic to r_a c_b,   r_a = sum_b p_ab,  c_b = sum_a p_ab.

Each is a strict local minimum modulo K.

**Theorem 4 (attainment).** Let dA, dB >= 2, K = U(dA) x U(dB), and T a realizable standard Young tableau of shape dA x dB.
- There is a nonempty open set of simple spectra on which the global minimisers of C1 on the orbit form exactly the class of
  T. The class is the K-orbit of the placement of the sorted spectrum in T's order, together with its transpose when dA = dB.
- The same holds for I, on full-rank probability vectors.
- Consequently, the classes that occur as global minimisers, for simple (full-rank) spectra, are exactly the realizable
  tableaux, taken modulo transpose when dA = dB.

**Corollary 5 (identification).** For both C1 and I, the number of global-minimiser classes is 1, 5, 14, 42, 18 for shapes
2x2, 2x3, 2x4, 2x5, 3x3. Each class is attained on a nonempty open set of spectra.
- For mutual information these are McConnell-Jennings's numerically found counts [MJ, Sec. I]. Their notion counts the
  arrangements realising the classical minimum, up to local permutations and transpose. By [JJR] the minimum over the orbit is
  classical, so the two notions agree on simple full-rank spectra.
- At 2x3, McConnell [McConnell] had shown that all five classes occur.

## 5. Proofs of Theorems 1 and 2

**Lemma 5.1 (second-order torus-fixedness; ascent direction).** Assume (SC). Let x be critical and suppose Phi''_x(A) <= 0
for every A in u(d) with [A, Lambda] = 0, where Lambda = Lambda(x).
- Then x commutes with every Y in k_H that commutes with Lambda.
- If k in K satisfies k Lambda k^dag in t_H, then k x k^dag is T-fixed.
- Ascent direction. Let x be critical, Y in k_H with [Y, Lambda] = 0 and [Y, x] != 0, and put A* = [x, Y]. Then
  [A*, Lambda] = 0 and Phi''_x(A*) >= c_F ||A*||^4 / ||Y||^2. Here c_F > 0 is the least value of D^2F(l)[Z, Z] over unit
  traceless Z in k_H; c_F = 2 for ||.||^2.
- So for F = ||.||^2, with ||H||^2 C1 = ||H||^2 - Phi: (||H||^2 C1)''(A*) / ||A*||^2 <= -2 ||[Y, x]||^2 / ||Y||^2.

*Proof.*
- If [A, Lambda] = 0, the second-order term DF(P x'') = -<[A, Lambda], [A, x]> in (2.1) vanishes identically. So
  Phi''_x(A) = D^2F(P[A, x], P[A, x]).
- Tr [A, x] = 0, and P preserves the trace, so P[A, x] is traceless. Then (SC) and Phi''_x(A) <= 0 give P[A, x] = 0.
- Let Y be in k_H with [Y, Lambda] = 0, and put A* = [x, Y], which is skew-Hermitian. By the Jacobi identity,
  [Lambda, A*] = [[Lambda, x], Y] + [x, [Lambda, Y]] = 0, using (Crit).
- Moreover <Y, [A*, x]> = Tr([x, Y] A*) = Tr(A* A*) = -||A*||^2. Since Y is in k_H, <Y, P[A*, x]> = -||A*||^2.
- Under the hypothesis, P[A*, x] = 0, so A* = 0.
- Without the hypothesis, Cauchy-Schwarz gives ||P[A*, x]|| >= ||A*||^2 / ||Y||, and hence
  Phi''_x(A*) = D^2F(P[A*, x], P[A*, x]) >= c_F ||A*||^4 / ||Y||^2.
- Gauge. Every element of the Lie algebra k_H is K-conjugate into t_H (maximal torus theorem). Phi is K-invariant and Lambda
  is equivariant, so k x k^dag satisfies the hypotheses with Lambda in t_H. Every Y in t_H commutes with that Lambda, so
  k x k^dag commutes with t_H. QED.

For F = ||.||^2 the first conclusion at a true local maximum is Biliotti's (K3). Lemma 5.1 adds the second-order hypothesis,
general F, and the explicit direction A*. The factor 2 is proved by the Cauchy-Schwarz step; no tightness is claimed.

**Proposition 5.2 (block form at a T-fixed point).** Let D be T-fixed, and write A in u(d)_off by its entries
A_{c'c} (with A_{cc'} = -conj(A_{c'c})).
- The entries of a same-weight pair contribute 0 to Phi''_D.
- The entry z of a non-root pair (c -> c') contributes -2 (Lambda_c' - Lambda_c)(E_c' - E_c) |z|^2.
- For each positive root alpha, the alpha-pair entries z = (z_p), z_p = A_{c'c}, contribute

      B_alpha(z) = D^2F(Y_z, Y_z) - 2 alpha(Lambda) sum_p Delta_p |z_p|^2,

  where Y_z := P[A, D] lies in V_alpha and ||Y_z||^2 = 2 |sum_p conj(a_p) Delta_p z_p|^2 / W_alpha.
- Phi''_D is the sum of these contributions; there are no cross terms. For F = ||.||^2,

      B_alpha(z) = -4 Q_alpha(z),   Q_alpha(z) := alpha(l) sum_p Delta_p |z_p|^2 - |sum_p conj(a_p) Delta_p z_p|^2 / W_alpha.

*Proof.*
- Decompose A by weight difference beta: A_beta maps V_mu into V_{mu+beta}. D, l and Lambda have weight 0, so [A_beta, D]
  and [A_beta, Lambda] have weight beta.
- Components of different weight are orthogonal, and P respects the decomposition, because k_H is ad(t_H)-invariant. Its
  weight-beta part is t_H for beta = 0, V_beta (+) V_{-beta} for beta = +-alpha a root, and 0 otherwise. So both terms of
  Phi'' split over the pairs {beta, -beta}.
- For one pair p = (c -> c'): [A, D] has entries -Delta_p z_p at (c', c) and its conjugate at (c, c'), and [A, Lambda] has
  the same with Lambda_c' - Lambda_c. This gives <[A, Lambda], [A, D]> = 2 (Lambda_c' - Lambda_c) Delta_p |z_p|^2.
- Lambda lies in t_H, so Lambda_c' - Lambda_c = beta(Lambda) is the same for every pair of difference beta.
- Same weight (beta = 0): [A_0, D] is off-diagonal inside the blocks V_mu, so it is orthogonal to t_H, and P[A_0, D] = 0.
  Also [A_0, Lambda] = 0.
- Non-root beta: P kills weight beta, so only the second term survives.
- Root alpha: [A, D] = -(Z + Z^dag) with Z = sum_p Delta_p z_p e_c' e_c^dag. Its projection onto
  V_alpha = {w E_alpha + conj(w) E_alpha^dag} has w = -Tr(E_alpha^dag Z) / W_alpha = -sum_p conj(a_p) Delta_p z_p / W_alpha.
  Then ||Y_z||^2 = 2 |w|^2 W_alpha.
- For F = ||.||^2: D^2F = 2<.,.> and Lambda = 2l. QED.

**Lemma 5.3 (toral identity).** alpha(l) W_alpha = sum_p w_p Delta_p.

*Proof.* [E_alpha, E_alpha^dag] has weight 0, lies in Lie(K) tensor C and is Hermitian, so it lies in t_H, a subspace of
k_H. Since D - l is orthogonal to k_H, Tr(l [E_alpha, E_alpha^dag]) = Tr(D [E_alpha, E_alpha^dag]). The left side is
Tr([l, E_alpha] E_alpha^dag) = alpha(l) W_alpha. The right side is Tr([D, E_alpha] E_alpha^dag) = sum_p (E_c' - E_c) |a_p|^2.
QED.

**Lemma 5.4 (variance identity).** Let S = {p : a_p != 0}, and write z_p = a_p y_p for p in S. Then

  W_alpha Q_alpha(z) = (1/2) sum_{p,q in S} w_p w_q Delta_p Delta_q |y_p - y_q|^2 + alpha(l) W_alpha sum_{p not in S} Delta_p |z_p|^2.

*Proof.* By Lemma 5.3, W_alpha alpha(l) = sum_q w_q Delta_q. So the S-part of W_alpha Q_alpha is
(sum_q w_q Delta_q)(sum_p w_p Delta_p |y_p|^2) - |sum_p w_p Delta_p y_p|^2. This equals
sum_{p,q} w_p w_q Delta_p Delta_q (|y_p|^2 - Re y_p conj(y_q)), which is (1/2) sum_{p,q} w_p w_q Delta_p Delta_q |y_p - y_q|^2.
QED.

**Lemma 5.5 (sign of a root block).** Assume every Delta_p != 0, which holds under simple spectrum.
- (i) If alpha(l) Delta_p > 0 for every alpha-pair, then Q_alpha >= 0. Its kernel is {z = y a : y in C}, which is exactly the
  coordinate image of k_alpha: B = y E_alpha - conj(y) E_alpha^dag has entries z_p = y a_p.
- (ii) If alpha(l) Delta_p < 0 for some p, then Q_alpha(e_p) = alpha(l) Delta_p - w_p Delta_p^2 / W_alpha < 0.
- (iii) If alpha(l) = 0, then Q_alpha(z) < 0 for z_p := a_p Delta_p, and by Lemma 5.3 the Delta_p with p in S take both signs.

*Proof.* In (i) every term of Lemma 5.4 is >= 0. The sum vanishes iff y is constant on S and z = 0 off S. (ii) is direct. In
(iii), Q_alpha(z) = -(sum_p w_p Delta_p^2)^2 / W_alpha < 0, and sum_p w_p Delta_p = 0 with w_p > 0 on S forces both signs.
QED.

**Lemma 5.6 (kappa_alpha).** If alpha(l) != 0, then D^2F(Y, Y) = kappa_alpha ||Y||^2 for all Y in V_alpha, with
kappa_alpha = alpha(Lambda)/alpha(l). Under (SC), kappa_alpha > 0. For F = ||.||^2, kappa_alpha = 2.

*Proof.*
- For B in k_alpha, F(e^{sB} l e^{-sB}) = F(l) for all s. Differentiating twice at s = 0 gives
  D^2F([B, l], [B, l]) = <[B, Lambda], [B, l]>.
- For B = y E_alpha - conj(y) E_alpha^dag: [B, l] = -alpha(l) (y E_alpha + conj(y) E_alpha^dag) and
  [B, Lambda] = -alpha(Lambda) (y E_alpha + conj(y) E_alpha^dag).
- As y ranges over C, [B, l] ranges over all of V_alpha. Hence the identity.
- Elements of V_alpha are off-diagonal, hence traceless, so (SC) gives kappa_alpha > 0. QED.

**Lemma 5.7 (exact Givens curve).** Let D be T-fixed and {c, c'} a non-root pair. Let R_theta rotate span{e_c, e_c'} by theta
and fix the other e's, and put D_theta = R_theta D R_theta^dag and dE = E_c - E_c'. Then

  P(D_theta) = l - sin^2(theta) dE u,   u = P(e_c e_c^dag - e_c' e_c'^dag) != 0.

So Phi(D_theta) = F(l - sin^2(theta) dE u). In particular:
- for F = ||.||^2: Phi(D_theta) = Phi(D) - 2 sin^2(theta) dE (l_c - l_c') + sin^4(theta) dE^2 ||u||^2;
- for general F with Lambda_c = Lambda_c': Phi(D_theta) - Phi(D) = (1/2) sin^4(theta) dE^2 D^2F(u, u) + o(sin^4 theta).

*Proof.*
- In the plane, R_theta diag(E_c, E_c') R_theta^dag has diagonal (E_c - s^2 dE, E_c' + s^2 dE), with s = sin(theta), and
  off-diagonal entries s cos(theta) dE at the pair (c, c').
- The off-diagonal part has non-root weight, so P kills it.
- <l, u> = <l, e_c e_c^dag - e_c' e_c'^dag> = l_c - l_c', and <Lambda, u> = Lambda_c - Lambda_c'.
- u != 0 (one line): since mu(c) != mu(c'), there is X in t_H with mu(c)(X) != mu(c')(X), and
  <X, u> = <X, e_c e_c^dag - e_c' e_c'^dag> = mu(c)(X) - mu(c')(X) != 0. For product weights, c = (a, b), c' = (a', b') with
  a != a': take X = |a><a| (x) 1, which gives <X, u> = 1.
- u is traceless and lies in k_H, so D^2F(u, u) > 0 under (SC).
- The general-F line is Taylor's formula for F at l, since the increment has size O(s^2). QED.

**Necessity in Theorems 1(b) and 2.** Let D be a T-fixed local maximum, so Phi''_D <= 0. Go through the pairs.
- **Non-root pair.** Proposition 5.2 gives (Lambda_c - Lambda_c')(E_c - E_c') >= 0. If Lambda_c = Lambda_c', Lemma 5.7 gives
  Phi(D_theta) > Phi(D) for small theta != 0, a contradiction. So the inequality is strict.
- **Root alpha with alpha(l) != 0.** By Lemma 5.6, B_alpha = -2 kappa_alpha Q_alpha with kappa_alpha > 0, so Q_alpha >= 0. By
  Lemma 5.5, alpha(l) Delta_p > 0 for every alpha-pair. Then
  (Lambda_c' - Lambda_c) Delta_p = alpha(Lambda) Delta_p = kappa_alpha alpha(l) Delta_p > 0.
- **Root alpha with alpha(l) = 0 (the case left open in v0.2 for general F).** By Lemma 5.5(iii), Delta_p takes both
  signs on S.
  - If alpha(Lambda) != 0, choose p in S with alpha(Lambda) Delta_p < 0 and z = e_p. Here
    ||Y||^2 = 2 |a_p Delta_p|^2 / W_alpha > 0, although the bound D^2F(Y, Y) >= 0 does not even need Y != 0. Then
    B_alpha(e_p) = D^2F(Y, Y) - 2 alpha(Lambda) Delta_p >= -2 alpha(Lambda) Delta_p > 0.
  - If alpha(Lambda) = 0, take z_p = a_p Delta_p. Then ||Y_z||^2 = 2 (sum_p w_p Delta_p^2)^2 / W_alpha > 0, so (SC) gives
    B_alpha(z) = D^2F(Y_z, Y_z) > 0.
  - Either way Phi''_D takes a positive value. One positive second derivative along an orbit curve s -> e^{sA} D e^{-sA}
    through the critical point D excludes a local maximum, since Phi = Phi(D) + (s^2/2) Phi''_D(A) + o(s^2) along it. So
    alpha(l) != 0 at a local maximum.
  - For F = ||.||^2 the sub-branch alpha(l) = 0 with alpha(Lambda) != 0 is empty, because alpha(Lambda) = 2 alpha(l).
  - Only the sign of D^2F is used, never its value on V_alpha. The relation between Lambda and l is never used.
- **Same-weight pairs** impose nothing.

So a local maximum satisfies (O_F). For F = ||.||^2, (O_F) is (O) because Lambda = 2l. There Proposition 5.2 gives
B_alpha = -4 Q_alpha directly, and Lemma 5.5 finishes without Lemma 5.6. These arguments also prove part (d), with the
explicit directions e_p, (a_p Delta_p) and the Givens curve.

**Sufficiency and the Morse-Bott step.** Assume (O_F).

*Step 1 (the Hessian).*
- Root alpha. Every alpha-pair has alpha(Lambda) Delta_p > 0, so all Delta_p share the sign of alpha(Lambda). By Lemma 5.3,
  alpha(l) is a positive-weight mean of the Delta_p, so it is nonzero with the same sign.
- Lemma 5.6 then gives B_alpha = -2 kappa_alpha Q_alpha with kappa_alpha > 0. Lemma 5.5(i) gives B_alpha <= 0 with kernel
  the k_alpha line.
- Non-root entries contribute strictly negative terms; same-weight entries contribute 0.
- Hence Phi''_D <= 0, with kernel exactly

      K_D := (sum_alpha k_alpha) (+) {same-weight off-diagonal entries}.

- Because Lie(K) = Lie(T) (+) sum_alpha k_alpha, and Lie(T) and the diagonal commute with D, we have
  {[B, D] : B in K_D} = {[B, D] : B in Lie(K) + Lie(G_T)} = T_D N_D. This proves the kernel statement in (c).

*Step 2 (every point of N_D is critical, with the same value).*
- Take g in G_T. For X in t_H, X is scalar on each V_mu, so Tr(X g D g^dag) = Tr(X D).
- For X in V_alpha, X maps between different weight spaces while g D g^dag preserves them, so Tr(X g D g^dag) = 0 = Tr(X D).
- Hence P(g D g^dag) = l, so Phi(g D g^dag) = Phi(D) and Lambda(g D g^dag) = Lambda.
- Lambda lies in t_H, so it commutes with g D g^dag, which is therefore critical by (Crit). K-invariance extends both facts
  to all of N_D.

*Step 3 (slice and submersion).*
- Let A_S be the orthogonal complement of K_D in u(d)_off (Frobenius metric). Define
  Psi(k, g, A) = k g e^A D e^{-A} g^dag k^dag on K x G_T x A_S.
- Its differential at (1, 1, 0) is (B, C, A) -> [B + C + A, D]. Since Lie(K) + Lie(G_T) + A_S contains u(d)_off, and
  diagonal matrices commute with D, this differential is onto T_D O_H.
- So Psi is a submersion at (1, 1, 0). Every x near D can be written x = Psi(k, g, A) with g near 1 and A small.
- Put h(g, A) := Phi(g e^A D e^{-A} g^dag), so that Phi(x) = h(g, A) by K-invariance.
- By Step 2, h(g, 0) = Phi(D), and the derivative of h in A vanishes at A = 0 for every g, because g D g^dag is critical.
- The second derivative of h in A at (1, 0) is Phi''_D restricted to A_S. That restriction is negative definite: a negative
  semidefinite form is definite on any complement of its kernel.
- h is C^2 (F is C^2 and the other maps are smooth), so there are lambda > 0 and a neighbourhood of (1, 0) on which the
  second derivative of h in A is at most -lambda ||.||^2. Negative definiteness is an open condition on a continuous
  second derivative, which is why such a neighbourhood exists.
- Take A small enough that the whole segment tA, t in [0, 1], stays inside that neighbourhood, and g close enough to 1.
  Taylor's formula with integral remainder along t -> tA then gives Phi(x) = h(g, A) <= Phi(D) - (lambda/2) ||A||^2.
- Equality forces A = 0, i.e. x = k g D g^dag k^dag lies in N_D. Conversely Phi = Phi(D) on N_D.

This proves (c), and the "if" direction of (b). It also answers the higher-order question. Under (O_F) the transverse
Hessian is definite, so no higher-order term can lower Phi. A semidefinite transverse Hessian arises only at a non-root tie,
and that tie is decided exactly by Lemma 5.7.

**Part (a).** For F = ||.||^2 this is K3. Lemma 5.1 gives it under the weaker second-order hypothesis, and for every F
satisfying (SC). QED (Theorems 1 and 2).

**Proof of Theorem 3.**
- Put F(Y) = -S(Tr_B Y) - S(Tr_A Y) on {Y in k_H : Tr_B Y > 0, Tr_A Y > 0}, with K = U(dA) x U(dB). One checks
  Tr_B P(rho) = rho_A and Tr_A P(rho) = rho_B.
- A full-rank rho has full-rank marginals, so P(O_rho) lies in the domain. F is smooth and K-invariant.
- (SC). D^2(-S)(sigma)[tau, tau] > 0 for traceless tau != 0 at full-rank sigma. In sigma's eigenbasis it equals
  sum_ij |tau_ij|^2 m(sigma_i, sigma_j), where m(s, t) = (log s - log t)/(s - t) for s != t and m(s, s) = 1/s.
- If a traceless Y in k_H had Tr_B Y = 0 and Tr_A Y = 0, then <X (x) 1, Y> = Tr(X Tr_B Y) = 0, and likewise for 1 (x) X.
  So Y would be orthogonal to k_H, hence Y = 0. Therefore D^2F(Y, Y) > 0 for every nonzero traceless Y in k_H.
- The gradient is Lambda = log rho_A (x) 1 + 1 (x) log rho_B + 2 * 1, which lies in k_H.
- S(rho') is constant on the orbit, so I = -Phi - S(rho), and local minima of I are local maxima of Phi.
- The weights of U(dA) x U(dB) on product basis vectors are distinct. So T-fixed means diagonal in a product basis,
  N_D = K.D, and every pair of distinct cells is a root pair or a non-root pair.
- (O_F) reads (log r_a c_b - log r_a' c_b')(p_ab - p_a'b') > 0 for all distinct cells, i.e. p is order-isomorphic to r (x) c.
- Theorem 2 applies. Lambda is not a linear function of l. The proof of Theorem 2 uses Lambda only through K-equivariance
  (Lemma 5.6) and sign conditions, so the referee's objection to deducing Theorem 3 from the F = ||.||^2 case does not arise.
  QED.

## 6. Proof of Theorem 4 (attainment)

**Notation.**
- Cells are C = [dA] x [dB], with d = dA dB. A grid is a vector in R^C. The additive grids
  A = {(x_a + y_b)_{ab}} form a subspace of dimension dA + dB - 1 containing the constants.
- A placement is a bijection sigma : C -> [d]. For sorted lambda in R^d, (Pi_sigma lambda)_c = lambda_{sigma(c)}.
- sigma ~ tau if tau = sigma o pi with pi a row x column permutation of C, composed with the transpose when dA = dB. Placements
  in one class give states that are K-conjugate, or K-conjugate after the SWAP, which is a symmetry of both C1 and I.
- A tableau T is identified with the placement c -> T(c) + 1.

**Step 0 (reduction to placements).**
- C1: a global minimiser exists because the orbit is compact. It is a local minimum, so by Theorem 1(a) it is K-conjugate to
  a T-fixed point, i.e. a placement D_sigma(lambda). P maps diagonal matrices onto the additive diagonal ones orthogonally, so
  ||H||^2 C1(D_sigma) = g_sigma(lambda) := dist(Pi_sigma lambda, A)^2.
- I: the minimum lies on a classical product-diagonal state [JJR]. Independently, Theorem 3 with Theorem 2(a) gives the same.
  I(D_sigma(p)) = H(r^sigma) + H(c^sigma) - H(p), where H is the Shannon entropy and r^sigma, c^sigma are the row and column
  sums of Pi_sigma p.

**Step 1 (cell permutations preserving A).** *Lemma 6.1.* If a permutation pi of C satisfies {G o pi : G in A} = A, then pi is
a row x column permutation, composed with the transpose when dA = dB.

*Proof.*
- (i) Every 0-1 grid M in A is a union of rows or a union of columns (column-difference argument). For columns b, b', the
  difference M_ab - M_ab' = y_b - y_b' does not depend on a. Entries lie in {0, 1}, so either the two columns are equal, or
  one is all 1 and the other all 0.
  - If all columns are equal, M depends only on the row: a union of rows.
  - Otherwise some two columns differ, so both are constant. Every other column equals one of them or differs from it, so it
    is constant too. Then M is a union of columns.
- (ii) Since pi preserves A, so does pi^{-1}. For a cell set R write pi(R) for its image; its indicator is
  1_R o pi^{-1}, so it lies in A whenever 1_R does. Row indicators R_a and column indicators C_b lie in A, and
  |R_a intersect C_b| = 1. So each pi(R_a), pi(C_b) is a union of rows or of columns, with the same cardinality and the same
  incidence numbers.
  - Suppose pi(R_a) is a union of k >= 1 columns, so k dA = dB. Consider any pi(C_b), which has dA cells.
  - If pi(C_b) is a single column, it meets pi(R_a) in 0 or dA >= 2 cells, never in 1.
  - If pi(C_b) is a union of j rows, j dB = dA, then the intersection has j k cells. Equal to 1 forces j = k = 1 and dA = dB.
- Hence, if dA != dB, every pi(R_a) is a single row. Then every pi(C_b) is a single column: a union of rows would meet a
  row in dB >= 2 cells.
- If dA = dB and some pi(R_a) is a column, the same incidence count makes every pi(C_b) a row and every pi(R_a') a column. So
  pi composed with the transpose maps rows to rows.
- In both cases pi(a, b) = pi(R_a) intersect pi(C_b) has the stated form. QED.
- The divisible case, where dA divides dB and a union of dB/dA columns has the size of a row, is covered by the incidence
  count.

**Step 2 (witness for C1).**
- V_sigma := Pi_sigma^{-1} A has dimension dA + dB - 1, and g_sigma(lambda) = dist(lambda, V_sigma)^2 because Pi_sigma is an
  isometry.
- Pi_tau Pi_sigma^{-1} acts on grids as G -> G o (sigma^{-1} o tau). So by Lemma 6.1, V_sigma = V_tau iff sigma ~ tau. For
  sigma not ~ T, V_sigma intersect V_T is therefore a proper subspace of V_T.
- Let C_T be the set of (x, y) in R^{dA+dB} with x_a + y_b strictly ordered as T. It is open, and nonempty because T is
  realizable.
- On C_T the sorted vector of values is L(x, y) := Pi_T^{-1}(x (+) y). This is a linear map of R^{dA+dB} onto V_T. Preimages
  of proper subspaces of V_T are proper subspaces, and finitely many of them cannot cover the open set C_T.
- Choose (x, y) in C_T outside all of them and put lambda* = L(x, y). Then lambda* is simple, g_T(lambda*) = 0, and
  g_sigma(lambda*) > 0 for every sigma not ~ T.
- **Genericity is necessary.** For lambda = (0, 1, ..., 7) in shape 2x4 there are three zero-residual classes:
  [[0,1,2,3],[4,5,6,7]], [[0,1,4,5],[2,3,6,7]] and [[0,2,4,6],[1,3,5,7]].

**Step 3 (open set for C1).**
- Let delta = min over sigma not ~ T of dist(lambda*, V_sigma) > 0, and gap = min_i (lambda*_{i+1} - lambda*_i).
- Suppose ||lambda - lambda*|| < min(delta, gap)/2. Then lambda is sorted and simple, and
  g_T(lambda) <= ||lambda - lambda*||^2 < delta^2/4 < g_sigma(lambda), because distance to a subspace is 1-Lipschitz.
- By Step 0 the global minimisers are exactly the placements sigma ~ T.

**Step 4 (mutual information).**
- Take (u, v) in C_T outside the same finitely many subspaces, and put p*_ab proportional to e^{u_a + v_b}, sorted. Then
  Pi_T p* is a product distribution, so I(D_T(p*)) = 0.
- For any sigma: I(D_sigma(p*)) = 0 iff Pi_sigma p* is a product distribution, iff log Pi_sigma p* is in A, iff log p* is in
  V_sigma.
- log p* = L(u, v) - (log Z) 1, and 1 lies in every V_sigma. So log p* is in V_sigma iff L(u, v) is, which fails for every
  sigma not ~ T. Hence delta_MI := min over sigma not ~ T of I(D_sigma(p*)) > 0.
- Each p -> I(D_sigma(p)) is continuous on the simplex, and there are finitely many sigma. So some neighbourhood of p*
  consists of simple, full-rank, sorted p with I(D_T(p)) < delta_MI/2 < I(D_sigma(p)) for all sigma not ~ T.
- S(p) cancels in the difference, and the Fannes-Audenaert continuity bound for the marginal entropies gives an explicit
  positive radius. Only its existence is used here.

**Step 5 (which classes occur).**
- Every global minimiser is a local minimiser. By Theorem 1 (for C1) or Theorem 3 (for I), its cell order is the order of an
  additive grid (l, or log r_a + log c_b). After sorting rows and columns, that order is a realizable tableau.
- Conversely, Steps 2-4 realise every realizable tableau.
- Each class with a realizable order contains exactly one tableau. If a tableau T is realised by (x, y), then x and y are
  increasing. For pi = (row permutation rho) x (column permutation gamma), T o pi is realised by (x o rho, y o gamma), and it
  is a tableau only if both are increasing, i.e. pi = id.
- For dA = dB, no tableau is row-column equivalent to its transpose. Suppose T^t = T o pi. The entry 0 sits at (0, 0) in both,
  so pi fixes (0, 0) and maps row 0 to row 0. Then the first row of T^t, which is the first column of T, equals the first row
  of T as a set of entries. These two sets come from cells that share only (0, 0), and all entries are distinct, which is a
  contradiction since dA >= 2.
- Realizability is closed under transpose (swap x and y). So the classes are the realizable tableaux, halved when
  dA = dB. QED.

**Counts (proof of Corollary 5).**
- Every SYT of shape 2xn, n = 2..5, is realizable. The script farkas3x3.py checks an exact integer witness for each of the
  2 + 5 + 14 + 42 tableaux.
- Of the 42 SYT of shape 3x3, 36 have exact integer witnesses. The other 6 have exact Farkas certificates (Appendix A).
- Hence: 2x2: 2/2 = 1; 2x3: 5; 2x4: 14; 2x5: 42; 3x3: 36/2 = 18.
- The six non-realizable 3x3 tableaux form three transpose pairs. So among the 21 = 42/2 classes of 3x3 tables in [JJR],
  exactly three never occur as minimisers. That is why the count is 18.

## 7. Corollaries

**Corollary 6 (bipartite realms; upper bounds).** Let K = U(m) x U(n) and simple spectrum.
- The local minima of C1 are the K-orbits of placements whose additive fit l_ab = (row mean) + (column mean) - (grand mean)
  orders all mn cells as the spectrum does. Every pair of distinct cells is a root pair (same row or same column) or a
  non-root pair.
- For a fixed spectrum, the number of local-minimum classes (modulo the SWAP when m = n) is at most the number of realizable
  tableaux, halved when m = n: 1 (2x2), 18 (3x3), 1430 (2x8), 3330 (4x4) [OEIS A211400].
- For 2x2 the bound 1 is a theorem: there is a unique local-minimum class for every simple spectrum.
- These are upper bounds for a single spectrum. At 3x3 a spectrum with 18 local-minimum classes was found numerically. At
  4x4 the largest number found is 1114 of 3330, so sharpness is not shown.

**Corollary 7 (qubits; exact global optimum).** Let K = U(2)^n and simple spectrum on C^{2^n}.
- Root pairs are bit strings at Hamming distance 1. The fit P(D) is c + sum_j w_j s_j on s in {0,1}^n. So a local minimum of
  C1 is a placement of the sorted spectrum along a coherent Boolean term order (the order of subset sums of generic weights)
  whose own fit realises that order.
- Counted modulo local unitaries and qubit permutations, the number of local-minimum classes is at most A009997(n):
  2, 14, 516, 124,187 for n = 3..6. These are upper bounds; they were attained numerically at n = 3 and 4.
- The global minimum of C1 over ALL unitaries is attained at a local minimum. So it equals the minimum, over coherent orders
  O, of the residual of the placement of the sorted spectrum along O.
- **Completeness of the enumeration.** Coherent orders (modulo symmetry) are the chambers of the arrangement
  {w : sum_{j in S} w_j = sum_{j in S'} w_j}, S != S' subsets of [n], inside the open cone w_1 > ... > w_n > 0.
  - The chamber graph of an arrangement inside an open convex cone is connected: a generic segment between two chambers
    crosses one hyperplane at a time.
  - So any set of chambers that contains one chamber and is closed under crossing every facet interior to the cone is
    complete.
  - One of our two enumerations checks exactly this closure, in exact integer arithmetic. Both return A009997(n) for
    n <= 6 (supplement, S4).
- A six-qubit spectrum is then decomposed exactly by evaluating 124,187 placements.

**Remarks (one line each; details in the supplement).**
- (Crit) can be tested directly on a given split. For the random nearest-neighbour chains of the supplement, the lattice
  half-cut fails it, so it is not a local optimum of C1.
- Theorem 1 applies to K = U(n) on wedge^N (fermions), which is multiplicity-free. The resulting Slater-diagonal test adds a
  self-consistency condition to coherence; we report only sampled counts.
- Peeling one qubit at a time, each step solved exactly, ends far above the exact optimum. The table, with N, the ratio and
  the baseline defined, is in the supplement (S5).

## 8. Comparison with Georgoulas-Robbin-Salamon, Lemma 3.5

We read [GRS] (arXiv:1311.0410v2, Chapter 3). Their standing setting is a closed Kahler manifold X with a compact Lie group G
acting by Kahler isometries, moment map mu, and f = (1/2)|mu|^2. Lemma 3.5 opens: "The covariant Hessian of f = 1/2|mu|^2
at a point x in X" [GRS, Lemma 3.5]. In paraphrase, its two formulas are:
- (3.9): at any x, the Hessian on a tangent vector xi is |dmu(x) xi|^2 plus the Hessian of the height function
  <mu(.), zeta> with zeta = mu(x) held fixed.
- (3.10): at a critical point, on the directions J L_x eta (eta in g), the Hessian equals |L_x^* L_x eta|^2 - |[mu(x), eta]|^2.
- Lemma 3.6 (attributed there to Lijing Wang) shows that (3.10) is >= 0 at every critical point.
- GRS state that these Hessian lemmas are not used elsewhere in their paper.

| | GRS | This paper |
|---|---|---|
| Space, group | closed Kahler X, compact G | X = O_H with the KKS form (closed); G = K.U(1) |
| Function | f = (1/2)\|mu\|^2 | Phi = \|\|P x\|\|^2 = 2f; Theorem 2: any F with (SC) |
| General formula | (3.9), at any point | (2.1) at critical points. The term D^2F(Px', Px') is 2\|dmu(x) xi\|^2, and DF(P x'') is twice the height-function Hessian. Same identity for F = \|\|.\|\|^2 |
| Directions treated | J L_x g only, (3.10) | all of T_D O_H at a T-fixed D, split into same-weight, non-root and root blocks (Prop. 5.2) |
| Sign information | (3.10) >= 0 on J L_x g (Lemma 3.6) | Lemmas 5.4-5.5: a root block is semidefinite iff the root gaps Delta_p share one sign, with kernel the k_alpha line; non-root ties decided by Lemma 5.7 |
| Conclusion | none about which critical points are maxima | Theorems 1-2 (b)-(d), strictness modulo N_D |

At a T-fixed point, J L_D k_alpha is at most one real 2-plane (L_D can kill k_alpha) inside the alpha-block, which has real dimension 2 x (number of
alpha-pairs). So (3.10) and Lemma 3.6 do not decide the sign of the block. Our block formula (Proposition 5.2) is (3.9)
specialised to torus-fixed points of a coadjoint orbit and written in root coordinates. We claim no novelty for it. What we
add is the toral identity and variance identity (Lemmas 5.3-5.4), the tie formula (Lemma 5.7), the slice argument, the
general-F extension (Lemma 5.6 and the alpha(l) = 0 case), and Theorems 3-4.

## 9. Scope and open problems

- **Proved here:** Theorems 1-4, Lemmas 5.1-5.7 and 6.1, and Corollaries 5-7 as stated. Corollaries 6 and 7 are upper bounds plus the
  exact-optimum statement.
- **Not proved:**
  - Sharpness of the single-spectrum bounds at 4x4 (largest found 1114 of 3330) and for n >= 5 qubits.
  - Degenerate spectra. The proofs use E_c != E_c' (Lemmas 5.5 and 5.7); nothing is claimed when the spectrum is degenerate.
  - The analogue on general Hamiltonian K-manifolds: a local maximum iff T-fixed with mu(x) interior to its own normal cone.
  - The equivalence of our landscape with the sorted-eigenvalue landscape minimised by released code [Loizeau code] is an
    observation, not a theorem.
- **Numerics** (Hessian cross-checks, censuses, timings, branch-and-bound, greedy ratios) are not used in any proof. They are
  listed in the supplement with their reproduction commands and limitations.

## Appendix A. Farkas certificates for the six non-realizable 3x3 tableaux

Tableaux are written row by row with entries 1..9, rows separated by "/". L(a, b) = x_a + y_b, with rows and columns indexed
0..2.
- A certificate is a set of strict inequalities L(p) < L(q), each true in the tableau order (p before q), with positive
  integer multipliers. The left-hand sides and the right-hand sides of the sum then contain the same x's and y's, so the sum
  reads 0 < 0, and no additive grid has that order.
- All six certificates use three inequalities with multiplier 1, a triple cancellation.
- The last column rewrites each certificate on the 8 consecutive inequalities L(rank k) < L(rank k+1), k = 0..7, the
  constraint set of the realizability LP.
- Every certificate was verified in exact integer arithmetic (theory/farkas3x3.py; theory/results/farkas3x3.json). The same
  script verifies exact integer witnesses for the other 36 tableaux.

| Tableau | Pair certificate (each multiplier 1) | Consecutive multipliers, k = 0..7 |
|---|---|---|
| 126/347/589 | L(0,1) < L(1,0); L(2,0) < L(0,2); L(1,2) < L(2,1) | 0,1,0,0,1,0,1,0 |
| 125/367/489 | L(0,1) < L(1,0); L(2,0) < L(0,2); L(1,2) < L(2,1) | 0,1,0,1,0,0,1,0 |
| 126/357/489 | L(0,1) < L(1,0); L(2,0) < L(0,2); L(1,2) < L(2,1) | 0,1,0,1,1,0,1,0 |
| 134/258/679 | L(1,0) < L(0,1); L(0,2) < L(2,0); L(2,1) < L(1,2) | 0,1,0,1,1,0,1,0 |
| 134/268/579 | L(1,0) < L(0,1); L(0,2) < L(2,0); L(2,1) < L(1,2) | 0,1,0,1,0,0,1,0 |
| 135/248/679 | L(1,0) < L(0,1); L(0,2) < L(2,0); L(2,1) < L(1,2) | 0,1,0,0,1,0,1,0 |

Example, first row: x0+y1 < x1+y0, x2+y0 < x0+y2 and x1+y2 < x2+y1. Summing gives x0+x1+x2+y0+y1+y2 < x0+x1+x2+y0+y1+y2.
The pairs (row 1, row 6), (row 2, row 5) and (row 3, row 4) are transposes of each other.

## Appendix B. Exact integer witnesses for the realizable tableaux

Each witness is a pair of integer vectors x (rows) and y (columns) with L(a, b) = x_a + y_b strictly increasing along the
tableau's rank order. Tableaux are written as in Appendix A (entries 1..mn, rows separated by "/"). Every witness was
produced and checked in exact integer arithmetic by theory/farkas3x3.py; the values below are copied from
theory/results/farkas3x3.json without change. Together with Appendix A this covers all 42 + 2 + 5 + 14 + 42 tableaux used
in Corollary 5.

**Shape 3x3** (36 tableaux; exact_ok is true for each in the JSON).

| Tableau | x (rows) | y (columns) |
|---|---|---|
| 123/456/789 | 0, 3, 6 | 0, 1, 2 |
| 123/457/689 | 0, 5, 8 | 0, 2, 4 |
| 123/458/679 | 0, 5, 7 | 0, 1, 4 |
| 123/467/589 | 0, 5, 7 | 0, 3, 4 |
| 123/468/579 | 0, 5, 6 | 0, 2, 4 |
| 124/356/789 | 0, 3, 8 | 0, 2, 4 |
| 124/357/689 | 0, 3, 6 | 0, 2, 4 |
| 124/358/679 | 0, 5, 8 | 0, 2, 6 |
| 124/367/589 | 0, 5, 8 | 0, 4, 6 |
| 124/368/579 | 0, 5, 7 | 0, 3, 6 |
| 125/346/789 | 0, 2, 7 | 0, 1, 4 |
| 125/347/689 | 0, 3, 8 | 0, 2, 6 |
| 125/348/679 | 0, 3, 6 | 0, 1, 5 |
| 126/348/579 | 0, 3, 6 | 0, 2, 7 |
| 127/348/569 | 0, 2, 4 | 0, 1, 6 |
| 125/368/479 | 0, 4, 5 | 0, 3, 6 |
| 126/358/479 | 0, 4, 6 | 0, 3, 8 |
| 127/358/469 | 0, 3, 4 | 0, 2, 7 |
| 134/256/789 | 0, 2, 7 | 0, 3, 4 |
| 134/257/689 | 0, 3, 8 | 0, 4, 6 |
| 134/267/589 | 0, 3, 6 | 0, 4, 5 |
| 135/246/789 | 0, 1, 6 | 0, 2, 4 |
| 135/247/689 | 0, 2, 7 | 0, 3, 6 |
| 136/247/589 | 0, 1, 5 | 0, 3, 6 |
| 136/248/579 | 0, 2, 6 | 0, 3, 8 |
| 137/248/569 | 0, 1, 4 | 0, 2, 7 |
| 135/267/489 | 0, 3, 6 | 0, 5, 7 |
| 135/268/479 | 0, 4, 6 | 0, 5, 8 |
| 136/257/489 | 0, 2, 6 | 0, 5, 8 |
| 136/258/479 | 0, 2, 4 | 0, 3, 6 |
| 137/258/469 | 0, 2, 4 | 0, 3, 8 |
| 145/267/389 | 0, 2, 4 | 0, 5, 6 |
| 145/268/379 | 0, 3, 4 | 0, 5, 7 |
| 146/257/389 | 0, 1, 4 | 0, 5, 7 |
| 146/258/379 | 0, 2, 4 | 0, 5, 8 |
| 147/258/369 | 0, 1, 2 | 0, 3, 6 |

**Shape 2x2** (2 tableaux; exact_ok is true for each in the JSON).

| Tableau | x (rows) | y (columns) |
|---|---|---|
| 12/34 | 0, 2 | 0, 1 |
| 13/24 | 0, 1 | 0, 2 |

**Shape 2x3** (5 tableaux; exact_ok is true for each in the JSON).

| Tableau | x (rows) | y (columns) |
|---|---|---|
| 123/456 | 0, 3 | 0, 1, 2 |
| 124/356 | 0, 3 | 0, 2, 4 |
| 125/346 | 0, 2 | 0, 1, 4 |
| 134/256 | 0, 2 | 0, 3, 4 |
| 135/246 | 0, 1 | 0, 2, 4 |

**Shape 2x4** (14 tableaux; exact_ok is true for each in the JSON).

| Tableau | x (rows) | y (columns) |
|---|---|---|
| 1234/5678 | 0, 4 | 0, 1, 2, 3 |
| 1235/4678 | 0, 4 | 0, 2, 3, 5 |
| 1236/4578 | 0, 4 | 0, 1, 3, 6 |
| 1237/4568 | 0, 3 | 0, 1, 2, 6 |
| 1245/3678 | 0, 4 | 0, 3, 5, 6 |
| 1246/3578 | 0, 3 | 0, 2, 4, 6 |
| 1247/3568 | 0, 3 | 0, 2, 4, 8 |
| 1256/3478 | 0, 2 | 0, 1, 4, 5 |
| 1257/3468 | 0, 2 | 0, 1, 4, 7 |
| 1345/2678 | 0, 3 | 0, 4, 5, 6 |
| 1346/2578 | 0, 3 | 0, 4, 6, 8 |
| 1347/2568 | 0, 2 | 0, 3, 4, 7 |
| 1356/2478 | 0, 2 | 0, 3, 6, 7 |
| 1357/2468 | 0, 1 | 0, 2, 4, 6 |

**Shape 2x5** (42 tableaux; exact_ok is true for each in the JSON).

| Tableau | x (rows) | y (columns) |
|---|---|---|
| 12345/678910 | 0, 5 | 0, 1, 2, 3, 4 |
| 12346/578910 | 0, 5 | 0, 2, 3, 4, 6 |
| 12347/568910 | 0, 5 | 0, 1, 3, 4, 7 |
| 12348/567910 | 0, 5 | 0, 1, 2, 4, 8 |
| 12349/567810 | 0, 4 | 0, 1, 2, 3, 8 |
| 12356/478910 | 0, 5 | 0, 3, 4, 6, 7 |
| 12357/468910 | 0, 5 | 0, 2, 4, 6, 8 |
| 12358/467910 | 0, 4 | 0, 2, 3, 5, 8 |
| 12359/467810 | 0, 4 | 0, 2, 3, 5, 10 |
| 12367/458910 | 0, 5 | 0, 1, 4, 7, 8 |
| 12368/457910 | 0, 4 | 0, 1, 3, 6, 9 |
| 12369/457810 | 0, 4 | 0, 1, 3, 6, 11 |
| 12378/456910 | 0, 3 | 0, 1, 2, 6, 8 |
| 12379/456810 | 0, 3 | 0, 1, 2, 6, 10 |
| 12456/378910 | 0, 5 | 0, 4, 6, 7, 8 |
| 12457/368910 | 0, 4 | 0, 3, 5, 6, 8 |
| 12458/367910 | 0, 5 | 0, 4, 6, 8, 12 |
| 12459/367810 | 0, 4 | 0, 3, 5, 6, 11 |
| 12467/358910 | 0, 4 | 0, 2, 5, 7, 8 |
| 12468/357910 | 0, 3 | 0, 2, 4, 6, 8 |
| 12469/357810 | 0, 3 | 0, 2, 4, 6, 10 |
| 12478/356910 | 0, 3 | 0, 2, 4, 8, 10 |
| 12479/356810 | 0, 3 | 0, 2, 4, 8, 12 |
| 12567/348910 | 0, 3 | 0, 1, 5, 6, 7 |
| 12568/347910 | 0, 3 | 0, 1, 5, 7, 9 |
| 12569/347810 | 0, 2 | 0, 1, 4, 5, 8 |
| 12578/346910 | 0, 2 | 0, 1, 4, 7, 8 |
| 12579/346810 | 0, 2 | 0, 1, 4, 7, 10 |
| 13456/278910 | 0, 4 | 0, 5, 6, 7, 8 |
| 13457/268910 | 0, 4 | 0, 5, 7, 8, 10 |
| 13458/267910 | 0, 4 | 0, 5, 6, 8, 11 |
| 13459/267810 | 0, 3 | 0, 4, 5, 6, 10 |
| 13467/258910 | 0, 4 | 0, 5, 8, 10, 11 |
| 13468/257910 | 0, 3 | 0, 4, 6, 8, 10 |
| 13469/257810 | 0, 3 | 0, 4, 6, 8, 12 |
| 13478/256910 | 0, 2 | 0, 3, 4, 7, 8 |
| 13479/256810 | 0, 2 | 0, 3, 4, 7, 10 |
| 13567/248910 | 0, 3 | 0, 4, 8, 9, 10 |
| 13568/247910 | 0, 3 | 0, 4, 8, 10, 12 |
| 13569/247810 | 0, 2 | 0, 3, 6, 7, 10 |
| 13578/246910 | 0, 2 | 0, 3, 6, 9, 10 |
| 13579/246810 | 0, 1 | 0, 2, 4, 6, 8 |

## References

Labels: **OPENED**: the full text, or the cited section, was read for this project. **ABSTRACT**: abstract only.
**SNIPPET**: known only from a search snippet or a secondary source; to be opened before release.

**OPENED**
- [Adil] arXiv:2403.10895.
- [Biliotti] L. Biliotti, On the moment map on symplectic manifolds, arXiv:math/0605270. Prop. 1.2 and Prop. 2.1 re-read
  for v0.3.
- [Cao] arXiv:2606.10004.
- [CDKW] arXiv:1204.0741.
- [Castillo-Labbe] arXiv:2306.00082.
- [Chu-Driessel] Chu, Driessel, SIAM J. Numer. Anal. 27:1050 (1990).
- [Costa Rico-Shteyner] arXiv:2601.14158.
- [CPR] Cotler, Penington, Ranard, arXiv:1702.06142.
- [CS] Carroll, Singh, arXiv:2005.12938.
- [GRS] V. Georgoulas, J. W. Robbin, D. A. Salamon, The moment-weight inequality and the Hilbert-Mumford criterion,
  arXiv:1311.0410 (v2). Chapter 3, Lemmas 3.1-3.6, read for v0.3.
- [Helmke-Moore] Helmke, Moore, Optimization and Dynamical Systems.
- [JJR] Jevtic, Jennings, Rudolph, PRL 108, 110403 (arXiv:1112.3372); PRA 85, 052121.
- [Klyachko] arXiv:quant-ph/0409113.
- [LMS] arXiv:2303.02782.
- [Loizeau code] github.com/nicolasloizeau/quantum_mereology.
- [LS] arXiv:2409.01391.
- [McConnell] arXiv:1209.2674.
- [MJ] McConnell, Jennings, arXiv:1205.3517, Section I.
- [SFD] Soulas, Franzmann, Di Biagio, arXiv:2512.07468.
- [Soulas] arXiv:2506.21173.
- [St] Stoica, arXiv:2603.07674 (abstract and Sec. 5.2) and arXiv:2401.01793.
- [Tegmark] arXiv:1401.1219.
- [Tuma] Zenodo 22096373.
- [Woodward] arXiv:0912.1132.
- [Zanardi] arXiv:2212.14340.
- OEIS A211400, A009997.

- [Maclagan] D. Maclagan, Boolean term orders and the root system B_n, arXiv:math/9809134 (opened 2026-10-10, ar5iv full
  text): Definition 2.4 (coherent = induced by a weight vector) and the enumeration table, coherent counts 1, 1, 2, 14, 516,
  124187, 214580603 for n = 1..7.
- [Mallows-Vanderbei] C. Mallows and R. J. Vanderbei, Which Young tableaux can represent an outer sum?, J. Integer Seq. 18
  (2015), Article 15.9.1 (opened 2026-10-10, journal page): definition of an outer sum and its tableau; all 2 x n tableaux are
  realizable; higher shapes are not all realizable; sequence A211400. The PDF with the small-shape counts was not opened;
  counts are taken from OEIS A211400.

**ABSTRACT**
- [Araujo] arXiv:2302.09194.
- [Fine-Gill] T. Fine and J. Gill, The enumeration of comparative probability relations, Ann. Probab. 4 (1976) 667-673
  (abstract opened 2026-10-10, Project Euclid; full text paywalled): computer enumerations for n <= 6 and bounds on the
  number of relations admitting an agreeing probability measure.

**NOT OPENED (classical; statements taken from the opened secondary sources named)**
- [Kirwan] F. Kirwan, Cohomology of Quotients in Symplectic and Algebraic Geometry, Princeton 1984. The criticality statement
  used here (K2) is taken from [Woodward, Lemma 7.1.1] and [GRS, Lemma 3.1], both opened.
- [Ness] L. Ness, A stratification of the null cone via the moment map, Amer. J. Math. 106 (1984) 1281-1329. Same statement,
  same opened sources.
