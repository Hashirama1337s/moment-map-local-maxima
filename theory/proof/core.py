"""Proof core (self-contained; independent of probe/impl_a).
f(U) = ||R||_F^2, H' = U^dag H U, L = P(H') (Frobenius projection onto X(x)1 + 1(x)Y), R = H' - L.
Curves: U -> U expm(iK), K Hermitian.  At a critical point the Hessian quadratic form is
   Q(K) = -2 ||P(i[H',K])||^2 + 2 <[L,K],[H',K]>          (derived in PROOF-GAP.md, eq. (2))
"""
import numpy as np
from scipy.linalg import expm

def ptrB(Z, m, n):  # Tr_B Z  (m x m)
    return np.einsum('ajbj->ab', Z.reshape(m, n, m, n))

def ptrA(Z, m, n):  # Tr_A Z  (n x n)
    return np.einsum('iaib->ab', Z.reshape(m, n, m, n))

def P(Z, m, n):
    d = m * n
    return (np.kron(ptrB(Z, m, n), np.eye(n)) / n + np.kron(np.eye(m), ptrA(Z, m, n)) / m
            - np.trace(Z) * np.eye(d) / d)

def comm(A, B):
    return A @ B - B @ A

def ip(A, B):  # real Frobenius inner product <A,B> = Re Tr(A^dag B)
    return float(np.real(np.vdot(A, B)))

def herm_basis(d):
    """Real-orthonormal basis of d x d Hermitian matrices (d^2 elements)."""
    out = []
    for j in range(d):
        E = np.zeros((d, d), complex); E[j, j] = 1; out.append(E)
    s = 1 / np.sqrt(2)
    for j in range(d):
        for k in range(j + 1, d):
            E = np.zeros((d, d), complex); E[j, k] = E[k, j] = s; out.append(E)
            E = np.zeros((d, d), complex); E[j, k] = 1j * s; E[k, j] = -1j * s; out.append(E)
    return np.array(out)

def cost(U, H, m, n):
    Hp = U.conj().T @ H @ U
    R = Hp - P(Hp, m, n)
    return float(np.real(np.vdot(R, R)))

def parts(U, H, m, n):
    Hp = U.conj().T @ H @ U
    Hp = (Hp + Hp.conj().T) / 2
    L = P(Hp, m, n)
    return Hp, L

def resid(Hp, L):
    """r = [L, H'];  Riemannian gradient of f is G = -2 i r (Hermitian), ||grad|| = 2||r||."""
    return comm(L, Hp)

def hess_analytic(Hp, L, m, n, basis):
    """Hessian matrix in the real basis (valid at critical points)."""
    A = np.array([P(1j * comm(Hp, K), m, n) for K in basis])
    B = np.array([comm(L, K) for K in basis])
    C = np.array([comm(Hp, K) for K in basis])
    nb = len(basis)
    Af = A.reshape(nb, -1); Bf = B.reshape(nb, -1); Cf = C.reshape(nb, -1)
    G = np.real(Af.conj() @ Af.T)
    X = np.real(Bf.conj() @ Cf.T)
    return -2 * G + X + X.T

def hess_autograd(U, H, m, n, basis):
    """Independent check: torch autograd Hessian of f(U expm(i sum x_p K_p)) at x = 0."""
    import torch
    torch.set_default_dtype(torch.float64)
    d = m * n
    Ut = torch.tensor(U, dtype=torch.complex128); Ht = torch.tensor(H, dtype=torch.complex128)
    Bt = torch.tensor(basis, dtype=torch.complex128)
    def Pt(Z):
        Z4 = Z.reshape(m, n, m, n)
        TB = torch.einsum('ajbj->ab', Z4); TA = torch.einsum('iaib->ab', Z4)
        return (torch.kron(TB, torch.eye(n, dtype=Z.dtype)) / n + torch.kron(torch.eye(m, dtype=Z.dtype), TA) / m
                - torch.trace(Z) * torch.eye(d, dtype=Z.dtype) / d)
    def f(x):
        K = torch.einsum('p,pij->ij', x.to(torch.complex128), Bt)
        V = Ut @ torch.linalg.matrix_exp(1j * K)
        Hp = V.conj().T @ Ht @ V
        R = Hp - Pt(Hp)
        return torch.real(torch.sum(R.conj() * R))
    x0 = torch.zeros(len(basis), dtype=torch.float64)
    Hs = torch.autograd.functional.hessian(f, x0)
    return Hs.numpy()

def jac_resid(Hp, L, m, n, basis):
    """J[p,q] = < K_p , i * d r (K_q) >,  d r(K) = [P(i[H',K]), H'] + [L, i[H',K]]  (exact derivative)."""
    cols = []
    for K in basis:
        dH = 1j * comm(Hp, K)
        dr = comm(P(dH, m, n), Hp) + comm(L, dH)
        cols.append(1j * dr)
    Cf = np.array(cols).reshape(len(basis), -1)
    Bf = basis.reshape(len(basis), -1)
    return np.real(Bf.conj() @ Cf.T)  # [p, q]

def coords(Z, basis):  # coefficients of Hermitian Z in basis
    return np.real(np.einsum('pij,ij->p', basis.conj(), Z))

def from_coords(x, basis):
    return np.einsum('p,pij->ij', x, basis)

def lm_critical(U, H, m, n, basis, tol=1e-13, maxit=200, mu0=1e-3, verbose=False):
    """Levenberg-Marquardt on r(U) = [L,H'] = 0 (finds critical points of any index)."""
    mu = mu0
    Hp, L = parts(U, H, m, n)
    r = coords(1j * resid(Hp, L), basis); nr = np.linalg.norm(r)
    for it in range(maxit):
        if nr < tol: break
        J = jac_resid(Hp, L, m, n, basis)
        JTJ = J.T @ J; g = J.T @ r
        while True:
            xi = -np.linalg.lstsq(JTJ + mu * np.eye(len(basis)), g, rcond=None)[0]
            Un = U @ expm(1j * from_coords(xi, basis))
            Hpn, Ln = parts(Un, H, m, n)
            rn = coords(1j * resid(Hpn, Ln), basis); nrn = np.linalg.norm(rn)
            if nrn < nr:
                U, Hp, L, r, nr = Un, Hpn, Ln, rn, nrn
                mu = max(mu / 5, 1e-12 * max(1.0, float(np.trace(JTJ)) / len(basis))); break
            mu *= 4
            if mu > 1e12: return U, nr, it
        if verbose: print(it, nr, mu)
    return U, nr, it

def descend(U, H, m, n, iters=4000, tol=1e-10):
    """Riemannian gradient descent with Armijo backtracking along U expm(-i t G)."""
    t = 0.1
    f0 = cost(U, H, m, n)
    for it in range(iters):
        Hp, L = parts(U, H, m, n)
        G = -2j * comm(L, Hp); G = (G + G.conj().T) / 2
        g2 = ip(G, G)
        if np.sqrt(g2) < tol: break
        while True:
            Un = U @ expm(-1j * t * G)
            fn = cost(Un, H, m, n)
            if fn <= f0 - 0.3 * t * g2: break
            t *= 0.5
            if t < 1e-14: return U, f0, np.sqrt(g2)
        U, f0 = Un, fn; t = min(t * 2, 10.0)
    Hp, L = parts(U, H, m, n)
    return U, f0, 2 * np.linalg.norm(comm(L, Hp))

# ----------------- classification helpers -----------------
def is_assignment(Hp, m, n, tol=1e-7):
    """Gauge-invariant test (non-degenerate H'): eigenbasis of H' is a tensor-product basis.
    Returns (bool, max Schmidt s2, max non-0/1 overlap among A- and B-factors)."""
    w, V = np.linalg.eigh(Hp)
    xs, ys, s2max = [], [], 0.0
    for i in range(m * n):
        M = V[:, i].reshape(m, n)
        Uu, s, Vh = np.linalg.svd(M)
        s2max = max(s2max, s[1] if len(s) > 1 else 0.0)
        xs.append(Uu[:, 0]); ys.append(Vh[0].conj())
    def worst(vs):
        G = np.abs(np.array(vs).conj() @ np.array(vs).T) ** 2
        return float(np.max(np.minimum(G, 1 - G)))
    ov = max(worst(xs), worst(ys))
    return (s2max < tol and ov < tol), float(s2max), ov

def local_gaps(Hp, m, n):
    a = np.linalg.eigvalsh(ptrB(Hp, m, n)); b = np.linalg.eigvalsh(ptrA(Hp, m, n))
    ga = np.min(np.diff(a)) if m > 1 else np.inf
    gb = np.min(np.diff(b)) if n > 1 else np.inf
    return float(ga), float(gb)

def gauge_to_product(Hp, m, n):
    """Local unitary V(x)W diagonalising Tr_B H', Tr_A H'; returns rotated H'."""
    _, V = np.linalg.eigh(ptrB(Hp, m, n)); _, W = np.linalg.eigh(ptrA(Hp, m, n))
    T = np.kron(V, W)
    return T.conj().T @ Hp @ T

def commutant_form(Hp, L, m, n, tol_rel=1e-8):
    """N1 form Q(K) = -2||P(i[H',K])||^2 restricted to {K : [L,K]=0}. Returns its min eigenvalue
    and the minimiser K (Hermitian).  Works in the eigenbasis of L (exact block structure)."""
    d = m * n
    w, V = np.linalg.eigh(L)
    scale = max(1.0, np.max(np.abs(w)))
    # group eigenvalues of L into clusters
    groups, cur = [], [0]
    for i in range(1, d):
        if abs(w[i] - w[cur[-1]]) < tol_rel * scale: cur.append(i)
        else: groups.append(cur); cur = [i]
    groups.append(cur)
    basis = []
    s = 1 / np.sqrt(2)
    for g in groups:
        for j in g:
            E = np.zeros((d, d), complex); E[j, j] = 1; basis.append(E)
        for x in range(len(g)):
            for y in range(x + 1, len(g)):
                j, k = g[x], g[y]
                E = np.zeros((d, d), complex); E[j, k] = E[k, j] = s; basis.append(E)
                E = np.zeros((d, d), complex); E[j, k] = 1j * s; E[k, j] = -1j * s; basis.append(E)
    basis = np.array([V @ E @ V.conj().T for E in basis])
    A = np.array([P(1j * comm(Hp, K), m, n) for K in basis]).reshape(len(basis), -1)
    Q = -2 * np.real(A.conj() @ A.T)
    ev, evec = np.linalg.eigh(Q)
    Kmin = np.einsum('p,pij->ij', evec[:, 0], basis)
    return float(ev[0]), Kmin, [len(g) for g in groups]
