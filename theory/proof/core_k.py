"""k-partite proof core.  H on C^{d1} (x) ... (x) C^{dk}; P1 = Frobenius projection onto 1-local operators
span{1 (x) .. X_j .. (x) 1};  L = P1(H'),  f = ||H' - L||^2,  curves U expm(iK), K Hermitian.
  P1(Z) = sum_j emb_j(Tr_{not j} Z) * d_j/d  -  (k-1) Tr(Z)/d * 1.
At a critical point the Hessian quadratic form is Q(K) = -2||P1(i[H',K])||^2 + 2<[L,K],[H',K]>.
"""
import numpy as np
from itertools import product
from scipy.linalg import expm
from core import comm, ip, herm_basis, coords, from_coords

LET = 'abcdefghijklmnopqrstuvwxyz'

def ptr_keep(Z, dims, j):
    """Tr over all factors except j  ->  d_j x d_j."""
    k = len(dims)
    row = list(LET[:k]); col = list(LET[k:2 * k])
    for i in range(k):
        if i != j: col[i] = row[i]
    return np.einsum(''.join(row) + ''.join(col) + '->' + row[j] + col[j], Z.reshape(tuple(dims) * 2))

def emb(X, dims, j):
    out = np.array([[1.0 + 0j]])
    for i, di in enumerate(dims):
        out = np.kron(out, X if i == j else np.eye(di))
    return out

def P1(Z, dims):
    d = int(np.prod(dims)); k = len(dims)
    out = -(k - 1) * np.trace(Z) / d * np.eye(d, dtype=complex)
    for j, dj in enumerate(dims):
        out = out + emb(ptr_keep(Z, dims, j), dims, j) * dj / d
    return out

def cost(U, H, dims):
    Hp = U.conj().T @ H @ U; R = Hp - P1(Hp, dims)
    return float(np.real(np.vdot(R, R)))

def parts(U, H, dims):
    Hp = U.conj().T @ H @ U; Hp = (Hp + Hp.conj().T) / 2
    return Hp, P1(Hp, dims)

def hess_analytic(Hp, L, dims, basis):
    nb = len(basis)
    A = np.array([P1(1j * comm(Hp, K), dims) for K in basis]).reshape(nb, -1)
    B = np.array([comm(L, K) for K in basis]).reshape(nb, -1)
    C = np.array([comm(Hp, K) for K in basis]).reshape(nb, -1)
    X = np.real(B.conj() @ C.T)
    return -2 * np.real(A.conj() @ A.T) + X + X.T

def hess_autograd(U, H, dims, basis):
    import torch
    d = int(np.prod(dims)); k = len(dims)
    Ut = torch.tensor(U, dtype=torch.complex128); Ht = torch.tensor(H, dtype=torch.complex128)
    Bt = torch.tensor(np.array(basis), dtype=torch.complex128)
    embs = []  # projectors via explicit partial-trace einsums
    def ptk(Z, j):
        row = list(LET[:k]); col = list(LET[k:2 * k])
        for i in range(k):
            if i != j: col[i] = row[i]
        return torch.einsum(''.join(row) + ''.join(col) + '->' + row[j] + col[j], Z.reshape(tuple(dims) * 2))
    def embt(X, j):
        out = torch.ones((1, 1), dtype=torch.complex128)
        for i, di in enumerate(dims):
            out = torch.kron(out, X if i == j else torch.eye(di, dtype=torch.complex128))
        return out
    def P1t(Z):
        out = -(k - 1) * torch.trace(Z) / d * torch.eye(d, dtype=torch.complex128)
        for j, dj in enumerate(dims): out = out + embt(ptk(Z, j), j) * dj / d
        return out
    def f(x):
        K = torch.einsum('p,pij->ij', x.to(torch.complex128), Bt)
        V = Ut @ torch.linalg.matrix_exp(1j * K); Hp = V.conj().T @ Ht @ V; R = Hp - P1t(Hp)
        return torch.real(torch.sum(R.conj() * R))
    return torch.autograd.functional.hessian(f, torch.zeros(len(basis), dtype=torch.float64)).numpy()

def jac_resid(Hp, L, dims, basis):
    cols = []
    for K in basis:
        dH = 1j * comm(Hp, K); cols.append(1j * (comm(P1(dH, dims), Hp) + comm(L, dH)))
    Cf = np.array(cols).reshape(len(basis), -1); Bf = np.array(basis).reshape(len(basis), -1)
    return np.real(Bf.conj() @ Cf.T)

def lm_critical(U, H, dims, basis, tol=1e-14, maxit=200):
    mu = 1e-3; Hp, L = parts(U, H, dims)
    r = coords(1j * comm(L, Hp), basis); nr = np.linalg.norm(r)
    for it in range(maxit):
        if nr < tol: break
        J = jac_resid(Hp, L, dims, basis); JTJ = J.T @ J; g = J.T @ r
        while True:
            xi = -np.linalg.lstsq(JTJ + mu * np.eye(len(basis)), g, rcond=None)[0]
            Un = U @ expm(1j * from_coords(xi, basis)); Hpn, Ln = parts(Un, H, dims)
            rn = coords(1j * comm(Ln, Hpn), basis); nrn = np.linalg.norm(rn)
            if nrn < nr:
                U, Hp, L, r, nr = Un, Hpn, Ln, rn, nrn
                mu = max(mu / 5, 1e-12 * max(1.0, float(np.trace(JTJ)) / len(basis))); break
            mu *= 4
            if mu > 1e12: return U, nr
    return U, nr

def descend(U, H, dims, iters=3000, tol=1e-6):
    t = 0.1; f0 = cost(U, H, dims)
    for it in range(iters):
        Hp, L = parts(U, H, dims); G = -2j * comm(L, Hp); G = (G + G.conj().T) / 2; g2 = ip(G, G)
        if np.sqrt(g2) < tol: break
        while True:
            Un = U @ expm(-1j * t * G); fn = cost(Un, H, dims)
            if fn <= f0 - 0.3 * t * g2: break
            t *= 0.5
            if t < 1e-14: return U
        U, f0 = Un, fn; t = min(t * 2, 10.0)
    return U

def is_assignment(Hp, dims, tol=1e-7):
    """Eigenbasis of (simple) H' is a k-fold tensor-product basis: every eigenvector fully product (all one-site reduced
    states pure) and, per site, the factor vectors pairwise parallel or orthogonal."""
    w, V = np.linalg.eigh(Hp); d = len(w); k = len(dims)
    impure, facs = 0.0, [[] for _ in range(k)]
    for i in range(d):
        rho = np.outer(V[:, i], V[:, i].conj())
        for j in range(k):
            ev, evec = np.linalg.eigh(ptr_keep(rho, dims, j))
            impure = max(impure, 1 - ev[-1]); facs[j].append(evec[:, -1])
    ov = 0.0
    for j in range(k):
        G = np.abs(np.array(facs[j]).conj() @ np.array(facs[j]).T) ** 2
        ov = max(ov, float(np.max(np.minimum(G, 1 - G))))
    return bool(impure < tol and ov < tol), float(impure), ov

def local_gaps(Hp, dims):
    return [float(np.min(np.diff(np.linalg.eigvalsh(ptr_keep(Hp, dims, j))))) for j in range(len(dims))]

def gauge(Hp, dims):
    T = np.array([[1.0 + 0j]])
    for j in range(len(dims)):
        _, V = np.linalg.eigh(ptr_keep(Hp, dims, j)); T = np.kron(T, V)
    return T

def commutant_form(Hp, L, dims, tol_rel=1e-8):
    d = Hp.shape[0]; w, V = np.linalg.eigh(L); scale = max(1.0, np.max(np.abs(w)))
    groups, cur = [], [0]
    for i in range(1, d):
        if abs(w[i] - w[cur[-1]]) < tol_rel * scale: cur.append(i)
        else: groups.append(cur); cur = [i]
    groups.append(cur); basis = []; s = 1 / np.sqrt(2)
    for g in groups:
        for j in g:
            E = np.zeros((d, d), complex); E[j, j] = 1; basis.append(E)
        for x in range(len(g)):
            for y in range(x + 1, len(g)):
                a, b = g[x], g[y]
                E = np.zeros((d, d), complex); E[a, b] = E[b, a] = s; basis.append(E)
                E = np.zeros((d, d), complex); E[a, b] = 1j * s; E[b, a] = -1j * s; basis.append(E)
    basis = np.array([V @ E @ V.conj().T for E in basis])
    A = np.array([P1(1j * comm(Hp, K), dims) for K in basis]).reshape(len(basis), -1)
    ev = np.linalg.eigvalsh(-2 * np.real(A.conj() @ A.T))
    return float(ev[0]), [len(g) for g in groups]

def d2_reassign(Hp, dims, tol_rel=1e-8):
    """D2-k: gauge L diagonal, re-place each L-level block's eigenvalues on the block's product states.
    Returns (||H'-L||^2, ||D-L||^2, f(D) = ||D - P1 D||^2, max off-block leakage of H' in the gauge basis)."""
    T = gauge(Hp, dims); H2 = T.conj().T @ Hp @ T; H2 = (H2 + H2.conj().T) / 2; L2 = P1(H2, dims)
    Lv = np.real(np.diag(L2)); d = len(Lv); scale = max(1.0, np.max(np.abs(Lv)))
    Dv = np.zeros(d); used = np.zeros(d, bool); leak = 0.0
    for i in range(d):
        if used[i]: continue
        S = [j for j in range(d) if abs(Lv[j] - Lv[i]) < tol_rel * scale]; used[S] = True
        Dv[S] = np.linalg.eigvalsh(H2[np.ix_(S, S)])
        notS = [j for j in range(d) if j not in S]
        if notS: leak = max(leak, float(np.max(np.abs(H2[np.ix_(S, notS)]))))
    D = np.diag(Dv).astype(complex)
    a = float(np.real(np.vdot(H2 - L2, H2 - L2))); b = float(np.real(np.vdot(D - L2, D - L2)))
    R = D - P1(D, dims)
    return a, b, float(np.real(np.vdot(R, R))), leak, float(np.linalg.norm(L2 - np.diag(np.diag(L2))))
