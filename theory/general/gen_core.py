"""General-K theorem core.  Self-contained.
K = compact connected subgroup of U(d), given by Hermitian generators spanning k_H (iX in Lie K) and a list of
commuting Hermitian generators spanning t_H (maximal torus).  P = Frobenius (Re Tr) projection onto span(k_H, 1).
f(U) = ||H' - P(H')||^2 = ||H||^2 - ||P(H')||^2,  H' = U^dag H U,  curves U expm(isK), K Hermitian.
At a critical point ([L,H'] = 0, L = P(H')):  Q(K) = -2||P(i[H',K])||^2 + 2<[L,K],[H',K]>.
All group constructions return generators in a basis where the standard basis is a T-weight basis.
"""
import numpy as np, itertools, math
from scipy.linalg import expm

# ----------------------------------------------------------------- basics
def comm(A, B): return A @ B - B @ A
def ip(A, B): return float(np.real(np.vdot(A, B)))

def herm_basis(d):
    out = []
    for j in range(d):
        E = np.zeros((d, d), complex); E[j, j] = 1; out.append(E)
    s = 1 / np.sqrt(2)
    for j in range(d):
        for k in range(j + 1, d):
            E = np.zeros((d, d), complex); E[j, k] = E[k, j] = s; out.append(E)
            E = np.zeros((d, d), complex); E[j, k] = 1j * s; E[k, j] = -1j * s; out.append(E)
    return np.array(out)

def coords(Z, basis): return np.array([ip(K, Z) for K in basis])
def from_coords(x, basis): return np.einsum('p,pij->ij', x, basis)

def real_span_onb(mats, tol=1e-10):
    """Real-orthonormal (Re Tr) basis of the real span of the given Hermitian matrices."""
    mats = np.array(mats); d = mats.shape[1]
    V = mats.reshape(len(mats), -1); R = np.concatenate([V.real, V.imag], axis=1)
    _, s, Vt = np.linalg.svd(R, full_matrices=False)
    r = int(np.sum(s > tol * s[0])); B = Vt[:r]
    return (B[:, :d * d] + 1j * B[:, d * d:]).reshape(r, d, d)

# ----------------------------------------------------------------- groups
class Group:
    def __init__(self, name, gens, tgens, W, note=''):
        gens = [np.asarray(g, complex) for g in gens]; tgens = [np.asarray(g, complex) for g in tgens]
        d = gens[0].shape[0]
        # rotate to a weight basis if t is not diagonal
        if max(np.abs(t - np.diag(np.diag(t))).max() for t in tgens) > 1e-12:
            x = np.random.default_rng(12345).normal(size=len(tgens))
            _, V = np.linalg.eigh(sum(c * t for c, t in zip(x, tgens)))
            gens = [V.conj().T @ g @ V for g in gens]; tgens = [V.conj().T @ g @ V for g in tgens]
            # inside degenerate weight spaces V is arbitrary; that is fine
        for t in tgens: assert np.abs(t - np.diag(np.diag(t))).max() < 1e-9, 'torus not simultaneously diagonal'
        self.name, self.d, self.W, self.note = name, d, W, note
        self.gens = gens; self.tgens = [np.diag(np.real(np.diag(t))).astype(complex) for t in tgens]
        self.B = real_span_onb(gens + [np.eye(d)])          # ONB of span(k_H, 1)
        self.Bk = real_span_onb(gens)                       # ONB of k_H
        T = np.array([np.real(np.diag(t)) for t in self.tgens] + [np.ones(d)]).T
        Qt, s, _ = np.linalg.svd(T, full_matrices=False); Qt = Qt[:, s > 1e-10 * s[0]]
        self.Qt = Qt                                        # ONB (columns) of diag part t + R1 in R^d
        self.Pdiag = Qt @ Qt.T                              # P restricted to diagonal matrices (weight basis)
        self.rank_t1 = Qt.shape[1]
        # weight labels / weight spaces
        wl = np.round(Qt, 9); keys = {}; lab = np.zeros(d, int)
        for c in range(d):
            k = tuple(wl[c]); lab[c] = keys.setdefault(k, len(keys))
        self.label = lab; self.mult = np.bincount(lab)
        self.multfree = bool(np.all(self.mult == 1))
        # sanity: span(k,1) must contain t+R1
        for j in range(Qt.shape[1]):
            D = np.diag(Qt[:, j]).astype(complex); assert np.linalg.norm(self.P(D) - D) < 1e-9
        # dimensions: k_H, t_H (rank), root directions
        self.dimk = len(self.Bk)
        Td = real_span_onb([np.diag(np.diag(b)) for b in self.Bk]) if True else None
        self.rank = len(real_span_onb(self.tgens))
        self.nroot = self.dimk - self.rank
        self.roots = self._roots()

    def P(self, Z):
        c = np.real(np.einsum('mij,ij->m', self.B.conj(), Z))
        return np.einsum('m,mij->ij', c, self.B)

    def _roots(self):
        """Root vectors as matrices: for each nonzero weight difference alpha, the alpha-component of k_C."""
        d = self.d; wl = self.Qt; out = []
        diffs = {}
        for c in range(d):
            for cp in range(d):
                if self.label[c] == self.label[cp]: continue
                a = tuple(np.round(wl[cp] - wl[c], 8)); diffs.setdefault(a, []).append((cp, c))   # |cp><c| shifts c -> cp
        xi0 = np.random.default_rng(777).normal(size=wl.shape[1])
        for a, pairs in diffs.items():
            if np.dot(a, xi0) <= 0: continue                     # positive half only
            mask = np.zeros((d, d), bool)
            for (cp, c) in pairs: mask[cp, c] = True
            comps = [np.where(mask, g, 0) for g in self.gens]
            V = np.array([x.reshape(-1) for x in comps]); s = np.linalg.svd(V, compute_uv=False)
            r = int(np.sum(s > 1e-9 * max(1, s[0]))) if s[0] > 1e-12 else 0
            if r == 0: continue
            assert r == 1, f'root space of dim {r}'
            Ea = comps[int(np.argmax([np.linalg.norm(x) for x in comps]))]
            out.append(dict(alpha=np.array(a), E=Ea / np.linalg.norm(Ea), pairs=pairs))
        assert 2 * len(out) == self.nroot, (2 * len(out), self.nroot)
        return out

def emb(X, dims, j):
    out = np.array([[1.0 + 0j]])
    for i, di in enumerate(dims): out = np.kron(out, X if i == j else np.eye(di))
    return out

def g_product(dims):
    gens = [emb(b, dims, j) for j, dj in enumerate(dims) for b in herm_basis(dj)]
    tgens = [emb(np.diag(np.eye(dj)[a]).astype(complex), dims, j) for j, dj in enumerate(dims) for a in range(dj)]
    return Group('U(' + ')xU('.join(map(str, dims)) + ')', gens, tgens, int(np.prod([math.factorial(x) for x in dims])))

def _jw(n):
    a = np.array([[0, 1], [0, 0]], complex); Z = np.diag([1, -1]).astype(complex); I = np.eye(2)
    ops = []
    for i in range(n):
        o = np.array([[1.0 + 0j]])
        for k in range(n): o = np.kron(o, Z if k < i else (a if k == i else I))
        ops.append(o)
    return ops

def g_fermions(n, N):
    c = _jw(n); idx = [s for s in range(2 ** n) if bin(s).count('1') == N]
    def restrict(M): return M[np.ix_(idx, idx)]
    gens, tgens = [], []
    for i in range(n):
        ni = restrict(c[i].conj().T @ c[i]); gens.append(ni); tgens.append(ni)
        for j in range(i + 1, n):
            Eij = restrict(c[i].conj().T @ c[j]); gens += [Eij + Eij.conj().T, 1j * (Eij - Eij.conj().T)]
    return Group(f'fermions U({n}) on Lambda^{N}', gens, tgens, math.factorial(n))

def g_bosons(n, N):
    occs = [o for o in itertools.product(range(N + 1), repeat=n) if sum(o) == N]; ix = {o: k for k, o in enumerate(occs)}
    d = len(occs); gens, tgens = [], []
    for i in range(n):
        ni = np.diag([o[i] for o in occs]).astype(complex); gens.append(ni); tgens.append(ni)
        for j in range(i + 1, n):
            E = np.zeros((d, d), complex)
            for o in occs:
                if o[j] > 0:
                    o2 = list(o); o2[j] -= 1; o2[i] += 1; E[ix[tuple(o2)], ix[o]] = np.sqrt((o[i] + 1) * o[j])
            gens += [E + E.conj().T, 1j * (E - E.conj().T)]
    return Group(f'bosons U({n}) on Sym^{N}', gens, tgens, math.factorial(n))

def spin_mats(j):
    m = np.arange(j, -j - 1, -1); d = len(m); Jp = np.zeros((d, d), complex)
    for k in range(1, d): Jp[k - 1, k] = np.sqrt(j * (j + 1) - m[k] * (m[k] + 1))
    return (Jp + Jp.conj().T) / 2, (Jp - Jp.conj().T) / 2j, np.diag(m).astype(complex)

def g_spin(js):
    js = list(js); mats = [spin_mats(j) for j in js]
    from scipy.linalg import block_diag
    J = [block_diag(*[m[a] for m in mats]) for a in range(3)]
    return Group('SU(2) spin ' + '+'.join(str(j) for j in js), J, [J[2]], 2)

def g_collective(nq):
    sx = np.array([[0, 1], [1, 0]], complex) / 2; sy = np.array([[0, -1j], [1j, 0]]) / 2; sz = np.diag([0.5, -0.5]).astype(complex)
    S = [sum(emb(s, [2] * nq, q) for q in range(nq)) for s in (sx, sy, sz)]
    return Group(f'SU(2) collective on {nq} qubits', S, [S[2]], 2)

def g_u1(charges):
    D = np.diag(np.array(charges, float)).astype(complex)
    return Group(f'U(1) charges {list(charges)}', [D], [D], 1)

def g_su3_adjoint():
    l = np.zeros((8, 3, 3), complex)
    l[0][0, 1] = l[0][1, 0] = 1; l[1][0, 1] = -1j; l[1][1, 0] = 1j; l[2][0, 0] = 1; l[2][1, 1] = -1
    l[3][0, 2] = l[3][2, 0] = 1; l[4][0, 2] = -1j; l[4][2, 0] = 1j; l[5][1, 2] = l[5][2, 1] = 1
    l[6][1, 2] = -1j; l[6][2, 1] = 1j; l[7] = np.diag([1, 1, -2]) / np.sqrt(3)
    e = l / np.sqrt(2)
    gens = []
    for a in range(8):
        M = np.array([[np.trace(e[b].conj().T @ comm(l[a], e[c])) for c in range(8)] for b in range(8)])
        gens.append((M + M.conj().T) / 2)
    return Group('SU(3) adjoint', gens, [gens[2], gens[7]], 6)

# ----------------------------------------------------------------- f, Hessians, critical points
def parts(G, U, H):
    Hp = U.conj().T @ H @ U; Hp = (Hp + Hp.conj().T) / 2
    return Hp, G.P(Hp)

def cost(G, U, H):
    Hp, L = parts(G, U, H); return float(np.real(np.vdot(Hp - L, Hp - L)))

def hess_analytic(G, Hp, L, basis):
    nb = len(basis)
    A = np.array([G.P(1j * comm(Hp, K)) for K in basis]).reshape(nb, -1)
    Bm = np.array([comm(L, K) for K in basis]).reshape(nb, -1)
    C = np.array([comm(Hp, K) for K in basis]).reshape(nb, -1)
    X = np.real(Bm.conj() @ C.T)
    return -2 * np.real(A.conj() @ A.T) + X + X.T

def hess_autograd(G, U, H, basis):
    import torch
    c128 = torch.complex128
    Ut = torch.tensor(U, dtype=c128); Ht = torch.tensor(H, dtype=c128); Bt = torch.tensor(np.array(basis), dtype=c128)
    PB = torch.tensor(G.B, dtype=c128)
    def f(x):
        K = torch.einsum('p,pij->ij', x.to(c128), Bt)
        V = Ut @ torch.linalg.matrix_exp(1j * K); Hp = V.conj().T @ Ht @ V
        c = torch.real(torch.einsum('mij,ij->m', PB.conj(), Hp))
        L = torch.einsum('m,mij->ij', c.to(c128), PB); R = Hp - L
        return torch.real(torch.sum(R.conj() * R))
    return torch.autograd.functional.hessian(f, torch.zeros(len(basis), dtype=torch.float64)).numpy()

def jac_resid(G, Hp, L, basis):
    cols = []
    for K in basis:
        dH = 1j * comm(Hp, K); cols.append(1j * (comm(G.P(dH), Hp) + comm(L, dH)))
    Cf = np.array(cols).reshape(len(basis), -1); Bf = np.array(basis).reshape(len(basis), -1)
    return np.real(Bf.conj() @ Cf.T)

def lm_critical(G, U, H, basis, tol=1e-13, maxit=200):
    mu = 1e-3; Hp, L = parts(G, U, H)
    r = coords(1j * comm(L, Hp), basis); nr = np.linalg.norm(r)
    for it in range(maxit):
        if nr < tol: break
        J = jac_resid(G, Hp, L, basis); JTJ = J.T @ J; g = J.T @ r
        while True:
            xi = -np.linalg.lstsq(JTJ + mu * np.eye(len(basis)), g, rcond=None)[0]
            Un = U @ expm(1j * from_coords(xi, basis)); Hpn, Ln = parts(G, Un, H)
            rn = coords(1j * comm(Ln, Hpn), basis); nrn = np.linalg.norm(rn)
            if nrn < nr:
                U, Hp, L, r, nr = Un, Hpn, Ln, rn, nrn
                mu = max(mu / 5, 1e-12 * max(1.0, float(np.trace(JTJ)) / len(basis))); break
            mu *= 4
            if mu > 1e12: return U, nr
    return U, nr

def descend_batch(G, H, U0, iters=4000, device='cuda', gtol=1e-10):
    """Batched Riemannian gradient descent with per-sample Armijo step control (torch)."""
    import torch
    c128 = torch.complex128; dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    Ht = torch.tensor(H, dtype=c128, device=dev); PB = torch.tensor(G.B, dtype=c128, device=dev)
    U = torch.tensor(np.array(U0), dtype=c128, device=dev); n = U.shape[0]
    nH2 = float(np.real(np.vdot(H, H)))
    def ev(U):
        Hp = U.mH @ Ht @ U; Hp = (Hp + Hp.mH) / 2
        c = torch.real(torch.einsum('mij,nij->nm', PB.conj(), Hp))
        L = torch.einsum('nm,mij->nij', c.to(c128), PB)
        return nH2 - (c ** 2).sum(1), Hp, L
    f, Hp, L = ev(U); t = torch.full((n,), 0.05 / max(nH2, 1e-12), dtype=torch.float64, device=dev)
    for it in range(iters):
        Gr = -2j * (L @ Hp - Hp @ L); Gr = (Gr + Gr.mH) / 2; g2 = torch.real((Gr.conj() * Gr).sum((1, 2)))
        if float(torch.sqrt(g2).max()) < gtol * nH2: break
        Un = U @ torch.linalg.matrix_exp(-1j * t[:, None, None].to(c128) * Gr)
        fn, Hpn, Ln = ev(Un)
        ok = fn <= f - 0.3 * t * g2
        U = torch.where(ok[:, None, None], Un, U); f = torch.where(ok, fn, f)
        Hp = torch.where(ok[:, None, None], Hpn, Hp); L = torch.where(ok[:, None, None], Ln, L)
        t = torch.where(ok, torch.clamp(t * 1.5, max=5.0 / max(nH2, 1e-12)), t * 0.5)
        t = torch.clamp(t, min=1e-16)
    Gr = -2j * (L @ Hp - Hp @ L); gn = torch.sqrt(torch.real((Gr.conj() * Gr).sum((1, 2))))
    return U.cpu().numpy(), f.cpu().numpy(), gn.cpu().numpy(), it

# ----------------------------------------------------------------- classification
def centralizer_in_k(G, L, tol=1e-8):
    """ONB of {X in span(k_H,1) : [X,L] = 0}."""
    M = np.array([comm(b, L).reshape(-1) for b in G.B]).T
    M = np.concatenate([M.real, M.imag], axis=0)
    _, s, Vt = np.linalg.svd(M); scale = max(1.0, np.linalg.norm(L))
    null = Vt[np.sum(s > tol * scale):]
    Z = np.einsum('km,mij->kij', null, G.B)
    return real_span_onb(list(Z)) if len(Z) else np.zeros((0, G.d, G.d), complex)

def N1_form(G, Hp, L, tol_rel=1e-8):
    """min of Q on the L-commutant (Q = -2||P(i[H',K])||^2 there)."""
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
    A = np.array([G.P(1j * comm(Hp, K)) for K in basis]).reshape(len(basis), -1)
    return float(np.linalg.eigvalsh(-2 * np.real(A.conj() @ A.T))[0]), [len(g) for g in groups]

def classify(G, U, H, basis):
    Hp, L = parts(G, U, H); nH2 = float(np.real(np.vdot(H, H))); tol = 1e-8 * nH2
    grad = 2 * np.linalg.norm(comm(L, Hp))
    ev = np.linalg.eigvalsh(hess_analytic(G, Hp, L, basis))
    Z = centralizer_in_k(G, L)
    cdef = 0.0
    if len(Z):
        Mz = np.array([comm(Hp, X).reshape(-1) for X in Z]); cdef = float(np.linalg.norm(Mz, 2))
    tfixed = cdef < 1e-6 * np.sqrt(nH2)
    n1, _ = N1_form(G, Hp, L)
    out = dict(grad=float(grad), f=float(np.real(np.vdot(Hp - L, Hp - L))), hess_min=float(ev[0]),
               n_neg=int(np.sum(ev < -tol)), n_zero=int(np.sum(np.abs(ev) <= tol)), dimZ=len(Z), comm_def=cdef,
               tfixed=bool(tfixed), N1=n1, cand=bool(ev[0] >= -tol))
    if tfixed:
        E, V = np.linalg.eigh(Hp)
        wv = np.array([[np.real(V[:, i].conj() @ X @ V[:, i]) for X in Z] for i in range(len(E))])
        ell = np.array([np.real(V[:, i].conj() @ L @ V[:, i]) for i in range(len(E))])
        labs = np.round(wv, 6); same = np.all(np.abs(labs[:, None, :] - labs[None, :, :]) < 1e-5, axis=2)
        dE = E[:, None] - E[None, :]; dL = ell[:, None] - ell[None, :]
        off = ~same
        oi = float(np.min((dE * dL)[off])) if off.any() else np.inf
        out.update(ordiso_min=oi, ordiso=bool(oi > 1e-9 * nH2), key=make_key(ell, wv @ wv.T),
                   wmult=sorted(np.unique(labs, axis=0, return_counts=True)[1].tolist()))
    return out

def make_key(ell_sorted, gram):
    return (tuple(np.round(ell_sorted, 5)), tuple(np.round(gram, 5).reshape(-1)))

# ----------------------------------------------------------------- T-fixed assignments
def assignments(G, lam, chunk=200000):
    """Yield (perms, ellprime, ordiso) in chunks. perm tau: sorted-eigenvalue index i -> weight-basis vector tau[i].
    Non-multiplicity-free: deduplicated by sorting inside weight spaces."""
    d = G.d; lam = np.sort(lam); lab = G.label
    it = itertools.permutations(range(d)); seen = set()
    while True:
        block = list(itertools.islice(it, chunk))
        if not block: break
        T = np.array(block)
        if not G.multfree:
            labs = lab[T]  # label sequence along eigenvalue order
            keep = []
            for r, row in enumerate(labs):
                k = tuple(row)
                if k not in seen: seen.add(k); keep.append(r)
            T = T[keep]
            if len(T) == 0: continue
        E = np.zeros((len(T), d)); np.put_along_axis(E, T, lam[None, :], axis=1)   # E[c] for weight vector c
        ell = E @ G.Pdiag.T
        ellp = np.take_along_axis(ell, T, axis=1)                                      # ell at eigenvalue i
        labs = lab[T]; ch = labs[:, 1:] != labs[:, :-1]; df = np.diff(ellp, axis=1)
        sc = 1e-9 * max(1.0, float(np.sum(lam ** 2)))
        ordiso = np.all(np.where(ch, df > sc, df > -sc), axis=1)
        yield T, ellp, ordiso

def D_from_perm(G, lam, tau):
    E = np.zeros(G.d); E[np.array(tau)] = np.sort(lam); return E

def U_from_E(H, E):
    """U with U^dag H U = diag(E) (E a permutation of spec H)."""
    lam, V = np.linalg.eigh(H); order = np.argsort(lam)
    rank = {i: k for k, i in enumerate(np.argsort(E))}
    cols = [V[:, order[rank[c]]] for c in range(len(E))]
    return np.array(cols).T

def block_hessian_spectrum(G, E):
    """Closed-form Hessian spectrum at T-fixed D = diag(E) (weight basis): see GENERAL.md section 3."""
    d = G.d; ell = G.Pdiag @ E; ev = [0.0] * d; done = np.zeros((d, d), bool)
    for r in G.roots:
        pairs = r['pairs']; Ea = r['E']; W = float(np.sum(np.abs(Ea) ** 2))
        a = np.array([Ea[cp, c] for (cp, c) in pairs]); Dl = np.array([E[cp] - E[c] for (cp, c) in pairs])
        al = np.array([ell[cp] - ell[c] for (cp, c) in pairs]); alpha_xi = float(np.mean(al))
        assert np.max(np.abs(al - alpha_xi)) < 1e-8 * max(1, np.abs(E).max())
        v = a * Dl; M = alpha_xi * np.diag(Dl) - np.outer(v, v.conj()) / W
        for x in np.linalg.eigvalsh(M): ev += [2 * x, 2 * x]
        for (cp, c) in pairs: done[cp, c] = done[c, cp] = True
    for c in range(d):
        for cp in range(c + 1, d):
            if done[c, cp]: continue
            x = 0.0 if G.label[c] == G.label[cp] else 2 * (ell[c] - ell[cp]) * (E[c] - E[cp])
            ev += [x, x]
    return np.sort(np.array(ev))

def gue(d, seed):
    rng = np.random.default_rng(seed); X = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    return (X + X.conj().T) / (2 * np.sqrt(d))

def haar(d, rng):
    X = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)); Q, R = np.linalg.qr(X)
    return Q * (np.diag(R) / np.abs(np.diag(R)))
