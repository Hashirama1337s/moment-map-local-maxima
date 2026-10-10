#!/usr/bin/env python3
"""
impl_b.py -- BLIND implementation B of the QMEREO probe.

Implements CRITERIA.md (including Amendment 1) in NumPy/SciPy only (CPU).
Written without reading impl_a.py, probe.py, or any log in the tree.

Usage:
    py -3.11 impl_b.py EVAL <seed_H> <seed_S> <seed_U> <n>
        -> prints one line:  cost1 cost2 cost3 D   (8 significant digits, D = D(U, identity))
    py -3.11 impl_b.py EVAL <seed_H> <seed_S> <seed_U> <n> --alt
        -> same line, followed by '#'-prefixed lines with values under alternative
           readings of the ambiguous points (documented in impl_b_NOTES.md)
    py -3.11 impl_b.py SELFTEST
        -> PASS/FAIL per check

Conventions (see impl_b_NOTES.md for every choice and why):
  * qubit 1 is the leftmost tensor factor (most significant bit); subsystem A = qubits 1..n/2,
    so Tr_B traces out the LAST n/2 qubits and the reference algebra is M_{d_A} (x) 1_{d_B}.
  * a state vector psi of length d reshapes row-major to Psi[a, b] (a in A, b in B); rho_A = Psi Psi^dagger.
"""
import sys
import numpy as np
from scipy.linalg import expm

# ----------------------------------------------------------------------------------------------
# basic operators
# ----------------------------------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def kron_all(ops):
    out = np.array([[1.0 + 0j]])
    for op in ops:
        out = np.kron(out, op)
    return out


def site_ops(n, i, P, Q=None):
    """P on qubit i (0-based), optionally Q on qubit i+1, identity elsewhere."""
    ops = [I2] * n
    ops[i] = P
    if Q is not None:
        ops[i + 1] = Q
    return kron_all(ops)


def ptrace_B(M, dA, dB):
    return np.einsum('abcb->ac', M.reshape(dA, dB, dA, dB))


def ptrace_A(M, dA, dB):
    return np.einsum('abad->bd', M.reshape(dA, dB, dA, dB))


def fro2(M):
    return float(np.real(np.vdot(M, M)))


def op_norm(H):
    return float(np.max(np.abs(np.linalg.eigvalsh(H))))


# ----------------------------------------------------------------------------------------------
# planted chain, Haar scramble, random U
# ----------------------------------------------------------------------------------------------
def build_H_plant(n, seed_H):
    """Open XYZ chain with random fields; draw order: (Jx,Jy,Jz) per bond i=1..n-1, then (hx,hy,hz) per site."""
    rng = np.random.default_rng(seed_H)
    d = 2 ** n
    H = np.zeros((d, d), dtype=complex)
    for i in range(n - 1):
        Jx, Jy, Jz = rng.standard_normal(3)
        H += Jx * site_ops(n, i, SX, SX) + Jy * site_ops(n, i, SY, SY) + Jz * site_ops(n, i, SZ, SZ)
    for i in range(n):
        hx, hy, hz = rng.standard_normal(3)
        H += hx * site_ops(n, i, SX) + hy * site_ops(n, i, SY) + hz * site_ops(n, i, SZ)
    return H


def haar_unitary(d, rng):
    """A1: real Gaussian matrix first, then imaginary; QR; phase fix diag(R)/|diag(R)| applied to the columns of Q."""
    Gr = rng.standard_normal((d, d))
    Gi = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(Gr + 1j * Gi)
    dR = np.diag(R)
    ph = dR / np.abs(dR)
    return Q * ph[None, :]


def build_S(d, seed_S):
    return haar_unitary(d, np.random.default_rng(seed_S))


def build_U(d, seed_U):
    """A2: G real Gaussian d x d, K = (G + G^T)/2, U = expm(iK)."""
    rng = np.random.default_rng(seed_U)
    G = rng.standard_normal((d, d))
    K = 0.5 * (G + G.T)
    return expm(1j * K)


# ----------------------------------------------------------------------------------------------
# C1: minimal interaction
# ----------------------------------------------------------------------------------------------
def decompose(Hp, dA, dB):
    d = dA * dB
    HA = ptrace_B(Hp, dA, dB) / dB
    HB = ptrace_A(Hp, dA, dB) / dA
    c = np.trace(Hp) / d
    Hint = Hp - np.kron(HA, np.eye(dB)) - np.kron(np.eye(dA), HB) + c * np.eye(d)
    return HA, HB, c, Hint


def cost1(H, Hint):
    return fro2(Hint) / fro2(H)


# ----------------------------------------------------------------------------------------------
# C2: Carroll-Singh Schwinger entropy
# ----------------------------------------------------------------------------------------------
def ggm_basis(m):
    """A3 ordering: symmetric (j<k, row-major), antisymmetric (j<k, row-major), diagonal l=1..m-1. Frobenius-orthonormal."""
    L = []
    s2 = np.sqrt(2.0)
    for j in range(m):
        for k in range(j + 1, m):
            M = np.zeros((m, m), dtype=complex)
            M[j, k] = 1 / s2
            M[k, j] = 1 / s2
            L.append(M)
    for j in range(m):
        for k in range(j + 1, m):
            M = np.zeros((m, m), dtype=complex)
            M[j, k] = -1j / s2      # -i (E_jk - E_kj)/sqrt2
            M[k, j] = 1j / s2
            L.append(M)
    for l in range(1, m):
        M = np.zeros((m, m), dtype=complex)
        for j in range(l):
            M[j, j] = 1.0
        M[l, l] = -l
        M *= np.sqrt(1.0 / (l * (l + 1)))
        L.append(M)
    return np.array(L)


def _gram(Hint, T):
    """T: (N, d, d) operators. Returns M[j,j'] = Re Tr([Hint,T_j]^dagger [Hint,T_j'])."""
    C = Hint @ T - T @ Hint
    Cf = C.reshape(C.shape[0], -1)
    return np.real(Cf.conj() @ Cf.T)


def cpo_solve(Hint, LA, LB, dA, dB, n_starts=8, max_iter=200, rtol=1e-12):
    """A3: exact alternating minimisation of f(a,b) = ||[Hint, O_A (x) O_B]||_F^2 with ||a||=||b||=1."""
    d = dA * dB
    nA, nB = LA.shape[0], LB.shape[0]

    def f_val(a, b):
        OA = np.tensordot(a, LA, axes=(0, 0))
        OB = np.tensordot(b, LB, axes=(0, 0))
        T = np.kron(OA, OB)
        return fro2(Hint @ T - T @ Hint)

    best = None
    for s in range(n_starts):
        rng = np.random.default_rng(s)
        a = rng.standard_normal(nA)
        b = rng.standard_normal(nB)
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        f_old = f_val(a, b)
        f_new = f_old
        for _ in range(max_iter):
            OB = np.tensordot(b, LB, axes=(0, 0))
            TA = np.einsum('jac,bd->jabcd', LA, OB).reshape(nA, d, d)
            w, V = np.linalg.eigh(_gram(Hint, TA))
            a = V[:, 0].copy()
            OA = np.tensordot(a, LA, axes=(0, 0))
            TB = np.einsum('ac,kbd->kabcd', OA, LB).reshape(nB, d, d)
            w, V = np.linalg.eigh(_gram(Hint, TB))
            b = V[:, 0].copy()
            f_new = float(w[0])
            if abs(f_new - f_old) < rtol * max(abs(f_old), 1e-300):
                break
            f_old = f_new
        f_fin = f_val(a, b)
        if best is None or f_fin < best[0]:
            best = (f_fin, a.copy(), b.copy())
    return best


def fix_phases(V):
    """Column phase convention: the largest-modulus component of each eigenvector is made real and positive
    (ties -> lowest index). CRITERIA.md / the paper do not state a phase convention for |b_k>; see NOTES."""
    idx = np.argmax(np.abs(V), axis=0)
    ph = V[idx, np.arange(V.shape[1])]
    ph = ph / np.abs(ph)
    return V * np.conj(ph)[None, :]


def schwinger_terms(Hp, OA, OB, dA, dB, env="mixed", phase_fix=True):
    """Returns arrays (Sddot_lin[j], Sddot_pointer[j]) over the d_A peaked states.
    env="mixed" (Amendment 2, binding): rho_j(0) = |a_j><a_j| (x) 1_B/d_B.
    env="pure"  (sensitivity variant): rho_j(0) = |psi_j><psi_j| with |psi_j> = |a_j> (x) |u_B>, Eq. 27,
                |u_B> = (1/sqrt d_B) sum_k |b_k> with the phase convention of fix_phases (if phase_fix)."""
    wA, VA = np.linalg.eigh(OA)
    if env == "pure":
        wB, VB = np.linalg.eigh(OB)
        if phase_fix:
            VB = fix_phases(VB)
        uB = VB.sum(axis=1) / np.sqrt(dB)
    S_lin = np.zeros(dA)
    S_pt = np.zeros(dA)
    for j in range(dA):
        if env == "pure":
            psi = np.kron(VA[:, j], uB)
            rho = np.outer(psi, psi.conj())
        else:
            rho = np.kron(np.outer(VA[:, j], VA[:, j].conj()), np.eye(dB) / dB)
        comm1 = Hp @ rho - rho @ Hp
        rho_d = -1j * comm1
        rho_dd = -(Hp @ comm1 - comm1 @ Hp)
        rA = ptrace_B(rho, dA, dB)
        rAd = ptrace_B(rho_d, dA, dB)
        rAdd = ptrace_B(rho_dd, dA, dB)
        S_lin[j] = -2.0 * float(np.real(np.trace(rAdd @ rA) + np.trace(rAd @ rAd)))
        p = np.real(np.einsum('ij,jk,ki->i', VA.conj().T, rA, VA))
        pd = np.real(np.einsum('ij,jk,ki->i', VA.conj().T, rAd, VA))
        pdd = np.real(np.einsum('ij,jk,ki->i', VA.conj().T, rAdd, VA))
        S_pt[j] = -2.0 * float(np.sum(pd ** 2 + p * pdd))
    return S_lin, S_pt


def cost2(Hp, Hint, Hop, dA, dB, return_details=False):
    LA = ggm_basis(dA)
    LB = ggm_basis(dB)
    f, a, b = cpo_solve(Hint, LA, LB, dA, dB)
    OA = np.tensordot(a, LA, axes=(0, 0))
    OB = np.tensordot(b, LB, axes=(0, 0))
    S_lin, S_pt = schwinger_terms(Hp, OA, OB, dA, dB)
    c2 = float(np.mean(np.maximum(S_lin, S_pt))) / Hop ** 2
    if return_details:
        return c2, dict(f=f, a=a, b=b, OA=OA, OB=OB, S_lin=S_lin, S_pt=S_pt)
    return c2


# ----------------------------------------------------------------------------------------------
# C3: Adil et al. purity cost, Eq. 12 discretisation, inner Adam per A4
# ----------------------------------------------------------------------------------------------
DT3 = 0.2
ADAM_LR = 0.05
ADAM_STEPS = 200
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8


def make_cost3_fn(Hp, Hop, dA, dB):
    """Returns cost_grad(theta) -> (cost, grad) with theta = [Re a, Im a, Re b, Im b] (unnormalised alpha, beta)."""
    d = dA * dB
    Hn = Hp / Hop                       # time unit 1/||H||_op
    K = 2 * d                           # number of steps T/dt = 2d
    w, W = np.linalg.eigh(Hn)
    V = (W * np.exp(-1j * w * DT3)) @ W.conj().T     # one-step propagator exp(-i Hn dt)
    Vd = V.conj().T

    def cost_grad(theta):
        al = theta[0:dA] + 1j * theta[dA:2 * dA]
        be = theta[2 * dA:2 * dA + dB] + 1j * theta[2 * dA + dB:]
        na, nb = np.linalg.norm(al), np.linalg.norm(be)
        ah, bh = al / na, be / nb
        psi = np.kron(ah, bh)
        cost = 0.0
        gs = []
        for k in range(K + 1):
            if k > 0:
                psi = V @ psi
            Psi = psi.reshape(dA, dB)
            rA = Psi @ Psi.conj().T
            P = float(np.real(np.vdot(rA, rA)))
            cost += 1.0 - P
            gs.append(-(2.0 * rA @ Psi).reshape(-1))   # d(1-P_k)/d conj(psi_k)
        # backward: dcost/d conj(psi_0) = sum_k (V^dagger)^k g_k, Horner form
        acc = gs[K]
        for k in range(K - 1, -1, -1):
            acc = Vd @ acc + gs[k]
        G = (acc / K).reshape(dA, dB)      # prefactor dt/T = 1/K
        cost /= K
        h_a = G @ bh.conj()
        h_b = G.T @ ah.conj()
        ga = (h_a - ah * np.real(np.vdot(h_a, ah))) / na     # chain rule through normalisation
        gb = (h_b - bh * np.real(np.vdot(h_b, bh))) / nb
        grad = np.concatenate([2 * ga.real, 2 * ga.imag, 2 * gb.real, 2 * gb.imag])
        return cost, grad

    return cost_grad


def adam_minimise(cost_grad, theta0, lr=ADAM_LR, steps=ADAM_STEPS):
    theta = theta0.astype(float).copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    traj_min = np.inf
    for t in range(1, steps + 1):
        c, g = cost_grad(theta)
        traj_min = min(traj_min, c)
        m = ADAM_B1 * m + (1 - ADAM_B1) * g
        v = ADAM_B2 * v + (1 - ADAM_B2) * g * g
        mh = m / (1 - ADAM_B1 ** t)
        vh = v / (1 - ADAM_B2 ** t)
        theta -= lr * mh / (np.sqrt(vh) + ADAM_EPS)
    c_final, _ = cost_grad(theta)
    traj_min = min(traj_min, c_final)
    return c_final, traj_min, theta


def cost3(Hp, Hop, dA, dB, return_details=False, n_starts=8):
    cg = make_cost3_fn(Hp, Hop, dA, dB)
    finals, trajmins = [], []
    for s in range(n_starts):
        rng = np.random.default_rng(s)
        ar = rng.standard_normal(dA)
        ai = rng.standard_normal(dA)
        br = rng.standard_normal(dB)
        bi = rng.standard_normal(dB)
        theta0 = np.concatenate([ar, ai, br, bi])
        c_final, c_traj, _ = adam_minimise(cg, theta0)
        finals.append(c_final)
        trajmins.append(c_traj)
    best = float(min(finals))
    if return_details:
        return best, dict(finals=finals, trajmins=trajmins)
    return best


# ----------------------------------------------------------------------------------------------
# gauge-invariant distance (A5 closed form) and the projector form (self-test only)
# ----------------------------------------------------------------------------------------------
def alg_basis(U, dA, dB, side):
    """Orthonormal basis (rows, flattened d^2) of U (M_{dA} (x) 1) U^dag ('A') or U (1 (x) M_{dB}) U^dag ('B')."""
    d = dA * dB
    Ur = U.reshape(d, dA, dB)
    if side == 'A':
        X = np.einsum('pjb,qkb->jkpq', Ur, Ur.conj()) / np.sqrt(dB)
        return X.reshape(dA * dA, d * d)
    X = np.einsum('paj,qak->jkpq', Ur, Ur.conj()) / np.sqrt(dA)
    return X.reshape(dB * dB, d * d)


def distance(U, V, dA, dB):
    XA = alg_basis(U, dA, dB, 'A')
    YA = alg_basis(V, dA, dB, 'A')
    YB = alg_basis(V, dA, dB, 'B')
    sA = float(np.sum(np.abs(XA.conj() @ YA.T) ** 2))
    sB = float(np.sum(np.abs(XA.conj() @ YB.T) ** 2))
    DA = np.sqrt(max(0.0, 2.0 * (1.0 - sA / dA ** 2)))
    DAp = np.sqrt(max(0.0, 2.0 * (1.0 - sB / dA ** 2)))
    return float(min(DA, DAp))


def distance_projector_form(U, V, dA, dB):
    """Literal CRITERIA definition: projectors on HS space as d^2 x d^2 matrices (self-test cross-check)."""
    def proj(B):
        return B.T @ B.conj()      # sum_i vec(X_i) vec(X_i)^dagger
    PU = proj(alg_basis(U, dA, dB, 'A'))
    PV = proj(alg_basis(V, dA, dB, 'A'))
    PVp = proj(alg_basis(V, dA, dB, 'B'))
    nU = np.sqrt(fro2(PU))
    return float(min(np.sqrt(fro2(PU - PV)), np.sqrt(fro2(PU - PVp))) / nU)


def swap_unitary(dA, dB):
    d = dA * dB
    Sw = np.zeros((d, d), dtype=complex)
    for a in range(dA):
        for b in range(dB):
            Sw[b * dA + a, a * dB + b] = 1.0
    return Sw


# ----------------------------------------------------------------------------------------------
# harness
# ----------------------------------------------------------------------------------------------
def setup(seed_H, seed_S, seed_U, n):
    d = 2 ** n
    dA = dB = 2 ** (n // 2)
    Hplant = build_H_plant(n, seed_H)
    S = build_S(d, seed_S)
    H = S @ Hplant @ S.conj().T
    U = build_U(d, seed_U)
    return dict(n=n, d=d, dA=dA, dB=dB, Hplant=Hplant, S=S, H=H, U=U)


def evaluate(seed_H, seed_S, seed_U, n, alt=False):
    st = setup(seed_H, seed_S, seed_U, n)
    dA, dB, H, U = st['dA'], st['dB'], st['H'], st['U']
    Hp = U.conj().T @ H @ U
    HA, HB, c, Hint = decompose(Hp, dA, dB)
    Hop = op_norm(H)
    c1 = cost1(H, Hint)
    c2, det2 = cost2(Hp, Hint, Hop, dA, dB, return_details=True)
    c3, det3 = cost3(Hp, Hop, dA, dB, return_details=True)
    D = distance(U, np.eye(dA * dB), dA, dB)
    print(f"{c1:.8g} {c2:.8g} {c3:.8g} {D:.8g}")
    if alt:
        S_lin_p, S_pt_p = schwinger_terms(Hp, det2['OA'], det2['OB'], dA, dB, env="pure", phase_fix=True)
        c2_pure = float(np.mean(np.maximum(S_lin_p, S_pt_p))) / Hop ** 2
        S_lin_raw, S_pt_raw = schwinger_terms(Hp, det2['OA'], det2['OB'], dA, dB, env="pure", phase_fix=False)
        c2_raw = float(np.mean(np.maximum(S_lin_raw, S_pt_raw))) / Hop ** 2
        c2_lin_only = float(np.mean(det2['S_lin'])) / Hop ** 2
        print(f"# cpo_f_min={det2['f']:.8g}  ||H||_op={Hop:.8g}  ||H||_F^2={fro2(H):.8g}")
        print(f"# cost2 pure-state variant (Eq. 27 |u_B>, largest-component-positive phases): {c2_pure:.8g}")
        print(f"# cost2 pure-state variant with raw numpy.eigh phases for |b_k>: {c2_raw:.8g}")
        print(f"# cost2 using S_lin only (no max with S_pointer): {c2_lin_only:.8g}")
        print(f"# cost2 S_lin per state: {' '.join(f'{x:.6g}' for x in det2['S_lin'] / Hop ** 2)}")
        print(f"# cost2 S_pt  per state: {' '.join(f'{x:.6g}' for x in det2['S_pt'] / Hop ** 2)}")
        print(f"# cost3 finals per seed:   {' '.join(f'{x:.8g}' for x in det3['finals'])}")
        print(f"# cost3 traj-min per seed: {' '.join(f'{x:.8g}' for x in det3['trajmins'])}")
        print(f"# cost3 alt (min over trajectory and seeds): {min(det3['trajmins']):.8g}")
        print(f"# D(U,S) (plant recovery distance per A6): {distance(U, st['S'], dA, dB):.8g}")


def selftest():
    results = []

    def check(name, ok, info=""):
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {info}")

    # Zero-distance bar: the literal A5 closed form sqrt(2(1 - s/d_A^2)) has a sqrt(machine-eps) floor
    # (~3e-8 at d=64) because s/d_A^2 = 1 - O(1e-16) is squared-rooted; 1e-6 is far above that floor and
    # far below any physically relevant scale (plant-recovery threshold 0.05).
    ZTOL = 1e-6
    for n in (4, 6):
        d = 2 ** n
        dA = dB = 2 ** (n // 2)
        I = np.eye(d)
        D11 = distance(I, I, dA, dB)
        check(f"D(1,1) = 0 (n={n})", D11 < ZTOL, f"D={D11:.3e}")

        rng = np.random.default_rng(123)
        WA = haar_unitary(dA, rng)
        WB = haar_unitary(dB, rng)
        Dloc = distance(np.kron(WA, WB), I, dA, dB)
        check(f"D(W_A (x) W_B, 1) = 0 (n={n})", Dloc < ZTOL, f"D={Dloc:.3e}")

        Dsw = distance(swap_unitary(dA, dB), I, dA, dB)
        check(f"D(swap-permutation, 1) = 0 (n={n})", Dsw < ZTOL, f"D={Dsw:.3e}")

        Uh = haar_unitary(d, np.random.default_rng(7))
        Dh = distance(Uh, I, dA, dB)
        check(f"D(Haar, 1) > 0.3 (n={n})", Dh > 0.3, f"D={Dh:.6f}")

        # extras (not required by the task, labelled as such)
        Dinv = distance(Uh @ np.kron(WA, WB), Uh, dA, dB)
        check(f"[extra] D(U (W_A (x) W_B), U) = 0 for Haar U (n={n})", Dinv < ZTOL, f"D={Dinv:.3e}")
        if n == 4:
            Vh = haar_unitary(d, np.random.default_rng(8))
            Dc, Dp = distance(Uh, Vh, dA, dB), distance_projector_form(Uh, Vh, dA, dB)
            check("[extra] A5 closed form == projector form (n=4)", abs(Dc - Dp) < 1e-10, f"{Dc:.10f} vs {Dp:.10f}")

    # H_int partial traces and Sddot_lin >= 0 on the (1,2,3,4) instance, plus (4,5,6,4)
    for (sh, ss, su, n) in [(1, 2, 3, 4), (4, 5, 6, 4)]:
        st = setup(sh, ss, su, n)
        dA, dB, H, U = st['dA'], st['dB'], st['H'], st['U']
        Hp = U.conj().T @ H @ U
        HA, HB, c, Hint = decompose(Hp, dA, dB)
        scale = np.sqrt(fro2(H))
        tB = np.sqrt(fro2(ptrace_B(Hint, dA, dB))) / scale
        tA = np.sqrt(fro2(ptrace_A(Hint, dA, dB))) / scale
        check(f"H_int partial traces vanish ({sh},{ss},{su},{n})", tA < 1e-12 and tB < 1e-12,
              f"|Tr_A|/|H|={tA:.2e} |Tr_B|/|H|={tB:.2e}")
        herm = np.sqrt(fro2(Hint - Hint.conj().T)) / scale
        check(f"[extra] H_int Hermitian ({sh},{ss},{su},{n})", herm < 1e-12, f"{herm:.2e}")
        Hop = op_norm(H)
        c2, det = cost2(Hp, Hint, Hop, dA, dB, return_details=True)
        check(f"Sddot_lin(0) >= 0 for all peaked states ({sh},{ss},{su},{n})", np.all(det['S_lin'] >= -1e-10),
              f"min={det['S_lin'].min():.3e} max={det['S_lin'].max():.3e}")
        # extra: the CPO solution is a stationary point: f equals both Rayleigh quotients
        check(f"[extra] CPO f >= 0 and finite ({sh},{ss},{su},{n})", np.isfinite(det['f']) and det['f'] >= 0,
              f"f={det['f']:.6e}")

    # extra: analytic cost3 gradient vs central finite differences (n=4)
    st = setup(1, 2, 3, 4)
    dA, dB, H, U = st['dA'], st['dB'], st['H'], st['U']
    Hp = U.conj().T @ H @ U
    cg = make_cost3_fn(Hp, op_norm(H), dA, dB)
    rng = np.random.default_rng(99)
    th = rng.standard_normal(2 * dA + 2 * dB)
    c0, g = cg(th)
    eps = 1e-6
    gfd = np.zeros_like(th)
    for i in range(len(th)):
        e = np.zeros_like(th)
        e[i] = eps
        gfd[i] = (cg(th + e)[0] - cg(th - e)[0]) / (2 * eps)
    rel = np.linalg.norm(g - gfd) / np.linalg.norm(gfd)
    check("[extra] cost3 analytic gradient == finite differences (n=4)", rel < 1e-6, f"rel err={rel:.2e}")
    # extra: cost3 invariant to the unnormalised scale of alpha, beta
    c1_, _ = cg(th)
    th2 = th.copy()
    th2[:2 * dA] *= 3.0
    th2[2 * dA:] *= 0.2
    c2_, _ = cg(th2)
    check("[extra] cost3 scale-invariant in (alpha, beta)", abs(c1_ - c2_) < 1e-12, f"{c1_:.12f} vs {c2_:.12f}")

    print("ALL PASS" if all(results) else f"{sum(not r for r in results)} FAILED")


# ----------------------------------------------------------------------------------------------
# Amendment 3: readout R5, local-minimum test at the lattice cut
# ----------------------------------------------------------------------------------------------
def herm_from_P(P):
    return 0.5 * (P + P.T) + 0.5j * (P - P.T)


def P_from_herm(K):
    """Inverse of herm_from_P on Hermitian K: P = Re K + Im K (Re K symmetric, Im K antisymmetric)."""
    return K.real + K.imag


def U_of_P(S, Pvec, d):
    return S @ expm(1j * herm_from_P(Pvec.reshape(d, d)))


def herm_basis(m):
    """Orthonormal Hermitian basis of M_m: the GGM set plus 1/sqrt(m)."""
    return np.concatenate([ggm_basis(m), (np.eye(m, dtype=complex) / np.sqrt(m))[None]], axis=0)


def gauge_split(dA, dB):
    """Returns (G, N): orthonormal bases (columns, length d^2) of the gauge subspace
    {P : herm(P) = W_A (x) 1 + 1 (x) W_B} and of its orthogonal complement (the non-gauge subspace)."""
    d = dA * dB
    vecs = [P_from_herm(np.kron(W, np.eye(dB))).ravel() for W in herm_basis(dA)]
    vecs += [P_from_herm(np.kron(np.eye(dA), W)).ravel() for W in herm_basis(dB)]
    M = np.array(vecs).T
    Uu, s, _ = np.linalg.svd(M, full_matrices=True)
    r = int(np.sum(s > 1e-10))
    assert r == dA * dA + dB * dB - 1, (r, s)
    return Uu[:, :r], Uu[:, r:]


def make_r5_f(crit, st, mode="full"):
    """f(Pvec) = C_crit(U(P)). crit 2: mode 'full' (CPO re-solved) or 'fixed' (CPO from P = 0)."""
    H, S, dA, dB, d = st['H'], st['S'], st['dA'], st['dB'], st['d']
    Hop = op_norm(H)
    HF2 = fro2(H)
    fixed = None
    if crit == 2 and mode == "fixed":
        Hp0 = S.conj().T @ H @ S
        _, _, _, Hint0 = decompose(Hp0, dA, dB)
        LA, LB = ggm_basis(dA), ggm_basis(dB)
        f_, a, b = cpo_solve(Hint0, LA, LB, dA, dB)
        fixed = (np.tensordot(a, LA, axes=(0, 0)), np.tensordot(b, LB, axes=(0, 0)))

    def f(Pvec):
        U = U_of_P(S, Pvec, d)
        Hp = U.conj().T @ H @ U
        HA, HB, c, Hint = decompose(Hp, dA, dB)
        if crit == 1:
            return fro2(Hint) / HF2
        if crit == 2:
            if fixed is not None:
                S_lin, S_pt = schwinger_terms(Hp, fixed[0], fixed[1], dA, dB)
                return float(np.mean(np.maximum(S_lin, S_pt))) / Hop ** 2
            return cost2(Hp, Hint, Hop, dA, dB)
        if crit == 3:
            return cost3(Hp, Hop, dA, dB)
        raise ValueError(crit)

    return f


def fd_gradient(f, N, h=1e-4):
    """Central differences along the orthonormal columns of N; returns the coefficient vector."""
    return np.array([(f(h * N[:, k]) - f(-h * N[:, k])) / (2 * h) for k in range(N.shape[1])])


def fd_hessian_full(f, dim, h=1e-3, f0=None):
    """Central-difference Hessian on the full P basis (dim x dim)."""
    if f0 is None:
        f0 = f(np.zeros(dim))
    Hs = np.zeros((dim, dim))
    E = np.eye(dim) * h
    for i in range(dim):
        Hs[i, i] = (f(2 * E[i]) - 2 * f0 + f(-2 * E[i])) / (4 * h * h)
        for j in range(i + 1, dim):
            v = (f(E[i] + E[j]) - f(E[i] - E[j]) - f(-E[i] + E[j]) + f(-E[i] - E[j])) / (4 * h * h)
            Hs[i, j] = v
            Hs[j, i] = v
    return Hs


def eig_counts(lams, tol=1e-9):
    return int(np.sum(lams < -tol)), int(np.sum(np.abs(lams) <= tol)), int(np.sum(lams > tol))


def classify(f0, gnorm, n_neg):
    if abs(gnorm) / abs(f0) >= 1e-6:
        return "NOT CRITICAL"
    if n_neg is None:
        return "UNDETERMINED (Hessian not computed)"
    return "STRICT LOCAL MIN" if n_neg == 0 else "SADDLE"


def r5(seed_H, seed_S, n, crit, hessian=None):
    """hessian: None -> auto (full basis for crit 1 and the fixed-CPO C2 variant; not computed for full C2 / C3)."""
    import time
    st = setup(seed_H, seed_S, 0, n)
    dA, dB, d = st['dA'], st['dB'], st['d']
    dim = d * d
    Gb, N = gauge_split(dA, dB)
    f = make_r5_f(crit, st)
    t0 = time.time()
    f0 = f(np.zeros(dim))
    t_eval = time.time() - t0
    c = fd_gradient(f, N)
    gnorm = float(np.linalg.norm(c))
    t_grad = time.time() - t0
    lines = [f"# n={n} d={d} dim(P)={dim} gauge dim={Gb.shape[1]} non-gauge dim={N.shape[1]}; "
             f"gradient: central differences h=1e-4 along the full orthonormal non-gauge basis ({N.shape[1]} directions); "
             f"t_eval={t_eval:.3g}s t_grad={t_grad:.1f}s"]
    do_hess = hessian if hessian is not None else (crit == 1)
    counts = None
    if do_hess:
        Hs = fd_hessian_full(f, dim, f0=f0)
        lams = np.linalg.eigvalsh(N.T @ Hs @ N)
        counts = eig_counts(lams)
        lams_g = np.linalg.eigvalsh(Gb.T @ Hs @ Gb)
        lines.append(f"# Hessian: central differences h=1e-3 on the full {dim}-basis, projected to non-gauge; "
                     f"min eig={lams.min():.6g} max eig={lams.max():.6g}; gauge-block max|eig|={np.max(np.abs(lams_g)):.3g}; "
                     f"t_total={time.time() - t0:.1f}s")
    if crit == 1:
        # analytic gradient: U(P) = S e^{iK}, dH'/dK = i[H', K]  =>  df/dK = 2 i [H_int, H'] / ||H||_F^2 (HS gradient),
        # mapped to P-space by P = Re + Im (same map as P_from_herm).
        S, H = st['S'], st['H']
        Hp = S.conj().T @ H @ S
        _, _, _, Hint = decompose(Hp, dA, dB)
        G_an = 2j * (Hint @ Hp - Hp @ Hint) / fro2(H)
        g_an = N.T @ P_from_herm(G_an).ravel()
        g_raw = N.T @ P_from_herm(1j * (Hp @ Hint - Hint @ Hp)).ravel()   # the coordinator's i[H', H_int], unscaled
        rel = np.linalg.norm(c - g_an) / np.linalg.norm(g_an)
        cosang = float(c @ g_raw / (np.linalg.norm(c) * np.linalg.norm(g_raw)))
        kappa = float(c @ g_raw / (g_raw @ g_raw))
        gauge_part = np.linalg.norm(Gb.T @ P_from_herm(G_an).ravel()) / np.linalg.norm(P_from_herm(G_an).ravel())
        lines.append(f"# C1 identity: |g_fd - g_an|/|g_an| = {rel:.3e} with g_an = non-gauge part of "
                     f"(2/||H||_F^2) * Pmap(i[H_int, H']) = -(2/||H||_F^2) * Pmap(i[H', H_int]); "
                     f"cos(g_fd, Pmap(i[H',H_int])) = {cosang:+.8f}, best scalar kappa = {kappa:.8g} "
                     f"(expected -2/||H||_F^2 = {-2 / fro2(H):.8g}); gauge fraction of the analytic gradient = {gauge_part:.2e}")
    if crit == 2:
        ff = make_r5_f(2, st, mode="fixed")
        f0f = ff(np.zeros(dim))
        cf = fd_gradient(ff, N)
        gf = float(np.linalg.norm(cf))
        lines.append(f"# C2 fixed-CPO variant: f0={f0f:.8g} grad_norm={gf:.8g} "
                     f"({'NOT CRITICAL' if gf / abs(f0f) >= 1e-6 else 'critical by the 1e-6 bar'})")
        if hessian is not False:
            Hs = fd_hessian_full(ff, dim, f0=f0f)
            lams = np.linalg.eigvalsh(N.T @ Hs @ N)
            nn, nz, npos = eig_counts(lams)
            lines.append(f"# C2 fixed-CPO Hessian (full basis, projected): n_neg={nn} n_zero={nz} n_pos={npos} "
                         f"min eig={lams.min():.6g} max eig={lams.max():.6g}; full-C2 Hessian NOT computed "
                         f"(131k CPO re-solves exceeds the time budget); t_total={time.time() - t0:.1f}s")
    if crit == 3 and not do_hess:
        lines.append("# C3 Hessian NOT computed: 131k inner re-optimisations exceeds the time budget")
    cls = classify(f0, gnorm, None if counts is None else counts[0])
    cnt = " ".join(str(x) for x in counts) if counts is not None else "not_computed not_computed not_computed"
    print(f"{f0:.8g} {gnorm:.8g} {cnt} {cls}")
    for ln in lines:
        print(ln)


def main(argv):
    if len(argv) >= 2 and argv[1] == "SELFTEST":
        selftest()
        return
    if len(argv) >= 6 and argv[1] == "R5":
        seed_H, seed_S, n, crit = int(argv[2]), int(argv[3]), int(argv[4]), int(argv[5])
        r5(seed_H, seed_S, n, crit)
        return
    if len(argv) >= 6 and argv[1] == "EVAL":
        seed_H, seed_S, seed_U, n = int(argv[2]), int(argv[3]), int(argv[4]), int(argv[5])
        alt = "--alt" in argv[6:]
        evaluate(seed_H, seed_S, seed_U, n, alt=alt)
        return
    print(__doc__)
    sys.exit(2)


if __name__ == "__main__":
    main(sys.argv)
