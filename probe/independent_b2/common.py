"""implementation B2 common helpers.

Conventions
  * spectrum lam(n, k) = sorted eigenvalues of impl_b.build_H_plant(n, 1000 + k)  (implementation B's "chain(1000+k)").
  * bitstring s is an integer 0..N-1, qubit 1 = most significant bit: s_j = (s >> (n-1-j)) & 1, j = 0..n-1.
  * z_j(s) = 2 s_j - 1  (= +-1).
  * a placement is a rank array r of length N:  E_s = lam[r[s]]  (rank 0 = lowest eigenvalue).
  * cost(r) = ||E - L||^2 / ||lam||^2,  L = least-squares fit of E on span{1, z_1..z_n}.
    The functions 1, z_j are mutually orthogonal over {0,1}^n with squared norm N, hence
    ||E - L||^2 = sum lam^2 - (sum lam)^2 / N - sum_j (z_j . E)^2 / N.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.dirname(HERE)
if PROBE not in sys.path:
    sys.path.insert(0, PROBE)
import impl_b as B  # noqa: E402  (only build_H_plant is used)


def spectrum(n, k):
    H = B.build_H_plant(n, 1000 + k)
    return np.sort(np.linalg.eigvalsh(H))


def bits(n):
    """S[s, j] = s_j in {0,1}, shape (N, n)."""
    N = 2 ** n
    s = np.arange(N)
    return np.array([(s >> (n - 1 - j)) & 1 for j in range(n)], dtype=np.int64).T


def zmat(n):
    return 2 * bits(n) - 1


def fit_v(E, Z):
    """v_j = z_j . E  (works for E of shape (N,) or (R, N))."""
    return E @ Z


def cost_of_E(E, lam, Z):
    N = len(lam)
    lam2 = float(np.dot(lam, lam))
    v = E @ Z
    F = np.sum(v * v, axis=-1)
    tot = np.sum(E, axis=-1)
    return (lam2 - tot * tot / N - F / N) / lam2


def cost(r, lam, Z):
    return float(cost_of_E(lam[np.asarray(r)], lam, Z))


def cost_lstsq(r, lam, Z):
    """Independent check of cost via an explicit least-squares fit."""
    E = lam[np.asarray(r)]
    A = np.hstack([np.ones((len(E), 1)), Z.astype(float)])
    coef, *_ = np.linalg.lstsq(A, E, rcond=None)
    res = E - A @ coef
    return float(np.dot(res, res) / np.dot(lam, lam))


def ranks_from_order(order):
    """order[i] = bitstring at position i (ascending)  ->  r[s] = position of s."""
    r = np.empty(len(order), dtype=np.int64)
    r[np.asarray(order)] = np.arange(len(order))
    return r


def is_order_isomorphic(r, lam, Z, tol=1e-12):
    """True iff the 1-local fit L of E_s = lam[r[s]] has exactly the ordering r (strict, with margin).
    Returns (flag, min gap of sorted L)."""
    E = lam[np.asarray(r)]
    v = E @ Z
    L = Z @ v  # L - c0 up to the positive factor 1/N; ordering is what matters
    order = np.argsort(L, kind="stable")
    gaps = np.diff(L[order])
    ok = np.array_equal(ranks_from_order(order), np.asarray(r))
    return bool(ok and gaps.min() > tol * max(1.0, np.abs(L).max())), float(gaps.min())
