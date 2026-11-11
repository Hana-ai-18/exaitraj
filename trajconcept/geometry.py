"""Bieu dien song anh X <-> (p1, beta1, nu, omega) (nhu SAMO).

X: (n, L, 2). Thoi gian la chi so (dt = 1). nu = log toc do (L-1), omega = toc do re (L-2).
Song anh chinh xac tru khi co buoc dung hoan toan (toc do = 0): khi do ta san toc do o S_MIN
(sai so <= S_MIN moi buoc, kiem thu o tests/test_core.py).
"""
from __future__ import annotations
import numpy as np

S_MIN = 1e-4


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def encode(X: np.ndarray):
    X = np.asarray(X, dtype=np.float64)
    d = np.diff(X, axis=1)                                  # (n, L-1, 2)
    s = np.maximum(np.hypot(d[..., 0], d[..., 1]), S_MIN)
    beta = np.arctan2(d[..., 1], d[..., 0])
    # buoc dung hoan toan: huong khong xac dinh -> keo huong truoc do (khong tao quay gia)
    stopped = np.hypot(d[..., 0], d[..., 1]) < S_MIN
    if stopped.any():
        for i in np.where(stopped.any(1))[0]:
            for k in range(beta.shape[1]):
                if stopped[i, k]:
                    beta[i, k] = beta[i, k - 1] if k > 0 else 0.0
    omega = wrap(np.diff(beta, axis=1))                     # (n, L-2)
    return dict(p1=X[:, 0], beta1=beta[:, 0], nu=np.log(s), omega=omega)


def decode(p1, beta1, nu, omega):
    n = nu.shape[0]
    beta = np.concatenate([beta1[:, None], beta1[:, None] + np.cumsum(omega, axis=1)], axis=1)
    s = np.exp(nu)
    d = np.stack([s * np.cos(beta), s * np.sin(beta)], axis=-1)
    X = np.concatenate([p1[:, None, :], p1[:, None, :] + np.cumsum(d, axis=1)], axis=1)
    return X


def u_of(enc):
    """phan du u = concat(nu, omega), (n, 2L-3)."""
    return np.concatenate([enc["nu"], enc["omega"]], axis=1)


def split_u(u, L):
    return u[:, : L - 1], u[:, L - 1:]


def linear_basis(L):
    """Co so truc chuan V (4, 2L-3): [muc nu, xu huong nu, muc omega, xu huong omega].
    Cac muc K tuyen tinh theo (nu, omega) nam het trong span(V) -> twin giu CHINH XAC (SAMO, muc 5.3.2)."""
    nl, ol = L - 1, L - 2
    def lvl_trend(n):
        t = np.arange(n) - (n - 1) / 2.0
        a = np.ones(n) / np.sqrt(n)
        b = t / np.linalg.norm(t)
        return a, b
    a1, b1 = lvl_trend(nl); a2, b2 = lvl_trend(ol)
    V = np.zeros((4, nl + ol))
    V[0, :nl] = a1; V[1, :nl] = b1; V[2, nl:] = a2; V[3, nl:] = b2
    return V
