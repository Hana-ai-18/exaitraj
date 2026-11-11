"""Kiem thu phan can torch: toan tu nhieu, DTW, twin giu K tuyen tinh chinh xac, chieu giam sai so K."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trajconcept.config import quick
from trajconcept.data import make_synthetic
from trajconcept.knowledge import compute_K, ALL_NAMES
from trajconcept import explain as ex
from trajconcept.twins import TwinGenerator
from trajconcept.geometry import encode


def test_perturbation_operators_eq2_to_eq4():
    cfg = quick(); rng = np.random.RandomState(0)
    X = np.cumsum(np.random.RandomState(1).randn(20, 2), 0)
    m = np.zeros(20, bool); m[5:9] = True
    # Eq.(3): nhan he so s
    Y = ex.apply_op("scale", X, m, cfg, rng)
    assert np.allclose(Y[m], cfg.scale_factor * X[m]) and np.allclose(Y[~m], X[~m])
    # Eq.(4): xoay quanh GOC bang theta
    Y = ex.apply_op("rotate", X, m, cfg, rng)
    c, s = np.cos(cfg.rot_angle), np.sin(cfg.rot_angle)
    assert np.allclose(Y[m], X[m] @ np.array([[c, -s], [s, c]]).T) and np.allclose(Y[~m], X[~m])
    assert np.allclose(np.linalg.norm(Y[m], axis=1), np.linalg.norm(X[m], axis=1))     # xoay bao toan khoang cach toi goc
    # Eq.(2): nhieu Gauss N(0, sigma^2) cong vao diem trong doan, thong ke dung
    big = np.zeros((4000, 2)); mm = np.ones(4000, bool)
    Y = ex.apply_op("noise", big, mm, cfg, np.random.RandomState(0))
    assert abs(Y.std() - cfg.noise_sigma) < 0.05 and abs(Y.mean()) < 0.05
    # 'straight': cac diem trong doan nam tren duong thang noi hai dau
    Y = ex.apply_op("straight", X, m, cfg, rng)
    u_ = Y[9] - Y[4]; v_ = Y[5:9] - Y[4]; d = u_[0] * v_[:, 1] - u_[1] * v_[:, 0]; assert np.abs(d).max() < 1e-9


def test_twin_keeps_linear_K_exactly_and_projection_reduces_error():
    ts = make_synthetic(300, 32, seed=1)
    X = ts.X.astype(np.float64)
    gen = TwinGenerator("gauss", n_pca=8, seed=0, proj_steps=0).fit(X[:200])
    tw0, e0, _ = gen.twins(X[200:215], B=4, pool=16, seed=1, steps=0)
    gen.proj_steps = 40
    tw1, e1, _ = gen.twins(X[200:215], B=4, pool=16, seed=1)
    K = compute_K(X[200:215])
    lin = [ALL_NAMES.index(n) for n in ("samo_speed_level", "samo_speed_trend", "samo_head_cos", "samo_head_sin", "samo_turn_level", "samo_turn_trend")]
    for tw in (tw0, tw1):
        Kt = compute_K(tw.reshape(-1, 32, 2)).reshape(15, 4, -1)
        err = np.abs(Kt[:, :, lin] - K[:, None, lin]).max()
        assert err < 1e-5, err                      # muc K tuyen tinh theo (nu, omega) phai CHINH XAC (clip nu duoc chieu lai)
    assert np.median(e1) <= np.median(e0) + 1e-9    # chieu khong lam te di
    # twin hop le & khac goc
    assert np.isfinite(tw1).all() and np.abs(tw1 - X[200:215, None]).max() > 1e-3


def test_dose_response_monotone_on_synthetic_latent():
    from trajconcept import concepts as cc
    from trajconcept.config import quick
    from trajconcept.geometry import encode
    rs = np.random.RandomState(0); L = 16
    X = np.cumsum(rs.randn(60, L, 2), 1)
    cfg = quick(); cfg.t_min_frac = 0.0
    def lat(Xa):                                   # dac trung 0 = nu (log toc do) tai tung buoc, lap buoc cuoi cho du L diem
        nu = encode(np.asarray(Xa, np.float64))["nu"]
        Z = np.zeros((len(Xa), L, 3)); Z[:, :-1, 0] = nu; Z[:, -1, 0] = nu[:, -1]
        return Z
    v = np.array([1.0, 0, 0])
    d_nu = np.full(L - 1, 0.2); d_om = np.zeros(L - 2)
    r = cc.dose_response(lat, v, X, d_nu, d_om, cfg)
    assert r["rho"] > 0.99 and r["effect"] > 0, r                      # tang don dieu theo lieu
    r0 = cc.dose_response(lat, v, X, np.zeros(L - 1), d_om, cfg)       # mau hinh rong -> khong doi
    assert abs(r0["effect"]) < 1e-9 and max(r0["k_drift"]) < 1e-9, r0
    print("OK test_dose_response_monotone_on_synthetic_latent")


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("OK", k)
