"""Kiem thu cac phan thuan numpy: song anh hinh hoc, K, chia doan, cong thuc Fidelity."""
import numpy as np, itertools, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trajconcept.geometry import encode, decode, linear_basis, u_of
from trajconcept.data import make_synthetic
from trajconcept.knowledge import compute_K, ALL_NAMES
from trajconcept.segment import segment
from trajconcept.xai_metrics import precision_X, expected_random_precision, fidelity


def test_roundtrip():
    ts = make_synthetic(200, 32)
    X = ts.X.astype(np.float64)
    e = encode(X)
    Xr = decode(e["p1"], e["beta1"], e["nu"], e["omega"])
    assert np.abs(X - Xr).max() < 1e-4, np.abs(X - Xr).max()


def test_basis_orthonormal_and_K_linear_in_span():
    L = 32
    V = linear_basis(L)
    assert np.allclose(V @ V.T, np.eye(4), atol=1e-9)
    ts = make_synthetic(100, L)
    e = encode(ts.X.astype(np.float64)); u = u_of(e)
    a = u @ V.T
    u2 = u - a @ V + a @ V               # giu a, khong doi
    assert np.allclose(u, u2)
    # mean(nu) va slope(nu) la ham tuyen tinh cua a
    K = compute_K(ts.X)
    nm = ALL_NAMES.index("samo_speed_level"); sl = ALL_NAMES.index("samo_speed_trend")
    assert np.allclose(K[:, nm], a[:, 0] / np.sqrt(L - 1), atol=1e-6)
    t = np.arange(L - 1) - (L - 2) / 2.0
    assert np.allclose(K[:, sl], a[:, 1] / np.linalg.norm(t), atol=1e-6)


def test_precision_eq8_hand_cases():
    # flips o hang 1 va 3 trong T=4: P = (1*1/1 + 1*2/3)/2
    assert abs(precision_X(np.array([1, 0, 1, 0.])) - (1 + 2 / 3) / 2) < 1e-12
    assert precision_X(np.array([1, 1, 0, 0.])) == 1.0          # xep hang hoan hao
    assert precision_X(np.ones(5)) == 1.0                       # moi doan deu doi lop -> P = 1 (paper noi ro)
    assert abs(precision_X(np.array([0, 0, 1.])) - 1 / 3) < 1e-12
    assert np.isnan(precision_X(np.zeros(3)))


def test_expected_random_precision_matches_montecarlo():
    rng = np.random.RandomState(0)
    for T, R in [(5, 1), (6, 3), (10, 4), (8, 7)]:
        base = np.array([1] * R + [0] * (T - R), float)
        mc = np.mean([precision_X(rng.permutation(base)) for _ in range(40000)])
        assert abs(mc - expected_random_precision(T, R)) < 4e-3, (T, R, mc, expected_random_precision(T, R))
    # exact enumeration
    T, R = 5, 2
    vals = [precision_X(np.array(p, float)) for p in set(itertools.permutations([1] * R + [0] * (T - R)))]
    assert abs(np.mean(vals) - expected_random_precision(T, R)) < 1e-12


def test_fidelity_R_P_H():
    fl = [np.array([1, 0, 0], bool), np.array([0, 0, 0], bool), np.array([1, 1, 1], bool), np.array([0, 1, 0, 1], bool)]
    th = [np.array([3., 1, 0]), np.array([1., 2, 3]), np.array([.1, .2, .3]), np.array([9., 8, 7, 6])]
    r = fidelity(fl, th)
    assert abs(r["R"] - 3 / 4) < 1e-12                            # 3/4 quy dao giai thich duoc
    # P_X: traj0 = 1, traj2 = 1 (R=T), traj3: ranking flips at idx [1,3] -> order [0,1,2,3] -> f=[0,1,0,1]
    p3 = precision_X(np.array([0, 1, 0, 1.]))
    assert abs(r["P"] - (1 + 1 + p3) / 3) < 1e-12
    assert abs(r["H"] - 2 * r["P"] * r["R"] / (r["P"] + r["R"])) < 1e-12
    # quy dao co thong tin: traj0 (R=1<3) va traj3 (R=2<4); traj2 (R=T) khong
    assert abs(r["frac_informative"] - 2 / 4) < 1e-12


def test_segmentation_masks_partition():
    ts = make_synthetic(20, 48)
    for m in ["rdp", "mdl", "window"]:
        for P in ts.X[:10].astype(float):
            M = segment(P, m)
            assert M.shape[1] == 48 and M.shape[0] >= 2
            assert (M.sum(0) == 1).all(), m           # moi diem thuoc dung mot doan




def test_softk_matches_numpy_K():
    import torch
    from trajconcept.softk import soft_K
    ts = make_synthetic(150, 40, seed=3)
    X = ts.X.astype(np.float64)
    e = encode(X); K = compute_K(X)
    t = lambda a: torch.tensor(a, dtype=torch.float64)
    Kh = soft_K(t(e["p1"]), t(e["beta1"]), t(e["nu"]), t(e["omega"]), soft=False).numpy()
    # dac trung 'sinuosity' va std dung ddof khac nhau o numpy (std ddof=0 trong ca hai) -> phai khop
    bad = [(n, float(np.abs(K[:, i] - Kh[:, i]).max())) for i, n in enumerate(ALL_NAMES) if not np.allclose(K[:, i], Kh[:, i], atol=1e-4, rtol=1e-4)]
    assert not bad, bad




def test_local_metrics_hand_cases():
    from trajconcept.local_metrics import relevant_mass_ratio, iou_topk, leak_overlap
    # 4 doan khong chong lap tren 8 diem; t*=5 thuoc doan 2 (diem 4-5)
    M = np.zeros((4, 8), bool)
    for j in range(4): M[j, 2 * j:2 * j + 2] = True
    phi_all = np.array([0, 0, 1, 0.0])            # toan bo khoi luong o doan lien quan
    assert abs(relevant_mass_ratio(phi_all, M, 5) - 4.0) < 1e-12           # 1 / (1/4)
    assert abs(relevant_mass_ratio(np.ones(4), M, 5) - 1.0) < 1e-12        # deu = ngau nhien
    assert relevant_mass_ratio(np.zeros(4), M, 5) is None
    assert iou_topk(phi_all, M, 5) == 1.0
    assert iou_topk(np.array([1.0, 0, 0, 0]), M, 5) == 0.0
    assert iou_topk(np.array([0.5, 0.4, 0.3, 0.1]), M, 5) == 0.0
    # leak_overlap: phi trung hoan toan kappa tap trung -> T ; deu -> 1
    assert abs(leak_overlap(np.ones(4), np.ones(4)) - 1.0) < 1e-12
    assert abs(leak_overlap(np.array([1.0, 0, 0, 0]), np.array([1.0, 0, 0, 0])) - 4.0) < 1e-12
    assert abs(leak_overlap(np.array([1.0, 0, 0, 0]), np.array([0, 1.0, 0, 0]))) < 1e-12


def test_paired_signflip_p_exact():
    from trajconcept.concepts import paired_signflip_p
    assert abs(paired_signflip_p([1, 1, 1], [0, 0, 0]) - 1 / 8) < 1e-12        # p nho nhat = 1/2^n
    assert paired_signflip_p([0, 0, 0], [0, 0, 0]) == 1.0
    assert paired_signflip_p([-1, -1, -1], [0, 0, 0]) == 1.0
    p = paired_signflip_p([0.5, 0.3, 0.8, 0.1], [0.0, 0.0, 0.0, 0.0])           # 4 so duong: 1/16
    assert abs(p - 1 / 16) < 1e-12
    rs = np.random.RandomState(0); a = rs.randn(40) + 1.0
    assert paired_signflip_p(a, np.zeros(40)) < 0.01                             # n > 12: Monte Carlo, hieu ung ro


def test_boot_ci_covers_mean():
    from trajconcept.pipeline import boot_ci
    d = np.random.RandomState(1).randn(200) * 2 + 3
    m, lo, hi = boot_ci(d, B=500)
    assert lo < 3 < hi and abs(m - d.mean()) < 1e-12 and hi - lo < 1.0


def test_deletion_insertion_ranks_true_segment_better():
    from trajconcept.config import quick
    from trajconcept import explain as ex
    L = 25; t = np.arange(L)
    y = np.zeros(L); y[13:18] = 1.0 * np.sin((t[13:18] - 13) * np.pi / 4); y[1:6] = 3.0 * np.sin((t[1:6] - 1) * np.pi / 4)   # A (lien quan, nho) o 13..17 ; B (khong lien quan, lon) o 1..5
    X = np.stack([t.astype(float), y], 1); cfg = quick()
    masks = np.zeros((4, L), bool)
    for j in range(4): masks[j, 6 * j: 6 * j + 7 if j < 3 else L] = True
    def fpred(Xs):                                           # chi phu thuoc do lech cua vung A
        dev = np.abs(Xs[:, 13:18, 1]).mean(1); p = 1 / (1 + np.exp(-(dev - 0.3) * 6))
        return np.stack([p, 1 - p], 1)
    good = np.array([0.0, 0.0, 1.0, 0.0]); bad = np.array([1.0, 0.0, 0.0, 0.0])
    dg, ig = ex.deletion_insertion(X, masks, good, fpred, 0, cfg); db, ib = ex.deletion_insertion(X, masks, bad, fpred, 0, cfg)
    assert dg < db - 0.05 and ig > ib + 0.05, (dg, db, ig, ib)       # xep dung: xoa -> P giam som, chen -> P tang som (tren truc ngan sach ADE)


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("OK", k)
