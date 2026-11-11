"""The tri thuc K (v1.0-draft) tinh tu quy dao. Xem The_tri_thuc_K_v1.md.

TANG (tich luy): K0 hinh hoc + thong ke buoc | K1 + sinh thai van dong (Benhamou, Zheng) | K2 + SAMO-5 + dong hoc (PaHaW).
LUU Y TRUNG THUC: (1) day la mot BAN RUT GON DUNG CHUNG cho moi bo (de mot) - bang theo-bo trong de xuat
chua duoc cai dat; (2) cac nguong HCR/SR/VCR lay tu Zheng et al. nhung ap vao toc do chuan hoa (buoc trung vi = 1),
nen SR dung nguong tuong doi 0.25 * buoc trung vi; (3) cong thuc sinuosity (Benhamou 2004) ghi tu tri nho: [CAN KIEM TRA NGUON].
"""
from __future__ import annotations
import numpy as np
from .geometry import encode

TIER_NAMES = ["K0_geometry_steps", "K1_ecology_zheng", "K2_samo_kinematics"]

K0 = ["mean_speed", "std_speed", "path_len", "net_disp", "bbox_w", "bbox_h", "disp_cos", "disp_sin",
      "mean_abs_turn", "std_turn", "top3_speed", "top3_acc"]
K1 = ["straightness", "sinuosity", "HCR", "SR", "VCR", "turn_ac1", "speed_ac1"]
K2 = ["samo_speed_level", "samo_speed_trend", "samo_head_cos", "samo_head_sin", "samo_turn_level",
      "samo_turn_trend", "NCV", "NCA", "jerk_rms"]
ALL_NAMES = K0 + K1 + K2
TIER_COLS = [list(range(0, len(K0))),
             list(range(0, len(K0) + len(K1))),
             list(range(0, len(ALL_NAMES)))]


def tier_cols(r: int):
    return TIER_COLS[r]


def _slope(v):
    n = v.shape[1]
    t = np.arange(n) - (n - 1) / 2.0
    return (v * t).sum(1) / (t ** 2).sum()


def _ac1(v):
    a = v[:, :-1] - v[:, :-1].mean(1, keepdims=True)
    b = v[:, 1:] - v[:, 1:].mean(1, keepdims=True)
    return (a * b).sum(1) / (np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1)) + 1e-9)


def compute_K(X: np.ndarray) -> np.ndarray:
    """X (n, L, 2) -> (n, 28). Thuan numpy, khong phu thuoc mo hinh."""
    X = np.asarray(X, np.float64)
    e = encode(X)
    nu, om, b1 = e["nu"], e["omega"], e["beta1"]
    s = np.exp(nu)
    d = np.diff(X, axis=1)
    path = s.sum(1)
    net_v = X[:, -1] - X[:, 0]
    net = np.hypot(net_v[:, 0], net_v[:, 1])
    ext = X.max(1) - X.min(1)
    srt = np.sort(s, axis=1)
    acc = np.diff(s, axis=1)
    top3_acc = np.sort(np.abs(acc), axis=1)[:, -3:].mean(1)
    c = np.cos(om).mean(1)
    p = s.mean(1); bcv = s.std(1) / (p + 1e-9)
    sin_denom = (1 + c) / (1 - c + 1e-6) + bcv ** 2
    sinuosity = 2.0 * (p * sin_denom) ** -0.5                      # Benhamou 2004 [CAN KIEM TRA]
    hcr = (np.abs(om) > np.deg2rad(19)).sum(1) / (path + 1e-9)     # Zheng 2008
    sr = (s < 0.25).sum(1) / (path + 1e-9)
    vcr = ((np.abs(acc) / (s[:, :-1] + 1e-9)) > 0.26).sum(1) / (path + 1e-9)
    ds_ = np.sign(acc)
    ncv = (ds_[:, 1:] * ds_[:, :-1] < 0).sum(1)
    acc2 = np.diff(acc, axis=1)
    ds2 = np.sign(acc2)
    nca = (ds2[:, 1:] * ds2[:, :-1] < 0).sum(1)
    cols = [
        s.mean(1), s.std(1), path, net, ext[:, 0], ext[:, 1],
        net_v[:, 0] / (net + 1e-9), net_v[:, 1] / (net + 1e-9),
        np.abs(om).mean(1), om.std(1), srt[:, -3:].mean(1), top3_acc,
        net / (path + 1e-9), sinuosity, hcr, sr, vcr, _ac1(om), _ac1(nu),
        nu.mean(1), _slope(nu), np.cos(b1), np.sin(b1), om.mean(1), _slope(om),
        ncv.astype(float), nca.astype(float), np.sqrt((acc2 ** 2).mean(1)),
    ]
    K = np.stack(cols, 1)
    assert K.shape[1] == len(ALL_NAMES), (K.shape, len(ALL_NAMES))
    return np.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
