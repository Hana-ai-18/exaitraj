"""Thuoc do Fidelity cua XAI4Traj (Le, Bui, Vo 2025, Eq. 7-10) va ban hieu chinh (P_info, P_rand).

GHI CHU DOI CHIEU VOI PAPER
 - Eq.(7) Recall: ban trich van ban cua PDF bi vo cong thuc (hai dau tong long nhau); theo VAN XUOI cua paper
   ("proportion of data points that can be explained... any of the perturbations change the result"),
   ta cai dat Recall = ti le quy dao co IT NHAT MOT doan lam doi lop. [XAC NHAN LAI voi ban PDF goc truoc khi nop]
 - Eq.(8): P_X = 1/min(T,R) * sum_{i=1..T} 1{flip tai hang i} * r_i / i, r_i = so doan gay doi lop trong i hang dau.
 - Eq.(9): P = trung binh P_X tren cac quy dao giai thich duoc (R >= 1).
 - Eq.(10): H = 2PR/(P+R) (khong xac dinh khi R = 0 -> NaN).
"""
from __future__ import annotations
import numpy as np


def precision_X(flips_ranked: np.ndarray) -> float:
    """flips_ranked: (T,) 0/1 theo thu tu giam dan tam quan trong. Eq.(8). Yeu cau co >= 1 flip."""
    f = np.asarray(flips_ranked, dtype=float)
    T = len(f); R = f.sum()
    if R < 1:
        return float("nan")
    r = np.cumsum(f)
    i = np.arange(1, T + 1)
    return float((f * r / i).sum() / min(T, R))


def expected_random_precision(T: int, R: int) -> float:
    """E[P_X] khi thu tu la hoan vi ngau nhien (cong thuc dong, da kiem bang Monte Carlo trong tests)."""
    if R < 1:
        return float("nan")
    if T == 1:
        return 1.0
    i = np.arange(1, T + 1)
    # E[f_i r_i / i] = (R/T)(1+(i-1)(R-1)/(T-1))/i ; chia cho min(T,R)=R  =>
    return float((1.0 / T) * np.sum((1.0 + (i - 1) * (R - 1) / (T - 1)) / i))


def fidelity(flips_list, theta_list):
    """flips_list[k]: (T_k,) bool, doan j co lam doi lop khi nhieu rieng no khong.
    theta_list[k]: (T_k,) diem quan trong (cang lon cang quan trong).
    Tra ve dict R, P, H (dung paper) va P_info, P_rand, P_info_minus_rand, frac_informative."""
    n = len(flips_list)
    explainable, P_all, P_info, P_info_rand = [], [], [], []
    for fl, th in zip(flips_list, theta_list):
        fl = np.asarray(fl, bool)
        T, R = len(fl), int(fl.sum())
        if th is None:                       # surrogate chi thay mot lop (repo: change = 0) -> khong giai thich duoc
            explainable.append(False)
            continue
        th = np.asarray(th, float)
        if R >= 1:
            explainable.append(True)
            order = np.argsort(-th, kind="stable")
            p = precision_X(fl[order].astype(float))
            P_all.append(p)
            if R <= T - 1:                                  # quy dao 'co thong tin'
                P_info.append(p); P_info_rand.append(expected_random_precision(T, R))
        else:
            explainable.append(False)
    Rec = float(np.mean(explainable)) if n else float("nan")
    P = float(np.mean(P_all)) if P_all else float("nan")
    H = float(2 * P * Rec / (P + Rec)) if (P_all and (P + Rec) > 0) else float("nan")
    Pi = float(np.mean(P_info)) if P_info else float("nan")
    Pr = float(np.mean(P_info_rand)) if P_info_rand else float("nan")
    return dict(R=Rec, P=P, H=H, P_info=Pi, P_rand=Pr,
                P_info_minus_rand=(Pi - Pr) if P_info else float("nan"),
                frac_informative=len(P_info) / max(n, 1), n=n)
