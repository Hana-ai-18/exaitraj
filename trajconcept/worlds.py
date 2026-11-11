"""Benchmark 'cai dap an' (muc 7.1 nhom 2): giu nguyen quy dao that, THAY NHAN bang nhan sinh tu
  score_c = a * (W K_std)_c + b * beta_c * h*(X) + Gumbel,   y = argmax_c score_c
trong do h* la mot yeu to NAM NGOAI K: tinh tu quy dao roi LOAI phan giai thich duoc boi K (phan du cheo-gap, GB).
 - ho 'freq' : cong suat dai tan cao cua toc do re omega (khong dinh vi duoc)
 - ho 'where': do sau cua doan CHAM LAI sau nhat (co vi tri t* -> dung de do IoU / khoi luong dung)
b = 0 la THE GIOI RONG: nhan chi phu thuoc K => phuong phap phai bao 'khong co gi'.
CHUA CAI DAT (ghi ro trong bai): ho 'dia ly an', the gioi Sigma-Lognormal, nguoi di bo luc xa hoi, loi tat cai san.
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score
from .geometry import encode, u_of, linear_basis
from .knowledge import compute_K


def raw_factor(X, family):
    """Tra ve h_raw (n,) va t_star (n,) (chi so diem, -1 neu khong co vi tri)."""
    e = encode(np.asarray(X, np.float64)); L = X.shape[1]
    if family == "freq":
        om = e["omega"] - e["omega"].mean(1, keepdims=True)
        P = np.abs(np.fft.rfft(om, axis=1)) ** 2
        hi = P[:, int(0.6 * P.shape[1]):].sum(1)
        return np.log(hi + 1e-9), -np.ones(len(X), int)
    if family == "where":
        s = np.exp(e["nu"]); w = 3
        k = np.ones(w) / w
        sm = np.stack([np.convolve(si, k, mode="same") for si in s])
        t = sm[:, w:-w].argmin(1) + w                          # chi so buoc
        depth = (s.mean(1) - sm[np.arange(len(s)), t]) / (s.mean(1) + 1e-9)
        return depth, t + 1                                    # diem sau buoc t
    raise ValueError(family)


def residualize(h, K, seed=0):
    """Loai phan giai thich duoc boi K (GB cheo-gap). Tra ve h* chuan hoa va R2 CV cua h* theo K (bang chung ngoai-K)."""
    from scipy.stats import rankdata, norm
    h = norm.ppf((rankdata(h) - 0.5) / len(h))        # chuan hoa hang -> Gauss TRUOC khi loai K (tranh duoi nang lam yeu tin hieu)
    kf = KFold(5, shuffle=True, random_state=seed)
    gb = lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=100, random_state=seed)
    pred = cross_val_predict(gb(), K, h, cv=kf)
    r2_before = r2_score(h, pred)
    res = h - pred
    res = (res - res.mean()) / (res.std() + 1e-9)
    pred2 = cross_val_predict(HistGradientBoostingRegressor(max_depth=4, max_iter=150, random_state=seed + 1), K, res, cv=kf)
    return res, float(r2_before), float(r2_score(res, pred2))


def make_world(ts, family, b, seed, n_classes=3, a=1.5):
    """Tra ve (y, info). Khong thay doi X. info: h_star, t_star, r2 ngoai-K, ..."""
    rng = np.random.RandomState(seed + 17)
    K = compute_K(ts.X); Ks = (K - K.mean(0)) / (K.std(0) + 1e-9)
    h, t_star = raw_factor(ts.X, family)
    hstar, r2_b, r2_a = residualize(h, Ks, seed)
    W = rng.randn(Ks.shape[1], n_classes) / np.sqrt(Ks.shape[1])
    kscore = Ks @ W; kscore = (kscore - kscore.mean(0)) / (kscore.std(0) + 1e-9)
    beta = np.linspace(-1, 1, n_classes)
    score = a * kscore + b * beta[None] * hstar[:, None] + rng.gumbel(size=(len(Ks), n_classes)) * 0.7
    y = score.argmax(1)
    return y, dict(h_star=hstar, t_star=t_star, r2_h_given_K_before=r2_b, r2_h_given_K_after=r2_a,
                   valid=bool(r2_a < 0.05), family=family, b=b, class_counts=np.bincount(y, minlength=n_classes).tolist())
