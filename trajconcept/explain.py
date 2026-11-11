"""Ban do doan cua TrajConcept (phi^MU) va cac phep nhieu/thuoc do dung de DANH GIA no theo giao thuc Fidelity cua XAI4Traj
(Le, Bui, Vo, Neurocomputing 658, 131691). KHONG con phuong phap giai thich nao khac (XAI4Traj, LIME, SHAP, occlusion, IG da bo).

Giu lai tu XAI4Traj chi la THUOC DO + giao thuc danh gia:
 - Chia doan: RDP / MDL / cua so truot 5% (segment.py, da doi chieu voi repo chinh thuc DAIR-Group/XAI4Traj).
 - Phep nhieu dung de THU lat nhan khi nhieu RIENG tung doan (Eq.2-4): gaussian / rotation / scaling.
     profile 'advisor'   : mac dinh REPO (noise std 3 x scale 1.5 = 4.5; s = 1.2; theta = pi/18) trong TOA DO THO, xoay/co gian quanh goc (0,0) tho
                           (can TrajSet.origin); quy ve khong gian dau vao cua mo hinh.
     profile 'calibrated': khong gian chuan hoa (buoc trung vi = 1): sigma 1, s 1.5, theta pi/4, quanh diem dau.
 - Fidelity R/P/H (Eq.7-10) trong xai_metrics.py. Precision Eq.(8) = ap_at_k cua repo khi dung dung (repo co loi: relevant_class la list => luon 1.0).
 Paper KHONG neu sigma, s, theta, L: gia tri 'advisor' la mac dinh trong repo, chua chac la gia tri da in Bang 2-8.
"""
from __future__ import annotations
import numpy as np
from .segment import segment
from .xai_metrics import fidelity

EPS = 1e-6


# ------------------------------------------------------------------ toan tu nhieu
def apply_op(op, X, mask, cfg, rng, twin=None, gen=None):
    """X (L,2), mask (L,) bool -> X' (L,2)."""
    Y = X.copy()
    if not mask.any():
        return Y
    fr = getattr(cfg, "frame", None)
    if fr is not None and op in ("noise", "scale", "rotate"):
        # Toa do THO (profile 'advisor'): p_tho = X*scale + origin; phep nhieu theo Eq.2-4 voi tham so tho; quy ve lai (X' khong dich lai ve goc)
        sc, o = fr[0], np.array(fr[1:3])
        P = Y[mask].astype(np.float64) * sc + o
        if op == "noise":
            P = P + rng.randn(int(mask.sum()), 2) * cfg.noise_sigma
        elif op == "scale":
            P = cfg.scale_factor * P
        else:
            c, s_ = np.cos(cfg.rot_angle), np.sin(cfg.rot_angle)
            P = P @ np.array([[c, -s_], [s_, c]]).T
        Y[mask] = ((P - o) / sc).astype(Y.dtype)
        return Y
    if op == "noise":
        Y[mask] += rng.randn(int(mask.sum()), 2) * cfg.noise_sigma
    elif op == "scale":
        Y[mask] = cfg.scale_factor * Y[mask]
    elif op == "rotate":
        c, s = np.cos(cfg.rot_angle), np.sin(cfg.rot_angle)
        R = np.array([[c, -s], [s, c]])
        Y[mask] = Y[mask] @ R.T
    elif op == "straight":
        idx = np.where(mask)[0]; L = len(X)
        # tach cac khoang lien tiep
        brk = np.where(np.diff(idx) > 1)[0]
        starts = np.concatenate([[0], brk + 1]); ends = np.concatenate([brk, [len(idx) - 1]])
        for s_, e_ in zip(starts, ends):
            a, b = idx[s_], idx[e_]
            left = max(a - 1, 0); right = min(b + 1, L - 1)
            if left == a:       # khoang bat dau o diem 0: giu diem 0
                left = 0
            if a == 0:
                pts = np.arange(1, b + 1) if b >= 1 else np.array([], int)
            else:
                pts = np.arange(a, b + 1)
            if len(pts) == 0:
                continue
            if b == L - 1:
                pts = pts[pts < L - 1]; right = L - 1
            if len(pts) == 0:
                continue
            w = (pts - left) / max(right - left, 1)
            Y[pts] = X[left] + w[:, None] * (X[right] - X[left])
    elif op == "twin":
        Y = gen.redraw_segment(X, twin, mask)
    else:
        raise ValueError(op)
    return Y



def explain_phi_mu(X, masks, fpred, cstar, gen, twins, cfg, normalize=False):
    """phi_j = E_{twin} | g(X) - g(X with residual redrawn in segment j) |, g = log-odds lop cstar. Khong am.
    normalize=True: chia cho ADE trung binh cua phep redraw (do nhay tren MOT DON VI thay doi hinh hoc) - khu hieu ung
    'doan dai/thay doi nhieu thi logit doi nhieu' bat duoc o kiem tra Adebayo (E8)."""
    T = len(masks); B = len(twins)
    def lo(p):
        p = np.clip(p[:, cstar], 1e-6, 1 - 1e-6); return np.log(p) - np.log(1 - p)
    g0 = lo(fpred(X[None].astype(np.float32)))[0]
    Xs = np.empty((T * B, *X.shape), np.float32)
    for j in range(T):
        for b in range(B):
            Xs[j * B + b] = gen.redraw_segment(X.astype(np.float64), twins[b].astype(np.float64), masks[j])
    g = lo(fpred(Xs)).reshape(T, B)
    phi = np.abs(g - g0).mean(1)
    if normalize:
        ade = np.linalg.norm(Xs - X[None], axis=-1).mean(1).reshape(T, B).mean(1)
        phi = phi / (ade + 1e-3)
    return phi


def phi_mu_dense(X, fpred, cstar, gen, twins, cfg, w=3, normalize=False):
    """phi^MU o DO PHAN GIAI TUNG DIEM, khong phu thuoc cach chia doan: cua so truot w diem (buoc 1), diem i nhan trung binh diem
    cua cac cua so chua no. Dung de ve (F5): cho thay ban do khong phai la 'khoi mau theo doan'."""
    L = len(X)
    masks = []
    for a in range(0, L - w + 1):
        m = np.zeros(L, bool); m[a:a + w] = True; masks.append(m)
    ph = explain_phi_mu(X, masks, fpred, cstar, gen, twins, cfg, normalize=normalize)
    num = np.zeros(L); den = np.zeros(L)
    for a, v in enumerate(ph):
        num[a:a + w] += v; den[a:a + w] += 1
    return num / np.maximum(den, 1)


# ------------------------------------------------------------------ danh gia: lat nhan khi nhieu RIENG tung doan
def flips_native(X, masks, fpred, c0, op, cfg, rng):
    """Dung nhu paper: moi doan nhieu RIENG bang chinh toan tu cua phuong phap, MOT mau/doan."""
    Zs = np.stack([apply_op(op, X, m, cfg, rng) for m in masks]).astype(np.float32)
    return fpred(Zs).argmax(1) != c0


def flips_common(X, masks, fpred, c0, op, cfg, rng, m=8, gen=None, twins=None):
    """Toan tu CHUNG cho moi phuong phap: lat nhan neu >= 1/2 trong m lan nhieu (noise) hoac m twin (twin)."""
    T = len(masks); out = np.zeros(T, bool)
    for j in range(T):
        if op == "twin":
            Zs = np.stack([apply_op("twin", X, masks[j], cfg, rng, twin=twins[b % len(twins)], gen=gen) for b in range(m)])
        else:
            Zs = np.stack([apply_op(op, X, masks[j], cfg, rng) for _ in range(m)])
        out[j] = (fpred(Zs.astype(np.float32)).argmax(1) != c0).mean() >= 0.5
    return out


def deletion_insertion(X, masks, theta, fpred, cstar, cfg, rng=None):
    """Duong cong xoa/chen theo NGAN SACH ADE (sai so vi tri trung binh giua quy dao sua va goc), 'straight' lam toan tu xoa.
    Tra ve (auc_del, auc_ins): x = ADE/ADE_full (xoa) hoac 1 - ADE/ADE_full (chen), y = P(lop goc). Xoa: THAP tot. Chen: CAO tot."""
    T, L = len(masks), len(X)
    order = np.argsort(-np.asarray(theta), kind="stable")
    allm = np.ones(L, bool); allm[0] = allm[-1] = False
    Xfull = apply_op("straight", X, allm, cfg, np.random.RandomState(0))
    ade_full = np.linalg.norm(Xfull - X, axis=1).mean() + 1e-9
    dels, adel, ins, ains = [], [], [], []
    union = np.zeros(L, bool)
    Xd, Xi = [X], [Xfull]
    for j in order:
        union |= masks[j]
        Xd.append(apply_op("straight", X, union, cfg, np.random.RandomState(0)))
        Xk = Xfull.copy(); Xk[union] = X[union]; Xi.append(Xk)
    Pd = fpred(np.stack(Xd).astype(np.float32))[:, cstar]; Pi = fpred(np.stack(Xi).astype(np.float32))[:, cstar]
    xd = np.array([np.linalg.norm(x - X, axis=1).mean() for x in Xd]) / ade_full
    xi = 1 - np.array([np.linalg.norm(x - X, axis=1).mean() for x in Xi]) / ade_full
    gd = np.linspace(0, 1, 21)
    cd = np.interp(gd, np.maximum.accumulate(np.minimum(xd, 1.0)), Pd)
    ci = np.interp(gd, np.maximum.accumulate(np.clip(xi, 0, 1)), Pi)
    _trap = getattr(np, "trapezoid", None) or np.trapz          # numpy < 2 khong co trapezoid
    return float(_trap(cd, gd)), float(_trap(ci, gd))


# ------------------------------------------------------------------ tong hop
def seg_eps(cfg, seg):
    return cfg.mdl_eps if seg == "mdl" else cfg.rdp_eps


def segmentations(X, cfg, which=("rdp", "mdl", "window")):
    return {w: segment(X.astype(np.float64), w, seg_eps(cfg, w), cfg.window_frac) for w in which}



def run_explainers(model, gen, twins_loc, Xs, cfg, seg="rdp", methods=None, seed=0, cfg_fn=None):
    """phi^MU (va ban chuan hoa) cho moi quy dao voi cach chia doan `seg`. Tra ve (dict[method] = list theta, masks list, c0)."""
    methods = methods or ["trajconcept", "trajconcept_n"]
    fpred = model.proba
    P0 = fpred(Xs.astype(np.float32)); c0 = P0.argmax(1)
    allmask = [segment(X.astype(np.float64), seg, seg_eps(cfg, seg), cfg.window_frac) for X in Xs]
    res = {m: [] for m in methods}
    cfg0 = cfg
    for i, X in enumerate(Xs):
        cfg = cfg_fn(i) if cfg_fn is not None else cfg0
        M = allmask[i]; masks = [M[j] for j in range(len(M))]
        for m in methods:
            res[m].append(explain_phi_mu(X, masks, fpred, c0[i], gen, twins_loc[i], cfg, normalize=(m == "trajconcept_n")))
    return res, allmask, c0


OPNAME = {"noise": "gauss", "scale": "scal", "rotate": "rot"}      # ten cot: gaussian / scaling / rotation nhu bang cua paper


def evaluate_fidelity(model, gen, twins_loc, Xs, masks_list, c0, theta_by_method, cfg, seed=0, common=True, auc=True, cfg_fn=None):
    """Voi moi phuong phap (phi^MU, phi^MU chuan hoa) va moi phep nhieu p in {noise, scale, rotate}: R/P/H dung dinh nghia Eq.7-10
    + P_info, P_rand (muc ngau nhien, giai tich) + frac_informative -> cot f"{gauss|scal|rot}_{R,P,H,...}".
    common=True them R/P/H duoi toan tu twin (lat nhan neu >= 1/2 trong m=8 twin); auc=True them AUC xoa/chen theo ADE."""
    fpred = model.proba; rng = np.random.RandomState(seed + 99)
    fl = {"noise": [], "scale": [], "rotate": [], "common_twin": []}
    for i, X in enumerate(Xs):
        masks = [masks_list[i][j] for j in range(len(masks_list[i]))]
        cfg_i = cfg_fn(i) if cfg_fn is not None else cfg
        for op in ("noise", "scale", "rotate"):
            fl[op].append(flips_native(X, masks, fpred, c0[i], op, cfg_i, rng))
        if common:
            fl["common_twin"].append(flips_common(X, masks, fpred, c0[i], "twin", cfg, rng, gen=gen, twins=twins_loc[i]))
    rows = {}
    for m, thetas in theta_by_method.items():
        r = {}
        keys = [(OPNAME[o], fl[o]) for o in ("noise", "scale", "rotate")] + ([("ctwin", fl["common_twin"])] if common else [])
        for key, f_ in keys:
            for k, v in fidelity(f_, thetas).items():
                r[f"{key}_{k}"] = v
        if auc:
            ad = [deletion_insertion(Xs[i], [masks_list[i][j] for j in range(len(masks_list[i]))], thetas[i], fpred, c0[i], cfg) for i in range(len(Xs))]
            r["auc_del_ade"] = float(np.mean([a for a, b in ad])); r["auc_ins_ade"] = float(np.mean([b for a, b in ad]))
        rows[m] = r
    return rows, fl
