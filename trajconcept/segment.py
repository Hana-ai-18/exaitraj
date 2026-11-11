"""Chia doan: RDP, MDL, cua so truot (5% so diem) - 3 cach cua XAI4Traj.
DOI CHIEU voi repo chinh thuc DAIR-Group/XAI4Traj (traj_xai/src/segmentation/segmentation.py):
 - RDP: thu vien `rdp`, epsilon TUYET DOI (don vi toa do tho); o day epsilon TUONG DOI theo duong cheo khung bao.
 - MDL: ban cai dat trong repo KHONG phai TRACLUS day du: chi phi doan = tong khoang cach vuong goc cua cac diem toi
   duong thang dau-cuoi; mo rong doan cho den khi chi phi > epsilon (mac dinh 0.8, tuyet doi). `mdl` o day = dung cong thuc do
   (epsilon tuong doi theo duong cheo khung bao); ban TRACLUS (paper trich dan Lee et al. 2007) con la `mdl_traclus`.
 - Cua so truot: paper (Fig. 3b) = chia deu thanh cac doan do dai bang nhau (phan hoach). Repo dung buoc co dinh step=5 voi cua
   so = 5% do dai nen cac doan CHONG LAP hoac HO (ghep lai khong con la quy dao goc). Ta theo MO TA CUA PAPER (phan hoach).
Doan tra ve dang mat na bool (T, L): doan j gom cac diem (b_j, b_{j+1}] (doan dau gom ca b_0)."""
from __future__ import annotations
import numpy as np


def _rdp(P, eps):
    n = len(P)
    keep = np.zeros(n, bool); keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        d = P[b] - P[a]; nrm = np.hypot(*d)
        seg = P[a + 1:b] - P[a]
        dist = np.hypot(seg[:, 0], seg[:, 1]) if nrm < 1e-12 else np.abs(d[0] * seg[:, 1] - d[1] * seg[:, 0]) / nrm
        k = int(np.argmax(dist))
        if dist[k] > eps:
            m = a + 1 + k; keep[m] = True
            stack += [(a, m), (m, b)]
    return np.where(keep)[0]


def _mdl_traclus(P):
    """Phan doan xap xi theo TRACLUS (Lee, Han, Whang 2007)."""
    n = len(P); cps = [0]; start = 0; length = 1
    lg = lambda v: np.log2(max(v, 1e-9))
    def seg_len(a, b): return float(np.hypot(*(P[b] - P[a])))
    def cost_par(a, b):
        h = lg(seg_len(a, b)); dh = 0.0
        base = P[b] - P[a]; bl = np.hypot(*base)
        for k in range(a, b):
            v = P[k + 1] - P[k]; vl = np.hypot(*v)
            if bl < 1e-12 or vl < 1e-12: continue
            ang = np.arccos(np.clip(np.dot(base, v) / (bl * vl), -1, 1))
            dperp = abs(base[0] * (P[k] - P[a])[1] - base[1] * (P[k] - P[a])[0]) / bl
            dh += lg(dperp) + lg(vl * np.sin(ang))
        return h + dh
    def cost_nopar(a, b):
        return lg(sum(seg_len(k, k + 1) for k in range(a, b)))
    while start + length < n:
        curr = start + length
        if cost_par(start, curr) > cost_nopar(start, curr):
            cps.append(curr - 1); start = curr - 1; length = 1
        else:
            length += 1
    cps.append(n - 1)
    return np.unique(cps)


def _mdl_cost_repo(seg):
    """Chi phi MDL cua repo chinh thuc: tong khoang cach vuong goc toi duong thang dau-cuoi (inf neu <2 diem hoac dau=cuoi)."""
    if len(seg) < 2:
        return float("inf")
    d = seg[-1] - seg[0]; n = np.hypot(*d)
    if n == 0:
        return float("inf")
    u = d / n
    proj = (seg - seg[0]) @ u
    rec = seg[0] + np.outer(proj, u)
    return float(np.linalg.norm(seg - rec, axis=1).sum())


def _mdl_repo(P, eps):
    """Port tu repo: tra ve cac chi so diem ngat (0, ..., L-1)."""
    cps = [0]; start = 0
    for i in range(1, len(P)):
        if _mdl_cost_repo(P[start:i + 1]) > eps:
            cps.append(i - 1); start = i
    cps.append(len(P) - 1)
    return np.unique(cps)


def _window(L, frac):
    w = max(2, int(round(frac * L)))
    b = list(range(0, L - 1, w)) + [L - 1]
    return np.unique(b)


def _bp_to_mask(bp, L):
    T = len(bp) - 1
    M = np.zeros((T, L), bool)
    for j in range(T):
        lo = bp[j] if j == 0 else bp[j] + 1
        M[j, lo:bp[j + 1] + 1] = True
    return M


def segment(P: np.ndarray, method: str, eps: float = 0.05, window_frac: float = 0.05,
            max_T: int = 24, min_T: int = 2) -> np.ndarray:
    """P: (L,2). Tra ve mat na (T, L). eps tuong doi theo duong cheo khung bao (nhu RDP epsilon trong paper)."""
    L = len(P)
    diag = float(np.hypot(*(P.max(0) - P.min(0)))) + 1e-9
    if method == "rdp":
        e = eps * diag
        for _ in range(12):
            bp = _rdp(P, e)
            if len(bp) - 1 <= max_T: break
            e *= 1.5
    elif method == "mdl":
        bp = _mdl_repo(P, eps * diag)
        if len(bp) - 1 > max_T:
            bp = bp[np.linspace(0, len(bp) - 1, max_T + 1).astype(int)]
    elif method == "mdl_traclus":
        bp = _mdl_traclus(P)
        if len(bp) - 1 > max_T:
            bp = bp[np.linspace(0, len(bp) - 1, max_T + 1).astype(int)]
    elif method == "window":
        bp = _window(L, window_frac)
        if len(bp) - 1 > max_T:
            bp = _window(L, 1.0 / max_T)
    else:
        raise ValueError(method)
    if len(bp) - 1 < min_T:                 # duong thang -> khong co gi de tach, chia doi
        bp = np.unique([0, L // 2, L - 1])
    return _bp_to_mask(bp, L)
