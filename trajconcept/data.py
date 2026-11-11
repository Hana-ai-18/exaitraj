"""Nap 7 bo du lieu cua XAI4Traj qua pactus, lay mau lai ve do dai co dinh L, chuan hoa.

Chuan hoa: dich diem dau ve goc, chia cho do dai buoc trung vi cua CA BO (mot so vo huong,
giu thong tin do lon). Diem dau tho duoc luu trong TrajSet.origin: profile 'advisor' cua XAI4Traj (Eq. 2-4, repo) dung phep nhieu
trong TOA DO THO (xoay/co gian quanh goc (0,0) THAT, nhieu theo don vi tho) roi quy ve khong gian dau vao cua mo hinh.
"""
from __future__ import annotations
import os
import numpy as np
from dataclasses import dataclass
from typing import Optional
from .config import RESAMPLE_L


@dataclass
class TrajSet:
    name: str
    X: np.ndarray            # (n, L, 2) float32 da chuan hoa
    y: np.ndarray            # (n,) int
    meta: np.ndarray         # (n, 4): do dai goc, thoi luong, dt trung binh, dt std (cho bo loc 'loi tat')
    groups: Optional[np.ndarray]
    classes: list
    scale: float
    origin: Optional[np.ndarray] = None   # (n, 2) diem dau THO (truoc khi dich ve goc) - de dung phep nhieu cua XAI4Traj trong toa do tho

    @property
    def n_classes(self):
        return len(self.classes)


def _resample(r: np.ndarray, t: np.ndarray, L: int):
    r = np.asarray(r, dtype=np.float64)[:, :2]
    t = np.asarray(t, dtype=np.float64)
    if len(t) != len(r) or len(r) < 2 or not np.all(np.diff(t) > 0):
        t = np.arange(len(r), dtype=np.float64)
    tt = (t - t[0]) / max(t[-1] - t[0], 1e-12)
    g = np.linspace(0, 1, L)
    return np.stack([np.interp(g, tt, r[:, 0]), np.interp(g, tt, r[:, 1])], 1), t


REPO = "yupidevs/trajectory-datasets"


def _download_json(name: str, data_dir: str) -> str:
    """Tai bo du lieu cua pactus (GitHub release cua yupidevs/trajectory-datasets) KHONG qua pactus/TensorFlow
    (import pactus keo theo TensorFlow, va TF + PyTorch cung tien trinh co the xung dot LLVM)."""
    import requests, zipfile, io as _io
    os.makedirs(data_dir, exist_ok=True)
    out = os.path.join(data_dir, f"{name}.json")
    if os.path.exists(out):
        return out
    # 1) cache san co cua pactus (neu da tung chay pactus tren may nay)
    import site
    for base in site.getsitepackages() + [site.getusersitepackages()]:
        cand = os.path.join(base, "pactus", ".cache", "datasets", name, f"{name}.json")
        if os.path.exists(cand):
            return cand
    url = None
    # 2) giong pactus: liet ke tag bang `git ls-remote` roi do file release (github.com, khong qua api.github.com)
    import subprocess
    try:
        outp = subprocess.run(["git", "ls-remote", "--tags", f"https://github.com/{REPO}"], capture_output=True, text=True, timeout=60).stdout
        tags = [ln.split("refs/tags/")[-1].strip() for ln in outp.splitlines() if "refs/tags/" in ln and not ln.endswith("^{}")]
        tags = sorted(tags, reverse=True)
    except Exception:
        tags = []
    for tag in tags:
        cand = f"https://github.com/{REPO}/releases/download/{tag}/{name}.zip"
        try:
            if requests.head(cand, timeout=30).status_code in (200, 302):
                url = cand; break
        except Exception:
            continue
    # 3) du phong: GitHub API
    if url is None:
        try:
            r = requests.get(f"https://api.github.com/repos/{REPO}/releases?per_page=30", timeout=60)
            r.raise_for_status()
            for rel in r.json():
                for a_ in rel.get("assets", []):
                    if a_["name"] == f"{name}.zip":
                        url = a_["browser_download_url"]; break
                if url: break
        except Exception:
            pass
    if url is None:
        raise RuntimeError(f"Khong tai duoc {name}.zip tu {REPO}. Tai thu cong (xem README) va dat {name}.json vao thu muc {data_dir}")
    z = zipfile.ZipFile(_io.BytesIO(requests.get(url, timeout=600).content))
    member = [m for m in z.namelist() if m.endswith(".json")][0]
    with open(out, "wb") as fh:
        fh.write(z.read(member))
    return out


def _load_pactus(name: str, data_dir: str = None):
    """Doc JSON cua pactus -> (list (r, t), labels) bang numpy thuan."""
    import json as _json
    data_dir = data_dir or os.environ.get("TRAJ_DATA_DIR", "data")
    path = _download_json(name, data_dir)
    with open(path) as fh:
        d = _json.load(fh)
    trajs = []
    for t in d["trajs"]:
        ax = np.asarray(t["axes"], dtype=np.float64)           # (n_axes, n)
        n = ax.shape[1]
        tt = np.asarray(t["t"], dtype=np.float64) if t.get("t") is not None else t.get("t_0", 0.0) + (t.get("dt") or 1.0) * np.arange(n)
        trajs.append((ax.T[:, :2], tt))
    return trajs, list(d["labels"])


def load_dataset(name: str, max_n: int = 3000, max_len_raw: int = 2000, seed: int = 0,
                 L: Optional[int] = None) -> TrajSet:
    if name.startswith("synthetic"):
        return make_synthetic(n=max_n, L=L or 32, seed=seed)
    L = L or RESAMPLE_L[name]
    trajs, labels = _load_pactus(name)
    classes = sorted(set(labels), key=lambda z: str(z))
    cidx = {c: i for i, c in enumerate(classes)}
    y_all = np.array([cidx[c] for c in labels])
    rng = np.random.RandomState(seed)
    # lay phan tang neu qua nhieu mau
    idx = np.arange(len(y_all))
    if len(idx) > max_n:
        keep = []
        for c in range(len(classes)):
            ic = idx[y_all == c]
            k = max(2, int(round(max_n * len(ic) / len(idx))))
            keep.append(rng.choice(ic, min(k, len(ic)), replace=False))
        idx = np.sort(np.concatenate(keep))
    Xs, ys, ms = [], [], []
    for i in idx:
        r, t = trajs[int(i)]
        n_orig = len(r)
        if len(r) < 4:
            continue
        if len(r) > max_len_raw:
            sel = np.linspace(0, len(r) - 1, max_len_raw).astype(int)
            r, t = r[sel], t[sel]
        xy, tt = _resample(r, t, L)
        dt = np.diff(tt)
        Xs.append(xy); ys.append(y_all[i])
        ms.append([n_orig, float(tt[-1] - tt[0]), float(dt.mean()), float(dt.std())])
    X = np.stack(Xs); y = np.array(ys); meta = np.array(ms)
    P0 = X[:, 0].copy()                                    # diem dau THO
    X = X - X[:, :1]                                       # diem dau ve goc
    step = np.hypot(*np.moveaxis(np.diff(X, axis=1), -1, 0))
    scale = float(np.median(step[step > 0])) if (step > 0).any() else 1.0
    X = (X / scale).astype(np.float32)
    # bo cac lop qua it mau (< 5) de chia train/val/test duoc
    cnt = np.bincount(y, minlength=len(classes))
    ok = cnt[y] >= 5
    X, y, meta, P0 = X[ok], y[ok], meta[ok], P0[ok]
    remap = {c: i for i, c in enumerate(sorted(set(y.tolist())))}
    y = np.array([remap[c] for c in y])
    classes = [classes[c] for c in sorted(remap, key=lambda z: remap[z])]
    return TrajSet(name, X, y, meta, None, classes, scale, P0.astype(np.float64))


def make_synthetic(n=1200, L=32, n_classes=3, seed=0) -> TrajSet:
    """Du lieu gia lap offline (kiem thu, khong dung de ket luan). Lop = muc toc do + dao dong re."""
    rng = np.random.RandomState(seed)
    y = rng.randint(0, n_classes, n)
    nu = rng.randn(n, 1) * 0.2 + 0.3 * y[:, None] + rng.randn(n, L - 1) * 0.15
    om = rng.randn(n, L - 2) * (0.25 + 0.0 * y[:, None])
    osc = np.sin(np.arange(L - 2)[None] * (1.0 + y[:, None] * 0.5)) * 0.3 * (y[:, None] == 1)
    om = om + osc
    from .geometry import decode
    X = decode(np.zeros((n, 2)), rng.uniform(-np.pi, np.pi, n), nu, om).astype(np.float32)
    meta = np.stack([np.full(n, L), np.full(n, L - 1.0), np.ones(n), np.zeros(n)], 1)
    return TrajSet("synthetic", X, y, meta, None, list(range(n_classes)), 1.0)


def stratified_split(y, fractions=(0.7, 0.15, 0.15), seed=0, groups=None):
    """Chia train/val/test phan tang (theo nhom neu co). Tra ve 3 mang chi so."""
    rng = np.random.RandomState(seed)
    n = len(y)
    if groups is not None:
        ug = np.unique(groups)
        rng.shuffle(ug)
        cuts = np.cumsum([int(len(ug) * f) for f in fractions[:-1]])
        parts = np.split(ug, cuts)
        return [np.where(np.isin(groups, p))[0] for p in parts]
    out = [[] for _ in fractions]
    for c in np.unique(y):
        ic = np.where(y == c)[0]
        rng.shuffle(ic)
        cuts = (np.cumsum(fractions)[:-1] * len(ic)).astype(int)
        for k, part in enumerate(np.split(ic, cuts)):
            out[k].extend(part.tolist())
    return [np.array(sorted(o)) for o in out]
