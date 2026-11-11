"""Pipeline cho mot o (bo du lieu, black box, seed): chia du lieu, huan luyen f va g_K, cong khoang cach tri thuc (E1),
twin + MU-share + CRT (Buoc 2-3), trich khai niem + bo loc (Buoc 4-7). Co cache de chay lai duoc tren Kaggle."""
from __future__ import annotations
import os, pickle, time, hashlib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from .config import Config
from .data import TrajSet, stratified_split
from .knowledge import compute_K, tier_cols, ALL_NAMES
from .geometry import encode, u_of, linear_basis
from .models import (train_blackbox, GK, nll, bacc, ece, DeepClassifier, SklearnRaw, get_device)
from .twins import TwinGenerator
from . import concepts as cc


# ------------------------------------------------------------------ tien ich thong ke
def usable_flag(bb, bacc_, chance, ece_, gap, cfg):
    """Black box dung duoc khi: bacc >= chance + min_margin, ECE < max_ece, va (CHI voi mang no-ron) do lech train-test < max_gap.
    RF / k-NN ghi nho tap train (train acc ~ 1) nen 'gap' cua chung chi la 1 - acc test, KHONG phai dau hieu qua khop: khong dung lam dieu kien."""
    from .config import DEEP
    ok = (bacc_ >= chance + cfg.min_margin) and (ece_ < cfg.max_ece)
    if bb in DEEP:
        ok = ok and (gap < cfg.max_gap)
    return bool(ok)


def boot_ci(d, B=1000, seed=0, alpha=0.05):
    rng = np.random.RandomState(seed)
    d = np.asarray(d, float)
    m = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(B)])
    return float(d.mean()), float(np.percentile(m, 100 * alpha / 2)), float(np.percentile(m, 100 * (1 - alpha / 2)))


def per_sample_nll(p, y):
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1))


def per_sample_correct(p, y):
    return (p.argmax(1) == y).astype(float)


def balanced_correct(p, y):
    """Trong so mau de trung binh = do chinh xac can bang (de bootstrap ghep cap)."""
    w = 1.0 / np.bincount(y, minlength=p.shape[1])[y]
    return per_sample_correct(p, y) * w / w.mean()


# ------------------------------------------------------------------ o thi nghiem
class Cell:
    """Trang thai da huan luyen cua mot (dataset, black box, seed)."""
    def __init__(self, ts: TrajSet, bb: str, seed: int, cfg: Config):
        self.ts, self.bb, self.seed, self.cfg = ts, bb, seed, cfg
        self.n_classes = ts.n_classes
        self.timing = {}

    # --- chia du lieu
    def split(self):
        ts, seed = self.ts, self.seed
        tv, ho = stratified_split(ts.y, (0.7, 0.3), seed, ts.groups)
        tr_i, va_i = stratified_split(ts.y[tv], (0.85, 0.15), seed + 1, None if ts.groups is None else ts.groups[tv])
        self.tr, self.va = tv[tr_i], tv[va_i]
        d_i, c_i, f_i = stratified_split(ts.y[ho], (0.5, 0.25, 0.25), seed + 2, None if ts.groups is None else ts.groups[ho])
        self.ho, self.disc, self.conf, self.fin = ho, ho[d_i], ho[c_i], ho[f_i]

    # --- huan luyen
    def train(self):
        ts, cfg = self.ts, self.cfg
        X, y = ts.X, ts.y
        t0 = time.time()
        self.f = train_blackbox(self.bb, X[self.tr], y[self.tr], X[self.va], y[self.va], self.n_classes, cfg, self.seed)
        self.timing["train_f"] = time.time() - t0
        self.K = compute_K(X)
        self.sc = StandardScaler().fit(self.K[self.tr])
        self.Ks = self.sc.transform(self.K)
        t0 = time.time()
        self.gK = get_gK(self, cfg)
        self.timing["train_gK"] = time.time() - t0
        self.P_ho_f = self.f.proba(X[self.ho])
        self.P_tr_f = self.f.proba(X[self.tr][: min(len(self.tr), 3000)])
        self.P_ho_g = [g.proba(self.K[self.ho][:, tier_cols(r)]) for r, g in enumerate(self.gK)]

    def pf(self, X):
        return self.f.proba(np.asarray(X, np.float32))

    # --- kiem tra black box (muc 7.3)
    def bb_check(self):
        y = self.ts.y; yh = y[self.ho]
        cnt = np.bincount(y[self.tr], minlength=self.n_classes)
        maj = cnt.max() / cnt.sum()
        ba = bacc(self.P_ho_f, yh)
        acc_ho = (self.P_ho_f.argmax(1) == yh).mean()
        ntr = len(self.P_tr_f); acc_tr = (self.P_tr_f.argmax(1) == y[self.tr][:ntr]).mean()
        e = ece(self.P_ho_f, yh)
        chance = 1.0 / self.n_classes                  # do chinh xac CAN BANG ngau nhien = 1/C (khong phai ti le lop da so)
        ok = usable_flag(self.bb, ba, chance, e, acc_tr - acc_ho, self.cfg)
        return dict(bacc=ba, acc=float(acc_ho), nll=nll(self.P_ho_f, yh), ece=e, majority=float(maj),
                    chance=float(chance), gap_train_test=float(acc_tr - acc_ho), usable=bool(ok))

    # --- E1: khoang cach tri thuc theo tang (Delta_r) voi CI ghep cap
    def gate(self, B=None):
        B = B or self.cfg.bootstrap_B
        yh = self.ts.y[self.ho]
        rows = []
        chance_nll = nll(np.tile(np.bincount(self.ts.y[self.tr], minlength=self.n_classes) / len(self.tr), (len(yh), 1)), yh)
        Sf = -per_sample_nll(self.P_ho_f, yh)
        for r, Pg in enumerate(self.P_ho_g):
            Sg = -per_sample_nll(Pg, yh)
            d_nll, lo, hi = boot_ci(Sf - Sg, B, self.seed)
            d_ba, lo_b, hi_b = boot_ci(balanced_correct(self.P_ho_f, yh) - balanced_correct(Pg, yh), B, self.seed)
            S_f, S_g, S_c = -nll(self.P_ho_f, yh), -nll(Pg, yh), -chance_nll
            rows.append(dict(tier=r, S_f_nll=S_f, S_g_nll=S_g, delta_nll=d_nll, delta_nll_lo=lo, delta_nll_hi=hi,
                             bacc_f=bacc(self.P_ho_f, yh), bacc_g=bacc(Pg, yh), delta_bacc=d_ba, delta_bacc_lo=lo_b,
                             delta_bacc_hi=hi_b,
                             known_ratio=(S_g - S_c) / (S_f - S_c) if (S_f - S_c) > 1e-6 else float("nan"),
                             gK_family=self.gK[r].family))
        return rows

    # --- twin
    def build_twin_gen(self, kind=None, k_cols=None, seed_off=0):
        cfg = self.cfg
        g = TwinGenerator(kind or cfg.twin_gen, cfg.pca_dim, cfg.flow_steps, cfg.flow_hidden,
                          str(get_device(cfg.device)) if (kind or cfg.twin_gen) == "flow" else "cpu",
                          seed=self.seed + seed_off, k_cols=k_cols, proj_steps=cfg.twin_proj_steps, newton_iters=cfg.twin_newton_iters)
        t0 = time.time(); g.fit(self.ts.X[self.tr]); self.timing["twin_fit"] = time.time() - t0
        return g

    def make_twins(self, gen, idx, B=None, pool=None):
        B = B or self.cfg.n_twin; pool = pool or self.cfg.twin_pool
        t0 = time.time()
        tw, eps, acc = gen.twins(self.ts.X[idx], B, pool, seed=self.seed + 5, tau=self.cfg.twin_tau)
        self.timing["twins"] = time.time() - t0
        return tw.astype(np.float32), eps, acc


# ------------------------------------------------------------------ g_K dung chung giua cac black box (chi phu thuoc du lieu+seed)
_GK_MEM = {}


def get_gK(cell, cfg):
    key = (cell.ts.name, cell.seed, cfg.gk_budget, cfg.max_n, len(cell.ts.y))
    if key in _GK_MEM:
        return _GK_MEM[key]
    os.makedirs(cfg.cache_dir, exist_ok=True)
    h = hashlib.md5(str((key, cfg.name, cell.ts.y[:200].tolist())).encode()).hexdigest()[:8]
    path = os.path.join(cfg.cache_dir, f"gK__{cell.ts.name}__s{cell.seed}__{h}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            g = pickle.load(fh)
    else:
        y = cell.ts.y; K = cell.K
        g = [GK(cell.n_classes, cfg.gk_budget, cell.seed).fit(K[cell.tr][:, tier_cols(r)], y[cell.tr],
                                                              K[cell.va][:, tier_cols(r)], y[cell.va]) for r in range(3)]
        with open(path, "wb") as fh:
            pickle.dump(g, fh)
    _GK_MEM[key] = g
    return g


# ------------------------------------------------------------------ cache
CACHE_FIELDS = ("deep_rich", "name", "max_n", "max_len_raw", "epochs", "batch_size", "lr", "hidden", "patience", "gk_budget", "min_margin", "max_gap", "max_ece")


def _key(cfg, ds, bb, seed):
    """Khoa cache chi gom tham so anh huong HUAN LUYEN black box/g_K: doi tham so giai thich/E4/twin KHONG bat huan luyen lai."""
    sig = str({k: getattr(cfg, k) for k in CACHE_FIELDS})
    h = hashlib.md5((sig + ds + bb + str(seed)).encode()).hexdigest()[:8]
    return os.path.join(cfg.cache_dir, f"{ds}__{bb}__s{seed}__{h}.pkl")


def get_cell(ts: TrajSet, bb: str, seed: int, cfg: Config, use_cache=True, verbose=True) -> Cell:
    os.makedirs(cfg.cache_dir, exist_ok=True)
    path = _key(cfg, ts.name, bb, seed)
    if use_cache and os.path.exists(path):
        with open(path, "rb") as fh:
            cell = pickle.load(fh)
        if isinstance(cell.f, DeepClassifier):
            cell.f.dev = get_device(cfg.device); cell.f.net.to(cell.f.dev)
        cell.cfg = cfg
        if verbose: print(f"[cache] {ts.name}/{bb}/s{seed}")
        return cell
    cell = Cell(ts, bb, seed, cfg)
    cell.split(); cell.train()
    if isinstance(cell.f, DeepClassifier):
        cell.f.net.cpu()
    with open(path, "wb") as fh:
        pickle.dump(cell, fh)
    if isinstance(cell.f, DeepClassifier):
        cell.f.net.to(cell.f.dev)
    if verbose: print(f"[trained] {ts.name}/{bb}/s{seed} in {cell.timing}")
    return cell


# ------------------------------------------------------------------ Buoc 3: MU-share va CRT
def mu_and_crt(cell: Cell, X, y, twins, alpha=0.05, tier=2):
    """X (n,L,2), y (n,), twins (n,B,L,2). g(X) = log-odds lop f du doan cho X.
    MU = E_X Var_b g(X'_b) / Var_X g(X).
    CRT THO  : T0 = mean log p_f(y|X); T_b = mean log p_f(y|X'_b); p = (1+#{T_b >= T0})/(B+1).
    CRT DOI CHUNG (chinh): thong ke = S(f) - S(g_K) (ky nang cua f tru ky nang cua mo hinh chi dung K). Vi g_K KHONG THE dung
      thong tin ngoai K, moi sut giam cua g_K duoi twin la 'ro ri K' (twin chua dang tri thuc tuyet doi); lay hieu
      de khu phan do. leak_gK = S(g;X) - mean_b S(g;X'_b) bao ro muc ro ri do."""
    n, B = twins.shape[:2]; L = X.shape[1]
    p0 = cell.pf(X); c0 = p0.argmax(1)
    flat = twins.reshape(-1, L, 2)
    pt = cell.pf(flat).reshape(n, B, -1)
    lo = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6)) - np.log(1 - np.clip(p, 1e-6, 1 - 1e-6))
    g0 = lo(p0[np.arange(n), c0]); gt = lo(pt[np.arange(n), :, c0])           # (n,B)
    mu = float(gt.var(1).mean() / (g0.var() + 1e-9))
    ll0 = np.log(np.clip(p0[np.arange(n), y], 1e-9, 1))
    llb = np.log(np.clip(pt[np.arange(n)[:, None], np.arange(B)[None, :], y[:, None]], 1e-9, 1))
    T0, Tb = float(ll0.mean()), llb.mean(0)
    p_raw = float((1 + (Tb >= T0).sum()) / (B + 1))
    cols = tier_cols(tier); g = cell.gK[tier]
    pg0 = g.proba(compute_K(np.asarray(X, np.float64))[:, cols])
    pgt = g.proba(compute_K(flat.astype(np.float64))[:, cols]).reshape(n, B, -1)
    gl0 = np.log(np.clip(pg0[np.arange(n), y], 1e-9, 1))
    glb = np.log(np.clip(pgt[np.arange(n)[:, None], np.arange(B)[None, :], y[:, None]], 1e-9, 1))
    T0c, Tbc = float((ll0 - gl0).mean()), (llb - glb[:, :, ]).mean(0)
    p_ctrl = float((1 + (Tbc >= T0c).sum()) / (B + 1))
    leak = float(gl0.mean() - glb.mean())
    dg = np.abs(gt - g0[:, None]).mean(1)
    mu_x = gt.var(1) / (gt.var(1) + (g0 - gt.mean(1)) ** 2 + 1e-6)
    return dict(MU_share=mu, T0=T0, Tb_mean=float(Tb.mean()), Delta_CRT=float(T0 - Tb.mean()), p_raw=p_raw, p_value=p_ctrl,
                Delta_CRT_ctrl=float(T0c - Tbc.mean()), leak_gK=leak, B=B, reject=bool(p_ctrl < alpha), reject_raw=bool(p_raw < alpha),
                dg=dg, mu_x=mu_x, Tb=Tb)


# ------------------------------------------------------------------ chat luong twin (E2)
def residual_features(X):
    X = np.asarray(X, np.float64); L = X.shape[1]
    e = encode(X); V = linear_basis(L); u = u_of(e); r = u - (u @ V.T) @ V
    nu, om = r[:, : L - 1], r[:, L - 1:]
    def psd_bands(a, nb=6):
        P = np.abs(np.fft.rfft(a, axis=1)) ** 2
        edges = np.linspace(0, P.shape[1], nb + 1).astype(int)
        return np.stack([np.log(P[:, edges[i]:max(edges[i + 1], edges[i] + 1)].mean(1) + 1e-9) for i in range(nb)], 1)
    def ac(a, k):
        return (a[:, :-k] * a[:, k:]).mean(1) / (a.var(1) + 1e-9)
    f = [psd_bands(nu), psd_bands(om)] + [np.stack([ac(nu, k) for k in (1, 2, 3)] + [ac(om, k) for k in (1, 2, 3)], 1)]
    from scipy.stats import kurtosis, skew
    f.append(np.stack([kurtosis(nu, axis=1), skew(nu, axis=1), kurtosis(om, axis=1), skew(om, axis=1)], 1))
    return np.nan_to_num(np.concatenate(f, 1))


def c2st(X_real, X_fake, seed=0):
    """C2ST tren dac trung phan du; real va fake DA GHEP theo cung K (fake la twin cua real). 0.5 = khong phan biet duoc."""
    A, Bf = residual_features(X_real), residual_features(X_fake)
    F = np.concatenate([A, Bf]); y = np.r_[np.zeros(len(A)), np.ones(len(Bf))].astype(int)
    kf = StratifiedKFold(5, shuffle=True, random_state=seed)
    pr = cross_val_predict(HistGradientBoostingClassifier(max_depth=3, max_iter=100, random_state=seed), F, y, cv=kf)
    return float((pr == y).mean())


def twin_leak(cell: Cell, X, y, twins, cfg):
    """Ro ri K cua twin: mo hinh chi dung K (g_K tang 2) mat bao nhieu log-likelihood khi thay X bang twin. Ly tuong = 0."""
    from .pipeline import get_gK
    if not hasattr(cell, "K"):
        cell.K = compute_K(cell.ts.X)
    g = get_gK(cell, cfg)[2]; cols = tier_cols(2); n, B = twins.shape[:2]
    p0 = g.proba(compute_K(np.asarray(X, np.float64))[:, cols])
    pt = g.proba(compute_K(twins.reshape(-1, *X.shape[1:]).astype(np.float64))[:, cols]).reshape(n, B, -1)
    l0 = np.log(np.clip(p0[np.arange(n), y], 1e-9, 1)).mean()
    lb = np.log(np.clip(pt[np.arange(n)[:, None], np.arange(B)[None], y[:, None]], 1e-9, 1)).mean()
    return float(l0 - lb)


def twin_quality(cell: Cell, gen, X, twins, eps, acc):
    """Nhom B: eps_K, C2ST, ti le NN, hop le vat ly, da dang. twins (n,B,L,2)."""
    Xtr = cell.ts.X[cell.tr].astype(np.float64)
    n, B = twins.shape[:2]
    one = twins[:, 0].astype(np.float64)
    nn_ = NearestNeighbors(n_neighbors=1).fit(Xtr.reshape(len(Xtr), -1))
    d_twin = nn_.kneighbors(one.reshape(n, -1))[0][:, 0]
    d_real = nn_.kneighbors(X.reshape(n, -1).astype(np.float64))[0][:, 0]
    flat = twins.reshape(-1, *X.shape[1:]).astype(np.float64)
    lim = np.abs(Xtr).max() * 1.5
    valid = np.isfinite(flat).all((1, 2)) & (np.abs(flat).max((1, 2)) <= lim)
    L = X.shape[1]; V = linear_basis(L)
    u = u_of(encode(X.astype(np.float64))); r = u - (u @ V.T) @ V
    ut = u_of(encode(flat)); rt = (ut - (ut @ V.T) @ V).reshape(n, B, -1)
    div = float((rt.var(1).mean(0) / (r.var(0) + 1e-9)).mean())
    return dict(eps_K_median=float(np.median(eps[:, 0])), eps_K_mean_best=float(eps.mean()),
                eps_Kavg_median=float(np.median(gen.last_eps_mean[:, 0])), accept_rate=float(acc.mean()),
                C2ST=c2st(X, twins[:, 0], cell.seed), nn_ratio_median=float(np.median(d_twin / (d_real + 1e-9))),
                valid_frac=float(valid.mean()), diversity=div)


# ------------------------------------------------------------------ Buoc 4-7: khai niem + bo loc
def run_concepts(cell: Cell, gen, twins_disc, idx_disc, cfg, tier=2, source="both", norm="l1", layer=-1,
                 K_cols=None, verbose=False, n_init=None):
    """Tra ve dict: concepts (list record), funnel (so luong sau tung bo loc), stats LP.
    Chi dung cho DeepClassifier. Khai niem duoc trich tren tap KHAM PHA, loc tren tap XAC NHAN."""
    f: DeepClassifier = cell.f
    n_init = n_init or cfg.teach_inits
    p_thr = max(cfg.alpha, 1.0 / 2 ** n_init) + 1e-12            # p nho nhat cua kiem dinh hoan vi dau la 1/2^n
    ts = cell.ts; X, y = ts.X, ts.y
    cols = K_cols if K_cols is not None else tier_cols(tier)
    Ks = cell.Ks[:, cols] if K_cols is None else cell.Ks[:, cols]
    lat = cc.Latent(f, X[cell.tr], layer)
    out = dict(concepts=[], funnel={}, lp={})
    # ---- Buoc 4: du lieu tuong phan
    groups, st = [], {}
    if source in ("twin", "both"):
        g1, st1 = cc.twin_pairs(f, lat, X[idx_disc], y[idx_disc], twins_disc)
        groups += g1; out["lp"]["twin"] = {k: v for k, v in st1.items() if k != "gap_all"}
    if source in ("disagree", "both"):
        gK_p = cell.gK[tier].proba(cell.K[idx_disc][:, tier_cols(tier)])
        g2, st2 = cc.disagree_pairs(f, gK_p, Ks[idx_disc], X[idx_disc], y[idx_disc], cell.pf(X[idx_disc]))
        groups += g2; out["lp"]["disagree"] = st2
    if not groups:
        out["funnel"] = dict(candidates=0); return out
    # ---- Buoc 5
    V, lpst = cc.extract_candidates(groups, lat, cfg, norm=norm, seed=cell.seed, max_groups=cfg.n_explain * 2)
    out["lp"].update(lpst)
    concepts, sizes = cc.cluster_concepts(V, cfg.cluster_cos_thr)
    out["funnel"]["candidates"] = len(concepts)
    if not concepts:
        return out
    # ---- chuan bi cho bo loc tren tap XAC NHAN
    ic = cell.conf
    Zc = lat(X[ic]); t0 = int(cfg.t_min_frac * Zc.shape[1])
    Zc_traj = Zc.mean(1)
    # Z_a: may dung & g_K sai ; Z_h: g_K dung (tren tap kham pha)
    pf_d = cell.pf(X[idx_disc]).argmax(1); pg_d = cell.gK[tier].predict(cell.K[idx_disc][:, tier_cols(tier)])
    sa = (pf_d == y[idx_disc]) & (pg_d != y[idx_disc]); sh = (pg_d == y[idx_disc])
    Zd = lat(X[idx_disc])[:, t0:, :]
    Za = Zd[sa].reshape(-1, Zd.shape[-1]) if sa.any() else np.zeros((0, Zd.shape[-1]))
    Zh = Zd[sh].reshape(-1, Zd.shape[-1]) if sh.any() else np.zeros((0, Zd.shape[-1]))
    W = cc.probe_matrix(lat(X[cell.tr][:1500]).mean(1), Ks[cell.tr][:1500])
    meta = ts.meta[ic]
    # vecto probe K tot nhat lam doi chung (muc K cho mo hinh f 'biet' ro nhat) : cot probe co R2 cao nhat
    r2 = [np.corrcoef(Zc_traj @ W[:, j], Ks[ic][:, j])[0, 1] ** 2 if Ks[ic][:, j].std() > 0 else 0 for j in range(W.shape[1])]
    probe_v = W[:, int(np.nanargmax(r2))]; probe_v = probe_v / (np.linalg.norm(probe_v) + 1e-12)
    n_pass = dict(candidates=len(concepts), useful=0, novel=0, teachable=0, not_shortcut=0, all=0)
    for k, v in enumerate(concepts):
        c = cc.concept_score(Zc, v, "max", t0)
        rec = dict(id=k, size=sizes[k], v=v)
        # (a) co ich
        U, (ulo, uhi), _ = cc.usefulness(Ks[ic], c, y[ic], cell.n_classes, seed=cell.seed, alpha=cfg.alpha / len(concepts))   # Bonferroni qua cac khai niem ung vien
        rec.update(U=U, U_lo=ulo, U_hi=uhi, useful=bool(ulo > 0))
        # (b) moi
        NS, _ = cc.schut_novelty(v, Za, Zh, cfg.k_max_svd, cell.seed)
        rho = cc.orth_ratio(v, W); NK = cc.functional_novelty(c, Ks[ic], cell.seed)
        rec.update(N_S=NS, rho=rho, N_K=NK,
                   novel=bool((NS > 0) and rho >= cfg.rho0 and NK >= cfg.N0))
        rec["novel_schut_only"] = bool(NS > 0); rec["novel_probe_only"] = bool(rho >= cfg.rho0 and NK >= cfg.N0)
        # (d) khong phai loi tat
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.model_selection import KFold
        from sklearn.metrics import r2_score
        try:
            pr = cross_val_predict(HistGradientBoostingRegressor(max_depth=3, max_iter=60, random_state=0),
                                   np.nan_to_num(meta), c, cv=KFold(5, shuffle=True, random_state=0))
            r2m = float(r2_score(c, pr))
        except Exception:
            r2m = float("nan")
        # chuyen nhom: nhom = nua tren/duoi cua do dai goc
        grp = meta[:, 0] > np.median(meta[:, 0])
        Ut = []
        for a_, b_ in ((grp, ~grp), (~grp, grp)):
            if a_.sum() > 30 and b_.sum() > 30 and len(np.unique(y[ic][b_])) > 1:
                Ut.append(cc.usefulness(Ks[ic][b_], c[b_], y[ic][b_], cell.n_classes, seed=cell.seed, B=50)[0])
        rec.update(R2_meta=r2m, U_transfer=float(np.mean(Ut)) if Ut else float("nan"),
                   not_shortcut=bool((np.isnan(r2m) or r2m <= 0.3) and (not Ut or np.mean(Ut) > 0)))
        out["concepts"].append(rec)
        n_pass["useful"] += rec["useful"]
    # (c) day duoc: chi tinh cho cac khai niem co ich + moi (tiet kiem tinh toan); bao cao ro dieu nay
    for rec in out["concepts"]:
        rec["teachable"] = None
        if rec["useful"]:                      # tinh cho moi khai niem co ich (de co duong cong ve hinh); phieu loc dem 'novel' rieng
            curve, info = cc.student_curve(f, rec["v"], lat, X[cell.disc], X[ic], cfg, cfg.teach_sizes, cell.seed,
                                           n_init=n_init, other_vecs={"probe_K": probe_v})
            nmax = max(curve["concept"].keys())
            a = curve["concept"][nmax]
            beats = all(a.mean() > curve[k][nmax].mean() for k in curve if k != "concept")
            p_rand = cc.paired_signflip_p(a, curve["random_samples"][nmax])
            re = info["rand_eval"]
            # 4 o doi chung kieu Schut (Fig.4): train{concept, random} x test{concept, random-position}
            cc_, rc_ = a.mean(), curve["random_samples"][nmax].mean()
            cr_, rr_ = re["concept"][nmax].mean(), re["random_samples"][nmax].mean()
            rec.update(teach_curve={k: {int(n): vv.tolist() for n, vv in d.items()} for k, d in curve.items()},
                       teach_info={k: v_ for k, v_ in info.items() if k != "rand_eval"},
                       teach_rand_eval={k: {int(n): vv.tolist() for n, vv in d.items()} for k, d in re.items()},
                       teach4=dict(concept_to_concept=float(cc_), random_to_concept=float(rc_),
                                   concept_to_random=float(cr_), random_to_random=float(rr_)),
                       student_naive=bool(info["student_naive"]), T_A=float(cc_ - rc_), T_A_p=p_rand,
                       teachable=bool(beats and p_rand <= p_thr and info["student_naive"]))
    for rec in out["concepts"]:                                # thu can thiep lieu-dap ung (thong tin, khong phai bo loc)
        rec["dose"] = None
        if rec["useful"]:
            try:
                pat = cc.concept_pattern(rec["v"], lat, X[cell.conf], cfg)
                rec["dose"] = cc.dose_response(lat, rec["v"], X[cell.fin][:200], pat["nu_diff"], pat["om_diff"], cfg, gen=gen)
            except Exception as ex_:
                rec["dose"] = dict(error=str(ex_))
    for rec in out["concepts"]:
        n_pass["novel"] += int(rec["useful"] and rec["novel"])
        n_pass["teachable"] += int(rec["useful"] and rec["novel"] and bool(rec["teachable"]))
        n_pass["not_shortcut"] += int(rec["useful"] and rec["novel"] and bool(rec["teachable"]) and rec["not_shortcut"])
    n_pass["all"] = n_pass["not_shortcut"]
    out["funnel"] = n_pass
    out["probe_v"] = probe_v; out["lat"] = lat; out["W"] = W
    return out
