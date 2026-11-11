"""Buoc 4-7: du lieu tuong phan, trich khai niem bang toi uu loi (kieu Schut), bon bo loc.

DIEU CHINH SO VOI SCHUT (ghi ro trong bai):
 (1) rang buoc '>= 1' thay cho '>= 0' (loai nghiem v = 0), co bien long xi >= 0 co phat C (bao ti le khong can long);
 (2) trang thai an duoc chuan hoa theo trung binh/do lech chuan cua tung chieu tren tap train;
 (3) chi dung cac buoc t >= t_min (RNN nhan qua: o dau chuoi hai trang thai gan nhu trung nhau);
 (4) 'hoc tro' chon checkpoint som nhat co do trung < max(0.2, 1/|C| + 0.1) (Schut dung 0.2 voi nuoc co; voi |C| nho 0.2 la duoi muc doan mo).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score
from .models import DeepClassifier, nll
from .geometry import encode, u_of


# ------------------------------------------------------------------ tien ich
def log_odds_true(p, y):
    pt = np.clip(p[np.arange(len(y)), y], 1e-6, 1 - 1e-6)
    return np.log(pt) - np.log(1 - pt)


class Latent:
    """Chuan hoa trang thai an theo thong ke train (moi chieu)."""
    def __init__(self, model: DeepClassifier, Xtrain, layer=-1):
        self.m, self.layer = model, layer
        Z = model.latents(Xtrain[: min(len(Xtrain), 1500)], layer)
        self.mu = Z.reshape(-1, Z.shape[-1]).mean(0); self.sd = Z.reshape(-1, Z.shape[-1]).std(0) + 1e-6

    def __call__(self, X):
        return (self.m.latents(X, self.layer) - self.mu) / self.sd         # (n, L, H)


def concept_score(Z, v, mode="max", t_min=0):
    s = Z[:, t_min:, :] @ v
    return s.max(1) if mode == "max" else s.mean(1)


# ------------------------------------------------------------------ Buoc 4: du lieu tuong phan
def twin_pairs(model, lat: Latent, X, y, twins, delta_q=0.5):
    """Voi moi X goc: chon (X+, X-) trong {X} U twins co log-odds lop that cao nhat/thap nhat.
    Bo cac nhom co chenh < delta (phan vi delta_q cua chenh lech - giong Schut bo rollout co gia tri gan nhau).
    Tra ve danh sach nhom, moi nhom = dict(Zp, Zm) (P, L, H) va gap."""
    n, B = twins.shape[:2]
    allX = np.concatenate([X[:, None], twins], 1)                           # (n, B+1, L, 2)
    flat = allX.reshape(-1, *X.shape[1:]).astype(np.float32)
    p = model.proba(flat).reshape(n, B + 1, -1)
    lo = np.stack([log_odds_true(p[i], np.full(B + 1, y[i])) for i in range(n)])        # (n, B+1)
    gap = lo.max(1) - lo.min(1)
    thr = np.quantile(gap, delta_q)
    groups = []
    for i in range(n):
        if gap[i] < thr or gap[i] <= 0:
            continue
        o = np.argsort(lo[i])
        pairs = [(o[-1], o[0])]
        if B + 1 >= 4:
            pairs.append((o[-2], o[1]))
        groups.append(dict(i=i, pos=[allX[i, a] for a, b in pairs], neg=[allX[i, b] for a, b in pairs], gap=float(gap[i])))
    stats = dict(n_groups=len(groups), n_total=n, gap_median=float(np.median(gap)), gap_all=gap)
    return groups, stats


def disagree_pairs(model, gK_proba, K_std, X, y, p_f, k_nn=1):
    """Cap bat dong: X+ = thuc, f dung & g_K sai; X- = lang gieng gan nhat trong khong gian K ma CA HAI dung."""
    ok_f = p_f.argmax(1) == y; ok_g = gK_proba.argmax(1) == y
    pos = np.where(ok_f & ~ok_g)[0]; both = np.where(ok_f & ok_g)[0]
    if len(pos) < 3 or len(both) < 3:
        return [], dict(n_pos=len(pos), n_both=len(both))
    nn_ = NearestNeighbors(n_neighbors=k_nn).fit(K_std[both])
    _, ind = nn_.kneighbors(K_std[pos])
    groups = {}
    for a, row in zip(pos, ind):
        b = both[row[0]]
        groups.setdefault(int(y[a]), dict(i=int(a), pos=[], neg=[]))
        groups[int(y[a])]["pos"].append(X[a]); groups[int(y[a])]["neg"].append(X[b])
    return list(groups.values()), dict(n_pos=len(pos), n_both=len(both))


# ------------------------------------------------------------------ Buoc 5: toi uu loi
def solve_concept(D, C=5.0, norm="l1"):
    """D: (m, H) cac hang (z+ - z-)_t. min ||v||_1 + C sum xi  s.t.  D v + xi >= 1, xi >= 0.
    Tra ve v (H,), feasible (xi ~ 0), mean_slack."""
    m, H = D.shape
    if norm == "l1":
        c = np.concatenate([np.ones(2 * H), C * np.ones(m)])
        A = np.concatenate([-D, D, -np.eye(m)], 1)
        res = linprog(c, A_ub=A, b_ub=-np.ones(m), bounds=[(0, None)] * (2 * H + m), method="highs")
        if res.status != 0:
            return None, False, np.inf
        x = res.x; v = x[:H] - x[H:2 * H]; xi = x[2 * H:]
        return v, bool(xi.sum() < 1e-6), float(xi.mean())
    if norm == "l2":                            # ablation: huong chenh trung binh (nghiem L2 cua bai toan rut gon)
        v = D.mean(0); v = v / (np.linalg.norm(v) + 1e-12)
        xi = np.maximum(0, 1 - D @ (v * (1.0 / max(1e-9, np.median(D @ v)))))
        return v, bool(xi.sum() < 1e-6), float(xi.mean())
    raise ValueError(norm)


def extract_candidates(groups, lat: Latent, cfg, norm="l1", seed=0, max_groups=None):
    """Mot LP cho moi nhom. Tra ve (V (g,H) da chuan hoa don vi, feasible_rate, slack_list)."""
    rng = np.random.RandomState(seed)
    if max_groups and len(groups) > max_groups:
        groups = [groups[i] for i in rng.choice(len(groups), max_groups, replace=False)]
    vs, feas, slack = [], [], []
    for g in groups:
        P = np.stack(g["pos"]).astype(np.float32); N = np.stack(g["neg"]).astype(np.float32)
        if len(P) > cfg.lp_max_pairs:
            P, N = P[: cfg.lp_max_pairs], N[: cfg.lp_max_pairs]
        Zp, Zn = lat(P), lat(N)
        t0 = int(cfg.t_min_frac * Zp.shape[1])
        D = (Zp - Zn)[:, t0:, :].reshape(-1, Zp.shape[-1])
        v, ok, sl = solve_concept(D, cfg.lp_slack_C, norm)
        if v is None or not np.isfinite(v).all() or np.linalg.norm(v) < 1e-9:
            continue
        vs.append(v); feas.append(ok); slack.append(sl)
    stats = dict(n_lp=len(groups), n_ok=len(vs), feasible_rate=float(np.mean(feas)) if feas else float("nan"),
                 mean_slack=float(np.mean(slack)) if slack else float("nan"))
    return (np.stack(vs) if vs else np.zeros((0, 1))), stats


def cluster_concepts(V, cos_thr=0.5, min_size=3, max_concepts=12):
    """Gom cum theo cosine (khong phan biet dau khong dung: dau co nghia -> khong dao dau)."""
    if len(V) < min_size:
        return [], []
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    cl = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                 distance_threshold=1 - cos_thr).fit(Vn)
    concepts, sizes = [], []
    for k in np.unique(cl.labels_):
        mem = Vn[cl.labels_ == k]
        if len(mem) >= min_size:
            c = mem.mean(0); concepts.append(c / np.linalg.norm(c)); sizes.append(len(mem))
    o = np.argsort(sizes)[::-1][:max_concepts]
    return [concepts[i] for i in o], [sizes[i] for i in o]


# ------------------------------------------------------------------ Bo loc (a): co ich
def usefulness(K_std, c, y, n_classes, seed=0, B=300, n_splits=5, alpha=0.05):
    """U(v) = S(g_K (+) c) - S(g_K) bang log-loss ngoai-mau (cross-fit) tren tap CHUA TUNG huan luyen f.
    Tra ve U, CI (1-alpha) bootstrap theo mau, mang chenh theo mau. alpha nho (Bonferroni) can B lon: B >= 20/alpha."""
    cnt = np.bincount(y); cnt = cnt[cnt > 0]
    ns = int(min(n_splits, cnt.min()))
    if ns < 2 or len(y) < 20:                       # qua it mau de cross-fit
        return float("nan"), (float("nan"), float("nan")), np.zeros(len(y))
    kf = StratifiedKFold(n_splits=ns, shuffle=True, random_state=seed)
    def oof(F):
        P = np.zeros((len(y), n_classes))
        for tr, te in kf.split(F, y):
            m = HistGradientBoostingClassifier(max_depth=3, max_iter=80, learning_rate=0.1, random_state=seed).fit(F[tr], y[tr])
            P[np.ix_(te, m.classes_)] = m.predict_proba(F[te])
        P = 0.98 * P + 0.02 / n_classes
        return -np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1))
    l0 = oof(K_std); l1 = oof(np.concatenate([K_std, c[:, None]], 1))
    d = l0 - l1
    rng = np.random.RandomState(seed)
    B = max(B, int(20 / alpha))
    bs = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(B)]
    return float(d.mean()), (float(np.percentile(bs, 100 * alpha / 2)), float(np.percentile(bs, 100 * (1 - alpha / 2)))), d


# ------------------------------------------------------------------ Bo loc (b): moi
def schut_novelty(v, Z_a, Z_h, k_max=8, seed=0):
    """Z_a, Z_h: (N, H) trang thai (da chuan hoa) tren cac buoc cua quy dao 'may dung, g_K sai' / 'g_K dung'.
    N_S = min_k [err_h(k) - err_a(k)] ; nhan neu > 0. Can so mau bang nhau (cat theo so nho hon)."""
    rng = np.random.RandomState(seed)
    n = min(len(Z_a), len(Z_h))
    if n < 50:
        return float("nan"), None
    Za = Z_a[rng.choice(len(Z_a), n, replace=False)]; Zh = Z_h[rng.choice(len(Z_h), n, replace=False)]
    def basis(Z):
        _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
        return Vt
    Ua, Uh = basis(Za), basis(Zh)
    vn = v / (np.linalg.norm(v) + 1e-12)
    ks = range(1, min(k_max, Ua.shape[0]) + 1)
    ea = [np.linalg.norm(vn - Ua[:k].T @ (Ua[:k] @ vn)) for k in ks]
    eh = [np.linalg.norm(vn - Uh[:k].T @ (Uh[:k] @ vn)) for k in ks]
    return float(min(h - a for a, h in zip(ea, eh))), (ea, eh)


def probe_matrix(Z_traj, K_std, alpha=10.0):
    """Probe tuyen tinh (ridge) du doan tung muc K tu trang thai trung binh theo thoi gian. Tra ve W (H, m)."""
    r = Ridge(alpha=alpha).fit(Z_traj, K_std)
    return r.coef_.T


def orth_ratio(v, W):
    Q, _ = np.linalg.qr(W)
    vp = v - Q @ (Q.T @ v)
    return float(np.linalg.norm(vp) / (np.linalg.norm(v) + 1e-12))


def functional_novelty(c, K_std, seed=0):
    """N_K = 1 - R2_CV(c | K) (hoi quy GB, 5-fold)."""
    kf = KFold(5, shuffle=True, random_state=seed)
    pr = cross_val_predict(HistGradientBoostingRegressor(max_depth=3, max_iter=100, random_state=seed), K_std, c, cv=kf)
    return float(np.clip(1 - r2_score(c, pr), 0, 1))


# ------------------------------------------------------------------ Bo loc (c): day duoc cho hoc tro
def _agree(a, b):
    return float((a == b).mean())


def student_curve(f: DeepClassifier, v, lat: Latent, X_pool, X_eval, cfg, sizes, seed=0, n_init=3,
                  other_vecs=None, agree_max=None):
    """Hoc tro A (kieu Schut): checkpoint som cua f, chung cat bang KL tren `n` mau khai niem
    (n/2 diem c_v cao nhat + n/2 thap nhat trong X_pool), do do trung voi f tren mau khai niem GIU LAI (X_eval).
    Doi chung: (i) n mau ngau nhien, (ii) mau cua vecto ngau nhien, (iii) cac vecto khac (probe K) neu co.
    Tra ve dict[cond] -> dict[size] -> mang agree theo n_init khoi tao."""
    import torch, torch.nn.functional as F_
    C = f.C
    agree_max = agree_max or max(0.2, 1.0 / C + 0.1)
    Zp = lat(X_pool); Ze = lat(X_eval)
    t0 = int(cfg.t_min_frac * Zp.shape[1])
    sp = concept_score(Zp, v, "max", t0); se = concept_score(Ze, v, "max", t0)
    q = max(5, len(se) // 4)
    o = np.argsort(se); eval_idx = np.concatenate([o[:q], o[-q:]])
    Xe = X_eval[eval_idx]; pe_f = f.predict(Xe)
    ridx = np.random.RandomState(seed + 99).choice(len(X_eval), min(len(X_eval), len(Xe)), replace=False)   # vi tri NGAU NHIEN (khong chon theo khai niem)
    Xr = X_eval[ridx]; pr_f = f.predict(Xr)
    soft_pool = f.proba(X_pool)
    rng = np.random.RandomState(seed)

    # chon checkpoint som: muon nhat ma do trung voi f tren Xe < agree_max
    chosen = f.ckpts[0][1]; chosen_ep = 0
    for ep, st in f.ckpts:
        m = f.with_state(st)
        if _agree(m.predict(Xe), pe_f) <= agree_max:
            chosen, chosen_ep = st, ep
        else:
            break

    agree_pre = _agree(f.with_state(chosen).predict(Xe), pe_f)          # hoc tro TRUOC khi day: phai chua biet khai niem
    student_naive = bool(agree_pre <= agree_max)

    def train_student(idx, s):
        stu = DeepClassifier(f.kind, f.C, f.L, f.hidden, str(f.dev), seed=s, rich=getattr(f, "rich", False))
        stu.net.load_state_dict(chosen)
        stu.net.xy_scale.copy_(f.net.xy_scale); stu.net.d_scale.copy_(f.net.d_scale)
        net = stu.net; net.train()
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
        Xt = torch.tensor(X_pool[idx], dtype=torch.float32, device=stu.dev)
        tgt = torch.tensor(soft_pool[idx], dtype=torch.float32, device=stu.dev)
        for ep in range(cfg.teach_epochs):
            perm = torch.randperm(len(idx))
            for i in range(0, len(idx), 64):
                b = perm[i:i + 64]
                loss = F_.kl_div(F_.log_softmax(net(Xt[b]), -1), tgt[b], reduction="batchmean")
                opt.zero_grad(); loss.backward(); opt.step()
        stu.T = 1.0
        return stu

    conds = {"concept": lambda n, s: np.concatenate([np.argsort(sp)[: n // 2], np.argsort(sp)[-(n - n // 2):]]),
             "random_samples": lambda n, s: np.random.RandomState(s).choice(len(X_pool), n, replace=False)}
    # vecto ngau nhien (cung chuan)
    rv = np.random.RandomState(seed + 7).randn(*v.shape); rv /= np.linalg.norm(rv)
    srv = concept_score(Zp, rv, "max", t0)
    conds["random_vector"] = lambda n, s: np.concatenate([np.argsort(srv)[: n // 2], np.argsort(srv)[-(n - n // 2):]])
    for name, ov in (other_vecs or {}).items():
        so = concept_score(Zp, ov, "max", t0)
        conds[name] = (lambda so_: (lambda n, s: np.concatenate([np.argsort(so_)[: n // 2], np.argsort(so_)[-(n - n // 2):]])))(so)
    out = {k: {} for k in conds}; rand_eval = {k: {} for k in conds}
    for n in sizes:
        n = min(n, len(X_pool) - 1)
        for name, fn in conds.items():
            vals, vals_r = [], []
            for s in range(n_init):
                idx = np.asarray(fn(n, seed * 100 + s))
                stu = train_student(idx, seed * 100 + s)
                vals.append(_agree(stu.predict(Xe), pe_f)); vals_r.append(_agree(stu.predict(Xr), pr_f))
            out[name][n] = np.array(vals); rand_eval[name][n] = np.array(vals_r)
    return out, dict(ckpt_epoch=chosen_ep, agree_max=agree_max, n_eval=len(Xe), agree_pre=agree_pre, student_naive=student_naive,
                     rand_eval=rand_eval)


def paired_signflip_p(a, b, n_perm=5000, seed=0):
    """Kiem dinh hoan vi dau cho chenh ghep cap d = a - b (mot phia: trung binh d > 0).
    n <= 12: liet ke chinh xac 2^n dau (p nho nhat = 1/2^n). n > 12: Monte Carlo voi hieu chinh +1."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    if n == 0:
        return float("nan")
    obs = d.mean()
    if n <= 12:
        signs = np.array(np.meshgrid(*[[1, -1]] * n)).reshape(n, -1).T
        return float(((signs * d).mean(1) >= obs - 1e-12).mean())
    signs = np.random.RandomState(seed).choice([1, -1], (n_perm, n))
    return float((1 + ((signs * d).mean(1) >= obs - 1e-12).sum()) / (n_perm + 1))


# ------------------------------------------------------------------ Buoc 7: dien giai
def concept_pattern(v, lat: Latent, X, cfg, q=0.2):
    """Mau hinh dac trung trong khong gian dau vao: chenh trung binh cua phan du (nu, omega) giua q% cao nhat va thap nhat
    c_v (sau khi tru muc/xu huong tuyen tinh de chi con phan ngoai K tuyen tinh) + pho cong suat; dai khai niem c_t."""
    from .geometry import linear_basis
    Z = lat(X); t0 = int(cfg.t_min_frac * Z.shape[1])
    s = concept_score(Z, v, "max", t0)
    o = np.argsort(s); k = max(3, int(q * len(s)))
    lo, hi = o[:k], o[-k:]
    L = X.shape[1]; V = linear_basis(L)
    u = u_of(encode(X.astype(np.float64))); r = u - (u @ V.T) @ V
    d = r[hi].mean(0) - r[lo].mean(0)
    nu_d, om_d = d[: L - 1], d[L - 1:]
    psd = lambda a: np.abs(np.fft.rfft(a - a.mean())) ** 2
    ct = (Z @ v)                                      # (n, L) dai khai niem theo thoi gian
    return dict(nu_diff=nu_d, om_diff=om_d, psd_nu=psd(nu_d), psd_om=psd(om_d), c_t_hi=ct[hi].mean(0), c_t_lo=ct[lo].mean(0),
                score=s)


def dose_response(lat: Latent, v, X, d_nu, d_om, cfg, alphas=(0.0, 0.5, 1.0, 2.0), seed=0, gen=None):
    """Thu CAN THIEP (lieu-dap ung): dich phan du (nu, omega) cua quy dao theo mau hinh cua khai niem voi lieu alpha
    (alpha = 1: chenh trung binh giua nhom c_v cao va thap). Neu v that su dai dien cho mau hinh do, diem khai niem c_v phai
    TANG DON DIEU theo alpha. Phan tuyen tinh a = V^T u giu nguyen; K phi tuyen co the troi -> bao cao k_drift
    (trung vi theo quy dao cua max_j |dK_j|/sigma_j) de biet muc nhiem K.
    Neu truyen `gen` (bo sinh twin): them phien ban DOI CHUNG K - sau khi dich, chieu lai ve K = K goc (Adam + Gauss-Newton cua gen)
    de loai hieu ung do K troi; ket qua trong cac khoa *_ctrl. Phan nay moi tra loi cau hoi 'khai niem tac dong NGOAI K'.
    Tra ve dict(alphas, mean_c, rho (Spearman alpha~mean_c), effect (c(alpha_max)-c(0))/sd c(0), k_drift (theo alpha))."""
    from .geometry import encode as _enc, decode as _dec, u_of as _u
    from .knowledge import compute_K
    from scipy.stats import spearmanr
    X = np.asarray(X, np.float64); L = X.shape[1]
    e = _enc(X); u = _u(e); K0 = compute_K(X); sdK = K0.std(0) + 1e-9
    t0 = int(cfg.t_min_frac * L)
    means, drift, c0 = [], [], None
    for a in alphas:
        un = u.copy(); un[:, : L - 1] += a * d_nu; un[:, L - 1:] += a * d_om
        Xa = _dec(e["p1"], e["beta1"], un[:, : L - 1], un[:, L - 1:])
        c = concept_score(lat(Xa.astype(np.float32)), v, "max", t0)
        if c0 is None: c0 = c
        means.append(float(c.mean()))
        drift.append(float(np.median((np.abs(compute_K(Xa) - K0) / sdK).max(1))))
    rho = float(spearmanr(alphas, means)[0]) if len(set(means)) > 1 else 0.0
    eff = float((means[-1] - means[0]) / (c0.std() + 1e-9))
    res = dict(alphas=list(alphas), mean_c=means, rho=rho, effect=eff, k_drift=drift)
    if gen is not None:
        Kt = K0[:, gen.cols]; a_rows = u @ gen.V.T; cm, cd = [], []
        for a in alphas:
            un = u.copy(); un[:, : L - 1] += a * d_nu; un[:, L - 1:] += a * d_om
            un[:, : L - 1] = np.clip(un[:, : L - 1], gen.nu_lo, gen.nu_hi)
            un = un - ((un @ gen.V.T) - a_rows) @ gen.V
            if a > 0:
                un = gen._project(un, a_rows, e["p1"], e["beta1"], Kt, gen.proj_steps)
                if gen.newton_iters > 0:
                    un = gen._newton(un, a_rows, e["p1"], e["beta1"], Kt, gen.newton_iters)
            Xa = _dec(e["p1"], e["beta1"], un[:, : L - 1], un[:, L - 1:])
            cm.append(float(concept_score(lat(Xa.astype(np.float32)), v, "max", t0).mean()))
            cd.append(float(np.median((np.abs(compute_K(Xa) - K0) / sdK)[:, gen.cols].max(1))))
        res.update(mean_c_ctrl=cm, k_drift_ctrl=cd, rho_ctrl=float(spearmanr(alphas, cm)[0]) if len(set(cm)) > 1 else 0.0,
                   effect_ctrl=float((cm[-1] - cm[0]) / (c0.std() + 1e-9)))
    return res
