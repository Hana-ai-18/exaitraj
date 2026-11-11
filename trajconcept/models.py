"""6 black box cua XAI4Traj (RF, KNN, GRU, LSTM, Transformer, TrajFormer) + g_K (mo hinh chi dung K).

TRUNG THUC VE CAI DAT
 - RF/KNN o day chay tren TOA DO tho (da lay mau lai) = 'RF-raw' trong de xuat (muc 7.2). RF-K nam trong g_K.
 - TrajFormer: ban 'lite' (nhung cuc bo bang Conv1d tren [x,y,dx,dy,toc do,goc re] + Transformer). KHONG gom buoc
   gop patch thich nghi cua Liang et al. 2022. Neu can so khop tuyet doi, thay bang ma goc va giu giao dien DeepClassifier.
 - Dau vao mang sau: [x, y, dx, dy] chuan hoa theo thong ke train.
"""
from __future__ import annotations
import copy, math, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, balanced_accuracy_score

EPS = 1e-6


def get_device(pref="auto"):
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def seed_all(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def prefeat(x, xy_scale, d_scale):
    """x: (B,L,2) tensor toa do. -> (B,L,4) [x,y,dx,dy] chuan hoa (vi sai bang 0 o buoc dau)."""
    d = torch.cat([torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]], 1)
    return torch.cat([x / xy_scale, d / d_scale], -1)


class SeqNet(nn.Module):
    """Tra ve (logits, [z_layer1, z_layer2, ...]) moi z co dang (B,L,H). Pool trung binh theo thoi gian."""
    def __init__(self, kind, L, n_classes, hidden=64, n_layers=2, dropout=0.1, rich=False):
        super().__init__()
        self.kind, self.L, self.H, self.nl = kind, L, hidden, n_layers
        self.rich = bool(rich) and kind in ("gru", "lstm", "transformer")       # trajformer da co dac trung toc do/goc re
        in_dim = 6 if self.rich else 4
        self.register_buffer("xy_scale", torch.tensor(1.0))
        self.register_buffer("d_scale", torch.tensor(1.0))
        if kind in ("gru", "lstm"):
            Rnn = nn.GRU if kind == "gru" else nn.LSTM
            self.inp = nn.Linear(in_dim, hidden)
            self.rnn = nn.ModuleList([Rnn(hidden, hidden, batch_first=True) for _ in range(n_layers)])
        elif kind in ("transformer", "trajformer"):
            if kind == "trajformer":
                self.conv = nn.Conv1d(6, hidden, 3, padding=1)    # dac trung cuc bo: +toc do, +goc re
            else:
                self.inp = nn.Linear(in_dim, hidden)
            self.pos = nn.Parameter(torch.randn(1, L, hidden) * 0.02)
            self.enc = nn.ModuleList([nn.TransformerEncoderLayer(hidden, 4, 2 * hidden, dropout, batch_first=True,
                                                                 norm_first=True) for _ in range(n_layers)])
            self.norm = nn.LayerNorm(hidden)
        else:
            raise ValueError(kind)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x, return_z=False):
        f = prefeat(x, self.xy_scale, self.d_scale)
        if self.rich:                                  # them toc do va goc re: dac trung cuc bo ma mang tho kho tu hoc tu (dx, dy)
            sp = f[..., 2:4].norm(dim=-1, keepdim=True)
            ang = torch.atan2(f[..., 3:4], f[..., 2:3])
            dang = torch.cat([torch.zeros_like(ang[:, :1]),
                              torch.atan2(torch.sin(ang[:, 1:] - ang[:, :-1]), torch.cos(ang[:, 1:] - ang[:, :-1]))], 1)
            f = torch.cat([f, sp, dang], -1)
        zs = []
        if self.kind in ("gru", "lstm"):
            h = self.inp(f)
            for r in self.rnn:
                h, _ = r(h); zs.append(h)
        else:
            if self.kind == "trajformer":
                sp = f[..., 2:4].norm(dim=-1, keepdim=True)
                ang = torch.atan2(f[..., 3:4], f[..., 2:3])
                dang = torch.cat([torch.zeros_like(ang[:, :1]),
                                  torch.atan2(torch.sin(ang[:, 1:] - ang[:, :-1]), torch.cos(ang[:, 1:] - ang[:, :-1]))], 1)
                g = torch.cat([f, sp, dang], -1).transpose(1, 2)
                h = self.conv(g).transpose(1, 2)
            else:
                h = self.inp(f)
            h = h + self.pos
            for e in self.enc:
                h = e(h); zs.append(h)
            zs[-1] = self.norm(zs[-1]); h = zs[-1]
        logits = self.head(self.drop(h.mean(1)))
        return (logits, zs) if return_z else logits


class DeepClassifier:
    """GRU/LSTM/Transformer/TrajFormer-lite. Co: hieu chinh nhiet do, checkpoint tung epoch (cho 'hoc tro' muc 5.7c)."""
    def __init__(self, kind, n_classes, L, hidden=64, device="auto", seed=0, rich=False):
        self.kind, self.C, self.L, self.hidden, self.rich = kind, n_classes, L, hidden, bool(rich)
        self.dev = get_device(device); self.seed = seed
        seed_all(seed)
        self.net = SeqNet(kind, L, n_classes, hidden, rich=rich).to(self.dev)
        self.T = 1.0
        self.ckpts = []                        # [(epoch, state_dict cpu)]
        self.history = []

    # ---------------- huan luyen
    def fit(self, X, y, Xv, yv, epochs=40, bs=128, lr=2e-3, patience=6, keep_ckpts=True, verbose=False):
        dev = self.dev
        X = torch.tensor(X, dtype=torch.float32); y_t = torch.tensor(y, dtype=torch.long)
        self.net.xy_scale.fill_(float(X.std()) + EPS)
        d = X[:, 1:] - X[:, :-1]
        self.net.d_scale.fill_(float(d.std()) + EPS)
        Xv_t = torch.tensor(Xv, dtype=torch.float32, device=dev); yv_t = torch.tensor(yv, dtype=torch.long, device=dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-3)
        steps = epochs * math.ceil(len(X) / bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
        best, best_state, bad = 1e9, None, 0
        if keep_ckpts:
            self.ckpts.append((0, copy.deepcopy({k: v.cpu() for k, v in self.net.state_dict().items()})))
        g = torch.Generator().manual_seed(self.seed)
        for ep in range(1, epochs + 1):
            self.net.train()
            perm = torch.randperm(len(X), generator=g)
            for i in range(0, len(X), bs):
                idx = perm[i:i + bs]
                xb, yb = X[idx].to(dev), y_t[idx].to(dev)
                loss = F.cross_entropy(self.net(xb), yb)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step(); sched.step()
            self.net.eval()
            with torch.no_grad():
                lv = F.cross_entropy(self.net(Xv_t), yv_t).item()
            self.history.append(lv)
            if keep_ckpts:
                self.ckpts.append((ep, copy.deepcopy({k: v.cpu() for k, v in self.net.state_dict().items()})))
            if lv < best - 1e-4:
                best, bad = lv, 0
                best_state = copy.deepcopy({k: v.cpu() for k, v in self.net.state_dict().items()}); best_ep = ep
            else:
                bad += 1
                if bad >= patience:
                    break
            if verbose:
                print(ep, round(lv, 4))
        self.net.load_state_dict(best_state); self.best_epoch = best_ep
        self.ckpts = [(e, s) for (e, s) in self.ckpts if e <= best_ep]     # chi giu checkpoint truoc diem tot nhat
        self._calibrate(Xv, yv)
        return self

    def _calibrate(self, Xv, yv):
        lg = torch.tensor(self.logits(Xv), dtype=torch.float32); yt = torch.tensor(yv)
        logT = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=50)
        def closure():
            opt.zero_grad(); l = F.cross_entropy(lg / logT.exp(), yt); l.backward(); return l
        opt.step(closure)
        self.T = float(logT.exp().clamp(0.05, 20.0))

    # ---------------- suy luan
    @torch.no_grad()
    def logits(self, X, bs=1024):
        self.net.eval(); out = []
        for i in range(0, len(X), bs):
            xb = torch.tensor(np.asarray(X[i:i + bs]), dtype=torch.float32, device=self.dev)
            out.append(self.net(xb).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.C))

    def proba(self, X):
        lg = self.logits(X) / self.T
        lg = lg - lg.max(1, keepdims=True)
        p = np.exp(lg); return p / p.sum(1, keepdims=True)

    def predict(self, X):
        return self.proba(X).argmax(1)

    @torch.no_grad()
    def latents(self, X, layer=-1, bs=1024):
        """(n, L, H) trang thai an o lop `layer` (-1 = lop cuoi)."""
        self.net.eval(); out = []
        for i in range(0, len(X), bs):
            xb = torch.tensor(np.asarray(X[i:i + bs]), dtype=torch.float32, device=self.dev)
            _, zs = self.net(xb, return_z=True)
            out.append(zs[layer].cpu().numpy())
        return np.concatenate(out)

    def proba_torch(self, x):
        """cho Integrated Gradients: x tensor (B,L,2) co grad."""
        self.net.eval()
        return F.softmax(self.net(x) / self.T, -1)

    def clone_from_state(self, state):
        m = DeepClassifier(self.kind, self.C, self.L, self.hidden, str(self.dev), self.seed, rich=self.rich)
        m.net.load_state_dict(state); m.T = 1.0
        return m

    def with_state(self, state):
        m = copy.copy(self); m.net = copy.deepcopy(self.net); m.net.load_state_dict(state); m.T = 1.0
        return m


class SklearnRaw:
    """RF-raw / KNN tren toa do tho da lay mau lai; bao xac suat lam muot de log-loss huu han."""
    def __init__(self, kind, n_classes, seed=0, n_estimators=200):
        self.kind, self.C = kind, n_classes
        if kind == "rf":
            self.m = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=seed, min_samples_leaf=1)
        elif kind == "knn":
            self.m = KNeighborsClassifier(n_neighbors=7, weights="distance")
        else:
            raise ValueError(kind)

    def _flat(self, X):
        X = np.asarray(X); return X.reshape(len(X), -1)

    def fit(self, X, y, Xv=None, yv=None, *a, **k):
        self.m.fit(self._flat(X), y); self.T = 1.0
        if Xv is not None:                              # hieu chinh nhiet do tren validation (dieu kien ECE trong muc 7.3)
            raw = self._raw(Xv); lg = np.log(raw)
            grid = np.exp(np.linspace(-1.5, 2.0, 36))
            nl = [-np.mean(np.log(np.clip(self._soft(lg / T)[np.arange(len(yv)), yv], 1e-9, 1))) for T in grid]
            self.T = float(grid[int(np.argmin(nl))])
        return self

    def _raw(self, X):
        p = np.zeros((len(X), self.C)); p[:, self.m.classes_] = self.m.predict_proba(self._flat(X))
        p = 0.98 * p + 0.02 / self.C                   # lam muot nhe tranh log(0)
        return p / p.sum(1, keepdims=True)

    @staticmethod
    def _soft(lg):
        lg = lg - lg.max(1, keepdims=True); e = np.exp(lg); return e / e.sum(1, keepdims=True)

    def proba(self, X):
        return self._soft(np.log(self._raw(X)) / self.T)

    def predict(self, X):
        return self.proba(X).argmax(1)


def make_blackbox(kind, n_classes, L, cfg, seed):
    if kind in ("rf", "knn"):
        return SklearnRaw(kind, n_classes, seed, n_estimators=200 if cfg.name != "quick" else 80)
    return DeepClassifier(kind, n_classes, L, cfg.hidden, cfg.device, seed, rich=getattr(cfg, "deep_rich", False))


def train_blackbox(kind, X, y, Xv, yv, n_classes, cfg, seed):
    L = X.shape[1]
    m = make_blackbox(kind, n_classes, L, cfg, seed)
    if isinstance(m, DeepClassifier):
        m.fit(X, y, Xv, yv, epochs=cfg.epochs, bs=cfg.batch_size, lr=cfg.lr, patience=cfg.patience)
    else:
        m.fit(X, y, Xv, yv)
    return m


# ------------------------------------------------------------------ g_K: mo hinh chi dung the K
class GK:
    """Mo hinh tot nhat chi nhan K_r. Chon theo log-loss validation trong {HistGB, RF, MLP}, moi ho `budget` cau hinh
    (cung ngan sach tinh chinh voi f). Co the tai hien: seed co dinh."""
    def __init__(self, n_classes, budget=3, seed=0):
        self.C, self.budget, self.seed = n_classes, budget, seed
        self.sc = StandardScaler()

    def _cands(self):
        b, s = self.budget, self.seed
        gb = [dict(max_depth=d, learning_rate=lr, max_iter=it, l2_regularization=1.0, random_state=s,
                   early_stopping=False) for d, lr, it in [(3, 0.1, 150), (5, 0.05, 250), (None, 0.05, 200), (4, 0.1, 100)][:b]]
        rf = [dict(n_estimators=n, min_samples_leaf=l, n_jobs=-1, random_state=s)
              for n, l in [(300, 1), (300, 3), (500, 2), (200, 5)][:b]]
        mlp = [dict(hidden_layer_sizes=h, alpha=a, max_iter=600, early_stopping=True, random_state=s)
               for h, a in [((64, 64), 1e-3), ((128,), 1e-2), ((128, 64), 1e-3), ((32,), 1e-2)][:b]]
        return [("hgb", HistGradientBoostingClassifier, c) for c in gb] + [("rf", RandomForestClassifier, c) for c in rf] + \
               [("mlp", MLPClassifier, c) for c in mlp]

    def _proba(self, m, Z):
        p = np.zeros((len(Z), self.C)); p[:, m.classes_] = m.predict_proba(Z)
        p = 0.98 * p + 0.02 / self.C
        return p / p.sum(1, keepdims=True)

    def fit(self, K, y, Kv, yv):
        self.sc.fit(K); Z, Zv = self.sc.transform(K), self.sc.transform(Kv)
        best = (1e9, None); self.table = []
        for fam, cls, kw in self._cands():
            try:
                m = cls(**kw).fit(Z, y)
                ll = log_loss(yv, self._proba(m, Zv), labels=list(range(self.C)))
            except Exception as e:                       # mot ho hong khong lam hong ca gate
                self.table.append((fam, str(kw), float("nan"))); continue
            self.table.append((fam, str(kw), ll))
            if ll < best[0]:
                best = (ll, (fam, m))
        self.family, self.m = best[1]; self.val_ll = best[0]
        return self

    def proba(self, K):
        return self._proba(self.m, self.sc.transform(K))

    def predict(self, K):
        return self.proba(K).argmax(1)


# ------------------------------------------------------------------ thuoc do ky nang & hieu chinh
def nll(p, y):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1))))


def bacc(p, y):
    return float(balanced_accuracy_score(y, p.argmax(1)))


def ece(p, y, bins=10):
    conf = p.max(1); pred = p.argmax(1); acc = (pred == y).astype(float)
    e = 0.0; edges = np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)
