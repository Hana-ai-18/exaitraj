"""Bo sinh twin dang-tri-thuc (muc 5.3 cua de xuat).

X = D(p1, beta1, u),  u = (nu, omega).  Chieu u len co so V (muc/xu huong cua nu, omega) -> a (4 so), phan du r = u - V a.
Twin: GIU p1, beta1 va a (CHINH XAC), rut r' ~ p(r | K) roi LOC DANG-TRI-THUC bang lay mau loai bo tren cac muc K con lai
(phi tuyen). Ba bo sinh: 'gauss' (Gauss co dieu kien), 'flow' (flow matching co dieu kien), 'naive' (r tu mau khac ngau nhien,
khong dieu kien hoa -> cho ablation).

Sau khi lay mau: CHIEU bang Adam tren ban K kha vi (softk.py, thay cho Gauss-Newton cua muc 5.3.3a) roi chon B ung vien
co sai so K THAT nho nhat (lay mau loai bo 5.3.3b). eps_K bao theo K that.
"""
from __future__ import annotations
import numpy as np, torch, torch.nn as nn
from .geometry import encode, decode, u_of, linear_basis
from .knowledge import compute_K


class CondFlow(nn.Module):
    """Conditional flow matching (Lipman 2023) tren he so PCA da chuan hoa."""
    def __init__(self, m, dc, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(m + dc + 1, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, m))
    def forward(self, x, t, c):
        return self.net(torch.cat([x, c, t], -1))


class TwinGenerator:
    def __init__(self, kind="flow", n_pca=12, flow_steps=600, flow_hidden=128, device="cpu", seed=0, k_cols=None, proj_steps=40, newton_iters=0):
        self.kind, self.n_pca, self.flow_steps, self.flow_hidden = kind, n_pca, flow_steps, flow_hidden
        self.dev = torch.device(device); self.seed = seed
        self.k_cols = k_cols                       # cot K dung de dieu kien hoa & kiem tra dang tri thuc
        self.proj_steps = proj_steps; self.newton_iters = newton_iters; self.disc_idx = []; self.proj_lr = 0.05; self.proj_maxw = 0.0

    # -------------------------------------------------- huan luyen
    def fit(self, X):
        rng = np.random.RandomState(self.seed); torch.manual_seed(self.seed)
        X = np.asarray(X, np.float64); n, L = X.shape[0], X.shape[1]; self.L = L
        e = encode(X); u = u_of(e)
        self.V = linear_basis(L)
        a = u @ self.V.T; r = u - a @ self.V
        self.r_mean = r.mean(0)
        U, S, Vt = np.linalg.svd(r - self.r_mean, full_matrices=False)
        m = min(self.n_pca, Vt.shape[0]); self.m = m
        self.P = Vt[:m]                                                 # (m, 2L-3)
        coef = (r - self.r_mean) @ self.P.T                             # (n, m)
        resid = (r - self.r_mean) - coef @ self.P
        self.tail_sd = resid.std(0)                                     # nhieu duoi doc lap (phuong sai rat nho)
        self.c_mu, self.c_sd = coef.mean(0), coef.std(0) + 1e-9
        self.z = (coef - self.c_mu) / self.c_sd                         # he so chuan hoa
        K = compute_K(X)
        self.K_all_sd = K.std(0) + 1e-9
        cols = self.k_cols if self.k_cols is not None else list(range(K.shape[1]))
        # bo cac muc K HANG SO tren bo nay (vd net_disp = 0 voi UCI Characters): chia cho sigma ~ 0 la vo nghia
        self.dropped_cols = [c for c in cols if K[:, c].std() < 1e-6]
        cols = [c for c in cols if K[:, c].std() >= 1e-6]
        self.cols = cols
        from .knowledge import ALL_NAMES
        self.disc_idx = [i for i, j in enumerate(cols) if ALL_NAMES[j] in ("NCV", "NCA")]
        Kc = K[:, cols]; self.k_mu, self.k_sd = Kc.mean(0), Kc.std(0) + 1e-9
        C = (Kc - self.k_mu) / self.k_sd
        self.nu_lo, self.nu_hi = np.percentile(e["nu"], 0.05), np.percentile(e["nu"], 99.95)
        self.coef_pool = self.z.copy()
        if self.kind == "gauss":
            Cb = np.concatenate([C, np.ones((n, 1))], 1)
            lam = 1.0 * np.eye(Cb.shape[1]); lam[-1, -1] = 0
            self.W = np.linalg.solve(Cb.T @ Cb + lam * n * 1e-3, Cb.T @ self.z)
            res = self.z - Cb @ self.W
            self.res_cov = np.cov(res.T) + 1e-6 * np.eye(m)
            self.res_chol = np.linalg.cholesky(self.res_cov)
        elif self.kind == "flow":
            self.flow = CondFlow(m, C.shape[1], self.flow_hidden).to(self.dev)
            opt = torch.optim.Adam(self.flow.parameters(), lr=2e-3)
            Zt = torch.tensor(self.z, dtype=torch.float32, device=self.dev)
            Ct = torch.tensor(C, dtype=torch.float32, device=self.dev)
            bs = min(256, n)
            for it in range(self.flow_steps):
                idx = torch.randint(0, n, (bs,), device=self.dev)
                x1, c = Zt[idx], Ct[idx] + 0.05 * torch.randn(bs, Ct.shape[1], device=self.dev)
                x0 = torch.randn_like(x1); t = torch.rand(bs, 1, device=self.dev)
                xt = (1 - t) * x0 + t * x1
                loss = ((self.flow(xt, t, c) - (x1 - x0)) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            self.flow_loss = float(loss)
            self.flow.eval()
        elif self.kind != "naive":
            raise ValueError(self.kind)
        return self

    # -------------------------------------------------- lay mau
    def _sample_coef(self, Cq, rng):
        """Cq: (R, dc) dieu kien chuan hoa -> (R, m) he so chuan hoa (vector hoa theo R)."""
        R, m = len(Cq), self.m
        if self.kind == "naive":
            return self.coef_pool[rng.randint(0, len(self.coef_pool), R)]
        if self.kind == "gauss":
            mean = np.concatenate([Cq, np.ones((R, 1))], 1) @ self.W
            return mean + rng.randn(R, m) @ self.res_chol.T
        with torch.no_grad():
            x = torch.tensor(rng.randn(R, m), dtype=torch.float32, device=self.dev)
            c = torch.tensor(Cq, dtype=torch.float32, device=self.dev)
            steps = 24
            for i in range(steps):
                t = torch.full((R, 1), i / steps, device=self.dev)
                x = x + self.flow(x, t, c) / steps
        return x.cpu().numpy()

    def _project(self, u0, a_rows, p1, b1, Kt, steps, lam=0.05, lr=None):
        """Chieu cac u ve mat K = Kt bang Adam tren ban K kha vi (giu a chinh xac, giu u gan u0)."""
        from .softk import soft_K
        L = self.L
        V = torch.tensor(self.V, dtype=torch.float32); a_t = torch.tensor(a_rows, dtype=torch.float32)
        u = torch.tensor(u0, dtype=torch.float32).clone().requires_grad_(True)
        u0t = torch.tensor(u0, dtype=torch.float32)
        p1t = torch.tensor(p1, dtype=torch.float32); b1t = torch.tensor(b1, dtype=torch.float32)
        Ktt = torch.tensor(Kt, dtype=torch.float32)
        sd = torch.tensor(self.K_all_sd[self.cols], dtype=torch.float32)
        cols = torch.tensor(self.cols)
        lr = self.proj_lr if lr is None else lr
        opt = torch.optim.Adam([u], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.05)
        lo, hi = float(self.nu_lo), float(self.nu_hi)
        for _ in range(steps):
            opt.zero_grad()
            Ks = soft_K(p1t, b1t, u[:, : L - 1], u[:, L - 1:])[:, cols]
            zq = ((Ks - Ktt) / sd) ** 2
            loss = zq.mean(1) + self.proj_maxw * zq.max(1).values + lam * ((u - u0t) ** 2).mean(1)
            loss.sum().backward(); opt.step(); sched.step()
            with torch.no_grad():           # re-chieu: giu a chinh xac; kep nu
                u.data[:, : L - 1].clamp_(lo, hi)
                u.data -= ((u.data @ V.T) - a_t) @ V
        return u.detach().numpy().astype(np.float64)

    def _newton(self, u0, a_rows, p1, b1, Kt, iters=8, lam=1e-2):
        """Tinh chinh Gauss-Newton co giam chan (Levenberg) tren K kha vi: buoc d nam trong khong gian rong cua V (giu a chinh xac),
        giai  min ||d||  s.t.  J d = -r, r=(K_soft-Kt)/sd.  Chap nhan buoc chi khi tong binh phuong sai so giam (line search)."""
        from .softk import soft_K
        L = self.L
        V = torch.tensor(self.V, dtype=torch.float32)
        p1t = torch.tensor(p1, dtype=torch.float32); b1t = torch.tensor(b1, dtype=torch.float32)
        Ktt = torch.tensor(Kt, dtype=torch.float32)
        sd = torch.tensor(self.K_all_sd[self.cols], dtype=torch.float32)
        cols = torch.tensor(self.cols); lo, hi = float(self.nu_lo), float(self.nu_hi)
        u = torch.tensor(u0, dtype=torch.float32)
        def res(uu):
            return (soft_K(p1t, b1t, uu[:, : L - 1], uu[:, L - 1:])[:, cols] - Ktt) / sd
        for _ in range(iters):
            uu = u.clone().requires_grad_(True)
            r = res(uu); R, m = r.shape
            J = torch.stack([torch.autograd.grad(r[:, j].sum(), uu, retain_graph=True)[0] for j in range(m)], 1)   # (R,m,D)
            J = J - (J @ V.T) @ V                                      # chi di chuyen trong null(V)
            A = J @ J.transpose(1, 2) + lam * torch.eye(m).expand(R, m, m)
            d = -(J.transpose(1, 2) @ torch.linalg.solve(A, r.detach().unsqueeze(2))).squeeze(2)
            f0 = (r.detach() ** 2).sum(1)
            step = torch.ones(R, 1); best = u.clone(); bestf = f0.clone()
            for _ls in range(4):
                un = u + step * d
                un[:, : L - 1] = un[:, : L - 1].clamp(lo, hi)
                with torch.no_grad():
                    fn = (res(un) ** 2).sum(1)
                better = fn < bestf
                best[better] = un[better]; bestf = torch.where(better, fn, bestf)
                step = step * 0.5
            u = best
        return u.detach().numpy().astype(np.float64)

    def twins(self, X, B=16, pool=64, seed=0, tau=0.25, steps=None, chunk=16):
        """X: (n,L,2) -> twins (n,B,L,2), eps (n,B) = max_j |K_j(X')-K_j(X)|/sigma_j (K THAT, tren cac cot dieu kien),
        accept (n,) = ti le ung vien co eps <= tau.  Sau goi: self.last_eps_mean (n,B) = trung binh theo j.
        steps: so buoc chieu (None -> self.proj_steps)."""
        steps = self.proj_steps if steps is None else steps
        rng = np.random.RandomState(seed)
        X = np.asarray(X, np.float64); n, L = X.shape[:2]
        e = encode(X); u = u_of(e); a = u @ self.V.T
        K = compute_K(X)[:, self.cols]
        C = (K - self.k_mu) / self.k_sd
        out = np.zeros((n, B, L, 2)); eps = np.zeros((n, B)); epm = np.zeros((n, B)); acc = np.zeros(n)
        sd = self.K_all_sd[self.cols]
        for s0 in range(0, n, chunk):
            ix = np.arange(s0, min(n, s0 + chunk)); m = len(ix)
            rep = lambda arr: np.repeat(arr[ix], pool, 0)
            z = self._sample_coef(rep(C), rng)
            coef = z * self.c_sd + self.c_mu
            r = self.r_mean + coef @ self.P + rng.randn(m * pool, len(self.tail_sd)) * self.tail_sd
            a_rows = rep(a)
            u2 = a_rows @ self.V + r
            u2[:, : L - 1] = np.clip(u2[:, : L - 1], self.nu_lo, self.nu_hi)
            u2 = u2 - ((u2 @ self.V.T) - a_rows) @ self.V
            if steps > 0:
                u2 = self._project(u2, a_rows, rep(e["p1"]), rep(e["beta1"]), rep(K), steps)
            if self.newton_iters > 0:
                u2 = self._newton(u2, a_rows, rep(e["p1"]), rep(e["beta1"]), rep(K), self.newton_iters)
            Xc = decode(rep(e["p1"]), rep(e["beta1"]), u2[:, : L - 1], u2[:, L - 1:])
            Kc = compute_K(Xc)[:, self.cols]
            zq = np.abs(Kc - rep(K)) / sd
            ep, epmean = zq.max(1).reshape(m, pool), zq.mean(1).reshape(m, pool)
            Xc = Xc.reshape(m, pool, L, 2)
            # chon ung vien: uu tien KHOP CHINH XAC cac muc dem (NCV, NCA: roi rac, khong chieu duoc) roi toi sai so lon nhat
            mism = (zq[:, self.disc_idx] > 1e-9).any(1).reshape(m, pool) if len(self.disc_idx) else np.zeros((m, pool), bool)
            score = ep + 100.0 * mism
            for k in range(m):
                o = np.argsort(score[k])[:B]
                out[ix[k]], eps[ix[k]], epm[ix[k]] = Xc[k, o], ep[k, o], epmean[k, o]
                acc[ix[k]] = (ep[k] <= tau).mean()
        self.last_eps_mean = epm
        return out, eps, acc

    # -------------------------------------------------- twin cuc bo theo doan (cho phi^MU)
    def redraw_segment(self, X, twin, mask):
        """Giu K toan quy dao; thay phan du (nu, omega) TRONG doan `mask` (L,) bang cua twin, roi chieu lai a ve gia tri goc.
        X, twin: (L,2); tra ve (L,2)."""
        L = len(X)
        e = encode(X[None].astype(np.float64)); t = encode(twin[None].astype(np.float64))
        u = u_of(e)[0]; ut = u_of(t)[0]
        a0 = u @ self.V.T
        in_nu = mask[1:]                        # buoc i->i+1 thuoc doan neu diem i+1 thuoc doan
        in_om = mask[2:]
        sel = np.concatenate([in_nu, in_om])
        un = np.where(sel, ut, u)
        un = un - ((un @ self.V.T) - a0) @ self.V
        return decode(e["p1"], e["beta1"], un[None, : L - 1], un[None, L - 1:])[0]
