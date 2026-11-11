"""Cau hinh tap trung. Moi sieu tham so deu o day de 'co dinh truoc' (muc 7.6 cua de xuat)."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple
import json, os

DATASETS = ["animals", "geolife", "hurdat2", "uci_pen_digits",
            "uci_movement_libras", "uci_characters", "mnist_stroke"]
BLACK_BOXES = ["rf", "knn", "gru", "lstm", "transformer", "trajformer"]
DEEP = ["gru", "lstm", "transformer", "trajformer"]

# so diem lay mau lai (L) theo bo. Pen digits goc chi co 8 diem.
RESAMPLE_L = {"animals": 64, "geolife": 64, "hurdat2": 32, "uci_pen_digits": 16,
              "uci_movement_libras": 45, "uci_characters": 64, "mnist_stroke": 48}


@dataclass
class Config:
    name: str = "quick"
    out_dir: str = "results"
    cache_dir: str = "cache"
    seeds: List[int] = field(default_factory=lambda: [0])
    datasets: List[str] = field(default_factory=lambda: ["uci_pen_digits"])
    black_boxes: List[str] = field(default_factory=lambda: ["rf", "knn", "gru"])
    max_n: int = 1500                 # tran so mau moi bo (lay phan tang)
    max_len_raw: int = 2000           # cat quy dao qua dai truoc khi lay mau lai
    # --- huan luyen black box
    epochs: int = 30
    batch_size: int = 128
    lr: float = 2e-3
    hidden: int = 64
    patience: int = 6
    # --- g_K
    gk_budget: int = 3                # so cau hinh moi ho mo hinh (cung ngan sach voi f)
    bootstrap_B: int = 200
    # --- twin
    n_twin: int = 39                  # B trong muc 5.4 (CRT: p nho nhat = 1/(B+1) = 0.025 khi B=39)
    twin_pool: int = 48               # so ung vien truoc khi loc dang-tri-thuc
    twin_tau: float = 0.25            # nguong sai so dang tri thuc (don vi sigma_K) khi loc
    twin_newton_iters: int = 8        # buoc Gauss-Newton sau Adam (0 = tat)
    twin_proj_steps: int = 40         # so buoc chieu ve mat K=const (0 = tat)
    twin_eps_ok: float = 0.5          # twin 'dung duoc' neu median eps_K(max) <= nguong nay (md muc 6.2 de xuat 0.05: khong dat duoc voi K 28 chieu)
    twin_c2st_ok: float = 0.65
    twin_gen: str = "flow"            # gauss | flow | naive
    flow_steps: int = 600
    flow_hidden: int = 128
    pca_dim: int = 12
    # --- khai niem
    lp_slack_C: float = 5.0
    lp_max_pairs: int = 30
    t_min_frac: float = 0.25
    cluster_cos_thr: float = 0.5
    k_max_svd: int = 8
    rho0: float = 0.8
    N0: float = 0.7
    teach_sizes: Tuple[int, ...] = (10, 50, 100, 300)
    teach_epochs: int = 25
    teach_inits: int = 5              # so khoi tao hoc tro; p nho nhat cua kiem dinh hoan vi dau = 1/2^n (5 -> 0,031)
    # --- giai thich cuc bo
    n_explain: int = 60               # so quy dao kiem tra duoc giai thich moi cau hinh
    L_perturb: int = 64               # so mau nhieu moi quy dao (XAI4Traj dung chung cho moi thi nghiem)
    rdp_eps: float = 0.05             # epsilon RDP tren toa do chuan hoa (paper: 0.005-0.5)
    window_frac: float = 0.05         # cua so truot = 5% so diem (dung nhu paper)
    noise_sigma: float = 1.0          # 'sigma*s' cua Eq.(2), tinh theo do dai buoc trung vi (paper khong neu gia tri)
    scale_factor: float = 1.5         # s cua Eq.(3)
    rot_angle: float = 0.7853981633974483  # theta cua Eq.(4): pi/4
    mdl_eps: float = 0.05             # epsilon cua MDL (chi phi = tong khoang cach vuong goc), tuong doi theo duong cheo khung bao
    ops_profile: str = "advisor"      # 'advisor': tham so mac dinh cua repo (std 3 x scale 1.5, s 1.2, theta pi/18) quy ve buoc trung vi | 'calibrated'
    ops_profiles: List[str] = field(default_factory=lambda: ["advisor", "calibrated"])   # E4 chay ca hai
    frame: Optional[tuple] = None     # (scale, ox, oy): neu co, Eq.2-4 chay trong toa do THO (do Config.ops_for dat cho tung quy dao)
    segs: List[str] = field(default_factory=lambda: ["mdl", "rdp", "window"])             # 3 cach chia doan cua paper
    e4_seeds: List[int] = field(default_factory=lambda: [0])                              # E4 ton nhieu thoi gian: 1 seed black box
    e4_n: int = 0                     # so quy dao giai thich trong E4 (0 = dung n_explain)
    # --- benchmark cai dap an
    worlds_datasets: List[str] = field(default_factory=lambda: ["uci_pen_digits"])
    worlds_bbs: List[str] = field(default_factory=lambda: ["gru"])      # black boxes dung cho the gioi cai dap an (E3)
    worlds_b: List[float] = field(default_factory=lambda: [0.0, 1.0, 2.0, 4.0])
    worlds_families: List[str] = field(default_factory=lambda: ["freq", "where"])
    crt_B: int = 99
    alpha: float = 0.05
    # --- dieu kien toi thieu cho black box (muc 7.3)
    min_margin: float = 0.15
    max_gap: float = 0.15
    max_ece: float = 0.10
    device: str = "auto"

    def ops_for(self, scale: float, profile: str = None, origin=None) -> "Config":
        """Config voi tham so toan tu nhieu (Eq.2-4) theo profile.
        'advisor': mac dinh REPO XAI4Traj trong toa do THO: noise std 3 x scale 1.5 = 4.5; s = 1.2; theta = pi/18; xoay/co gian quanh goc
        (0,0) THAT cua toa do tho. Can `origin` (diem dau tho cua quy dao, TrajSet.origin[i]) de quy doi: p_tho = X*scale + origin.
        'calibrated': khong gian chuan hoa (buoc trung vi = 1), mac dinh cua ta (sigma 1, s 1.5, theta pi/4), quay quanh diem dau."""
        import dataclasses, math
        profile = profile or self.ops_profile
        if profile == "advisor":
            o = (0.0, 0.0) if origin is None else (float(origin[0]), float(origin[1]))
            return dataclasses.replace(self, noise_sigma=3.0 * 1.5, scale_factor=1.2, rot_angle=math.pi / 18, frame=(float(scale), o[0], o[1]))
        return dataclasses.replace(self, frame=None)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def quick() -> Config:
    """Smoke test tren CPU (vai phut)."""
    return Config(name="quick", gk_budget=1, epochs=12, n_twin=24, twin_pool=48, flow_steps=300, n_explain=20, L_perturb=40,
                  bootstrap_B=100, twin_proj_steps=25, twin_newton_iters=4, lp_max_pairs=20, teach_sizes=(10, 50), teach_epochs=10, hidden=48, pca_dim=8, teach_inits=3)


def kaggle_t4() -> Config:
    """Cau hinh day du cho Kaggle T4 (xem notebook). Ngan sach da canh de nam trong ~9-12 gio, co cache nen chay lai duoc."""
    return Config(
        name="full", seeds=[0, 1, 2, 3, 4], datasets=list(DATASETS),
        black_boxes=list(BLACK_BOXES), max_n=6000, epochs=60, hidden=96, patience=8,
        gk_budget=4, bootstrap_B=1000, n_twin=39, twin_pool=64, flow_steps=3000,
        flow_hidden=256, pca_dim=16, n_explain=150, e4_n=60, L_perturb=256,
        worlds_datasets=["uci_pen_digits", "uci_movement_libras", "animals"], worlds_bbs=["gru", "transformer"],
        worlds_b=[0.0, 0.5, 1.0, 2.0, 4.0], crt_B=199, lp_max_pairs=60, twin_proj_steps=300,
        teach_sizes=(10, 50, 100, 500), teach_epochs=40)


def medium() -> Config:
    """Cau hinh trung gian: 3 seed, bo dataset nho, de tien chay thu truoc khi chay full."""
    c = kaggle_t4()
    c.name = "medium"; c.seeds = [0, 1, 2]
    c.datasets = ["uci_pen_digits", "uci_movement_libras", "uci_characters", "animals"]
    c.max_n = 3000; c.epochs = 40; c.flow_steps = 1500; c.n_explain = 100; c.e4_n = 40; c.L_perturb = 128
    c.bootstrap_B = 500; c.crt_B = 99; c.twin_proj_steps = 200; c.worlds_datasets = ["uci_pen_digits", "uci_movement_libras"]
    return c


PRESETS = {"quick": quick, "medium": medium, "full": kaggle_t4}
