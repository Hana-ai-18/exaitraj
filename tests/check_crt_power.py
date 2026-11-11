"""KIEM TRA THONG KE cua CRT doi chung (khong phu thuoc mang no-ron): black box 'oracle' f(X) = HGB[K(X), h_raw(X)] thuc su
dung mot dac trung NGOAI K tinh truc tiep tu quy dao. Ky vong: the gioi rong (b=0) -> khong bac bo; b lon -> bac bo.
Chay: python tests/check_crt_power.py [dataset] (cham, vai phut)."""
import sys, os, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, dataclasses
from sklearn.ensemble import HistGradientBoostingClassifier
from trajconcept.config import quick
from trajconcept.data import load_dataset
from trajconcept import worlds as W
from trajconcept.knowledge import compute_K
from trajconcept.pipeline import Cell, get_gK, mu_and_crt, twin_leak


class OracleF:
    def __init__(self, fam, C, seed):
        self.fam, self.C = fam, C
        self.m = HistGradientBoostingClassifier(max_depth=3, max_iter=100, random_state=seed)
    def feats(self, X):
        X = np.asarray(X, np.float64); return np.c_[compute_K(X), W.raw_factor(X, self.fam)[0]]
    def fit(self, X, y): self.m.fit(self.feats(X), y); return self
    def proba(self, X):
        p = np.zeros((len(X), self.C)); p[:, self.m.classes_] = self.m.predict_proba(self.feats(X))
        p = 0.98 * p + 0.02 / self.C; return p / p.sum(1, keepdims=True)


def main(ds="uci_characters", fam="freq"):
    cfg = quick(); cfg.name = "crtcheck"; cfg.cache_dir = "/tmp/crtcheck_cache"; cfg.twin_proj_steps = int(os.environ.get('STEPS', 40)); cfg.flow_steps = 500; cfg.n_twin = 39; cfg.twin_pool = 48
    ts0 = load_dataset(ds, 2500)
    BS = [float(x) for x in os.environ.get('BS', '0,2,4').split(',')]; SEEDS = [int(x) for x in os.environ.get('SEEDS', '0,1').split(',')]
    for b in BS:
        for seed in SEEDS:
            y, info = W.make_world(ts0, fam, b, seed)
            ts = dataclasses.replace(ts0, name=f"{ds}-chk-{fam}-{b}", y=y, classes=list(range(len(np.unique(y)))))
            c = Cell(ts, "rf", seed, cfg); c.split(); c.K = compute_K(ts.X); c.gK = get_gK(c, cfg)
            c.f = OracleF(fam, len(np.unique(y)), seed).fit(ts.X[c.tr], ts.y[c.tr])
            gen = c.build_twin_gen(); idx = c.ho[:300]
            tw, eps, acc = c.make_twins(gen, idx, B=cfg.n_twin, pool=cfg.twin_pool)
            d = mu_and_crt(c, ts.X[idx], ts.y[idx], tw)
            print(f"b={b} seed={seed} eps_K={np.median(eps[:,0]):.2f} leak_gK={d['leak_gK']:.3f} DeltaCRT_raw={d['Delta_CRT']:.3f} ctrl={d['Delta_CRT_ctrl']:.3f} "
                  f"p_raw={d['p_raw']:.3f} p_ctrl={d['p_value']:.3f}  dropped_K_cols={len(gen.dropped_cols)}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
