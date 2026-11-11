"""FPR cua CRT o THE GIOI RONG (b=0) voi nhieu ban lap (nhan thay doi, twin co dinh vi twin khong phu thuoc nhan).
Do: FPR cua CRT tho, CRT doi chung g_K, va CRT 'null thuc nghiem' (nguong tinh tu cac the gioi placebo, kiem tra cheo chia doi).
Chay: REPS=40 STEPS=100 NEWTON=8 DS=uci_pen_digits python tests/crt_fpr.py"""
import sys, os, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, dataclasses
from trajconcept.config import quick
from trajconcept.data import load_dataset
from trajconcept import worlds as W
from trajconcept.knowledge import compute_K, tier_cols
from trajconcept.pipeline import Cell, mu_and_crt
from trajconcept.models import GK
sys.path.insert(0, os.path.dirname(__file__))
from check_crt_power import OracleF


def main():
    ds = os.environ.get("DS", "uci_pen_digits"); fam = os.environ.get("FAM", "freq"); R = int(os.environ.get("REPS", 40))
    cfg = quick(); cfg.name = "fpr"; cfg.cache_dir = "/tmp/fpr_cache"; cfg.twin_proj_steps = int(os.environ.get("STEPS", 100))
    cfg.twin_newton_iters = int(os.environ.get("NEWTON", 8)); cfg.flow_steps = 400; cfg.n_twin = int(os.environ.get("NT", 39)); cfg.twin_pool = int(os.environ.get("POOL", 48))
    ts0 = load_dataset(ds, int(os.environ.get("N", 2500)))
    y0, _ = W.make_world(ts0, fam, 0.0, 0)
    ts = dataclasses.replace(ts0, name=f"{ds}-fpr", y=y0, classes=list(range(len(np.unique(y0)))))
    c = Cell(ts, "rf", 0, cfg); c.split(); K = compute_K(ts.X); c.K = K
    gen = c.build_twin_gen(); idx = c.ho[:int(os.environ.get("NH", 300))]
    t0 = time.time(); tw, eps, acc = c.make_twins(gen, idx, B=cfg.n_twin, pool=cfg.twin_pool)
    print(f"twins: {time.time()-t0:.0f}s eps_K med={np.median(eps[:,0]):.2f} frac<=0.5={(eps[:,0]<=0.5).mean():.2f}", flush=True)
    rows = []
    for r in range(1, R + 1):
        y, _ = W.make_world(ts0, fam, 0.0, r); C = len(np.unique(y))
        f = OracleF(fam, C, r).fit(ts.X[c.tr], y[c.tr])
        cols = tier_cols(2)
        g = GK(C, cfg.gk_budget, r).fit(K[c.tr][:, cols], y[c.tr], K[c.va][:, cols], y[c.va])
        c.f = f; c.gK = [g, g, g]; c.n_classes = C
        d = mu_and_crt(c, ts.X[idx], y[idx], tw)
        rows.append((d["p_raw"], d["p_value"], d["Delta_CRT_ctrl"], d["Delta_CRT"]))
        print(f"rep {r}: p_raw={d['p_raw']:.3f} p_ctrl={d['p_value']:.3f} dctrl={d['Delta_CRT_ctrl']:.4f}", flush=True)
    a = np.array(rows)
    from scipy.stats import beta
    def fpr(v):
        k, n = int((v < 0.05).sum()), len(v); lo = beta.ppf(.025, k, n - k + 1) if k else 0; hi = beta.ppf(.975, k + 1, n - k) if k < n else 1
        return f"{k/n:.3f} [{lo:.3f},{hi:.3f}]"
    print("FPR raw :", fpr(a[:, 0])); print("FPR ctrl:", fpr(a[:, 1]))
    d = a[:, 2]; h = len(d) // 2; crit = np.quantile(d[:h], 0.95)
    print(f"null thuc nghiem: nguong(95% tren nua dau)={crit:.4f}; FPR tren nua sau = {(d[h:] > crit).mean():.3f} (n={len(d)-h})")


if __name__ == "__main__":
    main()
