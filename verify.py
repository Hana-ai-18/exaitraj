"""KIEM CHUNG phuong phap bang mot 'cell manh' va in bang ket luan PASS/FAIL cho 6 tuyen bo (xem trajconcept/claims.py).

  python verify.py --smoke --out v_smoke --cache c_smoke          # thu script chay duoc tren CPU (vai phut, KHONG ket luan)
  python verify.py --out verify_out --cache verify_cache --budget-hours 8   # ban that tren Kaggle T4; chay lai cung lenh de TIEP TUC
  python verify.py --claims-only --out verify_out                  # chi tinh lai bang ket luan tu CSV da co

Thu tu (cai gi xong truoc thi co ket qua truoc): null (C1) -> worlds (C2,C3) -> e1 (C5) -> e8 (C4) -> e5 (C5) -> e4, e4w (C6)."""
import argparse, os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trajconcept.config import PRESETS
from trajconcept.store import Store, BudgetExceeded
from trajconcept import experiments as E, claims


def stage_null(cfg, S):
    ds = cfg.worlds_datasets[0]
    for bb in cfg.worlds_bbs:
        for seed in sorted(set(cfg.seeds) | set(cfg.null_seeds if bb == cfg.worlds_bbs[0] else cfg.null_seeds[:8])):
            if S.has("v_null", dataset=ds, seed=seed, bb=bb): continue
            r = E._safe(E._run_world_cell, cfg, S, ds, "where", 0.0, seed, bb, tag="v_null", concepts=False)
            if r:
                S.add("v_null", r[0]); E._log("NULL", ds, seed, bb, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r[0].items() if k in ("bacc", "delta_nll", "p_crt", "p_raw", "detect_ctrl", "detect_raw")})


def stage_worlds(cfg, S):
    ds = cfg.worlds_datasets[0]
    for b in [x for x in cfg.worlds_b if x > 0]:
        for fam in cfg.worlds_families:
            for seed in cfg.seeds:
                for bb in cfg.worlds_bbs:
                    if S.has("v_worlds", dataset=ds, family=fam, b=b, seed=seed, bb=bb): continue
                    r = E._safe(E._run_world_cell, cfg, S, ds, fam, b, seed, bb, tag="v_worlds")
                    if r:
                        S.add("v_worlds", r[0]); E._log("WORLD", fam, b, seed, bb, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r[0].items()
                                                                                    if k in ("bacc", "delta_nll", "p_crt", "detect_crt", "rec_all", "rec_rand", "rec_pca")})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--out", default="verify_out"); ap.add_argument("--cache", default="verify_cache")
    ap.add_argument("--budget-hours", type=float, default=None); ap.add_argument("--claims-only", action="store_true")
    a = ap.parse_args(argv)
    cfg = PRESETS["verify_smoke" if a.smoke else "verify"](); cfg.out_dir, cfg.cache_dir = a.out, a.cache
    S = Store(a.out); open(os.path.join(a.out, "config.json"), "w").write(cfg.to_json())
    if a.budget_hours: Store.deadline = time.time() + a.budget_hours * 3600
    if a.claims_only: Store.deadline = 0.0
    steps = [("null", stage_null), ("worlds", stage_worlds), ("e1", E.e1_gate), ("e8", lambda c, s: E.e8_sanity(c, s, bb="gru")),
             ("e5", E.e5_concepts), ("e4", E.e4_explain), ("e4w", lambda c, s: E.e4_worlds_local(c, s, ds=c.worlds_datasets[0], fam=c.worlds_families[0], bb="gru"))]
    t0 = time.time()
    if not a.claims_only:
        for name, fn in steps:
            print(f"\n===== {name} =====", flush=True)
            try:
                fn(cfg, S)
            except BudgetExceeded as b:
                print(f"[{name}] het ngan sach thoi gian tai '{b}'. Chay lai cung lenh de tiep tuc.", flush=True); break
            print(f"[{name}] xong sau {time.time() - t0:.0f}s", flush=True)
    Store.deadline = None
    df = claims.evaluate(S)
    df.to_csv(os.path.join(a.out, "claims.csv"), index=False)
    md = claims.to_md_table(df) if hasattr(claims, "to_md_table") else df.to_string()
    from trajconcept.store import to_md
    open(os.path.join(a.out, "claims.md"), "w").write("# Bang ket luan\n\n" + to_md(df) + "\n\n" + claims.summarize(df) + "\n")
    print("\n" + df.to_string(max_colwidth=70)); print("\nTOM TAT:", claims.summarize(df))
    if a.smoke: print("\n(SMOKE: chi de kiem tra script chay duoc, KHONG phai ket luan.)")


if __name__ == "__main__":
    main()
