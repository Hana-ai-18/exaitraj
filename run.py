"""Run the experiments.

  python run.py --preset quick --stage all                 # smoke test of everything (CPU, ~15 min)
  python run.py --preset full  --stage cpu                 # CPU-friendly part (run on your own computer)
  python run.py --preset full  --stage gpu --budget-hours 8   # GPU part (Kaggle T4); re-run the same command to resume
  python run.py --preset full  --stage report --out merged    # rebuild tables + figures from merged CSVs, computes nothing

Stages
  cpu    E1 (random forest, k-NN), E2 (twin quality), E4 (fidelity, random forest and k-NN)
  gpu    E1, E3, E4, E4w, E5, E7, E8, E9, E10 for the neural black boxes (gru, lstm, transformer, trajformer)
  report E1..E15 + figures from the CSV files already in --out (never trains anything; unfinished units are skipped)
  all    everything, in order
Every finished unit is stored in <out>/*.csv, trained models in <cache>/, so any command can be interrupted and re-run."""
import argparse, os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trajconcept.config import PRESETS, DEEP
from trajconcept.store import Store, BudgetExceeded
from trajconcept import experiments as E
from trajconcept import plots

ALL = ["e1", "e2", "e3", "e4", "e4w", "e5", "e7", "e8", "e9", "e10", "e15"]
STAGES = {"cpu": ["e1", "e2", "e4"], "gpu": ["e1", "e3", "e4", "e4w", "e5", "e7", "e8", "e9", "e10"], "report": ALL, "all": ALL}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="quick", choices=list(PRESETS))
    ap.add_argument("--stage", default=None, choices=list(STAGES), help="cpu | gpu | report | all (default all)")
    ap.add_argument("--exp", default=None, help="explicit list, e.g. e1,e2 (overrides --stage's list)")
    ap.add_argument("--datasets", default=None); ap.add_argument("--bbs", default=None); ap.add_argument("--seeds", default=None)
    ap.add_argument("--out", default="results"); ap.add_argument("--cache", default="cache")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--budget-hours", type=float, default=None, help="stop cleanly after this many hours (the unit being computed is finished); re-run to resume")
    a = ap.parse_args(argv)
    stage = a.stage or ("all" if a.exp is None else None)
    cfg = PRESETS[a.preset]()
    cfg.out_dir, cfg.cache_dir = a.out, a.cache
    if a.datasets: cfg.datasets = a.datasets.split(",")
    if a.seeds: cfg.seeds = [int(s) for s in a.seeds.split(",")]
    if a.bbs: cfg.black_boxes = a.bbs.split(",")
    elif stage == "cpu": cfg.black_boxes = [b for b in cfg.black_boxes if b not in DEEP]
    elif stage == "gpu": cfg.black_boxes = [b for b in cfg.black_boxes if b in DEEP]
    S = Store(a.out)
    open(os.path.join(a.out, "config.json"), "w").write(cfg.to_json())
    exps = a.exp.split(",") if a.exp else STAGES[stage]
    t0 = time.time()
    if stage == "report":
        Store.deadline = 0.0                      # any unfinished unit raises BudgetExceeded -> skipped, nothing is computed
    elif a.budget_hours:
        Store.deadline = t0 + a.budget_hours * 3600
    fn = dict(e1=E.e1_gate, e2=E.e2_twin_quality, e3=lambda c, s: E.e3_worlds(c, s, bbs=tuple(c.worlds_bbs)), e4=E.e4_explain,
              e4w=E.e4_worlds_local, e5=E.e5_concepts, e7=E.e7_rediscovery, e8=E.e8_sanity, e9=E.e9_ablation,
              e10=E.e10_sensitivity, e15=E.e15_failure)
    print(f"preset={a.preset} stage={stage or 'custom'} datasets={cfg.datasets} black_boxes={cfg.black_boxes} seeds={cfg.seeds}", flush=True)
    for e in exps:
        print(f"\n===== {e} =====", flush=True)
        try:
            fn[e](cfg, S)
        except BudgetExceeded as b:
            if stage == "report":
                print(f"[{e}] some units are not finished -> skipped in the report", flush=True)
                continue
            print(f"[{e}] TIME BUDGET REACHED ({a.budget_hours} h) at '{b}'. Re-run the same command to continue.", flush=True)
            break
        print(f"[{e}] done after {time.time() - t0:.0f}s", flush=True)
    if not a.no_plots:
        print(json.dumps(plots.make_all(S), indent=1, default=str))
    print("DONE. Results in:", a.out)


if __name__ == "__main__":
    main()
