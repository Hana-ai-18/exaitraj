"""Merge result folders produced on different machines (e.g. results_cpu/ from your computer and results_gpu/ from Kaggle).

  python merge_results.py merged results_cpu results_gpu
  python run.py --preset full --stage report --out merged

Concatenates every *.csv (dropping exact duplicate rows), copies artifacts/ and keeps config.json of the LAST folder."""
import sys, os, shutil, glob
import pandas as pd


def main(out, *srcs):
    os.makedirs(out, exist_ok=True)
    names = sorted({os.path.basename(f) for s in srcs for f in glob.glob(os.path.join(s, "*.csv"))})
    for n in names:
        dfs = [pd.read_csv(os.path.join(s, n)) for s in srcs if os.path.exists(os.path.join(s, n))]
        d = pd.concat(dfs, ignore_index=True).drop_duplicates()
        d.to_csv(os.path.join(out, n), index=False); print(f"{n}: {len(d)} rows from {len(dfs)} folders")
    os.makedirs(os.path.join(out, "artifacts"), exist_ok=True)
    for s in srcs:
        for f in glob.glob(os.path.join(s, "artifacts", "*.pkl")):
            shutil.copy(f, os.path.join(out, "artifacts", os.path.basename(f)))
    if os.path.exists(os.path.join(srcs[-1], "config.json")):
        shutil.copy(os.path.join(srcs[-1], "config.json"), os.path.join(out, "config.json"))


if __name__ == "__main__":
    if len(sys.argv) < 3: sys.exit(__doc__)
    main(*sys.argv[1:])
