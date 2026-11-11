"""Luu ket qua tang dan ra CSV (de chay lai/tiep tuc duoc tren Kaggle) + artifact pickle cho ve hinh."""
from __future__ import annotations
import os, pickle, json
import numpy as np
import pandas as pd


def to_md(df: pd.DataFrame, floatfmt=3) -> str:
    """Markdown table without needing tabulate. Columns whose values are all integers (counts, 0/1 flags) are printed as integers."""
    if df.empty:
        return "(empty)\n"
    cols = list(df.columns)
    allint = {}
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce")
        allint[c] = bool(v.notna().all() and np.isfinite(v).all() and (v == v.round()).all() and df[c].dtype != object)
    def fmt(v, c):
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(v): return "n/a"
            return str(int(v)) if allint[c] else f"{v:.{floatfmt}f}"
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c], c) for c in cols) + " |")
    return "\n".join(lines) + "\n"


LEGEND = ("\n*n/a = not defined (it is NOT zero): P and H need R > 0; delta_sd needs >= 2 seeds; concept columns exist only for neural networks. "
          "Numbers that are all integers are shown without decimals; 1/0 columns such as usable are fractions of runs.*\n")


class BudgetExceeded(Exception):
    """Het ngan sach thoi gian: dung SACH SE giua cac don vi (don vi dang chay khong bi cat ngang)."""


class Store:
    deadline = None            # time.time() toi da; None = khong gioi han

    def __init__(self, out_dir):
        self.out = out_dir
        self.art = os.path.join(out_dir, "artifacts"); self.fig = os.path.join(out_dir, "figs"); self.tab = os.path.join(out_dir, "tables")
        for d in (self.out, self.art, self.fig, self.tab):
            os.makedirs(d, exist_ok=True)

    def csv(self, name):
        return os.path.join(self.out, f"{name}.csv")

    def load(self, name) -> pd.DataFrame:
        p = self.csv(name)
        return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

    def has(self, name, **key):
        done = self._done(name, key)
        if not done and Store.deadline is not None:
            import time
            if time.time() > Store.deadline:
                raise BudgetExceeded(name)
        return done

    def _done(self, name, key):
        df = self.load(name)
        if df.empty or any(k not in df.columns for k in key):
            return False
        m = np.ones(len(df), bool)
        for k, v in key.items():
            m &= (df[k].astype(str) == str(v)).values
        return bool(m.any())

    def add(self, name, rows):
        rows = [rows] if isinstance(rows, dict) else rows
        if not rows:
            return
        df = pd.concat([self.load(name), pd.DataFrame(rows)], ignore_index=True)
        df.to_csv(self.csv(name), index=False)

    def save_art(self, name, obj):
        with open(os.path.join(self.art, name + ".pkl"), "wb") as fh:
            pickle.dump(obj, fh)

    def load_art(self, name, default=None):
        p = os.path.join(self.art, name + ".pkl")
        if not os.path.exists(p):
            return default
        with open(p, "rb") as fh:
            return pickle.load(fh)

    def save_table(self, name, df, title=""):
        with open(os.path.join(self.tab, name + ".md"), "w") as fh:
            fh.write((f"### {title}\n\n" if title else "") + to_md(df) + LEGEND)
