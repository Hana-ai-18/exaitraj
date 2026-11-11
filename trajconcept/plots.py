"""Five key figures (English only). Each figure supports ONE claim of the paper:
 Fig 1  Knowledge gate      : does the black box know MORE than the best model that uses only the expert card K?   (E1)
 Fig 2  Planted-answer test : is the method silent when nothing is hidden (b=0) and does it find/localise it (b>0)? (E3, E4w)
 Fig 3  Teachability        : can a naive student learn the concept, and only from concept examples?               (E5/E6)
 Fig 4  Fidelity R/P/H      : does the importance map really identify the segments the model relies on?             (E4)
 Fig 5  Worked example      : original vs. isometric twin (same K) and the resulting importance maps               (E4)
Figures with no data are skipped (never drawn empty)."""
from __future__ import annotations
import os, glob, pickle, warnings
warnings.filterwarnings("ignore", message="Mean of empty slice")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .store import Store

PAL = ["#2a6fbb", "#d1495b", "#3b8f5f", "#8c6bb1", "#e0a030", "#4d4d4d", "#17a2b8", "#a05a2c"]
BBC = {"rf": "#4d4d4d", "knn": "#e0a030", "gru": "#2a6fbb", "lstm": "#17a2b8", "transformer": "#d1495b", "trajformer": "#8c6bb1"}
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25})


def _preset(S: Store):
    try:
        import json
        return json.load(open(os.path.join(S.out, "config.json"))).get("name", "")
    except Exception:
        return ""


def _save(fig, S: Store, name, note=None):
    if _preset(S) == "quick":
        fig.text(0.5, 1.09, "QUICK PRESET: smoke test only, not evidence", ha="center", va="bottom", fontsize=9, color="#c0392b", fontweight="bold")
    if note:
        fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=7.5, color="0.3", wrap=True)
    p = os.path.join(S.fig, name + ".png"); fig.tight_layout(); fig.savefig(p, bbox_inches="tight"); plt.close(fig); return p


# ---------------------------------------------------------------- Fig 1
def fig1_knowledge_gate(S: Store):
    df = S.load("e1_gate")
    if df.empty: return None
    dss = list(df.dataset.unique()); n = len(dss); cols = min(4, n); rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.0 * rows), squeeze=False)
    for ax, ds in zip(axs.ravel(), dss):
        for bb, g in df[df.dataset == ds].groupby("bb"):
            a = g.groupby("tier").agg(m=("delta_nll", "mean"), lo=("delta_nll_lo", "mean"), hi=("delta_nll_hi", "mean"))
            ax.plot(a.index, a.m, "-o", color=BBC.get(bb, "k"), label=bb, ms=3)
            ax.fill_between(a.index, a.lo, a.hi, color=BBC.get(bb, "k"), alpha=0.12)
        ax.axhline(0, color="k", lw=0.8); ax.set_title(ds, fontsize=9)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["K0\ngeometry", "K1\n+ecology", "K2\n+kinematics"], fontsize=7)
        ax.set_ylabel("Knowledge gap Δ (log-loss)")
    for ax in axs.ravel()[n:]: ax.axis("off")
    axs[0, 0].legend(fontsize=7, ncol=2, title="black box", title_fontsize=7)
    fig.suptitle("Fig 1 – Does the black box know more than the expert card K?", y=1.02)
    return _save(fig, S, "Fig1_knowledge_gate",
                 "Δ > 0 (band above the line): the black box beats the best model that only sees the K features of that tier -> there is something to discover.\n"
                 "Δ <= 0: no knowledge gap, so no machine-unique concept should be claimed. Bands: 95% bootstrap CI.")


# ---------------------------------------------------------------- Fig 2
def fig2_planted_answer(S: Store):
    df = S.load("e3_worlds")
    if df.empty: return None
    loc = S.load("e4_worlds_local")
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.3))
    for k, fam in enumerate(df.family.unique()):
        g = df[df.family == fam].groupby("b").agg(ctrl=("detect_ctrl", "mean"), raw=("detect_raw", "mean"), rec=("rec_all", "mean"))
        axs[0].plot(g.index, g.ctrl, "-o", color=PAL[k], label=f"{fam}: controlled CRT (used)")
        axs[0].plot(g.index, g.raw, ":^", color=PAL[k], alpha=0.55, label=f"{fam}: raw CRT (invalid)")
        axs[1].plot(g.index, g.rec, "-o", color=PAL[k], label=fam)
    axs[0].axhline(0.05, color="k", lw=0.8, ls=":"); axs[0].set_ylim(-0.02, 1.02)
    axs[0].set_xlabel("planted-signal strength b  (b = 0: empty world)"); axs[0].set_ylabel("detection rate")
    axs[0].set_title("Detection rate vs. planted strength"); axs[0].legend(fontsize=6.5)
    axs[1].set_xlabel("planted-signal strength b"); axs[1].set_ylabel("|corr| (best concept, planted factor)")
    axs[1].set_title("Recovery of the planted factor"); axs[1].legend(fontsize=7); axs[1].set_ylim(0, 1)
    if not loc.empty:
        a = loc.groupby("method").agg(r=("relevant_mass_ratio", "mean")).sort_values("r")
        axs[2].barh([m.replace("trajconcept_n", "φ^MU normalised").replace("trajconcept", "φ^MU") for m in a.index], a.r, color=PAL[0])
        axs[2].axvline(1, color="k", ls=":")
    axs[2].set_title("Localisation of the planted segment"); axs[2].set_xlabel("importance mass on true segment / chance")
    fig.suptitle("Fig 2 – Planted-answer benchmark: the ground truth is known", y=1.03)
    return _save(fig, S, "Fig2_planted_answer",
                 "Left: dotted line = 5% false-alarm level. The raw CRT fires even at b=0 (twins leak K), so only the controlled CRT is used. "
                 "Middle: higher is better. Right: 1 = chance, >1 = better than chance.")


# ---------------------------------------------------------------- Fig 3
def fig3_teachability(S: Store):
    df = S.load("e6_teach")
    if df.empty: return None
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.3), gridspec_kw=dict(width_ratios=[1.2, 1]))
    names = {"concept": "concept examples", "random_samples": "random examples", "random_vector": "random direction", "probe_K": "known-K direction"}
    for k, c in enumerate(names):
        g = df[df.cond == c]
        if g.empty: continue
        a = g.groupby("n").agg(m=("agree", "mean"), s=("agree", "std")).fillna(0)
        axs[0].errorbar(a.index, a.m, a.s, marker="o", ms=3, color=PAL[k], label=names[c], capsize=2, lw=2.2 if c == "concept" else 1.2)
    axs[0].set_xscale("log"); axs[0].set_xlabel("number of teaching examples"); axs[0].set_ylabel("agreement with the model (held-out)")
    axs[0].set_title("A naive student taught with…"); axs[0].legend(fontsize=7)
    nmax = df.n.max(); cells = {}
    for tr in ("concept", "random_samples"):
        for te, suf in (("concept examples", ""), ("random positions", "@random_pos")):
            g = df[(df.cond == tr + suf) & (df.n == nmax)]
            if not g.empty: cells[(tr, te)] = (g.agree.mean(), g.agree.std() if len(g) > 1 else 0.0)
    if cells:
        w = 0.36
        for i, tr in enumerate(("concept", "random_samples")):
            xs = [0 + i * w - w / 2, 1 + i * w - w / 2]
            ys = [cells.get((tr, te), (np.nan, 0)) for te in ("concept examples", "random positions")]
            axs[1].bar(xs, [y[0] for y in ys], w, yerr=[y[1] for y in ys], color=PAL[0] if tr == "concept" else PAL[4], capsize=2,
                       label=f"trained on {names[tr]}")
        axs[1].set_xticks([0, 1]); axs[1].set_xticklabels(["tested on\nconcept examples", "tested on\nrandom positions"])
        axs[1].set_ylabel("agreement with the model"); axs[1].set_title(f"Four-way control (n={int(nmax)})"); axs[1].legend(fontsize=7)
    fig.suptitle("Fig 3 – Is the concept teachable?", y=1.03)
    return _save(fig, S, "Fig3_teachability",
                 "Student = early checkpoint of the same network that does not yet know the concept. A real concept helps only when the student is taught with concept "
                 "examples and tested on the concept (blue bar, left).\nIf concept-trained students also win on random positions the knowledge is generic, not concept-specific. Error bars: s.d. over concepts/seeds.")


# ---------------------------------------------------------------- Fig 4
def fig4_fidelity(S: Store):
    df = S.load("e4_explain")
    if df.empty: return None
    df = df[df.method == "trajconcept"]
    if df.empty: return None
    profs = [p for p in ("advisor", "calibrated") if p in set(df.profile)]
    fig, axs = plt.subplots(1, len(profs), figsize=(4.8 * len(profs), 3.3), squeeze=False)
    for ax, pf in zip(axs[0], profs):
        sub = df[df.profile == pf]; bbs = list(sub.bb.unique()); x = np.arange(len(bbs)); w = 0.26
        for j, (m, lab) in enumerate((("R", "R: share of trajectories the perturbation flips"), ("P", "P: precision of the map"), ("H", "H: harmonic mean"))):
            vals = [np.nanmean([sub[(sub.bb == b)][f"{p}_{m}"].astype(float).mean() for p in ("gauss", "scal", "rot")]) for b in bbs]
            ax.bar(x + (j - 1) * w, np.nan_to_num(vals, nan=0.0), w, color=PAL[j], label=lab)
            for xi, v in zip(x + (j - 1) * w, vals):
                if not np.isfinite(v): ax.text(xi, 0.01, "n/a", ha="center", fontsize=6, rotation=90, color="0.3")
        pr = [np.nanmean([sub[(sub.bb == b)][f"{p}_P_rand"].astype(float).mean() for p in ("gauss", "scal", "rot")]) for b in bbs]
        ax.plot(x, pr, "k_", ms=14, mew=2, label="P expected for a random map")
        ax.set_xticks(x); ax.set_xticklabels(bbs); ax.set_ylim(0, 1.0)
        ax.set_title({"advisor": "Perturbation strength: repository defaults", "calibrated": "Perturbation strength: calibrated (stronger)"}[pf], fontsize=8)
    axs[0, 0].legend(fontsize=6.5, loc="upper left")
    fig.suptitle("Fig 4 – Fidelity of the importance map φ^MU (mean over 3 segmentations x 3 perturbations)", y=1.03)
    return _save(fig, S, "Fig4_fidelity",
                 "Perturb the most / least important segments and see whether the prediction flips (advisor protocol). P above the black tick = better than a random map. "
                 "n/a = undefined because R = 0\n(the perturbation never changed the prediction, so there is nothing to explain; this is NOT a zero).")


# ---------------------------------------------------------------- Fig 5
def _draw_map(ax, X, masks, theta):
    th = np.asarray(theta, float); th = (th - th.min()) / (th.max() - th.min() + 1e-9); cmap = plt.get_cmap("coolwarm")
    for j, m in enumerate(masks):
        idx = np.where(m)[0]
        if not len(idx): continue
        pts = X[max(idx[0] - 1, 0): idx[-1] + 1]
        if len(pts) > 1: ax.plot(pts[:, 0], pts[:, 1], color=cmap(th[j]), lw=2.8)
    ax.plot(*X[0], "ko", ms=3)


def _draw_dense(ax, X, v):
    v = (v - v.min()) / (v.max() - v.min() + 1e-9); cmap = plt.get_cmap("coolwarm")
    for i in range(len(X) - 1):
        ax.plot(X[i:i + 2, 0], X[i:i + 2, 1], color=cmap(0.5 * (v[i] + v[i + 1])), lw=2.8)
    ax.plot(*X[0], "ko", ms=3)


def fig5_example(S: Store, k=3):
    fs = [x for x in sorted(glob.glob(os.path.join(S.art, "e4_explain__*_rdp_*.pkl")))]
    best = None
    for x in fs:
        d = pickle.load(open(x, "rb"))
        if d.get("dense") is None: continue
        if best is None or len(d["X"]) > len(best[1]["X"]): best = (x, d)
    if best is None: return None
    f, d = best; k = min(k, len(d["X"]))
    fig, axs = plt.subplots(k, 3, figsize=(7.6, 2.4 * k), squeeze=False)
    titles = ["Original (blue) vs. isometric twin\n(red dashed, same K)", "φ^MU per segment", "φ^MU per point"]
    for r in range(k):
        X = d["X"][r]
        axs[r, 0].plot(X[:, 0], X[:, 1], color=PAL[0], lw=2); axs[r, 0].plot(*X[0], "ko", ms=3)
        if d.get("twins") is not None and len(d["twins"]) > r:
            tw = d["twins"][r][0]; axs[r, 0].plot(tw[:, 0], tw[:, 1], "--", color=PAL[1], lw=1.6)
        _draw_map(axs[r, 1], X, [d["masks"][r][s] for s in range(len(d["masks"][r]))], d["theta"]["trajconcept"][r]) if d["theta"]["trajconcept"][r] is not None else None
        _draw_dense(axs[r, 2], X, d["dense"][r])
        for c in range(3):
            axs[r, c].set_aspect("equal"); axs[r, c].set_xticks([]); axs[r, c].set_yticks([]); axs[r, c].grid(False)
            if r == 0: axs[0, c].set_title(titles[c], fontsize=8)
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1)); sm.set_array([])
    cb = fig.colorbar(sm, ax=axs[:, 1:].ravel().tolist(), shrink=0.6, pad=0.02); cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"]); cb.set_label("machine-unique importance", fontsize=8)
    fig.suptitle("Fig 5 – Worked example", y=1.0)
    p = os.path.join(S.fig, "Fig5_example.png")
    fig.text(0.5, 0.0, "Black dot = start. The twin keeps all 28 expert-card features K but redraws the rest; the map shows where redrawing changes the model's output.\n"
             "'Per segment' is the form scored by Fig 4 (segments from RDP); 'per point' shows the same signal without segmentation.", ha="center", va="top", fontsize=7, color="0.3")
    if _preset(S) == "quick":
        fig.text(0.5, 1.09, "QUICK PRESET: smoke test only, not evidence", ha="center", va="bottom", fontsize=9, color="#c0392b", fontweight="bold")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); return p


def make_all(S: Store):
    out = {}
    for fn in (fig1_knowledge_gate, fig2_planted_answer, fig3_teachability, fig4_fidelity, fig5_example):
        try:
            out[fn.__name__] = fn(S)
        except Exception as e:
            out[fn.__name__] = f"ERROR {e!r}"
    return out
