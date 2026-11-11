"""Bang ket luan tu dong: 6 tuyen bo can chung minh, tieu chi dat/khong dat DAT TRUOC khi chay (xem THRESH).
Moi dong: PASS | FAIL | INCONCLUSIVE (chua du du lieu hoac black box khong qua cong) -> KHONG BAO GIO ep thanh PASS."""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.stats import spearmanr

THRESH = dict(
    alpha=0.05, fpr_max=0.10, n_null_min=12,           # C1
    power_min=0.80, power_b=4.0,                       # C2
    rec_min=0.50, rec_margin_ci_lo=0.0,                # C3
    adebayo_max=0.25,                                  # C4
    stable_frac=2 / 3,                                 # C5
    primary_method="trajconcept_n",                     # phuong phap CHINH cua C4/C6 (ban chuan hoa theo ADE, sinh ra tu kiem tra Adebayo o E8 quick); phuong phap con lai chi la INFO
    mass_ratio_min=1.2,                                # C6
)


def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def boot_ci(x, B=2000, seed=0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2: return (np.nan, np.nan, np.nan)
    r = np.random.RandomState(seed); m = np.array([x[r.randint(0, len(x), len(x))].mean() for _ in range(B)])
    return (float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def _row(c, crit, obs, n, v, note=""):
    return dict(claim=c, criterion=crit, observed=obs, n=n, verdict=v, note=note)


def evaluate(S, T=THRESH) -> pd.DataFrame:
    R = []
    nul = pd.concat([S.load("v_null"), S.load("v_worlds")[lambda d: d.get("b", pd.Series(dtype=float)) == 0] if not S.load("v_worlds").empty else pd.DataFrame()], ignore_index=True)
    if not nul.empty:
        nul = nul.drop_duplicates(["bb", "seed"])             # b=0 co the xuat hien o ca hai nguon; ho 'freq'/'where' trung nhau khi b=0
    W = S.load("v_worlds")
    # ---------------- C1: khong bao sai
    for bb in sorted(nul["bb"].unique()) if not nul.empty else []:
        d = nul[nul.bb == bb]; k = int(d["detect_ctrl"].sum()); n = len(d); lo, hi = wilson(k, n); fpr = k / n
        v = "INCONCLUSIVE" if n < T["n_null_min"] else ("FAIL" if (lo > T["alpha"] or fpr > T["fpr_max"]) else "PASS")
        R.append(_row(f"C1 sai duong CRT co kiem soat [{bb}]", f"FPR<={T['fpr_max']} va CI duoi<={T['alpha']}", f"{fpr:.3f} [{lo:.3f},{hi:.3f}]", n, v,
                      "that: 'detect_ctrl' tai b=0; ma bao sai cua CRT tho (thong ke) o duoi" ))
        raw = d["detect_raw"].mean(); R.append(_row(f"   (tham chieu) CRT THO [{bb}]", "khong co nguong (chi de so sanh)", f"{raw:.3f}", n, "INFO", "du kien >> alpha: ly do phai dung thong ke doi chung"))
    if nul.empty: R.append(_row("C1 sai duong", "", "n/a", 0, "INCONCLUSIVE", "chua chay"))
    # ---------------- C2: suc manh + dose-response
    if not W.empty:
        for (bb, fam), d in W.groupby(["bb", "family"]):
            hi_b = d[d.b >= T["power_b"]]
            if hi_b.empty: continue
            gate = d.groupby("seed")["detect_gate"].mean().mean()
            pw = hi_b["detect_crt"].mean(); n = len(hi_b)
            pos = d[d.b > 0]; rho = spearmanr(d["b"], d["delta_nll"])[0] if d["b"].nunique() > 2 else np.nan
            mu0 = d[d.b == 0]["detect_crt"].mean() if (d.b == 0).any() else np.nan
            v = "PASS" if (pw >= T["power_min"] and (np.isnan(rho) or rho > 0)) else ("INCONCLUSIVE" if n < 3 else "FAIL")
            R.append(_row(f"C2 suc manh [{bb}/{fam}]", f"power(b>={T['power_b']:.0f})>={T['power_min']} va delta tang theo b", f"power={pw:.2f}, rho(b,delta)={rho:.2f}, gap>0 (gate)={gate:.2f}",
                          n, v, f"power tai b=0 (sai bao)={mu0:.2f}"))
        # ---------------- C3: tim dung yeu to
        d = W[(W.b >= 2) & W["rec_all"].notna() & W["rec_rand"].notna()] if "rec_rand" in W else pd.DataFrame()
        if len(d):
            for fam, dd in d.groupby("family"):
                base = np.maximum(dd["rec_rand"], dd["rec_pca"]); m = (dd["rec_all"] - base).values
                mean, lo, hi = boot_ci(m); rec = dd["rec_all"].mean()
                v = "PASS" if (rec >= T["rec_min"] and lo > T["rec_margin_ci_lo"]) else ("INCONCLUSIVE" if len(dd) < 4 else "FAIL")
                R.append(_row(f"C3 phuc hoi yeu to cai san [{fam}]", f"|corr| >= {T['rec_min']} va hon baseline (ngau nhien/PCA) co CI>0",
                              f"rec={rec:.2f}; vuot baseline {mean:+.2f} [{lo:+.2f},{hi:+.2f}]", len(dd), v, "chi tinh cac o b>=2 (co tin hieu)"))
    # ---------------- C4: Adebayo
    e8 = S.load("e8_sanity")
    if not e8.empty and "spearman_full_random" in e8:
        for m, d in e8.groupby("method"):
            vals = {k: d[k].mean() for k in ("spearman_label_shuffle", "spearman_full_random")}; worst = max(vals.values())
            v = "INCONCLUSIVE" if not np.isfinite(worst) else ("PASS" if worst <= T["adebayo_max"] else "FAIL")
            note = "chi co y nghia neu mo hinh goc da train du (E1 usable)"
            if m != T["primary_method"]: note = f"INFO, se la {v}; " + note; v = "INFO"
            R.append(_row(f"C4 sanity Adebayo [{m}]", f"Spearman sau xao nhan / khoi tao lai <= {T['adebayo_max']}", ", ".join(f"{k.replace('spearman_','')}={x:.2f}" for k, x in vals.items()), len(d), v, note))
    else:
        R.append(_row("C4 sanity Adebayo", "", "n/a", 0, "INCONCLUSIVE", "chua chay e8 (hoac ban cu khong co cot full_random)"))
    # ---------------- C5: du lieu that
    e1, f5 = S.load("e1_gate"), S.load("e5_funnel")
    if not e1.empty and not f5.empty and "tier" in e1:
        g = e1[e1.tier == 2]
        for bb, d in g.groupby("bb"):
            seeds_gap = d[(d["delta_nll_lo"] > 0)]["seed"].tolist()
            if not seeds_gap:
                R.append(_row(f"C5 concept ngoai K tren du lieu that [{bb}]", "CI cua gap(f - g_K) > 0 roi concept qua loc o >=2/3 seed", f"khong seed nao co gap>0 (delta_lo<=0, median delta={d['delta_nll'].median():+.3f})",
                              len(d), "INCONCLUSIVE", "ket qua am hop le: K da du cho black box nay (khong co gi de tim)"))
                continue
            fd = f5[(f5.bb == bb) & (f5.seed.isin(seeds_gap))]
            ok = int((fd["funnel_all"] >= 1).sum()); n = len(seeds_gap)
            v = "PASS" if (n >= 2 and ok / n >= T["stable_frac"]) else ("INCONCLUSIVE" if n < 2 else "FAIL")
            R.append(_row(f"C5 concept ngoai K tren du lieu that [{bb}]", "qua het bo loc useful/novel/teachable/not_shortcut o >=2/3 seed co gap>0", f"{ok}/{n} seed", n, v, ""))
    else:
        R.append(_row("C5 concept ngoai K tren du lieu that", "", "n/a", 0, "INCONCLUSIVE", "chua chay e1/e5"))
    # ---------------- C6: fidelity
    e4 = S.load("e4_explain"); e4w = S.load("e4_worlds_local")
    if not e4.empty:
        for m in sorted(e4.method.unique()):
            d = e4[e4.method == m]; cols = [c for c in d.columns if c.endswith("_P_info_minus_rand") and not c.startswith("ctwin")]
            x = np.concatenate([d[c].dropna().values for c in cols]) if cols else np.array([])
            mean, lo, hi = boot_ci(x)
            mr = e4w[e4w.method == m]["relevant_mass_ratio"].mean() if not e4w.empty else np.nan
            v = "INCONCLUSIVE" if (not np.isfinite(lo)) else ("PASS" if (lo > 0 and (np.isnan(mr) or mr >= T["mass_ratio_min"])) else "FAIL")
            note = "P chi xac dinh khi R>0 (n/a khong tinh)"
            if m != T["primary_method"]: note = f"INFO, se la {v}; " + note; v = "INFO"
            R.append(_row(f"C6 fidelity (XAI4Traj la thuoc do) [{m}]", f"P_info - P_ngau_nhien co CI>0 va khoi luong dung >= {T['mass_ratio_min']} (world 'where')",
                          f"{mean:+.3f} [{lo:+.3f},{hi:+.3f}]; mass_ratio={mr:.2f}", len(x), v, note))
    else:
        R.append(_row("C6 fidelity", "", "n/a", 0, "INCONCLUSIVE", "chua chay e4"))
    return pd.DataFrame(R)


def summarize(df: pd.DataFrame) -> str:
    core = df[df.verdict.isin(["PASS", "FAIL", "INCONCLUSIVE"])]
    gist = []
    for c in ("C1", "C2", "C3", "C4", "C5", "C6"):
        v = core[core.claim.str.startswith(c)]["verdict"].tolist()
        if not v: gist.append(f"{c}: chua co du lieu"); continue
        gist.append(f"{c}: " + ("FAIL" if "FAIL" in v else ("PASS" if all(x == 'PASS' for x in v) else ("PASS mot phan" if "PASS" in v else "INCONCLUSIVE"))))
    return " | ".join(gist)
