"""Cac thi nghiem E1-E10, E15 (xem TrajConcept_de_xuat_day_du.md muc 8). E11-E12 la hinh (plots.py).
KHONG co ma cho E13 (thi diem voi nguoi) va E14 (bao + moi truong): can quy trinh dao duc / du lieu ngoai."""
from __future__ import annotations
import time, dataclasses, traceback
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .config import Config, DEEP
from .data import load_dataset, TrajSet
from .pipeline import (get_cell, Cell, mu_and_crt, twin_quality, twin_leak, run_concepts, boot_ci)
from .models import DeepClassifier
from .knowledge import compute_K, tier_cols, ALL_NAMES, K0, K1, K2
from . import explain as ex
from . import concepts as cc
from . import worlds as W
from .local_metrics import leak_overlap, relevant_mass_ratio, iou_topk
from .store import Store

_DS_CACHE = {}


def get_ds(name, cfg, seed=0):
    k = (name, cfg.max_n, seed)
    if k not in _DS_CACHE:
        _DS_CACHE[k] = load_dataset(name, cfg.max_n, cfg.max_len_raw, seed=0)
    return _DS_CACHE[k]


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:
        _log("LOI:", fn.__name__, repr(e)); traceback.print_exc(); return None


def _cap(idx, n, seed=0):
    return idx if len(idx) <= n else np.sort(np.random.RandomState(seed).choice(idx, n, replace=False))


# ===================================================================== E1: cong khoang cach tri thuc
def e1_gate(cfg: Config, S: Store):
    for ds in cfg.datasets:
        ts = get_ds(ds, cfg)
        for bb in cfg.black_boxes:
            for seed in cfg.seeds:
                if S.has("e1_gate", dataset=ds, bb=bb, seed=seed):
                    continue
                cell = _safe(get_cell, ts, bb, seed, cfg)
                if cell is None:
                    continue
                chk = cell.bb_check(); rows = []
                for g in cell.gate():
                    rows.append(dict(dataset=ds, bb=bb, seed=seed, n_classes=ts.n_classes, n_ho=len(cell.ho), **chk, **g))
                S.add("e1_gate", rows)
                _log("E1", ds, bb, seed, {k: round(v, 3) if isinstance(v, float) else v for k, v in chk.items()})
    df = S.load("e1_gate")
    if not df.empty:
        agg = df.groupby(["dataset", "bb", "tier"]).agg(
            usable=("usable", "mean"), bacc_f=("bacc_f", "mean"), bacc_g=("bacc_g", "mean"), S_f=("S_f_nll", "mean"),
            S_g=("S_g_nll", "mean"), delta_nll=("delta_nll", "mean"), delta_sd=("delta_nll", "std"),
            ci_lo=("delta_nll_lo", "mean"), ci_hi=("delta_nll_hi", "mean"), known_ratio=("known_ratio", "mean"),
            n_seeds=("seed", "nunique")).reset_index()
        agg["pass_gate"] = (agg["ci_lo"] > 0) & (agg["tier"] == 2)
        S.save_table("table1_gate", agg, "Table 1 - Knowledge gap (log-loss; delta_nll > 0 means the black box beats the K-only model)")
    return df


# ===================================================================== E2: chat luong twin
def e2_twin_quality(cfg: Config, S: Store, gens=("gauss", "flow", "naive"), n_eval=250):
    for ds in cfg.datasets:
        ts = get_ds(ds, cfg)
        for seed in cfg.seeds:
            c = Cell(ts, "rf", seed, cfg); c.split()
            idx = _cap(c.fin, n_eval, seed)
            for g in gens:
                if S.has("e2_twin", dataset=ds, gen=g, seed=seed):
                    continue
                def run():
                    old = cfg.twin_gen
                    gen = c.build_twin_gen(kind=g)
                    tw, eps, acc = c.make_twins(gen, idx, B=min(cfg.n_twin, 8), pool=cfg.twin_pool)
                    q = twin_quality(c, gen, ts.X[idx], tw, eps, acc)
                    q["leak_gK"] = twin_leak(c, ts.X[idx], ts.y[idx], tw, cfg)
                    q["usable"] = bool(q["C2ST"] <= cfg.twin_c2st_ok and q["eps_K_median"] <= cfg.twin_eps_ok and q["valid_frac"] >= 0.95)
                    S.add("e2_twin", dict(dataset=ds, gen=g, seed=seed, n=len(idx), **q))
                    _log("E2", ds, g, seed, {k: round(v, 3) if isinstance(v, float) else v for k, v in q.items()})
                _safe(run)
    df = S.load("e2_twin")
    if not df.empty:
        S.save_table("table2_twin", df.groupby(["dataset", "gen"]).mean(numeric_only=True).drop(columns=["seed"]).reset_index(),
                     "Table 2 - Twin quality (C2ST close to 0.5 and small eps_K are good; leak_gK = K-only model loss caused by the twin)")
    return df


# ===================================================================== tien ich cho the gioi / thuc
def _concept_corr_stats(cell, out, h_star, t_star, idx_eval):
    """Rec(h*), precision khai niem, sai so vi tri tren idx_eval (tap kiem tra cuoi)."""
    if not out.get("concepts"):
        return dict(rec_all=0.0, rec_pass=0.0, concept_precision=float("nan"), loc_err=float("nan"))
    lat = out["lat"]; X = cell.ts.X[idx_eval]
    Z = lat(X); t0 = int(cell.cfg.t_min_frac * Z.shape[1])
    cors, locs, passes = [], [], []
    for rec in out["concepts"]:
        s = cc.concept_score(Z, rec["v"], "max", t0)
        cr = np.corrcoef(s, h_star[idx_eval])[0, 1] if s.std() > 0 else 0.0
        cors.append(abs(cr)); passes.append(bool(rec.get("teachable") and rec.get("not_shortcut")) or bool(rec["useful"] and rec["novel"]))
        if (t_star[idx_eval] >= 0).all():
            th = (Z[:, t0:, :] @ rec["v"]).argmax(1) + t0
            locs.append(np.abs(th - t_star[idx_eval]).mean() / X.shape[1])
        rec["corr_h"] = float(cr)
    cors = np.array(cors); passes = np.array(passes)
    return dict(rec_all=float(cors.max()), rec_pass=float(cors[passes].max()) if passes.any() else 0.0,
                concept_precision=float((cors >= 0.5).mean()), loc_err=float(min(locs)) if locs else float("nan"))


def _run_world_cell(cfg, S, ds, fam, b, seed, bb="gru", gen_kind=None, tag="e3_worlds", extra=None, run_loc=False,
                    source="both", norm="l1", layer=-1, n_init=None):
    ts0 = get_ds(ds, cfg)
    y, info = W.make_world(ts0, fam, b, seed)
    if len(np.unique(y)) < 2 or np.bincount(y).min() < 12:
        _log("world bi suy bien", ds, fam, b); return None
    ts = dataclasses.replace(ts0, name=f"{ds}-W{fam}-b{b}", y=y, classes=list(range(len(np.unique(y)))))
    remap = {c: i for i, c in enumerate(np.unique(y))}; ts.y = np.array([remap[v] for v in y]); ts.classes = list(range(len(remap)))
    cell = get_cell(ts, bb, seed, cfg, verbose=False)
    gate = cell.gate()[2]
    gen = cell.build_twin_gen(kind=gen_kind)
    idx_ho = _cap(cell.ho, 400, seed)
    tw, eps, acc = cell.make_twins(gen, idx_ho, B=cfg.n_twin, pool=cfg.twin_pool)
    det = mu_and_crt(cell, ts.X[idx_ho], ts.y[idx_ho], tw, cfg.alpha)
    row = dict(dataset=ds, family=fam, b=b, seed=seed, bb=bb, valid_world=info["valid"], r2_h_K=info["r2_h_given_K_after"],
               bacc=float(gate["bacc_f"]), delta_nll=gate["delta_nll"], delta_lo=gate["delta_nll_lo"], delta_hi=gate["delta_nll_hi"],
               MU=det["MU_share"], p_crt=det["p_value"], p_raw=det["p_raw"], leak_gK=det["leak_gK"],
               twin_ok=bool(np.median(eps[:, 0]) <= cfg.twin_eps_ok),
               detect_crt=bool(det["reject"] and np.median(eps[:, 0]) <= cfg.twin_eps_ok), detect_ctrl=bool(det["reject"]),
               detect_raw=bool(det["reject_raw"]), detect_gate=bool(gate["delta_nll_lo"] > 0),
               eps_K=float(np.median(eps[:, 0])))
    # khai niem (chi DeepClassifier)
    if isinstance(cell.f, DeepClassifier):
        idx_d = _cap(cell.disc, 200, seed)
        tw_d, _, _ = cell.make_twins(gen, idx_d, B=min(cfg.n_twin, 12), pool=cfg.twin_pool)
        out = run_concepts(cell, gen, tw_d, idx_d, cfg, tier=2, source=source, norm=norm, layer=layer, n_init=n_init)
        row.update({f"funnel_{k}": v for k, v in out["funnel"].items()})
        row.update({f"lp_{k}": v for k, v in out["lp"].items() if isinstance(v, (int, float))})
        if "twin" in out["lp"]:
            row["lp_twin_groups"] = out["lp"]["twin"].get("n_groups")
        row.update(_concept_corr_stats(cell, out, info["h_star"], info["t_star"], cell.fin))
        row["n_concepts"] = len(out["concepts"])
        S.save_art(f"{tag}__{ds}_{fam}_b{b}_s{seed}", dict(dg=det["dg"], rec=[{k: v for k, v in r.items() if k not in ("teach_curve",)}
                                                                              for r in out["concepts"]]))
    if extra:
        row.update(extra)
    return row, cell, gen, info


# ===================================================================== E3: the gioi cai dap an
def e3_worlds(cfg: Config, S: Store, bbs=("gru",)):
    for ds in cfg.worlds_datasets:
        for fam in cfg.worlds_families:
            for b in cfg.worlds_b:
                for seed in cfg.seeds:
                    for bb in bbs:
                        if S.has("e3_worlds", dataset=ds, family=fam, b=b, seed=seed, bb=bb):
                            continue
                        r = _safe(_run_world_cell, cfg, S, ds, fam, b, seed, bb)
                        if r:
                            S.add("e3_worlds", r[0]); _log("E3", ds, fam, b, seed, bb,
                                                           {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r[0].items()
                                                            if k in ("bacc", "delta_nll", "p_crt", "detect_crt", "rec_all", "funnel_candidates", "funnel_all")})
    df = S.load("e3_worlds")
    if not df.empty:
        agg = df.groupby(["dataset", "family", "b"]).agg(
            power_crt=("detect_crt", "mean"), power_ctrl=("detect_ctrl", "mean"), power_raw=("detect_raw", "mean"), power_gate=("detect_gate", "mean"),
            twin_ok=("twin_ok", "mean"), leak=("leak_gK", "mean"), rec_h=("rec_all", "mean"),
            concept_prec=("concept_precision", "mean"), loc_err=("loc_err", "mean"), MU=("MU", "mean"),
            cand=("funnel_candidates", "mean"), final=("funnel_all", "mean"), n=("seed", "count")).reset_index()
        S.save_table("table3_worlds", agg, "Table 3 - Planted-answer worlds: detection rate (b>0 power, b=0 false-alarm rate) and recovery of the planted factor h*")
    return df


# ===================================================================== E4: ban do doan phi^MU theo giao thuc Fidelity (kieu Bang 2-8)
PERT = [("noise", "gaussian"), ("rotate", "rotation"), ("scale", "scaling")]


def e4_explain(cfg: Config, S: Store, bbs=None, tag="e4_explain"):
    """Voi MOI bo du lieu x black box x cach chia doan {mdl, rdp, window}: R/P/H (Eq.7-10 cua XAI4Traj, dung lam THUOC DO) cua ban do doan
    phi^MU duoi 3 phep nhieu danh gia {gaussian, rotation, scaling}, hai profile tham so nhieu ('advisor', 'calibrated').
    Chi co TrajConcept: khong phuong phap giai thich nao khac."""
    bbs = bbs or cfg.black_boxes
    for ds in cfg.datasets:
        ts = get_ds(ds, cfg)
        for bb in bbs:
            for seed in cfg.e4_seeds:
                todo = [(pf, sg) for pf in cfg.ops_profiles for sg in cfg.segs if not S.has(tag, dataset=ds, bb=bb, seed=seed, seg=sg, profile=pf)]
                if not todo:
                    continue
                def run():
                    cell = get_cell(ts, bb, seed, cfg, verbose=False)
                    gen = cell.build_twin_gen()
                    idx = _cap(cell.fin, cfg.e4_n or cfg.n_explain, seed)
                    tw, eps, _ = cell.make_twins(gen, idx, B=8, pool=cfg.twin_pool)
                    Xs = ts.X[idx]; chk = cell.bb_check()
                    phi_cache = {}
                    for pf, sg in todo:
                        org = ts.origin[idx] if ts.origin is not None else None
                        cfg_fn = (lambda i, pf=pf, org=org: cfg.ops_for(ts.scale, pf, None if org is None else org[i]))
                        cfg_o = cfg_fn(0)
                        t0 = time.time()
                        if sg not in phi_cache:                      # phi^MU khong phu thuoc profile nhieu
                            phi_cache[sg] = ex.run_explainers(cell.f, gen, tw, Xs, cfg_o, sg, ["trajconcept", "trajconcept_n"], seed, cfg_fn=cfg_fn)
                        th, masks, c0 = phi_cache[sg]
                        t_expl = time.time() - t0
                        extra = (pf == "calibrated")
                        rows, fl = ex.evaluate_fidelity(cell.f, gen, tw, Xs, masks, c0, th, cfg_o, seed, common=extra, auc=extra, cfg_fn=cfg_fn)
                        for m, r in rows.items():
                            r.update(dataset=ds, bb=bb, seed=seed, seg=sg, profile=pf, method=m, usable=chk["usable"], n_traj=len(idx),
                                     mean_T=float(np.mean([len(x) for x in masks])), time_per_traj=t_expl / len(idx))
                            r["leak_overlap"], r["leak_spearman"] = _leak(cell, Xs, masks, th[m])
                            S.add(tag, r)
                        dense = [ex.phi_mu_dense(Xs[i], cell.f.proba, c0[i], gen, tw[i], cfg_o) for i in range(min(12, len(Xs)))] if (sg == "rdp" and pf == todo[0][0]) else None
                        S.save_art(f"{tag}__{ds}_{bb}_s{seed}_{sg}_{pf}", dict(X=Xs[:12], masks=[masks[i] for i in range(min(12, len(masks)))],
                                                                           theta={m: th[m][:12] for m in th}, c0=c0[:12], twins=tw[:3], dense=dense))
                        _log("E4", ds, bb, pf, sg, {ex.OPNAME[o]: (None if not np.isfinite(rows["trajconcept"].get(f"{ex.OPNAME[o]}_H", np.nan))
                                                                  else round(rows["trajconcept"][f"{ex.OPNAME[o]}_H"], 2)) for o, _ in PERT})
                _safe(run)
    df = S.load(tag)
    if not df.empty:
        _e4_tables(df, S)
    return df


def _e4_tables(df, S):
    """Bang dang paper: hang = (chia doan, phep nhieu), cot = black box x (R, P, H); mot bang cho moi (bo du lieu, profile). Cung bang tong hop."""
    summ = []
    for (ds, pf), g in df.groupby(["dataset", "profile"]):
        rows = []
        for sg in [x for x in ("mdl", "rdp", "window") if x in set(g.seg)]:
            for op, opn in PERT:
                sub = g[(g.seg == sg) & (g.method == "trajconcept")]
                row = {"seg": sg, "perturb": opn}
                for _, r in sub.iterrows():
                    for k in ("R", "P", "H"):
                        row[f"{r.bb}_{k}"] = r.get(f"{ex.OPNAME[op]}_{k}", np.nan)
                rows.append(row)
        if rows:
            S.save_table(f"table4_{ds}_{pf}", pd.DataFrame(rows), f"Table 4 ({ds}, perturbation profile {pf}) - Fidelity R/P/H of phi^MU. n/a = undefined because R=0")
        for mk, label in (("trajconcept", "phi^MU"), ("trajconcept_n", "phi^MU normalised")):
            for bb, gb in g.groupby("bb"):
                vals = {"R": [], "P": [], "H": [], "Pinfo_minus_rand": [], "frac_informative": []}
                for op, _ in PERT:
                    for _, r in gb[gb.method == mk].iterrows():
                        o = ex.OPNAME[op]
                        for k, kk in (("R", "R"), ("P", "P"), ("H", "H"), ("Pinfo_minus_rand", "P_info_minus_rand"), ("frac_informative", "frac_informative")):
                            vals[k].append(r.get(f"{o}_{kk}", np.nan))
                summ.append(dict(dataset=ds, profile=pf, bb=bb, method=label,
                                 **{k: float(np.nanmean(v)) if np.isfinite(v).any() else np.nan for k, v in vals.items()}))
    if summ:
        sm = pd.DataFrame(summ)
        for pf, g in sm.groupby("profile"):
            S.save_table(f"table4_summary_{pf}", g.drop(columns="profile"),
                         f"Table 4 summary (profile {pf}): mean over 3 segmentations x 3 perturbations. Pinfo_minus_rand = precision above a random map; n/a = undefined because R=0")


def _leak(cell, Xs, masks_list, thetas):
    """Do chong lap voi 'tri thuc cu': kappa_j = |logit g_K(X) - logit g_K(X voi doan j bi xoa duong thang)|.
    leak_overlap = sum_j phi^_j kappa^_j (phi^, kappa^ chuan hoa tong = 1; mac dinh deu = 1/T) ; leak_spearman = Spearman(phi, kappa)."""
    gK = cell.gK[2]; cols = tier_cols(2); ov, sp = [], []
    for i, X in enumerate(Xs):
        th = thetas[i]
        if th is None:
            continue
        masks = [masks_list[i][j] for j in range(len(masks_list[i]))]
        Xm = np.stack([X] + [ex.apply_op("straight", X, m, cell.cfg, np.random.RandomState(0)) for m in masks])
        P = gK.proba(compute_K(Xm)[:, cols])
        c = cell.f.predict(X[None].astype(np.float32))[0]
        lo = np.log(np.clip(P[:, c], 1e-6, 1 - 1e-6)) - np.log(1 - np.clip(P[:, c], 1e-6, 1 - 1e-6))
        kap = np.abs(lo[1:] - lo[0]); phi = np.clip(th, 0, None)
        o = leak_overlap(phi, kap)
        if o is not None:
            ov.append(o)                                                               # chuan hoa de 1 = muc ngau nhien
            if len(phi) > 2 and phi.std() > 0 and kap.std() > 0:
                sp.append(spearmanr(phi, kap)[0])
    return (float(np.mean(ov)) if ov else float("nan")), (float(np.mean(sp)) if sp else float("nan"))


def e4_worlds_local(cfg: Config, S: Store, ds=None, fam="where", b=None, bb="gru"):
    """Khoi luong dung / IoU / ro sang K tren the gioi 'where' (co vi tri that t*)."""
    ds = ds or cfg.worlds_datasets[0]; b = b if b is not None else max(cfg.worlds_b)
    for seed in cfg.seeds:
        if S.has("e4_worlds_local", dataset=ds, seed=seed, bb=bb):
            continue
        def run():
            r = _run_world_cell(cfg, S, ds, fam, b, seed, bb, tag="e4w")
            if not r: return
            row, cell, gen, info = r
            idx = _cap(cell.fin, cfg.n_explain, seed)
            tw, _, _ = cell.make_twins(gen, idx, B=8, pool=cfg.twin_pool)
            Xs = cell.ts.X[idx]
            methods = ["trajconcept", "trajconcept_n"]
            org = cell.ts.origin[idx] if cell.ts.origin is not None else None
            cfg_fn = (lambda i: cfg.ops_for(cell.ts.scale, None, None if org is None else org[i]))
            cfg_o = cfg_fn(0)
            th, masks, c0 = ex.run_explainers(cell.f, gen, tw, Xs, cfg_o, "rdp", methods, seed, cfg_fn=cfg_fn)
            tstar = info["t_star"][idx]
            for m in methods:
                rel, iou, T_ = [], [], []
                for i in range(len(Xs)):
                    if th[m][i] is None: continue
                    M = masks[i]
                    r_, u_ = relevant_mass_ratio(th[m][i], M, tstar[i]), iou_topk(th[m][i], M, tstar[i])
                    if r_ is None: continue
                    rel.append(r_); iou.append(u_)                                             # rel > 1 = hon muc ngau nhien
                lk = _leak(cell, Xs, masks, th[m])
                S.add("e4_worlds_local", dict(dataset=ds, family=fam, b=b, seed=seed, bb=bb, method=m,
                                              relevant_mass_ratio=float(np.mean(rel)), iou_topk=float(np.mean(iou)),
                                              leak_overlap=lk[0], n=len(rel)))
        _safe(run)
    df = S.load("e4_worlds_local")
    if not df.empty:
        S.save_table("table4w_local", df.groupby(["method"]).mean(numeric_only=True).reset_index(),
                     "Table 4w - Localising the planted segment ('where' world): relevant mass ratio (1 = chance), IoU, overlap with K-only sensitivity")
    return df


# ===================================================================== E5/E6: khai niem tren du lieu that + hoc tro
def e5_concepts(cfg: Config, S: Store, bbs=None, only_deep=True):
    bbs = [b for b in (bbs or cfg.black_boxes) if (b in DEEP or not only_deep)]
    for ds in cfg.datasets:
        ts = get_ds(ds, cfg)
        for bb in bbs:
            for seed in cfg.seeds:
                if S.has("e5_funnel", dataset=ds, bb=bb, seed=seed):
                    continue
                def run():
                    cell = get_cell(ts, bb, seed, cfg, verbose=False)
                    gen = cell.build_twin_gen()
                    idx = _cap(cell.disc, 250, seed)
                    tw, eps, _ = cell.make_twins(gen, idx, B=min(cfg.n_twin, 12), pool=cfg.twin_pool)
                    t0 = time.time()
                    out = run_concepts(cell, gen, tw, idx, cfg, tier=2)
                    g = cell.gate()[2]
                    chk = cell.bb_check()
                    row = dict(dataset=ds, bb=bb, seed=seed, usable=chk["usable"], delta_R=g["delta_nll"], delta_R_lo=g["delta_nll_lo"],
                               gate_pass=bool(g["delta_nll_lo"] > 0), time_s=time.time() - t0,
                               **{f"funnel_{k}": v for k, v in out["funnel"].items()},
                               **{f"lp_{k}": v for k, v in out["lp"].items() if isinstance(v, (int, float))})
                    S.add("e5_funnel", row)
                    recs = []
                    pats = []
                    for r in out.get("concepts", []):
                        recs.append({k: (v if not isinstance(v, np.ndarray) else None) for k, v in r.items()
                                     if k not in ("v", "teach_curve", "teach_rand_eval", "teach4", "teach_info", "dose")})
                        recs[-1].update(dataset=ds, bb=bb, seed=seed)
                        if r.get("teach4"):
                            recs[-1].update({f"teach4_{k}": v for k, v in r["teach4"].items()})
                            recs[-1].update(agree_pre=r["teach_info"]["agree_pre"], agree_max=r["teach_info"]["agree_max"])
                        dz = r.get("dose")
                        if dz and "rho" in dz:
                            recs[-1].update(dose_rho=dz["rho"], dose_effect=dz["effect"], dose_kdrift_max=max(dz["k_drift"]),
                                            dose_rho_ctrl=dz.get("rho_ctrl"), dose_effect_ctrl=dz.get("effect_ctrl"), dose_kdrift_ctrl_max=max(dz["k_drift_ctrl"]) if "k_drift_ctrl" in dz else None)
                        if r.get("teach_rand_eval"):
                            for cond, d in r["teach_rand_eval"].items():
                                for n, vals in d.items():
                                    S.add("e6_teach", dict(dataset=ds, bb=bb, seed=seed, concept=r["id"], cond=cond + "@random_pos", n=n,
                                                           agree=float(np.mean(vals)), agree_sd=float(np.std(vals))))
                        if r.get("teach_curve"):
                            for cond, d in r["teach_curve"].items():
                                for n, vals in d.items():
                                    S.add("e6_teach", dict(dataset=ds, bb=bb, seed=seed, concept=r["id"], cond=cond, n=n, agree=float(np.mean(vals)),
                                                           agree_sd=float(np.std(vals))))
                        pats.append(cc.concept_pattern(r["v"], out["lat"], ts.X[cell.conf], cfg))
                    if recs: S.add("e5_concepts", recs)
                    S.save_art(f"e5__{ds}_{bb}_s{seed}", dict(patterns=[{k: v for k, v in p.items() if k != "score"} for p in pats],
                                                           recs=recs, X=ts.X[cell.conf][:8]))
                    _log("E5", ds, bb, seed, out["funnel"])
                _safe(run)
    return S.load("e5_funnel")


# ===================================================================== E7: tai phat hien
def e7_rediscovery(cfg: Config, S: Store, bbs=("gru",)):
    drops = [(1, K2, "K2 (SAMO+dong hoc)"), (0, K1, "K1 (sinh thai/Zheng)")]
    for ds in cfg.datasets:
        ts = get_ds(ds, cfg)
        for bb in bbs:
            for seed in cfg.seeds:
                for r, dropped_names, label in drops:
                    if S.has("e7_rediscovery", dataset=ds, bb=bb, seed=seed, kept_tier=r):
                        continue
                    def run():
                        cell = get_cell(ts, bb, seed, cfg, verbose=False)
                        if not isinstance(cell.f, DeepClassifier): return
                        cols = tier_cols(r)
                        gen = cell.build_twin_gen(k_cols=cols)
                        idx = _cap(cell.disc, 200, seed)
                        tw, _, _ = cell.make_twins(gen, idx, B=min(cfg.n_twin, 12), pool=cfg.twin_pool)
                        out = run_concepts(cell, gen, tw, idx, cfg, tier=r)
                        Kc = cell.Ks[cell.conf]
                        dropped_idx = [ALL_NAMES.index(n) for n in dropped_names]
                        rng = np.random.RandomState(seed)
                        junk = rng.randn(len(cell.conf))
                        best_c, best_j, junk_c = 0.0, "-", 0.0
                        if out.get("concepts"):
                            Z = out["lat"](ts.X[cell.conf]); t0 = int(cfg.t_min_frac * Z.shape[1])
                            for rec in out["concepts"]:
                                s = cc.concept_score(Z, rec["v"], "max", t0)
                                if s.std() == 0: continue
                                for j in dropped_idx:
                                    cr = abs(np.corrcoef(s, Kc[:, j])[0, 1]) if Kc[:, j].std() > 0 else 0
                                    if cr > best_c: best_c, best_j = cr, ALL_NAMES[j]
                                junk_c = max(junk_c, abs(np.corrcoef(s, junk)[0, 1]))
                        gate_r = cell.gate()
                        S.add("e7_rediscovery", dict(dataset=ds, bb=bb, seed=seed, kept_tier=r, dropped=label,
                                                     delta_extra=gate_r[r]["delta_nll"] - gate_r[2]["delta_nll"] if len(gate_r) > 2 else float("nan"),
                                                     n_concepts=len(out.get("concepts", [])), best_corr=best_c, best_item=best_j,
                                                     junk_corr=junk_c, funnel_all=out["funnel"].get("all", 0),
                                                     n_dropped_items=len(dropped_idx)))
                        _log("E7", ds, seed, label, round(best_c, 3), best_j, "junk", round(junk_c, 3))
                    _safe(run)
    df = S.load("e7_rediscovery")
    if not df.empty:
        S.save_table("table7_rediscovery", df.groupby(["dataset", "dropped"]).mean(numeric_only=True).reset_index().drop(columns=["seed", "kept_tier"], errors="ignore"),
                     "Table 7 - Rediscovery: |corr| of the best concept with a K item that was removed (junk = random control)")
    return df


# ===================================================================== E8: kiem tra hop le
def e8_sanity(cfg: Config, S: Store, bb="gru", n_traj=40):
    for ds in cfg.datasets:
        ts = get_ds(ds, cfg)
        for seed in cfg.seeds:
            if S.has("e8_sanity", dataset=ds, seed=seed):
                continue
            def run():
                cell = get_cell(ts, bb, seed, cfg, verbose=False)
                if not isinstance(cell.f, DeepClassifier): return
                gen = cell.build_twin_gen()
                idx = _cap(cell.fin, n_traj, seed)
                tw, _, _ = cell.make_twins(gen, idx, B=8, pool=cfg.twin_pool)
                Xs = ts.X[idx]; org = ts.origin[idx] if ts.origin is not None else None
                cfg_fn = (lambda i: cfg.ops_for(ts.scale, None, None if org is None else org[i])); cfg_o = cfg_fn(0)
                th_real, masks, _ = ex.run_explainers(cell.f, gen, tw, Xs, cfg_o, "rdp", ["trajconcept", "trajconcept_n"], seed, cfg_fn=cfg_fn)
                # (i) xao nhan
                rng = np.random.RandomState(seed + 3)
                ts_s = dataclasses.replace(ts, name=ts.name + "-shuf", y=rng.permutation(ts.y))
                cs = get_cell(ts_s, bb, seed, cfg, verbose=False)
                th_s, _, _ = ex.run_explainers(cs.f, gen, tw, Xs, cfg_o, "rdp", ["trajconcept", "trajconcept_n"], seed, cfg_fn=cfg_fn)
                # (ii) xao trong so lop cuoi
                import copy, torch
                f2 = copy.copy(cell.f); f2.net = copy.deepcopy(cell.f.net)
                for p_ in list(f2.net.head.parameters()):
                    torch.nn.init.normal_(p_, std=0.5) if p_.dim() > 1 else torch.nn.init.zeros_(p_)
                f2.T = 1.0
                th_w, _, _ = ex.run_explainers(f2, gen, tw, Xs, cfg_o, "rdp", ["trajconcept", "trajconcept_n"], seed, cfg_fn=cfg_fn)
                def sp(a, b_):
                    v = [spearmanr(x, y_)[0] for x, y_ in zip(a, b_) if x is not None and y_ is not None and len(x) > 2 and np.std(x) > 0 and np.std(y_) > 0]
                    return float(np.nanmean(v)) if v else float("nan")
                idx_ho = _cap(cs.ho, 300, seed); tw2, _, _ = cs.make_twins(gen, idx_ho, B=cfg.n_twin, pool=cfg.twin_pool)
                crt_s = mu_and_crt(cs, ts.X[idx_ho], ts_s.y[idx_ho], tw2, cfg.alpha)
                for m in ("trajconcept", "trajconcept_n"):
                    S.add("e8_sanity", dict(dataset=ds, seed=seed, bb=bb, method=m, spearman_label_shuffle=sp(th_real[m], th_s[m]),
                                            spearman_weight_random=sp(th_real[m], th_w[m]), crt_p_shuffled=crt_s["p_value"],
                                            crt_reject_shuffled=crt_s["reject"], acc_shuffled=float((cs.pf(ts.X[cs.ho]).argmax(1) == ts_s.y[cs.ho]).mean())))
                _log("E8", ds, seed, "ok")
            _safe(run)
    df = S.load("e8_sanity")
    if not df.empty:
        S.save_table("table8_sanity", df.groupby(["dataset", "method"]).mean(numeric_only=True).reset_index().drop(columns=["seed"], errors="ignore"),
                     "Table 8 - Sanity checks (Spearman near 0 is good; the CRT on the label-shuffled model must NOT reject)")
    return df


# ===================================================================== E9: ablation (tren the gioi cai san 'freq', b lon)
def e9_ablation(cfg: Config, S: Store, bb="gru"):
    ds = cfg.worlds_datasets[0]; fam = "freq"; b = max(cfg.worlds_b)
    variants = [("gen", "gauss", dict(gen_kind="gauss")), ("gen", "flow", dict(gen_kind="flow")), ("gen", "naive", dict(gen_kind="naive")),
                ("source", "twin", dict(source="twin")), ("source", "disagree", dict(source="disagree")), ("source", "both", dict(source="both")),
                ("extractor", "L1", dict(norm="l1")), ("extractor", "L2", dict(norm="l2")),
                ("layer", "last", dict(layer=-1)), ("layer", "first", dict(layer=0))]
    for seed in cfg.seeds:
        for grp, name, kw in variants:
            if S.has("e9_ablation", group=grp, variant=name, seed=seed):
                continue
            def run():
                r = _run_world_cell(cfg, S, ds, fam, b, seed, bb, tag="e9", **kw)
                if r:
                    row = r[0]
                    S.add("e9_ablation", dict(group=grp, variant=name, seed=seed, dataset=ds, p_crt=row["p_crt"], detect=row["detect_crt"], MU=row["MU"],
                                              candidates=row.get("funnel_candidates"), final=row.get("funnel_all"),
                                              rec_h=row.get("rec_all"), eps_K=row["eps_K"], feasible=row.get("lp_feasible_rate"),
                                              useful=row.get("funnel_useful")))
                    _log("E9", grp, name, seed, round(row.get("rec_all", float("nan")), 3))
            _safe(run)
    df = S.load("e9_ablation")
    if not df.empty:
        S.save_table("table9_ablation", df.groupby(["group", "variant"]).mean(numeric_only=True).reset_index().drop(columns=["seed"], errors="ignore"),
                     f"Table 9 - Ablation on the planted world ({ds}/freq/b={b}); rec_h = |corr| of the best concept with h*")
    return df


# ===================================================================== E10: do nhay
def e10_sensitivity(cfg: Config, S: Store, bb="gru"):
    ds = cfg.worlds_datasets[0]; fam = "freq"; b = max(cfg.worlds_b)
    for seed in cfg.seeds:
        if S.has("e10_sens", seed=seed, kind="B"):
            continue
        def run():
            ts0 = get_ds(ds, cfg); y, info = W.make_world(ts0, fam, b, seed)
            ts = dataclasses.replace(ts0, name=f"{ds}-W{fam}-b{b}", y=y)
            remap = {c: i for i, c in enumerate(np.unique(y))}; ts.y = np.array([remap[v] for v in y]); ts.classes = list(range(len(remap)))
            cell = get_cell(ts, bb, seed, cfg, verbose=False)
            idx = _cap(cell.ho, 300, seed)
            for kind in ("flow",):
                gen = cell.build_twin_gen(kind=kind)
                tw, eps, _ = cell.make_twins(gen, idx, B=max(cfg.n_twin, 32), pool=max(cfg.twin_pool, 64))
                for B in (8, 16, 32):
                    if tw.shape[1] < B: continue
                    d = mu_and_crt(cell, ts.X[idx], ts.y[idx], tw[:, :B], cfg.alpha)
                    S.add("e10_sens", dict(seed=seed, kind="B", value=B, MU=d["MU_share"], p_crt=d["p_value"], Delta_CRT=d["Delta_CRT"]))
            # tang K dung cho dieu kien hoa twin
            for r in (0, 1, 2):
                gen = cell.build_twin_gen(k_cols=tier_cols(r))
                tw, eps, _ = cell.make_twins(gen, idx, B=cfg.n_twin, pool=cfg.twin_pool)
                d = mu_and_crt(cell, ts.X[idx], ts.y[idx], tw, cfg.alpha)
                S.add("e10_sens", dict(seed=seed, kind="K_tier", value=r, MU=d["MU_share"], p_crt=d["p_value"], Delta_CRT=d["Delta_CRT"],
                                       eps_K=float(np.median(eps[:, 0]))))
            # cach chia doan cho giai thich cuc bo
            gen = cell.build_twin_gen()
            idx2 = _cap(cell.fin, max(20, cfg.n_explain // 3), seed)
            tw2, _, _ = cell.make_twins(gen, idx2, B=8, pool=cfg.twin_pool)
            for seg in ("rdp", "mdl", "window"):
                th, masks, c0 = ex.run_explainers(cell.f, gen, tw2, ts.X[idx2], cfg, seg, ["trajconcept"], seed)
                rows, _ = ex.evaluate_fidelity(cell.f, gen, tw2, ts.X[idx2], masks, c0, th, cfg, seed, auc=False)
                for m, r_ in rows.items():
                    S.add("e10_sens", dict(seed=seed, kind=f"seg_{seg}", value=m, gauss_H=r_["gauss_H"], ctwin_R=r_["ctwin_R"],
                                           ctwin_P_info_minus_rand=r_["ctwin_P_info_minus_rand"], n_seg=float(np.mean([len(M) for M in masks]))))
            for sg in (0.25, 0.5, 1.0, 2.0):                 # do manh phep nhieu DANH GIA (gaussian) anh huong R/P/H the nao
                c2 = dataclasses.replace(cfg, noise_sigma=sg)
                th, masks, c0 = ex.run_explainers(cell.f, gen, tw2, ts.X[idx2], c2, "rdp", ["trajconcept"], seed)
                rows, _ = ex.evaluate_fidelity(cell.f, gen, tw2, ts.X[idx2], masks, c0, th, c2, seed, common=False, auc=False)
                for m, r_ in rows.items():
                    S.add("e10_sens", dict(seed=seed, kind="op_sigma", value=f"{m}@{sg}", gauss_R=r_["gauss_R"], gauss_P=r_["gauss_P"], gauss_H=r_["gauss_H"],
                                           gauss_frac_informative=r_["gauss_frac_informative"], sigma=sg, method=m))
            _log("E10", seed, "ok")
        _safe(run)
    return S.load("e10_sens")


# ===================================================================== E15: phan tich that bai (tong hop)
def e15_failure(cfg: Config, S: Store):
    g = S.load("e1_gate"); t = S.load("e2_twin"); f5 = S.load("e5_funnel")
    if g.empty:
        return pd.DataFrame()
    g2 = g[g.tier == 2].groupby(["dataset", "bb"]).agg(delta_lo=("delta_nll_lo", "mean"), delta=("delta_nll", "mean"), usable=("usable", "mean")).reset_index()
    if not t.empty:
        t1 = t[t.gen == "flow"].groupby("dataset").agg(C2ST=("C2ST", "mean"), eps_K=("eps_K_median", "mean")).reset_index()
        g2 = g2.merge(t1, on="dataset", how="left")
    if not f5.empty:
        f5a = f5.groupby(["dataset", "bb"]).agg(lp_feasible=("lp_feasible_rate", "mean"), cand=("funnel_candidates", "mean"),
                                                 final=("funnel_all", "mean")).reset_index()
        g2 = g2.merge(f5a, on=["dataset", "bb"], how="left")
    def cause(r):
        if r["usable"] < 0.5: return "black box fails the minimum-quality check"
        if r["delta_lo"] <= 0: return "no knowledge gap (95% CI of delta includes 0)"
        if "C2ST" in r and r["C2ST"] > 0.65: return "twins are distinguishable from real data (C2ST > 0.65)"
        if r["bb"] not in DEEP: return "gap found, but concept extraction needs a neural network (not applicable)"
        if "lp_feasible" in r and r["lp_feasible"] < 0.3: return "concept optimisation rarely feasible (constraints too tight)"
        if "cand" in r and r["cand"] == 0: return "no stable concept candidates"
        if "final" in r and r["final"] == 0: return "concepts rejected by the four filters"
        return "concepts passed the four filters"
    g2["main_cause"] = g2.apply(cause, axis=1)
    S.save_table("table15_failure", g2, "Table 15 - Failure analysis per (dataset, black box)")
    g2.to_csv(S.csv("e15_failure"), index=False)
    return g2
