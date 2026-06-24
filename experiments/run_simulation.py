"""
MAIN RUN for simulation analysis — DEBUG VERSION

Aggiunge blocchi # ===== DEBUG ===== per diagnosticare:
  1. propagazione del SEED
  2. conteggi eventi per landmark x gruppo (DYNAMIC, per-bin)
  3. decomposizione con segno dei gap FPR/FNR (spiega il picco statico)
Rimuovi i blocchi DEBUG quando hai finito.
"""

import argparse
import gc
import os
from re import I
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, brier_score_loss
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


warnings.filterwarnings("ignore", category=FutureWarning)

# IMPORTS
from config import (
    SEED, DEVICE,
    ALPHA, BETA,
    EO_MODE_D,
    SCHEDULE_MODE_D,
    HORIZON, LANDMARKS_SIM, N_TEST_LANDMARKS,
    ID_COL, TIME_COL, EVENT_COL, SENS_COL,
    STATIC_COLS_SIM, TVC_COLS_SIM, CAT_COLS_SIM, ALL_NUM_COLS,
    ATTR_NAME, GROUP_NAMES_SIM,
    N_FOLDS, USE_WANDB, WANDB_ENTITY, WANDB_PROJECT,
    GRID_BETAS, GRID_ALPHAS, N_EPOCHS, LR, PW_CLIP,
)
from src.data.build_static        import build_static
from src.data.build_dynamic       import build_dynamic
from src.training.cross_validation import run_cv, build_summary_table, find_best_threshold
from src.training.grid_search      import run_grid_search, plot_tradeoff
from src.evaluation.fairness_metrics import (
    fairness_metrics, filter_sensitive, res_to_row,
    print_fairness_report, compute_adTPR_adFPR,
)
from src.evaluation.auc_fairness  import auc_fairness_single_attr
from src.evaluation.fairness_plots import (
    plot_separation_over_time_single, plot_auc_fairness_bar,
)


# Reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# ===== DEBUG: il SEED di config è arrivato fin qui? =====
print(f"[DEBUG seed @import] config.SEED={SEED}  "
      f"torch.initial_seed={torch.initial_seed()}  "
      f"np_state0={np.random.get_state()[1][0]}")
# =======================================================


def collapse_to_pd12(oof_hazard, event_bin, ids, lmk_vals, n_bins,
                     complete_only=True):
    """Da hazard per-bin (OOF) a PD(L, L+horizon) e label y12 per (soggetto, L)."""
    h = np.clip(oof_hazard, 1e-7, 1 - 1e-7)
    dfp = pd.DataFrame({
        "id": ids, "L": lmk_vals,
        "log1mh": np.log1p(-h),
        "ev": event_bin,
    })
    g    = dfp.groupby(["id", "L"], sort=False)
    surv = np.exp(g["log1mh"].sum())
    pd12 = (1.0 - surv).rename("pd12")
    y12  = g["ev"].max().rename("y12")
    cnt  = g.size().rename("n")
    out  = pd.concat([pd12, y12, cnt], axis=1).reset_index()
    if complete_only:
        out = out[out["n"] == n_bins]
    return out


# ===== DEBUG: conteggi eventi per landmark x gruppo (DYNAMIC per-bin) =====
def debug_event_counts(dynamic_data, label="DYNAMIC per-bin"):
    lmk = np.asarray(dynamic_data["lmk_vals"])
    y   = np.asarray(dynamic_data["y"])
    s   = np.asarray(dynamic_data["sensitive"])
    print(f"\n[DEBUG] conteggi eventi per landmark x gruppo ({label})")
    print(f"  {'L':>3} | {'g':>1} | {'n_rows':>7} | {'n_pos':>5} | {'prev':>6}")
    for L in sorted(np.unique(lmk)):
        for g in [0, 1]:
            mg = (lmk == L) & (s == g)
            n  = int(mg.sum())
            npos = int(y[mg].sum()) if n else 0
            prev = (y[mg].mean() if n else float("nan"))
            print(f"  {int(L):3d} | {g:1d} | {n:7d} | {npos:5d} | {prev:6.4f}")
# =========================================================================


# ===== DEBUG: decomposizione con segno dei gap (spiega la metrica separation) =====
def debug_signed_gaps(mname, yt_f, yb_f, sn_f):
    yt = np.asarray(yt_f); yb = np.asarray(yb_f); s = np.asarray(sn_f)
    def rates(g):
        m = s == g
        pos = (yt[m] == 1); neg = (yt[m] == 0)
        pred = yb[m]
        tpr = pred[pos].mean() if pos.sum() else float("nan")
        fpr = pred[neg].mean() if neg.sum() else float("nan")
        return tpr, fpr, 1.0 - tpr
    t0, f0, n0 = rates(0)
    t1, f1, n1 = rates(1)
    fpr_gap = f1 - f0
    fnr_gap = n1 - n0
    sum_abs  = abs(fpr_gap) + abs(fnr_gap)          # quello che minimizza il PENALTY (vecchio)
    abs_sum  = abs(fpr_gap + fnr_gap)               # quello che misura la METRICA separation*2
    print(f"   [DEBUG {mname}] "
          f"fpr_gap={fpr_gap:+.4f}  fnr_gap={fnr_gap:+.4f}  "
          f"| sum|.|={sum_abs:.4f}  |sum|={abs_sum:.4f}  sep(|sum|/2)={abs_sum/2:.4f}")
# ================================================================================


def parse_args():
    p = argparse.ArgumentParser(
        description="Run simulation experiment (fair/unfair/direct/proxy)."
    )
    p.add_argument("--data_dir", required=True,
                   help="Directory containing data_{scenario}.csv and test files")
    p.add_argument("--scenario", default="fair",
                   choices=["fair", "unfair", "direct", "proxy", "temporal"])
    p.add_argument("--config", default=None,
                   help="Path to YAML config (overrides config.py defaults)")
    p.add_argument("--grid_search", action="store_true",
                   help="Run grid search after CV")
    p.add_argument("--out_dir", default=None,
                   help="Output directory (default: outputs/simulation/{scenario})")
    return p.parse_args()


def load_config(config_path, args):
    cfg = dict(
        alpha=ALPHA, beta=BETA,
        eo_mode_d=EO_MODE_D,
        schedule_mode_d=SCHEDULE_MODE_D,
        horizon=HORIZON, landmarks=LANDMARKS_SIM,
        n_folds=N_FOLDS, use_wandb=USE_WANDB,
        grid_betas=GRID_BETAS, grid_alphas=GRID_ALPHAS,
    )
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            overrides = yaml.safe_load(f)
        cfg.update(overrides or {})
    return cfg


# Data loading
def load_raw(data_dir, scenario):
    path = os.path.join(data_dir, f"data_{scenario}.csv")
    df   = pd.read_csv(path)
    df = df.sort_values([ID_COL, TIME_COL])

    # Trend features
    trend_cols = []
    for col in ["X4", "X6"]:
        name = f"{col}_trend"
        df[name] = (
            df.groupby(ID_COL)[col]
            .transform(lambda x: x - x.shift(2))
        ).clip(-5, 5).fillna(0)
        trend_cols.append(name)

    for c in ALL_NUM_COLS + trend_cols + [TIME_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    # Sensitive attribute per loan
    sens_per_id = df.groupby(ID_COL)[SENS_COL].first().rename("sens_loan")
    df = df.merge(sens_per_id, on=ID_COL, how="left")

    # FirstEventTime
    first_event = (
        df[df[EVENT_COL] == 1]
        .groupby(ID_COL)[TIME_COL].min()
        .rename("FirstEventTime")
    )
    df = df.merge(first_event, on=ID_COL, how="left")

    # ===== DEBUG: composizione gruppi e base-rate per gruppo =====
    n_ids = df[ID_COL].nunique()
    ev_per_id = df.groupby(ID_COL)["FirstEventTime"].first().notna()
    sens_per  = df.groupby(ID_COL)["sens_loan"].first()
    print(f"\n[DEBUG raw] scenario={scenario}  n_ids={n_ids}")
    for g in sorted(sens_per.dropna().unique()):
        ids_g = sens_per[sens_per == g].index
        n_g   = len(ids_g)
        rate  = ev_per_id.loc[ids_g].mean()
        print(f"  gruppo s={int(g)}: n_id={n_g:6d}  default_rate={rate:.4f}")
    # =============================================================

    return df, trend_cols


# Fairness analysis
def run_fairness_analysis(
    y_static, static_oof, sens_static,
    y_dynamic, dynamic_oof, sens_dynamic, lmk_vals,
    out_dir, cfg, th_static, th_dynamic):

    ybin_static  = (static_oof  >= th_static ).astype(int)
    ybin_dynamic = (dynamic_oof >= th_dynamic).astype(int)

    # Aggregate
    agg_rows = []
    for mname, y_t, y_p, y_b, sens, th in [
        ("M_STATIC",  y_static,  static_oof,  ybin_static,  sens_static,  th_static),
        ("M_DYNAMIC", y_dynamic, dynamic_oof, ybin_dynamic, sens_dynamic, th_dynamic),
    ]:
        yt_f, yp_f, sn_f = filter_sensitive(y_t, y_p, sens)
        yb_f = (yp_f >= th).astype(int)
        res  = fairness_metrics(yt_f, yp_f, yb_f, sn_f, GROUP_NAMES_SIM, threshold=th)
        print_fairness_report(mname, res, GROUP_NAMES_SIM, label="AGGREGATE")
        # ===== DEBUG: decomposizione con segno dei gap =====
        debug_signed_gaps(mname, yt_f, yb_f, sn_f)
        # ===================================================
        agg_rows.append(res_to_row(res, GROUP_NAMES_SIM, {"model": mname}))

    df_agg = pd.DataFrame(agg_rows)
    df_agg.to_csv(out_dir / "fairness_aggregate.csv", index=False)

    # Dynamic per landmark
    dyn_rows = []
    for L in cfg["landmarks"]:
        mask = lmk_vals == L
        if mask.sum() < 20: continue
        yt_f, yp_f, sn_f = filter_sensitive(
            y_dynamic[mask], dynamic_oof[mask], sens_dynamic[mask]
        )
        if len(np.unique(yt_f)) < 2 or len(np.unique(sn_f)) < 2: continue
        yb_f = (yp_f >= th_dynamic).astype(int)
        res  = fairness_metrics(yt_f, yp_f, yb_f, sn_f, GROUP_NAMES_SIM, threshold=th_dynamic)
        dyn_rows.append(res_to_row(res, GROUP_NAMES_SIM,
                                   {"model": "M_DYNAMIC", "landmark": L}))

    df_dyn_lmk = pd.DataFrame(dyn_rows)
    df_dyn_lmk.to_csv(out_dir / "fairness_dynamic_by_landmark.csv", index=False)

    # adTPR / adFPR
    print("\n--- adTPR / adFPR ---")
    for mname, y_t, y_b, sens, tpts in [
        ("M_STATIC",  y_static,  ybin_static,  sens_static,  None),
        ("M_DYNAMIC", y_dynamic, ybin_dynamic, sens_dynamic, lmk_vals),
    ]:
        res = compute_adTPR_adFPR(y_t, y_b, sens, tpts)
        print(f"  {mname:<12} — adTPR={res['adTPR']:.4f}  adFPR={res['adFPR']:.4f}")

    # AUC fairness
    df_auc = auc_fairness_single_attr(
        df_dynamic=df_dyn_lmk, df_static_agg=df_agg,
        time_col_dyn="landmark",
    )
    df_auc.to_csv(out_dir / "fairness_auc_comparison.csv", index=False)
    print("\n--- AUC FAIRNESS ---")
    print(df_auc.to_string(index=False))

    # Plots
    static_vals = {
        k: df_agg[df_agg["model"] == "M_STATIC"][k].values[0]
        for k in ["independence", "separation", "sufficiency"]
        if k in df_agg.columns
    }

    plot_separation_over_time_single(
        df_time=df_dyn_lmk, time_col="landmark",
        title=f"Fairness — M_DYNAMIC",
        filename="fairness_dynamic_over_landmark.png",
        out_dir=out_dir, static_val=float(static_vals.get("separation", np.nan)),
        min_samples_per_group=20,
    )

    plot_auc_fairness_bar(df_auc=df_auc, out_dir=out_dir,
                          filename="fairness_auc_comparison.png")

    print(f"\n Fairness outputs saved in: {out_dir}")
    return df_agg, df_dyn_lmk, df_auc


def main():
    args = parse_args()
    cfg  = load_config(args.config, args)

    # ===== DEBUG: seed effettivo a inizio main =====
    print(f"[DEBUG seed @main] config.SEED={SEED}  torch.initial_seed={torch.initial_seed()}")
    # ===============================================

    # Output directory
    out_dir = Path(args.out_dir) if args.out_dir else \
              Path("outputs") / "simulation" / args.scenario
    out_dir.mkdir(parents=True, exist_ok=True)

    run_tag = (
        f"simulation_{args.scenario}"
        f"_S:{cfg['beta']}"
        f"_D:{cfg['alpha']}_{cfg['eo_mode_d']}"
    )

    if cfg["use_wandb"]:
      import wandb
      wandb.init(
          project = WANDB_PROJECT,
          entity  = WANDB_ENTITY,
          name    = run_tag,
          config  = {
              "beta":            cfg["beta"],
              "alpha":           cfg["alpha"],
              "eo_mode_d":       cfg["eo_mode_d"],
              "schedule_mode_d": cfg["schedule_mode_d"],
              "horizon":         cfg["horizon"],
              "n_folds":         cfg["n_folds"],
              "landmarks":       cfg["landmarks"],
              "n_epochs":        N_EPOCHS,
              "lr":              LR,
              "pw_clip":         PW_CLIP,
          }
      )

    print(f"\n{'='*60}")
    print(f"  Scenario : {args.scenario}")
    print(f"  Data dir : {args.data_dir}")
    print(f"{'='*60}\n")

    # Load raw data
    print("Loading raw data...")
    df, trend_cols = load_raw(args.data_dir, args.scenario)

    enc_cat = OneHotEncoder(handle_unknown="ignore",
                             sparse_output=False, dtype=np.float32)
    enc_cat.fit(df[CAT_COLS_SIM])

    # Build datasets
    print("\nBuilding STATIC dataset...")
    static_data = build_static(
        df=df,
        static_cols=STATIC_COLS_SIM, cat_cols=CAT_COLS_SIM,
        horizon=cfg["horizon"],
        id_col=ID_COL, time_col=TIME_COL,
        first_event_col="FirstEventTime",
        sens_col="sens_loan", enc_cat=enc_cat,
    )

    print("\nBuilding DYNAMIC dataset...")
    dynamic_data = build_dynamic(
        df=df,
        static_cols=STATIC_COLS_SIM, tvc_cols=TVC_COLS_SIM,
        cat_cols=CAT_COLS_SIM, landmarks=cfg["landmarks"],
        horizon=cfg["horizon"], delta=cfg.get("delta", 1),
        id_col=ID_COL, time_col=TIME_COL,
        first_event_col="FirstEventTime",
        sens_col="sens_loan", enc_cat=enc_cat,
    )

    # ===== DEBUG: conteggi eventi per landmark x gruppo (simulati) =====
    debug_event_counts(dynamic_data)
    # ==================================================================

    del df; gc.collect()

    # Cross-validation
    train_kwargs = dict(
        beta=cfg["beta"], alpha=cfg["alpha"],
        eo_mode_d=cfg["eo_mode_d"],
        schedule_mode_d=cfg["schedule_mode_d"],
    )

    print("\nTraining M_STATIC...")
    res_static = run_cv(
        X=static_data["X"], y=static_data["y"],
        groups=static_data["groups"], sensitive=static_data["sensitive"],
        model_name="static", n_splits=cfg["n_folds"], **train_kwargs,
    )

    print("\nTraining M_DYNAMIC...")
    res_dynamic = run_cv(
        X=dynamic_data["X"], y=dynamic_data["y"],
        groups=dynamic_data["groups"], sensitive=dynamic_data["sensitive"],
        time_arr=dynamic_data["lmk_vals"], subj_ids=dynamic_data["groups"],
        model_name="dynamic", n_splits=cfg["n_folds"],
        landmarks=cfg["landmarks"], **train_kwargs,
    )

    # ===== DEBUG: lo static_oof cambia tra run/seed? (firma rapida) =====
    so = res_static["oof_preds"]
    print(f"[DEBUG static_oof] mean={so.mean():.6f}  std={so.std():.6f}  "
          f"sum={so.sum():.4f}  hash={hash(so.tobytes()) & 0xffffffff}")
    do = res_dynamic["oof_preds"]
    print(f"[DEBUG dyn_oof]    mean={do.mean():.6f}  std={do.std():.6f}  "
          f"sum={do.sum():.4f}  hash={hash(do.tobytes()) & 0xffffffff}")
    # Se l'hash dello static_oof è identico tra due seed diversi -> il seed
    # non sta cambiando lo split/init dello static. Cambia il SEED in config
    # e rilancia: questi due hash DEVONO cambiare.
    # ===================================================================

    # --- COLLAPSE hazard per-bin -> PD-(L, L+horizon) ---
    n_bins = cfg["horizon"] // cfg.get("delta", 1)

    pd12_df = collapse_to_pd12(
        oof_hazard    = res_dynamic["oof_preds"],
        event_bin     = dynamic_data["y"],
        ids           = dynamic_data["groups"],
        lmk_vals      = dynamic_data["lmk_vals"],
        n_bins        = n_bins,
        complete_only = True,
    )

    dyn_pd  = pd12_df["pd12"].to_numpy()
    dyn_y12 = pd12_df["y12"].to_numpy()
    dyn_L   = pd12_df["L"].to_numpy()
    dyn_ids = pd12_df["id"].to_numpy()

    th_dynamic = find_best_threshold(dyn_y12, dyn_pd)
    dyn_auc    = roc_auc_score(dyn_y12, dyn_pd)
    dyn_brier  = brier_score_loss(dyn_y12, dyn_pd)
    print(f"\nM_DYNAMIC (PD-12) AUC={dyn_auc:.4f}  Brier={dyn_brier:.4f}")

    # --- collapse del sensitive a livello (soggetto, L) ---
    bin_ids = dynamic_data["groups"]
    id2g = pd.Series(dynamic_data["sensitive"], index=bin_ids)
    id2g = id2g[~id2g.index.duplicated(keep="first")]
    dyn_sens_collapsed = pd.Series(dyn_ids).map(id2g).to_numpy()

    # Summary
    summary = pd.DataFrame([
        {"Model": "M_STATIC",  **res_static["summary"]},
        {"Model": "M_DYNAMIC", "AUC_Mean": dyn_auc, "Brier_Mean": dyn_brier},
    ])
    print("\n=== CV RESULTS ===")
    print(summary.to_string(index=False))
    summary.to_csv(out_dir / "cv_results.csv", index=False)

    if cfg["use_wandb"]:
        import wandb
        for _, row in summary.iterrows():
            m = row["Model"].lower()
            wandb.log({
                f"{m}/AUC_Mean":   row.get("AUC_Mean"),
                f"{m}/AUC_SD":     row.get("AUC_SD"),
                f"{m}/Brier_Mean": row.get("Brier_Mean"),
                f"{m}/Brier_SD":   row.get("Brier_SD"),
                f"{m}/F1_Mean":    row.get("F1_Mean"),
                f"{m}/F1_SD":      row.get("F1_SD"),
            })

    # Fairness analysis
    print("\n" + "="*60)
    print("FAIRNESS ANALYSIS")
    print("="*60)

    th_static = res_static["threshold"]
    # ===== DEBUG: soglie usate per la fairness =====
    print(f"[DEBUG th] th_static={th_static:.5f}  th_dynamic={th_dynamic:.5f}")
    # ===============================================

    df_agg, df_dyn_lmk, df_auc = run_fairness_analysis(
        y_static=static_data["y"],
        static_oof=res_static["oof_preds"],
        sens_static=static_data["sensitive"],
        y_dynamic=dyn_y12,
        dynamic_oof=dyn_pd,
        sens_dynamic=dyn_sens_collapsed,
        lmk_vals=dyn_L,
        out_dir=out_dir, cfg=cfg,
        th_static=th_static, th_dynamic=th_dynamic
    )

    if cfg["use_wandb"]:
        import wandb
        for _, row in df_agg.iterrows():
            prefix = f"{row['model'].lower()}/aggregate"
            wandb.log({f"{prefix}/separation": row.get("separation")})
        for _, row in df_auc.iterrows():
            m = row["metric"]
            wandb.log({
                f"auc_fairness/{m}/M_STATIC":  row["AUC_M_STATIC"],
                f"auc_fairness/{m}/M_DYNAMIC": row["AUC_M_DYNAMIC"],
            })
        for _, row in df_dyn_lmk.iterrows():
            L = int(row["landmark"])
            wandb.log({f"dynamic/landmark_{L}/separation": row.get("separation")})
        wandb.finish()

    # Grid search
    if args.grid_search:
        df_grid = run_grid_search(
            X_static=static_data["X"], y_static=static_data["y"],
            grp_static=static_data["groups"], sens_static=static_data["sensitive"],
            X_dynamic=dynamic_data["X"], y_dynamic=dynamic_data["y"],
            grp_dynamic=dynamic_data["groups"], sens_dynamic=dynamic_data["sensitive"],
            lmk_vals=dynamic_data["lmk_vals"],
            group_names=GROUP_NAMES_SIM,
            betas=cfg["grid_betas"], alphas=cfg["grid_alphas"],
            n_folds=cfg["n_folds"],
            eo_mode_d=cfg["eo_mode_d"],
            schedule_mode_d=cfg["schedule_mode_d"],
            out_dir=out_dir, run_tag=run_tag,
        )
        plot_tradeoff(df_grid, out_dir=out_dir, run_tag=run_tag)

    print(f"\n All outputs saved in: {out_dir}")


if __name__ == "__main__":
    main()