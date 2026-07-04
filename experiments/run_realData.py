"""
MAIN RUN for real data analysis

Reads data matched by data_generation/realData/,
builds the two datasets, runs CV, fairness analysis, and grid search.
"""

import argparse
import gc
import os

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import yaml
from sklearn.preprocessing import OneHotEncoder
import sys
from pathlib import Path

from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=FutureWarning)

# IMPORTS
from config import (
    SEED, DEVICE,
    ALPHA, BETA, 
    EO_MODE_D,
    SCHEDULE_MODE_D, 
    HORIZON_MONTHS, LANDMARKS,
    STATIC_COLS, TVC_COLS, CAT_COLS,
    FAIR_ATTR, GROUP_NAMES, DELTA,
    N_FOLDS, USE_WANDB, WANDB_ENTITY, WANDB_PROJECT,
    GRID_BETAS, GRID_ALPHAS,N_EPOCHS,LR, PW_CLIP,
)
from src.data.build_static        import build_static
from src.data.build_dynamic       import build_dynamic
from src.training.run_train import run_cv, build_summary_table, find_best_threshold,run_grid_search, plot_tradeoff,make_splits
from src.evaluation.fairness_metrics import (
    fairness_metrics, filter_sensitive, res_to_row,
    print_fairness_report, compute_adTPR_adFPR,
)
from src.evaluation.auc_fairness  import auc_fairness_all_models
from src.evaluation.fairness_plots import (
    plot_separation_over_time, plot_auc_fairness_bar,
)


# Reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# From hazard per_bin to PD(L, L+horizon)
def collapse_to_pdh(oof_hazard, event_bin, ids, lmk_vals, n_bins,
                     complete_only=True):
    # Numerical stability
    h = np.clip(oof_hazard, 1e-7, 1 - 1e-7)
    dfp = pd.DataFrame({
        "id": ids, "L": lmk_vals,
         # log(1 - hazard) -> to translate product in sum
        "log1mh": np.log1p(-h),  
        "ev": event_bin,
    })
    # Group bin of the same subject and landmark
    g    = dfp.groupby(["id", "L"], sort=False)
    # prod(1 - h) = exp(sum(log(1 - h)))
    surv = np.exp(g["log1mh"].sum())  
    # The probability of default is 1-Surv 
    pdh = (1.0 - surv).rename("pdh")
    # Indicate if there is the event in any of the bin 
    yh  = g["ev"].max().rename("yh")
    # Number of bin for (id, L)
    cnt  = g.size().rename("n")
    out  = pd.concat([pdh, yh, cnt], axis=1).reset_index()
    # Require all bins                   
    if complete_only:   
        out = out[out["n"] == n_bins]
    return out


def plot_pd_by_landmark_group(dyn_pd, dyn_L, sens_arr, group_names, out_dir, title, filename):
    df = pd.DataFrame({"L": dyn_L, "pd": dyn_pd, "sens": sens_arr})
    agg = df.groupby(["L", "sens"])["pd"].mean().unstack()
    fig, ax = plt.subplots(figsize=(8,5))
    for g in agg.columns:
        ax.plot(agg.index, agg[g], marker="o", label=group_names.get(g, str(g)))
    ax.set_xlabel("Landmark"); ax.set_ylabel("PD-H media predetta")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    plt.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run real-data experiment."
    )
    p.add_argument("--data_path", required=True,
                   help="Path to panel_all_years_sampled.csv")
    p.add_argument("--fair_attr", default="SEX",
                   choices=["SEX", "RACE", "AGE"])
    p.add_argument("--config", default=None,
                   help="Path to YAML config")
    p.add_argument("--grid_search", action="store_true",
                   help="Run grid search after CV")
    p.add_argument("--out_dir", default=None,
                   help="Output directory")
    return p.parse_args()


def load_config(config_path):
    cfg = dict(
        alpha=ALPHA, beta=BETA, 
        eo_mode_d=EO_MODE_D, delta=DELTA,
        schedule_mode_d=SCHEDULE_MODE_D, 
        horizon=HORIZON_MONTHS, landmarks=LANDMARKS,
        n_folds=N_FOLDS, use_wandb=USE_WANDB,
        grid_betas=GRID_BETAS, grid_alphas=GRID_ALPHAS,
        n_epochs= N_EPOCHS,lr = LR, pw_clip=  PW_CLIP,
   
    )
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            overrides = yaml.safe_load(f)
        cfg.update(overrides or {})
    return cfg


def run_feature_importance(static_data, dynamic_data, 
                            res_static, res_dynamic,
                            out_dir, use_wandb=False):
    import matplotlib.pyplot as plt

    print("\n" + "="*60)
    print("FEATURE IMPORTANCE")
    print("="*60)

    for name, data, res in [
        ("M_STATIC",  static_data,  res_static),
        ("M_DYNAMIC", dynamic_data, res_dynamic),
    ]:
        model  = res["model_last"]
        scaler = res["scaler_last"]
        X      = data["X"]
        y      = data["y"]
        feature_names = data["feature_names"]

        # Scale with the same scaler used in training
        X_s = scaler.transform(X).astype(np.float32)
        X_s = np.nan_to_num(X_s, nan=0., posinf=5., neginf=-5.)

        # AUC baseline
        model.eval()
        with torch.no_grad():
            baseline_preds = torch.sigmoid(
                model(torch.tensor(X_s, device=DEVICE))
            ).cpu().numpy()
        baseline_auc = roc_auc_score(y, baseline_preds)

        # Permutation importance
        importances = []
        for i in range(X_s.shape[1]):
            X_perm = X_s.copy()
            np.random.shuffle(X_perm[:, i])  
            with torch.no_grad():
                perm_preds = torch.sigmoid(
                    model(torch.tensor(X_perm, device=DEVICE))
                ).cpu().numpy()
            perm_auc = roc_auc_score(y, perm_preds)
            # AUC decrease
            importances.append(baseline_auc - perm_auc) 

        df_imp = pd.DataFrame({
            "feature":    feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)

        print(f"\n--- {name} (baseline AUC={baseline_auc:.4f}) ---")
        print(df_imp.head(15).to_string(index=False))
        df_imp.to_csv(out_dir / f"feature_importance_{name.lower()}.csv", index=False)

        # Plot top 15
        top = df_imp.head(35)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top["feature"][::-1], top["importance"][::-1], color="#4C72B0")
        ax.set_xlabel("AUC drop (↑ more important)")
        ax.set_title(f"Permutation Importance — {name}")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plot_path = out_dir / f"feature_importance_{name.lower()}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)

        if use_wandb:
            import wandb
            wandb.log({f"feature_importance/{name}": wandb.Image(str(plot_path))})

#  Fairness analysis 
def run_fairness_analysis(
    y_static, static_oof, sens_by_attr_static,
    y_dynamic, dynamic_oof, sens_by_attr_dynamic, lmk_vals,
    out_dir, cfg,  th_static, th_dynamic):

    attrs = ["SEX", "RACE", "AGE"]

    ybin_static  = (static_oof  >= th_static ).astype(int)
    ybin_dynamic = (dynamic_oof >= th_dynamic).astype(int)

    agg_rows = []
    dyn_rows = []

    for attr_name in attrs:
        group_names = GROUP_NAMES[attr_name]
        s_stat = sens_by_attr_static[attr_name]
        s_dyn  = sens_by_attr_dynamic[attr_name]

        print(f"\n{'='*50}\n  {attr_name}\n{'='*50}")

        # Aggregate
        for mname, y_t, y_p, y_b, sens, th in [
            ("M_STATIC",  y_static,  static_oof,  ybin_static,  s_stat, th_static),
            ("M_DYNAMIC", y_dynamic, dynamic_oof, ybin_dynamic, s_dyn,  th_dynamic),
        ]:
            yt_f, yp_f, sn_f = filter_sensitive(y_t, y_p, sens)
            yb_f = (yp_f >= th).astype(int)
            res  = fairness_metrics(yt_f, yp_f, yb_f, sn_f,
                                    group_names, threshold=th)
            print_fairness_report(mname, res, group_names, label="AGGREGATE")
            agg_rows.append(res_to_row(res, group_names,
                                       {"attr": attr_name, "model": mname}))
        MIN_GROUP = 50 
        # Dynamic per landmark
        for L in cfg["landmarks"]:
            mask = lmk_vals == L
            yt_f, yp_f, sn_f = filter_sensitive(
                y_dynamic[mask], dynamic_oof[mask], s_dyn[mask]
            )
            if len(np.unique(yt_f)) < 2 or len(np.unique(sn_f)) < 2: continue
            counts = np.array([(sn_f == g).sum() for g in np.unique(sn_f)])
            n_group_min = int(counts.min())
            if n_group_min < MIN_GROUP: continue
            yb_f = (yp_f >= th_dynamic).astype(int)
            res  = fairness_metrics(yt_f, yp_f, yb_f, sn_f,
                                    group_names, threshold=th_dynamic)
            dyn_rows.append(res_to_row(res, group_names,
                                       {"attr": attr_name,
                                        "model": "M_DYNAMIC",
                                        "landmark": L}))

        # adTPR / adFPR
        print(f"\n  adTPR / adFPR — {attr_name}")
        for mname, y_t, y_b, sens, tpts in [
            ("M_STATIC",  y_static,  ybin_static,  s_stat, None),
            ("M_DYNAMIC", y_dynamic, ybin_dynamic, s_dyn,  lmk_vals),
        ]:
            res = compute_adTPR_adFPR(y_t, y_b, sens, tpts)
            print(f"    {mname:<12} adTPR={res['adTPR']:.4f}  adFPR={res['adFPR']:.4f}")

    df_agg     = pd.DataFrame(agg_rows)
    df_dyn_lmk = pd.DataFrame(dyn_rows)


    df_agg.to_csv(out_dir / "fairness_aggregate.csv", index=False)
    df_dyn_lmk.to_csv(out_dir / "fairness_dynamic_by_landmark.csv", index=False)
 
    # AUC fairness
    df_auc = auc_fairness_all_models(
        df_dynamic=df_dyn_lmk, df_static_agg=df_agg,
        time_col_dyn="landmark",
        min_samples_per_group=50,
    )
    df_auc.to_csv(out_dir / "auc_fairness_comparison.csv", index=False)
    print("\n=== AUC FAIRNESS ===")
    print(df_auc.to_string(index=False))


    # Plots
    plot_separation_over_time(
        df=df_dyn_lmk, time_col="landmark",
        title="Fairness — M_DYNAMIC by landmark",
        filename="fairness_dynamic_by_landmark.png",
        out_dir=out_dir, static_df=df_agg, min_samples_per_group=100,
    )

    for attr_name in attrs:
        sub = df_auc[df_auc["attr"] == attr_name].drop(columns="attr")
        plot_auc_fairness_bar(
            df_auc=sub, out_dir=out_dir, attr_name=attr_name,
            filename=f"fairness_auc_{attr_name}.png",
        )
   
    print(f"\nFairness outputs saved in: {out_dir}")
    return df_agg, df_dyn_lmk, df_auc



def main():
    args = parse_args()
    cfg  = load_config(args.config)
    print(SEED)

    out_dir = Path(args.out_dir) if args.out_dir else \
              Path("outputs") / "realData" / args.fair_attr
    out_dir.mkdir(parents=True, exist_ok=True)

    run_tag = (
        f"realData_{args.fair_attr}"
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
                "fair_attr":       args.fair_attr,
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
                "seed": SEED,
            }
        )

    print(f"\n{'='*60}")
    print(f"  Dataset   :  REAL")
    print(f"  Attr      : {args.fair_attr}")
    print(f"{'='*60}\n")

    # Load preprocessed data 
    df = pd.read_csv(args.data_path, low_memory=False)

    # Sensitive arrays for all three attributes (needed for fairness loop)
    sens_col_map = {
        "SEX":  "sex_bin_loan",
        "RACE": "race_bin_loan",
        "AGE":  "age_bin_loan",
    }
    
    df["sens_loan"] = df[sens_col_map[args.fair_attr]] 

    trend_cols = ["bd_pct_trend", "estimated_ltv_trend", "current_upb_trend"]

    enc_cat = OneHotEncoder(handle_unknown="ignore",
                             sparse_output=False, dtype=np.float32)
    enc_cat.fit(df[CAT_COLS])

    # Build datasets
    print("\nBuilding STATIC dataset...")
    static_data = build_static(
        df=df,
        static_cols=STATIC_COLS, cat_cols=CAT_COLS,
        horizon=cfg["horizon"],
        id_col="loan_sequence_number", time_col="loan_age",
        first_event_col="FirstDefaultAge",
        sens_col="sens_loan", enc_cat=enc_cat,
    )

    print("\nBuilding DYNAMIC dataset...")
    dynamic_data = build_dynamic(
        df=df,
        static_cols=STATIC_COLS, tvc_cols=TVC_COLS,
        cat_cols=CAT_COLS, landmarks=cfg["landmarks"],
        horizon=cfg["horizon"],delta=cfg.get("delta", 4),  
        id_col="loan_sequence_number", time_col="loan_age",
        first_event_col="FirstDefaultAge",
        sens_col="sens_loan", enc_cat=enc_cat,
    )


    # Collect sensitive arrays for all attributes
    static_sens_by_attr = {}
    dyn_sens_by_attr    = {}

    for attr_name, col in sens_col_map.items():
        # reindex from original df
        st_ids  = pd.Series(static_data["groups"])
        dy_ids  = pd.Series(dynamic_data["groups"])

        per_loan = df.groupby("loan_sequence_number")[col].first()

        static_sens_by_attr[attr_name] = st_ids.map(per_loan).to_numpy()
        dyn_sens_by_attr[attr_name]    = dy_ids.map(per_loan).to_numpy()

    del df; gc.collect()

    #  CV 

    t_min = float(min(cfg["landmarks"]))
    t_max = float(max(cfg["landmarks"]))

    train_kwargs = dict(
        beta=cfg["beta"], alpha=cfg["alpha"],
        eo_mode_d=cfg["eo_mode_d"], schedule_mode_d=cfg["schedule_mode_d"],
        t_min=t_min, t_max=t_max,
    )

    splits_d = make_splits(dynamic_data["y"], dynamic_data["groups"], n_splits=cfg["n_folds"])
    splits_s = make_splits(static_data["y"],  static_data["groups"],  n_splits=cfg["n_folds"])

    print("\nTraining M_STATIC...")
    res_static = run_cv(
    X=static_data["X"], y=static_data["y"],
    groups=static_data["groups"], sensitive=static_data["sensitive"],
    model_name="static", n_splits=cfg["n_folds"],
    group_names=GROUP_NAMES[args.fair_attr],  
    splits=splits_s,                           
    **train_kwargs,
    )
        
    n_bins = cfg["horizon"] // cfg.get("delta", 4)

    print("\nTraining M_DYNAMIC...")
    
    res_dynamic = run_cv(
    X=dynamic_data["X"], y=dynamic_data["y"],
    groups=dynamic_data["groups"], sensitive=dynamic_data["sensitive"],
    time_arr=dynamic_data["lmk_vals"], subj_ids=dynamic_data["groups"],
    model_name="dynamic", n_splits=cfg["n_folds"],
    landmarks=cfg["landmarks"], collapse_pdh=True, n_bins=n_bins,
    group_names=GROUP_NAMES[args.fair_attr], 
    splits=splits_d,                            
    **train_kwargs,
    )
    mask_d = res_dynamic["is_test"]
    mask_s = res_static["is_test"]

    pdh_df = collapse_to_pdh(
        oof_hazard = res_dynamic["oof_preds"][mask_d], # hazard per bin
        event_bin  = dynamic_data["y"][mask_d], # Event per bin 
        ids        = dynamic_data["groups"][mask_d], # Subject indication
        lmk_vals   = dynamic_data["lmk_vals"][mask_d], # Landamrk indication
        n_bins     = n_bins,
        complete_only = True,
    )


    
    # Indicate the probability of default at from L to h
    dyn_pd   = pdh_df["pdh"].to_numpy()
    # Indicate if there is the event in any of the bin 
    dyn_yh  = pdh_df["yh"].to_numpy()
    dyn_L    = pdh_df["L"].to_numpy()
    dyn_ids  = pdh_df["id"].to_numpy()
    
    # Threshold on pdh
    th_dynamic = find_best_threshold(dyn_yh, dyn_pd)



    summary = build_summary_table({
        "M_STATIC":  res_static,
        "M_DYNAMIC": res_dynamic,
    })


    print("\n=== CV RESULTS ===")
    print(summary.to_string(index=False))
    summary.to_csv(out_dir / "cv_results.csv", index=False)


    if cfg["use_wandb"]:
        import wandb
        for _, row in summary.iterrows():
            m = row["Model"].lower()
            wandb.log({
                f"{m}/AUC_Mean":   row["AUC_Mean"],
                f"{m}/AUC_SD":     row["AUC_SD"],
                f"{m}/Brier_Mean": row["Brier_Mean"],
                f"{m}/Brier_SD":   row["Brier_SD"],
                f"{m}/F1_Mean":    row["F1_Mean"],
                f"{m}/F1_SD":      row["F1_SD"],
            })
    run_feature_importance(
        static_data  = static_data,
        dynamic_data = dynamic_data,
        res_static   = res_static,
        res_dynamic  = res_dynamic,
        out_dir      = out_dir,
        use_wandb    = cfg["use_wandb"],
    )
    # Fairness analysis
    print("\n" + "="*60)
    print("FAIRNESS ANALYSIS")
    print("="*60)

    th_static  = res_static["threshold"]


    bin_ids = dynamic_data["groups"]
    dyn_sens_collapsed = {}
    # Assign a unique sensitive attribute per (ID,L)
    for attr, sarr in dyn_sens_by_attr.items():
        id2g = pd.Series(sarr, index=bin_ids)
        id2g = id2g[~id2g.index.duplicated(keep="first")] 
        dyn_sens_collapsed[attr] = pd.Series(dyn_ids).map(id2g).to_numpy()


    plot_pd_by_landmark_group(
    dyn_pd=dyn_pd,
    dyn_L=dyn_L,
    sens_arr=dyn_sens_collapsed[args.fair_attr],  
    group_names=GROUP_NAMES[args.fair_attr],
    out_dir=out_dir,
    title=f"PD-H per landmark — M_DYNAMIC (α={cfg['alpha']})",
    filename=f"pd_by_landmark_{args.fair_attr}_alpha{cfg['alpha']}.png",
    )

    df_agg, df_dyn_lmk, df_auc = run_fairness_analysis(
    y_static=static_data["y"][mask_s],
    static_oof=res_static["oof_preds"][mask_s],
    sens_by_attr_static=static_sens_by_attr,
    y_dynamic=dyn_yh,                    
    dynamic_oof=dyn_pd,                      
    sens_by_attr_dynamic=dyn_sens_collapsed, 
    lmk_vals=dyn_L,                     
    out_dir=out_dir, cfg=cfg,
    th_static=th_static, th_dynamic=th_dynamic,
    )

    if cfg["use_wandb"]:
        import wandb
        # Aggregate fairness
        for _, row in df_agg.iterrows():
            prefix = f"{row['model'].lower()}/{args.fair_attr}/aggregate"
            wandb.log({
                f"{prefix}/separation":   row.get("separation"),
                })

        # AUC fairness
        for _, row in df_auc.iterrows():
            m = row["metric"]
            wandb.log({
                f"auc_fairness/{m}/M_STATIC":  row["AUC_M_STATIC"],
                f"auc_fairness/{m}/M_DYNAMIC": row["AUC_M_DYNAMIC"],
            })

        # Dynamic per landmark
        for _, row in df_dyn_lmk.iterrows():
            L = int(row["landmark"])
            wandb.log({
                f"dynamic/{args.fair_attr}/landmark_{L}/separation":   row.get("separation"),
            })

        for attr_name in ["SEX", "RACE", "AGE"]:
            img_path = out_dir / f"fairness_auc_{attr_name}.png"
            if img_path.exists():
                wandb.log({f"fairness_plot/{attr_name}": wandb.Image(str(img_path))})
        
        sep_plot = out_dir / "fairness_dynamic_by_landmark.png"
        if sep_plot.exists():
            wandb.log({"fairness_separation_plot": wandb.Image(str(sep_plot))})


    # Grid search
    if args.grid_search:
        print("\n" + "="*60)
        print("GRID SEARCH")
        print("="*60)
        df_grid = run_grid_search(
            X_static=static_data["X"], y_static=static_data["y"],
            grp_static=static_data["groups"],
            sens_static=static_data["sensitive"],
            X_dynamic=dynamic_data["X"], y_dynamic=dynamic_data["y"],
            grp_dynamic=dynamic_data["groups"],
            sens_dynamic=dynamic_data["sensitive"],
            lmk_vals=dynamic_data["lmk_vals"],
            group_names=GROUP_NAMES[args.fair_attr],
            betas=cfg["grid_betas"], alphas=cfg["grid_alphas"],
            n_folds=cfg["n_folds"],
            eo_mode_d=cfg["eo_mode_d"],
            schedule_mode_d=cfg["schedule_mode_d"],
            n_bins=cfg["horizon"] // cfg.get("delta", 4),
            splits_static=splits_s,
            splits_dynamic=splits_d, 
            out_dir=out_dir, run_tag=run_tag,
        )
        plot_tradeoff(df_grid, out_dir=out_dir, run_tag=run_tag)
        if cfg["use_wandb"]:
             
          df_grid.to_csv(out_dir / "grid_search_results.csv", index=False)
          
          for img_path in out_dir.glob(f"*{run_tag}*.png"):
              wandb.log({f"grid_search/{img_path.stem}": wandb.Image(str(img_path))})
          

    if cfg["use_wandb"]:
        import wandb
        wandb.finish()

    print(f"\nAll outputs saved in: {out_dir}")


if __name__ == "__main__":
    main()