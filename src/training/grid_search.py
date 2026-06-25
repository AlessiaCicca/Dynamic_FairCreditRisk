"""
Grid search on the AUC vs Separation trade-off for the two MLP models.
Separation is measured as AUC of the fairness curve over time,
consistent with the main run.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from matplotlib.lines import Line2D

import torch

from src.training.train_mlp import train_mlp
from src.evaluation.fairness_metrics import (
    fairness_metrics, filter_sensitive)
from src.training.cross_validation import find_best_threshold

warnings.filterwarnings("ignore")


DEFAULT_BETAS  = [0.0, 0.3, 0.5, 0.7, 1.0]
DEFAULT_ALPHAS = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2]


MODEL_STYLES = {
    "M_STATIC":  {"color": "#3A6BC4", "marker": "o", "coef_label": "β"},
    "M_DYNAMIC": {"color": "#D4612A", "marker": "s", "coef_label": "α"},
}

def _collapse_fold(hazard, event_bin, ids, lmk, n_bins, complete_only=True):
    h = np.clip(hazard, 1e-7, 1 - 1e-7)
    d = pd.DataFrame({
        "id": ids, "L": lmk,
        "log1mh": np.log1p(-h),
        "ev": event_bin,
    })
    g = d.groupby(["id", "L"], sort=False)
    out = pd.DataFrame({
        "pdh": 1.0 - np.exp(g["log1mh"].sum()),
        "yh":  g["ev"].max(),
        "n":   g.size(),
    }).reset_index()
    if complete_only:
        out = out[out["n"] == n_bins]
    return out
    


# MAIN RUN to which we will assign all the different value for the coefficients
def _run_cv(
    model_tag,
    X, y, grp, sens, time_arr,
    beta, alpha,
    group_names,
    n_folds=5,
    eo_mode_d="mean",
    schedule_mode_d="flat",
    fixed_th=None,          
    n_bins=None):            

    # Splits the data into 5 folds ensuring that the same loan is not both in train and test
    gkf       = GroupKFold(n_splits=n_folds)
    # Empty Array to save OOF predictions
    oof_preds = np.zeros(len(y), dtype=np.float64)

    thresholds = []
    fold_aucs = []   

    # For each fold: train the model on the train, predict on the test
    for tr_idx, te_idx in gkf.split(X, y, grp):
        time_tr = time_arr[tr_idx] if time_arr is not None else None
        p_te, p_tr, _, _ = train_mlp(
            X[tr_idx], y[tr_idx], X[te_idx], y[te_idx],
            sensitive_tr    = sens[tr_idx],
            time_tr         = time_tr,
            subj_ids_tr     = grp[tr_idx],
            model_name      = model_tag,
            beta=beta, alpha=alpha,
            eo_mode_d       = eo_mode_d,
            schedule_mode_d = schedule_mode_d,
            verbose         = False,
        )
        oof_preds[te_idx] = p_te
        # Find the threshold that maximize F1
        if time_arr is not None:
            tr_pdh = _collapse_fold(p_tr, y[tr_idx], grp[tr_idx], time_arr[tr_idx], n_bins)
            te_pdh = _collapse_fold(p_te, y[te_idx], grp[te_idx], time_arr[te_idx], n_bins)
            best_th = find_best_threshold(tr_pdh["yh"], tr_pdh["pdh"])
            if len(np.unique(te_pdh["yh"])) > 1:
                fold_aucs.append(roc_auc_score(te_pdh["yh"].astype(int), te_pdh["pdh"]))
        else:
            best_th = find_best_threshold(y[tr_idx], p_tr)
            if len(np.unique(y[te_idx])) > 1:
                fold_aucs.append(roc_auc_score(y[te_idx].astype(int), p_te))
        thresholds.append(best_th)

    th = float(np.mean(thresholds))

    #  hazard per-bin -> PD-H 
    if time_arr is not None:
        if n_bins is None:
            raise ValueError("n_bins required")
        h = np.clip(oof_preds, 1e-7, 1 - 1e-7)
        dfp = pd.DataFrame({
            "id": grp, "L": time_arr,
            "log1mh": np.log1p(-h),
            "ev": y,
            "sens": sens,
        })
        g    = dfp.groupby(["id", "L"], sort=False)
        surv = np.exp(g["log1mh"].sum())
        pdh  = (1.0 - surv).rename("pdh")
        yh   = g["ev"].max().rename("yh")
        sh   = g["sens"].first().rename("sh") 
        cnt  = g.size().rename("n")
        Lh   = g["L"].first().rename("Lh")
        coll = pd.concat([pdh, yh, sh, cnt, Lh], axis=1).reset_index(drop=True)
        coll = coll[coll["n"] == n_bins]       
        
        eval_preds = coll["pdh"].to_numpy()
        eval_y     = coll["yh"].to_numpy().astype(int)
        eval_sens  = coll["sh"].to_numpy()
        eval_time  = coll["Lh"].to_numpy()
    else:
        eval_preds = oof_preds
        eval_y     = y.astype(int)
        eval_sens  = sens
        eval_time  = None

  
    def compute_separation(eval_th):
        if eval_time is not None:
            time_rows = []
            for t in sorted(np.unique(eval_time)):
                mask = eval_time == t
                yt_f, yp_f, sn_f = filter_sensitive(
                    eval_y[mask], eval_preds[mask], eval_sens[mask]
                )
                if len(np.unique(yt_f)) < 2 or len(np.unique(sn_f)) < 2:
                    continue
                counts = pd.Series(sn_f).value_counts()
                if counts.min() < 50:
                    continue
                
                yb_f = (yp_f >= eval_th).astype(int)
                res  = fairness_metrics(yt_f, yp_f, yb_f, sn_f,
                                        group_names, threshold=eval_th)
                axioms = res.get("axioms", {})
                time_rows.append({
                    "t":          t,
                    "separation": axioms.get("separation", np.nan),
                })

            df_t = pd.DataFrame(time_rows)

            def trapz_norm(col):
                sub = df_t.dropna(subset=[col])
                if len(sub) < 3:
                    return np.nan
                t_v = sub["t"].values.astype(float)
                v   = sub[col].values.astype(float)
                t_n = (t_v - t_v.min()) / (t_v.max() - t_v.min() + 1e-9)
                return float(np.trapezoid(v, t_n))

            sep_auc  = trapz_norm("separation")
            sep_mean = df_t["separation"].mean() if not df_t.empty else np.nan
            return sep_auc, sep_mean
        else:
            yt_f, yp_f, sn_f = filter_sensitive(
                eval_y, eval_preds, eval_sens
            )
            yb_f   = (yp_f >= eval_th).astype(int)
            res    = fairness_metrics(yt_f, yp_f, yb_f, sn_f,
                                      group_names, threshold=eval_th)
            axioms = res.get("axioms", {})
            s = axioms.get("separation", np.nan)
            return s, s

    sep_auc, sep_mean = compute_separation(th)

    if fixed_th is not None:
        sep_auc_fixed, sep_mean_fixed = compute_separation(float(fixed_th))
    else:
        sep_auc_fixed, sep_mean_fixed = np.nan, np.nan

    return {
        "auc_mean":              float(np.nanmean(fold_aucs)) if fold_aucs else np.nan,
        "separation_auc":        sep_auc,
        "separation_mean":       sep_mean,
        "separation_auc_fixed":  sep_auc_fixed,
        "separation_mean_fixed": sep_mean_fixed,
        "threshold":             float(th),
        "fixed_threshold":       float(fixed_th) if fixed_th is not None else np.nan,
    }


# Main grid search run
def run_grid_search(
    X_static, y_static, grp_static, sens_static,
    X_dynamic, y_dynamic, grp_dynamic, sens_dynamic, lmk_vals,
    group_names,
    betas=None,
    alphas=None,
    n_folds=5,
    eo_mode_d="mean",
    schedule_mode_d="flat",
    n_bins=None,               
    out_dir=Path("outputs"),
    run_tag="run"):
    
    # Seed setup
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    if betas  is None: betas  = DEFAULT_BETAS
    if alphas is None: alphas = DEFAULT_ALPHAS

    records = []

    # M_STATIC
    print("=" * 60)
    print("GRID SEARCH — M_STATIC")
    print("=" * 60)
    static_base_th = None     
    for beta in betas:
        print(f"  beta={beta:.2f} ...", end=" ", flush=True)
        r = _run_cv(
            "static", X_static, y_static, grp_static, sens_static, None,
            beta=beta, alpha=0.0,
            group_names=group_names, n_folds=n_folds,
            eo_mode_d=eo_mode_d,
            fixed_th=static_base_th,
        )
        if beta == 0.0:
            static_base_th = r["threshold"]
            r["separation_auc_fixed"]  = r["separation_auc"]
            r["separation_mean_fixed"] = r["separation_mean"]
            r["fixed_threshold"]       = static_base_th
        records.append({"model": "M_STATIC", "coef": beta,
                         "coef_name": "beta", **r})
        print(f"AUC={r['auc_mean']:.4f}  sep={r['separation_auc']:.4f}  "
              f"sep_fix={r['separation_auc_fixed']:.4f}")

    # M_DYNAMIC
    print("\n" + "=" * 60)
    print("GRID SEARCH — M_DYNAMIC")
    print("=" * 60)
    dyn_base_th = None    
    for alpha in alphas:
        print(f"  alpha={alpha:.2f} ...", end=" ", flush=True)
        r = _run_cv(
            "dynamic", X_dynamic, y_dynamic, grp_dynamic, sens_dynamic,
            lmk_vals, beta=0.0, alpha=alpha,
            group_names=group_names, n_folds=n_folds,
            eo_mode_d=eo_mode_d,
            schedule_mode_d=schedule_mode_d,
            fixed_th=dyn_base_th,
            n_bins=n_bins,
        )
        if alpha == 0.0:
            dyn_base_th = r["threshold"]
            r["separation_auc_fixed"]  = r["separation_auc"]
            r["separation_mean_fixed"] = r["separation_mean"]
            r["fixed_threshold"]       = dyn_base_th
        records.append({"model": "M_DYNAMIC", "coef": alpha,
                         "coef_name": "alpha", **r})
        print(f"AUC={r['auc_mean']:.4f}  sep={r['separation_auc']:.4f}  "
              f"sep_fix={r['separation_auc_fixed']:.4f}")

    df_grid = pd.DataFrame(records)
    csv_path = out_dir / f"grid_tradeoff_{run_tag}.csv"
    df_grid.to_csv(csv_path, index=False)
    print(df_grid.to_string(index=False))

    print_best_points(df_grid, out_dir)

    return df_grid


# Compute best trade-off point for each model
def _compute_best(df_grid):
    """
    M_STATIC:  minimizes separation using normalized AUC vs sep trade-off score
    M_DYNAMIC: minimizes separation subject to AUC >= static baseline AUC (beta=0)
    """

    static_baseline = df_grid[
        (df_grid["model"] == "M_STATIC") & (df_grid["coef"] == 0.0)
    ]["auc_mean"].values
    static_auc = float(static_baseline[0]) if len(static_baseline) > 0 else 0.0

    sub_s_all = df_grid[df_grid["model"] == "M_STATIC"]\
                    .dropna(subset=["auc_mean", "separation_auc"])\
                    .reset_index(drop=True)
    auc_min = sub_s_all["auc_mean"].min()
    auc_max = sub_s_all["auc_mean"].max()
    sep_min = sub_s_all["separation_auc"].min()
    sep_max = sub_s_all["separation_auc"].max()

    def trade_score(auc, sep):
        auc_n = (auc - auc_min) / (auc_max - auc_min + 1e-9)
        sep_n = (sep - sep_min) / (sep_max - sep_min + 1e-9)
        return auc_n - sep_n

    best_per_model = {}

    if not sub_s_all.empty:
        scores = np.array([trade_score(a, s) for a, s in
                           zip(sub_s_all["auc_mean"], sub_s_all["separation_auc"])])
        best_per_model["M_STATIC"] = sub_s_all.iloc[np.argmax(scores)]

    sub_d = df_grid[df_grid["model"] == "M_DYNAMIC"]\
                .dropna(subset=["auc_mean", "separation_auc"])\
                .reset_index(drop=True)
    feasible_d = sub_d[sub_d["auc_mean"] >= static_auc]
    if feasible_d.empty:
        feasible_d = sub_d  # fallback
    best_per_model["M_DYNAMIC"] = feasible_d.loc[
        feasible_d["separation_auc"].idxmin()
    ]

    return best_per_model, trade_score


def print_best_points(df_grid, out_dir):
    best_per_model, _ = _compute_best(df_grid)

    print("\n=== BEST COEFFICIENT ===")
    summary_rows = []
    for model_name, best in best_per_model.items():
        summary_rows.append({
            "model":          model_name,
            "best_coef":      best["coef"],
            "coef_name":      best["coef_name"],
            "auc_mean":       round(best["auc_mean"],       4),
            "separation_auc": round(best["separation_auc"], 4),
        })
        print(f"  {model_name:<12}  {best['coef_name']}={best['coef']:.2f}"
              f"  AUC={best['auc_mean']:.4f}"
              f"  sep_auc={best['separation_auc']:.4f}")

    pd.DataFrame(summary_rows).to_csv(out_dir / "grid_best_points.csv", index=False)


def plot_tradeoff(df_grid, out_dir, run_tag="run"):
    best_per_model, trade_score = _compute_best(df_grid)

    # Static baseline AUC (beta=0) — used as constraint for dynamic
    static_auc_baseline = df_grid[
        (df_grid["model"] == "M_STATIC") & (df_grid["coef"] == 0.0)
    ]["auc_mean"].values
    static_auc_baseline = float(static_auc_baseline[0]) if len(static_auc_baseline) > 0 else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("AUC and Separation AUC as a function of λ\n",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax, (model_name, style) in zip(axes, MODEL_STYLES.items()):
        sub = df_grid[df_grid["model"] == model_name]\
                .dropna(subset=["auc_mean", "separation_auc"])\
                .sort_values("coef").reset_index(drop=True)

        if sub.empty:
            ax.set_title(f"{model_name} — no data")
            continue

        coefs    = sub["coef"].to_numpy()
        aucs     = sub["auc_mean"].to_numpy()
        seps     = sub["separation_auc"].to_numpy()
        color    = style["color"]
        clabel   = style["coef_label"]

        # Different best_idx criterion for static vs dynamic
        if model_name == "M_DYNAMIC":
            feasible_mask = aucs >= static_auc_baseline
            if feasible_mask.any():
                feasible_seps = np.where(feasible_mask, seps, np.inf)
                best_idx = int(np.argmin(feasible_seps))
            else:
                best_idx = int(np.argmin(seps))
        else:
            best_idx = int(np.argmax([trade_score(a, s) for a, s in zip(aucs, seps)]))

        ax2 = ax.twinx()
        ax.plot(coefs, aucs, color=color, linewidth=2.2,
                marker=style["marker"], markersize=7, zorder=3)
        ax2.plot(coefs, seps, color=color, linewidth=2.2, linestyle="--",
                 marker=style["marker"], markersize=7, alpha=0.55, zorder=3)

        # Draw vertical line at static_auc_baseline for dynamic plot
        if model_name == "M_DYNAMIC":
            ax.axhline(y=static_auc_baseline, color="gray", linestyle=":",
                       linewidth=1.5, alpha=0.7, label=f"Static AUC baseline ({static_auc_baseline:.3f})")

        for c, a in zip(coefs, aucs):
            ax.annotate(f"{clabel}={c:.1f}", xy=(c, a),
                        xytext=(0, 6), textcoords="offset points",
                        fontsize=7, color=color, ha="center", va="bottom")

        for vals, ax_ in [(aucs, ax), (seps, ax2)]:
            ax_.scatter([coefs[best_idx]], [vals[best_idx]],
                        s=320, marker="*", color="gold",
                        edgecolors=color, linewidths=1.5, zorder=6)

        ax.annotate(
            f"best: {clabel}={coefs[best_idx]:.1f}\n"
            f"AUC={aucs[best_idx]:.3f}\n"
            f"sep_auc={seps[best_idx]:.4f}",
            xy=(coefs[best_idx], aucs[best_idx]),
            xytext=(12, -28), textcoords="offset points",
            fontsize=8, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.85),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
        )

        ax.set_xlabel(f"λ  ({clabel})", fontsize=11)
        ax.set_ylabel("AUC  (↑ higher)", fontsize=10, color=color)
        ax2.set_ylabel("Separation AUC  (↓ fairer)", fontsize=10, color=color)
        ax.tick_params(axis="y", labelcolor=color)
        ax2.tick_params(axis="y", labelcolor=color)
        ax.set_title(model_name, fontsize=12, fontweight="bold", color=color)
        ax.grid(alpha=0.2, linestyle="--")

        legend_handles = [
            Line2D([0], [0], color=color, linewidth=2,
                   marker=style["marker"], label="AUC (solid)"),
            Line2D([0], [0], color=color, linewidth=2, linestyle="--",
                   marker=style["marker"], alpha=0.55, label="Separation AUC (dashed)"),
            Line2D([0], [0], marker="*", color="gold", markersize=11,
                   markeredgecolor=color, linewidth=0, label="Best λ* (★)"),
        ]
        if model_name == "M_DYNAMIC":
            legend_handles.append(
                Line2D([0], [0], color="gray", linestyle=":", linewidth=1.5,
                       label=f"Static AUC baseline ({static_auc_baseline:.3f})")
            )
        ax.legend(handles=legend_handles, fontsize=8, loc="lower left", framealpha=0.9)

    plt.tight_layout()
    plot_path = out_dir / f"tradeoff_{run_tag}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    return plot_path
