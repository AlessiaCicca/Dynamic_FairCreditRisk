"""
src/training/grid_search.py

Grid search on the AUC vs Separation trade-off for the three MLP models.
Separation is measured as AUC of the fairness curve over time,
consistent with the main run.

Usage (from experiments/run_simulation.py after building datasets):

    from src.training.grid_search import run_grid_search, plot_tradeoff

    df_grid = run_grid_search(
        X_static=X_static, y_static=y_static, grp_static=grp_static,
        sens_static=sens_static,
        X_dynamic=X_dynamic, y_dynamic=y_dynamic, grp_dynamic=grp_dynamic,
        sens_dynamic=sens_dynamic, lmk_vals=lmk_vals,
        X_pp=X_pp, y_pp=y_pp, grp_pp=grp_pp,
        sens_pp=sens_pp, pp_ages=pp_ages,
        group_names=GROUP_NAMES,
        eo_mode_d=EO_MODE_D, eo_mode_p=EO_MODE_P,
        out_dir=OUT_DIR, run_tag=run_tag,
    )
    plot_tradeoff(df_grid, out_dir=OUT_DIR, run_tag=run_tag)
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

# ── Default sweep values ──────────────────────────────────────────────────────

DEFAULT_BETAS  = [0.0, 0.3, 0.5, 0.7, 1.0]
DEFAULT_ALPHAS = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2]
DEFAULT_GAMMAS = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2]

MODEL_STYLES = {
    "M_STATIC":  {"color": "#3A6BC4", "marker": "o", "coef_label": "β"},
    "M_DYNAMIC": {"color": "#D4612A", "marker": "s", "coef_label": "α"},
    "M_PP":      {"color": "#2E8B57", "marker": "^", "coef_label": "γ"},
}


# ── Single CV run ─────────────────────────────────────────────────────────────

def _run_cv(
    model_tag: str,
    X, y, grp, sens, time_arr,
    beta: float, alpha: float, gamma: float,
    group_names: dict,
    n_folds: int = 5,
    eo_mode_d: str = "mean",
    eo_mode_p: str = "mean",
    schedule_mode_d: str = "flat",
    schedule_mode_p: str = "flat",
) -> dict:
    """Run GroupKFold CV for one model / one coefficient value."""

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    gkf       = GroupKFold(n_splits=n_folds)
    oof_preds = np.zeros(len(y), dtype=np.float64)

    for tr_idx, te_idx in gkf.split(X, y, grp):
        time_tr = time_arr[tr_idx] if time_arr is not None else None
        p_te, _, _, _ = train_mlp(
            X[tr_idx], y[tr_idx], X[te_idx], y[te_idx],
            sensitive_tr    = sens[tr_idx],
            time_tr         = time_tr,
            subj_ids_tr     = grp[tr_idx],
            model_name      = model_tag,
            beta=beta, alpha=alpha, gamma=gamma,
            eo_mode_d       = eo_mode_d,
            eo_mode_p       = eo_mode_p,
            schedule_mode_d = schedule_mode_d,
            schedule_mode_p = schedule_mode_p,
            verbose         = False,
        )
        oof_preds[te_idx] = p_te

    th = find_best_threshold(y.astype(int), oof_preds)

    # ── Mean OOF AUC ─────────────────────────────────────────────────────────
    fold_aucs = []
    for _, te_idx in gkf.split(X, y, grp):
        if len(np.unique(y[te_idx])) > 1:
            fold_aucs.append(
                roc_auc_score(y[te_idx].astype(int), oof_preds[te_idx])
            )

    # ── Fairness AUC ─────────────────────────────────────────────────────────
    if time_arr is not None:
        time_rows = []
        for t in sorted(np.unique(time_arr)):
            mask = time_arr == t
            if mask.sum() < 20:
                continue
            yt_f, yp_f, sn_f = filter_sensitive(
                y[mask].astype(int), oof_preds[mask], sens[mask]
            )
            if len(np.unique(yt_f)) < 2 or len(np.unique(sn_f)) < 2:
                continue
            yb_f = (yp_f >= th).astype(int)
            res  = fairness_metrics(yt_f, yp_f, yb_f, sn_f,
                                    group_names, threshold=th)
            axioms = res.get("axioms", {})
            time_rows.append({
                "t":            t,
                "separation":   axioms.get("separation",   np.nan),
                "independence": axioms.get("independence", np.nan),
                "sufficiency":  axioms.get("sufficiency",  np.nan),
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
        ind_auc  = trapz_norm("independence")
        suf_auc  = trapz_norm("sufficiency")
        sep_mean = df_t["separation"].mean() if not df_t.empty else np.nan

    else:
        yt_f, yp_f, sn_f = filter_sensitive(
            y.astype(int), oof_preds, sens
        )
        yb_f   = (yp_f >= th).astype(int)
        res    = fairness_metrics(yt_f, yp_f, yb_f, sn_f,
                                  group_names, threshold=th)
        axioms = res.get("axioms", {})
        sep_auc  = axioms.get("separation",   np.nan)
        ind_auc  = axioms.get("independence", np.nan)
        suf_auc  = axioms.get("sufficiency",  np.nan)
        sep_mean = sep_auc

    return {
        "auc_mean":         float(np.nanmean(fold_aucs)) if fold_aucs else np.nan,
        "separation_auc":   sep_auc,
        "separation_mean":  sep_mean,
        "independence_auc": ind_auc,
        "sufficiency_auc":  suf_auc,
        "threshold":        float(th),
    }


# ── Main grid search ──────────────────────────────────────────────────────────

def run_grid_search(
    X_static, y_static, grp_static, sens_static,
    X_dynamic, y_dynamic, grp_dynamic, sens_dynamic, lmk_vals,
    X_pp, y_pp, grp_pp, sens_pp, pp_ages,
    group_names: dict,
    betas:  list = None,
    alphas: list = None,
    gammas: list = None,
    n_folds: int = 5,
    eo_mode_d: str = "mean",
    eo_mode_p: str = "mean",
    schedule_mode_d: str = "flat",
    schedule_mode_p: str = "flat",
    out_dir: Path = Path("outputs"),
    run_tag: str = "run",
) -> pd.DataFrame:
    """
    Run grid search over fairness penalty coefficients for all three models.

    Returns
    -------
    pd.DataFrame with one row per (model, coefficient value)
    """
    if betas  is None: betas  = DEFAULT_BETAS
    if alphas is None: alphas = DEFAULT_ALPHAS
    if gammas is None: gammas = DEFAULT_GAMMAS

    records = []

    # ── M_STATIC ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("GRID SEARCH — M_STATIC")
    print("=" * 60)
    for beta in betas:
        print(f"  beta={beta:.2f} ...", end=" ", flush=True)
        r = _run_cv(
            "static", X_static, y_static, grp_static, sens_static, None,
            beta=beta, alpha=0.0, gamma=0.0,
            group_names=group_names, n_folds=n_folds,
            eo_mode_d=eo_mode_d, eo_mode_p=eo_mode_p,
        )
        records.append({"model": "M_STATIC", "coef": beta,
                         "coef_name": "beta", **r})
        print(f"AUC={r['auc_mean']:.4f}  sep_auc={r['separation_auc']:.4f}")

    # ── M_DYNAMIC ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GRID SEARCH — M_DYNAMIC")
    print("=" * 60)
    for alpha in alphas:
        print(f"  alpha={alpha:.2f} ...", end=" ", flush=True)
        r = _run_cv(
            "dynamic", X_dynamic, y_dynamic, grp_dynamic, sens_dynamic,
            lmk_vals, beta=0.0, alpha=alpha, gamma=0.0,
            group_names=group_names, n_folds=n_folds,
            eo_mode_d=eo_mode_d, eo_mode_p=eo_mode_p,
            schedule_mode_d=schedule_mode_d,
        )
        records.append({"model": "M_DYNAMIC", "coef": alpha,
                         "coef_name": "alpha", **r})
        print(f"AUC={r['auc_mean']:.4f}  sep_auc={r['separation_auc']:.4f}")

    # ── M_PP ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GRID SEARCH — M_PP")
    print("=" * 60)
    for gamma in gammas:
        print(f"  gamma={gamma:.2f} ...", end=" ", flush=True)
        r = _run_cv(
            "person_period", X_pp, y_pp, grp_pp, sens_pp, pp_ages,
            beta=0.0, alpha=0.0, gamma=gamma,
            group_names=group_names, n_folds=n_folds,
            eo_mode_d=eo_mode_d, eo_mode_p=eo_mode_p,
            schedule_mode_p=schedule_mode_p,
        )
        records.append({"model": "M_PP", "coef": gamma,
                         "coef_name": "gamma", **r})
        print(f"AUC={r['auc_mean']:.4f}  sep_auc={r['separation_auc']:.4f}")

    df_grid = pd.DataFrame(records)
    csv_path = out_dir / f"grid_tradeoff_{run_tag}.csv"
    df_grid.to_csv(csv_path, index=False)
    print(f"\n✓ Saved: {csv_path}")
    print(df_grid.to_string(index=False))

    # Best points summary
    _print_best_points(df_grid, out_dir)

    return df_grid


def _print_best_points(df_grid: pd.DataFrame, out_dir: Path) -> None:
    """Print and save best trade-off points per model."""
    auc_min = df_grid["auc_mean"].min()
    auc_max = df_grid["auc_mean"].max()
    sep_min = df_grid["separation_auc"].min()
    sep_max = df_grid["separation_auc"].max()

    print("\n=== BEST POINTS (max AUC − normalised Separation AUC) ===")
    summary_rows = []
    for model_name in ["M_STATIC", "M_DYNAMIC", "M_PP"]:
        sub = (
            df_grid[df_grid["model"] == model_name]
            .dropna(subset=["auc_mean", "separation_auc"])
            .reset_index(drop=True)
        )
        if sub.empty:
            continue
        aucs     = sub["auc_mean"].to_numpy()
        seps     = sub["separation_auc"].to_numpy()
        auc_norm = (aucs - auc_min) / (auc_max - auc_min + 1e-9)
        sep_norm = (seps - sep_min) / (sep_max - sep_min + 1e-9)
        best     = sub.iloc[np.argmax(auc_norm - sep_norm)]
        summary_rows.append({
            "model":            model_name,
            "best_coef":        best["coef"],
            "coef_name":        best["coef_name"],
            "auc_mean":         round(best["auc_mean"],         4),
            "separation_auc":   round(best["separation_auc"],   4),
            "independence_auc": round(best["independence_auc"], 4),
            "sufficiency_auc":  round(best["sufficiency_auc"],  4),
        })
        print(f"  {model_name:<12}  {best['coef_name']}={best['coef']:.2f}"
              f"  AUC={best['auc_mean']:.4f}"
              f"  sep_auc={best['separation_auc']:.4f}")

    df_best = pd.DataFrame(summary_rows)
    out_path = out_dir / "grid_best_points.csv"
    df_best.to_csv(out_path, index=False)
    print(f"\n✓ Saved: {out_path}")


# ── Trade-off plot ────────────────────────────────────────────────────────────

def plot_tradeoff(df_grid: pd.DataFrame, out_dir: Path,
                  run_tag: str = "run") -> Path:
    """
    One panel per model: AUC (solid) and Separation AUC (dashed) vs λ.
    Stars mark the best trade-off point.
    """
    auc_min = df_grid["auc_mean"].min()
    auc_max = df_grid["auc_mean"].max()
    sep_min = df_grid["separation_auc"].min()
    sep_max = df_grid["separation_auc"].max()

    def trade_score(auc, sep):
        auc_n = (auc - auc_min) / (auc_max - auc_min + 1e-9)
        sep_n = (sep - sep_min) / (sep_max - sep_min + 1e-9)
        return auc_n - sep_n

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle(
        "AUC and Separation AUC as a function of λ — one panel per model\n"
        "★ marks the best trade-off point (max AUC − min Separation AUC, normalised)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    for ax, (model_name, style) in zip(axes, MODEL_STYLES.items()):
        sub = (
            df_grid[df_grid["model"] == model_name]
            .dropna(subset=["auc_mean", "separation_auc"])
            .sort_values("coef")
            .reset_index(drop=True)
        )
        if sub.empty:
            ax.set_title(f"{model_name} — no data")
            continue

        coefs    = sub["coef"].to_numpy()
        aucs     = sub["auc_mean"].to_numpy()
        seps     = sub["separation_auc"].to_numpy()
        color    = style["color"]
        clabel   = style["coef_label"]
        scores   = np.array([trade_score(a, s) for a, s in zip(aucs, seps)])
        best_idx = np.argmax(scores)

        ax2 = ax.twinx()
        ax.plot(coefs, aucs, color=color, linewidth=2.2,
                marker=style["marker"], markersize=7, zorder=3)
        ax2.plot(coefs, seps, color=color, linewidth=2.2, linestyle="--",
                 marker=style["marker"], markersize=7, alpha=0.55, zorder=3)

        for k, (c, a) in enumerate(zip(coefs, aucs)):
            ax.annotate(
                f"{clabel}={c:.1f}",
                xy=(c, a), xytext=(0, 6), textcoords="offset points",
                fontsize=7, color=color, ha="center", va="bottom",
            )

        ax.scatter([coefs[best_idx]], [aucs[best_idx]],
                   s=320, marker="*", color="gold",
                   edgecolors=color, linewidths=1.5, zorder=6)
        ax2.scatter([coefs[best_idx]], [seps[best_idx]],
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
        ax.set_ylabel("AUC  (↑ higher is better)", fontsize=10, color=color)
        ax2.set_ylabel("Separation AUC  (↓ fairer)", fontsize=10, color=color)
        ax.tick_params(axis="y", labelcolor=color)
        ax2.tick_params(axis="y", labelcolor=color)
        ax.set_title(model_name, fontsize=12, fontweight="bold",
                     color=color, pad=10)
        ax.grid(alpha=0.2, linestyle="--")

        legend_elems = [
            Line2D([0], [0], color=color, linewidth=2,
                   marker=style["marker"], markersize=7,
                   label="AUC (solid)"),
            Line2D([0], [0], color=color, linewidth=2, linestyle="--",
                   marker=style["marker"], markersize=7, alpha=0.55,
                   label="Separation AUC (dashed)"),
            Line2D([0], [0], marker="*", color="gold", markersize=11,
                   markeredgecolor=color, linewidth=0,
                   label="Best trade-off (★)"),
        ]
        ax.legend(handles=legend_elems, fontsize=8,
                  loc="lower left", framealpha=0.9)

    plt.tight_layout()
    plot_path = out_dir / f"tradeoff_{run_tag}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✓ Saved: {plot_path}")
    return plot_path
