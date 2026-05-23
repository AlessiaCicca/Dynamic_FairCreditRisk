"""
src/evaluation/fairness_plots.py

Fairness visualisation functions.
Two variants for the time-series plot:
  - plot_fairness_over_time_single : simulation (one attribute, no color legend)
  - plot_fairness_over_time        : real dataset (SEX/RACE/AGE color-coded)
Plus:
  - plot_auc_fairness_bar   : grouped bar chart (both simulation and real)
  - plot_fairness_bootstrap : mean ± CI with reliability shading
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


ATTR_COLORS = {
    "SEX":  "tab:blue",
    "RACE": "tab:orange",
    "AGE":  "tab:green",
}


# ── Simulation: single attribute ──────────────────────────────────────────────

def plot_fairness_over_time_single(
    df_time: pd.DataFrame,
    time_col: str,
    title: str,
    filename: str,
    out_dir: Path,
    static_val_dict: dict = None,
    min_samples_per_group: int = 50,
) -> Path:
    """
    Plot independence / separation / sufficiency over time for one attribute.
    Breaks the curve where n_group_min < threshold.
    Optionally draws a horizontal dashed line for the static baseline.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ["independence", "separation", "sufficiency"]

    for ax, metric in zip(axes, metrics):
        subset = df_time.sort_values(time_col).copy()
        if "n_group_min" in subset.columns:
            subset.loc[subset["n_group_min"] < min_samples_per_group,
                       metric] = np.nan

        x     = subset[time_col].to_numpy(dtype=float)
        y     = subset[metric].to_numpy(dtype=float)
        valid = ~np.isnan(y)

        if valid.sum() > 0:
            boundaries = np.where(np.diff(valid.astype(int)) != 0)[0] + 1
            segments   = np.split(np.arange(len(x)), boundaries)
            for seg in segments:
                if not valid[seg[0]]:
                    continue
                ax.plot(x[seg], y[seg], marker="o", markersize=4, linewidth=2)

        if static_val_dict is not None:
            sv = static_val_dict.get(metric)
            if sv is not None and not np.isnan(sv):
                ax.axhline(y=sv, linestyle="--", linewidth=1.2,
                           alpha=0.7, label="STATIC")

        ax.set_title(metric.capitalize())
        ax.set_xlabel(time_col)
        ax.set_ylabel("Value (lower = fairer)")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    path = out_dir / filename
    plt.savefig(path, dpi=150)
    plt.show()
    return path


# ── Real dataset: multiple attributes ────────────────────────────────────────

def plot_fairness_over_time(
    df: pd.DataFrame,
    time_col: str,
    title: str,
    filename: str,
    out_dir: Path,
    static_df: pd.DataFrame = None,
    attrs: list = None,
    min_samples_per_group: int = 50,
) -> Path:
    """
    Plot independence / separation / sufficiency over time for multiple
    sensitive attributes (SEX, RACE, AGE), color-coded.

    Parameters
    ----------
    df            : DataFrame with columns [attr, time_col, metric, n_group_min]
    time_col      : time column name
    title         : figure title
    filename      : output filename
    out_dir       : output directory
    static_df     : aggregate DataFrame with 'attr' and 'model' columns
                    for drawing static baselines (optional)
    attrs         : list of attribute names to plot (default: SEX, RACE, AGE)
    """
    if attrs is None:
        attrs = ["SEX", "RACE", "AGE"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics   = ["independence", "separation", "sufficiency"]

    for ax, metric in zip(axes, metrics):
        for attr_name in attrs:
            color  = ATTR_COLORS.get(attr_name, "tab:gray")
            subset = df[df["attr"] == attr_name].sort_values(time_col).copy()
            if subset.empty:
                continue

            if "n_group_min" in subset.columns:
                subset.loc[subset["n_group_min"] < min_samples_per_group,
                           metric] = np.nan

            x        = subset[time_col].to_numpy(dtype=float)
            y        = subset[metric].to_numpy(dtype=float)
            is_valid = ~np.isnan(y)
            if is_valid.sum() == 0:
                continue

            boundaries  = np.where(np.diff(is_valid.astype(int)) != 0)[0] + 1
            segments    = np.split(np.arange(len(x)), boundaries)
            first_label = True
            for seg in segments:
                if not is_valid[seg[0]]:
                    continue
                ax.plot(
                    x[seg], y[seg],
                    marker="o", markersize=4, color=color,
                    label=attr_name if first_label else "_nolegend_",
                )
                first_label = False

            # Static baseline
            if static_df is not None:
                static_row = static_df[
                    (static_df["attr"]  == attr_name) &
                    (static_df["model"] == "M_STATIC")
                ]
                if not static_row.empty and metric in static_row.columns:
                    sv = static_row[metric].values[0]
                    if not np.isnan(sv):
                        ax.axhline(
                            y=sv, color=color, linestyle="--",
                            linewidth=1.2, alpha=0.6,
                            label=f"{attr_name} (static)",
                        )

        ax.set_title(metric.capitalize())
        ax.set_xlabel(time_col)
        ax.set_ylabel("Value (lower = fairer)")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    path = out_dir / filename
    plt.savefig(path, dpi=150)
    plt.show()
    return path


# ── AUC fairness bar chart ────────────────────────────────────────────────────

def plot_auc_fairness_bar(
    df_auc: pd.DataFrame,
    out_dir: Path,
    attr_name: str = "",
    filename: str = "fairness_auc_comparison.png",
) -> Path:
    """
    Grouped bar chart comparing AUC-fairness across models.

    Works for both:
    - simulation (df_auc has columns [metric, AUC_M_STATIC, AUC_M_DYNAMIC, AUC_M_PP])
    - real dataset (same columns, one attr at a time or pre-filtered)
    """
    metrics = df_auc["metric"].tolist()
    models  = ["AUC_M_STATIC", "AUC_M_DYNAMIC", "AUC_M_PP"]
    labels  = ["M_STATIC",     "M_DYNAMIC",      "M_PP"]
    colors  = ["#4C72B0",      "#DD8452",         "#55A868"]
    x       = np.arange(len(metrics))
    width   = 0.22

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (col, label, color) in enumerate(zip(models, labels, colors)):
        vals = df_auc[col].to_numpy(dtype=float)
        bars = ax.bar(x + (i - 1) * width, vals, width=width,
                      label=label, color=color,
                      edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics], fontsize=11)
    ax.set_ylabel("AUC-fairness  (↓ fairer)", fontsize=10)
    suffix = f" — {attr_name}" if attr_name else ""
    ax.set_title(f"Fairness comparison{suffix}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    path = out_dir / filename
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Bootstrap CI plot ─────────────────────────────────────────────────────────

def plot_fairness_bootstrap(
    df_boot: pd.DataFrame,
    time_col: str,
    title: str,
    out_dir: Path,
    filename: str,
) -> Path:
    """
    Plot mean fairness curve with bootstrap CI bands.
    Unreliable time points are shaded in orange.
    """
    metrics = ["independence", "separation", "sufficiency"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, metric in zip(axes, metrics):
        sub = df_boot.sort_values(time_col).copy()
        x   = sub[time_col].values.astype(float)
        y   = sub[metric].values.astype(float)
        lo  = sub[f"{metric}_ci_lo"].values.astype(float)
        hi  = sub[f"{metric}_ci_hi"].values.astype(float)
        rel = sub[f"{metric}_reliable"].values

        ax.plot(x, y, marker="o", linewidth=2,
                color="#2196F3", label="mean")

        for i in range(len(x)):
            color = "#2196F3" if rel[i] else "#FF9800"
            ax.fill_between(
                [x[i] - 0.3, x[i] + 0.3],
                [lo[i], lo[i]], [hi[i], hi[i]],
                alpha=0.3, color=color,
            )

        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"{metric} (orange = unreliable)")
        ax.set_xlabel(time_col)
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    path = out_dir / filename
    plt.savefig(path, dpi=150)
    plt.show()
    return path
