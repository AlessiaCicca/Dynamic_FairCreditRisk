"""
Fairness visualisation functions.
Two variants for the time-series plot:
  - plot_fairness_over_time_single : simulation 
  - plot_fairness_over_time        : real dataset 
  - plot_auc_fairness_bar   : grouped bar chart (both simulation and real)
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


# Simulation: single sensitive attribute 
def plot_separation_over_time_single(
    df_time,
    time_col,
    title,
    filename,
    out_dir,
    static_val=None,
    min_samples_per_group=50,
):
    fig, ax = plt.subplots(figsize=(8, 5))

    subset = df_time.sort_values(time_col).copy()
  
    # Sets separation to NaN where the smallest group has too few samples 
    if "n_group_min" in subset.columns:
        subset.loc[subset["n_group_min"] < min_samples_per_group, "separation"] = np.nan

    x     = subset[time_col].to_numpy(dtype=float)
    y     = subset["separation"].to_numpy(dtype=float)
    valid = ~np.isnan(y)

    if valid.sum() > 0:
       # Splits the curve at NaN gaps and plots only the valid segments —
        boundaries = np.where(np.diff(valid.astype(int)) != 0)[0] + 1
        for seg in np.split(np.arange(len(x)), boundaries):
            if valid[seg[0]]:
                ax.plot(x[seg], y[seg], marker="o", markersize=4, linewidth=2)

    # Draws a horizontal dashed line for the static model baseline 
    if static_val is not None and not np.isnan(static_val):
        ax.axhline(y=static_val, linestyle="--", linewidth=1.2,
                   alpha=0.7, label="M_STATIC")

    ax.set_title("Separation over time")
    ax.set_xlabel(time_col)
    ax.set_ylabel("Separation (lower = fairer)")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    path = out_dir / filename
    plt.savefig(path, dpi=150)
    plt.show()
    plt.close(fig)
    return path

# Real dataset: multiple sensitive attributes 
def plot_separation_over_time(
    df,
    time_col,
    title,
    filename,
    out_dir,
    static_df=None,
    attrs=None,
    min_samples_per_group=50,
):
    if attrs is None:
        attrs = ["SEX", "RACE", "AGE"]

    fig, ax = plt.subplots(figsize=(10, 5))

    for attr_name in attrs:
        color  = ATTR_COLORS.get(attr_name, "tab:gray")
        subset = df[df["attr"] == attr_name].sort_values(time_col).copy()
        if subset.empty:
            continue
          
        # Sets separation to NaN where the smallest group has too few samples 
        if "n_group_min" in subset.columns:
            subset.loc[subset["n_group_min"] < min_samples_per_group,
                       "separation"] = np.nan

        x        = subset[time_col].to_numpy(dtype=float)
        y        = subset["separation"].to_numpy(dtype=float)
        is_valid = ~np.isnan(y)
        if is_valid.sum() == 0:
            continue

        # Splits the curve at NaN gaps and plots only the valid segments 
        boundaries  = np.where(np.diff(is_valid.astype(int)) != 0)[0] + 1
        first_label = True
        for seg in np.split(np.arange(len(x)), boundaries):
            if not is_valid[seg[0]]:
                continue
            ax.plot(x[seg], y[seg], marker="o", markersize=4, color=color,
                    label=attr_name if first_label else "_nolegend_")
            first_label = False

        # Draws a horizontal dashed line for the static model baseline 
        if static_df is not None:
            static_row = static_df[
                (static_df["attr"]  == attr_name) &
                (static_df["model"] == "M_STATIC")
            ]
            if not static_row.empty and "separation" in static_row.columns:
                sv = static_row["separation"].values[0]
                if not np.isnan(sv):
                    ax.axhline(y=sv, color=color, linestyle="--",
                               linewidth=1.2, alpha=0.6,
                               label=f"{attr_name} (static)")

    ax.set_title("Separation over time")
    ax.set_xlabel(time_col)
    ax.set_ylabel("Separation (lower = fairer)")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    path = out_dir / filename
    plt.savefig(path, dpi=150)
    plt.show()
    plt.close(fig)
    return path


# Bar Chart
def plot_auc_fairness_bar(
    df_auc, out_dir, attr_name: str = "", filename: str = "fairness_auc_comparison.png"):

    models = ["AUC_M_STATIC", "AUC_M_DYNAMIC", "AUC_M_PP"]
    labels = ["M_STATIC",     "M_DYNAMIC",      "M_PP"]
    colors = ["#4C72B0",      "#DD8452",         "#55A868"]

    # one group per attr (real) or single bar (simulation)
    attrs  = df_auc["attr"].unique() if "attr" in df_auc.columns else [""]
    x      = np.arange(len(attrs))
    width  = 0.22

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (col, label, color) in enumerate(zip(models, labels, colors)):
        vals = [
            float(df_auc[df_auc["attr"] == a][col].values[0])
            if "attr" in df_auc.columns
            else float(df_auc[col].values[0])
            for a in attrs
        ]
        bars = ax.bar(x + (i - 1) * width, vals, width=width,
                      label=label, color=color,
                      edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(attrs if len(attrs) > 1 else ["Separation"], fontsize=11)
    ax.set_ylabel("AUC-separation  (↓ fairer)", fontsize=10)
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
    return path

    plt.show()
    return path
