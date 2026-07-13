"""
AUC-fairness: area under the fairness curve over time.
  - auc_fairness_single_attr FOR simulation 
  - auc_fairness_all_models FOR real dataset (SEX, RACE, AGE)
"""

import numpy as np
import pandas as pd


# Integrates the separation curve over normalized time: area under the curve as the sum of trapezoids
def _safe_trapz(sub, time_col, metric, min_samples):
    # Sort by time column to ensure correct ordering for integration
    sub = sub.copy().sort_values(time_col)

    # Remove time points with too few samples per group
    if "n_group_min" in sub.columns:
        sub = sub[sub["n_group_min"] >= min_samples]

    # Remove rows where the metric is NaN
    sub = sub.dropna(subset=[metric])
  
    # Need at least 3 points for a meaningful integral
    if len(sub) < 3:
        return np.nan
      
    # Extract time and metric values as arrays
    t = sub[time_col].values.astype(float)
    v = sub[metric].values.astype(float)
  
    # Normalize time to [0, 1] so results are comparable across different time ranges
    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-9)
  
    return float(np.trapezoid(v, t_norm))


# Simulation: single sensitive attribute 
def auc_fairness_single_attr(
    df_dynamic, df_static_agg,
    time_col_dyn="landmark",
    min_samples_per_group=20,
):
    metric     = "separation"
  
    # Filter df_static_agg to get only the M_STATIC row
    static_row = df_static_agg[df_static_agg["model"] == "M_STATIC"]
    # Extract the separation value for M_STATIC
    static_val = (
        float(static_row[metric].values[0])
        if len(static_row) > 0 and metric in static_row.columns
        else np.nan
    )
    return pd.DataFrame([{
        "metric":        metric,
        "AUC_M_STATIC":  static_val,
        "AUC_M_DYNAMIC": _safe_trapz(df_dynamic, time_col_dyn, metric, min_samples_per_group),
    }])


# Real dataset: multiple sensitive attributes 
def auc_fairness_all_models(
    df_dynamic, df_static_agg,
    time_col_dyn="landmark",
    attrs=None,
    min_samples_per_group=20,
):
    if attrs is None:
        attrs = ["SEX", "RACE", "AGE"]
    metric = "separation"
    results = []

    for attr_name in attrs:
        # Filter df_static_agg to get only the M_STATIC row
        static_row = df_static_agg[
            (df_static_agg["attr"]  == attr_name) &
            (df_static_agg["model"] == "M_STATIC")]
        # Extract the separation value for M_STATIC
        static_val = (
            float(static_row[metric].values[0])
            if len(static_row) > 0 and metric in static_row.columns
            else np.nan)
        results.append({
            "attr":          attr_name,
            "metric":        metric,
            "AUC_M_STATIC":  static_val,
            "AUC_M_DYNAMIC": _safe_trapz(df_dynamic[df_dynamic["attr"] == attr_name],
                                          time_col_dyn, metric, min_samples_per_group),
        })

    return pd.DataFrame(results)

