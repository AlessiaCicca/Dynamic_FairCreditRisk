"""
Core fairness metrics: group-level statistics, separation and adTPR/adFPR

"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve


# Remove rows where sensitive attribute is not 0 or 1
def filter_sensitive(y_true, y_pred,sens_arr):
    valid = np.isin(sens_arr, [0, 1])
    return y_true[valid], y_pred[valid], sens_arr[valid]


def fairness_metrics(y_true, y_pred_proba, y_bin, sensitive, group_names, threshold):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred_proba, dtype=float)
    y_bin  = np.asarray(y_bin, dtype=int)
    sens   = np.asarray(sensitive)

    results = {"threshold": threshold}
    groups  = [g for g in [0, 1] if g in sens]

    # Iterates over S=0 and S=1 separately
    for g in groups:
        mask  = sens == g
        yt_g  = y_true[mask]
        yp_g  = y_pred[mask]
        yb_g  = y_bin[mask]
        n     = mask.sum()
        n_pos = (yt_g == 1).sum()
        n_neg = (yt_g == 0).sum()
        name  = group_names[g]

        # Counts true positives, false positives and false negatives for each group
        tp = ((yb_g == 1) & (yt_g == 1)).sum()
        fp = ((yb_g == 1) & (yt_g == 0)).sum()
        fn = ((yb_g == 0) & (yt_g == 1)).sum()

        results[name] = {
            "n":          int(n),
            "prev":       float(n_pos / n)       if n > 0      else np.nan,
            "tpr":        float(tp / n_pos)       if n_pos > 0  else np.nan,
            "fpr":        float(fp / n_neg)       if n_neg > 0  else np.nan,
            "fnr":        float(fn / n_pos)       if n_pos > 0  else np.nan,
        }

    if len(groups) == 2:
        m = results[group_names[0]]
        f = results[group_names[1]]
        # Measures how differently the model treats S=0 and S=1 given the true outcome.
        results["axioms"] = {
            "separation":   abs(
                (f["fpr"] - m["fpr"]) + (f["fnr"] - m["fnr"])
            ) / 2,
        }
    return results


def print_fairness_report(model_name, res,group_names, label: str = "AGGREGATE"):
    priv_name = group_names[0]
    prot_name = group_names[1]
    print(f"\n{'─'*50}")
    print(f"  {model_name}  [{label}]  (threshold = {res['threshold']:.4f})")
    print(f"{'─'*50}")
    print(f"  {'Metric':<22} {priv_name:>14} {prot_name:>14}")
    print(f"  {'-'*50}")

    rows = [
        ("N observations",    "n",    ".0f"),
        ("Base rate (prev.)", "prev", ".4f"),
        ("TPR (recall)",      "tpr",  ".4f"),
        ("FPR",               "fpr",  ".4f"),
        ("FNR (miss rate)",   "fnr",  ".4f"),
    ]
    for label_row, key, fmt in rows:
        vm = res[priv_name][key]
        vf = res[prot_name][key]
        vm_str = f"{vm:{fmt}}" if not np.isnan(vm) else "   N/A"
        vf_str = f"{vf:{fmt}}" if not np.isnan(vf) else "   N/A"
        print(f"  {label_row:<22} {vm_str:>14} {vf_str:>14}")



# Converts a fairness_metrics result into a flat dictionary row that will be then aggregated
def res_to_row(res, group_names, extra_cols={}):
    row = {**extra_cols, "threshold": res["threshold"]}
    if "axioms" in res:
        row.update(res["axioms"])
    n_vals = [res[name].get("n", np.nan) for name in group_names.values() if name in res]
    row["n_group_min"] = int(np.nanmin(n_vals)) if n_vals else np.nan
    return row


def compute_adTPR_adFPR(y_true, y_bin, sensitive,time_points=None):

    y_true    = np.asarray(y_true,    dtype=int)
    y_bin     = np.asarray(y_bin,     dtype=int)
    sensitive = np.asarray(sensitive)

    # Removes rows where the sensitive attribute is missing or invalid
    valid     = np.isin(sensitive, [0, 1])
    y_true    = y_true[valid]
    y_bin     = y_bin[valid]
    sensitive = sensitive[valid]

    time_pts  = (np.asarray(time_points)[valid]
                 if time_points is not None
                 else np.zeros(len(y_true), dtype=int))

    groups    = np.unique(sensitive)
    dTPR_list = []
    dFPR_list = []
    detail    = []

    # For each time point t, computes TPR and FPR separately for S=0 and S=1. 
    # Skips groups with fewer than 10 samples
    for t in np.unique(time_pts):
        mask_t        = time_pts == t
        tpr_per_group = []
        fpr_per_group = []

        for g in groups:
            mask_g = mask_t & (sensitive == g)
            if mask_g.sum() < 10:
                continue
            yt_g  = y_true[mask_g]
            yb_g  = y_bin[mask_g]
            n_pos = (yt_g == 1).sum()
            n_neg = (yt_g == 0).sum()
            tp    = ((yb_g == 1) & (yt_g == 1)).sum()
            fp    = ((yb_g == 1) & (yt_g == 0)).sum()
            tpr_per_group.append(tp / n_pos if n_pos > 0 else np.nan)
            fpr_per_group.append(fp / n_neg if n_neg > 0 else np.nan)

        tpr_per_group = [v for v in tpr_per_group if not np.isnan(v)]
        fpr_per_group = [v for v in fpr_per_group if not np.isnan(v)]
        if len(tpr_per_group) < 2 or len(fpr_per_group) < 2:
            continue

        # At each time point, measures the gap between the two groups and then avarege accross time. 
        dTPR = max(tpr_per_group) - min(tpr_per_group)
        dFPR = max(fpr_per_group) - min(fpr_per_group)
        dTPR_list.append(dTPR)
        dFPR_list.append(dFPR)

    return {
        "adTPR":  float(np.mean(dTPR_list)) if dTPR_list else np.nan,
        "adFPR":  float(np.mean(dFPR_list)) if dFPR_list else np.nan,
    }


