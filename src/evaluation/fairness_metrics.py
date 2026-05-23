"""
src/evaluation/fairness_metrics.py

Core fairness metrics: group-level statistics, axioms (independence,
separation, sufficiency), adTPR/adFPR, and bootstrap confidence intervals.

Works for both simulation (single sensitive attribute) and real dataset
(multiple attributes — caller loops over them).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve


# ── Threshold utilities ───────────────────────────────────────────────────────

def compute_threshold(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1-optimal classification threshold."""
    prec, rec, thr = precision_recall_curve(y_true, y_pred)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-8)
    return float(thr[np.argmax(f1)]) if len(thr) > 0 else 0.5


def filter_sensitive(y_true: np.ndarray, y_pred: np.ndarray,
                     sens_arr: np.ndarray):
    """Remove rows where sensitive attribute is not 0 or 1."""
    valid = np.isin(sens_arr, [0, 1])
    return y_true[valid], y_pred[valid], sens_arr[valid]


# ── Per-group metrics ─────────────────────────────────────────────────────────

def fairness_metrics(y_true, y_pred_proba, y_bin, sensitive,
                     group_names: dict, threshold: float) -> dict:
    """
    Compute per-group statistics and fairness axioms.

    Parameters
    ----------
    y_true       : binary ground-truth labels
    y_pred_proba : predicted probabilities
    y_bin        : binarised predictions (already thresholded)
    sensitive    : sensitive attribute array (0/1)
    group_names  : dict mapping {0: name_0, 1: name_1}
    threshold    : classification threshold used

    Returns
    -------
    dict with keys:
        threshold, {group_name: {metrics}}, gaps, axioms
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred_proba, dtype=float)
    y_bin  = np.asarray(y_bin, dtype=int)
    sens   = np.asarray(sensitive)

    results = {"threshold": threshold}
    groups  = [g for g in [0, 1] if g in sens]

    for g in groups:
        mask  = sens == g
        yt_g  = y_true[mask]
        yp_g  = y_pred[mask]
        yb_g  = y_bin[mask]
        n     = mask.sum()
        n_pos = (yt_g == 1).sum()
        n_neg = (yt_g == 0).sum()
        name  = group_names[g]

        tp = ((yb_g == 1) & (yt_g == 1)).sum()
        fp = ((yb_g == 1) & (yt_g == 0)).sum()
        fn = ((yb_g == 0) & (yt_g == 1)).sum()

        results[name] = {
            "n":          int(n),
            "prev":       float(n_pos / n)       if n > 0      else np.nan,
            "dp":         float(yb_g.mean()),
            "mean_score": float(yp_g.mean()),
            "tpr":        float(tp / n_pos)       if n_pos > 0  else np.nan,
            "fpr":        float(fp / n_neg)       if n_neg > 0  else np.nan,
            "fnr":        float(fn / n_pos)       if n_pos > 0  else np.nan,
            "ppv":        float(tp / (tp + fp))   if (tp+fp)>0  else np.nan,
            "auc":        roc_auc_score(yt_g, yp_g)
                          if len(np.unique(yt_g)) > 1 else np.nan,
        }

    if len(groups) == 2:
        m = results[group_names[0]]
        f = results[group_names[1]]
        results["gaps"] = {
            "dp_gap":         f["dp"]         - m["dp"],
            "tpr_gap":        f["tpr"]        - m["tpr"],
            "fpr_gap":        f["fpr"]        - m["fpr"],
            "ppv_gap":        f["ppv"]        - m["ppv"],
            "mean_score_gap": f["mean_score"] - m["mean_score"],
            "auc_gap":        f["auc"]        - m["auc"],
        }
        results["axioms"] = {
            "independence": abs(results["gaps"]["dp_gap"]),
            "separation":   abs(
                (f["fpr"] - m["fpr"]) + (f["fnr"] - m["fnr"])
            ) / 2,
            "sufficiency":  abs(results["gaps"]["ppv_gap"]),
        }

    return results


def print_fairness_report(model_name: str, res: dict,
                          group_names: dict, label: str = "AGGREGATE") -> None:
    """Pretty-print a fairness_metrics result dict."""
    priv_name = group_names[0]
    prot_name = group_names[1]
    print(f"\n{'─'*50}")
    print(f"  {model_name}  [{label}]  (threshold = {res['threshold']:.4f})")
    print(f"{'─'*50}")
    print(f"  {'Metric':<22} {priv_name:>14} {prot_name:>14} {'Gap':>12}")
    print(f"  {'-'*62}")

    rows = [
        ("N observations",     "n",          ".0f",  False),
        ("Base rate (prev.)",  "prev",       ".4f",  False),
        ("Dem. Parity P(ŷ=1)", "dp",         ".4f",  True),
        ("Mean score",         "mean_score", ".4f",  True),
        ("TPR (recall)",       "tpr",        ".4f",  True),
        ("FPR",                "fpr",        ".4f",  True),
        ("FNR (miss rate)",    "fnr",        ".4f",  True),
        ("Precision (PPV)",    "ppv",        ".4f",  True),
        ("AUC per group",      "auc",        ".4f",  True),
    ]
    gap_keys = {
        "dp": "dp_gap", "mean_score": "mean_score_gap",
        "tpr": "tpr_gap", "fpr": "fpr_gap",
        "ppv": "ppv_gap", "auc": "auc_gap",
    }
    for label_row, key, fmt, has_gap in rows:
        vm = res[priv_name][key]
        vf = res[prot_name][key]
        gk = gap_keys.get(key)
        gap_str = (f"{res['gaps'][gk]:>+.4f}"
                   if has_gap and gk and gk in res.get("gaps", {}) else "")
        vm_str = f"{vm:{fmt}}" if not np.isnan(vm) else "   N/A"
        vf_str = f"{vf:{fmt}}" if not np.isnan(vf) else "   N/A"
        print(f"  {label_row:<22} {vm_str:>10} {vf_str:>10} {gap_str:>12}")

    if "axioms" in res:
        print(f"\n  {'─'*40}")
        print(f"  Independence (SP):  {res['axioms']['independence']:.4f}")
        print(f"  Separation:         {res['axioms']['separation']:.4f}")
        print(f"  Sufficiency:        {res['axioms']['sufficiency']:.4f}")


def res_to_row(res: dict, group_names: dict, extra_cols: dict = {}) -> dict:
    """Flatten a fairness_metrics result dict into a single-row dict."""
    row = {**extra_cols, "threshold": res["threshold"]}
    if "axioms" in res:
        row.update(res["axioms"])
    if "gaps" in res:
        row.update(res["gaps"])

    n_vals = []
    for g, name in group_names.items():
        if name in res:
            n_vals.append(res[name].get("n", np.nan))
    row["n_group_min"] = int(np.nanmin(n_vals)) if n_vals else np.nan

    for g, name in group_names.items():
        if name in res:
            for k, v in res[name].items():
                row[f"{name}_{k}"] = v
    return row


# ── adTPR / adFPR ─────────────────────────────────────────────────────────────

def compute_adTPR_adFPR(y_true, y_bin, sensitive,
                         time_points=None) -> dict:
    """
    Average per-time-point TPR and FPR gap across groups.

    Parameters
    ----------
    y_true       : binary labels
    y_bin        : binarised predictions
    sensitive    : sensitive attribute (0/1)
    time_points  : array of time points parallel to data; None = aggregate

    Returns
    -------
    dict with keys adTPR, adFPR, detail (DataFrame)
    """
    y_true    = np.asarray(y_true,    dtype=int)
    y_bin     = np.asarray(y_bin,     dtype=int)
    sensitive = np.asarray(sensitive)

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

        dTPR = max(tpr_per_group) - min(tpr_per_group)
        dFPR = max(fpr_per_group) - min(fpr_per_group)
        dTPR_list.append(dTPR)
        dFPR_list.append(dFPR)
        detail.append({"time_point": t, "dTPR": dTPR, "dFPR": dFPR})

    return {
        "adTPR":  float(np.mean(dTPR_list)) if dTPR_list else np.nan,
        "adFPR":  float(np.mean(dFPR_list)) if dFPR_list else np.nan,
        "detail": pd.DataFrame(detail),
    }


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def fairness_metrics_bootstrap(y_true, y_pred, y_bin, sensitive,
                                group_names: dict, threshold: float = 0.5,
                                B: int = 500, ci: float = 0.95,
                                random_state: int = 42) -> dict:
    """
    Bootstrap confidence intervals for independence, separation, sufficiency.
    """
    rng   = np.random.RandomState(random_state)
    alpha = (1 - ci) / 2

    boot = {"independence": [], "separation": [], "sufficiency": []}
    n    = len(y_true)

    for _ in range(B):
        idx = rng.choice(n, size=n, replace=True)
        yt  = y_true[idx]
        yp  = y_pred[idx]
        yb  = (yp >= threshold).astype(int)
        sn  = sensitive[idx]

        if len(np.unique(sn)) < 2 or len(np.unique(yt)) < 2:
            continue
        try:
            res = fairness_metrics(yt, yp, yb, sn, group_names,
                                   threshold=threshold)
            for k in boot:
                boot[k].append(res.get("axioms", {}).get(k, np.nan))
        except Exception:
            continue

    out = {}
    for k, vals in boot.items():
        vals = np.array(vals)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            out[k] = out[f"{k}_ci_lo"] = out[f"{k}_ci_hi"] = np.nan
            out[f"{k}_reliable"] = False
        else:
            out[k]            = float(np.mean(vals))
            out[f"{k}_ci_lo"] = float(np.quantile(vals, alpha))
            out[f"{k}_ci_hi"] = float(np.quantile(vals, 1 - alpha))
            ci_width = out[f"{k}_ci_hi"] - out[f"{k}_ci_lo"]
            out[f"{k}_reliable"] = bool(
                ci_width < 0.15 or
                (out[k] > 0 and ci_width / (abs(out[k]) + 1e-9) < 2.0)
            )
    return out
