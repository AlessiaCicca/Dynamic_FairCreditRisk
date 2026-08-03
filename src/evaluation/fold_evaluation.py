"""
Fairness and Performance evaluation for each fold...the script is called by run_train.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from src.evaluation.fairness_metrics import (fairness_metrics, filter_sensitive, compute_adTPR_adFPR)


#   AUC, Brier score for single fold (F1 is not reported as a metric -- it is only
#   used internally by find_best_threshold to pick the operating threshold)
def metrics_all(y_true, p, threshold=0.5):
    p = np.clip(p, 0, 1)
    auc = roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else np.nan
    return dict(
        AUC=auc,
        Brier=brier_score_loss(y_true, p),
        Th=threshold,
    )

# Compute mean and sd accross all folds
def agg_mean_sd(list_of_dicts):
    out = {}
    for k in list_of_dicts[0].keys():
        vals = [d[k] for d in list_of_dicts]
        out[f"{k}_Mean"] = float(np.nanmean(vals))
        out[f"{k}_SD"] = float(np.nanstd(vals))
    return out


# AUC and separation on the static (aggregate) predictions
def eval_static(preds, y, sens, group_names, eval_th):
    yt_f, yp_f, sn_f = filter_sensitive(np.asarray(y).astype(int), preds, sens)
    if len(np.unique(yt_f)) < 2 or len(np.unique(sn_f)) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    auc = roc_auc_score(yt_f, yp_f)
    yb_f = (yp_f >= eval_th).astype(int)
    res = fairness_metrics(yt_f, yp_f, yb_f, sn_f, group_names, threshold=eval_th)
    s = res.get("axioms", {}).get("separation", np.nan)
    ad = compute_adTPR_adFPR(yt_f, yb_f, sn_f, None)   # no time -> un solo "landmark"
    return auc, s, s, ad["adTPR"], ad["adFPR"]


def integrate_curve(df_t, col, t_min=None, t_max=None):
    if df_t.empty or col not in df_t.columns:
        return np.nan
    sub = df_t.dropna(subset=[col])
    if len(sub) < 2:
        return np.nan
    t_v = sub["t"].to_numpy(float)
    v = sub[col].to_numpy(float)
    # normalizza sempre sull'intervallo REALMENTE osservato, a meno che il
    # chiamante non specifichi esplicitamente un range diverso. Il vecchio
    # default t_min=0.0 sottostimava sistematicamente l'integrale ogni volta
    # che i landmark non partivano da 0 (es. simulation con LANDMARKS_SIM che
    # parte da 1 o 2): l'area veniva calcolata solo sul range osservato ma
    # divisa per un intervallo piu' largo, deflazionando il risultato.
    if t_min is None:
        t_min = t_v.min()
    if t_max is None:
        t_max = t_v.max()
    if t_max - t_min <= 0:
        return np.nan
    area = np.trapezoid(v, t_v)
    return float(area / (t_max - t_min))


# Per-landmark AUC / Brier, used to build the integrated (time-normalized)
# performance curves for the dynamic model. Replaces the old pooled
# (person-period aggregate) performance computation. F1 is intentionally not
# reported here -- it is only used internally (via find_best_threshold) to
# pick the operating threshold `th`, which is still needed for fairness
# (separation) but is no longer surfaced as a performance metric.
def perf_by_landmark(y_true, preds, time_vals):
    rows = []
    for t in sorted(np.unique(time_vals)):
        mask = time_vals == t
        if mask.sum() == 0:
            continue
        yt_t, yp_t = y_true[mask], preds[mask]
        auc_t = roc_auc_score(yt_t, yp_t) if len(np.unique(yt_t)) > 1 else np.nan
        brier_t = brier_score_loss(yt_t, yp_t)
        rows.append({"t": t, "auc": auc_t, "brier": brier_t})
    return pd.DataFrame(rows)


def eval_dynamic_from_pdh(coll, sens_by_id, group_names, eval_th):
    """
    coll: DataFrame con (id, L, pdh, yh, n) gia' calcolato con il full-horizon
          PD-H (_collapse_fold_full_horizon), col modello del fold corretto.
    """
    coll = coll.reset_index(drop=True).copy()
    coll["sens"] = coll["id"].map(sens_by_id)

    eval_preds = coll["pdh"].to_numpy()
    eval_y     = coll["yh"].to_numpy().astype(int)
    eval_sens  = coll["sens"].to_numpy()
    eval_time  = coll["L"].to_numpy()

    df_perf          = perf_by_landmark(eval_y, eval_preds, eval_time)
    auc_integrated   = integrate_curve(df_perf, "auc")
    brier_integrated = integrate_curve(df_perf, "brier")

    time_rows = []
    for t in sorted(np.unique(eval_time)):
        mask = eval_time == t
        yt_f, yp_f, sn_f = filter_sensitive(eval_y[mask], eval_preds[mask], eval_sens[mask])
        if len(np.unique(yt_f)) < 2 or len(np.unique(sn_f)) < 2:
            continue
        if pd.Series(sn_f).value_counts().min() < 20:
            continue
        yb_f = (yp_f >= eval_th).astype(int)
        res = fairness_metrics(yt_f, yp_f, yb_f, sn_f, group_names, threshold=eval_th)
        time_rows.append({"t": t, "separation": res.get("axioms", {}).get("separation", np.nan)})

    df_t     = pd.DataFrame(time_rows)
    sep_auc  = integrate_curve(df_t, "separation")
    sep_mean = (df_t["separation"].mean()
                if (not df_t.empty and "separation" in df_t.columns) else np.nan)

    # adTPR / adFPR (Xie & Ge): media semplice dei gap |TPR_1-TPR_0| / |FPR_1-FPR_0|
    # per landmark. A differenza di SEP-AUC (integrale della curva), sono medie
    # semplici -> molto meno sensibili al rumore dei landmark con pochi dati.
    yb_all = (eval_preds >= eval_th).astype(int)
    ad = compute_adTPR_adFPR(eval_y, yb_all, eval_sens, eval_time)
    adtpr, adfpr = ad["adTPR"], ad["adFPR"]

    class _Result(tuple):
        pass
    result = _Result((auc_integrated, sep_auc, sep_mean, adtpr, adfpr))
    result.brier_integrated = brier_integrated
    result.df_perf = df_perf
    return result
