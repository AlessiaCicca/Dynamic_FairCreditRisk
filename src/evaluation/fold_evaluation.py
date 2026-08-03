#Fairness and Performance evaluation for each fold...the script is called by run_train.py

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from src.evaluation.fairness_metrics import (fairness_metrics, filter_sensitive, compute_adTPR_adFPR)


#  AUC, Brier score for single fold (F1 is not reported as a metric, it is only used internally by find_best_threshold to pick the operating threshold)
def metrics_all(y_true, p, threshold=0.5):
    p = np.clip(p, 0, 1)
    auc = roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else np.nan
    return dict(AUC=auc, Brier=brier_score_loss(y_true, p),Th=threshold)

# Compute mean and sd accross all folds
def agg_mean_sd(list_of_dicts):
    out = {}
    for k in list_of_dicts[0].keys():
        vals = [d[k] for d in list_of_dicts]
        out[f"{k}_Mean"] = float(np.nanmean(vals))
        out[f"{k}_SD"] = float(np.nanstd(vals))
    return out


# Separation, AUC, adTPR and adFPR on the static (aggregate) predictions
def eval_static(preds, y, sens, group_names, eval_th):
    yt_f, yp_f, sn_f = filter_sensitive(np.asarray(y).astype(int), preds, sens)
    # Information about both the group are required
    if len(np.unique(yt_f)) < 2 or len(np.unique(sn_f)) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    auc = roc_auc_score(yt_f, yp_f)
    yb_f = (yp_f >= eval_th).astype(int)
    res = fairness_metrics(yt_f, yp_f, yb_f, sn_f, group_names, threshold=eval_th)
    s = res.get("axioms", {}).get("separation", np.nan)
    ad = compute_adTPR_adFPR(yt_f, yb_f, sn_f, None)   # no time -> a single "landmark"
    # Separation is repeted twice for alignement with dynamic that store both sep_auc and sep_mean
    return auc, s, s, ad["adTPR"], ad["adFPR"]

# Separation, AUC, adTPR and adFPR on the dynamic predictions
def eval_dynamic_from_pdh(coll, sens_by_id, group_names, eval_th):
    # coll: (id, L, pdh, yh, n)
    
    coll = coll.reset_index(drop=True).copy()
    coll["sens"] = coll["id"].map(sens_by_id)
    
    eval_preds = coll["pdh"].to_numpy()
    eval_y = coll["yh"].to_numpy().astype(int)
    eval_sens = coll["sens"].to_numpy()
    eval_time = coll["L"].to_numpy()

    # Compute AUC and BRIER Score for each landmark (curve)
    df_perf = perf_by_landmark(eval_y, eval_preds, eval_time)

    # Integrate the curve
    auc_integrated = integrate_curve(df_perf, "auc")
    brier_integrated = integrate_curve(df_perf, "brier")

    time_rows = []
    # LOOP on landmark time -> compute fairness metrics for each landmark
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

    df_t = pd.DataFrame(time_rows)
    sep_auc = integrate_curve(df_t, "separation")
    sep_mean = (df_t["separation"].mean()
                if (not df_t.empty and "separation" in df_t.columns) else np.nan)

    # Binarization of the prediction
    yb_all = (eval_preds >= eval_th).astype(int)
    ad = compute_adTPR_adFPR(eval_y, yb_all, eval_sens, eval_time)
    adtpr, adfpr = ad["adTPR"], ad["adFPR"]

    return { "auc": auc_integrated,"sep_auc": sep_auc,"sep_mean": sep_mean,
        "adtpr": adtpr,"adfpr": adfpr,"brier": brier_integrated,"df_perf": df_perf}


# Compute the integrale of a generic curve store in df_t (type value/time)
def integrate_curve(df_t, col, t_min=None, t_max=None):

    # Remove landmarks for which we cannot compute metrics
    if df_t.empty or col not in df_t.columns:
        return np.nan
    sub = df_t.dropna(subset=[col])
    if len(sub) < 2:
        return np.nan

    t_v = sub["t"].to_numpy(float)
    v = sub[col].to_numpy(float)

    # Use the observed value as boundaries only if they are not provided in the config
    if t_min is None:
        t_min = t_v.min()
    if t_max is None:
        t_max = t_v.max()
    if t_max - t_min <= 0:
        return np.nan
        
    area = np.trapezoid(v, t_v)
    return float(area / (t_max - t_min))


# Compute performance (AUC / Brier) for each landmark
def perf_by_landmark(y_true, preds, time_vals):
    rows = []
    for t in sorted(np.unique(time_vals)):
        # Consider only value related to landmark t
        mask = time_vals == t
        if mask.sum() == 0:
            continue
            
        yt_t, yp_t = y_true[mask], preds[mask]
        auc_t = roc_auc_score(yt_t, yp_t) if len(np.unique(yt_t)) > 1 else np.nan
        brier_t = brier_score_loss(yt_t, yp_t)
        rows.append({"t": t, "auc": auc_t, "brier": brier_t})
    return pd.DataFrame(rows)


