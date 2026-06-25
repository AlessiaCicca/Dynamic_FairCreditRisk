"""
GroupKFold cross-validation.
Returns OOF predictions, per-fold metrics, and the last-fold model/scaler.

GroupKFold: A cross-validation method that splits the data into K folds while keeping all 
observations from the same subject in the same fold, preventing data leakage across train and 
test sets.

OOF (Out-of-Fold) predictions: Predictions generated for each subject when that subject belongs
to the test fold, ensuring that every prediction is produced by a model that was not trained on 
that subject.

"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score, precision_recall_curve

from src.training.train_mlp import train_mlp


# F1-optimal threshold
def find_best_threshold(y_true, p, max_th_quantile = 0.90):
    p = np.clip(p, 0, 1)
    prec, rec, thresholds = precision_recall_curve(y_true, p)
    max_th    = np.quantile(p, max_th_quantile)
    f1_scores = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-8)
    f1_scores[thresholds > max_th] = 0
    return float(thresholds[np.argmax(f1_scores)]) if len(thresholds) > 0 else 0.5

# From hazard per bin to PD(L,L+h)
# Same function of main run
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
    
#   AUC, Brier score, F1 for single fold
def metrics_all(y_true, p,threshold = 0.5):
    p   = np.clip(p, 0, 1)
    auc = roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else np.nan
    return dict(
        AUC   = auc,
        Brier = brier_score_loss(y_true, p),
        F1    = f1_score(y_true, (p >= threshold).astype(int), zero_division=0),
        Th    = threshold,
    )

# Compute mean and sd accross all folds
def agg_mean_sd(list_of_dicts: list) -> dict:
    out = {}
    for k in list_of_dicts[0].keys():
        vals = [d[k] for d in list_of_dicts]
        out[f"{k}_Mean"] = float(np.nanmean(vals))
        out[f"{k}_SD"]   = float(np.nanstd(vals))
    return out

def run_cv(X, y, groups, sensitive,
           time_arr=None, subj_ids=None,
           model_name="", n_splits=5,
           landmarks=None,  collapse_pdh=False, n_bins=None,
           **train_kwargs):
    # GroupKFold splits by subject — same subject never in both train and test
    gkf          = GroupKFold(n_splits=n_splits)
    oof_preds    = np.zeros(len(y), dtype=np.float64)
    metrics_list = []
    model_last   = None
    scaler_last  = None

    thresholds = []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        # train MLP on training fold, get predictions on both train and test
        p_te, p_tr, model, scaler = train_mlp(
            X[tr], y[tr], X[te], y[te],
            sensitive_tr = sensitive[tr] if sensitive is not None else None,
            time_tr      = time_arr[tr]  if time_arr  is not None else None,
            subj_ids_tr  = subj_ids[tr]  if subj_ids  is not None else None,
            model_name   = model_name,
            verbose      = (fold == 0),
            **train_kwargs,
        )
        # store test predictions in the OOF array
        oof_preds[te] = p_te


        oof_preds[te] = p_te

        if collapse_pdh:
            tr_pdh = _collapse_fold(p_tr, y[tr], groups[tr], time_arr[tr], n_bins)
            te_pdh = _collapse_fold(p_te, y[te], groups[te], time_arr[te], n_bins)

            best_th = find_best_threshold(tr_pdh["yh"], tr_pdh["pdh"]) 
            thresholds.append(best_th)
            metrics_list.append(
                metrics_all(te_pdh["yh"].astype(int), te_pdh["pdh"], threshold=best_th)
            )
            auc_print = metrics_list[-1]["AUC"]
        else:
            best_th = find_best_threshold(y[tr], p_tr)
            
            thresholds.append(best_th)
            metrics_list.append(metrics_all(y[te].astype(int), p_te, threshold=best_th))
            auc_print = metrics_list[-1]["AUC"]

        print(
            f"  Fold {fold + 1}  |  "
            f"  pred_mean_test={p_te.mean():.4f}"
            f"  |  AUC: {auc_print:.4f}  |  th={best_th:.5f}"
        )
        if fold == n_splits - 1:
            model_last  = model
            scaler_last = scaler

    summary          = agg_mean_sd(metrics_list)
    summary["Model"] = model_name.upper()

    return dict(
        oof_preds   = oof_preds,
        metrics     = metrics_list,
        summary     = summary,
        threshold   = float(np.mean(thresholds)),
        model_last  = model_last,
        scaler_last = scaler_last,
    )


def build_summary_table(cv_results) :
    rows = []
    for name, res in cv_results.items():
        row = res["summary"].copy()
        row["Model"] = name
        rows.append(row)

    cols = [
        "Model", "AUC_Mean", "AUC_SD",
        "Brier_Mean", "Brier_SD",
        "F1_Mean", "F1_SD",
    ]
    df = pd.DataFrame(rows)
    return df[[c for c in cols if c in df.columns]]


