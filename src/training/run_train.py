"""
GroupKFold cross-validation for both final evaluation and grid search.

GroupKFold: A cross-validation method that splits the data into K folds while keeping all
observations from the same subject in the same fold, preventing data leakage across train and
test sets.

"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, f1_score, precision_recall_curve
)

from src.training.train_mlp import train_mlp
from src.evaluation.fairness_metrics import fairness_metrics, filter_sensitive, compute_adTPR_adFPR
from config import SEED


MODEL_STYLES = {
    "M_STATIC":  {"color": "#3A6BC4", "marker": "o", "coef_label": "beta"},
    "M_DYNAMIC": {"color": "#D4612A", "marker": "s", "coef_label": "alpha"},
}


def _reset_seed(fold, seed=SEED):
    """
    Reimposta lo stato RNG prima di ogni training, in modo DETERMINISTICO e
    dipendente solo dal fold (non da quanti training sono gia' stati fatti).

    Senza questo, il seed viene impostato una volta sola all'avvio e ogni
    training consuma lo stream RNG: run_cv e run_grid_search partono quindi da
    stati diversi e producono modelli DIVERSI a parita' di iperparametri
    (pesi iniziali diversi -> soglie diverse -> metriche diverse). Questo
    rendeva non confrontabili i numeri di cv_results.csv con quelli della grid
    search, e non riproducibile il singolo esperimento.
    """
    import torch
    s = seed + 1000 * fold
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# Split definition: Build the train/val/test folds once and return them as a list of
#   (train_idx, val_idx, test_idx) tuples.
def make_splits(y, groups, n_splits=5, val_size=0.5, seed=SEED):
    # GroupKFold splits by subject: same loan is never both in train and test+val set.
    gkf = GroupKFold(n_splits=n_splits)
    splits = []
    # GroupShuffleSplit splits the test+val fold into val/test, again by subject,
    # run_cv and run_grid_search receive that division
    for tr_idx, te_idx in gkf.split(np.zeros(len(y)), y, groups):
        gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        a_pos, b_pos = next(gss.split(te_idx, groups=groups[te_idx]))
        splits.append((tr_idx, te_idx[a_pos], te_idx[b_pos]))
    return splits

# F1-optimal threshold (F1 puro: nessun vincolo aggiuntivo sulla soglia).
#
# La versione precedente azzerava l'F1 per tutte le soglie sopra il 90esimo
# percentile delle predizioni, vincolando implicitamente il modello a non
# predire positivo piu' del 10% del campione. Non era un vincolo dichiarato ne'
# giustificato, e con una prevalenza reale del ~6% escludeva soglie legittime.
def find_best_threshold(y_true, p):
    p = np.clip(p, 0, 1)
    prec, rec, thresholds = precision_recall_curve(y_true, p)
    if len(thresholds) == 0:
        return 0.5
    f1_scores = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-8)
    return float(thresholds[np.argmax(f1_scores)])

# From hazard per bin to PD(L,L+h)
# Same function of main run
def _collapse_fold(hazard, event_bin, ids, lmk, n_bins, complete_only=True):
    h = np.clip(hazard, 1e-7, 1 - 1e-7)
    d = pd.DataFrame({
        "id": ids, "L": lmk,
        # log(1 - hazard) -> turns the product over bins into a sum
        "log1mh": np.log1p(-h),
        "ev": event_bin,
    })
    g = d.groupby(["id", "L"], sort=False)
    out = pd.DataFrame({
        "pdh": 1.0 - np.exp(g["log1mh"].sum()),
        "yh":  g["ev"].max(),
        "n":   g.size(),
    }).reset_index()
    # require all bins present for the (subject, landmark), TRANNE per i
    # soggetti con evento già risolto (yh==1): escono dal rischio appena
    # avviene l'evento (n < n_bins per costruzione), ma l'esito è comunque
    # completo e noto -> non vanno scartati come se fossero censoring.
    # Scartiamo solo i gruppi incompleti E senza evento (vero censoring).
    if complete_only:
        out = out[(out["n"] == n_bins) | (out["yh"] == 1)]
    return out



# Tanner-style PD-H (Tanner et al. 2021, JRSS-A -- createPredictData + cumprod).
#
# Il person-period dataset di TRAINING contiene solo i bin realmente vissuti dal
# soggetto: i bin successivi all'evento sono scartati, perche' il soggetto e'
# uscito dal risk set e non c'e' nulla da imparare (questo lo fa _collapse_fold /
# build_dynamic, ed e' corretto).
#
# Ma in PREDIZIONE il modello va interrogato su una griglia sintetica COMPLETA di
# tutti gli n_bins dell'orizzonte: e' una domanda puramente predittiva ("dato lo
# stato a L, che hazard assegni a ciascuno dei bin futuri?"), che NON dipende da
# cosa sia poi successo. Il prodotto di sopravvivenza e' quindi sempre su n_bins
# termini, per ogni soggetto.
#
# Senza questo, un default precoce (pochi bin osservati) ottiene un prodotto su
# meno fattori -> PD-H sistematicamente piu' basso -> AUC e soglia collassano.
def _collapse_fold_full_horizon(model, scaler, X, y, groups, lmk, bin_times,
                                 feat_names, idx, n_bins, delta, device,
                                 complete_only=True):
    import torch

    spl_idx = [i for i, f in enumerate(feat_names) if str(f).startswith("spl_")]
    if not spl_idx:
        raise ValueError("Nessuna colonna 'spl_*' in feature_names")

    d = pd.DataFrame({
        "row": idx,
        "id":  groups[idx],
        "L":   lmk[idx],
        "ev":  y[idx],
    })
    g   = d.groupby(["id", "L"], sort=False)
    out = pd.DataFrame({
        "yh":  g["ev"].max(),
        "n":   g.size(),
        "row": g["row"].first(),
    }).reset_index()

    if complete_only:
        out = out[(out["n"] == n_bins) | (out["yh"] == 1)].reset_index(drop=True)
    if len(out) == 0:
        return out.assign(pdh=[])

    n_groups = len(out)
    X_rep  = X[out["row"].to_numpy()]
    X_full = np.repeat(X_rep, n_bins, axis=0)

    # mappa bin_time -> valori spline (ricavata dal dataset originale)
    bt_to_spl = {}
    for bt_val in np.unique(bin_times):
        j = np.where(bin_times == bt_val)[0][0]
        bt_to_spl[bt_val] = X[j, spl_idx]

    L_rep     = np.repeat(out["L"].to_numpy(), n_bins)
    j_rep     = np.tile(np.arange(n_bins), n_groups)
    bt_target = L_rep + delta * j_rep
    for k, bt_val in enumerate(bt_target):
        if bt_val in bt_to_spl:
            X_full[k, spl_idx] = bt_to_spl[bt_val]

    Xs = scaler.transform(X_full).astype(np.float32)
    Xs = np.nan_to_num(Xs, nan=0., posinf=5., neginf=-5.)
    model.eval()
    with torch.no_grad():
        h = torch.sigmoid(model(torch.tensor(Xs, device=device))).cpu().numpy()
    h = np.clip(h, 1e-7, 1 - 1e-7)

    surv = np.prod(1.0 - h.reshape(n_groups, n_bins), axis=1)
    out["pdh"] = 1.0 - surv
    return out[["id", "L", "pdh", "yh", "n"]]


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
def _eval_static(preds, y, sens, group_names, eval_th):
    yt_f, yp_f, sn_f = filter_sensitive(np.asarray(y).astype(int), preds, sens)
    if len(np.unique(yt_f)) < 2 or len(np.unique(sn_f)) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    auc = roc_auc_score(yt_f, yp_f)
    yb_f = (yp_f >= eval_th).astype(int)
    res = fairness_metrics(yt_f, yp_f, yb_f, sn_f, group_names, threshold=eval_th)
    s = res.get("axioms", {}).get("separation", np.nan)
    ad = compute_adTPR_adFPR(yt_f, yb_f, sn_f, None)   # no time -> un solo "landmark"
    return auc, s, s, ad["adTPR"], ad["adFPR"]


def _integrate_curve(df_t, col, t_min=None, t_max=None):
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
def _perf_by_landmark(y_true, preds, time_vals):
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


def _eval_dynamic_from_pdh(coll, sens_by_id, group_names, eval_th):
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

    df_perf          = _perf_by_landmark(eval_y, eval_preds, eval_time)
    auc_integrated   = _integrate_curve(df_perf, "auc")
    brier_integrated = _integrate_curve(df_perf, "brier")

    time_rows = []
    for t in sorted(np.unique(eval_time)):
        mask = eval_time == t
        yt_f, yp_f, sn_f = filter_sensitive(eval_y[mask], eval_preds[mask], eval_sens[mask])
        if len(np.unique(yt_f)) < 2 or len(np.unique(sn_f)) < 2:
            continue
        if pd.Series(sn_f).value_counts().min() < 50:
            continue
        yb_f = (yp_f >= eval_th).astype(int)
        res = fairness_metrics(yt_f, yp_f, yb_f, sn_f, group_names, threshold=eval_th)
        time_rows.append({"t": t, "separation": res.get("axioms", {}).get("separation", np.nan)})

    df_t     = pd.DataFrame(time_rows)
    sep_auc  = _integrate_curve(df_t, "separation")
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


# Main function of the file
# For each pre-computed (train, val, test) fold: fit on train, pick the threshold
# on train, and store predictions into val / test
def _fit_predict(splits, X, y, groups, sensitive,
                 time_arr, subj_ids, model_name,
                 alpha, beta, n_bins, collapse_pdh,  verbose_folds=False,
                 bin_times=None, feat_names=None, delta=None, device="cpu",
                 **train_kwargs):

    is_dyn = time_arr is not None
    oof_val = np.zeros(len(y), dtype=np.float64)
    oof_test = np.zeros(len(y), dtype=np.float64)
    oof_full = np.zeros(len(y), dtype=np.float64)
    is_val = np.zeros(len(y), dtype=bool)
    is_test = np.zeros(len(y), dtype=bool)

    thresholds = []
    model_last = scaler_last = None
    fold_models = []   # (model, scaler) per fold -- serve per la predizione
                       # full-horizon, che va fatta col modello DEL PROPRIO fold
    for fold, (tr_idx, val_idx, test_idx) in enumerate(splits):
        te_idx = np.concatenate([val_idx, test_idx])

        # RNG deterministico per fold: garantisce che lo stesso fold, con gli
        # stessi iperparametri, produca sempre lo stesso modello -- sia in
        # run_cv sia in run_grid_search.
        _reset_seed(fold)

         # train MLP on the training fold, predict on the whole held-out (val + test)
        p_te, p_tr, model, scaler = train_mlp(
            X[tr_idx], y[tr_idx], X[te_idx], y[te_idx],
            sensitive_tr=sensitive[tr_idx] if sensitive is not None else None,
            time_tr=time_arr[tr_idx] if is_dyn else None,
            subj_ids_tr=subj_ids[tr_idx] if subj_ids is not None else None,
            model_name=model_name, alpha=alpha, beta=beta,
            verbose=(verbose_folds and fold == 0), **train_kwargs,
        )

        # Divide predictions in val and test
        pos = {idx: k for k, idx in enumerate(te_idx)}
        oof_val[val_idx] = p_te[[pos[i] for i in val_idx]]
        oof_test[test_idx] = p_te[[pos[i] for i in test_idx]]
        oof_full[te_idx] = p_te
        is_val[val_idx] = True
        is_test[test_idx] = True

        # Soglia scelta sul VALIDATION del fold (non sul training).
        # Il training e' in-sample: il modello ha gia' visto quei dati, quindi le
        # sue predizioni sono troppo confidenti e la soglia ne risulta distorta.
        # La soglia e' a tutti gli effetti un iperparametro della regola
        # decisionale -> va scelta su val, come il coefficiente alpha/beta.
        if collapse_pdh:
            val_pdh_th = _collapse_fold_full_horizon(
                model, scaler, X, y, groups, time_arr, bin_times, feat_names,
                val_idx, n_bins, delta, device)
            thresholds.append(find_best_threshold(val_pdh_th["yh"], val_pdh_th["pdh"]))
        else:
            p_val_th = p_te[[pos[i] for i in val_idx]]
            thresholds.append(find_best_threshold(y[val_idx], p_val_th))

        fold_models.append((model, scaler))
        model_last, scaler_last = model, scaler

        if verbose_folds:
            if collapse_pdh:
                val_pdh = _collapse_fold_full_horizon(
                    model, scaler, X, y, groups, time_arr, bin_times, feat_names,
                    val_idx, n_bins, delta, device)
                df_perf_fold = _perf_by_landmark(val_pdh["yh"].to_numpy().astype(int),
                                                 val_pdh["pdh"].to_numpy(),
                                                 val_pdh["L"].to_numpy())
                auc_fold = _integrate_curve(df_perf_fold, "auc")
                pm = val_pdh["pdh"].mean()
            else:
                val_pos  = [pos[i] for i in val_idx]
                p_val    = p_te[val_pos]
                auc_fold = (roc_auc_score(y[val_idx], p_val)
                            if len(np.unique(y[val_idx])) > 1 else float("nan"))
                pm = p_val.mean()
            print(f"  Fold {fold + 1}  |  pred_mean_val={pm:.4f}"
                  f"  |  AUC (val): {auc_fold:.4f}  |  th={thresholds[-1]:.5f}")

    return dict(
        oof_val=oof_val, oof_test=oof_test, oof_full=oof_full,
        is_val=is_val, is_test=is_test,
        threshold=float(np.mean(thresholds)),   # media (per retrocompatibilita')
        fold_thresholds=thresholds,             # soglia PER FOLD: ogni fold usa
                                                # la propria per valutare il suo test
        model_last=model_last, scaler_last=scaler_last,
        fold_models=fold_models,
    )


# Valutazione PER FOLD: ogni fold usa il PROPRIO modello per predire il proprio
# split (val o test). Usare un unico modello (es. l'ultimo fold) su tutti gli
# split sarebbe leakage: 4 fold su 5 avrebbero visto quei soggetti in training.
def _fairness_per_fold(fp, splits, which, X, y, groups, time_arr, sensitive,
                       group_names, th, n_bins, is_dyn,
                       bin_times=None, feat_names=None, delta=None, device="cpu",
                       use_fold_threshold=True):
    """
    use_fold_threshold=True: ogni fold usa la PROPRIA soglia (scelta sul suo val)
    per valutare il proprio split -- cosi' ogni fold e' un esperimento
    self-contained (train -> allena, val -> soglia, test -> riporta).
    Se False, usa la soglia `th` passata (serve per la valutazione a soglia
    FISSA del baseline nella grid search).
    """
    NAN5 = (np.nan,) * 5
    if group_names is None:
        return NAN5

    fold_ths = fp.get("fold_thresholds") if use_fold_threshold else None

    if not is_dyn:
        # ANCHE lo statico va valutato PER FOLD e poi mediato, coerentemente
        # col dinamico (e con cv_results.csv). Prima veniva calcolato una volta
        # sola su tutta la maschera out-of-fold (= aggregato), producendo un
        # numero non confrontabile con quello del dinamico.
        oof = fp["oof_val"] if which == "val" else fp["oof_test"]
        rows = []
        for k, (tr_idx, val_idx, test_idx) in enumerate(splits):
            idx = val_idx if which == "val" else test_idx
            if len(idx) == 0:
                continue
            th_k = fold_ths[k] if fold_ths is not None else th
            r = _eval_static(oof[idx], y[idx], sensitive[idx], group_names, th_k)
            rows.append(r)
        if not rows:
            return NAN5
        arr = np.asarray(rows, dtype=float)
        return tuple(np.nanmean(arr, axis=0))

    sens_by_id = pd.Series(sensitive, index=groups)
    sens_by_id = sens_by_id[~sens_by_id.index.duplicated(keep="first")]

    rows = []
    for k, (tr_idx, val_idx, test_idx) in enumerate(splits):
        idx = val_idx if which == "val" else test_idx
        model, scaler = fp["fold_models"][k]
        coll = _collapse_fold_full_horizon(
            model, scaler, X, y, groups, time_arr, bin_times, feat_names,
            idx, n_bins, delta, device)
        if len(coll) == 0:
            continue
        th_k = fold_ths[k] if fold_ths is not None else th
        rows.append(_eval_dynamic_from_pdh(coll, sens_by_id, group_names, th_k))

    if not rows:
        return NAN5
    arr = np.asarray(rows, dtype=float)
    return tuple(np.nanmean(arr, axis=0))


  # Main run: run cross_validation and perform grid_search if flag=True
def run(X, y, groups, sensitive, splits, group_names,
        time_arr=None, subj_ids=None, model_name="",
        n_bins=None, collapse_pdh=False, is_dynamic=False,
        grid_search=False, coefs=None, verbose_folds=False,
        bin_times=None, feat_names=None, delta=None, device="cpu",
        **train_kwargs):

    # for a combination of alpha and beta and call _fit_predict for training 
    def _one(alpha, beta):
        fp = _fit_predict(splits, X, y, groups, sensitive, time_arr, subj_ids,
                          model_name, alpha, beta, n_bins, collapse_pdh,
                          verbose_folds=verbose_folds,
                          bin_times=bin_times, feat_names=feat_names,
                          delta=delta, device=device, **train_kwargs)
        th = fp["threshold"]

        # Valutazione PER FOLD (non pooled): ogni fold predice il proprio split
        # col PROPRIO modello -- usare un unico modello su tutti gli split
        # sarebbe leakage (4 fold su 5 avrebbero visto quei soggetti in training).
        # Le metriche sono poi mediate tra fold, stessa metodologia di
        # cv_results.csv, cosi' selezione del coefficiente e report finale
        # usano lo stesso criterio.
        val  = _fairness_per_fold(fp, splits, "val",  X, y, groups, time_arr,
                                  sensitive, group_names, th, n_bins, is_dynamic,
                                  bin_times, feat_names, delta, device)
        test = _fairness_per_fold(fp, splits, "test", X, y, groups, time_arr,
                                  sensitive, group_names, th, n_bins, is_dynamic,
                                  bin_times, feat_names, delta, device)
        fp["val"], fp["test"] = val, test
        return fp

    if not grid_search:
        r = _one(train_kwargs.pop("alpha", 0.0), train_kwargs.pop("beta", 0.0))
        auc_val, sep_auc_val, sep_mean_val, adtpr_val, adfpr_val = r["val"]
        auc_test, sep_auc_test, sep_mean_test, adtpr_test, adfpr_test = r["test"]
        
        # Store all predictions (full/test/val) that will be used be the related functions
        return dict(
            oof_preds=r["oof_full"],
            oof_test=r["oof_test"], is_test=r["is_test"],
            oof_val=r["oof_val"], is_val=r["is_val"],
            threshold=r["threshold"],
            fold_thresholds=r["fold_thresholds"],
            fold_models=r["fold_models"],
            auc_test=auc_test, separation_auc_test=sep_auc_test, separation_mean_test=sep_mean_test,
            adtpr_test=adtpr_test, adfpr_test=adfpr_test,
            auc_val=auc_val, separation_auc_val=sep_auc_val, separation_mean_val=sep_mean_val,
            adtpr_val=adtpr_val, adfpr_val=adfpr_val,
            model_last=r["model_last"], scaler_last=r["scaler_last"],
        )


    records = []
    coef_name = "alpha" if is_dynamic else "beta"

    # The baseline (coef=0.0) threshold is used to evaluate separation at a FIXED
    # threshold for every other coefficient, isolating the effect of the fairness
    # loss from the effect of the F1-optimal threshold moving with the coefficient.
    # Evaluated first (regardless of its position in `coefs`) so it's available
    # for every row, and cached so it's never trained twice.
    results_cache = {}

    def _eval_coef(c):
        r = _one(c, 0.0) if is_dynamic else _one(0.0, c)
        results_cache[c] = r
        return r

    fixed_th = None
    if 0.0 in coefs:
        fixed_th = _eval_coef(0.0)["threshold"]

    for c in coefs:
        r = results_cache[c] if c in results_cache else _eval_coef(c)

        if fixed_th is not None:
            val_fixed = _fairness_per_fold(r, splits, "val", X, y, groups, time_arr,
                                           sensitive, group_names, fixed_th, n_bins,
                                           is_dynamic, bin_times, feat_names, delta, device,
                                           use_fold_threshold=False)
            test_fixed = _fairness_per_fold(r, splits, "test", X, y, groups, time_arr,
                                            sensitive, group_names, fixed_th, n_bins,
                                            is_dynamic, bin_times, feat_names, delta, device,
                                            use_fold_threshold=False)
        else:
            val_fixed = (np.nan,) * 5
            test_fixed = (np.nan,) * 5

        # Combined selection criterion: worst-case (max) of the mobile-threshold
        # and fixed-threshold separation. A coefficient is only considered good
        # if it's good under BOTH threshold policies, not just one -- guards
        # against picking a coefficient that looks good only because the F1
        # threshold happened to move favorably for it at that specific point.
        sep_val_mobile = r["val"][1]
        sep_val_fixed = val_fixed[1]
       
        records.append({
            "coef": c, "coef_name": coef_name,
            # selection is done on VAL, own (per-coefficient) F1-optimal threshold
            "auc_mean": r["val"][0], "separation_auc": r["val"][1], "separation_mean": r["val"][2],
            # adTPR/adFPR (medie semplici -> robuste al rumore, a differenza di SEP-AUC)
            "adTPR": r["val"][3], "adFPR": r["val"][4],
            # unbiased report on TEST, own threshold
            "auc_mean_test": r["test"][0], "separation_auc_test": r["test"][1],
            "separation_mean_test": r["test"][2],
            "adTPR_test": r["test"][3], "adFPR_test": r["test"][4],
            "threshold": r["threshold"],
            # same predictions, but re-evaluated at the FIXED (baseline) threshold
            "separation_auc_val_fixed": val_fixed[1], "separation_mean_val_fixed": val_fixed[2],
            "separation_auc_test_fixed": test_fixed[1], "separation_mean_test_fixed": test_fixed[2],
            "fixed_threshold": fixed_th,
        })
    return pd.DataFrame(records)


# Run Cross_Validation
# Final CV for one model at a fixed coefficient. 
def run_cv(X, y, groups, sensitive,
           time_arr=None, subj_ids=None,
           model_name="", n_splits=5,
           landmarks=None, collapse_pdh=False, n_bins=None,
           group_names=None, splits=None,
           val_size=0.5, split_seed=SEED,
           bin_times=None, feat_names=None, delta=None, device="cpu",
           **train_kwargs):
    
    if splits is None:
        splits = make_splits(y, groups, n_splits=n_splits, val_size=val_size, seed=split_seed)

    r = run(X, y, groups, sensitive, splits, group_names,
            time_arr=time_arr, subj_ids=subj_ids, model_name=model_name,
            n_bins=n_bins, collapse_pdh=collapse_pdh,
            is_dynamic=(time_arr is not None), grid_search=False, verbose_folds=True,
            bin_times=bin_times, feat_names=feat_names, delta=delta, device=device,
            **train_kwargs)

    # per-fold performance metrics on the test portion only, for the summary table.
    # For the dynamic model (collapse_pdh=True) this now uses the INTEGRATED
    # per-landmark AUC/Brier -- the pooled (person-period aggregate) version has
    # been removed, consistent with _eval_dynamic. F1 is not reported for either
    # model: the threshold is still chosen by maximizing F1 (find_best_threshold),
    # but F1 itself is no longer surfaced as a metric.
    metrics_list = []
    for k, (tr_idx, val_idx, test_idx) in enumerate(splits):
        if collapse_pdh:
            # full-horizon PD-H col modello DI QUESTO fold (no leakage)
            model_k, scaler_k = r["fold_models"][k]
            te_pdh = _collapse_fold_full_horizon(
                model_k, scaler_k, X, y, groups, time_arr, bin_times, feat_names,
                test_idx, n_bins, delta, device)
            df_perf = _perf_by_landmark(te_pdh["yh"].to_numpy().astype(int),
                                        te_pdh["pdh"].to_numpy(),
                                        te_pdh["L"].to_numpy())
            metrics_list.append(dict(
                AUC=_integrate_curve(df_perf, "auc"),
                Brier=_integrate_curve(df_perf, "brier"),
                Th=r["fold_thresholds"][k],
            ))
        else:
            metrics_list.append(metrics_all(y[test_idx].astype(int),
                                            r["oof_test"][test_idx],
                                            r["fold_thresholds"][k]))

    summary = agg_mean_sd(metrics_list)
    summary["Model"] = model_name.upper()

    oof_test_only = np.full(len(y), np.nan, dtype=np.float64)
    oof_test_only[r["is_test"]] = r["oof_test"][r["is_test"]]

    r["oof_preds"] = oof_test_only
    r["metrics"] = metrics_list
    r["summary"] = summary
    return r

# Loop on alpha and beta list
def run_grid_search(
    X_static, y_static, grp_static, sens_static,
    X_dynamic, y_dynamic, grp_dynamic, sens_dynamic, lmk_vals,
    group_names,
    betas=None, alphas=None,
    n_folds=5, eo_mode_d="mean", schedule_mode_d="flat",
    n_bins=None, val_size=0.5, split_seed=SEED,
    splits_static=None, splits_dynamic=None,
    bin_times=None, feat_names=None, delta=None, device="cpu",
    out_dir=Path("outputs"), run_tag="run",
):

    # NB: non serve piu' reimpostare i seed qui -- _fit_predict lo fa in modo
    # deterministico all'inizio di OGNI fold (_reset_seed), quindi run_cv e
    # run_grid_search producono gli stessi modelli a parita' di iperparametri.

    if betas is None:
        betas = [0.0, 0.3, 0.5, 0.7, 1.0]
    if alphas is None:
        alphas = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2]
    if splits_static is None:
        splits_static = make_splits(y_static, grp_static, n_folds, val_size, split_seed)
    if splits_dynamic is None:
        splits_dynamic = make_splits(y_dynamic, grp_dynamic, n_folds, val_size, split_seed)

    # M_STATIC (coefficient = beta)
    print( "\nGRID SEARCH — M_STATIC\n" + "=" * 60)
    df_s = run(X_static, y_static, grp_static, sens_static, splits_static, group_names,
               model_name="static", is_dynamic=False, eo_mode_d=eo_mode_d,
               grid_search=True, coefs=betas)
    df_s["model"] = "M_STATIC"
    for _, row in df_s.iterrows():
        print(f"  beta={row['coef']:.2f}  AUC={row['auc_mean']:.4f}  "
              f"sep={row['separation_auc']:.4f}  "
              f"adTPR={row['adTPR']:.4f}  adFPR={row['adFPR']:.4f}")

    # M_DYNAMIC (coefficient = alpha)
    print("\n" + "\nGRID SEARCH — M_DYNAMIC\n" + "=" * 60)
    df_d = run(X_dynamic, y_dynamic, grp_dynamic, sens_dynamic, splits_dynamic, group_names,
               time_arr=lmk_vals, subj_ids=grp_dynamic, model_name="dynamic",
               n_bins=n_bins, collapse_pdh=True, is_dynamic=True,
               eo_mode_d=eo_mode_d, schedule_mode_d=schedule_mode_d,
               grid_search=True, coefs=alphas,
               bin_times=bin_times, feat_names=feat_names, delta=delta, device=device)
    df_d["model"] = "M_DYNAMIC"
    for _, row in df_d.iterrows():
        print(f"  alpha={row['coef']:.2f}  AUC={row['auc_mean']:.4f}  "
              f"sep={row['separation_auc']:.4f}  "
              f"adTPR={row['adTPR']:.4f}  adFPR={row['adFPR']:.4f}")

    df_grid = pd.concat([df_s, df_d], ignore_index=True)
    out_dir = Path(out_dir)
    df_grid.to_csv(out_dir / f"grid_tradeoff_{run_tag}.csv", index=False)
    print(df_grid.to_string(index=False))
    return df_grid


def build_summary_table(cv_results):
    # Underlying values are unchanged: AUC_Mean/Brier_Mean are ordinary AUC/Brier
    # for M_STATIC, and the INTEGRATED per-landmark AUC/Brier for M_DYNAMIC
    # (see run_cv). Only the displayed column labels are generalized here to
    # make that distinction explicit without needing separate columns per model.
    rows = []
    for name, res in cv_results.items():
        row = res["summary"].copy()
        row["Model"] = name
        rows.append(row)
    df = pd.DataFrame(rows)
    rename_map = {
        "AUC_Mean": "AUC/iAUC Mean", "AUC_SD": "AUC/iAUC SD",
        "Brier_Mean": "BS/IBS Mean", "Brier_SD": "BS/IBS SD",
    }
    df = df.rename(columns=rename_map)
    cols = ["Model", "AUC/iAUC Mean", "AUC/iAUC SD", "BS/IBS Mean", "BS/IBS SD"]
    return df[[c for c in cols if c in df.columns]]


def plot_tradeoff(df_grid, out_dir, run_tag="run"):
    # No automatic "best" selection anymore -- the plot shows both the
    # mobile-threshold and fixed-threshold separation curves so the
    # coefficient can be chosen by inspection, with explicit reasoning.
    static_base = df_grid[(df_grid["model"] == "M_STATIC") &
                          (df_grid["coef"] == 0.0)]["auc_mean"].values
    static_base = float(static_base[0]) if len(static_base) > 0 else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("AUC and Separation (val) as a function of the fairness coefficient\n"
                 "solid = AUC | dashed = separation (mobile threshold) | "
                 "dotted = separation (fixed threshold)",
                 fontsize=12, fontweight="bold", y=1.04)

    for ax, (model_name, style) in zip(axes, MODEL_STYLES.items()):
        sub = df_grid[df_grid["model"] == model_name]\
            .dropna(subset=["auc_mean", "separation_auc"])\
            .sort_values("coef").reset_index(drop=True)
        if sub.empty:
            ax.set_title(f"{model_name} — no data")
            continue

        coefs = sub["coef"].to_numpy()
        aucs = sub["auc_mean"].to_numpy()
        seps_mobile = sub["separation_auc"].to_numpy()
        # fixed-threshold separation may be all-NaN if coef=0.0 wasn't in the grid
        seps_fixed = (sub["separation_auc_val_fixed"].to_numpy()
                      if "separation_auc_val_fixed" in sub.columns
                      else np.full_like(seps_mobile, np.nan))
        color, clabel = style["color"], style["coef_label"]

        ax2 = ax.twinx()
        ax.plot(coefs, aucs, color=color, linewidth=2.2, marker=style["marker"],
                markersize=7, zorder=3)
        ax2.plot(coefs, seps_mobile, color=color, linewidth=2.2, linestyle="--",
                 marker=style["marker"], markersize=7, alpha=0.55, zorder=3)
        if not np.all(np.isnan(seps_fixed)):
            ax2.plot(coefs, seps_fixed, color=color, linewidth=2.2, linestyle=":",
                     marker=style["marker"], markersize=6, alpha=0.85, zorder=4)

        # draw the static AUC baseline used as constraint for the dynamic model
        if model_name == "M_DYNAMIC":
            ax.axhline(static_base, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)

        ax.set_xlabel(f"coefficient ({clabel})", fontsize=11)
        ax.set_ylabel("AUC val  (higher is better)", fontsize=10, color=color)
        ax2.set_ylabel("Separation val  (lower is fairer)", fontsize=10, color=color)
        ax.tick_params(axis="y", labelcolor=color)
        ax2.tick_params(axis="y", labelcolor=color)
        ax.set_title(model_name, fontsize=12, fontweight="bold", color=color)
        ax.grid(alpha=0.2, linestyle="--")

        legend_handles = [
            Line2D([0], [0], color=color, linewidth=2, marker=style["marker"],
                   label="AUC val (solid)"),
            Line2D([0], [0], color=color, linewidth=2, linestyle="--",
                   marker=style["marker"], alpha=0.55, label="Separation val, mobile threshold (dashed)"),
            Line2D([0], [0], color=color, linewidth=2, linestyle=":",
                   marker=style["marker"], alpha=0.85, label="Separation val, fixed threshold (dotted)"),
        ]
        if model_name == "M_DYNAMIC":
            legend_handles.append(
                Line2D([0], [0], color="gray", linestyle=":", linewidth=1.5,
                       label=f"Static AUC baseline ({static_base:.3f})")
            )
        ax.legend(handles=legend_handles, fontsize=8, loc="lower left", framealpha=0.9)

    plt.tight_layout()
    plot_path = Path(out_dir) / f"tradeoff_{run_tag}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path