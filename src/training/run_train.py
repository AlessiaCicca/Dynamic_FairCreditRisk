"""
Entire training process with GroupKFold cross validation and Grid Search
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import (roc_auc_score, precision_recall_curve)

from src.training.train_mlp import train_mlp
from src.evaluation.fold_evaluation import integrate_curve, perf_by_landmark, metrics_all, agg_mean_sd, eval_static, eval_dynamic_from_pdh
from config import SEED


MODEL_STYLES = {"M_STATIC":  {"color": "#3A6BC4", "marker": "o", "coef_label": "beta"}, "M_DYNAMIC": {"color": "#D4612A", "marker": "s", "coef_label": "alpha"}}

# Reproducibility -> ensure that both run_cv and grid_search are initialized with the same random seed
def reset_seed(fold, seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

# F1-optimal threshold -> threshold for 0/1 definition as the value that maximize F1
def find_best_threshold(y_true, p):
    p = np.clip(p, 0, 1)
    prec, rec, thresholds = precision_recall_curve(y_true, p)
    if len(thresholds) == 0:
        return 0.5
    f1_scores = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-8)
    return float(thresholds[np.argmax(f1_scores)])



# From hazard per bin to PD(L,L+h)
# From (Tanner et al. 2021, createPredictData + cumprod function)
def collapse_fold_full_horizon(model, scaler, X, y, groups, lmk, bin_times,
                                 feat_names, idx, n_bins, delta, device, complete_only=True):

    
    spl_idx = [i for i, f in enumerate(feat_names) if str(f).startswith("spl_")]
    if not spl_idx:
        print("Nessuna colonna 'spl_*' in feature_names")
        
    # d is a DataFrame at bin level - g/out are at landmark level
    d = pd.DataFrame({"row": idx,"id":  groups[idx],"L":   lmk[idx],"ev":  y[idx],})                                    
    g  = d.groupby(["id", "L"], sort=False)
    out = pd.DataFrame({"yh":  g["ev"].max(),"n":   g.size(),"row": g["row"].first()}).reset_index()

    # Require all bins or default - discard censoring                              
    if complete_only:
        out = out[(out["n"] == n_bins) | (out["yh"] == 1)].reset_index(drop=True)
    if len(out) == 0:
        return out.assign(pdh=[])

    # Recreate n_bins row
    n_groups = len(out)
    X_rep  = X[out["row"].to_numpy()]
    X_full = np.repeat(X_rep, n_bins, axis=0)

    # Retrieve real spline
    bt_to_spl = {}
    for bt_val in np.unique(bin_times):
        j = np.where(bin_times == bt_val)[0][0]
        bt_to_spl[bt_val] = X[j, spl_idx]

    L_rep  = np.repeat(out["L"].to_numpy(), n_bins)
    j_rep = np.tile(np.arange(n_bins), n_groups)
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


# Main function of the file
# For each pre-computed (train, val, test) fold: fit on train, pick the threshold
# on train, and store predictions into val / test
def fit_predict(splits, X, y, groups, sensitive,
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
    fold_models = []   # (model, scaler) per fold
    
    for fold, (tr_idx, val_idx, test_idx) in enumerate(splits):
        te_idx = np.concatenate([val_idx, test_idx])
        reset_seed(fold)

        # train MLP on the training fold, predict on the whole held-out (val + test)
        p_te, p_tr, model, scaler = train_mlp(
            X[tr_idx], y[tr_idx], X[te_idx], y[te_idx],
            sensitive_tr=sensitive[tr_idx] if sensitive is not None else None,
            time_tr=time_arr[tr_idx] if is_dyn else None,
            subj_ids_tr=subj_ids[tr_idx] if subj_ids is not None else None,
            model_name=model_name, alpha=alpha, beta=beta,
            verbose=(verbose_folds and fold == 0), **train_kwargs)

        # Divide predictions in val and test
        pos = {idx: k for k, idx in enumerate(te_idx)}
        oof_val[val_idx] = p_te[[pos[i] for i in val_idx]]
        oof_test[test_idx] = p_te[[pos[i] for i in test_idx]]
        oof_full[te_idx] = p_te
        is_val[val_idx] = True
        is_test[test_idx] = True

        if collapse_pdh:
            val_pdh_th = collapse_fold_full_horizon(
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
                val_pdh = collapse_fold_full_horizon(model, scaler, X, y, groups, time_arr, bin_times, feat_names,val_idx, n_bins, delta, device)
                df_perf_fold = perf_by_landmark(val_pdh["yh"].to_numpy().astype(int),val_pdh["pdh"].to_numpy(),val_pdh["L"].to_numpy())
                auc_fold = integrate_curve(df_perf_fold, "auc")
                pm = val_pdh["pdh"].mean()
            else:
                val_pos  = [pos[i] for i in val_idx]
                p_val    = p_te[val_pos]
                auc_fold = (roc_auc_score(y[val_idx], p_val)
                            if len(np.unique(y[val_idx])) > 1 else float("nan"))
                pm = p_val.mean()
            print(f"  Fold {fold + 1}  |  pred_mean_val={pm:.4f}"
                  f"  |  AUC (val): {auc_fold:.4f}  |  th={thresholds[-1]:.5f}")

    return dict( oof_val=oof_val, oof_test=oof_test, oof_full=oof_full,
        is_val=is_val, is_test=is_test,
        threshold=float(np.mean(thresholds)),  fold_thresholds=thresholds,
        model_last=model_last, scaler_last=scaler_last,fold_models=fold_models)



def fairness_per_fold(fp, splits, which, X, y, groups, time_arr, sensitive,
                       group_names, th, n_bins, is_dyn,
                       bin_times=None, feat_names=None, delta=None, device="cpu",
                       use_fold_threshold=True):
    
    NAN5 = (np.nan,) * 5
    if group_names is None:
        return NAN5

    fold_ths = fp.get("fold_thresholds") if use_fold_threshold else None

    if not is_dyn:
        oof = fp["oof_val"] if which == "val" else fp["oof_test"]
        rows = []
        for k, (tr_idx, val_idx, test_idx) in enumerate(splits):
            idx = val_idx if which == "val" else test_idx
            if len(idx) == 0:
                continue
            th_k = fold_ths[k] if fold_ths is not None else th
            r = eval_static(oof[idx], y[idx], sensitive[idx], group_names, th_k)
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
        coll = collapse_fold_full_horizon(
            model, scaler, X, y, groups, time_arr, bin_times, feat_names,
            idx, n_bins, delta, device)
        if len(coll) == 0:
            continue
        th_k = fold_ths[k] if fold_ths is not None else th
        rows.append(eval_dynamic_from_pdh(coll, sens_by_id, group_names, th_k))

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

    # for a combination of alpha and beta and call fit_predict for training 
    def _one(alpha, beta):
        fp = fit_predict(splits, X, y, groups, sensitive, time_arr, subj_ids,
                          model_name, alpha, beta, n_bins, collapse_pdh,
                          verbose_folds=verbose_folds,
                          bin_times=bin_times, feat_names=feat_names,
                          delta=delta, device=device, **train_kwargs)
        th = fp["threshold"]
        val  = fairness_per_fold(fp, splits, "val",  X, y, groups, time_arr, sensitive, group_names, th, n_bins, is_dynamic,
                                  bin_times, feat_names, delta, device)
        test = fairness_per_fold(fp, splits, "test", X, y, groups, time_arr,
                                  sensitive, group_names, th, n_bins, is_dynamic, bin_times, feat_names, delta, device)
        fp["val"], fp["test"] = val, test
        return fp

    if not grid_search:
        r = _one(train_kwargs.pop("alpha", 0.0), train_kwargs.pop("beta", 0.0))
        auc_val, sep_auc_val, sep_mean_val, adtpr_val, adfpr_val = r["val"]
        auc_test, sep_auc_test, sep_mean_test, adtpr_test, adfpr_test = r["test"]
        
        # Store all predictions (full/test/val) that will be used be the related functions
        return dict( oof_preds=r["oof_full"],
            oof_test=r["oof_test"], is_test=r["is_test"],
            oof_val=r["oof_val"], is_val=r["is_val"],
            threshold=r["threshold"], fold_thresholds=r["fold_thresholds"],
            fold_models=r["fold_models"],
            auc_test=auc_test, separation_auc_test=sep_auc_test, separation_mean_test=sep_mean_test,
            adtpr_test=adtpr_test, adfpr_test=adfpr_test,
            auc_val=auc_val, separation_auc_val=sep_auc_val, separation_mean_val=sep_mean_val,
            adtpr_val=adtpr_val, adfpr_val=adfpr_val,
            model_last=r["model_last"], scaler_last=r["scaler_last"] )


    records = []
    coef_name = "alpha" if is_dynamic else "beta"
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
            val_fixed = fairness_per_fold(r, splits, "val", X, y, groups, time_arr,sensitive, group_names, fixed_th, n_bins,
                                           is_dynamic, bin_times, feat_names, delta, device, use_fold_threshold=False)
            test_fixed = fairness_per_fold(r, splits, "test", X, y, groups, time_arr,sensitive, group_names, fixed_th, n_bins,
                                            is_dynamic, bin_times, feat_names, delta, device,use_fold_threshold=False)
        else:
            val_fixed = (np.nan,) * 5
            test_fixed = (np.nan,) * 5


        sep_val_mobile = r["val"][1]
        sep_val_fixed = val_fixed[1]
       
        records.append({
            "coef": c, "coef_name": coef_name,
            "auc_mean": r["val"][0], "separation_auc": r["val"][1], "separation_mean": r["val"][2],
            "adTPR": r["val"][3], "adFPR": r["val"][4],
            "auc_mean_test": r["test"][0], "separation_auc_test": r["test"][1],
            "separation_mean_test": r["test"][2],
            "adTPR_test": r["test"][3], "adFPR_test": r["test"][4],
            "threshold": r["threshold"],
            "separation_auc_val_fixed": val_fixed[1], "separation_mean_val_fixed": val_fixed[2],
            "separation_auc_test_fixed": test_fixed[1], "separation_mean_test_fixed": test_fixed[2],
            "fixed_threshold": fixed_th})
    return pd.DataFrame(records)


# Run Cross_Validation
# Final CV for one model at a fixed coefficient. 
def run_cv(X, y, groups, sensitive,time_arr=None, subj_ids=None,
           model_name="", n_splits=5, landmarks=None, collapse_pdh=False, n_bins=None, group_names=None, splits=None,
           val_size=0.5, split_seed=SEED, bin_times=None, feat_names=None, delta=None, device="cpu",**train_kwargs):
    
    if splits is None:
        splits = make_splits(y, groups, n_splits=n_splits, val_size=val_size, seed=split_seed)

    r = run(X, y, groups, sensitive, splits, group_names,
            time_arr=time_arr, subj_ids=subj_ids, model_name=model_name,
            n_bins=n_bins, collapse_pdh=collapse_pdh,
            is_dynamic=(time_arr is not None), grid_search=False, verbose_folds=True,
            bin_times=bin_times, feat_names=feat_names, delta=delta, device=device,
            **train_kwargs)


    metrics_list = []
    for k, (tr_idx, val_idx, test_idx) in enumerate(splits):
        if collapse_pdh:
            # full-horizon PD-H col modello DI QUESTO fold (no leakage)
            model_k, scaler_k = r["fold_models"][k]
            te_pdh = collapse_fold_full_horizon(
                model_k, scaler_k, X, y, groups, time_arr, bin_times, feat_names,
                test_idx, n_bins, delta, device)
            df_perf = perf_by_landmark(te_pdh["yh"].to_numpy().astype(int),
                                        te_pdh["pdh"].to_numpy(),
                                        te_pdh["L"].to_numpy())
            metrics_list.append(dict(
                AUC=integrate_curve(df_perf, "auc"),
                Brier=integrate_curve(df_perf, "brier"),
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
    t_min=0.0, t_max=48.0,  
    out_dir=Path("outputs"), run_tag="run",
):


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
               grid_search=True, coefs=betas,
               t_min=t_min, t_max=t_max) 
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
               bin_times=bin_times, feat_names=feat_names, delta=delta, device=device,
               t_min=t_min, t_max=t_max) 
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
