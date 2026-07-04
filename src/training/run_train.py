"""
Unified GroupKFold cross-validation for both final evaluation and grid search.

GroupKFold: A cross-validation method that splits the data into K folds while keeping all
observations from the same subject in the same fold, preventing data leakage across train and
test sets.

OOF (Out-of-Fold) predictions: Predictions generated for each subject when that subject belongs
to the held-out fold, ensuring that every prediction is produced by a model that was not trained
on that subject.

The train/val/test division is decided ONCE (make_splits) and reused by both entry points, so it
is univocal: what the grid search uses as `val` (to select the coefficient) stays disjoint from
`test` (used to report it), instead of relying on a matching random seed in two places.
  - train : fit the model and pick the F1-optimal threshold
  - val   : select the fairness coefficient (grid search only)
  - test  : unbiased report of the selected coefficient
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, f1_score, precision_recall_curve
)

from src.training.train_mlp import train_mlp
from src.evaluation.fairness_metrics import fairness_metrics, filter_sensitive
from config import SEED


MODEL_STYLES = {
    "M_STATIC":  {"color": "#3A6BC4", "marker": "o", "coef_label": "beta"},
    "M_DYNAMIC": {"color": "#D4612A", "marker": "s", "coef_label": "alpha"},
}


# --------------------------------------------------------------------------- #
#  Split defined upfront (single source of truth)
# --------------------------------------------------------------------------- #
def make_splits(y, groups, n_splits=5, val_size=0.5, seed=SEED):
    """
    Build the train/val/test folds ONCE and return them as a list of
    (train_idx, val_idx, test_idx) tuples.
    - GroupKFold splits by subject: same loan is never both in train and held-out.
    - GroupShuffleSplit splits the held-out fold into val/test, again by subject,
      with a fixed seed. This is THE division: run_cv and run_grid_search receive it,
      they do not recreate it.
    """
    gkf = GroupKFold(n_splits=n_splits)
    splits = []
    for tr_idx, te_idx in gkf.split(np.zeros(len(y)), y, groups):
        gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        a_pos, b_pos = next(gss.split(te_idx, groups=groups[te_idx]))
        splits.append((tr_idx, te_idx[a_pos], te_idx[b_pos]))
    return splits


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def find_best_threshold(y_true, p, max_th_quantile=0.90):
    """F1-optimal threshold, capped at a high quantile of the scores."""
    p = np.clip(p, 0, 1)
    prec, rec, thresholds = precision_recall_curve(y_true, p)
    if len(thresholds) == 0:
        return 0.5
    max_th = np.quantile(p, max_th_quantile)
    f1_scores = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-8)
    f1_scores[thresholds > max_th] = 0
    return float(thresholds[np.argmax(f1_scores)])


def _collapse_fold(hazard, event_bin, ids, lmk, n_bins, complete_only=True):
    """From per-bin hazard to PD(L, L+h), aggregating by (subject, landmark)."""
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
    # require all bins present for the (subject, landmark)
    if complete_only:
        out = out[out["n"] == n_bins]
    return out


def metrics_all(y_true, p, threshold=0.5):
    """AUC, Brier score, F1 for a single set of predictions."""
    p = np.clip(p, 0, 1)
    auc = roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else np.nan
    return dict(
        AUC=auc,
        Brier=brier_score_loss(y_true, p),
        F1=f1_score(y_true, (p >= threshold).astype(int), zero_division=0),
        Th=threshold,
    )


def agg_mean_sd(list_of_dicts):
    """Compute mean and sd across all folds."""
    out = {}
    for k in list_of_dicts[0].keys():
        vals = [d[k] for d in list_of_dicts]
        out[f"{k}_Mean"] = float(np.nanmean(vals))
        out[f"{k}_SD"] = float(np.nanstd(vals))
    return out


# --------------------------------------------------------------------------- #
#  Separation evaluation on a set of predictions
# --------------------------------------------------------------------------- #
def _eval_static(preds, y, sens, group_names, eval_th):
    """AUC and separation on the static (aggregate) predictions."""
    yt_f, yp_f, sn_f = filter_sensitive(np.asarray(y).astype(int), preds, sens)
    if len(np.unique(yt_f)) < 2 or len(np.unique(sn_f)) < 2:
        return np.nan, np.nan, np.nan
    auc = roc_auc_score(yt_f, yp_f)
    yb_f = (yp_f >= eval_th).astype(int)
    res = fairness_metrics(yt_f, yp_f, yb_f, sn_f, group_names, threshold=eval_th)
    s = res.get("axioms", {}).get("separation", np.nan)
    return auc, s, s


def _eval_dynamic(preds, y, ids, time, sens, group_names, eval_th, n_bins):
    """
    Collapse per-bin hazards to PD-H, then compute AUC and the separation curve
    over landmarks; the fairness AUC is the area under that curve (normalized time).
    """
    coll = _collapse_fold(preds, y, ids, time, n_bins).reset_index(drop=True)

    # one sensitive value per subject
    sens_by_id = pd.Series(sens, index=ids)
    sens_by_id = sens_by_id[~sens_by_id.index.duplicated(keep="first")]
    coll["sens"] = coll["id"].map(sens_by_id)

    eval_preds = coll["pdh"].to_numpy()
    eval_y = coll["yh"].to_numpy().astype(int)
    eval_sens = coll["sens"].to_numpy()
    eval_time = coll["L"].to_numpy()

    auc = roc_auc_score(eval_y, eval_preds) if len(np.unique(eval_y)) > 1 else np.nan

    # separation per landmark
    time_rows = []
    for t in sorted(np.unique(eval_time)):
        mask = eval_time == t
        yt_f, yp_f, sn_f = filter_sensitive(eval_y[mask], eval_preds[mask], eval_sens[mask])
        if len(np.unique(yt_f)) < 2 or len(np.unique(sn_f)) < 2:
            continue
        # skip landmarks with too few samples in the smallest group
        if pd.Series(sn_f).value_counts().min() < 50:
            continue
        yb_f = (yp_f >= eval_th).astype(int)
        res = fairness_metrics(yt_f, yp_f, yb_f, sn_f, group_names, threshold=eval_th)
        time_rows.append({"t": t, "separation": res.get("axioms", {}).get("separation", np.nan)})

    df_t = pd.DataFrame(time_rows)

    # integrate the separation curve over normalized time (area under the curve)
    def trapz_norm(col):
        if df_t.empty or col not in df_t.columns:
            return np.nan
        sub = df_t.dropna(subset=[col])
        if len(sub) < 3:
            return np.nan
        t_v = sub["t"].to_numpy(float)
        v = sub[col].to_numpy(float)
        t_n = (t_v - t_v.min()) / (t_v.max() - t_v.min() + 1e-9)
        return float(np.trapezoid(v, t_n))

    sep_auc = trapz_norm("separation")
    sep_mean = (df_t["separation"].mean()
                if (not df_t.empty and "separation" in df_t.columns) else np.nan)
    return auc, sep_auc, sep_mean


# --------------------------------------------------------------------------- #
#  Shared engine: fit one coefficient over the given splits
# --------------------------------------------------------------------------- #
def _fit_predict(splits, X, y, groups, sensitive,
                 time_arr, subj_ids, model_name,
                 alpha, beta, n_bins, collapse_pdh, **train_kwargs):
    """
    For each pre-computed (train, val, test) fold: fit on train, pick the threshold
    on train, and store predictions into val / test / full OOF arrays. The split is
    taken as-is from `splits` (see make_splits), never recreated here.
    """
    is_dyn = time_arr is not None
    oof_val = np.zeros(len(y), dtype=np.float64)
    oof_test = np.zeros(len(y), dtype=np.float64)
    oof_full = np.zeros(len(y), dtype=np.float64)
    is_val = np.zeros(len(y), dtype=bool)
    is_test = np.zeros(len(y), dtype=bool)

    thresholds = []
    model_last = scaler_last = None

    for tr_idx, val_idx, test_idx in splits:
        te_idx = np.concatenate([val_idx, test_idx])

        # train MLP on the training fold, predict on the whole held-out (val + test)
        p_te, p_tr, model, scaler = train_mlp(
            X[tr_idx], y[tr_idx], X[te_idx], y[te_idx],
            sensitive_tr=sensitive[tr_idx] if sensitive is not None else None,
            time_tr=time_arr[tr_idx] if is_dyn else None,
            subj_ids_tr=subj_ids[tr_idx] if subj_ids is not None else None,
            model_name=model_name, alpha=alpha, beta=beta,
            verbose=False, **train_kwargs,
        )

        # map held-out predictions back to global val / test / full positions
        pos = {idx: k for k, idx in enumerate(te_idx)}
        oof_val[val_idx] = p_te[[pos[i] for i in val_idx]]
        oof_test[test_idx] = p_te[[pos[i] for i in test_idx]]
        oof_full[te_idx] = p_te
        is_val[val_idx] = True
        is_test[test_idx] = True

        # threshold is ALWAYS chosen on train (no leakage)
        if collapse_pdh:
            tr_pdh = _collapse_fold(p_tr, y[tr_idx], groups[tr_idx], time_arr[tr_idx], n_bins)
            thresholds.append(find_best_threshold(tr_pdh["yh"], tr_pdh["pdh"]))
        else:
            thresholds.append(find_best_threshold(y[tr_idx], p_tr))

        model_last, scaler_last = model, scaler

    return dict(
        oof_val=oof_val, oof_test=oof_test, oof_full=oof_full,
        is_val=is_val, is_test=is_test,
        threshold=float(np.mean(thresholds)),
        model_last=model_last, scaler_last=scaler_last,
    )


def _fairness(oof, mask, y, groups, time_arr, sensitive, group_names, th, n_bins, is_dyn):
    """Aggregate AUC + separation on the subset selected by `mask`."""
    if group_names is None or mask.sum() == 0:
        return np.nan, np.nan, np.nan
    if is_dyn:
        return _eval_dynamic(oof[mask], y[mask], groups[mask], time_arr[mask],
                             sensitive[mask], group_names, th, n_bins)
    return _eval_static(oof[mask], y[mask], sensitive[mask], group_names, th)


# --------------------------------------------------------------------------- #
#  Single entry point: the `grid_search` flag is the only branch
# --------------------------------------------------------------------------- #
def run(X, y, groups, sensitive, splits, group_names,
        time_arr=None, subj_ids=None, model_name="",
        n_bins=None, collapse_pdh=False, is_dynamic=False,
        grid_search=False, coefs=None, **train_kwargs):
    """
    Run cross-validation on the pre-computed `splits` (univocal division).

    grid_search=False : one fixed coefficient (alpha/beta in train_kwargs); reports on TEST.
    grid_search=True  : loop over `coefs`; select on VAL, report on TEST.

    In both cases val and test come from the same `splits`, so selection (val) and
    report (test) are guaranteed disjoint.
    """
    def _one(alpha, beta):
        fp = _fit_predict(splits, X, y, groups, sensitive, time_arr, subj_ids,
                          model_name, alpha, beta, n_bins, collapse_pdh, **train_kwargs)
        th = fp["threshold"]
        val = _fairness(fp["oof_val"], fp["is_val"], y, groups, time_arr,
                        sensitive, group_names, th, n_bins, is_dynamic)
        test = _fairness(fp["oof_test"], fp["is_test"], y, groups, time_arr,
                         sensitive, group_names, th, n_bins, is_dynamic)
        fp["val"], fp["test"] = val, test
        return fp

    # ---- final evaluation: single coefficient ----
    if not grid_search:
        r = _one(train_kwargs.pop("alpha", 0.0), train_kwargs.pop("beta", 0.0))
        auc_val, sep_auc_val, sep_mean_val = r["val"]
        auc_test, sep_auc_test, sep_mean_test = r["test"]
        return dict(
            oof_preds=r["oof_full"],           # full held-out for downstream descriptive analysis
            oof_test=r["oof_test"], is_test=r["is_test"],
            oof_val=r["oof_val"], is_val=r["is_val"],
            threshold=r["threshold"],
            auc_test=auc_test, separation_auc_test=sep_auc_test, separation_mean_test=sep_mean_test,
            auc_val=auc_val, separation_auc_val=sep_auc_val, separation_mean_val=sep_mean_val,
            model_last=r["model_last"], scaler_last=r["scaler_last"],
        )

    # ---- grid search: one row per coefficient ----
    records = []
    coef_name = "alpha" if is_dynamic else "beta"
    for c in coefs:
        r = _one(c, 0.0) if is_dynamic else _one(0.0, c)
        records.append({
            "coef": c, "coef_name": coef_name,
            # selection is done on VAL
            "auc_mean": r["val"][0], "separation_auc": r["val"][1], "separation_mean": r["val"][2],
            # unbiased report on TEST
            "auc_mean_test": r["test"][0], "separation_auc_test": r["test"][1],
            "separation_mean_test": r["test"][2],
            "threshold": r["threshold"],
        })
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
#  Backward-compatible wrappers (same names used by the experiments)
# --------------------------------------------------------------------------- #
def run_cv(X, y, groups, sensitive,
           time_arr=None, subj_ids=None,
           model_name="", n_splits=5,
           landmarks=None, collapse_pdh=False, n_bins=None,
           group_names=None, splits=None,
           val_size=0.5, split_seed=SEED, **train_kwargs):
    """
    Final CV for one model at a fixed coefficient. If `splits` is not given it is
    built here with make_splits (so the caller can share the SAME splits between
    run_cv and run_grid_search).
    """
    if splits is None:
        splits = make_splits(y, groups, n_splits=n_splits, val_size=val_size, seed=split_seed)

    r = run(X, y, groups, sensitive, splits, group_names,
            time_arr=time_arr, subj_ids=subj_ids, model_name=model_name,
            n_bins=n_bins, collapse_pdh=collapse_pdh,
            is_dynamic=(time_arr is not None), grid_search=False, **train_kwargs)

    # per-fold performance metrics on the full held-out, for the summary table
    metrics_list = []
    for tr_idx, val_idx, test_idx in splits:
        te_idx = np.concatenate([val_idx, test_idx])
        if collapse_pdh:
            te_pdh = _collapse_fold(r["oof_preds"][te_idx], y[te_idx],
                                    groups[te_idx], time_arr[te_idx], n_bins)
            metrics_list.append(metrics_all(te_pdh["yh"].astype(int),
                                            te_pdh["pdh"], r["threshold"]))
        else:
            metrics_list.append(metrics_all(y[te_idx].astype(int),
                                            r["oof_preds"][te_idx], r["threshold"]))

    summary = agg_mean_sd(metrics_list)
    summary["Model"] = model_name.upper()
    r["metrics"] = metrics_list
    r["summary"] = summary
    return r


def run_grid_search(
    X_static, y_static, grp_static, sens_static,
    X_dynamic, y_dynamic, grp_dynamic, sens_dynamic, lmk_vals,
    group_names,
    betas=None, alphas=None,
    n_folds=5, eo_mode_d="mean", schedule_mode_d="flat",
    n_bins=None, val_size=0.5, split_seed=SEED,
    splits_static=None, splits_dynamic=None,
    out_dir=Path("outputs"), run_tag="run",
):
    """
    Grid search over the AUC vs Separation trade-off for the two MLP models.
    Separation is the AUC of the fairness curve over time. The coefficient is
    selected on VAL and reported on TEST, using the same splits as run_cv when
    `splits_static` / `splits_dynamic` are passed in.
    """
    np.random.seed(SEED)
    try:
        import torch
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)
    except Exception:
        pass

    if betas is None:
        betas = [0.0, 0.3, 0.5, 0.7, 1.0]
    if alphas is None:
        alphas = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2]
    if splits_static is None:
        splits_static = make_splits(y_static, grp_static, n_folds, val_size, split_seed)
    if splits_dynamic is None:
        splits_dynamic = make_splits(y_dynamic, grp_dynamic, n_folds, val_size, split_seed)

    # M_STATIC (coefficient = beta)
    print("=" * 60 + "\nGRID SEARCH — M_STATIC\n" + "=" * 60)
    df_s = run(X_static, y_static, grp_static, sens_static, splits_static, group_names,
               model_name="static", is_dynamic=False, eo_mode_d=eo_mode_d,
               grid_search=True, coefs=betas)
    df_s["model"] = "M_STATIC"
    for _, row in df_s.iterrows():
        print(f"  beta={row['coef']:.2f}  AUC(val)={row['auc_mean']:.4f}  "
              f"sep(val)={row['separation_auc']:.4f}  sep(test)={row['separation_auc_test']:.4f}")

    # M_DYNAMIC (coefficient = alpha)
    print("\n" + "=" * 60 + "\nGRID SEARCH — M_DYNAMIC\n" + "=" * 60)
    df_d = run(X_dynamic, y_dynamic, grp_dynamic, sens_dynamic, splits_dynamic, group_names,
               time_arr=lmk_vals, subj_ids=grp_dynamic, model_name="dynamic",
               n_bins=n_bins, collapse_pdh=True, is_dynamic=True,
               eo_mode_d=eo_mode_d, schedule_mode_d=schedule_mode_d,
               grid_search=True, coefs=alphas)
    df_d["model"] = "M_DYNAMIC"
    for _, row in df_d.iterrows():
        print(f"  alpha={row['coef']:.2f}  AUC(val)={row['auc_mean']:.4f}  "
              f"sep(val)={row['separation_auc']:.4f}  sep(test)={row['separation_auc_test']:.4f}")

    df_grid = pd.concat([df_s, df_d], ignore_index=True)
    out_dir = Path(out_dir)
    df_grid.to_csv(out_dir / f"grid_tradeoff_{run_tag}.csv", index=False)
    print(df_grid.to_string(index=False))
    print_best_points(df_grid, out_dir)
    return df_grid


# --------------------------------------------------------------------------- #
#  Best coefficient (ALWAYS selected on val) and reporting
# --------------------------------------------------------------------------- #
def _compute_best(df_grid):
    """
    M_STATIC:  maximizes a normalized AUC-vs-separation trade-off score.
    M_DYNAMIC: minimizes separation subject to AUC >= static baseline AUC (beta=0).
    Selection uses the VAL columns only.
    """
    static_baseline = df_grid[(df_grid["model"] == "M_STATIC") &
                              (df_grid["coef"] == 0.0)]["auc_mean"].values
    static_auc = float(static_baseline[0]) if len(static_baseline) > 0 else 0.0

    sub_s = df_grid[df_grid["model"] == "M_STATIC"]\
        .dropna(subset=["auc_mean", "separation_auc"]).reset_index(drop=True)

    best = {}
    trade_score = None
    if not sub_s.empty:
        auc_min, auc_max = sub_s["auc_mean"].min(), sub_s["auc_mean"].max()
        sep_min, sep_max = sub_s["separation_auc"].min(), sub_s["separation_auc"].max()

        def trade_score(auc, sep):
            an = (auc - auc_min) / (auc_max - auc_min + 1e-9)
            sn = (sep - sep_min) / (sep_max - sep_min + 1e-9)
            return an - sn

        scores = np.array([trade_score(a, s) for a, s in
                           zip(sub_s["auc_mean"], sub_s["separation_auc"])])
        best["M_STATIC"] = sub_s.iloc[np.argmax(scores)]

    sub_d = df_grid[df_grid["model"] == "M_DYNAMIC"]\
        .dropna(subset=["auc_mean", "separation_auc"]).reset_index(drop=True)
    if not sub_d.empty:
        feasible = sub_d[sub_d["auc_mean"] >= static_auc]
        if feasible.empty:
            feasible = sub_d  # fallback
        best["M_DYNAMIC"] = feasible.loc[feasible["separation_auc"].idxmin()]

    return best, trade_score


def print_best_points(df_grid, out_dir):
    best, _ = _compute_best(df_grid)
    print("\n=== BEST COEFFICIENT (selected on VAL, test also reported) ===")
    rows = []
    for model_name, b in best.items():
        rows.append({
            "model": model_name, "best_coef": b["coef"], "coef_name": b["coef_name"],
            "auc_val": round(b["auc_mean"], 4),
            "separation_auc_val": round(b["separation_auc"], 4),
            "auc_test": round(b.get("auc_mean_test", np.nan), 4),
            "separation_auc_test": round(b.get("separation_auc_test", np.nan), 4),
        })
        print(f"  {model_name:<12}  {b['coef_name']}={b['coef']:.2f}  "
              f"AUC(val)={b['auc_mean']:.4f}  sep(val)={b['separation_auc']:.4f}  "
              f"sep(test)={b.get('separation_auc_test', np.nan):.4f}")
    pd.DataFrame(rows).to_csv(Path(out_dir) / "grid_best_points.csv", index=False)


def build_summary_table(cv_results):
    """Stack the per-model summary dicts into one table."""
    rows = []
    for name, res in cv_results.items():
        row = res["summary"].copy()
        row["Model"] = name
        rows.append(row)
    cols = ["Model", "AUC_Mean", "AUC_SD", "Brier_Mean", "Brier_SD", "F1_Mean", "F1_SD"]
    df = pd.DataFrame(rows)
    return df[[c for c in cols if c in df.columns]]


def plot_tradeoff(df_grid, out_dir, run_tag="run"):
    """Plot AUC (val) and Separation AUC (val) against the fairness coefficient."""
    best, trade_score = _compute_best(df_grid)
    static_base = df_grid[(df_grid["model"] == "M_STATIC") &
                          (df_grid["coef"] == 0.0)]["auc_mean"].values
    static_base = float(static_base[0]) if len(static_base) > 0 else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("AUC and Separation AUC (val) as a function of the fairness coefficient",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax, (model_name, style) in zip(axes, MODEL_STYLES.items()):
        sub = df_grid[df_grid["model"] == model_name]\
            .dropna(subset=["auc_mean", "separation_auc"])\
            .sort_values("coef").reset_index(drop=True)
        if sub.empty:
            ax.set_title(f"{model_name} — no data")
            continue

        coefs = sub["coef"].to_numpy()
        aucs = sub["auc_mean"].to_numpy()
        seps = sub["separation_auc"].to_numpy()
        color, clabel = style["color"], style["coef_label"]

        # different best_idx criterion for static vs dynamic
        if model_name == "M_DYNAMIC":
            feas = aucs >= static_base
            best_idx = int(np.argmin(np.where(feas, seps, np.inf))) if feas.any() \
                else int(np.argmin(seps))
        else:
            best_idx = int(np.argmax([trade_score(a, s) for a, s in zip(aucs, seps)])) \
                if trade_score is not None else 0

        ax2 = ax.twinx()
        ax.plot(coefs, aucs, color=color, linewidth=2.2, marker=style["marker"],
                markersize=7, zorder=3)
        ax2.plot(coefs, seps, color=color, linewidth=2.2, linestyle="--",
                 marker=style["marker"], markersize=7, alpha=0.55, zorder=3)

        # draw the static AUC baseline used as constraint for the dynamic model
        if model_name == "M_DYNAMIC":
            ax.axhline(static_base, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)

        for vals, ax_ in [(aucs, ax), (seps, ax2)]:
            ax_.scatter([coefs[best_idx]], [vals[best_idx]], s=320, marker="*",
                        color="gold", edgecolors=color, linewidths=1.5, zorder=6)

        ax.set_xlabel(f"coefficient ({clabel})", fontsize=11)
        ax.set_ylabel("AUC val  (higher is better)", fontsize=10, color=color)
        ax2.set_ylabel("Separation AUC val  (lower is fairer)", fontsize=10, color=color)
        ax.tick_params(axis="y", labelcolor=color)
        ax2.tick_params(axis="y", labelcolor=color)
        ax.set_title(model_name, fontsize=12, fontweight="bold", color=color)
        ax.grid(alpha=0.2, linestyle="--")

        legend_handles = [
            Line2D([0], [0], color=color, linewidth=2, marker=style["marker"],
                   label="AUC val (solid)"),
            Line2D([0], [0], color=color, linewidth=2, linestyle="--",
                   marker=style["marker"], alpha=0.55, label="Separation AUC val (dashed)"),
            Line2D([0], [0], marker="*", color="gold", markersize=11,
                   markeredgecolor=color, linewidth=0, label="Best coefficient"),
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