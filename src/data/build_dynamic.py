"""
Builds the landmark discrete-time survival dataset from the longitudinal panel.
n_bins rows per (subject, landmark): covariates frozen at x(L), one row per
future bin (L+delta*j, L+delta*(j+1)]. Target event_bin = default within that bin.
Rows censored before the bin end are dropped

"""

import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import SplineTransformer


def build_dynamic(
    df,
    static_cols,
    tvc_cols,
    cat_cols,
    landmarks,
    horizon,
    delta=6,
    id_col="ID",
    time_col="Time",
    first_event_col="FirstEventTime",
    sens_col="sens_loan",
    enc_cat=None,
):

    #  Δx = x(L) - x(L-delta)
    trend_base_cols = ["bd_pct", "current_upb", "estimated_ltv", "current_interest_rate"]
    trend_cols = []
    for col in trend_base_cols:
        if col not in df.columns:
            continue
        tname = f"{col}_trend{delta}"
        s = df.groupby(id_col)[col].transform(lambda x: x - x.shift(delta))
        lo, hi = s.quantile(0.01), s.quantile(0.99)
        df[tname] = s.clip(lo, hi).fillna(0.0)
        trend_cols.append(tname)

    tvc_cols = list(tvc_cols) + trend_cols
    
    # For each landmark L:
    # - keeps only rows at time L (snapshot)
    # - keeps only subjects still at risk — those who have not yet experienced the event
    # - computes future_event = 1 if default occurs between L and L+horizon

    # Number of intervals
    n_bins = horizon // delta
    last_obs = df.groupby(id_col)[time_col].max()  
    lm_rows = []
    
    for L in landmarks:
        # Landamarking require covariate fixed at L: x(L)
        snap0 = df[df[time_col] == L].copy()      
        # Mantain only subjects at risk
        snap0 = snap0[snap0[first_event_col].isna() | (snap0[first_event_col] > L)].copy()
        snap0["_last_obs"] = snap0[id_col].map(last_obs)

        # Loop on BIN
        for j in range(n_bins):
            # Bin -> (b0, b1]
            b0 = L + delta * j        
            b1 = L + delta * (j + 1)  
            fe = snap0[first_event_col]

            # Default in the bin
            ev       = fe.notna() & (fe > b0) & (fe <= b1)
            # At risk in the bin
            at_risk  = fe.isna()  | (fe > b0)      
            # Observed until the end of the bin
            observed = ev | (snap0["_last_obs"] >= b1) 

            # All bin should have the same covariate x(L)
            row = snap0[at_risk & observed].copy()

            # Target
            row["event_bin"] = (
                row[first_event_col].notna()
                & (row[first_event_col] > b0)
                & (row[first_event_col] <= b1)
            ).astype(np.int8)
            row["landmark"] = np.int8(L)
            row["bin_time"] = np.int16(b0)
            lm_rows.append(row)

    # Final dataset
    landmark_df = pd.concat(lm_rows, ignore_index=True)
    del lm_rows

    # Categorical encoding
    if enc_cat is None:
        enc_cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        enc_cat.fit(landmark_df[cat_cols])
    cats              = enc_cat.transform(landmark_df[cat_cols])
    cat_feature_names = list(enc_cat.get_feature_names_out(cat_cols))

    # Temporal baseline hazard: one-hot of bin_time (loan age at the bin)
    all_bin_times = sorted({L + delta * j for L in landmarks for j in range(n_bins)})

    n_knots   = 4         
    spline_deg = 3         

    spline_tf = SplineTransformer(
        n_knots=n_knots, degree=spline_deg,
        include_bias=False,         
        knots="quantile",            
    )
    # fit sul range completo dei bin_time osservati
    spline_tf.fit(np.asarray(all_bin_times, dtype=np.float64).reshape(-1, 1))

    lmk_spl = spline_tf.transform(
        landmark_df[["bin_time"]].to_numpy(dtype=np.float64)
    ).astype(np.float32)
    lmk_feature_names = [f"spl_{i}" for i in range(lmk_spl.shape[1])]

    all_num_cols = static_cols + tvc_cols

    # Replaces missing values with the column median 
    medians      = landmark_df[all_num_cols].median()
    num = np.hstack([
        landmark_df[static_cols].fillna(medians[static_cols]).to_numpy(dtype=np.float32),
        landmark_df[tvc_cols].fillna(medians[tvc_cols]).to_numpy(dtype=np.float32),
    ])
    

    # Builds the final feature matrix by concatenating all parts
    X = np.hstack([num, cats, lmk_spl])

    # Extracts vectors needed for training

    y  = landmark_df["event_bin"].to_numpy(dtype=np.int8)  
    bin_time_vals = landmark_df["bin_time"].to_numpy()       
    groups    = landmark_df[id_col].to_numpy()
    sensitive = landmark_df[sens_col].to_numpy(dtype=np.float64)
    lmk_vals  = landmark_df["landmark"].to_numpy()
    

    # List of all column names
    feature_names = static_cols + tvc_cols + cat_feature_names + lmk_feature_names

    del cats, landmark_df
    gc.collect()

    return dict(
        X             = X,
        y             = y,
        groups        = groups,
        sensitive     = sensitive,
        lmk_vals      = lmk_vals,
        bin_time_vals = bin_time_vals, 
        enc_cat       = enc_cat,
        medians       = medians,
        feature_names = feature_names,
    )
