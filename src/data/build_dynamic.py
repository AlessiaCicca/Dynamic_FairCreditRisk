"""
Builds the landmark discrete-time survival dataset from the longitudinal panel:
-> n_bins rows per (subject, landmark)
-> covariates frozen at x(L)
-> Target event_bin = default within that bin.
"""

import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import SplineTransformer


def build_dynamic(df,static_cols,tvc_cols,cat_cols,landmarks,horizon,
    delta=6,id_col="ID",time_col="Time",first_event_col="FirstEventTime",sens_col="sens_loan",enc_cat=None,enc_lmk=None,):

    #  Δx = x(L) - x(L-delta)
    trend_base_cols = ["bd_pct", "current_upb", "estimated_ltv", "current_interest_rate"]
    trend_cols = []
    for col in trend_base_cols:
        if col not in df.columns:
            continue
        tname = f"{col}_trend{delta}"
        s = df.groupby(id_col)[col].transform(lambda x: x - x.shift(delta))
        lb, hb = s.quantile(0.01), s.quantile(0.99)
        df[tname] = s.clip(lb, hb).fillna(0.0)
        trend_cols.append(tname)

    tvc_cols = list(tvc_cols) + trend_cols
    
    # For each landmark L keeps only rows at time L, subjects still at risk and calculates if default occurs between L and L+horizon

    # Number of intervals
    n_bins = horizon // delta
    last_obs = df.groupby(id_col)[time_col].max()  
    lm_rows = []
    for L in landmarks:
        # Landamarking require covariate fixed at L: x(L)
        snap0 = df[df[time_col] == L].copy()      
        # Mantain only subjects at risk
        snap0 = snap0[snap0[first_event_col].isna() | (snap0[first_event_col] > L)].copy()
        snap0["last_obs"] = snap0[id_col].map(last_obs)

        # Loop on BIN
        for j in range(n_bins):
            # Bin -> (b0, b1]
            b0 = L + delta * j        
            b1 = L + delta * (j + 1)  
            fe = snap0[first_event_col]

            # Default in the bin
            event = fe.notna() & (fe > b0) & (fe <= b1)
            # At risk in the bin
            at_risk  = fe.isna()  | (fe > b0)      
            # Observed until the end of the bin
            observed = event | (snap0["last_obs"] >= b1) 

            # All bin should have the same covariate x(L)
            row = snap0[at_risk & observed].copy()

            # Target in the BIN: occurs + > bo + < b1
            row["event_bin"] = ( row[first_event_col].notna() & (row[first_event_col] > b0)  & (row[first_event_col] <= b1)).astype(np.int8)
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
    cats = enc_cat.transform(landmark_df[cat_cols])
    cat_feature_names = list(enc_cat.get_feature_names_out(cat_cols))

    # Temporal features: one-hot oflandmark and spline of bin_time
    all_bin_times = sorted({L + delta * j for L in landmarks for j in range(n_bins)})

    n_knots   = 4         
    spline_deg = 3         

    spline_tf = SplineTransformer( n_knots=n_knots, degree=spline_deg,include_bias=False, knots="quantile")
    spline_tf.fit(np.asarray(all_bin_times, dtype=np.float64).reshape(-1, 1))

    lmk_spl = spline_tf.transform(landmark_df[["bin_time"]].to_numpy(dtype=np.float64)).astype(np.float32)
    lmk_feature_names = [f"spl_{i}" for i in range(lmk_spl.shape[1])]

    # One-hot encoding of the landmark L itself
    if enc_lmk is None:
        enc_lmk = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        enc_lmk.fit(landmark_df[["landmark"]])
    lmk_oh = enc_lmk.transform(landmark_df[["landmark"]])
    lmk_oh_feature_names = list(enc_lmk.get_feature_names_out(["landmark"]))

    all_num_cols = static_cols + tvc_cols

    # Replaces missing values with the column median 
    medians = landmark_df[all_num_cols].median()
    num_static = landmark_df[static_cols].fillna(medians[static_cols])
    num_tvc = landmark_df[tvc_cols].fillna(medians[tvc_cols])
  
    num = np.hstack([num_static.to_numpy(dtype=np.float32),num_tvc.to_numpy(dtype=np.float32),])

    # Builds the final feature matrix by concatenating all parts
    # ...bd_pct_trend4,current_upb_trend4,estimated_ltv_trend4,current_interest_rate_trend4,occupancy_status_orig_I,occupancy_status_orig_P,occupancy_status_orig_S,
    #    loan_purpose_orig_C,loan_purpose_orig_N,loan_purpose_orig_P,spl_0,spl_1,spl_2,spl_3,spl_4,landmark_0,landmark_4,landmark_8,landmark_12,landmark_16,landmark_20,
    #    landmark_24,landmark_28,landmark_32,landmark_36,landmark_40,landmark_44,landmark_48,y,groups,sensitive,lmk_vals,bin_time_vals
   
    X = np.hstack([num, cats, lmk_spl, lmk_oh])

    # Extracts vectors needed for training

    y  = landmark_df["event_bin"].to_numpy(dtype=np.int8)  
    bin_time_vals = landmark_df["bin_time"].to_numpy()       
    groups = landmark_df[id_col].to_numpy()
    sensitive = landmark_df[sens_col].to_numpy(dtype=np.float64)
    lmk_vals = landmark_df["landmark"].to_numpy()
    

    # List of all column names
    feature_names = static_cols + tvc_cols + cat_feature_names + lmk_feature_names + lmk_oh_feature_names

    del cats, landmark_df
    gc.collect()

    return dict(X = X, y = y, groups = groups,
        sensitive = sensitive, lmk_vals = lmk_vals,
        bin_time_vals = bin_time_vals, enc_cat = enc_cat,
        enc_lmk = enc_lmk, medians = medians, feature_names = feature_names)
