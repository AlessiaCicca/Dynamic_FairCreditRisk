"""
Builds the landmark (dynamic) dataset from the raw longitudinal panel.
One row per subject × landmark, features from the snapshot at that landmark.
Target: default in (landmark, landmark + horizon].

"""

import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def build_dynamic(
    df,
    static_cols,
    tvc_cols,
    cat_cols,
    landmarks,
    horizon,
    id_col="ID",
    time_col="Time",
    first_event_col="FirstEventTime",
    sens_col="sens_loan",
    enc_cat=None,
):

    lm_rows = []
    
    # For each landmark L:
    # - keeps only rows at time L (snapshot)
    # - keeps only subjects still at risk — those who have not yet experienced the event
    # - computes future_event = 1 if default occurs between L and L+horizon
    
    for L in landmarks:
        snap = df[df[time_col] == L].copy()
        snap = snap[snap[first_event_col].isna() | (snap[first_event_col] > L)].copy()
        snap["future_event"] = (
            snap[first_event_col].notna() &
            (snap[first_event_col] > L) &
            (snap[first_event_col] <= L + horizon)
        ).astype(np.int8)
        # Create landamrk columns
        snap["landmark"] = np.int8(L)
        lm_rows.append(snap)

    landmark_df = pd.concat(lm_rows, ignore_index=True)
    del lm_rows


    # Categorical encoding of landmark -> convert the landamrk column in one-hot
    if enc_cat is None:
        enc_cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        enc_cat.fit(landmark_df[cat_cols])
    enc_lmk = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    enc_lmk.fit(np.array(landmarks).reshape(-1, 1))
    cats              = enc_cat.transform(landmark_df[cat_cols])
    lmk_oh            = enc_lmk.transform(landmark_df[["landmark"]])
    cat_feature_names = list(enc_cat.get_feature_names_out(cat_cols))
    lmk_feature_names = [f"lmk_{L}" for L in landmarks]
    # → ["lmk_3", "lmk_6", "lmk_9", "lmk_12"]

    all_num_cols = static_cols + tvc_cols

    # Replaces missing values with the column median 
    medians      = landmark_df[all_num_cols].median()
    num = np.hstack([
        landmark_df[static_cols].fillna(medians[static_cols]).to_numpy(dtype=np.float32),
        landmark_df[tvc_cols].fillna(medians[tvc_cols]).to_numpy(dtype=np.float32),
    ])

    # Builds the final feature matrix by concatenating all parts
    X = np.hstack([num, cats, lmk_oh])

    # Extracts vectors needed for training
    y         = landmark_df["future_event"].to_numpy(dtype=np.int8)
    groups    = landmark_df[id_col].to_numpy()
    sensitive = landmark_df[sens_col].to_numpy()
    lmk_vals  = landmark_df["landmark"].to_numpy()

    # List of all column names
    feature_names = static_cols + tvc_cols + cat_feature_names + lmk_feature_names

    del cats, lmk_oh, landmark_df
    gc.collect()

    return dict(
        X             = X,
        y             = y,
        groups        = groups,
        sensitive     = sensitive,
        lmk_vals      = lmk_vals,
        enc_cat       = enc_cat,
        enc_lmk       = enc_lmk,
        medians       = medians,
        feature_names = feature_names,
    )
