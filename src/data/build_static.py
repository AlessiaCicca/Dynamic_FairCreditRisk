"""
Builds the static dataset (t=0) from the raw longitudinal panel.
One row per subject, features from first observation only.
Target: default within HORIZON (12) months from origination.
"""

import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def build_static(
    df,
    static_cols,
    cat_cols,
    horizon,
    id_col="ID",
    time_col="Time",
    first_event_col="FirstEventTime",
    sens_col="sens_loan",
    enc_cat=None,
):

    # Take first observation per subject 
    static_df = ( df.sort_values(time_col).groupby(id_col).first().reset_index())

    # Target
    static_df["target_static"] = (static_df[first_event_col].notna() & (static_df[first_event_col] <= horizon)).astype(np.int8)

    # Categorical encoding
    if enc_cat is None:
        enc_cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        enc_cat.fit(static_df[cat_cols])
    cats = enc_cat.transform(static_df[cat_cols])
    cat_feature_names = list(enc_cat.get_feature_names_out(cat_cols))

    # Replaces missing values with the column median 
    medians = static_df[static_cols].median()
    num     = static_df[static_cols].fillna(medians).to_numpy(dtype=np.float32)

    # Builds the final feature matrix by concatenating numerical and categorical variables
    X = np.hstack([num, cats])

    # Extracts vectors needed for training
    y         = static_df["target_static"].to_numpy(dtype=np.int8)
    groups    = static_df[id_col].to_numpy()
    sensitive = static_df[sens_col].to_numpy()

    # List of all column names
    feature_names = static_cols + cat_feature_names

    del cats
    gc.collect()

    return dict(
        X             = X,
        y             = y,
        groups        = groups,
        sensitive     = sensitive,
        enc_cat       = enc_cat,
        medians       = medians,
        feature_names = feature_names,
    )
