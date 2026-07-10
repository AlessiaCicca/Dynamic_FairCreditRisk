"""
Preprocessing pipeline for real data analysis.
Input:  panel_sampled.csv  
Output: panel_clean.csv    

Steps:
    1. Load raw panel (select relevant columns)
    2. Numeric conversions + fix errate notazioni
    3. Outlier capping
    4. Compute bd_pct (balance deviation)
    5. Compute trend features
    6. Compute FirstDefaultAge
    7. Encode demographics → binary (sex_bin, race_bin, age_bin)
    8. Propagate per-loan demographic columns
    9. Filter loans with too few observations
    10. Save panel_clean.csv
"""

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import STATIC_COLS, TVC_COLS, CAT_COLS



PIPELINE_COLS = [
    "loan_sequence_number",               # ID
    "loan_age",                           # Time
    "current_loan_delinquency_status",    
    "applicant_sex",                 
    "derived_race",                    
    "applicant_age",   
    "loan_amount"                
]

RAW_COLS = (PIPELINE_COLS + STATIC_COLS + 
    [c for c in TVC_COLS if c != "bd_pct"and c !="current_upb_delta" ] + 
    CAT_COLS)

NUMERIC_COLS = [
    "loan_age",
    "credit_score", "original_dti", "original_ltv",
     "interest_rate", "loan_term",
    "num_borrowers",
    "current_upb", "current_interest_rate", "estimated_ltv", "loan_amount"
]


MISSING_CODES = {
    "credit_score": [9999],
    "original_dti":    [999], 
     "original_ltv": [999],
     "num_borrowers": [99],
     "estimated_ltv":[999,998],
     "occupancy_status_orig": [9],
     "loan_purpose_orig": [9],

}

VALID_RANGES = {
    "credit_score":          [(300, 850)],   # Data Dictionary
    "original_dti":          [(1, 65)],      # Data Dictionary
    "original_ltv":          [(0, 998)],     # Data Dictionary
    "num_borrowers":         [(1, 10)],      # Data Dictionary
    "current_interest_rate": [(0, None)],    # Peng & Lessmann (2026)
    "estimated_ltv":         [(0, 500)],     # Peng & Lessmann (2026)
    "current_upb":           [(0, None)],    # Peng & Lessmann (2026)
}

VALID_VALUES = {
    "occupancy_status_orig":  ["P", "I", "S"],  
    "loan_purpose_orig":      ["P", "C", "N", "R"],  
}


# Returns 1 if loan is in default (delinquency status != 0), 0 otherwise
def _is_default(s):
    num = pd.to_numeric(s, errors="coerce")
    return (num.notna() & (num != 0)).astype(np.int8).values

# Computes the theoretical amortization schedule balance at age a. Used to compute bd_pct.
def _scheduled_balance(orig_upb, r, N, a):
    try:
        orig_upb = float(orig_upb); r = float(r)
        N = int(float(N));          a = float(a)
    except Exception:
        return np.nan
    if any(np.isnan(x) for x in [orig_upb, r, N, a]):
        return np.nan
    if r > 1:
        r /= 100.0
    if r < 0 or N <= 0 or N > 1000:
        return np.nan
    rm = r / 12.0
    if abs(rm) < 1e-10:
        return max(0.0, orig_upb - (orig_upb / N) * a)
    a = np.clip(a, 0, N)
    try:
        num = (1 + rm) ** N - (1 + rm) ** a
        den = (1 + rm) ** N - 1
        return max(0.0, orig_upb * num / den) if den != 0 else np.nan
    except OverflowError:
        return np.nan


# Load raw panel selecting only relevant columns.
def load_panel(path):
    df = pd.read_csv(path, usecols=RAW_COLS, low_memory=False)
    return df


# Replace Freddie Mac special missing codes with NaN
def replace_missing_codes(df):
    for col, codes in MISSING_CODES.items():
        if col in df.columns:
            df[col] = df[col].replace(codes, np.nan)
    return df

# Replace values outside valid domain with NaN
def replace_invalid_ranges(df):
    # Numerical Range
    for col, ranges in VALID_RANGES.items():
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if col not in df.columns:
            continue
        n_before   = df[col].notna().sum()
        valid_mask = pd.Series(False, index=df.index)
        for (lo, hi) in ranges:
                    mask = pd.Series(True, index=df.index)
                    if lo is not None:
                        mask &= df[col] >= lo
                    if hi is not None:
                        mask &= df[col] <= hi
                    valid_mask |= mask
        df[col] = df[col].where(valid_mask | df[col].isna(), other=np.nan)
        n_replaced = n_before - df[col].notna().sum()
      
    # Categorical Range
    for col, valid_vals in VALID_VALUES.items():
        if col not in df.columns:
            continue
        n_before = df[col].notna().sum()
        df[col]  = df[col].where(df[col].isin(valid_vals), other=np.nan)
        n_replaced = n_before - df[col].notna().sum()
    return df

# Numeric conversions 
def convert_numerics(df):
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    return df

# Encoding first_time_homebuyer
def encode_first_time_homebuyer(df):
    if "first_time_homebuyer" in df.columns:
        df["first_time_homebuyer"] = df["first_time_homebuyer"].map(
            {"Y": 1, "N": 0}
        )
    return df

# Outlier capping
def cap_outliers(df, lower=0.01, upper=0.99):
    # Escludi colonne già gestite da replace_invalid_ranges
    already_bounded = set(VALID_RANGES.keys()) | set(VALID_VALUES.keys())
    cols_to_cap = [c for c in NUMERIC_COLS if c not in already_bounded]
    
    for col in cols_to_cap:
        if col not in df.columns:
            continue
        p_low  = df[col].quantile(lower)
        p_high = df[col].quantile(upper)
        df[col] = df[col].clip(p_low, p_high)
    return df


#  Balance deviation from theoretical amortization schedule.
#   bd_pct = (current_upb - scheduled_balance) / scheduled_balance
# Following Peng & Lessmann (2026) — "Incorporating data drift into survival analysis for credit risk"
def compute_bd_pct(df):
    print("  Computing bd_pct...")
    df["b_sched"] = df.apply(
        lambda r: _scheduled_balance(
            r["loan_amount"], r["interest_rate"],
            r["loan_term"],   r["loan_age"]
        ), axis=1
    ).astype("float32")
    df["bd_pct"] = (
        (df["current_upb"] - df["b_sched"]) / df["b_sched"]
    ).replace([np.inf, -np.inf], np.nan).clip(-2, 2).astype("float32")
    df.drop(columns=["b_sched"], inplace=True)
    return df


# Trend features 
def compute_trends(df):
    df = df.sort_values(["loan_sequence_number", "loan_age"])
    for col in ["bd_pct", "estimated_ltv", "current_upb"]:
        if col not in df.columns:
            continue
        df[f"{col}_trend"] = (
            df.groupby("loan_sequence_number")[col]
            .transform(lambda x: x - x.shift(2))
        ).clip(-2, 2).fillna(0)
    return df

# First-order difference of current_upb.
# Captures short-term changes in repayment behavior.
# Reference: KAN paper (2024), ResE-BiLSTM (2025)
def compute_upb_delta(df):
    df = df.sort_values(["loan_sequence_number", "loan_age"])
    df["current_upb_delta"] = df.groupby("loan_sequence_number")["current_upb"].diff().fillna(0)
    return df



# FirstDefaultAge: Compute the loan age at first default event.
def compute_first_default_age(df):
    is_def = _is_default(df["current_loan_delinquency_status"])
    df["_is_default"] = is_def
    fd_age = (
        df[df["_is_default"] == 1]
        .groupby("loan_sequence_number")["loan_age"].min()
        .rename("FirstDefaultAge")
    )
    df = df.merge(fd_age, on="loan_sequence_number", how="left")
    df.drop(columns=["_is_default", "current_loan_delinquency_status"],
            inplace=True)

    n_loans = df["loan_sequence_number"].nunique()
    n_def   = df.groupby("loan_sequence_number")["FirstDefaultAge"].first().notna().sum()
    print(f"  Loans: {n_loans:,}  |  Defaulters: {n_def:,} ({n_def/n_loans:.1%})")
    return df


# Demographics → binary
def encode_demographics(df):
    if "applicant_sex" in df.columns:
        df["sex_bin"] = pd.to_numeric(df["applicant_sex"], errors="coerce").map(
            {1.0: 0, 2.0: 1}  # 1=Male→0, 2=Female→1, 3/6→NaN
        )
    # Race
    def race_map(x):
        if not isinstance(x, str): return np.nan
        x = x.strip().lower()
        if x in ["white", "asian","joint"]: return 0
        if x in ["black or african american",
                  "american indian or alaska native",
                  "native hawaiian or other pacific islander",
                  "2 or more minority races", "other"]: return 1
        return np.nan

    df["race_bin"] = df["derived_race"].apply(race_map)

    # Age
    df["age_bin"] = df["applicant_age"].map(
        {"<25": 1, "25-34": 0, "35-44": 0, "45-54": 0,
         "55-64": 0, "65-74": 0, ">74": 0}
    )

    return df


# Propagate per-loan demographics
def propagate_demographics(df):
    for col, name in [
        ("sex_bin",  "sex_bin_loan"),
        ("race_bin", "race_bin_loan"),
        ("age_bin",  "age_bin_loan"),
    ]:
        if col not in df.columns:
            continue
        per_loan = df.groupby("loan_sequence_number")[col].first().rename(name)
        df = df.merge(per_loan, on="loan_sequence_number", how="left")
    cols_to_drop = ["derived_sex", "derived_race", "applicant_age",
                    "sex_bin", "race_bin", "age_bin"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    return df



#  Categorical encoding
def encode_categoricals(df):
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df

# MAIN
def preprocess(path_in, path_out):
    df = load_panel(path_in)
    df = replace_missing_codes(df)
    df = replace_invalid_ranges(df)
    df = convert_numerics(df)
    df = cap_outliers(df)
    df = compute_bd_pct(df)
    df = compute_upb_delta(df)
    df = compute_trends(df)
    df = compute_first_default_age(df)
    df = encode_demographics(df)
    df = propagate_demographics(df)
    df = encode_categoricals(df)



    print(f"\nSaving: {path_out}")
    df.to_csv(path_out, index=False)
    size_gb = os.path.getsize(path_out) / (1024**3)
    print(f"  File size: {size_gb:.2f} GB")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path_in", required=True)

    parser.add_argument("--path_out", required=True)

    args = parser.parse_args()

    preprocess(
        path_in  = args.path_in,
        path_out = args.path_out)
