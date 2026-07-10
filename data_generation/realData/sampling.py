import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import  chi2_contingency
import argparse

import gc
import os
import sys
sys.path.append("/content/Dynamic_FairCreditRisk")

from config import (SAMPLING, RACE_MINORITY, TVC_COLS,STATIC_COLS)



def unique_panel(base_path,output_path):
  years = range(2018, 2025)
  all_cols = []
  for year in years:
      path = base_path + f"panel_{year}.csv"
      if os.path.exists(path):
          cols = pd.read_csv(path, nrows=0).columns.tolist()
          all_cols.append(set(cols))
  common_cols = set.intersection(*all_cols)
  print(f"Common columns: {len(common_cols)}")
  # Consider the first year as reference: mantain all the common columns with 2018's order
  ref_cols = pd.read_csv(base_path + "panel_2018.csv", nrows=0).columns.tolist()
  final_cols = [c for c in ref_cols if c in common_cols]

  # Create the unique file
  local_path = "/content/panel_full_tmp.csv"
  if os.path.exists(output_path):
    os.remove(output_path)
           
  first_write = True
  for year in years:
      path_year = base_path + f"panel_{year}.csv"

      if not os.path.exists(path_year): 
        print(f"{path_year} Missing ")
        continue

      print(f"Processing {year}...")

      for chunk in pd.read_csv( path_year, usecols=final_cols, chunksize=500_000, low_memory=False):
          # Force order
          chunk = chunk[final_cols]
          chunk.to_csv(local_path, mode="w" if first_write else "a",header=first_write,index=False)
          first_write = False
          del chunk
          gc.collect()


  import shutil
  shutil.copy(local_path, output_path)
  os.remove(local_path)

  # Check correct size
  size_gb = os.path.getsize(output_path) / (1024**3)
  print(f"File size: {size_gb:.2f} GB")
  print(f"File saved in: {output_path}")


def find_discriminated_loans(path, race_minority, chunksize=500_000):
    disc_loan_ids = set()
    for chunk in pd.read_csv(path, usecols=usecols_demo, dtype=str, chunksize=500_000):
        mask = (
            chunk["derived_race"].str.lower().str.strip().isin(race_minority)
            | chunk["applicant_age"].str.lower().str.strip().eq("<25")
            | chunk["derived_sex"].str.lower().str.strip().eq("female")
        )
        disc_loan_ids.update(chunk.loc[mask, "loan_sequence_number"].unique())
        del chunk
        gc.collect()
    return disc_loan_ids

# Let us to obtain the dataset in a form useful for sampling:
# loan_id | credit_score | upb    | ever_default | has_tvc_var | is_discriminated
def aggregate_loans(path, usecols_feat, tvc_cols, feature_cols,
                    disc_loan_ids, chunksize=500_000):
    loan_data = {}  
    
    for chunk in pd.read_csv(path, usecols=usecols_feat, chunksize=500_000, low_memory=False):
        chunk["current_loan_delinquency_status"] = pd.to_numeric(
            chunk["current_loan_delinquency_status"], errors="coerce"
        ).fillna(0)
        
        for col in tvc_cols:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        
        chunk["loan_sequence_number"] = chunk["loan_sequence_number"].astype(str)
        
        for lid, grp in chunk.groupby("loan_sequence_number"):
            if lid not in loan_data:
                loan_data[lid] = {
                    "is_disc":      lid in disc_loan_ids,
                    "ever_default": 0,
                    "first_row":    grp.iloc[0].to_dict(),
                    "upb_vals":     [],
                    "rate_vals":    [],
                    "ltv_vals":     [],
                }
            if grp["current_loan_delinquency_status"].max() > 0:
                loan_data[lid]["ever_default"] = 1
            loan_data[lid]["upb_vals"].extend(grp["current_upb"].dropna().tolist())
            loan_data[lid]["rate_vals"].extend(grp["current_interest_rate"].dropna().tolist())
            loan_data[lid]["ltv_vals"].extend(grp["estimated_ltv"].dropna().tolist())
        
        del chunk
        gc.collect()
    

    rows = []
    for lid, d in loan_data.items():
        rows.append({
            "loan_sequence_number": lid,
            "is_discriminated":     int(d["is_disc"]),
            "ever_default":         d["ever_default"],
            "has_tvc_var":          int(
                np.std(d["upb_vals"])  > 0 or
                np.std(d["rate_vals"]) > 0 or
                np.std(d["ltv_vals"])  > 0
            ),
            "derived_race":   str(d["first_row"].get("derived_race",  "nan")).lower().strip(),
            "derived_sex":    str(d["first_row"].get("derived_sex",   "nan")).lower().strip(),
            "applicant_age":  str(d["first_row"].get("applicant_age", "nan")).lower().strip(),
            **{k: v for k, v in d["first_row"].items() if k in feature_cols}
        })

    del loan_data
    loan_stats = pd.DataFrame(rows)
    gc.collect()

    print(f"\nTot Loan:        {len(loan_stats):,}")
    print(f"  Default rate:     {loan_stats['ever_default'].mean():.4f}")
    print(f"  Discriminated:     {loan_stats['is_discriminated'].mean():.2%}")
    print(f" TVC variability :    {loan_stats['has_tvc_var'].mean():.2%}")
    return loan_stats 


def weighted_sample(loan_stats, config):
    w_base    = config["w_base"]
    w_default = config["w_default"]
    w_disc    = config["w_disc"]
    w_tvc     = config["w_tvc"]
    n         = config["target_loans"]
    seed      = config["random_seed"]

    df = loan_stats.copy()
    # Initialize all weight equal to w_base=1.0
    df["weight"] = w_base
    df.loc[df["ever_default"]     == 1, "weight"] *= w_default
    df.loc[df["is_discriminated"] == 1, "weight"] *= w_disc
    df.loc[df["has_tvc_var"]      == 1, "weight"] *= w_tvc
  
    # Normalization from 0 to 1 (from weight to probability)
    df["weight"] = df["weight"] / df["weight"].sum()

    print(f"\nSampling...")
    df_sampled = df.sample(n=n, weights="weight", random_state=seed, replace=False)

    return df_sampled.drop(columns=["weight"])


# Re-read panel and keep only sampled loan IDs.
def filter_panel(path, sampled_ids, out_path, chunksize=500_000):
    first_write = True
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        chunk["loan_sequence_number"] = chunk["loan_sequence_number"].astype(str)
        filtered = chunk[chunk["loan_sequence_number"].isin(sampled_ids)]
        
        if len(filtered) > 0:
            filtered.to_csv(out_path,mode="w" if first_write else "a",
                header=first_write,index=False)
            first_write = False  
        del chunk
        gc.collect()
    size_gb = os.path.getsize(out_path) / (1024**3)
    print(f"  File size:         {size_gb:.2f} GB")
    print(f"  Saved in: {out_path}")
    return 



def compute_psi(expected, actual, n_bins=10):
    expected = np.array(expected, dtype=float)
    actual   = np.array(actual,   dtype=float)
    breakpoints = np.nanpercentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct   = np.histogram(actual,   bins=breakpoints)[0] / len(actual)
    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct   = np.where(actual_pct   == 0, 1e-6, actual_pct)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))

def psi_label(psi):
    if psi < 0.1:  return "OK"
    if psi < 0.2:  return "!!!"
    return "X"

def cramers_v(col, df_full, df_samp):
    cats = pd.Series(
        list(df_full[col].astype(str).unique()) +
        list(df_samp[col].astype(str).unique())
    ).unique()
    
    # Normalized proportion to avoid bias
    full_prop = df_full[col].astype(str).value_counts(normalize=True).reindex(cats, fill_value=0)
    samp_prop = df_samp[col].astype(str).value_counts(normalize=True).reindex(cats, fill_value=0)
    
    n = min(len(df_full), len(df_samp))
    full_counts = (full_prop * n).round().astype(int)
    samp_counts = (samp_prop * n).round().astype(int)
    
    contingency = np.array([full_counts.values, samp_counts.values])
    contingency = contingency[:, contingency.sum(axis=0) > 0]
    
    if contingency.shape[1] < 2:
        return np.nan
    
    chi2, _, _, _ = chi2_contingency(contingency)
    n_total = contingency.sum()
    k = min(contingency.shape) - 1
    return float(np.sqrt(chi2 / (n_total * k)) if k > 0 else np.nan)


def cramers_label(v):
    if v < 0.1:  return "OK"
    if v < 0.3:  return "!"
    if v < 0.5:  return "!!"
    return "X"


def evaluate_sampling(df_full_agg, df_sampled, feature_cols, race_minority):
    demo_cols    = ["derived_race", "derived_sex", "applicant_age"]

  
    print("\n=== PSI — Continuous Features ===")
    print(f"{'Feature':<25} {'PSI':>8} {'Result'}")
    print("-" * 50)
    for col in feature_cols:
        full_vals    = pd.to_numeric(df_full_agg[col], errors="coerce").dropna().values
        sampled_vals = pd.to_numeric(df_sampled[col],  errors="coerce").dropna().values
        psi = compute_psi(full_vals, sampled_vals)
        print(f"{col:<25} {psi:>8.4f}  {psi_label(psi)}")
        

    print("\n=== PSI — Demographic Features ===")
    print(f"{'Variable':<25} {'PSI':>8} {'Result'}")
    print("-" * 50)
    for col in demo_cols:
        if col not in df_full_agg.columns:
            continue
        cats     = df_full_agg[col].astype(str).value_counts().index.tolist()
        psi_vals = []
        for cat in cats:
            exp = (df_full_agg[col].astype(str) == cat).mean()
            act = (df_sampled[col].astype(str)  == cat).mean() if col in df_sampled.columns else 0
            exp = max(exp, 1e-6)
            act = max(act, 1e-6)
            psi_vals.append((act - exp) * np.log(act / exp))
        print(f"{col:<25} {sum(psi_vals):>8.4f}  {psi_label(sum(psi_vals))}")
  
    print("\n=== CRAMÉR'S V — Demographic Features ===")
    print(f"{'Variable':<25} {'Cramér V':>10} {'Result'}")
    print("-" * 50)
    for col in demo_cols:
        if col in df_full_agg.columns and col in df_sampled.columns:
            v = cramers_v(col, df_full_agg, df_sampled)
            print(f"{col:<25} {v:>10.4f}  {cramers_label(v)}")


    print("\n=== DEMOGRAPHIC PROPORTIONS: FULL vs SAMPLED ===\n")
    for col in demo_cols:
        if col not in df_full_agg.columns or col not in df_sampled.columns:
            continue
        print(f"{'─'*70}")
        print(f"{col.upper()}")
        print(f"{'Category':<45} {'Full':>8} {'Sampled':>10} {'Ratio':>8}")
        print(f"{'─'*70}")
        cats = sorted(
            pd.Series(list(df_full_agg[col].astype(str).unique()) +
                      list(df_sampled[col].astype(str).unique())).unique(),
            key=lambda x: (df_full_agg[col].astype(str)==x).mean(),
            reverse=True
        )
        for cat in cats:
            before = (df_full_agg[col].astype(str) == cat).mean()
            after  = (df_sampled[col].astype(str)  == cat).mean()
            if before < 0.001 and after < 0.001:
                continue
            is_disc = ""
            if col == "derived_race"  and cat in race_minority: is_disc = " ← disc"
            elif col == "derived_sex" and cat == "female":       is_disc = " ← disc"
            elif col == "applicant_age" and cat == "<25":        is_disc = " ← disc"
            print(f"{cat:<45} {before:>8.2%} {after:>10.2%} {after/(before+1e-9):>8.2f}x{is_disc}")
        print()

    print(f"\n=== DEFAULT RATE BY GROUP ===")
    print(f"{'Group':<30} {'Full':>8} {'Sampled':>10} {'Ratio':>8}")
    print("-" * 58)
    groups = {
        "Discriminated":     (df_full_agg["is_discriminated"]==1, df_sampled["is_discriminated"]==1),
        "Non discriminated": (df_full_agg["is_discriminated"]==0, df_sampled["is_discriminated"]==0),
        "Total":             (np.ones(len(df_full_agg), dtype=bool), np.ones(len(df_sampled), dtype=bool)),
    }
    for grp_name, (mask_full, mask_samp) in groups.items():
        dr_full = df_full_agg.loc[mask_full, "ever_default"].mean()
        dr_samp = df_sampled.loc[mask_samp, "ever_default"].mean()
        print(f"{grp_name:<30} {dr_full:>8.2%} {dr_samp:>10.2%} {dr_samp/(dr_full+1e-9):>8.2f}x")



RANDOM_SEED  = SAMPLING["random_seed"]
TARGET_LOANS = SAMPLING["target_loans"]
W_DEFAULT    = SAMPLING["w_default"]
W_DISC       = SAMPLING["w_disc"]
W_TVC        = SAMPLING["w_tvc"]
W_BASE       = SAMPLING["w_base"]



tvc_cols = [col for col in TVC_COLS if col not in ("bd_pct", "current_upb_delta")]
race_minority = RACE_MINORITY

usecols_demo = ["loan_sequence_number", "derived_race", "derived_sex", "applicant_age"]

usecols_feat = (
    ["loan_sequence_number", "current_loan_delinquency_status"] +
    usecols_demo[1:] +  
    STATIC_COLS +
    tvc_cols
)

feature_cols = STATIC_COLS + tvc_cols

# MAIN
def run_sampling(path, out_full, out_sampled, config):
    print(f"\n{'='*60}")
    print(f"  Target loans:  {config['target_loans']:,}")
    print(f"  Weights:       default={config['w_default']} "
          f"disc={config['w_disc']} tvc={config['w_tvc']}")
    print(f"{'='*60}\n")

    unique_panel(path, out_full)
    
    disc_loan_ids = find_discriminated_loans(out_full, RACE_MINORITY)
    loan_stats = aggregate_loans(out_full, usecols_feat, tvc_cols, feature_cols, disc_loan_ids)

    df_sampled = weighted_sample(loan_stats, config)

    sampled_ids = set(df_sampled["loan_sequence_number"].astype(str))
    filter_panel(out_full, sampled_ids,out_sampled )


    evaluate_sampling(loan_stats, df_sampled, feature_cols, race_minority)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", required=True,
        help="Path to panel_years.csv"
    )
    parser.add_argument(
        "--out_full", default="/content/drive/MyDrive/thesis_data/output/panel_full.csv")
    parser.add_argument(
        "--out_sampled", default="/content/drive/MyDrive/thesis_data/output/panel_sampled.csv")
    args = parser.parse_args()

    config = {
        "target_loans": SAMPLING["target_loans"],
        "w_default":    SAMPLING["w_default"],
        "w_disc":       SAMPLING["w_disc"],
        "w_tvc":        SAMPLING["w_tvc"],
        "w_base":       SAMPLING["w_base"],
        "random_seed":  SAMPLING["random_seed"],
    }

    run_sampling(path=args.path, out_full=args.out_full, out_sampled=args.out_sampled, config=config)
  