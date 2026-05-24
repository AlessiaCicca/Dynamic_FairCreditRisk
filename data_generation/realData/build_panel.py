"""
Build the longitudinal panel from Freddie Mac performance data + matched CSV.

"""

import os
import gc
import glob
import shutil
import zipfile
import argparse

import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq


PERF_COLS_FULL = [
    "loan_sequence_number",
    "monthly_reporting_period",
    "current_upb",
    "current_loan_delinquency_status",
    "loan_age",
    "remaining_months_to_maturity",
    "defect_settlement_date",
    "modifications_flag",
    "zero_balance_code",
    "zero_balance_effective_date",
    "current_interest_rate",
    "current_deferred_upb",
    "due_date_last_paid_installment",
    "mi_recoveries",
    "net_sales_proceeds",
    "non_mi_recoveries",
    "expenses",
    "legal_costs",
    "maintenance_costs",
    "taxes_and_insurance",
    "miscellaneous_expenses",
    "actual_loss",
    "modification_cost",
    "step_modification_flag",
    "deferred_payment_plan",
    "estimated_ltv",
    "zero_balance_removal_upb",
    "delinquent_accrued_interest",
    "delinquency_due_to_disaster",
    "borrower_assistance_status",
    "current_month_modification_loss",
    "interest_bearing_upb",
]

# Columns to keep in output 
PERF_COLS_KEEP = [
    "loan_sequence_number",
    "monthly_reporting_period",
    "current_upb",
    "current_loan_delinquency_status",
    "loan_age",
    "remaining_months_to_maturity",
    "modifications_flag",
    "zero_balance_code",
    "zero_balance_effective_date",
    "current_interest_rate",
    "current_deferred_upb",
    "estimated_ltv",
    "delinquency_due_to_disaster",
    "borrower_assistance_status",
]


# Extract performance files 
def extract_performance_zips(year, freddie_dir, perf_local):
    zip_pattern_sub  = os.path.join(freddie_dir, f"historical_data_{year}", f"historical_data_{year}Q*.zip")
    zip_pattern_flat = os.path.join(freddie_dir, f"historical_data_{year}Q*.zip")
    
    zip_files = glob.glob(zip_pattern_sub) or glob.glob(zip_pattern_flat)

    if not zip_files:
        raise FileNotFoundError(f"No zip files found for {year}." )

    extracted = []
    
    # Iterates over zip files in alphabetical/chronological order (Q1→Q2→Q3→Q4)
    for zpath in sorted(zip_files):
        # Opens the zip in read "r" mode
        with zipfile.ZipFile(zpath, "r") as z:
            time_files = [f for f in z.namelist() if "time" in f.lower() and f.endswith(".txt")]
            if not time_files:
                continue
            for tf in time_files:
                dest = os.path.join(perf_local, os.path.basename(tf))
                # Opens the file inside the zip and writes "wb" it to disk 
                if not os.path.exists(dest):
                    with z.open(tf) as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                extracted.append(dest)

    if not extracted:
        raise RuntimeError(f"No TIME files extracted for {year}.")
    return extracted


# Reads Freddie Mac performance files, keeps only loans present in the matched dataset, and saves them to a Parquet file.

PARQUET_SCHEMA = pa.schema([(col, pa.string()) for col in PERF_COLS_KEEP])
#Parquet is a file format for saving data tables to disk, like CSV, but designed for large datasets.

def load_performance(perf_txt_files, loan_ids, parquet_path, chunk_size=200_000):
    writer     = None
    for f in perf_txt_files:
        fname = os.path.basename(f)
        with open(f, "r") as fh:
            n_cols_file = len(fh.readline().split("|"))
        for chunk in pd.read_csv(f, sep="|", header=None,
                                  names=PERF_COLS_FULL[:n_cols_file],
                                  dtype=str, low_memory=False,
                                  chunksize=chunk_size, on_bad_lines="skip"):
            # Keeps only rows belonging to loans present in the matched dataset
            filtered = chunk[chunk["loan_sequence_number"].isin(loan_ids)]
            if filtered.empty:
                continue
            # Selects only the useful columns and fills missing ones with NaN     
            filtered = filtered.reindex(columns=PERF_COLS_KEEP)
            # Converts the chunk to Parquet format 
            table = pa.Table.from_pandas(filtered, schema=PARQUET_SCHEMA,
                                          preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(parquet_path, PARQUET_SCHEMA)
            writer.write_table(table)
                                      
            del filtered, table
            gc.collect()
                                      
    if writer:
        writer.close()
        
    # Reads the entire Parquet file written incrementally back into a single dataframe and returns it
    perf = pd.read_parquet(parquet_path)
    print(f"Unique loans in perf: {perf['loan_sequence_number'].nunique():,}")
    return perf



def add_time_columns(perf):
    perf = perf.copy()
    
    perf["monthly_reporting_period"] = perf["monthly_reporting_period"].str.strip()
    perf["period_year"] = perf["monthly_reporting_period"].str[:4].astype(int)
    perf["period_month"] = perf["monthly_reporting_period"].str[4:6].astype(int)
    perf["quarter"] = ((perf["period_month"] - 1) // 3 + 1).astype(int)

    perf["period_quarter"] = ( perf["period_year"].astype(str) + "Q" + perf["quarter"].astype(str))

    for col in [
        "current_upb",  "loan_age",  "remaining_months_to_maturity",
        "current_interest_rate", "estimated_ltv", "current_deferred_upb",]:
        if col in perf.columns:
            perf[col] = pd.to_numeric(perf[col], errors="coerce")


    # FirstEventTime definition
    perf["_is_default"] = (
    perf["current_loan_delinquency_status"].str.strip()
    .isin({"", "0", "00"})
    .map({True: 0, False: 1}))

    first_event = (
        perf[perf["_is_default"] == 1]
        .groupby("loan_sequence_number")["loan_age"]
        .min()
        .rename("FirstEventTime"))

    perf = perf.merge(first_event, on="loan_sequence_number", how="left") 
    perf.drop(columns=["_is_default"], inplace=True)

    return perf


# Inner join performance (one row per loan × month) with matched (one row per loan, time-invariant origination + HMDA data).
def build_panel(perf, matched):
    panel = perf.merge(matched, on="loan_sequence_number", how="inner")
    gc.collect()
    dups = panel.duplicated(subset=["loan_sequence_number", "monthly_reporting_period"]).sum()
    if dups > 0:
        print(f"WARNING: {dups:,} duplicate rows on (loan, month).")
    return panel

# Order columns and save
def save_panel(panel, matched, output_path):
    id_cols   = ["loan_sequence_number", "period_quarter", "period_year",
                 "quarter", "monthly_reporting_period"]
    perf_cols = ["loan_age", "remaining_months_to_maturity",
                 "current_upb", "current_interest_rate", "estimated_ltv",
                 "current_deferred_upb", "current_loan_delinquency_status",
                 "modifications_flag", "zero_balance_code",
                 "zero_balance_effective_date",
                 "delinquency_due_to_disaster", "borrower_assistance_status"]
    origination_cols = [c for c in matched.columns
                        if c != "loan_sequence_number"
                        and c not in set(id_cols + perf_cols + target_cols)]
    
    ordered   = [c for group in [id_cols, perf_cols, target_cols, origination_cols]
                 for c in group if c in panel.columns]
    # Order remaining columns
    ordered  += [c for c in panel.columns if c not in set(ordered)]

    panel = panel[ordered]
    panel.sort_values(["loan_sequence_number", "monthly_reporting_period"], inplace=True)
    panel.reset_index(drop=True, inplace=True)
    panel.to_csv(output_path, index=False)


# MAIN RUN
def run_build_panel(year, drive_root):
    freddie_dir  = os.path.join(drive_root, "freddie")
    output_dir   = os.path.join(drive_root, "output")
    perf_local   = os.path.join(drive_root, "perf_local_tmp")
    matched_path = os.path.join(output_dir, f"matched_{year}.csv")
    output_path  = os.path.join(output_dir, f"panel_{year}.csv")

    # Creates output folders if they don't exist yet.
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(perf_local, exist_ok=True)


    # Load matched Freddie Mac + HMDA
    matched  = pd.read_csv(matched_path, dtype=str, low_memory=False)
    loan_ids = set(matched["loan_sequence_number"].dropna().unique())
   
    # Extract + load + prepare performance 
    perf_txt_files = extract_performance_zips(year, freddie_dir, perf_local)
    parquet_path = os.path.join(drive_root, "perf_tmp.parquet")
    perf = load_performance(perf_txt_files, loan_ids, parquet_path)
    perf = add_time_columns(perf)

    # Build panel
    panel = build_panel(perf, matched)
    del perf
    gc.collect()

    save_panel(panel, matched, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drive_root", required=True,
        help="Root directory (e.g. /content/drive/MyDrive/thesis_data)"
    )
    parser.add_argument(
        "--year", type=int, required=True,
        help="Year to process (e.g. 2022)"
    )
    args = parser.parse_args()

    run_build_panel(year=args.year, drive_root=args.drive_root)
