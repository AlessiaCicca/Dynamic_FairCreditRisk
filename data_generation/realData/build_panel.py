"""
data_generation/fnma/build_panel.py

Build the longitudinal panel from Freddie Mac performance data + matched CSV.
Step 2 of the FNMA pipeline (run after match_hmda.py).

Usage:
    python build_panel.py --drive_root /path/to/thesis_data --year 2022

Input:
    drive_root/output/matched_{YEAR}.csv   <- from match_hmda.py
    drive_root/freddie/historical_data_{YEAR}/historical_data_{YEAR}Q*.zip

Output:
    drive_root/output/panel_{YEAR}.csv
    One row per (loan_sequence_number, monthly_reporting_period).

Column groups in output:
    [A] Identifiers    : loan_sequence_number, period_quarter, ...
    [B] Performance    : current_upb, loan_age, estimated_ltv, ...
    [C] Survival target: default, prepayment
    [D] Origination + HMDA (time-invariant, replicated from matched)
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

# ── Performance column layout (Freddie Mac manual) ────────────────────────────

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

# Columns to keep in output (drop financial recovery details)
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


# ── Step 1: extract performance files ────────────────────────────────────────

def extract_performance_zips(year: int, freddie_dir: str,
                              perf_local: str) -> list:
    """
    Extract TIME (performance) .txt files from Freddie Mac quarterly zips.
    Returns sorted list of extracted .txt paths.
    """
    zip_pattern_sub  = os.path.join(
        freddie_dir, f"historical_data_{year}",
        f"historical_data_{year}Q*.zip"
    )
    zip_pattern_flat = os.path.join(
        freddie_dir, f"historical_data_{year}Q*.zip"
    )
    zip_files = glob.glob(zip_pattern_sub) or glob.glob(zip_pattern_flat)

    if not zip_files:
        raise FileNotFoundError(
            f"\nNo zip files found for {year}.\n"
            f"Patterns checked:\n  {zip_pattern_sub}\n  {zip_pattern_flat}"
        )

    print(f"Found {len(zip_files)} zip files for {year}:")
    extracted = []

    for zpath in sorted(zip_files):
        zname = os.path.basename(zpath)
        with zipfile.ZipFile(zpath, "r") as z:
            contents   = z.namelist()
            time_files = [f for f in contents
                          if "time" in f.lower() and f.endswith(".txt")]

            if not time_files:
                print(f"  WARNING: no TIME file in {zname}. Contents: {contents}")
                continue

            for tf in time_files:
                out_name = os.path.basename(tf)
                dest     = os.path.join(perf_local, out_name)

                if os.path.exists(dest):
                    size_mb = os.path.getsize(dest) / 1e6
                    print(f"  {zname} -> {out_name} already extracted "
                          f"({size_mb:.0f} MB), skip")
                else:
                    print(f"  Extracting {out_name} from {zname}...",
                          end=" ", flush=True)
                    with z.open(tf) as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    size_mb = os.path.getsize(dest) / 1e6
                    print(f"OK ({size_mb:.0f} MB)")

                extracted.append(dest)

    if not extracted:
        raise RuntimeError(
            "No TIME files extracted. Check that zips contain files "
            "with 'time' in the name (e.g. time_2022Q1.txt)."
        )
    return sorted(extracted)


# ── Step 2: load and filter performance data ──────────────────────────────────

PARQUET_SCHEMA = pa.schema([(col, pa.string()) for col in PERF_COLS_KEEP])

def load_performance(perf_txt_files, loan_ids, parquet_path, chunk_size=200_000):
    writer     = None
    total_rows = 0

    for f in perf_txt_files:
        fname = os.path.basename(f)
        with open(f, "r") as fh:
            n_cols_file = len(fh.readline().split("|"))

        rows_this_file = 0
        for chunk in pd.read_csv(f, sep="|", header=None,
                                  names=PERF_COLS_FULL[:n_cols_file],
                                  dtype=str, low_memory=False,
                                  chunksize=chunk_size, on_bad_lines="skip"):
            filtered = chunk[chunk["loan_sequence_number"].isin(loan_ids)]
            if filtered.empty:
                continue
            cols = [c for c in PERF_COLS_KEEP if c in filtered.columns]
            filtered = filtered[cols].copy()
            for col in PERF_COLS_KEEP:
                if col not in filtered.columns:
                    filtered[col] = None
            filtered = filtered[PERF_COLS_KEEP]

            table = pa.Table.from_pandas(filtered, schema=PARQUET_SCHEMA,
                                          preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(parquet_path, PARQUET_SCHEMA)
            writer.write_table(table)

            rows_this_file += len(filtered)
            total_rows     += len(filtered)
            del filtered, table
            gc.collect()

        print(f"  {fname}: {rows_this_file:,} rows")

    if writer:
        writer.close()

    print(f"\nTotal performance rows: {total_rows:,}")
    perf = pd.read_parquet(parquet_path)
    print(f"Unique loans in perf: {perf['loan_sequence_number'].nunique():,}")
    return perf


# ── Step 3: add time columns ──────────────────────────────────────────────────

import numpy as np
import pandas as pd

def add_time_columns(perf: pd.DataFrame) -> pd.DataFrame:
    """Parse YYYYMM period and add year, month, quarter columns."""

    perf = perf.copy()

    perf["monthly_reporting_period"] = perf["monthly_reporting_period"].str.strip()
    perf["period_year"] = perf["monthly_reporting_period"].str[:4].astype(int)
    perf["period_month"] = perf["monthly_reporting_period"].str[4:6].astype(int)
    perf["quarter"] = ((perf["period_month"] - 1) // 3 + 1).astype(int)

    perf["period_quarter"] = (
        perf["period_year"].astype(str)
        + "Q"
        + perf["quarter"].astype(str)
    )

    # Numeric conversions
    for col in [
        "current_upb",
        "loan_age",
        "remaining_months_to_maturity",
        "current_interest_rate",
        "estimated_ltv",
        "current_deferred_upb",
    ]:
        if col in perf.columns:
            perf[col] = pd.to_numeric(perf[col], errors="coerce")

    # =========================
    # FirstEventTime (DEFAULT)
    # =========================

    delq = pd.to_numeric(perf["current_loan_delinquency_status"], errors="coerce")
    delq_str = perf["current_loan_delinquency_status"].astype(str).str.strip()

    is_default = np.where(
        delq.notna(),
        (delq != 0).astype(int),
        (~delq_str.isin({"", "0", "00"})).astype(int),
    )

    perf["_is_default"] = is_default

    first_event = (
        perf[perf["_is_default"] == 1]
        .groupby("loan_sequence_number")["loan_age"]
        .min()
        .rename("FirstEventTime")
    )

    perf = perf.merge(first_event, on="loan_sequence_number", how="left")
    perf.drop(columns=["_is_default"], inplace=True)

    return perf

# ── Step 4: merge with matched ────────────────────────────────────────────────

def build_panel(perf: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    """
    Inner join performance (one row per loan × month) with matched
    (one row per loan, time-invariant origination + HMDA data).
    """
    print("Merging performance x origination/HMDA...")
    panel = perf.merge(matched, on="loan_sequence_number", how="inner")
    gc.collect()

    print(f"Panel: {panel.shape[0]:,} rows x {panel.shape[1]} columns")
    print(f"Unique loans in panel: {panel['loan_sequence_number'].nunique():,}")

    dups = panel.duplicated(
        subset=["loan_sequence_number", "monthly_reporting_period"]
    ).sum()
    if dups > 0:
        print(f"WARNING: {dups:,} duplicate rows on (loan, month).")
    else:
        print("OK: no duplicates on (loan_sequence_number, month).")

    return panel


# ── Step 5: order columns and save ───────────────────────────────────────────

def save_panel(panel: pd.DataFrame, matched: pd.DataFrame,
               output_path: str) -> None:
    """Order columns and save panel to CSV."""

    id_cols = [
        "loan_sequence_number", "period_quarter", "period_year",
        "quarter", "monthly_reporting_period",
    ]
    perf_out_cols = [
        "loan_age", "remaining_months_to_maturity",
        "current_upb", "current_interest_rate", "estimated_ltv",
        "current_deferred_upb", "current_loan_delinquency_status",
        "modifications_flag", "zero_balance_code",
        "zero_balance_effective_date",
        "delinquency_due_to_disaster", "borrower_assistance_status",
    ]
    target_cols = [c for c in ["default", "prepayment"] if c in panel.columns]

    already_placed   = set(id_cols + perf_out_cols + target_cols)
    origination_cols = [
        c for c in matched.columns
        if c != "loan_sequence_number" and c not in already_placed
    ]

    ordered = []
    seen    = set()
    for group in [id_cols, perf_out_cols, target_cols, origination_cols]:
        for c in group:
            if c in panel.columns and c not in seen:
                ordered.append(c)
                seen.add(c)

    remaining = [c for c in panel.columns if c not in seen]
    ordered  += remaining

    panel = panel[ordered]
    panel.sort_values(
        ["loan_sequence_number", "monthly_reporting_period"], inplace=True
    )
    panel.reset_index(drop=True, inplace=True)

    panel.to_csv(output_path, index=False)
    size_mb = os.path.getsize(output_path) / 1e6

    print(f"\nSaved : {output_path}")
    print(f"Size  : {size_mb:.1f} MB")
    print(f"Rows  : {len(panel):,}")
    print(f"Cols  : {len(panel.columns)}")
    print()
    print(f"[A] Identifiers  : {[c for c in id_cols if c in panel.columns]}")
    print(f"[B] Performance  : {[c for c in perf_out_cols if c in panel.columns]}")
    print(f"[C] Survival tgt : {target_cols}")
    print(f"[D] Orig + HMDA  : "
          f"{len([c for c in origination_cols if c in panel.columns])} columns")
    if remaining:
        print(f"[E] Other        : {remaining}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_build_panel(year: int, drive_root: str) -> None:
    freddie_dir  = os.path.join(drive_root, "freddie")
    output_dir   = os.path.join(drive_root, "output")
    perf_local   = os.path.join(drive_root, "perf_local_tmp")
    matched_path = os.path.join(output_dir, f"matched_{year}.csv")
    output_path  = os.path.join(output_dir, f"panel_{year}.csv")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(perf_local, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Year        : {year}")
    print(f"  Freddie dir : {freddie_dir}")
    print(f"  Matched     : {matched_path}")
    print(f"  Output      : {output_path}")
    print(f"{'='*60}\n")

    if not os.path.exists(matched_path):
        raise FileNotFoundError(
            f"matched_{year}.csv not found at {matched_path}.\n"
            f"Run match_hmda.py --year {year} first."
        )

    # ── Load matched ──────────────────────────────────────────────────────────
    print(f"Loading matched: {matched_path}")
    matched  = pd.read_csv(matched_path, dtype=str, low_memory=False)
    loan_ids = set(matched["loan_sequence_number"].dropna().unique())
    print(f"  {len(matched):,} loans x {len(matched.columns)} columns")
    print(f"  Unique loan IDs: {len(loan_ids):,}\n")

    # ── Extract + load performance ────────────────────────────────────────────
    print(f"Extracting performance files for {year}...")
    perf_txt_files = extract_performance_zips(year, freddie_dir, perf_local)
    print(f"\nExtracted {len(perf_txt_files)} files:")
    for f in perf_txt_files:
        print(f"  {os.path.basename(f)}  ({os.path.getsize(f)/1e6:.0f} MB)")

    print("\nLoading performance data...")
    parquet_path = os.path.join(drive_root, "perf_tmp.parquet")
    perf = load_performance(perf_txt_files, loan_ids, parquet_path)

    # ── Add time columns ──────────────────────────────────────────────────────
    print("\nParsing time columns...")
    perf = add_time_columns(perf)

    # ── Build panel ───────────────────────────────────────────────────────────
    panel = build_panel(perf, matched)
    del perf
    gc.collect()

    # ── Save ──────────────────────────────────────────────────────────────────
    save_panel(panel, matched, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build longitudinal panel from Freddie Mac performance data."
    )
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
