"""
data_generation/fnma/sample_panel.py

Stratified sampling from multi-year Freddie Mac panels.
Produces panel_all_years_sampled.csv for model training.

Strategy:
  - Keep ALL defaulters (capped at MAX_DEFAULT, stratified by year)
  - Fill remaining slots with non-defaulters that have variable TVC
  - If still not enough, add flat non-defaulters
  - Target default rate = DEFAULT_RATE

Usage:
    python sample_panel.py --output_path /path/to/output --years 2018 2024

Input:
    output_path/panel_{YEAR}.csv  (one per year, from build_panel.py)

Output:
    output_path/panel_all_years_sampled.csv
"""

import os
import argparse
import numpy as np
import pandas as pd


COLS = [
    "loan_sequence_number",
    "current_loan_delinquency_status",
    "loan_age",
    "current_upb",
    "current_interest_rate",
    "estimated_ltv",
]


# ── Pass 1: aggregate per-loan statistics ────────────────────────────────────

def pass1_aggregate(output_path: str, years: range) -> pd.DataFrame:
    """
    Read all panel files in chunks and aggregate per-loan statistics:
    ever_default, origin_year, max_age, std of TVC columns.
    """
    loan_data = {}

    for year in years:
        filepath = os.path.join(output_path, f"panel_{year}.csv")
        if not os.path.exists(filepath):
            print(f"[SKIP] {filepath} not found")
            continue
        print(f"[PASS 1] {year}...")

        for chunk in pd.read_csv(filepath, low_memory=False,
                                  usecols=COLS, chunksize=100_000):
            chunk["del"] = pd.to_numeric(
                chunk["current_loan_delinquency_status"], errors="coerce"
            ).fillna(0)
            chunk["default"] = (chunk["del"] != 0).astype(int)

            for col in ["loan_age", "current_upb",
                        "current_interest_rate", "estimated_ltv"]:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

            for lid, grp in chunk.groupby("loan_sequence_number"):
                if lid not in loan_data:
                    loan_data[lid] = {
                        "ever_default": 0,
                        "origin_year":  year,
                        "max_age":      0,
                        "upb_vals":     [],
                        "rate_vals":    [],
                        "ltv_vals":     [],
                    }
                if grp["default"].max() == 1:
                    loan_data[lid]["ever_default"] = 1
                loan_data[lid]["max_age"] = max(
                    loan_data[lid]["max_age"],
                    grp["loan_age"].dropna().max()
                    if grp["loan_age"].notna().any() else 0,
                )
                loan_data[lid]["upb_vals"].extend(
                    grp["current_upb"].dropna().tolist()
                )
                loan_data[lid]["rate_vals"].extend(
                    grp["current_interest_rate"].dropna().tolist()
                )
                loan_data[lid]["ltv_vals"].extend(
                    grp["estimated_ltv"].dropna().tolist()
                )

    rows = []
    for lid, d in loan_data.items():
        rows.append({
            "loan_id":      lid,
            "ever_default": d["ever_default"],
            "origin_year":  d["origin_year"],
            "max_age":      d["max_age"],
            "upb_std":      np.std(d["upb_vals"])  if len(d["upb_vals"])  > 1 else 0,
            "rate_std":     np.std(d["rate_vals"]) if len(d["rate_vals"]) > 1 else 0,
            "ltv_std":      np.std(d["ltv_vals"])  if len(d["ltv_vals"])  > 1 else 0,
            "n_obs":        len(d["upb_vals"]),
        })

    del loan_data
    loan_stats = pd.DataFrame(rows)

    print(f"\nTotal loans  : {len(loan_stats):,}")
    print(f"Defaulters   : {loan_stats['ever_default'].sum():,} "
          f"({loan_stats['ever_default'].mean():.2%})")
    return loan_stats


# ── Stratified sampling ───────────────────────────────────────────────────────

def sample_loans(loan_stats: pd.DataFrame,
                 max_default: int,
                 default_rate: float,
                 random_seed: int = 42) -> np.ndarray:
    """
    Sample loan IDs:
      - All defaulters (capped at max_default, stratified by year)
      - Non-defaulters with variable TVC to reach target default_rate
      - Flat non-defaulters to fill remaining slots if needed
    """
    np.random.seed(random_seed)

    # ── Defaulters ────────────────────────────────────────────────────────────
    all_defaults = loan_stats[loan_stats["ever_default"] == 1]

    if len(all_defaults) > max_default:
        all_defaults = (
            all_defaults
            .groupby("origin_year", group_keys=False)
            .apply(lambda x: x.sample(
                n=max(1, int(max_default * len(x) / len(all_defaults))),
                random_state=random_seed,
            ))
        )

    g1_default = all_defaults["loan_id"].values
    n_default  = len(g1_default)

    # ── Non-defaulters with variable TVC ──────────────────────────────────────
    g2_tvc = loan_stats[
        (loan_stats["ever_default"] == 0) &
        (
            (loan_stats["upb_std"]  > 0) |
            (loan_stats["rate_std"] > 0) |
            (loan_stats["ltv_std"]  > 0)
        )
    ]["loan_id"].values

    target_non_default = int(n_default / default_rate) - n_default
    n_g2       = min(len(g2_tvc), target_non_default)
    sampled_g2 = np.random.choice(g2_tvc, size=n_g2, replace=False)

    # ── Flat non-defaulters (filler if needed) ────────────────────────────────
    remaining = target_non_default - n_g2
    if remaining > 0:
        g3_flat = loan_stats[
            (loan_stats["ever_default"] == 0) &
            (loan_stats["upb_std"] == 0)
        ]["loan_id"].values
        n_g3       = min(len(g3_flat), remaining)
        sampled_g3 = np.random.choice(g3_flat, size=n_g3, replace=False)
    else:
        sampled_g3 = np.array([])

    sampled_ids = np.concatenate([g1_default, sampled_g2, sampled_g3])

    total = len(sampled_ids)
    print(f"\nFinal sample : {total:,}")
    print(f"  Defaulters : {n_default:,}  ({n_default/total:.1%})")
    print(f"  TVC        : {n_g2:,}  ({n_g2/total:.1%})")
    print(f"  Flat       : {len(sampled_g3):,}  ({len(sampled_g3)/total:.1%})")

    return sampled_ids


# ── Pass 2: filter and concatenate ───────────────────────────────────────────

def pass2_filter(output_path: str, years: range,
                 sampled_set: set) -> pd.DataFrame:
    """
    Re-read all panel files and keep only sampled loan IDs.
    Adds source_year column for temporal train/test splitting.
    """
    chunks = []

    for year in years:
        filepath = os.path.join(output_path, f"panel_{year}.csv")
        if not os.path.exists(filepath):
            continue
        print(f"[PASS 2] {year}...")

        for chunk in pd.read_csv(filepath, low_memory=False, chunksize=100_000):
            filtered = chunk[chunk["loan_sequence_number"].isin(sampled_set)]
            if len(filtered) > 0:
                filtered = filtered.copy()
                filtered["source_year"] = year
                chunks.append(filtered)

    sampled_df = pd.concat(chunks, ignore_index=True)
    del chunks
    return sampled_df


# ── Main ──────────────────────────────────────────────────────────────────────

def run_sampling(output_path: str, years: range,
                 max_default: int = 10_000,
                 default_rate: float = 0.30,
                 random_seed: int = 42,
                 out_filename: str = "panel_all_years_sampled.csv") -> None:

    print(f"\n{'='*60}")
    print(f"  Years        : {list(years)}")
    print(f"  Max default  : {max_default:,}")
    print(f"  Default rate : {default_rate:.0%}")
    print(f"  Output dir   : {output_path}")
    print(f"{'='*60}\n")

    # Pass 1
    loan_stats  = pass1_aggregate(output_path, years)

    # Sample
    sampled_ids = sample_loans(loan_stats, max_default, default_rate, random_seed)
    sampled_set = set(sampled_ids)

    # Pass 2
    print("\nFiltering panel files...")
    sampled_df = pass2_filter(output_path, years, sampled_set)

    # Save
    out_file = os.path.join(output_path, out_filename)
    sampled_df.to_csv(out_file, index=False)

    print(f"\nSaved  : {out_file}")
    print(f"Rows   : {len(sampled_df):,}")
    print(f"Loans  : {sampled_df['loan_sequence_number'].nunique():,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stratified sampling from multi-year Freddie Mac panels."
    )
    parser.add_argument(
        "--output_path", required=True,
        help="Directory containing panel_{YEAR}.csv files and output"
    )
    parser.add_argument(
        "--years", nargs=2, type=int, default=[2018, 2024],
        metavar=("START", "END"),
        help="Year range inclusive (default: 2018 2024)"
    )
    parser.add_argument(
        "--max_default", type=int, default=10_000,
        help="Cap on total defaulters (default: 10000)"
    )
    parser.add_argument(
        "--default_rate", type=float, default=0.30,
        help="Target default rate in final dataset (default: 0.30)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--out_filename", default="panel_all_years_sampled.csv",
        help="Output filename (default: panel_all_years_sampled.csv)"
    )
    args = parser.parse_args()

    run_sampling(
        output_path  = args.output_path,
        years        = range(args.years[0], args.years[1] + 1),
        max_default  = args.max_default,
        default_rate = args.default_rate,
        random_seed  = args.seed,
        out_filename = args.out_filename,
    )
