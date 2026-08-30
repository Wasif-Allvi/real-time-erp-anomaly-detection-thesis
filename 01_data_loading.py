# ============================================================
# 01_data_loading.py
# Thesis: Deep Sequential Models for ERP Anomaly Detection
# ============================================================
# Stage 1 of the preprocessing pipeline (thesis Section 5.1).
# Loads both BPI Challenge 2020 XES files via pm4py, converts
# each to a flat pandas DataFrame, adds a dataset_source label,
# concatenates into one combined DataFrame, and saves it as
# events_raw_combined.csv in results/preprocessed/.
#
# Run standalone:  python 01_data_loading.py
# Or called from:  main.py -> load_raw_data()
# ============================================================

import os
import pandas as pd
import pm4py

import config
from config import (
    XES_DOMESTIC,
    XES_INTERNATIONAL,
    RAW_COMBINED_CSV,
    SOURCE_COL,
    CASE_ID_COL,
    TIMESTAMP_COL,
    ACTIVITY_COL,
)


# ------------------------------------------------------------
# Core loader
# ------------------------------------------------------------

def load_xes(path: str, source_label: str) -> pd.DataFrame:
    """
    Load a single XES file and return a flat DataFrame.

    Parameters
    ----------
    path         : absolute path to the .xes file
    source_label : string tag added to SOURCE_COL
                   ("domestic" / "international")

    Returns
    -------
    pd.DataFrame with one row per event and SOURCE_COL appended.
    """
    print(f"  Loading {source_label} XES from:\n    {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"XES file not found: {path}\n"
            "Check DATA_DIR and subfolder structure in config.py."
        )

    log = pm4py.read_xes(path)
    df  = pm4py.convert_to_dataframe(log)

    if TIMESTAMP_COL in df.columns:
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)

    df[SOURCE_COL] = source_label

    print(f"    Rows: {len(df):,}  |  "
          f"Cases: {df[CASE_ID_COL].nunique():,}  |  "
          f"Columns: {len(df.columns)}")
    return df


def load_raw_data(save: bool = True) -> pd.DataFrame:
    """
    Load both XES files, concatenate, and optionally save to CSV.

    Parameters
    ----------
    save : if True, writes combined DataFrame to RAW_COMBINED_CSV

    Returns
    -------
    pd.DataFrame — combined raw event log
    """
    config.make_output_dirs()

    print("=" * 60)
    print("01_data_loading.py — Loading BPI Challenge 2020 XES files")
    print("=" * 60)

    df_domestic      = load_xes(XES_DOMESTIC,      source_label="domestic")
    df_international = load_xes(XES_INTERNATIONAL, source_label="international")

    df_combined = pd.concat(
        [df_domestic, df_international], ignore_index=True
    )

    # --------------------------------------------------------
    # Verification summary
    # --------------------------------------------------------
    print()
    print("=" * 60)
    print("COMBINED DATASET SUMMARY")
    print("=" * 60)
    print(f"  Total rows (events)  : {len(df_combined):,}")
    print(f"  Total cases          : {df_combined[CASE_ID_COL].nunique():,}")
    print(f"  Total columns        : {len(df_combined.columns)}")
    print()

    print("  Row counts by source:")
    print(df_combined[SOURCE_COL].value_counts().to_string())
    print()

    print("  Case counts by source:")
    print(
        df_combined.groupby(SOURCE_COL)[CASE_ID_COL]
        .nunique()
        .to_string()
    )
    print()

    print("  Activity label counts (top 10):")
    print(df_combined[ACTIVITY_COL].value_counts().head(10).to_string())
    print()

    print("  All columns present in combined DataFrame:")
    for i, col in enumerate(sorted(df_combined.columns), 1):
        print(f"    {i:>3}. {col}")
    print()

    # --------------------------------------------------------
    # Required column check — uses actual dataset column names
    # confirmed from first run of this script
    # --------------------------------------------------------
    required_cols = [
        CASE_ID_COL,                          # case:concept:name
        ACTIVITY_COL,                         # concept:name
        TIMESTAMP_COL,                        # time:timestamp
        "org:resource",
        "org:role",
        "case:Amount",
        "case:RequestedAmount",
        "case:AdjustedAmount",
        "case:OriginalAmount",
        "case:Permit RequestedBudget",
        "case:Permit ID",
        "case:Permit BudgetNumber",
        "case:BudgetNumber",
        "case:Permit ProjectNumber",
        "case:Permit OrganizationalEntity",
    ]

    print("  Required column check:")
    all_present = True
    for col in required_cols:
        present = col in df_combined.columns
        status  = "OK" if present else "MISSING"
        print(f"    [{status}] {col}")
        if not present:
            all_present = False

    print()
    if all_present:
        print("  All required columns present.")
    else:
        print("  WARNING: Some required columns are missing.")
        print("  Check XES attribute names in the output above.")

    # --------------------------------------------------------
    # Missing value overview for required columns
    # --------------------------------------------------------
    print()
    print("  Missing value counts per required column:")
    for col in required_cols:
        if col in df_combined.columns:
            n_missing = df_combined[col].isna().sum()
            pct       = 100 * n_missing / len(df_combined)
            print(f"    {col:<40} {n_missing:>7,}  ({pct:.1f}%)")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    if save:
        df_combined.to_csv(RAW_COMBINED_CSV, index=False)
        print()
        print(f"  Raw combined CSV saved to:\n    {RAW_COMBINED_CSV}")

    print()
    print("  01_data_loading.py complete.")
    print("=" * 60)

    return df_combined


# ------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    df = load_raw_data(save=True)