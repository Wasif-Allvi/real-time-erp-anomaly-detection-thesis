# ============================================================
# 02_preprocessing.py
# Thesis: Deep Sequential Models for ERP Anomaly Detection
# Mustafa Wasif Allvi — University of Potsdam, M.Sc. Data Science
# ============================================================
# Preprocessing pipeline (thesis Section 5.1).
#
# Split: Train 70% / Val 10% / Test 20% at case level
# Leakage prevention: scaler fitted on TRAIN only
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

import config
from config import (
    RAW_COMBINED_CSV, PROCESSED_CSV, SPLIT_IDS_PATH,
    SCALER_PATH, ENCODER_PATH,
    CASE_ID_COL, ACTIVITY_COL, TIMESTAMP_COL,
    JOIN_KEY_COL, SOURCE_COL,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
    LOG_TRANSFORM_FEATURES,
    CATEGORICAL_FILL_TOKEN, NUMERICAL_FILL_VALUE,
    RANDOM_SEED, MODEL_INPUT_DIM, MIN_TRACE_LENGTH,
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
TEST_RATIO  = 0.20

KEEP_COLS = (
    [CASE_ID_COL, TIMESTAMP_COL, SOURCE_COL, JOIN_KEY_COL]
    + CATEGORICAL_FEATURES
    + [c for c in NUMERICAL_FEATURES if c not in ("delta_t", "cum_dur")]
)


def load_raw(path):
    print(f"  Loading raw data from:\n    {path}")
    df = pd.read_csv(path, low_memory=False)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
    print(f"  Loaded {len(df):,} rows, {df[CASE_ID_COL].nunique():,} cases")
    return df


def select_features(df):
    present = [c for c in KEEP_COLS if c in df.columns]
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        print(f"  WARNING: columns not found: {missing}")
    df = df[present].copy()
    print(f"  Selected {len(present)} columns")
    return df


def sort_events(df):
    df = df.sort_values([CASE_ID_COL, TIMESTAMP_COL]).reset_index(drop=True)
    print(f"  Sorted {len(df):,} events")
    return df


def handle_missing(df):
    cat_present = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    df[cat_present] = df[cat_present].fillna(CATEGORICAL_FILL_TOKEN)

    num_raw = [c for c in NUMERICAL_FEATURES
               if c not in ("delta_t", "cum_dur") and c in df.columns]
    df[num_raw] = df[num_raw].fillna(NUMERICAL_FILL_VALUE)

    if JOIN_KEY_COL in df.columns:
        df[JOIN_KEY_COL] = df[JOIN_KEY_COL].fillna(CATEGORICAL_FILL_TOKEN)

    print(f"  After imputation: {df.isnull().sum().sum()} nulls remain")
    return df


def derive_temporal(df):
    df["delta_t"] = (
        df.groupby(CASE_ID_COL)[TIMESTAMP_COL]
        .diff().dt.total_seconds().fillna(0.0)
    )
    df["cum_dur"] = df.groupby(CASE_ID_COL)[TIMESTAMP_COL].transform(
        lambda x: (x - x.min()).dt.total_seconds()
    )
    print(f"  delta_t range: [{df['delta_t'].min():.0f}s, "
          f"{df['delta_t'].max():.0f}s]  "
          f"mean {df['delta_t'].mean()/3600:.1f}h")
    print(f"  cum_dur mean:  {df['cum_dur'].mean()/86400:.1f} days")
    return df


def split_cases(df):
    all_cases = df[CASE_ID_COL].unique()
    total     = len(all_cases)

    # Step 1: separate test from remainder
    remainder, test_cases = train_test_split(
        all_cases, test_size=TEST_RATIO, random_state=RANDOM_SEED
    )

    # Step 2: split remainder into train and val
    # val fraction relative to remainder
    val_frac = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_cases, val_cases = train_test_split(
        remainder, test_size=val_frac, random_state=RANDOM_SEED
    )

    df_train = df[df[CASE_ID_COL].isin(train_cases)].copy()
    df_val   = df[df[CASE_ID_COL].isin(val_cases)].copy()
    df_test  = df[df[CASE_ID_COL].isin(test_cases)].copy()

    print(f"  Train: {len(train_cases):,} cases ({len(train_cases)/total*100:.0f}%), "
          f"{len(df_train):,} events")
    print(f"  Val:   {len(val_cases):,} cases ({len(val_cases)/total*100:.0f}%),  "
          f"{len(df_val):,} events")
    print(f"  Test:  {len(test_cases):,} cases ({len(test_cases)/total*100:.0f}%), "
          f"{len(df_test):,} events")

    assert len(set(train_cases) & set(val_cases))  == 0, "Train/Val overlap!"
    assert len(set(train_cases) & set(test_cases)) == 0, "Train/Test overlap!"
    assert len(set(val_cases)   & set(test_cases)) == 0, "Val/Test overlap!"
    print(f"  Overlap check: 0 cases in any two sets")

    split_ids = {
        "train": list(train_cases),
        "val":   list(val_cases),
        "test":  list(test_cases),
    }
    return df_train, df_val, df_test, split_ids


def encode_categoricals(df_full, df_train, df_val, df_test):
    encoders    = {}
    cat_present = [c for c in CATEGORICAL_FEATURES if c in df_full.columns]

    for col in cat_present:
        le  = LabelEncoder()
        le.fit(df_full[col].astype(str))
        enc = col + "_enc"
        df_train[enc] = le.transform(df_train[col].astype(str))
        df_val[enc]   = le.transform(df_val[col].astype(str))
        df_test[enc]  = le.transform(df_test[col].astype(str))
        encoders[col] = le
        print(f"    {col}: {len(le.classes_)} classes")

    return df_train, df_val, df_test, encoders


def scale_numericals(df_train, df_val, df_test):
    scalers = {}
    num_raw = [c for c in NUMERICAL_FEATURES
               if c not in ("delta_t", "cum_dur") and c in df_train.columns]
    all_num = num_raw + ["delta_t", "cum_dur"]

    for col in all_num:
        if col in LOG_TRANSFORM_FEATURES:
            df_train[col] = np.log1p(df_train[col])
            df_val[col]   = np.log1p(df_val[col])
            df_test[col]  = np.log1p(df_test[col])

        scaler     = MinMaxScaler()
        scaled_col = col + "_scaled"
        scaler.fit(df_train[[col]])
        df_train[scaled_col] = scaler.transform(df_train[[col]])
        df_val[scaled_col]   = scaler.transform(df_val[[col]])
        df_test[scaled_col]  = scaler.transform(df_test[[col]])
        scalers[col] = scaler
        print(f"    {col}: train range "
              f"[{scaler.data_min_[0]:.4f}, {scaler.data_max_[0]:.4f}] -> [0,1]")

    return df_train, df_val, df_test, scalers


def get_model_features(df_train):
    cat_enc = [c + "_enc"    for c in CATEGORICAL_FEATURES
               if c + "_enc" in df_train.columns]
    num_sc  = [c + "_scaled" for c in NUMERICAL_FEATURES
               if c + "_scaled" in df_train.columns]
    features = cat_enc + num_sc
    print(f"  Model input features: {len(features)}")
    for i, f in enumerate(features, 1):
        print(f"    {i:2d}. {f}")
    return features


def run_preprocessing(save=True):
    config.make_output_dirs()

    print("=" * 60)
    print("02_preprocessing.py — Preprocessing Pipeline")
    print(f"  Split: Train {TRAIN_RATIO*100:.0f}% / "
          f"Val {VAL_RATIO*100:.0f}% / "
          f"Test {TEST_RATIO*100:.0f}%")
    print("=" * 60)

    print("\nStep 1: Load raw data")
    df = load_raw(RAW_COMBINED_CSV)

    print("\nStep 2: Select features")
    df = select_features(df)

    print("\nStep 3: Sort events")
    df = sort_events(df)

    print("\nStep 4: Handle missing values")
    df = handle_missing(df)

    print("\nStep 5: Derive temporal features")
    df = derive_temporal(df)

    print("\nStep 6: Three-way case-level split")
    df_train, df_val, df_test, split_ids = split_cases(df)

    print("\nStep 7: Label-encode categoricals (fit on full dataset)")
    df_train, df_val, df_test, encoders = encode_categoricals(
        df, df_train, df_val, df_test
    )

    print("\nStep 8: Scale numericals (fit on train only)")
    df_train, df_val, df_test, scalers = scale_numericals(
        df_train, df_val, df_test
    )

    print("\nStep 9: Verify model features")
    model_features = get_model_features(df_train)

    if len(model_features) != MODEL_INPUT_DIM:
        print(f"\n  WARNING: expected {MODEL_INPUT_DIM}, "
              f"got {len(model_features)}")
    else:
        print(f"\n  Feature count verified: {len(model_features)} == {MODEL_INPUT_DIM}")

    if save:
        df_train.to_csv(PROCESSED_CSV.replace(".csv", "_train.csv"), index=False)
        df_val.to_csv(  PROCESSED_CSV.replace(".csv", "_val.csv"),   index=False)
        df_test.to_csv( PROCESSED_CSV.replace(".csv", "_test.csv"),  index=False)
        joblib.dump(split_ids,     SPLIT_IDS_PATH)
        joblib.dump(encoders,      ENCODER_PATH)
        joblib.dump(scalers,       SCALER_PATH)
        joblib.dump(model_features,
                    os.path.join(config.PREPROCESSED_DIR,
                                 "model_features.joblib"))
        print(f"\n  Saved: train/val/test CSVs, encoders, scalers, split IDs")

    print("\n  02_preprocessing.py complete.")
    print("=" * 60)
    return df_train, df_val, df_test, encoders, scalers, model_features


if __name__ == "__main__":
    run_preprocessing(save=True)