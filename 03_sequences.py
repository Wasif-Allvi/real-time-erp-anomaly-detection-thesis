# ============================================================
# 03_sequences.py
# Thesis: Deep Sequential Models for ERP Anomaly Detection
# Mustafa Wasif Allvi — University of Potsdam, M.Sc. Data Science
# ============================================================
# Prefix sequence creation (thesis Section 5.2).
# Creates sequences for train, val, and test splits.
# ============================================================

import os
import numpy as np
import pandas as pd
import joblib
from collections import Counter

import config
from config import (
    PROCESSED_CSV, PREPROCESSED_DIR,
    CASE_ID_COL, TIMESTAMP_COL,
    MAX_PREFIX_LENGTH, MODEL_INPUT_DIM, MIN_TRACE_LENGTH,
)


def load_processed(split):
    path = PROCESSED_CSV.replace(".csv", f"_{split}.csv")
    df   = pd.read_csv(path, low_memory=False)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
    print(f"  Loaded {split}: {len(df):,} events, "
          f"{df[CASE_ID_COL].nunique():,} cases")
    return df


def load_model_features():
    path     = os.path.join(PREPROCESSED_DIR, "model_features.joblib")
    features = joblib.load(path)
    print(f"  Model features: {len(features)}")
    return features


def create_prefix_sequences(df, model_features, target_col, split_name):
    sequences = []
    targets   = []
    case_ids  = []
    skipped   = 0

    for case_id, group in df.groupby(CASE_ID_COL):
        group = group.sort_values(TIMESTAMP_COL)
        if len(group) < MIN_TRACE_LENGTH:
            skipped += 1
            continue
        feat_mat = group[model_features].values.astype(np.float32)
        tgt_vec  = group[target_col].values.astype(np.int64)
        for i in range(1, len(feat_mat)):
            sequences.append(feat_mat[:i])
            targets.append(int(tgt_vec[i]))
            case_ids.append(case_id)

    print(f"  {split_name}: {len(sequences):,} sequences "
          f"(skipped {skipped} short traces)")
    return sequences, targets, case_ids


def pad_sequences(sequences, max_len, n_features):
    padded = np.zeros(
        (len(sequences), max_len, n_features), dtype=np.float32
    )
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        padded[i, :length, :] = seq[:length]
    return padded


def print_target_distribution(targets, encoders, split_name, top_n=5):
    total  = len(targets)
    counts = Counter(targets)
    print(f"  Target distribution ({split_name}, top {top_n}):")
    for act_int, count in counts.most_common(top_n):
        name = encoders["concept:name"].classes_[act_int]
        pct  = 100 * count / total
        print(f"    {pct:5.1f}%  ({count:6,})  -> {name}")


def run_sequence_creation(save=True):
    config.make_output_dirs()

    print("=" * 60)
    print("03_sequences.py — Prefix Sequence Creation")
    print("=" * 60)

    print("\nLoading processed data...")
    df_train = load_processed("train")
    df_val   = load_processed("val")
    df_test  = load_processed("test")

    model_features = load_model_features()
    encoders       = joblib.load(config.ENCODER_PATH)
    target_col     = "concept:name_enc"

    if target_col not in df_train.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    print("\nCreating prefix sequences...")
    train_seqs, train_tgts, train_cids = create_prefix_sequences(
        df_train, model_features, target_col, "train"
    )
    val_seqs, val_tgts, val_cids = create_prefix_sequences(
        df_val, model_features, target_col, "val"
    )
    test_seqs, test_tgts, test_cids = create_prefix_sequences(
        df_test, model_features, target_col, "test"
    )

    train_lens = [len(s) for s in train_seqs]
    print(f"\n  Sequence lengths (train):")
    print(f"    min: {min(train_lens)}  max: {max(train_lens)}  "
          f"mean: {np.mean(train_lens):.1f}")

    observed_max = max(
        max(len(s) for s in train_seqs),
        max(len(s) for s in val_seqs),
        max(len(s) for s in test_seqs),
    )
    max_len    = MAX_PREFIX_LENGTH
    n_features = len(model_features)

    if observed_max > max_len:
        print(f"\n  WARNING: observed max {observed_max} > "
              f"MAX_PREFIX_LENGTH {max_len}. Truncating.")
    else:
        print(f"\n  Max prefix length: {max_len} "
              f"(observed {observed_max}) — OK")

    print("\nPadding sequences...")
    train_padded = pad_sequences(train_seqs, max_len, n_features)
    val_padded   = pad_sequences(val_seqs,   max_len, n_features)
    test_padded  = pad_sequences(test_seqs,  max_len, n_features)

    print(f"  Train: {train_padded.shape}")
    print(f"  Val:   {val_padded.shape}")
    print(f"  Test:  {test_padded.shape}")

    print_target_distribution(train_tgts, encoders, "train")

    result = {
        "train_sequences" : train_padded,
        "val_sequences"   : val_padded,
        "test_sequences"  : test_padded,
        "train_targets"   : np.array(train_tgts, dtype=np.int64),
        "val_targets"     : np.array(val_tgts,   dtype=np.int64),
        "test_targets"    : np.array(test_tgts,  dtype=np.int64),
        "train_case_ids"  : train_cids,
        "val_case_ids"    : val_cids,
        "test_case_ids"   : test_cids,
        "model_features"  : model_features,
        "max_len"         : max_len,
        "n_features"      : n_features,
        "num_classes"     : len(encoders["concept:name"].classes_),
    }

    if save:
        seq_path = os.path.join(PREPROCESSED_DIR, "sequences.joblib")
        joblib.dump(result, seq_path)
        size_mb = os.path.getsize(seq_path) / (1024 * 1024)
        print(f"\n  Saved: {seq_path}  ({size_mb:.1f} MB)")

    print(f"\n  Summary:")
    print(f"    Train: {len(train_seqs):,} sequences")
    print(f"    Val:   {len(val_seqs):,} sequences")
    print(f"    Test:  {len(test_seqs):,} sequences")
    print(f"    Input shape: (B, {max_len}, {n_features})")
    print(f"    Num classes: {result['num_classes']}")

    print("\n  03_sequences.py complete.")
    print("=" * 60)
    return result


if __name__ == "__main__":
    run_sequence_creation(save=True)