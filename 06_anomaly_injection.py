# ============================================================
# 06_anomaly_injection.py
# Thesis: Deep Sequential Models for ERP Anomaly Detection
# Mustafa Wasif Allvi — University of Potsdam, M.Sc. Data Science
# ============================================================
# Anomaly injection framework (thesis Section 5.5).
#
# Injects five anomaly types into the TEST SET ONLY.
# Training and validation sets remain 100% clean.
#
# Anomaly types (thesis Section 3.6):
#   1. control_flow  — swap timestamps of two activities
#   2. data          — multiply case:Amount by 10x for EMPLOYEE role
#   3. resource      — assign EMPLOYEE role to an approval event
#   4. temporal      — shift middle event timestamp back 25 hours
#   5. cross_entity  — multiply case:Amount 5x to exceed permit budget
#
# Injection rate: 10% of test cases
# Output: test_injected.joblib with sequences, labels, types
# ============================================================

import os
import random
import json
import numpy as np
import pandas as pd
import joblib
from collections import Counter

import config
from config import (
    PREPROCESSED_DIR, METRICS_DIR,
    CASE_ID_COL, TIMESTAMP_COL, ACTIVITY_COL,
    ANOMALY_FRACTION, ANOMALY_RANDOM_SEED,
    ANOMALY_TYPES, ANOMALY_PARAMS,
    MODEL_INPUT_DIM, MAX_PREFIX_LENGTH, MIN_TRACE_LENGTH,
)


def load_encoders():
    return joblib.load(config.ENCODER_PATH)


def get_activity_int(encoders, activity_name):
    classes = list(encoders["concept:name"].classes_)
    return classes.index(activity_name) if activity_name in classes else -1


def get_role_int(encoders, role_name):
    classes = list(encoders["org:role"].classes_)
    return classes.index(role_name) if role_name in classes else -1


# ============================================================
# Five injection functions
# ============================================================

def inject_control_flow(case_df, encoders):
    case_df     = case_df.copy().reset_index(drop=True)
    payment_int = get_activity_int(encoders, "Payment Handled")
    request_int = get_activity_int(encoders, "Request Payment")
    act_col_enc = "concept:name_enc"

    if not (case_df[act_col_enc] == payment_int).any():
        return None
    if not (case_df[act_col_enc] == request_int).any():
        return None

    idx_payment = case_df[case_df[act_col_enc] == payment_int].index[0]
    idx_request = case_df[case_df[act_col_enc] == request_int].index[0]

    if idx_request >= idx_payment:
        return None

    ts_p = case_df.at[idx_payment, TIMESTAMP_COL]
    ts_r = case_df.at[idx_request, TIMESTAMP_COL]
    case_df.at[idx_payment, TIMESTAMP_COL] = ts_r
    case_df.at[idx_request, TIMESTAMP_COL] = ts_p

    case_df = case_df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    case_df = recompute_temporal(case_df)
    return case_df


def inject_data_anomaly(case_df, encoders):
    case_df      = case_df.copy().reset_index(drop=True)
    employee_int = get_role_int(encoders, "EMPLOYEE")
    role_col_enc = "org:role_enc"
    amount_col   = "case:Amount_scaled"
    employee_mask = case_df[role_col_enc] == employee_int

    if not employee_mask.any():
        return None

    idx        = case_df[employee_mask].index[0]
    multiplier = ANOMALY_PARAMS["data"]["amount_multiplier"]
    scalers    = joblib.load(config.SCALER_PATH)

    if "case:Amount" in scalers:
        scaler  = scalers["case:Amount"]
        current = case_df.at[idx, amount_col]
        original = scaler.inverse_transform([[current]])[0][0]
        inflated = original * multiplier
        new_scaled = float(scaler.transform([[inflated]])[0][0])
        case_df.at[idx, amount_col] = new_scaled
    else:
        case_df.at[idx, amount_col] = min(
            case_df.at[idx, amount_col] * multiplier, 2.0
        )
    return case_df


def inject_resource_anomaly(case_df, encoders):
    case_df      = case_df.copy().reset_index(drop=True)
    employee_int = get_role_int(encoders, "EMPLOYEE")
    role_col_enc = "org:role_enc"
    act_col_enc  = "concept:name_enc"

    approval_ints = [
        get_activity_int(encoders, name)
        for name in encoders["concept:name"].classes_
        if "APPROVED" in name
    ]
    approval_mask = case_df[act_col_enc].isin(approval_ints)

    if not approval_mask.any():
        return None

    idx = case_df[approval_mask].index[0]
    if case_df.at[idx, role_col_enc] == employee_int:
        return None

    case_df.at[idx, role_col_enc] = employee_int
    return case_df


def inject_temporal_anomaly(case_df, encoders):
    case_df = case_df.copy().sort_values(
        TIMESTAMP_COL
    ).reset_index(drop=True)

    if len(case_df) < 3:
        return None

    mid_idx = len(case_df) // 2
    shift   = pd.Timedelta(hours=25)
    case_df.at[mid_idx, TIMESTAMP_COL] = (
        case_df.at[mid_idx, TIMESTAMP_COL] - shift
    )
    case_df = case_df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    case_df = recompute_temporal(case_df)
    return case_df


def inject_cross_entity_anomaly(case_df, encoders):
    case_df    = case_df.copy().reset_index(drop=True)
    budget_col = "case:Permit RequestedBudget_scaled"
    amount_col = "case:Amount_scaled"

    if budget_col not in case_df.columns:
        return None

    budget_val = case_df[budget_col].iloc[0]
    if budget_val == 0.0:
        return None

    multiplier = ANOMALY_PARAMS["cross_entity"]["amount_multiplier"]
    scalers    = joblib.load(config.SCALER_PATH)

    if "case:Amount" in scalers:
        scaler  = scalers["case:Amount"]
        current = case_df[amount_col].iloc[0]
        original = scaler.inverse_transform([[current]])[0][0]
        inflated = original * multiplier
        new_scaled = float(scaler.transform([[inflated]])[0][0])
        case_df[amount_col] = new_scaled
    else:
        case_df[amount_col] = min(
            case_df[amount_col].iloc[0] * multiplier, 2.0
        )
    return case_df


# ============================================================
# Recompute temporal features after timestamp changes
# ============================================================

def recompute_temporal(case_df):
    scalers   = joblib.load(config.SCALER_PATH)
    delta_raw = (
        case_df[TIMESTAMP_COL].diff().dt.total_seconds().fillna(0.0)
    )
    cum_raw = (
        case_df[TIMESTAMP_COL] - case_df[TIMESTAMP_COL].iloc[0]
    ).dt.total_seconds()

    if "delta_t" in scalers:
        dt_log    = np.log1p(delta_raw.values.reshape(-1, 1))
        dt_scaled = scalers["delta_t"].transform(dt_log).flatten()
        case_df["delta_t_scaled"] = dt_scaled

    if "cum_dur" in scalers:
        cd_log    = np.log1p(cum_raw.values.reshape(-1, 1))
        cd_scaled = scalers["cum_dur"].transform(cd_log).flatten()
        case_df["cum_dur_scaled"] = cd_scaled

    return case_df


INJECTION_FUNCTIONS = {
    "control_flow" : inject_control_flow,
    "data"         : inject_data_anomaly,
    "resource"     : inject_resource_anomaly,
    "temporal"     : inject_temporal_anomaly,
    "cross_entity" : inject_cross_entity_anomaly,
}


# ============================================================
# MAIN
# ============================================================

def run_injection(save=True):
    config.make_output_dirs()

    print("=" * 60)
    print("06_anomaly_injection.py — Anomaly Injection")
    print("=" * 60)

    print("\nLoading test data and artefacts...")
    test_csv = config.PROCESSED_CSV.replace(".csv", "_test.csv")
    df_test  = pd.read_csv(test_csv, low_memory=False)
    df_test[TIMESTAMP_COL] = pd.to_datetime(
        df_test[TIMESTAMP_COL], utc=True
    )
    encoders       = load_encoders()
    model_features = joblib.load(
        os.path.join(PREPROCESSED_DIR, "model_features.joblib")
    )

    print(f"  Test events:      {len(df_test):,}")
    print(f"  Test cases:       {df_test[CASE_ID_COL].nunique():,}")
    print(f"  Anomaly fraction: {ANOMALY_FRACTION*100:.0f}%")

    random.seed(ANOMALY_RANDOM_SEED)
    all_test_cases = list(df_test[CASE_ID_COL].unique())
    n_inject       = int(len(all_test_cases) * ANOMALY_FRACTION)
    cases_to_inject = random.sample(all_test_cases, n_inject)

    type_assignments = {
        case_id: ANOMALY_TYPES[i % len(ANOMALY_TYPES)]
        for i, case_id in enumerate(cases_to_inject)
    }

    print(f"\n  Cases selected for injection: {n_inject:,}")
    print(f"  Type distribution (intended):")
    for t in ANOMALY_TYPES:
        count = sum(1 for v in type_assignments.values() if v == t)
        print(f"    {t:<20}: {count}")

    print("\nApplying injections...")
    df_injected = df_test.copy()
    df_injected["anomaly_label"] = 0
    df_injected["anomaly_type"]  = "none"

    successful = Counter()
    failed     = Counter()

    for case_id, anomaly_type in type_assignments.items():
        mask      = df_injected[CASE_ID_COL] == case_id
        case_data = df_injected[mask].copy()

        result = INJECTION_FUNCTIONS[anomaly_type](case_data, encoders)

        if result is not None:
            df_injected = df_injected[~mask]
            result["anomaly_label"] = 1
            result["anomaly_type"]  = anomaly_type
            df_injected = pd.concat(
                [df_injected, result], ignore_index=True
            )
            successful[anomaly_type] += 1
        else:
            failed[anomaly_type] += 1

    total_anomalous = df_injected[
        df_injected["anomaly_label"] == 1
    ][CASE_ID_COL].nunique()
    total_normal = df_injected[
        df_injected["anomaly_label"] == 0
    ][CASE_ID_COL].nunique()

    print(f"\n  Injection results:")
    print(f"    Successful: {sum(successful.values())}")
    print(f"    Failed:     {sum(failed.values())}")
    print(f"\n  Successful by type:")
    for t in ANOMALY_TYPES:
        print(f"    {t:<20}: {successful[t]}")
    if any(failed.values()):
        print(f"\n  Failed by type:")
        for t in ANOMALY_TYPES:
            if failed[t] > 0:
                print(f"    {t:<20}: {failed[t]}")

    anomaly_rate = total_anomalous / (total_normal + total_anomalous) * 100
    print(f"\n  Final test composition:")
    print(f"    Normal cases:    {total_normal:,}")
    print(f"    Anomalous cases: {total_anomalous:,}")
    print(f"    Anomaly rate:    {anomaly_rate:.1f}%")

    print("\nCreating labeled prefix sequences...")
    sequences  = []
    targets    = []
    case_ids   = []
    labels     = []
    types      = []
    target_col = "concept:name_enc"

    for case_id, group in df_injected.groupby(CASE_ID_COL):
        group = group.sort_values(TIMESTAMP_COL)
        if len(group) < MIN_TRACE_LENGTH:
            continue
        feat_matrix   = group[model_features].values.astype(np.float32)
        target_vector = group[target_col].values.astype(np.int64)
        a_label       = int(group["anomaly_label"].iloc[0])
        a_type        = group["anomaly_type"].iloc[0]

        for i in range(1, len(feat_matrix)):
            sequences.append(feat_matrix[:i])
            targets.append(int(target_vector[i]))
            case_ids.append(case_id)
            labels.append(a_label)
            types.append(a_type)

    n_features = len(model_features)
    padded     = np.zeros(
        (len(sequences), MAX_PREFIX_LENGTH, n_features),
        dtype=np.float32
    )
    for i, seq in enumerate(sequences):
        length = min(len(seq), MAX_PREFIX_LENGTH)
        padded[i, :length, :] = seq[:length]

    print(f"  Total sequences:     {len(sequences):,}")
    print(f"  Normal sequences:    "
          f"{sum(1 for l in labels if l == 0):,}")
    print(f"  Anomalous sequences: "
          f"{sum(1 for l in labels if l == 1):,}")
    print(f"\n  Anomaly type distribution in sequences:")
    type_counts = Counter(types)
    for t in ANOMALY_TYPES + ["none"]:
        print(f"    {t:<20}: {type_counts[t]:,}")

    result = {
        "sequences"      : padded,
        "targets"        : np.array(targets,  dtype=np.int64),
        "labels"         : np.array(labels,   dtype=np.int64),
        "types"          : types,
        "case_ids"       : case_ids,
        "model_features" : model_features,
        "anomaly_rate"   : anomaly_rate,
        "successful"     : dict(successful),
        "failed"         : dict(failed),
    }

    if save:
        inj_path = os.path.join(PREPROCESSED_DIR, "test_injected.joblib")
        joblib.dump(result, inj_path)
        print(f"\n  Saved: {inj_path}")

        summary = {
            "n_inject"       : n_inject,
            "successful"     : dict(successful),
            "failed"         : dict(failed),
            "anomaly_rate"   : anomaly_rate,
            "total_sequences": len(sequences),
        }
        with open(os.path.join(METRICS_DIR,
                               "injection_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    print("\n  06_anomaly_injection.py complete.")
    print("=" * 60)
    return result


if __name__ == "__main__":
    run_injection(save=True)