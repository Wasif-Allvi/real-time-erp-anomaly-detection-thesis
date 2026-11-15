# ============================================================
# 19_amount_range_check.py
# Thesis: Deep Sequential Models for ERP Anomaly Detection
# Mustafa Wasif Allvi, University of Potsdam, M.Sc. Data Science
# ============================================================
# How unusual is the size of an injected amount?
#
# Data anomalies multiply case:Amount on one event by 10, cross-entity
# anomalies multiply it on every event of a case by 5. This script
# compares the inflated amounts in the anomalous test prefixes with
# the amounts of the training cases:
#   - share of anomalous prefixes whose amount is above the largest
#     training amount (scaled value above 1),
#   - percentile of the inflated amount among the training cases
#     (median over the anomalous prefixes).
# Nothing is trained. Runs in a few seconds.
#
# Inputs : results/preprocessed/test_injected.joblib,
#          results/preprocessed/events_processed_train.csv,
#          results/metrics/event_labels.npz (13_event_level_evaluation.py)
# Output : results/metrics/amount_range_check.json
# Usage  : python 19_amount_range_check.py
# ============================================================

import json
import os

import joblib
import numpy as np
import pandas as pd

from config import METRICS_DIR, PREPROCESSED_DIR

AMOUNT_INDEX = 7                     # case:Amount_scaled in the 14 features


def main():
    inj = joblib.load(os.path.join(PREPROCESSED_DIR, "test_injected.joblib"))
    x = np.asarray(inj["sequences"])
    types = np.asarray(inj["types"])
    assert inj["model_features"][AMOUNT_INDEX] == "case:Amount_scaled"
    labels = np.load(os.path.join(METRICS_DIR, "event_labels.npz"))["event_labels"]

    train = pd.read_csv(os.path.join(PREPROCESSED_DIR, "events_processed_train.csv"),
                        usecols=["case:concept:name", "case:Amount_scaled"])
    case_amounts = np.sort(train.groupby("case:concept:name")["case:Amount_scaled"].first().values)
    train_max = float(train["case:Amount_scaled"].max())

    out = {"training_cases": int(len(case_amounts)), "training_max_scaled": train_max}
    for t in ("data", "cross_entity"):
        mask = (types == t) & (labels == 1)
        amount = x[mask][:, :, AMOUNT_INDEX].max(axis=1)          # inflated amount in the prefix
        pct = np.searchsorted(case_amounts, amount, side="right") / len(case_amounts) * 100
        out[t] = {
            "anomalous_prefixes": int(mask.sum()),
            "share_above_training_max_pct": float((amount > train_max).mean() * 100),
            "median_percentile_among_training_cases": float(np.median(pct)),
            "median_scaled_amount": float(np.median(amount)),
        }
        print(f"{t:13s} {mask.sum():4d} anomalous prefixes | above training max "
              f"{out[t]['share_above_training_max_pct']:5.1f}% | median percentile among "
              f"training cases {out[t]['median_percentile_among_training_cases']:5.1f}")

    path = os.path.join(METRICS_DIR, "amount_range_check.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
