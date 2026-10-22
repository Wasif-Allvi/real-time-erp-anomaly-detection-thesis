# ============================================================
# 09_baseline.py
# Non-deep baselines for ERP anomaly detection
#
# Purpose: establish whether deep sequential modeling is
# warranted, by scoring the same injected test partition with
# two classical unsupervised detectors fitted on the same
# clean training data and the same 14-feature schema.
#
#   Isolation Forest  — tree-based isolation of outliers
#   One-Class SVM     — boundary-based novelty detection
#
# Sequences are reduced to one feature vector per sequence by
# averaging over real (non-padded) timesteps, which mirrors the
# reconstruction target used by the autoencoder in 05b.
#
# Output: results/metrics/baseline_results.json
# ============================================================

import os
import json
import numpy as np
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

import config
from config import PREPROCESSED_DIR, METRICS_DIR, RANDOM_SEED

ANOMALY_TYPES = ["control_flow", "data", "resource", "temporal", "cross_entity"]


# ------------------------------------------------------------
# Sequence -> fixed-length feature vector
# ------------------------------------------------------------
def aggregate(sequences):
    """Mean over real timesteps, as in the autoencoder target.

    A padded timestep is an all-zero row, so the mask below keeps
    exactly the positions that carry an observed event.
    """
    mask = (np.abs(sequences).sum(axis=2) > 0).astype(np.float32)   # (B, L)
    n_real = np.clip(mask.sum(axis=1, keepdims=True), 1, None)      # (B, 1)
    summed = (sequences * mask[:, :, None]).sum(axis=1)             # (B, F)
    return summed / n_real


# ------------------------------------------------------------
# Metrics, matching 07_scoring.py so numbers are comparable
# ------------------------------------------------------------
def evaluate(scores, y_true, types, name):
    roc = float(roc_auc_score(y_true, scores))
    auprc = float(average_precision_score(y_true, scores))
    baseline = float(y_true.mean())

    prec, rec, thr = precision_recall_curve(y_true, scores)
    f1 = np.divide(
        2 * prec * rec, prec + rec,
        out=np.zeros_like(prec), where=(prec + rec) > 0,
    )
    best = int(np.argmax(f1))
    best_thr = float(thr[best]) if best < len(thr) else float(thr[-1])

    pred = (scores > best_thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())

    per_type = {}
    for t in ANOMALY_TYPES:
        m = (types == t)
        if m.sum() == 0:
            continue
        per_type[t] = {
            "detected": int((scores[m] > best_thr).sum()),
            "total": int(m.sum()),
            "rate_pct": float(100.0 * (scores[m] > best_thr).sum() / m.sum()),
        }

    res = {
        "method": name,
        "roc_auc": roc,
        "auprc": auprc,
        "auprc_random_baseline": baseline,
        "best_f1": float(f1[best]),
        "threshold": best_thr,
        "precision_at_threshold": float(prec[best]),
        "recall_at_threshold": float(rec[best]),
        "tp": tp, "fp": fp, "fn": fn,
        "type_detection": per_type,
    }

    print(f"\n  {name}")
    print(f"    ROC-AUC   : {roc:.4f}")
    print(f"    AUPRC     : {auprc:.4f}   (random {baseline:.4f})")
    print(f"    Best F1   : {f1[best]:.4f} at threshold {best_thr:.6f}")
    print(f"    Precision : {prec[best]:.4f}")
    print(f"    Recall    : {rec[best]:.4f}")
    for t, v in per_type.items():
        print(f"      {t:14} {v['rate_pct']:5.1f}%  ({v['detected']}/{v['total']})")
    return res


def main():
    print("=" * 60)
    print("09_baseline.py — Non-deep baselines")
    print("=" * 60)

    print("\nLoading clean training sequences...")
    data = joblib.load(os.path.join(PREPROCESSED_DIR, "sequences.joblib"))
    X_train = aggregate(data["train_sequences"])
    print(f"  Train: {X_train.shape[0]:,} sequences, {X_train.shape[1]} features")

    print("\nLoading injected test sequences...")
    inj = joblib.load(os.path.join(PREPROCESSED_DIR, "test_injected.joblib"))
    X_test = aggregate(inj["sequences"])
    y_true = np.asarray(inj["labels"])
    types = np.asarray(inj["types"])
    print(f"  Test : {X_test.shape[0]:,} sequences, "
          f"{int(y_true.sum()):,} anomalous ({100 * y_true.mean():.2f}%)")

    results = {}

    # ---- Isolation Forest -------------------------------------------------
    print("\nFitting Isolation Forest on clean training data...")
    iso = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    ).fit(X_train)
    # score_samples: higher = more normal, so negate for an anomaly score
    results["isolation_forest"] = evaluate(
        -iso.score_samples(X_test), y_true, types, "Isolation Forest"
    )

    # ---- One-Class SVM ----------------------------------------------------
    # Subsampled: OC-SVM is O(n^2) and 78k training rows is impractical on CPU.
    print("\nFitting One-Class SVM on a 10,000-sequence subsample...")
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(X_train), size=min(10000, len(X_train)), replace=False)
    svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(X_train[idx])
    results["one_class_svm"] = evaluate(
        -svm.score_samples(X_test), y_true, types, "One-Class SVM"
    )

    out = os.path.join(METRICS_DIR, "baseline_results.json")
    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")

    print("\n" + "=" * 60)
    print("Comparison with the deep detectors")
    print("=" * 60)
    try:
        ae = json.load(open(os.path.join(METRICS_DIR, "autoencoder_results.json")))
        print(f"  GRU autoencoder   ROC-AUC {ae['roc_auc']:.4f}  AUPRC {ae['auprc']:.4f}")
    except FileNotFoundError:
        pass
    for k, v in results.items():
        print(f"  {v['method']:17} ROC-AUC {v['roc_auc']:.4f}  AUPRC {v['auprc']:.4f}")


if __name__ == "__main__":
    main()
