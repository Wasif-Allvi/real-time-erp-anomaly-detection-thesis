# ============================================================
# 07_scoring.py
# Thesis: Deep Sequential Models for ERP Anomaly Detection
# Mustafa Wasif Allvi — University of Potsdam, M.Sc. Data Science
# ============================================================
# Composite anomaly scoring (thesis Section 5.6).
#
# For each trained model (LSTM, GRU, Transformer):
#   1. Load best weights from 05_train.py
#   2. Run inference on injected test sequences
#   3. Confidence score = 1 - max(softmax(logits))
#   4. Reconstruction error = weighted MSE(recon, numerical mean)
#   5. Composite = alpha * confidence + beta * recon_norm
#   6. Grid search over alpha/beta
#   7. Per-type detection rates at optimal threshold
# ============================================================

import os
import json
import importlib
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_recall_curve, accuracy_score,
)

import config
from config import (
    PREPROCESSED_DIR, MODELS_DIR, METRICS_DIR,
    DEVICE, BATCH_SIZE, COMPOSITE_ALPHA, COMPOSITE_BETA,
    ALPHA_GRID, MODEL_INPUT_DIM, N_CATEGORICAL, N_NUMERICAL,
    ANOMALY_TYPES, NUM_RUNS,
    FINANCIAL_INDICES_IN_NUMERICAL, FINANCIAL_RECON_WEIGHT,
)

models_module = importlib.import_module("04_models")
build_model   = models_module.build_model


# ============================================================
# Dataset
# ============================================================

class TestDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets   = torch.tensor(targets,   dtype=torch.long)
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


# ============================================================
# Reconstruction weights — same as training
# ============================================================

def build_recon_weights():
    weights = torch.ones(N_NUMERICAL)
    for idx in FINANCIAL_INDICES_IN_NUMERICAL:
        if idx < N_NUMERICAL:
            weights[idx] = FINANCIAL_RECON_WEIGHT
    return weights


# ============================================================
# Get mean numerical target (same as 05_train.py)
# ============================================================

def get_mean_numerical_target(sequences):
    num_slice     = sequences[:, :, N_CATEGORICAL:]
    mask          = (sequences.abs().sum(dim=2) > 0).float()
    mask_expanded = mask.unsqueeze(2)
    sum_num       = (num_slice * mask_expanded).sum(dim=1)
    n_real        = mask.sum(dim=1, keepdim=True).clamp(min=1)
    return sum_num / n_real


# ============================================================
# Compute scores from one model
# ============================================================

def compute_scores(model, loader, recon_weights, alpha, beta):
    model.eval()
    mse_fn = nn.MSELoss(reduction="none")
    confidence_scores = []
    recon_errors      = []
    predictions       = []
    true_labels       = []

    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(DEVICE)

            logits, recon = model(sequences)

            probs     = torch.softmax(logits, dim=1)
            max_probs = probs.max(dim=1).values
            conf      = (1 - max_probs).cpu().numpy()

            target_num = get_mean_numerical_target(sequences)
            sq_err     = mse_fn(recon, target_num)
            weighted   = (sq_err * recon_weights.to(DEVICE)).mean(dim=1)
            recon_err  = weighted.cpu().numpy()

            _, preds = logits.max(dim=1)
            confidence_scores.extend(conf)
            recon_errors.extend(recon_err)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.numpy())

    confidence  = np.array(confidence_scores)
    recon_error = np.array(recon_errors)

    recon_p99  = np.percentile(recon_error, 99)
    recon_norm = np.clip(recon_error / recon_p99, 0, 1) \
        if recon_p99 > 0 else recon_error

    composite = alpha * confidence + beta * recon_norm

    return {
        "confidence"  : confidence,
        "recon_error" : recon_error,
        "recon_norm"  : recon_norm,
        "composite"   : composite,
        "predictions" : np.array(predictions),
        "true_labels" : np.array(true_labels),
    }


# ============================================================
# Optimal threshold via F1 maximisation
# ============================================================

def find_threshold(scores, true_binary):
    precisions, recalls, thresholds = precision_recall_curve(
        true_binary, scores
    )
    f1s      = 2*precisions*recalls / (precisions+recalls+1e-8)
    best_idx = np.argmax(f1s)
    return float(thresholds[best_idx]), float(f1s[best_idx])


# ============================================================
# Evaluate one score array
# ============================================================

def evaluate_scores(scores, true_binary, anomaly_types, score_name):
    threshold, best_f1 = find_threshold(scores, true_binary)
    predicted = (scores > threshold).astype(int)

    try:
        roc_auc = float(roc_auc_score(true_binary, scores))
    except Exception:
        roc_auc = float("nan")

    try:
        auprc = float(average_precision_score(true_binary, scores))
    except Exception:
        auprc = float("nan")

    baseline_auprc = float(true_binary.mean())

    f1_w = float(f1_score(true_binary, predicted,
                           average="weighted", zero_division=0))
    f1_m = float(f1_score(true_binary, predicted,
                           average="macro",    zero_division=0))

    tp = int(((predicted == 1) & (true_binary == 1)).sum())
    fp = int(((predicted == 1) & (true_binary == 0)).sum())
    fn = int(((predicted == 0) & (true_binary == 1)).sum())
    precision_at_t = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_at_t    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    type_detection = {}
    for atype in ANOMALY_TYPES:
        mask     = np.array([t == atype for t in anomaly_types])
        if mask.sum() == 0:
            continue
        detected = int((scores[mask] > threshold).sum())
        total    = int(mask.sum())
        type_detection[atype] = {
            "detected" : detected,
            "total"    : total,
            "rate_pct" : detected / total * 100,
        }

    return {
        "score_name"            : score_name,
        "roc_auc"               : roc_auc,
        "auprc"                 : auprc,
        "auprc_random_baseline" : baseline_auprc,
        "best_f1"               : best_f1,
        "threshold"             : threshold,
        "precision_at_threshold": float(precision_at_t),
        "recall_at_threshold"   : float(recall_at_t),
        "tp"                    : tp,
        "fp"                    : fp,
        "fn"                    : fn,
        "f1_weighted"           : f1_w,
        "f1_macro"              : f1_m,
        "type_detection"        : type_detection,
    }


# ============================================================
# Grid search over alpha
# ============================================================

def grid_search_alpha(confidence, recon_norm, true_binary):
    best_roc   = -1.0
    best_alpha = COMPOSITE_ALPHA
    results    = {}

    for alpha in ALPHA_GRID:
        beta      = round(1.0 - alpha, 2)
        composite = alpha * confidence + beta * recon_norm
        try:
            roc = float(roc_auc_score(true_binary, composite))
        except Exception:
            roc = 0.0
        results[alpha] = roc
        if roc > best_roc:
            best_roc   = roc
            best_alpha = alpha

    return {
        "grid"       : results,
        "best_alpha" : best_alpha,
        "best_beta"  : round(1.0 - best_alpha, 2),
        "best_roc"   : best_roc,
    }


# ============================================================
# MAIN
# ============================================================

def run_scoring(save=True):
    config.make_output_dirs()

    print("=" * 60)
    print("07_scoring.py — Composite Anomaly Scoring")
    print("=" * 60)

    print("\nLoading injected test sequences...")
    inj_path = os.path.join(PREPROCESSED_DIR, "test_injected.joblib")
    inj_data = joblib.load(inj_path)

    sequences     = inj_data["sequences"]
    targets       = inj_data["targets"]
    true_binary   = inj_data["labels"]
    anomaly_types = inj_data["types"]

    print(f"  Total sequences:     {len(sequences):,}")
    print(f"  Anomalous sequences: {true_binary.sum():,}")
    print(f"  Normal sequences:    {(true_binary == 0).sum():,}")
    print(f"  Anomaly rate:        {true_binary.mean()*100:.1f}%")

    encoders    = joblib.load(config.ENCODER_PATH)
    num_classes = len(encoders["concept:name"].classes_)
    recon_weights = build_recon_weights()

    dataset = TestDataset(sequences, targets)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_model_results = {}
    model_names       = ["lstm", "gru", "transformer"]

    for model_name in model_names:
        print(f"\n{'=' * 60}")
        print(f"Scoring {model_name.upper()}")
        print(f"{'=' * 60}")

        run_confidences = []
        run_recon_norms = []

        for run_idx in range(1, NUM_RUNS + 1):
            weight_path = os.path.join(
                MODELS_DIR, f"{model_name}_run{run_idx}.pth"
            )
            if not os.path.exists(weight_path):
                print(f"  WARNING: not found: {weight_path}")
                continue

            model = build_model(model_name, num_classes)
            model.load_state_dict(
                torch.load(weight_path, map_location=DEVICE)
            )
            model.eval()
            print(f"  Run {run_idx}: loaded weights")

            scores = compute_scores(
                model, loader, recon_weights,
                COMPOSITE_ALPHA, COMPOSITE_BETA
            )
            run_confidences.append(scores["confidence"])
            run_recon_norms.append(scores["recon_norm"])

        mean_confidence = np.mean(run_confidences, axis=0)
        mean_recon_norm = np.mean(run_recon_norms,  axis=0)
        mean_composite  = (COMPOSITE_ALPHA * mean_confidence
                           + COMPOSITE_BETA * mean_recon_norm)

        print(f"\n  Score distributions (mean across {NUM_RUNS} runs):")
        print(f"    Confidence  — "
              f"normal: {mean_confidence[true_binary==0].mean():.4f}  "
              f"anomaly: {mean_confidence[true_binary==1].mean():.4f}  "
              f"gap: {mean_confidence[true_binary==1].mean()-mean_confidence[true_binary==0].mean():+.4f}")
        print(f"    Recon error — "
              f"normal: {mean_recon_norm[true_binary==0].mean():.4f}  "
              f"anomaly: {mean_recon_norm[true_binary==1].mean():.4f}  "
              f"gap: {mean_recon_norm[true_binary==1].mean()-mean_recon_norm[true_binary==0].mean():+.4f}")
        print(f"    Composite   — "
              f"normal: {mean_composite[true_binary==0].mean():.4f}  "
              f"anomaly: {mean_composite[true_binary==1].mean():.4f}  "
              f"gap: {mean_composite[true_binary==1].mean()-mean_composite[true_binary==0].mean():+.4f}")

        print(f"\n  Evaluation (alpha={COMPOSITE_ALPHA}, "
              f"beta={COMPOSITE_BETA}):")

        conf_result  = evaluate_scores(
            mean_confidence, true_binary,
            anomaly_types, "confidence_only"
        )
        recon_result = evaluate_scores(
            mean_recon_norm, true_binary,
            anomaly_types, "reconstruction_only"
        )
        comp_result  = evaluate_scores(
            mean_composite, true_binary,
            anomaly_types, "composite"
        )

        for res in [conf_result, recon_result, comp_result]:
            print(f"\n    [{res['score_name']}]")
            print(f"      ROC-AUC   : {res['roc_auc']:.4f}")
            print(f"      AUPRC     : {res['auprc']:.4f}  "
                  f"(random: {res['auprc_random_baseline']:.4f})")
            print(f"      Best F1   : {res['best_f1']:.4f}")
            print(f"      Precision : {res['precision_at_threshold']:.4f}")
            print(f"      Recall    : {res['recall_at_threshold']:.4f}")
            print(f"      Per-type detection rates:")
            for atype, det in res["type_detection"].items():
                print(f"        {atype:<20}: {det['rate_pct']:.1f}%  "
                      f"({det['detected']}/{det['total']})")

        print(f"\n  Grid search (alpha/beta):")
        grid_result = grid_search_alpha(
            mean_confidence, mean_recon_norm, true_binary
        )
        print(f"    Best alpha : {grid_result['best_alpha']}")
        print(f"    Best beta  : {grid_result['best_beta']}")
        print(f"    Best ROC   : {grid_result['best_roc']:.4f}")
        for alpha, roc in grid_result["grid"].items():
            marker = " <-- best" if alpha == grid_result["best_alpha"] \
                else ""
            print(f"      alpha={alpha:.1f} beta={1-alpha:.1f} "
                  f"ROC={roc:.4f}{marker}")

        all_model_results[model_name] = {
            "confidence_result"   : conf_result,
            "recon_result"        : recon_result,
            "composite_result"    : comp_result,
            "grid_search"         : grid_result,
            "score_distributions" : {
                "conf_normal"  : float(mean_confidence[true_binary==0].mean()),
                "conf_anomaly" : float(mean_confidence[true_binary==1].mean()),
                "recon_normal" : float(mean_recon_norm[true_binary==0].mean()),
                "recon_anomaly": float(mean_recon_norm[true_binary==1].mean()),
                "comp_normal"  : float(mean_composite[true_binary==0].mean()),
                "comp_anomaly" : float(mean_composite[true_binary==1].mean()),
            },
        }

    print(f"\n{'=' * 60}")
    print("CROSS-MODEL COMPARISON")
    print(f"{'=' * 60}")
    print(f"  {'Model':<15} {'Conf ROC':<12} {'Recon ROC':<12} "
          f"{'Comp ROC':<12} {'AUPRC':<10} {'Best Alpha'}")
    print(f"  {'-' * 65}")
    for model_name in model_names:
        res = all_model_results[model_name]
        print(f"  {model_name.upper():<15} "
              f"{res['confidence_result']['roc_auc']:<12.4f} "
              f"{res['recon_result']['roc_auc']:<12.4f} "
              f"{res['composite_result']['roc_auc']:<12.4f} "
              f"{res['composite_result']['auprc']:<10.4f} "
              f"{res['grid_search']['best_alpha']}")

    if save:
        def to_serialisable(obj):
            if isinstance(obj, (np.floating, float)): return float(obj)
            if isinstance(obj, (np.integer, int)):    return int(obj)
            if isinstance(obj, np.ndarray):           return obj.tolist()
            if isinstance(obj, dict):
                return {k: to_serialisable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_serialisable(i) for i in obj]
            return obj

        path = os.path.join(METRICS_DIR, "scoring_results.json")
        with open(path, "w") as f:
            json.dump(to_serialisable(all_model_results), f, indent=2)
        print(f"\n  Results saved: {path}")

    print("\n  07_scoring.py complete.")
    print("=" * 60)
    return all_model_results


if __name__ == "__main__":
    run_scoring(save=True)