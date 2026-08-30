# ============================================================
# 08_evaluate.py
# Thesis: Deep Sequential Models for ERP Anomaly Detection
# Mustafa Wasif Allvi — University of Potsdam, M.Sc. Data Science
# ============================================================
# Final evaluation and latency benchmarking (thesis Section 5.7).
#
# Produces:
#   1. Next-activity prediction on test set (accuracy, F1-macro)
#   2. Anomaly detection metrics from 07_scoring.py results
#   3. Latency and throughput for all models including autoencoder
#   4. Final comparison tables saved to JSON
# ============================================================

import os
import json
import time
import importlib
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve,
)

import config
from config import (
    PREPROCESSED_DIR, MODELS_DIR, METRICS_DIR,
    DEVICE, BATCH_SIZE, NUM_RUNS, MODEL_INPUT_DIM,
    N_CATEGORICAL, N_NUMERICAL,
    ANOMALY_TYPES, LATENCY_WARMUP_RUNS,
    LATENCY_MEASURE_RUNS, THROUGHPUT_BATCH_SIZE,
    THROUGHPUT_BATCHES, MAX_PREFIX_LENGTH,
    FINANCIAL_INDICES_IN_NUMERICAL, FINANCIAL_RECON_WEIGHT,
)

models_module = importlib.import_module("04_models")
build_model   = models_module.build_model

ae_module                  = importlib.import_module("05b_autoencoder")
GRUAutoencoder             = ae_module.GRUAutoencoder
compute_event_level_scores = ae_module.compute_event_level_scores
build_numerical_weights    = ae_module.build_numerical_weights
AE_HIDDEN_DIM              = ae_module.AE_HIDDEN_DIM
AE_NUM_LAYERS              = ae_module.AE_NUM_LAYERS
AE_DROPOUT                 = ae_module.AE_DROPOUT
N_NUMERICAL_AE             = ae_module.N_NUMERICAL
AE_RUNS                    = ae_module.AE_RUNS


# ============================================================
# Dataset
# ============================================================

class SequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets   = torch.tensor(targets,   dtype=torch.long)
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


# ============================================================
# Reconstruction weights
# ============================================================

def build_recon_weights():
    weights = torch.ones(N_NUMERICAL)
    for idx in FINANCIAL_INDICES_IN_NUMERICAL:
        if idx < N_NUMERICAL:
            weights[idx] = FINANCIAL_RECON_WEIGHT
    return weights


# ============================================================
# 1. Prediction evaluation on test set
# ============================================================

def evaluate_prediction(model_name, test_loader, num_classes):
    run_accs = []
    run_f1w  = []
    run_f1m  = []

    for run_idx in range(1, NUM_RUNS + 1):
        weight_path = os.path.join(
            MODELS_DIR, f"{model_name}_run{run_idx}.pth"
        )
        if not os.path.exists(weight_path):
            continue

        model = build_model(model_name, num_classes)
        model.load_state_dict(
            torch.load(weight_path, map_location=DEVICE)
        )
        model.eval()

        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences = sequences.to(DEVICE)
                logits, _ = model(sequences)
                _, preds  = logits.max(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        preds_arr  = np.array(all_preds)
        labels_arr = np.array(all_labels)

        run_accs.append(accuracy_score(labels_arr, preds_arr) * 100)
        run_f1w.append(f1_score(labels_arr, preds_arr,
                                average="weighted", zero_division=0))
        run_f1m.append(f1_score(labels_arr, preds_arr,
                                average="macro", zero_division=0))

    return {
        "model"             : model_name,
        "test_acc_mean"     : float(np.mean(run_accs)),
        "test_acc_std"      : float(np.std(run_accs)),
        "test_f1w_mean"     : float(np.mean(run_f1w)),
        "test_f1m_mean"     : float(np.mean(run_f1m)),
        "test_f1m_std"      : float(np.std(run_f1m)),
        "n_runs"            : len(run_accs),
    }


# ============================================================
# 2. Anomaly detection evaluation
# ============================================================

def evaluate_anomaly(scores, true_binary, anomaly_types, method_name):
    try:
        roc_auc = float(roc_auc_score(true_binary, scores))
    except Exception:
        roc_auc = float("nan")
    try:
        auprc = float(average_precision_score(true_binary, scores))
    except Exception:
        auprc = float("nan")

    baseline_auprc = float(true_binary.mean())

    precisions, recalls, thresholds = precision_recall_curve(
        true_binary, scores
    )
    f1s      = 2*precisions*recalls / (precisions+recalls+1e-8)
    best_idx = np.argmax(f1s)
    threshold = float(thresholds[best_idx])
    predicted = (scores > threshold).astype(int)
    best_f1   = float(f1s[best_idx])

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
        "method"                : method_name,
        "roc_auc"               : roc_auc,
        "auprc"                 : auprc,
        "auprc_random_baseline" : baseline_auprc,
        "best_f1"               : best_f1,
        "threshold"             : threshold,
        "precision_at_threshold": float(precision_at_t),
        "recall_at_threshold"   : float(recall_at_t),
        "tp": tp, "fp": fp, "fn": fn,
        "type_detection"        : type_detection,
    }


# ============================================================
# 3. Latency measurement
# ============================================================

def measure_latency(model, input_dim=MODEL_INPUT_DIM,
                     seq_len=MAX_PREFIX_LENGTH):
    model.eval()
    single = torch.zeros(1, seq_len, input_dim).to(DEVICE)

    with torch.no_grad():
        for _ in range(LATENCY_WARMUP_RUNS):
            _ = model(single)

    latencies = []
    with torch.no_grad():
        for _ in range(LATENCY_MEASURE_RUNS):
            t0 = time.perf_counter()
            _  = model(single)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

    return {
        "latency_mean_ms" : float(np.mean(latencies)),
        "latency_std_ms"  : float(np.std(latencies)),
        "latency_p50_ms"  : float(np.percentile(latencies, 50)),
        "latency_p95_ms"  : float(np.percentile(latencies, 95)),
    }


def measure_throughput(model, input_dim=MODEL_INPUT_DIM,
                        seq_len=MAX_PREFIX_LENGTH):
    model.eval()
    batch = torch.zeros(
        THROUGHPUT_BATCH_SIZE, seq_len, input_dim
    ).to(DEVICE)

    with torch.no_grad():
        for _ in range(5):
            _ = model(batch)

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(THROUGHPUT_BATCHES):
            _ = model(batch)
    t1 = time.perf_counter()

    return float(
        THROUGHPUT_BATCH_SIZE * THROUGHPUT_BATCHES / (t1 - t0)
    )


# ============================================================
# MAIN
# ============================================================

def run_evaluation(save=True):
    config.make_output_dirs()

    print("=" * 60)
    print("08_evaluate.py — Final Evaluation")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    seq_path = os.path.join(PREPROCESSED_DIR, "sequences.joblib")
    data     = joblib.load(seq_path)

    test_sequences = data["test_sequences"]
    test_targets   = data["test_targets"]
    num_classes    = data["num_classes"]
    encoders       = joblib.load(config.ENCODER_PATH)

    test_dataset = SequenceDataset(test_sequences, test_targets)
    test_loader  = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    print(f"  Test sequences: {len(test_sequences):,}")
    print(f"  Num classes:    {num_classes}")

    inj_path      = os.path.join(PREPROCESSED_DIR, "test_injected.joblib")
    inj_data      = joblib.load(inj_path)
    inj_sequences = inj_data["sequences"]
    inj_targets   = inj_data["targets"]
    true_binary   = inj_data["labels"]
    anomaly_types = inj_data["types"]

    inj_dataset = SequenceDataset(inj_sequences, inj_targets)
    inj_loader  = DataLoader(
        inj_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    numerical_weights = build_numerical_weights()
    recon_weights_joint = build_recon_weights()
    mse_fn = nn.MSELoss(reduction="none")

    # --------------------------------------------------------
    # 1. Prediction evaluation
    # --------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("1. NEXT-ACTIVITY PREDICTION ON TEST SET")
    print(f"{'=' * 60}")

    prediction_results = {}
    for model_name in ["lstm", "gru", "transformer"]:
        result = evaluate_prediction(
            model_name, test_loader, num_classes
        )
        prediction_results[model_name] = result
        print(f"\n  {model_name.upper()}:")
        print(f"    Test Acc  : {result['test_acc_mean']:.2f}% "
              f"(std {result['test_acc_std']:.2f}%)")
        print(f"    Test F1m  : {result['test_f1m_mean']:.4f} "
              f"(std {result['test_f1m_std']:.4f})")
        print(f"    Test F1w  : {result['test_f1w_mean']:.4f}")

    print(f"\n  {'Model':<15} {'Test Acc%':<12} {'Std%':<8} "
          f"{'F1 Macro':<12} {'F1 Weighted'}")
    print(f"  {'-' * 56}")
    for name, r in prediction_results.items():
        print(f"  {name.upper():<15} "
              f"{r['test_acc_mean']:<12.2f} "
              f"{r['test_acc_std']:<8.2f} "
              f"{r['test_f1m_mean']:<12.4f} "
              f"{r['test_f1w_mean']:.4f}")

    # --------------------------------------------------------
    # 2. Anomaly detection
    # --------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("2. ANOMALY DETECTION EVALUATION")
    print(f"{'=' * 60}")

    anomaly_results = {}

    def get_mean_numerical_target(sequences):
        num_slice     = sequences[:, :, N_CATEGORICAL:]
        mask          = (sequences.abs().sum(dim=2) > 0).float()
        mask_expanded = mask.unsqueeze(2)
        sum_num       = (num_slice * mask_expanded).sum(dim=1)
        n_real        = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return sum_num / n_real

    for model_name in ["lstm", "gru", "transformer"]:
        run_conf  = []
        run_recon = []

        for run_idx in range(1, NUM_RUNS + 1):
            wp = os.path.join(MODELS_DIR, f"{model_name}_run{run_idx}.pth")
            if not os.path.exists(wp):
                continue
            model = build_model(model_name, num_classes)
            model.load_state_dict(torch.load(wp, map_location=DEVICE))
            model.eval()

            conf_list  = []
            recon_list = []
            with torch.no_grad():
                for sequences, _ in inj_loader:
                    sequences = sequences.to(DEVICE)
                    logits, recon = model(sequences)
                    probs     = torch.softmax(logits, dim=1)
                    max_probs = probs.max(dim=1).values
                    conf      = (1 - max_probs).cpu().numpy()
                    target_num = get_mean_numerical_target(sequences)
                    sq_err     = mse_fn(recon, target_num)
                    weighted   = (sq_err *
                                  recon_weights_joint.to(DEVICE)
                                  ).mean(dim=1)
                    conf_list.extend(conf)
                    recon_list.extend(weighted.cpu().numpy())
            run_conf.append(np.array(conf_list))
            run_recon.append(np.array(recon_list))

        mean_conf  = np.mean(run_conf,  axis=0)
        mean_recon = np.mean(run_recon, axis=0)
        recon_p99  = np.percentile(mean_recon, 99)
        recon_norm = np.clip(mean_recon / recon_p99, 0, 1) \
            if recon_p99 > 0 else mean_recon
        mean_comp  = 0.5 * mean_conf + 0.5 * recon_norm

        conf_res  = evaluate_anomaly(mean_conf,  true_binary,
                                     anomaly_types,
                                     f"{model_name}_confidence")
        recon_res = evaluate_anomaly(recon_norm, true_binary,
                                     anomaly_types,
                                     f"{model_name}_joint_recon")
        comp_res  = evaluate_anomaly(mean_comp,  true_binary,
                                     anomaly_types,
                                     f"{model_name}_composite")

        anomaly_results[f"{model_name}_confidence"]  = conf_res
        anomaly_results[f"{model_name}_joint_recon"] = recon_res
        anomaly_results[f"{model_name}_composite"]   = comp_res

        print(f"\n  {model_name.upper()}:")
        print(f"    Confidence:         ROC {conf_res['roc_auc']:.4f}  "
              f"AUPRC {conf_res['auprc']:.4f}  "
              f"F1 {conf_res['best_f1']:.4f}")
        print(f"    Joint recon:        ROC {recon_res['roc_auc']:.4f}  "
              f"AUPRC {recon_res['auprc']:.4f}  "
              f"F1 {recon_res['best_f1']:.4f}")
        print(f"    Composite:          ROC {comp_res['roc_auc']:.4f}  "
              f"AUPRC {comp_res['auprc']:.4f}  "
              f"F1 {comp_res['best_f1']:.4f}")

    # Separate autoencoder
    print(f"\n  Separate autoencoder (numerical reconstruction):")
    ae_run_scores = []
    for run_idx in range(1, AE_RUNS + 1):
        ae_path = os.path.join(MODELS_DIR, f"autoencoder_run{run_idx}.pth")
        if not os.path.exists(ae_path):
            continue
        ae_model = GRUAutoencoder(
            input_dim   = MODEL_INPUT_DIM,
            n_numerical = N_NUMERICAL_AE,
            hidden_dim  = AE_HIDDEN_DIM,
            num_layers  = AE_NUM_LAYERS,
            dropout     = AE_DROPOUT,
        ).to(DEVICE)
        ae_model.load_state_dict(
            torch.load(ae_path, map_location=DEVICE)
        )
        scores = compute_event_level_scores(
            ae_model, inj_sequences, numerical_weights
        )
        ae_run_scores.append(scores)

    mean_ae = np.mean(ae_run_scores, axis=0)
    ae_res  = evaluate_anomaly(
        mean_ae, true_binary, anomaly_types, "separate_autoencoder"
    )
    anomaly_results["separate_autoencoder"] = ae_res
    print(f"    ROC-AUC:   {ae_res['roc_auc']:.4f}")
    print(f"    AUPRC:     {ae_res['auprc']:.4f}  "
          f"(random: {ae_res['auprc_random_baseline']:.4f})")
    print(f"    Best F1:   {ae_res['best_f1']:.4f}")
    print(f"    Precision: {ae_res['precision_at_threshold']:.4f}")
    print(f"    Recall:    {ae_res['recall_at_threshold']:.4f}")
    print(f"    Per-type detection rates:")
    for atype, det in ae_res["type_detection"].items():
        print(f"      {atype:<20}: {det['rate_pct']:.1f}%  "
              f"({det['detected']}/{det['total']})")

    # --------------------------------------------------------
    # 3. Latency
    # --------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("3. LATENCY AND THROUGHPUT")
    print(f"{'=' * 60}")

    latency_results = {}
    for model_name in ["lstm", "gru", "transformer"]:
        wp = os.path.join(MODELS_DIR, f"{model_name}_run1.pth")
        if not os.path.exists(wp):
            continue
        model = build_model(model_name, num_classes)
        model.load_state_dict(torch.load(wp, map_location=DEVICE))
        model.eval()

        lat    = measure_latency(model)
        thr    = measure_throughput(model)
        params = sum(p.numel() for p in model.parameters())

        latency_results[model_name] = {
            **lat,
            "throughput_events_per_sec": thr,
            "parameters"               : params,
        }
        print(f"\n  {model_name.upper()}:")
        print(f"    Latency mean: {lat['latency_mean_ms']:.3f} ms  "
              f"p95: {lat['latency_p95_ms']:.3f} ms")
        print(f"    Throughput:   {thr:,.0f} events/sec")
        print(f"    Parameters:   {params:,}")

    ae_model = GRUAutoencoder(
        input_dim   = MODEL_INPUT_DIM,
        n_numerical = N_NUMERICAL_AE,
        hidden_dim  = AE_HIDDEN_DIM,
        num_layers  = AE_NUM_LAYERS,
        dropout     = AE_DROPOUT,
    ).to(DEVICE)
    ae_path = os.path.join(MODELS_DIR, "autoencoder_run1.pth")
    if os.path.exists(ae_path):
        ae_model.load_state_dict(
            torch.load(ae_path, map_location=DEVICE)
        )
        ae_model.eval()
        ae_lat = measure_latency(ae_model)
        ae_thr = measure_throughput(ae_model)
        ae_params = sum(p.numel() for p in ae_model.parameters())
        latency_results["autoencoder"] = {
            **ae_lat,
            "throughput_events_per_sec": ae_thr,
            "parameters"               : ae_params,
        }
        print(f"\n  AUTOENCODER:")
        print(f"    Latency mean: {ae_lat['latency_mean_ms']:.3f} ms  "
              f"p95: {ae_lat['latency_p95_ms']:.3f} ms")
        print(f"    Throughput:   {ae_thr:,.0f} events/sec")
        print(f"    Parameters:   {ae_params:,}")

    print(f"\n  {'Model':<20} {'Latency(ms)':<14} {'p95(ms)':<12} "
          f"{'Events/sec':<18} {'Params'}")
    print(f"  {'-' * 72}")
    for name, r in latency_results.items():
        print(f"  {name.upper():<20} "
              f"{r['latency_mean_ms']:<14.3f} "
              f"{r['latency_p95_ms']:<12.3f} "
              f"{r['throughput_events_per_sec']:<18,.0f} "
              f"{r['parameters']:,}")

    # Save
    all_results = {
        "prediction" : prediction_results,
        "anomaly"    : anomaly_results,
        "latency"    : latency_results,
    }

    def to_serialisable(obj):
        if isinstance(obj, (np.floating, float)): return float(obj)
        if isinstance(obj, (np.integer, int)):    return int(obj)
        if isinstance(obj, np.ndarray):           return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serialisable(i) for i in obj]
        return obj

    if save:
        path = os.path.join(METRICS_DIR, "final_evaluation.json")
        with open(path, "w") as f:
            json.dump(to_serialisable(all_results), f, indent=2)
        print(f"\n  All results saved: {path}")

    print("\n  08_evaluate.py complete.")
    print("=" * 60)
    return all_results


if __name__ == "__main__":
    run_evaluation(save=True)