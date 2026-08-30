# ============================================================
# 05b_autoencoder.py
# Thesis: Deep Sequential Models for ERP Anomaly Detection
# Mustafa Wasif Allvi — University of Potsdam, M.Sc. Data Science
# ============================================================
# Option B: Separate GRU autoencoder trained on NORMAL sequences.
# Reconstructs NUMERICAL features only (indices 7-13).
# Categorical features (indices 0-6) are used as encoder input
# but excluded from the reconstruction loss and scoring.
#
# This is the correct design because:
#   - Categorical features are label-encoded integers (0-825)
#     which cannot be meaningfully reconstructed by MSE
#   - Numerical features are min-max scaled to [0,1]
#     which Sigmoid output matches exactly
#   - Financial anomalies manifest in numerical features
#     so numerical-only reconstruction targets the right signal
#
# Reference: Chandola et al. (2009) \cite{Chandola2009}
# ============================================================

import os
import time
import json
import random
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve
)

import config
from config import (
    PREPROCESSED_DIR, MODELS_DIR, METRICS_DIR,
    DEVICE, BATCH_SIZE, RANDOM_SEED, LOG_INTERVAL,
    MODEL_INPUT_DIM, ANOMALY_TYPES,
)

# ------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------
AE_HIDDEN_DIM = 64
AE_NUM_LAYERS = 2
AE_DROPOUT    = 0.1
AE_EPOCHS     = 50
AE_LR         = 1e-3
AE_RUNS       = 3

# Feature index ranges (zero-based from model_features list):
#   0-6:  categorical encoded integers (encoder input only)
#   7-13: numerical scaled [0,1] (encoder input + reconstruction target)
#
# Full list:
#   0: concept:name_enc
#   1: org:resource_enc
#   2: org:role_enc
#   3: case:Permit OrganizationalEntity_enc
#   4: case:Permit ProjectNumber_enc
#   5: case:Permit BudgetNumber_enc
#   6: case:BudgetNumber_enc
#   7: case:Amount_scaled                  <- numerical, financial
#   8: case:RequestedAmount_scaled         <- numerical, financial
#   9: case:AdjustedAmount_scaled          <- numerical, financial
#  10: case:OriginalAmount_scaled          <- numerical, financial
#  11: case:Permit RequestedBudget_scaled  <- numerical, financial
#  12: delta_t_scaled                      <- numerical, temporal
#  13: cum_dur_scaled                      <- numerical, temporal

N_CATEGORICAL = 7    # features 0-6: encoder input only
N_NUMERICAL   = 7    # features 7-13: encoder input + reconstruction target

# Financial feature indices within the NUMERICAL slice (0-based within slice)
# Slice indices 0-4 correspond to global indices 7-11
FINANCIAL_INDICES_IN_SLICE = [0, 1, 2, 3, 4]
FINANCIAL_WEIGHT           = 3.0


# ============================================================
# Dataset
# ============================================================

class NormalSequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


# ============================================================
# GRU Autoencoder — numerical features only reconstruction
# ============================================================

class GRUAutoencoder(nn.Module):
    """
    GRU autoencoder that reconstructs numerical features only.

    Encoder input : full 14-feature vector (categorical + numerical)
    Decoder input : LayerNorm(encoder_outputs)
    Reconstruction: numerical features only (7 values, indices 7-13)
    Output bounds : Sigmoid clamps reconstruction to [0,1]
                    matching min-max scaled numerical inputs

    This design is correct because:
    - Categorical integers (0-825) cannot be reconstructed by MSE
    - Numerical features are all in [0,1] range
    - Financial anomalies appear in numerical features
    """
    def __init__(self, input_dim: int, n_numerical: int,
                 hidden_dim: int, num_layers: int,
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim   = input_dim
        self.n_numerical = n_numerical
        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers

        # Encoder sees full feature vector
        self.encoder = nn.GRU(
            input_size  = input_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        self.enc_norm = nn.LayerNorm(hidden_dim)

        # Decoder also sees full context but outputs only numerical
        self.decoder = nn.GRU(
            input_size  = hidden_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        # Output: reconstruct only the n_numerical features
        # Sigmoid bounds to [0,1] matching min-max scaled inputs
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, n_numerical),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        """
        x      : (B, L, input_dim=14)
        returns: (B, L, n_numerical=7) — reconstructed numerical features
        """
        encoder_outputs, h_context = self.encoder(x)
        enc_norm    = self.enc_norm(encoder_outputs)
        decoder_out, _ = self.decoder(enc_norm, h_context)
        recon       = self.output_proj(decoder_out)   # (B, L, 7)
        return recon


# ============================================================
# Feature weights for numerical slice
# ============================================================

def build_numerical_weights() -> torch.Tensor:
    """
    Weight vector for the 7 numerical features.
    Financial features (slice indices 0-4) weighted 3x.
    Temporal features (slice indices 5-6) weighted 1x.
    """
    weights = torch.ones(N_NUMERICAL)
    for idx in FINANCIAL_INDICES_IN_SLICE:
        weights[idx] = FINANCIAL_WEIGHT
    return weights


# ============================================================
# Masked weighted MSE on numerical features only
# ============================================================

def masked_numerical_mse(recon: torch.Tensor,
                          target_numerical: torch.Tensor,
                          weights: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    ----------
    recon            : (B, L, N_NUMERICAL)
    target_numerical : (B, L, N_NUMERICAL) — numerical slice of input
    weights          : (N_NUMERICAL,)
    mask             : (B, L) — 1 for real timesteps, 0 for padding

    Returns
    -------
    scalar loss
    """
    sq_err       = (recon - target_numerical) ** 2 * weights.to(recon.device)
    mse_per_step = sq_err.mean(dim=2)                  # (B, L)
    masked_sum   = (mse_per_step * mask).sum()
    n_real       = mask.sum()
    if n_real == 0:
        return torch.tensor(0.0, requires_grad=True, device=recon.device)
    return masked_sum / n_real


# ============================================================
# Train one run
# ============================================================

def train_one_run(run_idx: int,
                  train_loader: DataLoader,
                  numerical_weights: torch.Tensor) -> dict:
    random.seed(RANDOM_SEED + run_idx + 100)
    np.random.seed(RANDOM_SEED + run_idx + 100)
    torch.manual_seed(RANDOM_SEED + run_idx + 100)

    model = GRUAutoencoder(
        input_dim   = MODEL_INPUT_DIM,
        n_numerical = N_NUMERICAL,
        hidden_dim  = AE_HIDDEN_DIM,
        num_layers  = AE_NUM_LAYERS,
        dropout     = AE_DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=AE_LR)
    history   = {"train_loss": [], "epoch_times": []}

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Run {run_idx + 1}/{AE_RUNS} — "
          f"GRUAutoencoder ({total_params:,} params)")
    print(f"  Reconstructing {N_NUMERICAL} numerical features only")
    print(f"  {'Epoch':<8} {'Loss':<14} {'Time(s)'}")
    print("  " + "-" * 32)

    for epoch in range(1, AE_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for x in train_loader:
            x = x.to(DEVICE)

            # Extract numerical target and real-timestep mask
            target_num = x[:, :, N_CATEGORICAL:]   # (B, L, 7) indices 7-13
            mask       = (x.abs().sum(dim=2) > 0).float()   # (B, L)

            optimizer.zero_grad()
            recon = model(x)
            loss  = masked_numerical_mse(
                recon, target_num, numerical_weights, mask
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()
            total_loss += loss.item()

        avg_loss   = total_loss / len(train_loader)
        epoch_time = time.time() - t0
        history["train_loss"].append(avg_loss)
        history["epoch_times"].append(epoch_time)

        if epoch % LOG_INTERVAL == 0 or epoch == 1 or epoch == AE_EPOCHS:
            print(f"  {epoch:<8} {avg_loss:<14.6f} {epoch_time:.1f}")

    print("  " + "-" * 32)
    print(f"  Final loss: {history['train_loss'][-1]:.6f}")
    return {"model": model, "history": history, "run_idx": run_idx}


# ============================================================
# Event-level reconstruction scoring
# ============================================================

def compute_event_level_scores(model: nn.Module,
                                sequences: np.ndarray,
                                numerical_weights: torch.Tensor) -> np.ndarray:
    """
    Compute weighted mean numerical reconstruction MSE per sequence.
    Only real (non-padded) timesteps contribute to the score.
    """
    model.eval()
    all_scores  = []
    tensor_data = torch.tensor(sequences, dtype=torch.float32)
    loader      = DataLoader(tensor_data, batch_size=BATCH_SIZE,
                             shuffle=False)

    with torch.no_grad():
        for batch in loader:
            batch      = batch.to(DEVICE)
            recon      = model(batch)                       # (B, L, 7)
            target_num = batch[:, :, N_CATEGORICAL:]       # (B, L, 7)
            mask       = (batch.abs().sum(dim=2) > 0).float()

            sq_err       = (recon - target_num) ** 2 * numerical_weights.to(DEVICE)
            mse_per_step = sq_err.mean(dim=2)
            masked_sum   = (mse_per_step * mask).sum(dim=1)
            n_real       = mask.sum(dim=1).clamp(min=1)
            score        = masked_sum / n_real

            all_scores.extend(score.cpu().numpy())

    return np.array(all_scores)


# ============================================================
# MAIN
# ============================================================

def run_autoencoder(save: bool = True) -> dict:
    config.make_output_dirs()

    print("=" * 60)
    print("05b_autoencoder.py — Separate Autoencoder (Option B)")
    print("=" * 60)
    print(f"  Architecture    : GRU seq2seq + LayerNorm + Sigmoid")
    print(f"  Encoder input   : {MODEL_INPUT_DIM} features (full)")
    print(f"  Recon target    : {N_NUMERICAL} numerical features only")
    print(f"  Hidden dim      : {AE_HIDDEN_DIM}")
    print(f"  Epochs          : {AE_EPOCHS}")
    print(f"  Runs            : {AE_RUNS}")
    print(f"  Training on     : NORMAL sequences only")
    print(f"  Financial weight: {FINANCIAL_WEIGHT}x")

    numerical_weights = build_numerical_weights()
    print(f"  Numerical weights: {numerical_weights.numpy()}")

    print("\nLoading training sequences...")
    seq_path        = os.path.join(PREPROCESSED_DIR, "sequences.joblib")
    data            = joblib.load(seq_path)
    train_sequences = data["train_sequences"]
    print(f"  Normal train sequences: {len(train_sequences):,}")
    print(f"  Shape: {train_sequences.shape}")

    train_dataset = NormalSequenceDataset(train_sequences)
    train_loader  = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    print(f"  Batches per epoch: {len(train_loader)}")

    ae_results  = []
    total_start = time.time()

    for run_idx in range(AE_RUNS):
        result = train_one_run(run_idx, train_loader, numerical_weights)
        ae_results.append(result)
        if save:
            ae_path = os.path.join(
                MODELS_DIR, f"autoencoder_run{run_idx + 1}.pth"
            )
            torch.save(result["model"].state_dict(), ae_path)
            print(f"  Saved: {ae_path}")

    total_elapsed = (time.time() - total_start) / 60
    print(f"\n  All {AE_RUNS} runs complete in {total_elapsed:.1f} minutes")

    print("\nEvaluating on injected test sequences...")
    inj_path       = os.path.join(PREPROCESSED_DIR, "test_injected.joblib")
    inj_data       = joblib.load(inj_path)
    test_sequences = inj_data["sequences"]
    true_binary    = inj_data["labels"]
    anomaly_types  = inj_data["types"]

    run_scores = []
    for run_idx, result in enumerate(ae_results):
        scores = compute_event_level_scores(
            result["model"], test_sequences, numerical_weights
        )
        run_scores.append(scores)
        gap = scores[true_binary==1].mean() - scores[true_binary==0].mean()
        print(f"  Run {run_idx + 1} — "
              f"normal: {scores[true_binary==0].mean():.4f}  "
              f"anomaly: {scores[true_binary==1].mean():.4f}  "
              f"gap: {gap:+.4f}")

    mean_scores  = np.mean(run_scores, axis=0)
    normal_mean  = float(mean_scores[true_binary == 0].mean())
    anomaly_mean = float(mean_scores[true_binary == 1].mean())
    gap          = anomaly_mean - normal_mean

    print(f"\n  Mean across {AE_RUNS} runs:")
    print(f"    Normal mean  : {normal_mean:.4f}")
    print(f"    Anomaly mean : {anomaly_mean:.4f}")
    print(f"    Gap          : {gap:+.4f}")

    try:
        roc_auc = float(roc_auc_score(true_binary, mean_scores))
    except Exception:
        roc_auc = float("nan")

    try:
        auprc = float(average_precision_score(true_binary, mean_scores))
    except Exception:
        auprc = float("nan")

    baseline_auprc = float(true_binary.mean())

    print(f"    ROC-AUC      : {roc_auc:.4f}")
    print(f"    AUPRC        : {auprc:.4f}  "
          f"(random baseline: {baseline_auprc:.4f})")

    precisions, recalls, thresholds = precision_recall_curve(
        true_binary, mean_scores
    )
    f1s       = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx  = np.argmax(f1s)
    threshold = float(thresholds[best_idx])
    predicted = (mean_scores > threshold).astype(int)
    best_f1   = float(f1s[best_idx])

    tp = int(((predicted == 1) & (true_binary == 1)).sum())
    fp = int(((predicted == 1) & (true_binary == 0)).sum())
    fn = int(((predicted == 0) & (true_binary == 1)).sum())
    precision_at_t = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_at_t    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print(f"    Best F1      : {best_f1:.4f}")
    print(f"    Threshold    : {threshold:.4f}")
    print(f"    Precision    : {precision_at_t:.4f}")
    print(f"    Recall       : {recall_at_t:.4f}")
    print(f"    TP={tp}  FP={fp}  FN={fn}")

    print(f"\n    Per-type detection rates:")
    type_detection = {}
    for atype in ANOMALY_TYPES:
        mask     = np.array([t == atype for t in anomaly_types])
        if mask.sum() == 0:
            continue
        detected = int((mean_scores[mask] > threshold).sum())
        total    = int(mask.sum())
        rate     = detected / total * 100
        type_detection[atype] = {
            "detected": detected, "total": total, "rate_pct": rate
        }
        print(f"      {atype:<20}: {rate:.1f}%  ({detected}/{total})")

    eval_results = {
        "roc_auc"               : roc_auc,
        "auprc"                 : auprc,
        "auprc_random_baseline" : baseline_auprc,
        "best_f1"               : best_f1,
        "threshold"             : threshold,
        "precision_at_threshold": precision_at_t,
        "recall_at_threshold"   : recall_at_t,
        "normal_mean"           : normal_mean,
        "anomaly_mean"          : anomaly_mean,
        "gap"                   : gap,
        "type_detection"        : type_detection,
    }

    if save:
        path = os.path.join(METRICS_DIR, "autoencoder_results.json")
        with open(path, "w") as f:
            json.dump(eval_results, f, indent=2)
        print(f"\n  Saved: {path}")

        with open(os.path.join(
                METRICS_DIR, "autoencoder_train_history.json"), "w") as f:
            json.dump(
                [{"run_idx": r["run_idx"], "history": r["history"]}
                 for r in ae_results], f, indent=2
            )

    print("\n  05b_autoencoder.py complete.")
    print("=" * 60)
    return {"ae_results": ae_results, "eval": eval_results}


if __name__ == "__main__":
    run_autoencoder(save=True)