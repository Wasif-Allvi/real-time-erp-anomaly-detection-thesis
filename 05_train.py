# ============================================================
# 05_train.py
# Thesis: Deep Sequential Models for ERP Anomaly Detection
# Mustafa Wasif Allvi — University of Potsdam, M.Sc. Data Science
# ============================================================
# Training pipeline with validation and early stopping.
#
# Key design decisions:
#   - No fixed epoch count — trains until early stopping fires
#   - Early stopping monitors VALIDATION loss (patience=15 epochs)
#   - Best model weights saved at epoch with lowest val loss
#   - Val metrics (loss, accuracy, F1-macro) logged every epoch
#   - LR scheduler reduces LR when val loss plateaus
#   - Reconstruction target: numerical features only (indices 7-13)
#   - Reconstruction output bounded to [0,1] via Sigmoid
#
# Convergence scenarios detected automatically:
#   A. Good: train loss down, val loss down then plateau -> stop
#   B. Overfit: train loss down, val loss up -> early stop saves best
#   C. Undertraining: both still decreasing -> MAX_EPOCHS ceiling
# ============================================================

import os
import time
import json
import random
import importlib
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

import config
from config import (
    PREPROCESSED_DIR, MODELS_DIR, METRICS_DIR,
    DEVICE, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    NUM_RUNS, RANDOM_SEED, LOG_INTERVAL,
    MODEL_INPUT_DIM, N_CATEGORICAL, N_NUMERICAL,
    LAMBDA_RECON, FINANCIAL_INDICES_IN_NUMERICAL,
    FINANCIAL_RECON_WEIGHT, USE_LR_SCHEDULER,
    LR_PATIENCE, LR_FACTOR, LR_MIN,
)

models_module = importlib.import_module("04_models")
build_model   = models_module.build_model

# Early stopping: stop when val loss does not improve for this many epochs
EARLY_STOPPING_PATIENCE = 15

# Safety ceiling — early stopping fires well before this in practice
MAX_EPOCHS = 300


# ============================================================
# Reconstruction feature weights
# ============================================================

def build_recon_weights() -> torch.Tensor:
    weights = torch.ones(N_NUMERICAL)
    for idx in FINANCIAL_INDICES_IN_NUMERICAL:
        if idx < N_NUMERICAL:
            weights[idx] = FINANCIAL_RECON_WEIGHT
    return weights


# ============================================================
# Dataset
# ============================================================

class ERPSequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets   = torch.tensor(targets,   dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


# ============================================================
# Seed
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ============================================================
# Numerical reconstruction target
# ============================================================

def get_mean_numerical_target(sequences: torch.Tensor) -> torch.Tensor:
    """
    Mean of numerical features (indices 7-13) over real timesteps.
    All values in [0,1] — MSE is bounded, no explosion possible.
    """
    num_slice     = sequences[:, :, N_CATEGORICAL:]        # (B, L, 7)
    mask          = (sequences.abs().sum(dim=2) > 0).float()  # (B, L)
    mask_expanded = mask.unsqueeze(2)
    sum_num       = (num_slice * mask_expanded).sum(dim=1)
    n_real        = mask.sum(dim=1, keepdim=True).clamp(min=1)
    return sum_num / n_real                                # (B, 7)


# ============================================================
# One training epoch
# ============================================================

def train_epoch(model, loader, optimizer, ce_loss, mse_loss,
                recon_weights):
    model.train()
    total_loss = total_ce = total_recon = 0.0
    correct = total = 0

    for sequences, labels in loader:
        sequences = sequences.to(DEVICE)
        labels    = labels.to(DEVICE)

        optimizer.zero_grad()
        logits, recon = model(sequences)

        loss_ce    = ce_loss(logits, labels)
        target_num = get_mean_numerical_target(sequences)
        sq_err     = mse_loss(recon, target_num)
        loss_recon = (sq_err * recon_weights.to(DEVICE)).mean()
        loss       = loss_ce + LAMBDA_RECON * loss_recon

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss  += loss.item()
        total_ce    += loss_ce.item()
        total_recon += loss_recon.item()

        _, predicted = logits.max(dim=1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    n = len(loader)
    return total_loss/n, total_ce/n, total_recon/n, 100.0*correct/total


# ============================================================
# One validation epoch
# ============================================================

def val_epoch(model, loader, ce_loss, mse_loss, recon_weights):
    model.eval()
    total_loss = total_ce = total_recon = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(DEVICE)
            labels    = labels.to(DEVICE)

            logits, recon = model(sequences)

            loss_ce    = ce_loss(logits, labels)
            target_num = get_mean_numerical_target(sequences)
            sq_err     = mse_loss(recon, target_num)
            loss_recon = (sq_err * recon_weights.to(DEVICE)).mean()
            loss       = loss_ce + LAMBDA_RECON * loss_recon

            total_loss  += loss.item()
            total_ce    += loss_ce.item()
            total_recon += loss_recon.item()

            _, predicted = logits.max(dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    n          = len(loader)
    preds_arr  = np.array(all_preds)
    labels_arr = np.array(all_labels)
    val_acc    = accuracy_score(labels_arr, preds_arr) * 100
    val_f1m    = f1_score(labels_arr, preds_arr,
                           average="macro", zero_division=0)

    return total_loss/n, total_ce/n, total_recon/n, val_acc, val_f1m


# ============================================================
# Single training run with early stopping
# ============================================================

def train_one_run(model_name, run_idx, train_loader, val_loader,
                  num_classes, recon_weights):
    set_seed(RANDOM_SEED + run_idx)

    model     = build_model(model_name, num_classes)
    ce_loss   = nn.CrossEntropyLoss()
    mse_loss  = nn.MSELoss(reduction="none")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    scheduler = None
    if USE_LR_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=LR_FACTOR,
            patience=LR_PATIENCE, min_lr=LR_MIN,
        )

    history = {
        "train_loss": [], "train_ce": [], "train_recon": [],
        "train_acc":  [],
        "val_loss":   [], "val_ce":   [], "val_recon":   [],
        "val_acc":    [], "val_f1m":  [],
        "learning_rates": [], "epoch_times": [],
    }

    best_val_loss   = float("inf")
    best_epoch      = 0
    patience_count  = 0
    best_state_dict = None

    print(f"\n  Run {run_idx + 1}/{NUM_RUNS} — {model_name.upper()}")
    print(f"  Patience: {EARLY_STOPPING_PATIENCE} epochs  |  "
          f"Max epochs: {MAX_EPOCHS}")
    print(f"  {'Ep':<6} {'TrLoss':<9} {'TrAcc%':<9} "
          f"{'ValLoss':<9} {'ValAcc%':<9} {'ValF1m':<8} "
          f"{'LR':<11} {'s':<6} {'*'}")
    print("  " + "-" * 76)

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()

        tr_loss, tr_ce, tr_recon, tr_acc = train_epoch(
            model, train_loader, optimizer, ce_loss, mse_loss, recon_weights
        )
        vl_loss, vl_ce, vl_recon, vl_acc, vl_f1m = val_epoch(
            model, val_loader, ce_loss, mse_loss, recon_weights
        )

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step(vl_loss)

        epoch_time = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_ce"].append(tr_ce)
        history["train_recon"].append(tr_recon)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_ce"].append(vl_ce)
        history["val_recon"].append(vl_recon)
        history["val_acc"].append(vl_acc)
        history["val_f1m"].append(vl_f1m)
        history["learning_rates"].append(current_lr)
        history["epoch_times"].append(epoch_time)

        improved = vl_loss < best_val_loss - 1e-6
        if improved:
            best_val_loss   = vl_loss
            best_epoch      = epoch
            patience_count  = 0
            best_state_dict = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            patience_count += 1

        if epoch % LOG_INTERVAL == 0 or epoch == 1 or improved:
            marker = "*" if improved else ""
            print(f"  {epoch:<6} {tr_loss:<9.4f} {tr_acc:<9.2f} "
                  f"{vl_loss:<9.4f} {vl_acc:<9.2f} {vl_f1m:<8.4f} "
                  f"{current_lr:<11.2e} {epoch_time:<6.0f} {marker}")

        if patience_count >= EARLY_STOPPING_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}. "
                  f"Best val loss {best_val_loss:.4f} at epoch {best_epoch}.")
            break

    if epoch == MAX_EPOCHS and patience_count < EARLY_STOPPING_PATIENCE:
        print(f"\n  Reached MAX_EPOCHS ({MAX_EPOCHS}). "
              f"Best val loss {best_val_loss:.4f} at epoch {best_epoch}.")

    print("  " + "-" * 76)
    print(f"  Best epoch:  {best_epoch}")
    print(f"  Val loss:    {best_val_loss:.4f}")
    print(f"  Val acc:     {history['val_acc'][best_epoch-1]:.2f}%")
    print(f"  Val F1-mac:  {history['val_f1m'][best_epoch-1]:.4f}")

    # Restore weights from best epoch
    model.load_state_dict(best_state_dict)

    return {
        "model"        : model,
        "history"      : history,
        "run_idx"      : run_idx,
        "model_name"   : model_name,
        "best_epoch"   : best_epoch,
        "best_val_loss": best_val_loss,
    }


# ============================================================
# MAIN
# ============================================================

def run_training(save=True):
    config.make_output_dirs()

    print("=" * 60)
    print("05_train.py — Training with Validation + Early Stopping")
    print("=" * 60)

    print("\nLoading sequences...")
    seq_path = os.path.join(PREPROCESSED_DIR, "sequences.joblib")
    data     = joblib.load(seq_path)

    train_sequences = data["train_sequences"]
    train_targets   = data["train_targets"]
    val_sequences   = data["val_sequences"]
    val_targets     = data["val_targets"]
    num_classes     = data["num_classes"]

    recon_weights = build_recon_weights()

    print(f"  Train sequences : {len(train_sequences):,}")
    print(f"  Val sequences   : {len(val_sequences):,}")
    print(f"  Num classes     : {num_classes}")
    print(f"  Device          : {DEVICE}")
    print(f"  Max epochs      : {MAX_EPOCHS} "
          f"(early stopping patience {EARLY_STOPPING_PATIENCE})")
    print(f"  LR scheduler    : patience={LR_PATIENCE} factor={LR_FACTOR}")
    print(f"  Lambda recon    : {LAMBDA_RECON}")
    print(f"  Recon weights   : {recon_weights.numpy()}")

    train_dataset = ERPSequenceDataset(train_sequences, train_targets)
    val_dataset   = ERPSequenceDataset(val_sequences,   val_targets)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    print(f"  Train batches   : {len(train_loader)}")
    print(f"  Val batches     : {len(val_loader)}")

    all_results = {}
    model_names = ["lstm", "gru", "transformer"]
    total_start = time.time()

    for model_name in model_names:
        print(f"\n{'=' * 60}")
        print(f"Training {model_name.upper()} "
              f"({NUM_RUNS} runs, early stopping)")
        print(f"{'=' * 60}")

        model_results = []
        run_start     = time.time()

        for run_idx in range(NUM_RUNS):
            result = train_one_run(
                model_name    = model_name,
                run_idx       = run_idx,
                train_loader  = train_loader,
                val_loader    = val_loader,
                num_classes   = num_classes,
                recon_weights = recon_weights,
            )
            model_results.append(result)

            if save:
                weight_path = os.path.join(
                    MODELS_DIR, f"{model_name}_run{run_idx + 1}.pth"
                )
                torch.save(result["model"].state_dict(), weight_path)
                print(f"  Saved best weights: {weight_path}")

        best_epochs  = [r["best_epoch"] for r in model_results]
        best_val_acc = [
            r["history"]["val_acc"][r["best_epoch"] - 1]
            for r in model_results
        ]
        best_val_f1m = [
            r["history"]["val_f1m"][r["best_epoch"] - 1]
            for r in model_results
        ]
        run_elapsed = time.time() - run_start

        print(f"\n  {model_name.upper()} Summary ({NUM_RUNS} runs):")
        print(f"    Best epochs  : {best_epochs}")
        print(f"    Val Acc mean : {np.mean(best_val_acc):.2f}%  "
              f"std {np.std(best_val_acc):.2f}%")
        print(f"    Val F1m mean : {np.mean(best_val_f1m):.4f}  "
              f"std {np.std(best_val_f1m):.4f}")
        print(f"    Time         : {run_elapsed/60:.1f} minutes")

        all_results[model_name] = model_results

        if save:
            history_path = os.path.join(
                METRICS_DIR, f"{model_name}_train_history.json"
            )
            serialisable = []
            for r in model_results:
                serialisable.append({
                    "run_idx"      : r["run_idx"],
                    "best_epoch"   : r["best_epoch"],
                    "best_val_loss": r["best_val_loss"],
                    "history"      : r["history"],
                })
            with open(history_path, "w") as f:
                json.dump(serialisable, f, indent=2)
            print(f"  History saved: {history_path}")

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"All models trained in {total_elapsed/60:.1f} minutes.")
    print(f"Best weights saved in: {MODELS_DIR}")
    print("=" * 60)

    return all_results


if __name__ == "__main__":
    run_training(save=True)