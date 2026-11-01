# ============================================================
# 12_export_raw_scores.py
# Per-run raw anomaly scores for the final evaluation (13_...)
#
# For every trained model run, scores two prefix sets:
#   - the injected test prefixes   (results/preprocessed/test_injected.joblib)
#   - the clean validation prefixes (results/preprocessed/sequences.joblib)
#
# Classifiers (LSTM, GRU, Transformer; runs 1-3):
#   confidence  = 1 - max softmax(logits)
#   recon error = weighted MSE between the reconstruction head output
#                 and the mean numerical vector over real timesteps
#                 (financial features x2), exactly as in 07/08
# Dedicated autoencoder (runs 1-3):
#   masked, weighted reconstruction MSE (financial x3), as in 05b
#
# The validation scores are used only to fix the 99th-percentile
# normalisation of the reconstruction error in 13, so that no test
# information enters it.
#
# Output: results/figures/raw_scores.npz
# ============================================================

import os
import importlib
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
from config import (PREPROCESSED_DIR, MODELS_DIR, DEVICE, BATCH_SIZE, NUM_RUNS,
                    N_CATEGORICAL, N_NUMERICAL, MODEL_INPUT_DIM,
                    FINANCIAL_INDICES_IN_NUMERICAL, FINANCIAL_RECON_WEIGHT)

build_model = importlib.import_module("04_models").build_model
ae_mod = importlib.import_module("05b_autoencoder")

OUT = os.path.join(config.RESULTS_DIR, "figures", "raw_scores.npz")


def recon_weights():
    w = torch.ones(N_NUMERICAL)
    for i in FINANCIAL_INDICES_IN_NUMERICAL:
        w[i] = FINANCIAL_RECON_WEIGHT
    return w


def mean_numerical_target(x):
    num = x[:, :, N_CATEGORICAL:]
    mask = (x.abs().sum(dim=2) > 0).float().unsqueeze(2)
    return (num * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


@torch.no_grad()
def classifier_scores(model, X):
    model.eval()
    w = recon_weights().to(DEVICE)
    mse = nn.MSELoss(reduction="none")
    conf, err = [], []
    for (x,) in DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                           batch_size=BATCH_SIZE, shuffle=False):
        x = x.to(DEVICE)
        logits, recon = model(x)
        conf.append((1 - torch.softmax(logits, dim=1).max(dim=1).values).cpu().numpy())
        err.append((mse(recon, mean_numerical_target(x)) * w).mean(dim=1).cpu().numpy())
    return np.concatenate(conf), np.concatenate(err)


def main():
    inj = joblib.load(os.path.join(PREPROCESSED_DIR, "test_injected.joblib"))
    seq = joblib.load(os.path.join(PREPROCESSED_DIR, "sequences.joblib"))
    X_test, X_val = inj["sequences"], seq["val_sequences"]
    num_classes = len(joblib.load(config.ENCODER_PATH)["concept:name"].classes_)
    out = {"case_labels": np.asarray(inj["labels"])}

    for name in ["lstm", "gru", "transformer"]:
        ct, et, cv, ev = [], [], [], []
        for r in range(1, NUM_RUNS + 1):
            model = build_model(name, num_classes)
            model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"{name}_run{r}.pth"),
                                             map_location=DEVICE))
            c, e = classifier_scores(model, X_test); ct.append(c); et.append(e)
            c, e = classifier_scores(model, X_val); cv.append(c); ev.append(e)
            print(f"  {name} run {r} done")
        out[f"test_conf_{name}"], out[f"test_err_{name}"] = np.array(ct), np.array(et)
        out[f"val_conf_{name}"], out[f"val_err_{name}"] = np.array(cv), np.array(ev)

    nw = ae_mod.build_numerical_weights()
    at, av = [], []
    for r in range(1, ae_mod.AE_RUNS + 1):
        ae = ae_mod.GRUAutoencoder(input_dim=MODEL_INPUT_DIM, n_numerical=ae_mod.N_NUMERICAL,
                                   hidden_dim=ae_mod.AE_HIDDEN_DIM, num_layers=ae_mod.AE_NUM_LAYERS,
                                   dropout=ae_mod.AE_DROPOUT).to(DEVICE)
        ae.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"autoencoder_run{r}.pth"),
                                      map_location=DEVICE))
        at.append(ae_mod.compute_event_level_scores(ae, X_test, nw))
        av.append(ae_mod.compute_event_level_scores(ae, X_val, nw))
        print(f"  autoencoder run {r} done")
    out["test_ae"], out["val_ae"] = np.array(at), np.array(av)

    np.savez_compressed(OUT, **out)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
