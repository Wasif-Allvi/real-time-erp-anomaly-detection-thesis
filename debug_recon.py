# debug_recon.py — run once to diagnose
import os
import numpy as np
import joblib
import torch
import importlib
from torch.utils.data import DataLoader, Dataset
import config

models_module = importlib.import_module("04_models")
build_model   = models_module.build_model

class TestDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets   = torch.tensor(targets,   dtype=torch.long)
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx], self.targets[idx]

inj_data = joblib.load(
    os.path.join(config.PREPROCESSED_DIR, "test_injected.joblib")
)
encoders    = joblib.load(config.ENCODER_PATH)
num_classes = len(encoders["concept:name"].classes_)

dataset = TestDataset(inj_data["sequences"], inj_data["targets"])
loader  = DataLoader(dataset, batch_size=64, shuffle=False)

model = build_model("lstm", num_classes)
model.load_state_dict(torch.load(
    os.path.join(config.MODELS_DIR, "lstm_run1.pth"),
    map_location=config.DEVICE
))
model.eval()

all_recon_errors = []
with torch.no_grad():
    for sequences, _ in loader:
        logits, recon = model(sequences)
        last_step = sequences[:, -1, :]
        mse = ((recon - last_step) ** 2).mean(dim=1)
        all_recon_errors.extend(mse.cpu().numpy())

errors = np.array(all_recon_errors)
true_binary = inj_data["labels"]

print(f"Recon error stats:")
print(f"  Min:    {errors.min():.8f}")
print(f"  Max:    {errors.max():.8f}")
print(f"  Mean:   {errors.mean():.8f}")
print(f"  Std:    {errors.std():.8f}")
print(f"\nBy label:")
print(f"  Normal mean:  {errors[true_binary==0].mean():.8f}")
print(f"  Anomaly mean: {errors[true_binary==1].mean():.8f}")
print(f"\nAre they truly zero?")
print(f"  Fraction exactly zero: {(errors == 0).mean():.4f}")