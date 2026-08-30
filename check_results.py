import os
import json

models_dir  = r'C:\Users\wasif\Desktop\Thesis\Thesis Code\results\models'
metrics_dir = r'C:\Users\wasif\Desktop\Thesis\Thesis Code\results\metrics'

print("=== MODEL WEIGHTS ===")
for f in sorted(os.listdir(models_dir)):
    size = os.path.getsize(os.path.join(models_dir, f)) / (1024 * 1024)
    print(f"  {f}  ({size:.1f} MB)")

print()
print("=== TRAINING HISTORIES ===")
for f in sorted(os.listdir(metrics_dir)):
    path = os.path.join(metrics_dir, f)
    size = os.path.getsize(path) / 1024
    print(f"  {f}  ({size:.1f} KB)")
    if f.endswith(".json"):
        with open(path) as jf:
            data = json.load(jf)
        for run in data:
            acc  = run["history"]["train_acc"][-1]
            loss = run["history"]["train_loss"][-1]
            epochs = len(run["history"]["train_acc"])
            print(f"    Run {run['run_idx'] + 1}: "
                  f"epochs={epochs}  "
                  f"final_acc={acc:.2f}%  "
                  f"final_loss={loss:.4f}")