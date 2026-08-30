import os
import json
import numpy as np

metrics_dir = r'C:\Users\wasif\Desktop\Thesis\Thesis Code\results\metrics'

models = ["lstm", "gru", "transformer"]

print("=" * 70)
print("TRAINING RESULTS SUMMARY — Full Implementation")
print("BPI Challenge 2020 | 50 epochs | 3 runs | CPU")
print("=" * 70)

for model_name in models:
    path = os.path.join(metrics_dir, f"{model_name}_train_history.json")
    with open(path) as f:
        data = json.load(f)

    print(f"\n{'=' * 70}")
    print(f"MODEL: {model_name.upper()}")
    print(f"{'=' * 70}")

    final_accs   = []
    final_losses = []
    best_accs    = []

    for run in data:
        run_idx  = run["run_idx"]
        acc_hist = run["history"]["train_acc"]
        loss_hist = run["history"]["train_loss"]
        ce_hist  = run["history"]["train_ce_loss"]
        recon_hist = run["history"]["train_recon_loss"]
        times    = run["history"]["epoch_times"]

        final_accs.append(acc_hist[-1])
        final_losses.append(loss_hist[-1])
        best_accs.append(max(acc_hist))

        print(f"\n  Run {run_idx + 1}:")
        print(f"  {'Epoch':<8} {'Loss':<10} {'CE':<10} "
              f"{'Recon':<10} {'Acc%':<10}")
        print(f"  {'-'*48}")

        # Print epochs 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50
        log_epochs = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        for ep in log_epochs:
            idx = ep - 1
            if idx < len(acc_hist):
                print(f"  {ep:<8} {loss_hist[idx]:<10.4f} "
                      f"{ce_hist[idx]:<10.4f} "
                      f"{recon_hist[idx]:<10.4f} "
                      f"{acc_hist[idx]:<10.2f}")

        avg_epoch_time = np.mean(times)
        total_time     = sum(times) / 60
        print(f"\n  Avg epoch time : {avg_epoch_time:.1f}s")
        print(f"  Total run time : {total_time:.1f} min")
        print(f"  Best acc       : {max(acc_hist):.2f}% (epoch {acc_hist.index(max(acc_hist))+1})")
        print(f"  Final acc      : {acc_hist[-1]:.2f}%")
        print(f"  Final loss     : {loss_hist[-1]:.4f}")

    print(f"\n  {'─'*48}")
    print(f"  ACROSS 3 RUNS:")
    print(f"  Final Accuracy  — mean: {np.mean(final_accs):.2f}%  "
          f"std: {np.std(final_accs):.2f}%  "
          f"range: [{min(final_accs):.2f}%, {max(final_accs):.2f}%]")
    print(f"  Final Loss      — mean: {np.mean(final_losses):.4f}  "
          f"std: {np.std(final_losses):.4f}")
    print(f"  Best Accuracy   — mean: {np.mean(best_accs):.2f}%  "
          f"std: {np.std(best_accs):.2f}%")

print(f"\n{'=' * 70}")
print("CROSS-MODEL COMPARISON (final epoch, mean across 3 runs)")
print(f"{'=' * 70}")
print(f"  {'Model':<15} {'Mean Acc%':<12} {'Std Acc%':<12} {'Mean Loss':<12}")
print(f"  {'-'*51}")

for model_name in models:
    path = os.path.join(metrics_dir, f"{model_name}_train_history.json")
    with open(path) as f:
        data = json.load(f)
    accs   = [r["history"]["train_acc"][-1]  for r in data]
    losses = [r["history"]["train_loss"][-1] for r in data]
    print(f"  {model_name.upper():<15} "
          f"{np.mean(accs):<12.2f} "
          f"{np.std(accs):<12.2f} "
          f"{np.mean(losses):<12.4f}")

print(f"\n  Note: Training accuracy only. Test accuracy computed in 08_evaluate.py")
print("=" * 70)