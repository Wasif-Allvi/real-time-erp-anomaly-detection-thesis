# ============================================================
# 10_significance.py
# Statistical significance of the architecture comparison
#
# Purpose: test whether the accuracy differences between LSTM,
# GRU and Transformer on the held-out test set are larger than
# what seed variation and test-set sampling alone would produce.
#
# Three complementary analyses, all on the same 22,238 test
# prefixes and the trained weights of the three seeded runs:
#
#   1. Paired t-test across seeds (n = 3 per architecture,
#      run i of model A paired with run i of model B).
#      Low power by construction; reported for completeness.
#
#   2. McNemar's exact test per seed (Dietterich, 1998):
#      compares the two classifiers prediction by prediction
#      on the identical test set.
#
#   3. Case-level cluster bootstrap of the accuracy difference
#      (run-averaged correctness). Prefixes of one case are not
#      independent, so whole cases are resampled. This gives a
#      95% confidence interval that respects that dependence.
#
# Output: results/metrics/significance_results.json
# ============================================================

import os
import json
import importlib

import numpy as np
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score

import config
from config import (
    PREPROCESSED_DIR, MODELS_DIR, METRICS_DIR,
    DEVICE, BATCH_SIZE, NUM_RUNS, RANDOM_SEED,
)

build_model = importlib.import_module("04_models").build_model

MODELS = ["lstm", "gru", "transformer"]
N_BOOT = 10000


class SequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


# ------------------------------------------------------------
# Predictions of every run on the test set
# ------------------------------------------------------------
def predict_all_runs(model_name, loader, num_classes):
    preds = []
    for run_idx in range(1, NUM_RUNS + 1):
        path = os.path.join(MODELS_DIR, f"{model_name}_run{run_idx}.pth")
        model = build_model(model_name, num_classes)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        out = []
        with torch.no_grad():
            for x, _ in loader:
                logits, _ = model(x.to(DEVICE))
                out.append(logits.argmax(dim=1).cpu().numpy())
        preds.append(np.concatenate(out))
    return np.stack(preds)  # (runs, N)


# ------------------------------------------------------------
# Statistical tests
# ------------------------------------------------------------
def mcnemar_exact(correct_a, correct_b):
    """Two-sided exact McNemar test on paired correctness vectors."""
    b = int(np.sum(correct_a & ~correct_b))   # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))   # A wrong, B right
    n = b + c
    p = float(stats.binomtest(min(b, c), n, 0.5).pvalue) if n > 0 else 1.0
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "p_value": p}


def cluster_bootstrap(diff, case_ids, n_boot=N_BOOT, seed=RANDOM_SEED):
    """95% CI of the mean per-prefix difference, resampling whole cases."""
    _, inv = np.unique(case_ids, return_inverse=True)
    n_cases = inv.max() + 1
    sum_d = np.bincount(inv, weights=diff, minlength=n_cases)
    n_pre = np.bincount(inv, minlength=n_cases).astype(float)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_cases, n_cases)
        boots[i] = sum_d[idx].sum() / n_pre[idx].sum()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    observed = float(diff.mean())
    # two-sided bootstrap p-value: share of resamples on the other side of zero
    # (if no resample crosses zero, report the resolution limit 1/n_boot)
    p = 2 * min(np.mean(boots <= 0), np.mean(boots >= 0))
    p = min(max(p, 1.0 / n_boot), 1.0)
    return {
        "observed_diff_pp": 100 * observed,
        "ci95_low_pp": 100 * float(lo),
        "ci95_high_pp": 100 * float(hi),
        "p_value": float(p),
        "p_is_upper_bound": bool(p <= 1.0 / n_boot),
        "n_cases": int(n_cases),
        "n_boot": n_boot,
    }


def main():
    print("=" * 60)
    print("10_significance.py — Significance of the architecture comparison")
    print("=" * 60)

    data = joblib.load(os.path.join(PREPROCESSED_DIR, "sequences.joblib"))
    X, y = data["test_sequences"], data["test_targets"]
    case_ids = np.asarray(data["test_case_ids"])
    num_classes = data["num_classes"]
    loader = DataLoader(SequenceDataset(X, y), batch_size=BATCH_SIZE, shuffle=False)
    print(f"  Test prefixes: {len(y):,} from {len(np.unique(case_ids)):,} cases")

    preds, correct, per_run = {}, {}, {}
    for m in MODELS:
        print(f"\nPredicting with {m.upper()} (three runs)...")
        preds[m] = predict_all_runs(m, loader, num_classes)
        correct[m] = preds[m] == y[None, :]
        accs = [100 * accuracy_score(y, p) for p in preds[m]]
        f1m = [f1_score(y, p, average="macro", zero_division=0) for p in preds[m]]
        per_run[m] = {
            "accuracy_pct": accs,
            "f1_macro": f1m,
            "acc_mean": float(np.mean(accs)),
            "acc_sd_sample": float(np.std(accs, ddof=1)),
            "acc_sd_population": float(np.std(accs, ddof=0)),
            "f1m_mean": float(np.mean(f1m)),
            "f1m_sd_sample": float(np.std(f1m, ddof=1)),
        }
        print("    accuracy per run : " + ", ".join(f"{a:.2f}" for a in accs))
        print("    F1-macro per run : " + ", ".join(f"{f:.4f}" for f in f1m))
        print(f"    mean {per_run[m]['acc_mean']:.2f}  "
              f"SD sample {per_run[m]['acc_sd_sample']:.2f}  "
              f"SD population {per_run[m]['acc_sd_population']:.2f}")

    pairs = {}
    print("\n" + "=" * 60)
    print("Pairwise comparisons (first model minus second)")
    print("=" * 60)
    for a, b in [("gru", "lstm"), ("gru", "transformer"), ("lstm", "transformer")]:
        key = f"{a}_vs_{b}"
        acc_a = np.array(per_run[a]["accuracy_pct"])
        acc_b = np.array(per_run[b]["accuracy_pct"])
        t_acc = stats.ttest_rel(acc_a, acc_b)
        f1_a = np.array(per_run[a]["f1_macro"])
        f1_b = np.array(per_run[b]["f1_macro"])
        t_f1 = stats.ttest_rel(f1_a, f1_b)

        mcn = [mcnemar_exact(correct[a][r], correct[b][r]) for r in range(NUM_RUNS)]
        diff = correct[a].mean(axis=0) - correct[b].mean(axis=0)
        boot = cluster_bootstrap(diff, case_ids)

        pairs[key] = {
            "mean_acc_diff_pp": float(acc_a.mean() - acc_b.mean()),
            "runs_where_first_is_better": int(np.sum(acc_a > acc_b)),
            "paired_t_accuracy": {"t": float(t_acc.statistic), "p_value": float(t_acc.pvalue), "df": NUM_RUNS - 1},
            "paired_t_f1_macro": {"t": float(t_f1.statistic), "p_value": float(t_f1.pvalue), "df": NUM_RUNS - 1},
            "mcnemar_per_seed": mcn,
            "cluster_bootstrap": boot,
        }
        print(f"\n  {a.upper()} vs {b.upper()}")
        print(f"    mean accuracy difference : {pairs[key]['mean_acc_diff_pp']:+.2f} pp "
              f"({pairs[key]['runs_where_first_is_better']}/3 seeds favour {a.upper()})")
        print(f"    paired t (accuracy)      : t = {t_acc.statistic:.2f}, p = {t_acc.pvalue:.3f}")
        print(f"    paired t (F1-macro)      : t = {t_f1.statistic:.2f}, p = {t_f1.pvalue:.3f}")
        for r, res in enumerate(mcn, 1):
            print(f"    McNemar seed run {r}       : {res['a_right_b_wrong']} vs "
                  f"{res['a_wrong_b_right']} discordant, p = {res['p_value']:.2e}")
        p_txt = (f"p < {boot['p_value']:.4f}" if boot["p_is_upper_bound"]
                 else f"p = {boot['p_value']:.4f}")
        print(f"    cluster bootstrap 95% CI : [{boot['ci95_low_pp']:+.2f}, "
              f"{boot['ci95_high_pp']:+.2f}] pp, {p_txt}")

    out = os.path.join(METRICS_DIR, "significance_results.json")
    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"per_run": per_run, "pairs": pairs}, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
