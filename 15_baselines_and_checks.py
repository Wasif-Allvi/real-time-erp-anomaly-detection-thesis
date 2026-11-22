# ============================================================
# 15_baselines_and_checks.py
# Thesis: Deep Sequential Models for ERP Anomaly Detection
# Mustafa Wasif Allvi, University of Potsdam, M.Sc. Data Science
# ============================================================
# Simple reference points and consistency checks for the detection
# results of 13_event_level_evaluation.py (first configuration, models in
# results/models). Nothing is retrained.
#
#   A. Macro-F1 conventions. The output vocabulary has 37 activity
#      classes, the test targets contain fewer. Macro-F1 is computed
#      (1) over the classes that occur in the test targets or
#      predictions (the convention used in the thesis) and (2) over all 37.
#   B. Single-run detection metrics. 13_... averages the scores of
#      the three runs of a model before it computes the metrics.
#      Here every run is evaluated on its own.
#   C. Properties of the clean data that matter for the injected
#      financial anomalies: the four amount attributes are compared
#      with each other, and the amount is checked for changes within
#      a case.
#   D. Simple baselines on the injected test prefixes:
#        - largest declared amount in the prefix,
#        - declared amount divided by the permit budget,
#        - two consistency rules that need no training:
#            R1  Amount differs from OriginalAmount,
#            R2  Amount changes within the prefix.
#   E. Consistent cross-entity injection. In the injected test set
#      only case:Amount is inflated. Here the three other amount
#      attributes follow it, so that only the size of the amount and
#      its relation to the permit budget remain as signals. The
#      saved autoencoders score this variant.
#
# Inputs : results/figures/raw_scores.npz, results/metrics/event_labels.npz
#          (12_..., 13_...), results/preprocessed/*, results/models/*.pth
# Output : results/metrics/baselines_and_checks.json
# Usage  : python 15_baselines_and_checks.py
# ============================================================

import importlib
import importlib.util
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             f1_score, roc_auc_score)

import config
from config import MODELS_DIR, PREPROCESSED_DIR, DEVICE, MODEL_INPUT_DIM, NUM_RUNS

ROOT = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "ev13", os.path.join(ROOT, "13_event_level_evaluation.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)
ex = importlib.import_module("12_export_raw_scores")
ae_mod = ex.ae_mod

OUT = os.path.join(ROOT, "results", "metrics", "baselines_and_checks.json")
MODELS = ["lstm", "gru", "transformer"]
I_AMOUNT, I_REQUESTED, I_ADJUSTED, I_ORIGINAL, I_BUDGET = 7, 8, 9, 10, 11


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def real_mask(X):
    return np.abs(X).sum(axis=2) > 0


def load_classifier(name, run, num_classes):
    model = ex.build_model(name, num_classes)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"{name}_run{run}.pth"),
                                     map_location=DEVICE))
    model.eval()
    return model


def load_autoencoder(run):
    ae = ae_mod.GRUAutoencoder(input_dim=MODEL_INPUT_DIM, n_numerical=ae_mod.N_NUMERICAL,
                               hidden_dim=ae_mod.AE_HIDDEN_DIM, num_layers=ae_mod.AE_NUM_LAYERS,
                               dropout=ae_mod.AE_DROPOUT).to(DEVICE)
    ae.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"autoencoder_run{run}.pth"),
                                  map_location=DEVICE))
    ae.eval()
    return ae


@torch.no_grad()
def predict(model, X, batch=2048):
    out = []
    for s in range(0, len(X), batch):
        logits, _ = model(torch.tensor(X[s:s + batch], dtype=torch.float32).to(DEVICE))
        out.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(out)


def lm_auc(s, y, types_ev, plen, t):
    return float(ev.length_matched_auc(s, (y == 1) & (types_ev == t), y == 0, plen)[0])


def basic(s, y, types_ev, plen):
    return {"roc_auc": float(roc_auc_score(y, s)),
            "auprc": float(average_precision_score(y, s)),
            "per_type_length_matched": {t: lm_auc(s, y, types_ev, plen, t) for t in ev.TYPES}}


def full(s, y, types_ev, plen, cases, case_type, folds, key_idx):
    """Same quantities as in 13_...: overall, held-out threshold metrics, per type with interval."""
    r = {"roc_auc": float(roc_auc_score(y, s)),
         "auprc": float(average_precision_score(y, s)),
         "crossfit": ev.crossfit(s, y, types_ev, folds),
         "per_type_length_matched": ev.per_type_auc(s, y, types_ev, plen, cases, case_type, key_idx)}
    return r


def show(name, r):
    pt = r["per_type_length_matched"]
    val = lambda t: pt[t]["auc"] if isinstance(pt[t], dict) else pt[t]
    extra = ""
    if "crossfit" in r:
        cf = r["crossfit"]
        extra = f" | held-out P {cf['precision']:.3f} R {cf['recall']:.3f} normal flagged {cf['normal_flagged_pct']:.2f}%"
    print(f"  {name:44s} ROC {r['roc_auc']:.4f} AUPRC {r['auprc']:.4f} | "
          + "  ".join(f"{t[:5]} {val(t):.3f}" for t in ev.TYPES) + extra)


# ------------------------------------------------------------
# A. macro-F1 conventions
# ------------------------------------------------------------
def macro_f1_conventions():
    seq = joblib.load(os.path.join(PREPROCESSED_DIR, "sequences.joblib"))
    X, y = seq["test_sequences"], np.asarray(seq["test_targets"])
    num_classes = int(seq["num_classes"])
    res = {"classes_in_vocabulary": num_classes,
           "classes_in_test_targets": int(len(np.unique(y))), "models": {}}
    print(f"A. Macro-F1: vocabulary {num_classes} classes, test targets contain {res['classes_in_test_targets']}")
    for m in MODELS:
        acc, f_present, f_all, n_pred, n_union = [], [], [], [], []
        for run in range(1, NUM_RUNS + 1):
            p = predict(load_classifier(m, run, num_classes), X)
            acc.append(100 * accuracy_score(y, p))
            f_present.append(f1_score(y, p, average="macro", zero_division=0))
            f_all.append(f1_score(y, p, average="macro", zero_division=0,
                                  labels=np.arange(num_classes)))
            n_pred.append(int(len(np.unique(p))))
            n_union.append(int(len(np.union1d(np.unique(y), np.unique(p)))))
        res["models"][m] = {
            "accuracy_pct": [float(a) for a in acc], "accuracy_mean": float(np.mean(acc)),
            "macro_f1_classes_present": [float(f) for f in f_present],
            "macro_f1_classes_present_mean": float(np.mean(f_present)),
            "macro_f1_all_classes": [float(f) for f in f_all],
            "macro_f1_all_classes_mean": float(np.mean(f_all)),
            "classes_predicted": n_pred, "classes_averaged_by_default": n_union}
        print(f"  {m:11s} accuracy {np.mean(acc):.2f}% | macro-F1 over {n_union} classes "
              f"{np.mean(f_present):.4f} | over all {num_classes} classes {np.mean(f_all):.4f} "
              f"| classes ever predicted {n_pred}")
    return res


# ------------------------------------------------------------
# B. single runs
# ------------------------------------------------------------
def single_runs(raw, y, types_ev, plen):
    print("B. Detection metrics of the single runs (13_... uses the mean score of the three runs)")
    res = {}
    keys = [(f"{m}_confidence", f"test_conf_{m}") for m in MODELS] \
        + [(f"{m}_joint_recon", f"test_err_{m}") for m in MODELS] + [("autoencoder", "test_ae")]
    for name, k in keys:
        S = raw[k]
        res[name] = {"mean_score": basic(S.mean(0), y, types_ev, plen),
                     "runs": [basic(s, y, types_ev, plen) for s in S]}
        cf = [r["per_type_length_matched"]["control_flow"] for r in res[name]["runs"]]
        ce = [r["per_type_length_matched"]["cross_entity"] for r in res[name]["runs"]]
        print(f"  {name:24s} ROC-AUC mean score {res[name]['mean_score']['roc_auc']:.4f}, runs "
              + ", ".join(f"{r['roc_auc']:.4f}" for r in res[name]["runs"])
              + " | control-flow runs " + ", ".join(f"{v:.3f}" for v in cf)
              + " | cross-entity runs " + ", ".join(f"{v:.3f}" for v in ce))
    return res


# ------------------------------------------------------------
# C. properties of the clean data
# ------------------------------------------------------------
def clean_data_properties():
    seq = joblib.load(os.path.join(PREPROCESSED_DIR, "sequences.joblib"))
    res = {}
    for part in ("train", "val", "test"):
        X = seq[f"{part}_sequences"]
        m = real_mask(X)
        a, o = X[:, :, I_AMOUNT][m], X[:, :, I_ORIGINAL][m]
        r, j = X[:, :, I_REQUESTED][m], X[:, :, I_ADJUSTED][m]
        has = o > 0
        same = (a[has] == o[has]) & (a[has] == r[has]) & (a[has] == j[has])
        amount = np.where(m, X[:, :, I_AMOUNT], np.nan)
        rng = np.nanmax(amount, axis=1) - np.nanmin(amount, axis=1)
        res[part] = {"events": int(m.sum()),
                     "events_with_original_amount": int(has.sum()),
                     "of_these_all_four_amounts_identical": int(same.sum()),
                     "prefixes": int(len(X)),
                     "prefixes_with_changing_amount": int((rng > 0).sum())}
    df = pd.read_csv(os.path.join(PREPROCESSED_DIR, "events_processed_test.csv"), low_memory=False)
    first = df.groupby("case:concept:name").first()
    b = first["case:Permit RequestedBudget"].astype(float)
    a = first["case:Amount"].astype(float)
    has = b > 0
    res["clean_test_cases_with_permit_budget"] = int(has.sum())
    res["of_these_amount_above_budget_pct"] = float(100 * (a[has] > b[has]).mean())
    res["of_these_amount_above_twice_budget_pct"] = float(100 * (a[has] > 2 * b[has]).mean())
    print("C. Clean data")
    for part in ("train", "val", "test"):
        p = res[part]
        print(f"  {part:5s} events with OriginalAmount {p['events_with_original_amount']:,}, all four amount "
              f"attributes identical in {p['of_these_all_four_amounts_identical']:,} | prefixes in which "
              f"Amount changes: {p['prefixes_with_changing_amount']} of {p['prefixes']:,}")
    print(f"  clean test cases with a permit budget: {res['clean_test_cases_with_permit_budget']}, amount above "
          f"budget in {res['of_these_amount_above_budget_pct']:.1f}%, above twice the budget in "
          f"{res['of_these_amount_above_twice_budget_pct']:.1f}%")
    return res


# ------------------------------------------------------------
# D. simple baselines
# ------------------------------------------------------------
def simple_baselines(X, y, types_ev, plen, cases, case_type, folds):
    m = real_mask(X)
    scalers = joblib.load(config.SCALER_PATH)   # one MinMaxScaler per numerical feature

    def unscale(col, v):
        sc = scalers[col]
        return v * (sc.data_max_[0] - sc.data_min_[0]) + sc.data_min_[0]

    raw_amount = unscale("case:Amount", X[:, :, I_AMOUNT])
    raw_budget = unscale("case:Permit RequestedBudget", X[:, :, I_BUDGET])
    amount_max = np.where(m, raw_amount, -np.inf).max(axis=1)
    budget = np.where(m, raw_budget, 0.0).max(axis=1)
    ratio = np.where(budget > 0, amount_max / np.maximum(budget, 1e-9), 0.0)
    a, o = X[:, :, I_AMOUNT], X[:, :, I_ORIGINAL]
    r1 = np.where(m & (o > 0), np.abs(a - o), 0.0).max(axis=1)
    am = np.where(m, a, np.nan)
    r2 = np.nanmax(am, axis=1) - np.nanmin(am, axis=1)
    rule = ((r1 > 1e-9) | (r2 > 1e-9)).astype(float)
    scores = {"largest_amount_in_prefix": amount_max,
              "amount_to_permit_budget": ratio,
              "rule_amount_differs_from_original_amount": (r1 > 1e-9).astype(float),
              "rule_amount_changes_within_prefix": (r2 > 1e-9).astype(float),
              "rule_either": rule}
    print("D. Simple baselines on the injected test prefixes")
    res = {}
    for i, (name, s) in enumerate(scores.items()):
        res[name] = full(s, y, types_ev, plen, cases, case_type, folds, 100 + i)
        show(name, res[name])
    flagged = rule > 0
    tp = int((flagged & (y == 1)).sum())
    res["rule_either"]["as_fixed_rule"] = {
        "prefixes_flagged": int(flagged.sum()), "true_positives": tp,
        "precision": float(tp / max(int(flagged.sum()), 1)), "recall": float(tp / int((y == 1).sum())),
        "share_of_anomalous_prefixes_flagged_by_type":
            {t: float(flagged[(y == 1) & (types_ev == t)].mean()) for t in ev.TYPES}}
    f = res["rule_either"]["as_fixed_rule"]
    print(f"  the two rules as a fixed rule: {f['prefixes_flagged']} prefixes flagged, "
          f"precision {f['precision']:.3f}, recall {f['recall']:.3f}")
    return res


# ------------------------------------------------------------
# E. consistent cross-entity injection
# ------------------------------------------------------------
def consistent_injection(X, raw, y, types_ev, plen, cases, case_type, folds):
    nw = ae_mod.build_numerical_weights()
    aes = [load_autoencoder(r) for r in range(1, ae_mod.AE_RUNS + 1)]
    base = np.array([ae_mod.compute_event_level_scores(ae, X, nw) for ae in aes])
    diff = float(np.abs(base - raw["test_ae"]).max())
    print(f"E. Consistent cross-entity injection (self-check: recomputed autoencoder scores differ "
          f"from raw_scores.npz by at most {diff:.1e})")

    m = real_mask(X)
    Xc = X.copy()
    sel = (types_ev == "cross_entity")[:, None] & m & (X[:, :, I_ORIGINAL] > 0)
    for f in (I_REQUESTED, I_ADJUSTED, I_ORIGINAL):
        Xc[:, :, f] = np.where(sel, X[:, :, I_AMOUNT], X[:, :, f])
    S = np.array([ae_mod.compute_event_level_scores(ae, Xc, nw) for ae in aes])

    res = {"self_check_max_abs_diff": diff,
           "prefixes_changed": int(sel.any(axis=1).sum()),
           "as_in_thesis": full(base.mean(0), y, types_ev, plen, cases, case_type, folds, 200),
           "consistent": full(S.mean(0), y, types_ev, plen, cases, case_type, folds, 201),
           "consistent_single_runs": [basic(s, y, types_ev, plen) for s in S]}
    show("autoencoder, injection as in the thesis", res["as_in_thesis"])
    show("autoencoder, all four amounts inflated", res["consistent"])
    ce = (y == 1) & (types_ev == "cross_entity")
    res["median_score_cross_entity"] = {"as_in_thesis": float(np.median(base.mean(0)[ce])),
                                        "consistent": float(np.median(S.mean(0)[ce]))}
    res["normal_score_median"] = float(np.median(base.mean(0)[y == 0]))
    res["normal_score_99th_percentile"] = float(np.percentile(base.mean(0)[y == 0], 99))
    print(f"  median autoencoder score of the cross-entity prefixes: "
          f"{res['median_score_cross_entity']['as_in_thesis']:.5f} -> "
          f"{res['median_score_cross_entity']['consistent']:.5f} "
          f"(normal prefixes: median {res['normal_score_median']:.5f}, "
          f"99th percentile {res['normal_score_99th_percentile']:.5f})")
    return res


# ------------------------------------------------------------
def main():
    inj = joblib.load(os.path.join(PREPROCESSED_DIR, "test_injected.joblib"))
    raw = np.load(ev.RAW)
    lab = np.load(ev.LAB)
    X = np.asarray(inj["sequences"], dtype=np.float32)
    cases = np.asarray(inj["case_ids"])
    types = np.asarray(inj["types"])
    y, plen = lab["event_labels"], lab["prefix_length"]
    assert (raw["case_labels"] == np.asarray(inj["labels"])).all(), "raw scores and injected test set differ"
    types_ev = np.where(y == 1, types, "none")
    case_type = {c: t for c, t in zip(cases, types) if t != "none"}
    folds = ev.make_folds(cases, case_type, ev.N_REP)

    out = {"macro_f1": macro_f1_conventions(),
           "single_runs": single_runs(raw, y, types_ev, plen),
           "clean_data": clean_data_properties(),
           "baselines": simple_baselines(X, y, types_ev, plen, cases, case_type, folds),
           "consistent_injection": consistent_injection(X, raw, y, types_ev, plen, cases, case_type, folds)}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved:", OUT)


if __name__ == "__main__":
    main()
