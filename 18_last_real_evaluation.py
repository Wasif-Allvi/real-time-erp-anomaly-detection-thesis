# ============================================================
# 18_last_real_evaluation.py
# Thesis: Deep Sequential Models for ERP Anomaly Detection
# Mustafa Wasif Allvi, University of Potsdam, M.Sc. Data Science
# ============================================================
# Full evaluation of the classifiers that read their output at the
# last real event (trained by 17_padding_robustness.py). It repeats
# the evaluation steps of the main pipeline for these models and
# writes the same result files into a separate folder, so that
# every table and figure can be produced for both read-outs.
#
#   step                              same method as        file in results/last_real/
#   1 training histories              05_train.py           <model>_train_history.json
#   2 significance of the comparison  10_significance.py    significance_results.json
#   3 raw anomaly scores              12_export_raw_scores  raw_scores.npz
#   4 event-level detection           13_event_level_...    event_level_results.json, event_labels.npz
#   5 single-run detection metrics    15_baselines_...      single_runs.json
#   6 prediction summary and latency  08_evaluate.py        final_evaluation.json, latency_details.json
#
# The dedicated autoencoder does not depend on the read-out of the
# classifiers. Its scores are copied from results/figures/raw_scores.npz.
#
# Latency. A model that stops at the last real event is faster on a
# short prefix than on a long one. The timings therefore use a
# prefix of 26 real events (the longest possible, worst case) and,
# in addition, a prefix of 4 events (the median length).
#
# Nothing in results/metrics, results/models or results/figures is changed.
#
# Usage  : python 18_last_real_evaluation.py
#          python 18_last_real_evaluation.py --skip-latency
#          (close other programs before the latency part)
# ============================================================

import argparse
import importlib
import importlib.util
import json
import os
import platform
import time

import joblib
import numpy as np
import torch

import config
from config import (PREPROCESSED_DIR, MODELS_DIR, RESULTS_DIR, DEVICE, MODEL_INPUT_DIM, NUM_RUNS,
                    MAX_PREFIX_LENGTH, LATENCY_WARMUP_RUNS, LATENCY_MEASURE_RUNS,
                    THROUGHPUT_BATCH_SIZE, THROUGHPUT_BATCHES)

ROOT = os.path.dirname(os.path.abspath(__file__))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s17 = _load("s17", "17_padding_robustness.py")
s10 = importlib.import_module("10_significance")
ex, ev, ae_mod = s17.ex, s17.ev, s17.ex.ae_mod

OUT_DIR = os.path.join(RESULTS_DIR, "last_real")
MODELS = ["lstm", "gru", "transformer"]
REPEATS = 5
BOUND_MS = 10.0


def load_new(name, run, num_classes):
    model = s17.build_last_real(name, num_classes)
    model.load_state_dict(torch.load(os.path.join(s17.NEW_MODELS_DIR, f"{name}_run{run}.pth"),
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


# ------------------------------------------------------------
# 1. training histories in the format of 05_train.py
# ------------------------------------------------------------
def write_histories():
    print("1. Training histories")
    for name in MODELS:
        runs = []
        for run in range(1, NUM_RUNS + 1):
            d = json.load(open(os.path.join(s17.NEW_MODELS_DIR, f"{name}_run{run}.json")))
            runs.append({"run_idx": run - 1, "best_epoch": d["best_epoch"],
                         "best_val_loss": d["best_val_loss"], "history": d["history"]})
        with open(os.path.join(OUT_DIR, f"{name}_train_history.json"), "w") as f:
            json.dump(runs, f, indent=2)
        best = [r["best_epoch"] for r in runs]
        stop = [len(r["history"]["val_loss"]) for r in runs]
        print(f"  {name:11s} best epochs {best} (mean {np.mean(best):.0f}), "
              f"stopped at {stop} (mean {np.mean(stop):.0f})")


# ------------------------------------------------------------
# 2. significance tests (10_significance.py on the new models)
# ------------------------------------------------------------
def significance():
    print("\n2. Significance of the comparison")
    s10.build_model = s17.build_last_real
    s10.MODELS_DIR = s17.NEW_MODELS_DIR
    s10.METRICS_DIR = OUT_DIR
    s10.main()


# ------------------------------------------------------------
# 3. raw scores (12_export_raw_scores.py on the new models)
# ------------------------------------------------------------
def raw_scores(num_classes):
    print("\n3. Raw anomaly scores")
    inj = joblib.load(os.path.join(PREPROCESSED_DIR, "test_injected.joblib"))
    seq = joblib.load(os.path.join(PREPROCESSED_DIR, "sequences.joblib"))
    X_test, X_val = inj["sequences"], seq["val_sequences"]
    old = np.load(ex.OUT)
    out = {"case_labels": old["case_labels"], "test_ae": old["test_ae"], "val_ae": old["val_ae"]}
    assert (out["case_labels"] == np.asarray(inj["labels"])).all()
    for name in MODELS:
        ct, et, cv, evl = [], [], [], []
        for run in range(1, NUM_RUNS + 1):
            model = load_new(name, run, num_classes)
            c, e = ex.classifier_scores(model, X_test); ct.append(c); et.append(e)
            c, e = ex.classifier_scores(model, X_val); cv.append(c); evl.append(e)
            print(f"  {name} run {run} scored")
        out[f"test_conf_{name}"], out[f"test_err_{name}"] = np.array(ct), np.array(et)
        out[f"val_conf_{name}"], out[f"val_err_{name}"] = np.array(cv), np.array(evl)
    path = os.path.join(OUT_DIR, "raw_scores.npz")
    np.savez_compressed(path, **out)
    print("  Saved:", path)
    return path


# ------------------------------------------------------------
# 4. event-level evaluation (13_event_level_evaluation.py)
# ------------------------------------------------------------
def event_level(raw_path):
    print("\n4. Event-level detection")
    ev.RAW = raw_path
    ev.OUT = os.path.join(OUT_DIR, "event_level_results.json")
    ev.LAB = os.path.join(OUT_DIR, "event_labels.npz")
    ev.main()


# ------------------------------------------------------------
# 5. single runs
# ------------------------------------------------------------
def single_runs(raw_path):
    print("\n5. Detection metrics of the single runs")
    raw = np.load(raw_path)
    lab = np.load(os.path.join(OUT_DIR, "event_labels.npz"))
    inj = joblib.load(os.path.join(PREPROCESSED_DIR, "test_injected.joblib"))
    y, plen = lab["event_labels"], lab["prefix_length"]
    types_ev = np.where(y == 1, np.asarray(inj["types"]), "none")
    res = {}
    for name in MODELS:
        for kind, key in (("confidence", "test_conf"), ("joint_recon", "test_err")):
            S = raw[f"{key}_{name}"]
            res[f"{name}_{kind}"] = {"mean_score": s17.detection_metrics(S.mean(0), y, types_ev, plen),
                                     "runs": [s17.detection_metrics(s, y, types_ev, plen) for s in S]}
            r = res[f"{name}_{kind}"]
            print(f"  {name + '_' + kind:24s} ROC-AUC mean score {r['mean_score']['roc_auc']:.4f}, runs "
                  + ", ".join(f"{x['roc_auc']:.4f}" for x in r["runs"]) + " | control-flow runs "
                  + ", ".join(f"{x['per_type_length_matched']['control_flow']:.3f}" for x in r["runs"]))
    with open(os.path.join(OUT_DIR, "single_runs.json"), "w") as f:
        json.dump(res, f, indent=2)


# ------------------------------------------------------------
# 6. prediction summary and latency
# ------------------------------------------------------------
def prefix_input(n_events, batch=1):
    """A prefix with n_events real events followed by zero padding."""
    x = torch.zeros(batch, MAX_PREFIX_LENGTH, MODEL_INPUT_DIM)
    x[:, :n_events, :] = 0.5
    return x.to(DEVICE)


@torch.no_grad()
def time_once(models, x):
    for _ in range(LATENCY_WARMUP_RUNS):
        for m in models:
            m(x)
    lat = []
    for _ in range(LATENCY_MEASURE_RUNS):
        t0 = time.perf_counter()
        for m in models:
            m(x)
        lat.append((time.perf_counter() - t0) * 1000)
    return {"latency_mean_ms": float(np.mean(lat)), "latency_std_ms": float(np.std(lat)),
            "latency_p50_ms": float(np.percentile(lat, 50)), "latency_p95_ms": float(np.percentile(lat, 95))}


@torch.no_grad()
def throughput(models, n_events):
    batch = prefix_input(n_events, THROUGHPUT_BATCH_SIZE)
    for _ in range(5):
        for m in models:
            m(batch)
    t0 = time.perf_counter()
    for _ in range(THROUGHPUT_BATCHES):
        for m in models:
            m(batch)
    return float(THROUGHPUT_BATCH_SIZE * THROUGHPUT_BATCHES / (time.perf_counter() - t0))


def measure(models, n_events):
    reps = sorted((time_once(models, prefix_input(n_events)) for _ in range(REPEATS)),
                  key=lambda r: r["latency_mean_ms"])
    r = dict(reps[len(reps) // 2])            # the median repeat
    r["forward_passes"] = len(models)
    r["events_in_prefix"] = n_events
    r["throughput_events_per_sec"] = throughput(models, n_events)
    r["within_10ms_bound"] = bool(r["latency_mean_ms"] < BOUND_MS)
    return r


def line(name, r):
    flag = "within" if r["within_10ms_bound"] else "ABOVE"
    print(f"  {name:40s} {r['forward_passes']} pass(es)  mean {r['latency_mean_ms']:7.3f} ms  "
          f"p95 {r['latency_p95_ms']:7.3f} ms  {r['throughput_events_per_sec']:8,.0f} ev/s  {flag} 10 ms")


def prediction_and_latency(num_classes, skip_latency):
    print("\n6. Prediction summary and latency")
    sig = json.load(open(os.path.join(OUT_DIR, "significance_results.json")))["per_run"]
    seq = joblib.load(os.path.join(PREPROCESSED_DIR, "sequences.joblib"))
    X, y = seq["test_sequences"], np.asarray(seq["test_targets"])
    final = {"prediction": {}, "latency": {}}
    nets = {name: [load_new(name, r, num_classes) for r in range(1, NUM_RUNS + 1)] for name in MODELS}
    nets["autoencoder"] = [load_autoencoder(r) for r in range(1, ae_mod.AE_RUNS + 1)]
    for name in MODELS:
        f1w, f1m37 = [], []
        for model in nets[name]:
            m = s17.prediction_metrics(y, s17.predict(model, X), num_classes)
            f1w.append(m["f1_weighted"]); f1m37.append(m["f1_macro_all_classes"])
        final["prediction"][name] = {
            "model": name, "test_acc_mean": sig[name]["acc_mean"], "test_acc_std": sig[name]["acc_sd_population"],
            "test_f1w_mean": float(np.mean(f1w)), "test_f1m_mean": sig[name]["f1m_mean"],
            "test_f1m_std": float(np.std(sig[name]["f1_macro"])),
            "test_f1m_all_37_classes_mean": float(np.mean(f1m37)), "n_runs": NUM_RUNS}
        p = final["prediction"][name]
        print(f"  {name:11s} accuracy {p['test_acc_mean']:.2f} +- {p['test_acc_std']:.2f}  "
              f"macro-F1 {p['test_f1m_mean']:.4f} +- {p['test_f1m_std']:.4f}  weighted F1 {p['test_f1w_mean']:.4f}")

    if not skip_latency:
        details = {"protocol": {"warmup_runs": LATENCY_WARMUP_RUNS, "measured_runs": LATENCY_MEASURE_RUNS,
                                "repeats": REPEATS, "reported": "median repeat", "device": str(DEVICE),
                                "torch_threads": torch.get_num_threads(), "torch": torch.__version__,
                                "processor": platform.processor()},
                   "worst_case_26_events": {}, "median_case_4_events": {}}
        for label, n in (("worst_case_26_events", MAX_PREFIX_LENGTH), ("median_case_4_events", 4)):
            print(f"\n  Prefix of {n} real events")
            d = details[label]
            d["single"] = {k: [measure([m], n) for m in nets[k]] for k in MODELS + ["autoencoder"]}
            for k in MODELS + ["autoencoder"]:
                line(f"{k} (run 1)", d["single"][k][0])
            d["three_runs"] = {k: measure(nets[k], n) for k in MODELS + ["autoencoder"]}
            for k in MODELS + ["autoencoder"]:
                line(f"{k} x 3", d["three_runs"][k])
            d["pair"] = {k: measure([nets[k][0], nets["autoencoder"][0]], n) for k in MODELS}
            for k in MODELS:
                line(f"{k} + autoencoder", d["pair"][k])
            d["all"] = {k: measure(nets[k] + nets["autoencoder"], n) for k in MODELS}
            for k in MODELS:
                line(f"{k} x 3 + autoencoder x 3", d["all"][k])
        for k in MODELS + ["autoencoder"]:      # table format of 08_evaluate.py: run 1, worst case
            r = dict(details["worst_case_26_events"]["single"][k][0])
            r["parameters"] = int(sum(p.numel() for p in nets[k][0].parameters()))
            final["latency"][k] = r
        with open(os.path.join(OUT_DIR, "latency_details.json"), "w") as f:
            json.dump(details, f, indent=2)
    with open(os.path.join(OUT_DIR, "final_evaluation.json"), "w") as f:
        json.dump(final, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-latency", action="store_true")
    args = ap.parse_args()
    missing = [f"{m}_run{r}" for m in MODELS for r in range(1, NUM_RUNS + 1)
               if not os.path.exists(os.path.join(s17.NEW_MODELS_DIR, f"{m}_run{r}.pth"))]
    if missing:
        raise SystemExit("Run 17_padding_robustness.py first. Missing: " + ", ".join(missing))
    os.makedirs(OUT_DIR, exist_ok=True)
    num_classes = int(joblib.load(os.path.join(PREPROCESSED_DIR, "sequences.joblib"))["num_classes"])

    write_histories()
    significance()
    raw_path = raw_scores(num_classes)
    event_level(raw_path)
    single_runs(raw_path)
    prediction_and_latency(num_classes, args.skip_latency)
    print("\nAll files written to", OUT_DIR)


if __name__ == "__main__":
    main()
