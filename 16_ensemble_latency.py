# ============================================================
# 16_ensemble_latency.py
# Thesis: Deep Sequential Models for ERP Anomaly Detection
# Mustafa Wasif Allvi, University of Potsdam, M.Sc. Data Science
# ============================================================
# Latency of the configurations that produce the reported detection
# scores. 08_evaluate.py times ONE forward pass of run 1 of every
# model. The detection metrics of 13_event_level_evaluation.py use
# the mean score of the three runs of a model, which costs three
# forward passes per event. This script measures both with the
# protocol of 08_evaluate.py (one sequence of 26 steps, warm-up
# runs, then timed runs, CPU):
#
#   single   one forward pass, for each of the three runs
#   three    the three runs one after the other
#   pair     one classifier and one autoencoder one after the other
#   all      three classifier runs and three autoencoder runs
#
# Nothing is retrained and no existing result file is changed.
#
# Output : results/metrics/ensemble_latency.json
# Usage  : python 16_ensemble_latency.py
#          (close other programs first, the timings react to load)
# ============================================================

import importlib
import json
import os
import platform
import time

import joblib
import numpy as np
import torch

import config
from config import (MODELS_DIR, METRICS_DIR, DEVICE, MODEL_INPUT_DIM, NUM_RUNS,
                    MAX_PREFIX_LENGTH, LATENCY_WARMUP_RUNS, LATENCY_MEASURE_RUNS,
                    THROUGHPUT_BATCH_SIZE, THROUGHPUT_BATCHES)

build_model = importlib.import_module("04_models").build_model
ae_mod = importlib.import_module("05b_autoencoder")

OUT = os.path.join(METRICS_DIR, "ensemble_latency.json")
MODELS = ["lstm", "gru", "transformer"]
REPEATS = 5          # the whole measurement is repeated, the median repeat is reported
BOUND_MS = 10.0      # real-time criterion of the thesis (DR4)


def load_models():
    num_classes = len(joblib.load(config.ENCODER_PATH)["concept:name"].classes_)
    nets = {}
    for name in MODELS:
        nets[name] = []
        for run in range(1, NUM_RUNS + 1):
            m = build_model(name, num_classes)
            m.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"{name}_run{run}.pth"),
                                         map_location=DEVICE))
            m.eval()
            nets[name].append(m)
    nets["autoencoder"] = []
    for run in range(1, ae_mod.AE_RUNS + 1):
        ae = ae_mod.GRUAutoencoder(input_dim=MODEL_INPUT_DIM, n_numerical=ae_mod.N_NUMERICAL,
                                   hidden_dim=ae_mod.AE_HIDDEN_DIM, num_layers=ae_mod.AE_NUM_LAYERS,
                                   dropout=ae_mod.AE_DROPOUT).to(DEVICE)
        ae.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"autoencoder_run{run}.pth"),
                                      map_location=DEVICE))
        ae.eval()
        nets["autoencoder"].append(ae)
    return nets


@torch.no_grad()
def time_once(models):
    """Protocol of 08_evaluate.measure_latency, applied to a list of models
    that are called one after the other for the same event."""
    single = torch.zeros(1, MAX_PREFIX_LENGTH, MODEL_INPUT_DIM).to(DEVICE)
    for _ in range(LATENCY_WARMUP_RUNS):
        for m in models:
            m(single)
    lat = []
    for _ in range(LATENCY_MEASURE_RUNS):
        t0 = time.perf_counter()
        for m in models:
            m(single)
        lat.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(lat)), float(np.percentile(lat, 95))


@torch.no_grad()
def throughput(models):
    batch = torch.zeros(THROUGHPUT_BATCH_SIZE, MAX_PREFIX_LENGTH, MODEL_INPUT_DIM).to(DEVICE)
    for _ in range(5):
        for m in models:
            m(batch)
    t0 = time.perf_counter()
    for _ in range(THROUGHPUT_BATCHES):
        for m in models:
            m(batch)
    return float(THROUGHPUT_BATCH_SIZE * THROUGHPUT_BATCHES / (time.perf_counter() - t0))


def measure(models):
    reps = sorted(time_once(models) for _ in range(REPEATS))
    mean_ms, p95_ms = reps[len(reps) // 2]
    return {"forward_passes": len(models), "latency_mean_ms": mean_ms, "latency_p95_ms": p95_ms,
            "latency_mean_ms_all_repeats": [r[0] for r in reps],
            "throughput_events_per_sec": throughput(models),
            "within_10ms_bound": bool(mean_ms < BOUND_MS)}


def line(name, r):
    flag = "within" if r["within_10ms_bound"] else "ABOVE"
    print(f"  {name:46s} {r['forward_passes']} pass(es)  mean {r['latency_mean_ms']:7.3f} ms  "
          f"p95 {r['latency_p95_ms']:7.3f} ms  {r['throughput_events_per_sec']:8,.0f} ev/s  {flag} 10 ms")


def main():
    nets = load_models()
    out = {"protocol": {"sequence_length": MAX_PREFIX_LENGTH, "warmup_runs": LATENCY_WARMUP_RUNS,
                        "measured_runs": LATENCY_MEASURE_RUNS, "repeats": REPEATS,
                        "reported": "median repeat of the mean latency",
                        "device": str(DEVICE), "torch_threads": torch.get_num_threads(),
                        "torch": torch.__version__, "processor": platform.processor()},
           "single": {}, "three_runs": {}, "pair_one_classifier_one_autoencoder": {},
           "all_three_classifier_and_three_autoencoder_runs": {}}

    print("Single models (one forward pass, as in Table 15)")
    for name in MODELS + ["autoencoder"]:
        out["single"][name] = [measure([m]) for m in nets[name]]
        for i, r in enumerate(out["single"][name], 1):
            line(f"{name} run {i}", r)

    print("\nMean score of the three runs of one model (three forward passes)")
    for name in MODELS + ["autoencoder"]:
        out["three_runs"][name] = measure(nets[name])
        line(f"{name} x 3", out["three_runs"][name])

    print("\nOne classifier and one autoencoder (run 1 each), both channels")
    for name in MODELS:
        r = measure([nets[name][0], nets["autoencoder"][0]])
        out["pair_one_classifier_one_autoencoder"][name] = r
        line(f"{name} + autoencoder", r)

    print("\nBoth channels with the mean score of three runs each (six forward passes)")
    for name in MODELS:
        r = measure(nets[name] + nets["autoencoder"])
        out["all_three_classifier_and_three_autoencoder_runs"][name] = r
        line(f"{name} x 3 + autoencoder x 3", r)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved:", OUT)


if __name__ == "__main__":
    main()
