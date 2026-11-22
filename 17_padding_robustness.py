# ============================================================
# 17_padding_robustness.py
# Thesis: Deep Sequential Models for ERP Anomaly Detection
# Mustafa Wasif Allvi, University of Potsdam, M.Sc. Data Science
# ============================================================
# Classifiers that read their output at the last real event. These are
# the classifiers reported in the thesis (release v2.0-submission).
#
# In the first configuration (04_models.py, 05_train.py) every prefix is
# padded with zero rows to 26 steps and all three classifiers read
# their output at position 26. The LSTM and the GRU therefore run
# through the padded steps before the read-out, and the Transformer
# attends over them without a padding mask.
#
# This script retrains the three classifiers with the padding
# handled explicitly and everything else unchanged:
#
#   LSTM, GRU    packed sequences, read-out at the last real event
#   Transformer  key-padding mask, read-out at the last real event
#
# Same data, same split, same seeds (42, 43, 44), same hyper-
# parameters, loss, optimizer, learning-rate schedule and early
# stopping: the training loop is imported from 05_train.py.
#
# The weights and results of the first configuration are NOT touched.
# New files only:
#   results/models_last_real/<model>_run<r>.pth   weights
#   results/models_last_real/<model>_run<r>.json  history and test metrics
#   results/metrics/padding_robustness.json       summary
#   results/metrics/padding_robustness_predictions.npz
# In padding_robustness.json the key "main_study" denotes the first
# configuration (read-out at position 26).
#
# Usage:
#   python 17_padding_robustness.py --smoke     2-minute test, saves nothing
#   python 17_padding_robustness.py             full run (several hours on CPU)
#   python 17_padding_robustness.py --models gru --runs 1 2
# A run that is already saved is skipped, so the script can be
# stopped and started again. Use --force to retrain.
# ============================================================

import argparse
import importlib
import importlib.util
import json
import os
import time

import joblib
import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, average_precision_score,
                             f1_score, roc_auc_score)

import config
from config import (PREPROCESSED_DIR, MODELS_DIR, METRICS_DIR, RESULTS_DIR, DEVICE,
                    BATCH_SIZE, RANDOM_SEED, LSTM_CONFIG, GRU_CONFIG, TRANSFORMER_CONFIG)

models_mod = importlib.import_module("04_models")
tr = importlib.import_module("05_train")
ex = importlib.import_module("12_export_raw_scores")
ROOT = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "ev13", os.path.join(ROOT, "13_event_level_evaluation.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

NEW_MODELS_DIR = os.path.join(RESULTS_DIR, "models_last_real")
OUT = os.path.join(METRICS_DIR, "padding_robustness.json")
OUT_PRED = os.path.join(METRICS_DIR, "padding_robustness_predictions.npz")
MODELS = ["lstm", "gru", "transformer"]


# ------------------------------------------------------------
# Models with explicit padding handling
# ------------------------------------------------------------
def real_lengths(x):
    """Number of real events per sequence. Padding rows are all zero and follow the real events."""
    return (x.abs().sum(dim=2) > 0).sum(dim=1).clamp(min=1)


class LSTMLastReal(models_mod.LSTMModel):
    def forward(self, x):
        lengths = real_lengths(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        last = self.dropout(h_n[-1])            # hidden state of the top layer at the last real event
        return self.classifier(last), self.recon_head(last)


class GRULastReal(models_mod.GRUModel):
    def forward(self, x):
        lengths = real_lengths(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        last = self.dropout(h_n[-1])
        return self.classifier(last), self.recon_head(last)


class TransformerMasked(models_mod.TransformerModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # keep the output at full length in every PyTorch version
        self.transformer.enable_nested_tensor = False
        if hasattr(self.transformer, "use_nested_tensor"):
            self.transformer.use_nested_tensor = False

    def forward(self, x):
        lengths = real_lengths(x)
        x = x[:, :int(lengths.max())]           # positions that are padding in the whole batch are dropped
        pad = torch.arange(x.size(1), device=x.device)[None, :] >= lengths[:, None]   # True = padding
        h = self.pos_encoding(self.input_proj(x))
        h = self.transformer(h, src_key_padding_mask=pad)
        last = h[torch.arange(x.size(0), device=x.device), lengths - 1]
        last = self.dropout(last)
        return self.classifier(last), self.recon_head(last)


def build_last_real(model_name, num_classes):
    name = model_name.lower()
    if name == "lstm":
        c = LSTM_CONFIG
        model = LSTMLastReal(input_dim=c["input_dim"], hidden_dim=c["hidden_dim"],
                             num_layers=c["num_layers"], num_classes=num_classes, dropout=c["dropout"])
    elif name == "gru":
        c = GRU_CONFIG
        model = GRULastReal(input_dim=c["input_dim"], hidden_dim=c["hidden_dim"],
                            num_layers=c["num_layers"], num_classes=num_classes, dropout=c["dropout"])
    elif name == "transformer":
        c = TRANSFORMER_CONFIG
        model = TransformerMasked(input_dim=c["input_dim"], d_model=c["d_model"], nhead=c["nhead"],
                                  num_layers=c["num_layers"], dim_feedforward=c["dim_feedforward"],
                                  num_classes=num_classes, dropout=c["dropout"])
    else:
        raise ValueError(model_name)
    model = model.to(DEVICE)
    print(f"  Built {model_name.upper()} (last-real-event read-out): "
          f"{sum(p.numel() for p in model.parameters()):,} parameters")
    return model


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------
@torch.no_grad()
def predict(model, X, batch=1024):
    model.eval()
    out = []
    for s in range(0, len(X), batch):
        logits, _ = model(torch.tensor(X[s:s + batch], dtype=torch.float32).to(DEVICE))
        out.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(out)


def prediction_metrics(y, p, num_classes):
    return {"accuracy_pct": float(100 * accuracy_score(y, p)),
            "f1_macro_classes_present": float(f1_score(y, p, average="macro", zero_division=0)),
            "f1_macro_all_classes": float(f1_score(y, p, average="macro", zero_division=0,
                                                   labels=np.arange(num_classes))),
            "f1_weighted": float(f1_score(y, p, average="weighted", zero_division=0))}


def detection_metrics(s, y, types_ev, plen):
    return {"roc_auc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s)),
            "per_type_length_matched":
                {t: float(ev.length_matched_auc(s, (y == 1) & (types_ev == t), y == 0, plen)[0])
                 for t in ev.TYPES}}


def summarize(values):
    return {"runs": [float(v) for v in values], "mean": float(np.mean(values)),
            "sd_population": float(np.std(values))}


# ------------------------------------------------------------
def smoke_test(data, num_classes):
    """Short end-to-end test on a small subset. Saves nothing."""
    print("SMOKE TEST: 4,000 training and 1,000 validation sequences, 2 epochs, one run per model")
    rng = np.random.default_rng(0)
    it = rng.choice(len(data["train_sequences"]), 4000, replace=False)
    iv = rng.choice(len(data["val_sequences"]), 1000, replace=False)
    train_loader = DataLoader(tr.ERPSequenceDataset(data["train_sequences"][it], data["train_targets"][it]),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(tr.ERPSequenceDataset(data["val_sequences"][iv], data["val_targets"][iv]),
                            batch_size=BATCH_SIZE, shuffle=False)
    tr.MAX_EPOCHS = 2
    X, y = data["test_sequences"][:3000], np.asarray(data["test_targets"][:3000])
    for name in MODELS:
        res = tr.train_one_run(name, 0, train_loader, val_loader, num_classes, tr.build_recon_weights())
        m = prediction_metrics(y, predict(res["model"], X), num_classes)
        # the output for a sequence must not depend on the padding or on the other sequences of the batch
        model = res["model"].eval()
        a = torch.tensor(X[:64], dtype=torch.float32)
        with torch.no_grad():
            together = model(a)[0]
            alone = torch.cat([model(a[i:i + 1, :int(real_lengths(a[i:i + 1])[0])])[0] for i in range(len(a))])
            d = (together - alone).abs().max().item()
        assert d < 1e-3, f"{name}: output changes with the padding ({d})"
        print(f"  {name:11s} ok: test accuracy after 2 short epochs {m['accuracy_pct']:.1f}%, "
              f"output independent of padding (max difference {d:.1e})")
    print("\nSmoke test passed. Start the full run with:  python 17_padding_robustness.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--runs", nargs="+", type=int, default=[1, 2, 3], choices=[1, 2, 3])
    args = ap.parse_args()

    tr.build_model = build_last_real          # the training loop of 05_train.py builds its model through this name
    data = joblib.load(os.path.join(PREPROCESSED_DIR, "sequences.joblib"))
    num_classes = int(data["num_classes"])
    if args.smoke:
        return smoke_test(data, num_classes)

    os.makedirs(NEW_MODELS_DIR, exist_ok=True)
    train_loader = DataLoader(tr.ERPSequenceDataset(data["train_sequences"], data["train_targets"]),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(tr.ERPSequenceDataset(data["val_sequences"], data["val_targets"]),
                            batch_size=BATCH_SIZE, shuffle=False)
    X_test, y_test = data["test_sequences"], np.asarray(data["test_targets"])
    recon_weights = tr.build_recon_weights()

    # ---- training (skips runs that are already saved)
    for name in args.models:
        for run in args.runs:
            wp = os.path.join(NEW_MODELS_DIR, f"{name}_run{run}.pth")
            jp = os.path.join(NEW_MODELS_DIR, f"{name}_run{run}.json")
            if os.path.exists(wp) and os.path.exists(jp) and not args.force:
                print(f"{name} run {run}: already trained, skipped")
                continue
            t0 = time.time()
            res = tr.train_one_run(name, run - 1, train_loader, val_loader, num_classes, recon_weights)
            minutes = (time.time() - t0) / 60
            torch.save(res["model"].state_dict(), wp)
            with open(jp, "w") as f:
                json.dump({"model": name, "run": run, "seed": RANDOM_SEED + run - 1,
                           "best_epoch": res["best_epoch"], "epochs_run": len(res["history"]["val_loss"]),
                           "best_val_loss": res["best_val_loss"], "training_minutes": minutes,
                           "history": res["history"]}, f)
            print(f"  saved {wp}  ({minutes:.1f} min)")

    # ---- evaluation of everything that is saved
    inj = joblib.load(os.path.join(PREPROCESSED_DIR, "test_injected.joblib"))
    lab = np.load(ev.LAB)
    y_ev, plen = lab["event_labels"], lab["prefix_length"]
    types_ev = np.where(y_ev == 1, np.asarray(inj["types"]), "none")
    X_inj = np.asarray(inj["sequences"], dtype=np.float32)

    out = {"read_out": {"main_study": "position 26, after the padded steps, no padding mask",
                        "robustness": "last real event, packed sequences (LSTM, GRU), key-padding mask (Transformer)"},
           "classes_in_vocabulary": num_classes, "classes_in_test_targets": int(len(np.unique(y_test))),
           "models": {}}
    preds = {}
    for name in MODELS:
        new_runs, old_runs, conf_new, conf_old = [], [], [], []
        for run in (1, 2, 3):
            wp = os.path.join(NEW_MODELS_DIR, f"{name}_run{run}.pth")
            if not os.path.exists(wp):
                continue
            model = build_last_real(name, num_classes)
            model.load_state_dict(torch.load(wp, map_location=DEVICE))
            p = predict(model, X_test)
            preds[f"{name}_run{run}"] = p.astype(np.int16)
            info = json.load(open(os.path.join(NEW_MODELS_DIR, f"{name}_run{run}.json")))
            r = prediction_metrics(y_test, p, num_classes)
            r.update({k: info[k] for k in ("seed", "best_epoch", "epochs_run", "training_minutes")})
            new_runs.append(r)
            conf_new.append(ex.classifier_scores(model, X_inj)[0])

            old = models_mod.build_model(name, num_classes)
            old.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"{name}_run{run}.pth"), map_location=DEVICE))
            old_runs.append(prediction_metrics(y_test, predict(old, X_test), num_classes))
            conf_old.append(ex.classifier_scores(old, X_inj)[0])
        if not new_runs:
            continue
        entry = {"last_real_event": {"runs": new_runs}, "main_study": {"runs": old_runs}}
        for key, runs in (("last_real_event", new_runs), ("main_study", old_runs)):
            for metric in ("accuracy_pct", "f1_macro_classes_present", "f1_macro_all_classes"):
                entry[key][metric] = summarize([r[metric] for r in runs])
        for key, conf in (("last_real_event", conf_new), ("main_study", conf_old)):
            conf = np.array(conf)
            entry[key]["confidence_detection"] = {
                "mean_score": detection_metrics(conf.mean(0), y_ev, types_ev, plen),
                "runs": [detection_metrics(c, y_ev, types_ev, plen) for c in conf]}
        out["models"][name] = entry

    done = [m for m in MODELS if m in out["models"] and len(out["models"][m]["last_real_event"]["runs"]) == 3]
    print("\n" + "=" * 78)
    print("Test accuracy and macro-F1 (mean of the saved runs, population SD)")
    print(f"  {'model':12s} {'read-out':18s} {'accuracy %':>22s} {'macro-F1':>22s}   runs (accuracy %)")
    for name in MODELS:
        if name not in out["models"]:
            continue
        for key, label in (("main_study", "position 26"), ("last_real_event", "last real event")):
            e = out["models"][name][key]
            a, f = e["accuracy_pct"], e["f1_macro_classes_present"]
            print(f"  {name:12s} {label:18s} {a['mean']:10.2f} +- {a['sd_population']:.2f}      "
                  f"{f['mean']:10.4f} +- {f['sd_population']:.4f}   "
                  + ", ".join(f"{v:.2f}" for v in a["runs"]))
    if len(done) == 3:
        for key, label in (("main_study", "position 26"), ("last_real_event", "last real event")):
            acc = {m: out["models"][m][key]["accuracy_pct"] for m in MODELS}
            order = sorted(MODELS, key=lambda m: -acc[m]["mean"])
            per_seed = [sorted(MODELS, key=lambda m: -acc[m]["runs"][i]) for i in range(3)]
            out.setdefault("ranking_by_accuracy", {})[key] = {"by_mean": order, "by_seed": per_seed}
            print(f"  ranking, {label:16s}: " + " > ".join(order) + "   | per seed: "
                  + " ; ".join(" > ".join(o) for o in per_seed))
        print("\nControl-flow ROC-AUC of the confidence score (length-matched), mean score and single runs")
        for name in MODELS:
            for key, label in (("main_study", "position 26"), ("last_real_event", "last real event")):
                d = out["models"][name][key]["confidence_detection"]
                print(f"  {name:12s} {label:18s} {d['mean_score']['per_type_length_matched']['control_flow']:.3f}   runs "
                      + ", ".join(f"{r['per_type_length_matched']['control_flow']:.3f}" for r in d["runs"]))
    else:
        print("  (ranking is printed once all nine runs are saved)")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    np.savez_compressed(OUT_PRED, y_true=y_test.astype(np.int16), **preds)
    print("\nSaved:", OUT)
    print("Saved:", OUT_PRED)


if __name__ == "__main__":
    main()
