# ============================================================
# 13_event_level_evaluation.py
# Final anomaly-detection evaluation reported in Chapter 7
#
# 1. Event-level ground truth. A prefix counts as anomalous only
#    if it contains at least one event altered by the injection
#    (06_anomaly_injection.py). Earlier prefixes of an injected
#    case are identical to the original case and count as normal,
#    because a real-time detector can only react to events that
#    have arrived. The first altered position of every injected
#    case is found by comparing it with its original in
#    events_processed_test.csv. Cases whose mutation changed
#    nothing (declared amount of zero) therefore stay normal.
# 2. Scores come from 12_export_raw_scores.py (three runs each,
#    averaged). Joint reconstruction is ranked on its raw error.
#    For the composite it is divided by the 99th percentile of
#    the errors on the clean VALIDATION prefixes and clipped to
#    [0, 1], so no test information enters the normalisation.
# 3. Overall ROC-AUC and AUPRC on all 22,238 test prefixes.
# 4. Threshold metrics (F1, precision, recall, per-type rates)
#    with cross-fitted thresholds: the test cases are split into
#    two halves (stratified by anomaly type), the F1-optimal
#    threshold is chosen on one half and applied to the other,
#    and the split is repeated 1,000 times (100 times for the
#    held-out choice of the composite weight).
# 5. Per-type ROC-AUC, length-matched: every anomalous prefix of
#    a type is compared only with normal prefixes of the same
#    length, since all scores depend on prefix length. 95%
#    intervals from a case-level cluster bootstrap (1,000).
# 6. Composite grid alpha in {0, 0.1, ..., 1} and held-out alpha
#    selection with the same case split as in 4.
# 7. Classical baselines (same settings as 09_baseline.py) and
#    the metrics of each autoencoder run.
#
# Inputs : results/figures/raw_scores.npz        (12_export_raw_scores.py)
#          results/preprocessed/test_injected.joblib
#          results/preprocessed/events_processed_test.csv
#          results/preprocessed/model_features.joblib
#          results/preprocessed/sequences.joblib (baselines)
# Outputs: results/metrics/event_level_results.json
#          results/metrics/event_labels.npz
# Runs without PyTorch.
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
PRE = os.path.join(RES, "preprocessed")
RAW = os.path.join(RES, "figures", "raw_scores.npz")
OUT = os.path.join(RES, "metrics", "event_level_results.json")
LAB = os.path.join(RES, "metrics", "event_labels.npz")

SEED = 42
N_REP = 1000        # repetitions of the two-fold case split (thresholds)
N_REP_ALPHA = 100   # repetitions for the held-out choice of the composite weight
N_BOOT = 1000       # cluster bootstrap resamples
TYPES = ["control_flow", "data", "resource", "temporal", "cross_entity"]
MODELS = ["lstm", "gru", "transformer"]
ALPHAS = [round(a, 1) for a in np.arange(0.0, 1.01, 0.1)]


# ------------------------------------------------------------
# 1. Event-level labels
# ------------------------------------------------------------
def event_level_labels(inj, model_features):
    cases = np.asarray(inj["case_ids"])
    y_case = np.asarray(inj["labels"])
    X = inj["sequences"]

    # prefix length = position of the prefix within its case (as built in 06)
    plen = np.zeros(len(cases), int)
    seen = {}
    for i, c in enumerate(cases):
        seen[c] = seen.get(c, 0) + 1
        plen[i] = seen[c]

    df = pd.read_csv(os.path.join(PRE, "events_processed_test.csv"), low_memory=False)
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)
    injected = set(cases[y_case == 1])
    clean = {c: g.sort_values("time:timestamp")[model_features].values.astype(np.float32)
             for c, g in df[df["case:concept:name"].isin(injected)].groupby("case:concept:name")}

    first_altered = {}
    for c in injected:
        idx = np.where(cases == c)[0]
        longest = idx[np.argmax(plen[idx])]
        n = plen[longest]
        rows, ref = X[longest, :n, :], clean[c][:n]
        k = next((j for j in range(n) if not np.allclose(rows[j], ref[j], atol=1e-5)), None)
        first_altered[c] = k

    y_ev = np.zeros(len(cases), int)
    for i in np.where(y_case == 1)[0]:
        k = first_altered[cases[i]]
        y_ev[i] = int(k is not None and plen[i] >= k + 1)
    return y_ev, plen, first_altered


# ------------------------------------------------------------
# Threshold helpers
# ------------------------------------------------------------
def f1_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)[:-1]
    return t[int(np.argmax(f1))]


def make_folds(cases, case_type, n_rep):
    """Two folds over cases per repetition, stratified by injected type
    (normal cases form their own stratum). Returns an (n_rep, n_prefixes) array."""
    uc, inv = np.unique(cases, return_inverse=True)
    ctype = np.array([case_type.get(c, "none") for c in uc])
    folds = np.zeros((n_rep, len(cases)), np.int8)
    for rep in range(n_rep):
        rng = np.random.default_rng([SEED, rep])
        fold_c = np.zeros(len(uc), np.int8)
        for t in TYPES + ["none"]:
            perm = rng.permutation(np.where(ctype == t)[0])
            fold_c[perm[1::2]] = 1
        folds[rep] = fold_c[inv]
    return folds


def crossfit(s, y, types, folds):
    """Mean over all repetitions of pooled held-out metrics."""
    rec = {"precision": [], "recall": [], "f1": [], "normal_flagged_pct": []}
    per_type = {t: [] for t in TYPES}
    for fold in folds:
        pred = np.zeros(len(s), bool)
        for f in (0, 1):
            tr, te = fold != f, fold == f
            pred[te] = s[te] >= f1_threshold(y[tr], s[tr])
        tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum(); fn = (~pred & (y == 1)).sum()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn)
        rec["precision"].append(p); rec["recall"].append(r)
        rec["f1"].append(2 * p * r / (p + r) if p + r else 0.0)
        rec["normal_flagged_pct"].append(100 * fp / (y == 0).sum())
        for t in TYPES:
            m = (y == 1) & (types == t)
            per_type[t].append(100 * pred[m].mean())
    out = {k: float(np.mean(v)) for k, v in rec.items()}
    out.update({k + "_sd": float(np.std(v)) for k, v in rec.items()})
    out["per_type_rate_pct"] = {t: float(np.mean(v)) for t, v in per_type.items()}
    return out


# ------------------------------------------------------------
# Length-matched AUC (vectorised over bootstrap weights)
# ------------------------------------------------------------
def _num_den(s, pos, W):
    """W: (B, n) weights. Returns per-row numerator and denominator of the AUC."""
    o = np.argsort(s, kind="mergesort")
    s, pos, W = s[o], pos[o], W[:, o]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    gp = np.add.reduceat(W * pos, starts, axis=1)
    gn = np.add.reduceat(W * ~pos, starts, axis=1)
    above = gp[:, ::-1].cumsum(1)[:, ::-1] - gp
    return (gn * (above + 0.5 * gp)).sum(1), gp.sum(1) * gn.sum(1)


def length_matched_auc(s, pos_mask, neg_mask, plen, W=None):
    num = 0.0; den = 0.0
    for L in np.unique(plen[pos_mask]):
        idx = np.where((pos_mask | neg_mask) & (plen == L))[0]
        if neg_mask[idx].sum() == 0:
            continue
        w = np.ones((1, len(idx))) if W is None else W[:, idx]
        n_, d_ = _num_den(s[idx], pos_mask[idx], w)
        num = num + n_; den = den + d_
    return num / den


def per_type_auc(s, y, types, plen, cases, case_type, key_idx):
    uc, inv = np.unique(cases, return_inverse=True)
    ctype = np.array([case_type.get(c, "none") for c in uc])
    res = {}
    for t in TYPES:
        pos = (y == 1) & (types == t)
        neg = (y == 0)
        point = float(length_matched_auc(s, pos, neg, plen)[0])
        # cluster bootstrap: type-t cases and all other cases resampled separately
        rng = np.random.default_rng([SEED, key_idx, TYPES.index(t)])
        a_ids = np.where(ctype == t)[0]; o_ids = np.where(ctype != t)[0]
        boots = []
        for chunk in range(0, N_BOOT, 100):
            B = min(100, N_BOOT - chunk)
            mult = np.zeros((B, len(uc)))
            for b in range(B):
                mult[b] += np.bincount(rng.choice(a_ids, len(a_ids)), minlength=len(uc))
                mult[b] += np.bincount(rng.choice(o_ids, len(o_ids)), minlength=len(uc))
            boots.append(length_matched_auc(s, pos, neg, plen, W=mult[:, inv]))
        lo, hi = np.percentile(np.concatenate(boots), [2.5, 97.5])
        res[t] = {"auc": point, "ci95": [float(lo), float(hi)], "n_prefixes": int(pos.sum())}
    res["all"] = float(length_matched_auc(s, y == 1, y == 0, plen)[0])
    return res


# ------------------------------------------------------------
# Baselines (identical settings to 09_baseline.py)
# ------------------------------------------------------------
def aggregate(seq):
    mask = (np.abs(seq).sum(axis=2) > 0).astype(np.float32)
    return (seq * mask[:, :, None]).sum(1) / np.clip(mask.sum(1, keepdims=True), 1, None)


def baseline_scores(X_test):
    data = joblib.load(os.path.join(PRE, "sequences.joblib"))
    X_train = aggregate(data["train_sequences"])
    Xt = aggregate(X_test)
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=SEED, n_jobs=-1).fit(X_train)
    idx = np.random.default_rng(SEED).choice(len(X_train), size=min(10000, len(X_train)), replace=False)
    svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(X_train[idx])
    return {"isolation_forest": -iso.score_samples(Xt), "one_class_svm": -svm.score_samples(Xt)}


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    inj = joblib.load(os.path.join(PRE, "test_injected.joblib"))
    mf = joblib.load(os.path.join(PRE, "model_features.joblib"))
    raw = np.load(RAW)
    cases = np.asarray(inj["case_ids"]); types = np.asarray(inj["types"])
    y_case = np.asarray(inj["labels"])
    assert (raw["case_labels"] == y_case).all(), "raw scores and injected test set differ"

    y, plen, first_altered = event_level_labels(inj, mf)
    case_type = {c: t for c, t in zip(cases, types) if t != "none"}
    types_ev = np.where(y == 1, types, "none")
    np.savez(LAB, event_labels=y, prefix_length=plen)

    # ---- label summary
    summary = {"n_prefixes": int(len(y)), "n_anomalous": int(y.sum()), "base_rate": float(y.mean()),
               "unchanged_cases": sorted([c for c, k in first_altered.items() if k is None]),
               "per_type": {}}
    for t in TYPES:
        m = types == t
        summary["per_type"][t] = {
            "cases": int(len(np.unique(cases[m]))),
            "cases_altered": int(sum(1 for c in np.unique(cases[m]) if first_altered[c] is not None)),
            "prefixes_of_cases": int(m.sum()), "anomalous_prefixes": int((m & (y == 1)).sum())}
    print("Event-level labels:", summary["n_anomalous"], "of", summary["n_prefixes"],
          f"({100 * summary['base_rate']:.2f}%)")
    for t, v in summary["per_type"].items():
        print(f"  {t:13} {v}")

    # ---- scores
    S = {}
    p99 = {}
    for m in MODELS:
        S[f"{m}_confidence"] = raw[f"test_conf_{m}"].mean(0)
        err = raw[f"test_err_{m}"].mean(0)
        S[f"{m}_joint_recon"] = err
        p99[m] = float(np.percentile(raw[f"val_err_{m}"].mean(0), 99))
        S[f"_{m}_recon_norm"] = np.clip(err / p99[m], 0, 1)
    S["autoencoder"] = raw["test_ae"].mean(0)
    S.update(baseline_scores(inj["sequences"]))

    folds = make_folds(cases, case_type, N_REP)
    results = {"labels": summary, "p99_validation": p99, "n_rep": N_REP, "n_rep_alpha": N_REP_ALPHA, "scorers": {}}
    report = [k for k in S if not k.startswith("_")]
    for k in report:
        s = S[k]
        r = {"roc_auc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s))}
        r["crossfit"] = crossfit(s, y, types_ev, folds)
        results["scorers"][k] = r
        print(f"{k:24} ROC {r['roc_auc']:.4f} AUPRC {r['auprc']:.4f} | held-out P {r['crossfit']['precision']:.3f} "
              f"R {r['crossfit']['recall']:.3f} F1 {r['crossfit']['f1']:.3f} normal flagged {r['crossfit']['normal_flagged_pct']:.1f}%")

    # ---- per-type, length-matched
    pt_keys = [f"{m}_confidence" for m in MODELS] + [f"{m}_joint_recon" for m in MODELS] + ["autoencoder"]
    for key_idx, k in enumerate(pt_keys):
        results["scorers"][k]["per_type_length_matched"] = per_type_auc(S[k], y, types_ev, plen, cases, case_type, key_idx)
        pt = results["scorers"][k]["per_type_length_matched"]
        print(f"{k:24} all {pt['all']:.3f} | " + "  ".join(
            f"{t[:5]} {pt[t]['auc']:.3f} [{pt[t]['ci95'][0]:.2f},{pt[t]['ci95'][1]:.2f}]" for t in TYPES))

    # ---- composite grid and held-out alpha selection
    results["composite"] = {}
    for m in MODELS:
        conf, rn = S[f"{m}_confidence"], S[f"_{m}_recon_norm"]
        grid = {a: float(roc_auc_score(y, a * conf + (1 - a) * rn)) for a in ALPHAS}
        chosen, held, recon_only = [], [], []
        for fold in folds[:N_REP_ALPHA]:
            for f in (0, 1):
                tr, te = fold != f, fold == f
                a = max(ALPHAS, key=lambda a: roc_auc_score(y[tr], a * conf[tr] + (1 - a) * rn[tr]))
                chosen.append(a)
                held.append(roc_auc_score(y[te], a * conf[te] + (1 - a) * rn[te]))
                recon_only.append(roc_auc_score(y[te], rn[te]))
        results["composite"][m] = {"grid_roc_auc": {str(a): v for a, v in grid.items()},
                                   "best_alpha": max(grid, key=grid.get),
                                   "heldout_alpha_counts": {str(a): chosen.count(a) for a in sorted(set(chosen))},
                                   "heldout_roc_auc_selected": float(np.mean(held)),
                                   "heldout_roc_auc_recon_only": float(np.mean(recon_only))}
        print(m, "composite grid", {a: round(v, 4) for a, v in grid.items()})

    # ---- autoencoder: per run, and top of the ranking
    results["autoencoder_runs"] = [{"roc_auc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s))}
                                   for s in raw["test_ae"]]
    s = S["autoencoder"]; o = np.argsort(-s, kind="mergesort"); yy = y[o]
    first_normal = int(np.argmax(yy == 0))
    prec = np.cumsum(yy) / np.arange(1, len(yy) + 1); recall = np.cumsum(yy) / yy.sum()
    above60 = np.where(prec >= 0.60)[0]
    results["autoencoder_top"] = {"pure_top_n": first_normal, "recall_at_pure_top": float(recall[first_normal - 1]),
                                  "max_recall_precision_ge_0.60": float(recall[above60.max()])}

    # ---- score dependence on prefix length (normal prefixes)
    results["normal_mean_by_length"] = {k: {int(L): float(S[k][(y == 0) & (plen == L)].mean()) for L in range(1, 9)}
                                       for k in ["gru_confidence", "gru_joint_recon", "autoencoder"]}

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved:", OUT)


if __name__ == "__main__":
    main()
