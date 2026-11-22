# ============================================================
# 14_robustness_checks.py
# Thesis: Deep Sequential Models for ERP Anomaly Detection
# Mustafa Wasif Allvi, University of Potsdam, M.Sc. Data Science
# ============================================================
# Two robustness checks of the event-level detection results
# produced by 13_event_level_evaluation.py (classifier scores of the
# first configuration):
#
#   A. Cross-entity class restricted to the altered cases in which
#      the fivefold inflation pushes the declared amount above the
#      permit budget (22 of 24). The two other cases are left out of
#      the class (they stay anomalous, so overall metrics are unchanged).
#
#   B. Temporal anomalies split into two subgroups:
#      - gap only:  the backdated event keeps its position,
#      - reordered: the backdated event moves ahead of its predecessor,
#                   so the categorical rows of the case are permuted.
#
# Both checks reuse the scores, event-level labels, length-matched
# per-type ROC-AUC, case-level bootstrap and held-out thresholds of
# 13_event_level_evaluation.py, with the same seeds. No PyTorch needed.
#
# Inputs : results/figures/raw_scores.npz, results/metrics/event_labels.npz,
#          results/metrics/event_level_results.json,
#          results/preprocessed/test_injected.joblib, model_features.joblib,
#          events_processed_test.csv   (stages 01-03 and 06)
# Output : results/metrics/robustness_checks.json
#
# Usage  : python 14_robustness_checks.py            (both checks)
#          python 14_robustness_checks.py --part A   (only one check,
#          python 14_robustness_checks.py --part B    merged into the file)
# ============================================================

import argparse
import importlib.util
import json
import os

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "ev13", os.path.join(ROOT, "13_event_level_evaluation.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

PRE = os.path.join(ROOT, "results", "preprocessed")
OUT = os.path.join(ROOT, "results", "metrics", "robustness_checks.json")
N_CAT = 7   # the first seven model features are categorical codes
TYPES_B = ["control_flow", "data", "resource",
           "temporal_gap_only", "temporal_reordered", "cross_entity"]


def load():
    inj = joblib.load(os.path.join(PRE, "test_injected.joblib"))
    mf = list(joblib.load(os.path.join(PRE, "model_features.joblib")))
    raw = np.load(ev.RAW)
    lab = np.load(ev.LAB)
    cases = np.asarray(inj["case_ids"])
    types = np.asarray(inj["types"]).astype("<U20")
    assert (raw["case_labels"] == np.asarray(inj["labels"])).all()
    S = {}
    for m in ev.MODELS:
        S[f"{m}_confidence"] = raw[f"test_conf_{m}"].mean(0)
        S[f"{m}_joint_recon"] = raw[f"test_err_{m}"].mean(0)
    S["autoencoder"] = raw["test_ae"].mean(0)
    df = pd.read_csv(os.path.join(PRE, "events_processed_test.csv"), low_memory=False)
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)
    G = {c: g.sort_values("time:timestamp") for c, g in df.groupby("case:concept:name")}
    return inj["sequences"], mf, cases, types, lab["event_labels"], lab["prefix_length"], S, G


def check_cross_entity(cases, types, y, plen, S, G):
    case_type = {c: t for c, t in zip(cases, types) if t != "none"}
    ce = sorted(set(cases[(types == "cross_entity") & (y == 1)]))
    ratio = {c: 5 * float(G[c]["case:Amount"].iloc[0])
             / float(G[c]["case:Permit RequestedBudget"].iloc[0]) for c in ce}
    below = sorted(c for c, r in ratio.items() if r <= 1.0)

    types_a = types.copy()
    types_a[np.isin(cases, below)] = "excluded"
    ct_a = {c: ("excluded" if c in below else t) for c, t in case_type.items()}
    reported = json.load(open(ev.OUT))["scorers"]

    per_type = {}
    for k, (name, s) in enumerate(S.items()):
        red = ev.per_type_auc(s, y, types_a, plen, cases, ct_a, k)["cross_entity"]
        per_type[name] = {"all_altered_cases": reported[name]["per_type_length_matched"]["cross_entity"],
                          "overrun_cases_only": red}
        print(f"  {name:24s} cross-entity ROC-AUC "
              f"{per_type[name]['all_altered_cases']['auc']:.4f} -> {red['auc']:.4f}")

    keep = ~np.isin(cases, below)
    folds = ev.make_folds(cases[keep], {c: t for c, t in ct_a.items() if c not in below}, ev.N_REP)
    cf = ev.crossfit(S["autoencoder"][keep], y[keep], types[keep], folds)
    return {"altered_cases": len(ce),
            "below_budget_cases": below,
            "amount_to_budget_ratio_after_inflation": {c: ratio[c] for c in below},
            "median_ratio_all_altered": float(np.median(list(ratio.values()))),
            "prefixes_left_out": int(np.isin(cases, below).sum()),
            "per_type_auc": per_type,
            "autoencoder_detection_rate_pct": {
                "all_altered_cases": reported["autoencoder"]["crossfit"]["per_type_rate_pct"]["cross_entity"],
                "overrun_cases_only": cf["per_type_rate_pct"]["cross_entity"]}}


def check_temporal(X, mf, cases, types, y, plen, S, G):
    reordered = []
    temporal_cases = sorted(set(cases[types == "temporal"]))
    for c in temporal_cases:
        idx = np.where(cases == c)[0]
        last = idx[np.argmax(plen[idx])]
        n = plen[last]
        orig = G[c][mf].values[:n].astype(np.float32)
        if not np.allclose(X[last, :n, :N_CAT], orig[:, :N_CAT]):
            reordered.append(c)

    types_b = types.copy()
    is_t = types == "temporal"
    types_b[is_t & np.isin(cases, reordered)] = "temporal_reordered"
    types_b[is_t & ~np.isin(cases, reordered)] = "temporal_gap_only"
    ct_b = {c: t for c, t in zip(cases, types_b) if t != "none"}

    saved = ev.TYPES
    ev.TYPES = TYPES_B
    try:
        groups = {}
        for t in ["temporal_gap_only", "temporal_reordered"]:
            groups[t] = {"cases": int(len(set(cases[types_b == t]))),
                         "anomalous_prefixes": int(((y == 1) & (types_b == t)).sum())}
        per_type = {}
        for k, (name, s) in enumerate(S.items()):
            r = ev.per_type_auc(s, y, types_b, plen, cases, ct_b, k)
            per_type[name] = {t: r[t] for t in ["temporal_gap_only", "temporal_reordered"]}
            print(f"  {name:24s} gap only {r['temporal_gap_only']['auc']:.3f} | "
                  f"reordered {r['temporal_reordered']['auc']:.3f}")
        folds = ev.make_folds(cases, ct_b, ev.N_REP)
        rates = {}
        for name in ["gru_confidence", "gru_joint_recon", "autoencoder"]:
            cf = ev.crossfit(S[name], y, types_b, folds)
            rates[name] = {t: cf["per_type_rate_pct"][t] for t in ["temporal_gap_only", "temporal_reordered"]}
            rates[name]["normal_flagged_pct"] = cf["normal_flagged_pct"]
    finally:
        ev.TYPES = saved
    return {"temporal_cases": len(temporal_cases), "subgroups": groups,
            "reordered_cases": reordered, "per_type_auc": per_type,
            "detection_rate_pct": rates}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A", "B", "all"], default="all")
    part = ap.parse_args().part

    X, mf, cases, types, y, plen, S, G = load()
    out = json.load(open(OUT)) if (part != "all" and os.path.exists(OUT)) else {}
    if part in ("A", "all"):
        print("A. Cross-entity class without the two below-budget cases")
        out["cross_entity_overrun_only"] = check_cross_entity(cases, types, y, plen, S, G)
    if part in ("B", "all"):
        print("B. Temporal anomalies split by reordering")
        out["temporal_subgroups"] = check_temporal(X, mf, cases, types, y, plen, S, G)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
