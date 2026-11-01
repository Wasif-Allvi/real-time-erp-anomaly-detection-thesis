# ============================================================
# 11_per_type_auc.py
# SUPERSEDED by 13_event_level_evaluation.py, which the thesis reports.
# This earlier version labels every prefix of an injected case as
# anomalous and compares each type with all normal prefixes. Kept for
# reference only.
#
# Threshold-free per-type detection analysis
#
# For every scorer and every injected anomaly type, computes the
# ROC-AUC of "sequences of that type" against "normal sequences"
# (0.5 = no discrimination). Unlike a detection rate at one
# threshold, this does not depend on how many normal sequences
# the scorer flags at the same time.
#
# Uncertainty: 95% case-level cluster bootstrap. Prefixes of one
# case are not independent, so whole anomalous cases and whole
# normal cases are resampled.
#
# Also reports, for each scorer, the share of normal sequences
# flagged at its F1-optimal threshold (from final_evaluation.json),
# which is the context needed to read per-type detection rates.
#
# Inputs : results/figures/score_cache.npz   (written by generate_plots.py)
#          results/preprocessed/test_injected.joblib
#          results/metrics/final_evaluation.json
# Output : results/metrics/per_type_auc.json
# Runs without PyTorch.
# ============================================================

import os
import json
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
CACHE = os.path.join(RES, "figures", "score_cache.npz")
INJ = os.path.join(RES, "preprocessed", "test_injected.joblib")
FINAL = os.path.join(RES, "metrics", "final_evaluation.json")
OUT = os.path.join(RES, "metrics", "per_type_auc.json")

TYPES = ["control_flow", "data", "resource", "temporal", "cross_entity"]
SCORERS = {  # cache key -> key in final_evaluation.json
    "lstm_confidence": "lstm_confidence",
    "gru_confidence": "gru_confidence",
    "transformer_confidence": "transformer_confidence",
    "lstm_recon_norm": "lstm_joint_recon",
    "gru_recon_norm": "gru_joint_recon",
    "transformer_recon_norm": "transformer_joint_recon",
    "autoencoder": "separate_autoencoder",
}
ALL_SCORERS = tuple(SCORERS)
N_BOOT = 1000
SEED = 42


def main():
    sc = np.load(CACHE)
    inj = joblib.load(INJ)
    y = np.asarray(inj["labels"])
    types = np.asarray(inj["types"])
    cases = np.asarray(inj["case_ids"])
    assert (y == sc["true_binary"]).all(), "score cache and injected test set differ"
    final = json.load(open(FINAL))["anomaly"]

    n_normal = int((y == 0).sum())
    case_codes, case_of_seq = np.unique(cases, return_inverse=True)
    normal_case_ids = np.unique(case_of_seq[y == 0])

    def cluster_weights(rng, case_ids):
        # bootstrap multiplicity of every case, mapped to its sequences
        draw = rng.choice(case_ids, len(case_ids))
        mult = np.bincount(draw, minlength=len(case_codes))
        return mult[case_of_seq]

    results = {}
    for key, fe_key in SCORERS.items():
        # one fixed seed per scorer and type, so results do not depend on run order
        s = sc[key]
        row = {"overall_roc_auc": float(roc_auc_score(y, s)),
               "normal_flagged_pct": 100.0 * final[fe_key]["fp"] / n_normal,
               "per_type": {}}
        for t in TYPES:
            m = (types == t) | (y == 0)
            auc = float(roc_auc_score(y[m], s[m]))
            a_case_ids = np.unique(case_of_seq[types == t])
            ym, sm = y[m], s[m]
            rng = np.random.default_rng(
                [SEED, list(ALL_SCORERS).index(key), TYPES.index(t)])
            boots = []
            for _ in range(N_BOOT):
                w = cluster_weights(rng, a_case_ids) * (types == t) \
                    + cluster_weights(rng, normal_case_ids) * (y == 0)
                boots.append(roc_auc_score(ym, sm, sample_weight=w[m]))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            row["per_type"][t] = {"roc_auc": auc, "ci95": [float(lo), float(hi)],
                                  "n_sequences": int((types == t).sum()),
                                  "n_cases": int(len(a_case_ids))}
        results[key] = row
        cells = "  ".join(f"{t[:5]} {v['roc_auc']:.3f} [{v['ci95'][0]:.2f},{v['ci95'][1]:.2f}]"
                          for t, v in row["per_type"].items())
        print(f"{key:24} flagged {row['normal_flagged_pct']:5.1f}%  {cells}")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
