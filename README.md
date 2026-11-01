# A Comparative Analysis of Deep Sequential Models for Real-Time Anomaly Detection in Heterogeneous ERP Process Flows

Code, trained models and result files for the Master's thesis of
**Mustafa Wasif Allvi**, M.Sc. Data Science, University of Potsdam (2026).

| | |
|---|---|
| **Institution** | University of Potsdam, Chair of Business Informatics, Processes and Systems |
| **First examiner** | Prof. Dr.-Ing. Marcus Grum |
| **Second examiner** | M.Sc. Tim Müller |
| **Dataset** | BPI Challenge 2020 (domestic and international declarations) |
| **Language / framework** | Python 3.10+, PyTorch 2.2 (CPU only), scikit-learn |

The repository contains everything needed to reproduce every table and figure
in Chapters 6 and 7 of the thesis, starting from the public event log. The
version that produced the reported results is tagged `v1.0-submission`.

---

## Contents

1. [Abstract](#abstract)
2. [Research questions](#research-questions)
3. [Approach](#approach)
4. [Main results](#main-results)
5. [Repository structure](#repository-structure)
6. [Installation](#installation)
7. [Data](#data)
8. [Reproducing the results](#reproducing-the-results)
9. [From result files to thesis tables and figures](#from-result-files-to-thesis-tables-and-figures)
10. [Reproducibility notes](#reproducibility-notes)
11. [Limitations](#limitations)
12. [Citation](#citation)

---

## Abstract

ERP systems record business process execution as event logs. Anomalies in
those logs, whether caused by fraud, process error or system malfunction,
carry financial and compliance consequences, and they are most useful when
they are detected while a case is still running. Existing deep learning work
addresses generic event logs, evaluates single architectures in isolation
and rarely reports the latency evidence that real-time use requires.

Following a design science approach, the thesis derives a multi-perspective
detection framework from six ERP process perspectives using the Construction
of a Concept of Neuronal Modeling (CoNM) methodology. The framework is
instantiated with three encoders (LSTM, GRU, Transformer) and evaluated on the
BPI Challenge 2020 log under one identical, seeded protocol, with five anomaly
types injected into the held-out test cases. The GRU predicts the next
activity most accurately and most stably. Detection is evaluated on the
prefixes in which an injected anomaly is already visible. A dedicated
sequence-to-sequence autoencoder is the strongest detector (ROC-AUC 0.702,
AUPRC 0.398, 6.6 times the random baseline), while two classical detectors
without sequence context stay close to random. The scoring channels are
complementary: the autoencoder detects the two financial anomaly types,
prediction confidence detects control-flow anomalies, and resource and
temporal anomalies escape every scorer. Every model runs below 6 ms per event
on a commodity CPU.

## Research questions

- **RQ1.** Which deep sequential architecture (LSTM, GRU or Transformer)
  achieves the most effective next-activity prediction on heterogeneous ERP
  process data, and what architectural characteristics explain the difference?
- **RQ2.** How effectively do prediction-confidence and reconstruction-based
  scoring detect the five injected ERP anomaly types, and how does their
  detection coverage differ by type?
- **RQ3.** Does the proposed detection artifact meet the latency and
  throughput requirements for real-time ERP process monitoring?

## Approach

```
BPI 2020 event log (XES)
  └─ preprocessing: 14 features across six perspectives, train-only scaling,
     case-level 70/10/20 split, prefix enumeration (max. 26 events)
       ├─ classifiers: LSTM | GRU | Transformer, each with two heads
       │    ├─ softmax over 37 activities  → confidence score
       │    └─ reconstruction of 7 numerical features → joint reconstruction score
       ├─ dedicated GRU sequence-to-sequence autoencoder (clean data only)
       │    └─ reconstruction error → autoencoder score
       └─ classical baselines: Isolation Forest, One-Class SVM
  └─ evaluation on the test partition with five injected anomaly types:
     control-flow, data, resource, temporal, cross-entity
```

How detection is evaluated (`13_event_level_evaluation.py`):

- **Event-level labels.** Injection alters whole cases, but detection is scored
  per prefix. A prefix counts as anomalous only once it contains an altered
  event. Earlier prefixes of an injected case are identical to the original and
  count as normal. This gives 1,349 anomalous prefixes out of 22,238 (6.07 %).
  Three injected cases with a declared amount of zero were not changed by the
  mutation and stay normal.
- **Held-out thresholds.** F1, precision, recall and per-type detection rates
  use an F1-optimal threshold chosen on one half of the test cases (stratified
  by anomaly type) and applied to the other half. The split is repeated
  1,000 times and the means are reported.
- **Per-type ROC-AUC.** Each anomaly type is compared with normal prefixes of
  the same length, because every score depends on prefix length. 95 %
  intervals come from a case-level cluster bootstrap (1,000 resamples).
- **Composite score.** The reconstruction error is divided by its 99th
  percentile on the clean validation prefixes. The weight α of the confidence
  score is searched over 0, 0.1, …, 1 and checked on held-out halves.

Every classifier is trained three times (seeds 42, 43, 44) with early stopping
and best-weight restoration. The autoencoder is trained three times (seeds 142,
143, 144) for a fixed 50 epochs. Its three runs are combined by averaging their
per-sequence scores.

## Main results

**Next-activity prediction** (22,238 held-out test prefixes, mean of three runs)

| Model | Test accuracy | F1-macro | Parameters |
|---|---|---|---|
| LSTM | 85.70 % (± 0.25) | 0.4145 | 219,308 |
| **GRU** | **86.50 % (± 0.14)** | **0.4283** | 167,852 |
| Transformer | 84.63 % (± 1.18) | 0.3978 | 74,924 |

± is the population standard deviation over the three seeds. The per-run values are in
`results/metrics/significance_results.json`.

The GRU was the most accurate model in every seed. On the test set its lead is statistically
significant: McNemar's test gives p ≤ 4.3 × 10⁻⁴ for every seed, and a case-level bootstrap gives
95 % confidence intervals for the accuracy difference of +0.62 to +0.98 pp over the LSTM and
+1.60 to +2.15 pp over the Transformer. Across training runs, paired t-tests with three seeds do
not reach significance for accuracy. They do for the GRU's F1-macro advantage over the LSTM
(p = 0.001). The order of the LSTM and the Transformer depends on the seed.

**Anomaly detection** (event-level labels, 6.07 % anomalous prefixes, random AUPRC = 0.0607;
precision and recall at a held-out F1-optimal threshold, mean of 1,000 splits)

| Method | ROC-AUC | AUPRC | Precision | Recall |
|---|---|---|---|---|
| Confidence (three classifiers) | 0.482 – 0.491 | 0.056 – 0.057 | 0.061 – 0.064 | 0.62 – 0.96 |
| Joint reconstruction (three classifiers) | 0.588 – 0.617 | 0.097 – 0.105 | 0.085 – 0.091 | 0.34 – 0.38 |
| **Dedicated GRU autoencoder** | **0.7022** | **0.3981** | **0.7910** | 0.2744 |
| Isolation Forest (classical baseline) | 0.5698 | 0.0695 | 0.0709 | 0.6666 |
| One-Class SVM (classical baseline) | 0.4800 | 0.0633 | 0.0610 | 0.5899 |

The three autoencoder runs on their own reach ROC-AUC 0.694 – 0.699 and AUPRC 0.381 – 0.393.

**Per-type ROC-AUC** (anomalous prefixes of one type vs. normal prefixes of the same length;
0.5 = random; bold = 95 % case-level bootstrap interval excludes 0.5)

| Scorer | Control-flow | Data | Resource | Temporal | Cross-entity | Normal prefixes flagged |
|---|---|---|---|---|---|---|
| GRU confidence | **0.90** | 0.51 | 0.51 | **0.43** | **0.59** | 93.0 % |
| GRU joint reconstruction | **0.43** | 0.56 | **0.56** | 0.52 | **0.87** | 21.6 % |
| Dedicated autoencoder | **0.32** | **0.79** | 0.55 | 0.46 | **0.97** | 0.5 % |

Values below 0.5 mean that the anomalous prefixes look more typical than normal
prefixes of the same length. All seven scorers are in
`results/metrics/event_level_results.json`.

**Composite score.** ROC-AUC falls steadily as the weight α of the confidence
score grows. The best value is α = 0, the reconstruction score alone, for all
three architectures, and the held-out check selected α = 0 in all 200 selections.

**Inference cost** (CPU only, AMD Ryzen 5 PRO 7530U)

| Model | Mean latency | p95 latency | Throughput |
|---|---|---|---|
| LSTM | 1.136 ms | 1.403 ms | 8,380 ev/s |
| GRU | 3.770 ms | 4.298 ms | 2,190 ev/s |
| Transformer | 1.115 ms | 1.508 ms | 7,113 ev/s |
| Autoencoder | 5.912 ms | 8.153 ms | 2,253 ev/s |

Each prefix sequence corresponds to the inference triggered by one arriving
event, so latency per sequence equals latency per event.

In short:

- Prediction confidence carries no signal over all anomaly types together, but it
  separates control-flow anomalies clearly once the reordered event has arrived.
- A reconstruction model trained without a competing classification objective
  detects the financial anomalies best.
- Classical detectors on order-free aggregates barely beat chance.
- Resource and temporal anomalies escape every scorer.
- A fixed linear combination of the two scores never beats reconstruction alone,
  so the channels are best run side by side.

## Repository structure

```
.
├── config.py                   paths, seeds and all hyperparameters
├── 01_data_loading.py          load and merge the two XES sub-logs
├── 02_preprocessing.py         feature schema, imputation, encoding, scaling
├── 03_sequences.py             prefix enumeration, padding, case-level split
├── 04_models.py                LSTM, GRU and Transformer definitions (imported)
├── 05_train.py                 classifier training, three seeded runs each
├── 05b_autoencoder.py          dedicated GRU autoencoder, three seeded runs
├── 06_anomaly_injection.py     five anomaly types, test partition only
├── 07_scoring.py               first evaluation: scoring and composite grid (case-level labels)
├── 08_evaluate.py              prediction metrics, latency, throughput (+ first anomaly evaluation)
├── 09_baseline.py              first baseline evaluation (case-level labels)
├── 10_significance.py          statistical comparison of the three architectures
├── 11_per_type_auc.py          superseded by 13, kept for reference
├── 12_export_raw_scores.py     per-run raw scores on test and validation prefixes (needs PyTorch)
├── 13_event_level_evaluation.py final anomaly-detection evaluation reported in the thesis
├── generate_plots.py           result figures, first version (needs PyTorch)
├── generate_plots_v2.py        print-ready result figures used in the thesis (no PyTorch)
├── check_results.py            ┐
├── debug_recon.py              ├ inspection utilities used during development
├── extract_training_summary.py ┘
├── thesis-code-prototype.ipynb first design cycle (Chapter 5): prototype LSTM, 81.22 % baseline
├── requirements.txt
└── results/
    ├── metrics/                JSON and NPZ files behind every reported number
    ├── models/                 trained weights (3 runs × 4 models)
    ├── figures/                first-version figures, score_cache.npz and raw_scores.npz
    ├── figures_v2/             figures used in the thesis (PDF and PNG)
    └── *.joblib                fitted encoders, scaler and feature list
```

## Installation

```bash
git clone https://github.com/Wasif-Allvi/real-time-erp-anomaly-detection-thesis.git
cd real-time-erp-anomaly-detection-thesis
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All experiments ran on CPU. No GPU is required.

## Data

The experiments use the public BPI Challenge 2020 event log (van Dongen, 2020),
specifically its *Domestic Declarations* and *International Declarations*
sub-logs (16,949 cases, 128,588 events in total):
<https://doi.org/10.4121/uuid:52fb97d4-4588-43c9-9d04-3604d4613b51>

Download both `.xes` files and place them in a `dataset/` folder next to this
repository, as configured in `config.py`. The event log itself is not
redistributed here.

## Reproducing the results

Run the scripts in order. Each one reads the outputs of the previous stages
from `results/` and writes its own.

```bash
python 01_data_loading.py
python 02_preprocessing.py
python 03_sequences.py
python 05_train.py                  # 3 runs × 3 classifiers
python 05b_autoencoder.py           # 3 autoencoder runs
python 06_anomaly_injection.py
python 08_evaluate.py               # prediction metrics, latency, throughput
python 10_significance.py
python 12_export_raw_scores.py      # per-run scores (needs PyTorch)
python 13_event_level_evaluation.py # all detection results in the thesis
python generate_plots_v2.py         # figures of Chapter 7
```

The trained weights in `results/models/` are included, so you can skip the two
training stages. `results/figures/raw_scores.npz` is included as well, so after
stages 1–3 and 6 have rebuilt the preprocessed data, `13_event_level_evaluation.py`
and `generate_plots_v2.py` run without PyTorch. `07_scoring.py`,
`09_baseline.py`, `11_per_type_auc.py` and `generate_plots.py` reproduce the
first, case-level evaluation and are not needed for the reported numbers.

## From result files to thesis tables and figures

| File | Used for |
|---|---|
| `results/metrics/event_level_results.json` | anomaly detection and per-type tables, per-type ROC-AUC, composite grid, autoencoder runs (Sec. 7.2–7.4, appendix) |
| `results/metrics/event_labels.npz` | event-level labels and prefix lengths of the 22,238 test prefixes |
| `results/figures/raw_scores.npz` | per-run scores on the test and validation prefixes (input of 13 and of the figures) |
| `results/metrics/final_evaluation.json` | prediction and latency results (Sec. 7.1, 7.5). Its anomaly part is the first, case-level evaluation |
| `results/metrics/significance_results.json` | statistical comparison of the architectures (Sec. 7.1) |
| `results/metrics/injection_summary.json` | injected anomaly counts per type, before removing the three unchanged cases |
| `results/metrics/*_train_history.json` | training curves and best epochs |
| `results/metrics/scoring_results.json`, `autoencoder_results.json`, `baseline_results.json`, `per_type_auc.json` | first, case-level evaluation, kept for reference |

| File in `results/figures_v2/` (suffix `_v2`) | Thesis figure |
|---|---|
| `fig7_1_val_loss` | Figure 4: validation loss, three runs per architecture |
| `fig7_2_lr_decay` | Figure 5: learning-rate decay |
| `fig7_3_pred_performance` | Figure 6: next-activity prediction comparison |
| `fig7_4_roc_curves` | Figure 7: ROC curves (event-level labels) |
| `fig7_5_pr_curves` | Figure 8: precision-recall curves (event-level labels) |
| `fig7_6_per_type_auc` | Figure 9: length-matched per-type ROC-AUC |
| `fig7_7_alpha_beta` | Figure 10: composite weighting, α = 0 … 1 |
| `fig7_8_latency` | Figure 11: latency against the 10 ms bound |

## Reproducibility notes

- Seeds: case split 42, anomaly injection 2024, classifier runs 42 / 43 / 44,
  autoencoder runs 142 / 143 / 144. Held-out splits and bootstrap resamples in
  `13_event_level_evaluation.py` are seeded as well.
- Scalers are fitted on the training partition only. The split is made at case
  level, and partition disjointness is verified.
- The normalization of the reconstruction score for the composite uses the
  clean validation prefixes, so no test information enters it.
- Isolation Forest results can differ in the fourth decimal between
  scikit-learn versions.
- Large intermediate files (`results/preprocessed/`) are excluded from version
  control. Stages 1–3 and 6 regenerate them deterministically.
- Hardware: AMD Ryzen 5 PRO 7530U (six cores), 16 GB RAM, no GPU.

## Limitations

- The evaluation uses a single institutional dataset.
- Anomaly ground truth comes from controlled injection, not from audited fraud cases.
- Thresholds and the composite weight are calibrated within the injected test
  partition (on held-out halves), not on independent production data.
- Categorical attributes enter the networks as label-encoded integers, and the
  Transformer does not mask the padded positions.
- Each architecture was trained three times.

The thesis discusses each of these in Section 8.7.

## Citation

```bibtex
@mastersthesis{Allvi2026,
  author  = {Mustafa Wasif Allvi},
  title   = {A Comparative Analysis of Deep Sequential Models for Real-Time
             Anomaly Detection in Heterogeneous ERP Process Flows},
  school  = {University of Potsdam},
  type    = {Master's thesis},
  address = {Potsdam, Germany},
  year    = {2026}
}
```

Dataset: B. F. van Dongen, *BPI Challenge 2020*, 4TU.ResearchData, 2020.
