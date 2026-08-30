# Real-Time Anomaly Detection in Heterogeneous ERP Process Flows

Implementation and experimental artifacts for the Master's thesis
**"A Comparative Analysis of Deep Sequential Models for Real-Time Anomaly
Detection in Heterogeneous ERP Process Flows"**, University of Potsdam,
Chair of Business Informatics, Processes and Systems.

---

## What this study investigates

Anomalies in ERP process flows — a payment executed before its approval, an
approval granted by a role without the authority, a declared amount exceeding
the budget authorised in a linked permit — carry financial and compliance
consequences, and are most useful when detected while the process is still
running. The thesis asks two questions:

1. **RQ1** — Which deep sequential architecture (LSTM, GRU or Transformer)
   achieves the most effective next-activity prediction on heterogeneous ERP
   process data, and what architectural characteristics explain the difference?
2. **RQ2** — Does the resulting detection framework meet the latency and
   throughput requirements for real-time ERP process monitoring?

The framework is designed following the Construction of a Concept of Neuronal
Modeling (CoNM) methodology and evaluated on the BPI Challenge 2020 event log
(16,949 cases, 128,588 events) with five injected anomaly types.

## Headline results

Next-activity prediction on 22,238 held-out test sequences, reported as the mean
of three seeded runs:

| Model | Test accuracy | F1-macro | Parameters | Mean latency | Throughput |
|---|---|---|---|---|---|
| LSTM | 85.70 % (± 0.25) | 0.4145 | 219,308 | 1.136 ms | 8,380 ev/s |
| **GRU** | **86.50 % (± 0.14)** | **0.4283** | 167,852 | 3.770 ms | 2,190 ev/s |
| Transformer | 84.63 % (± 1.18) | 0.3978 | 74,924 | 1.115 ms | 7,113 ev/s |

Anomaly detection on the injected test partition (9.04 % anomalous sequences,
random AUPRC baseline 0.0904):

| Scoring mechanism | ROC-AUC | AUPRC | Precision @ best F1 |
|---|---|---|---|
| Confidence (softmax) | 0.517 – 0.528 | ≈ 0.09 | ≈ 0.10 |
| Joint reconstruction | 0.574 – 0.582 | 0.118 – 0.123 | ≈ 0.12 |
| **Dedicated autoencoder** | **0.6301** | **0.3403** | **0.6909** |

Three findings follow. Prediction confidence alone is near-random for anomalies
that leave the activity sequence intact. A reconstruction model trained without
a competing classification objective detects them substantially better. And the
two mechanisms are complementary rather than redundant: confidence scoring
covers control-flow and resource anomalies, reconstruction covers data and
cross-entity anomalies, and neither covers all five types alone.

## Data

The experiments use the publicly available BPI Challenge 2020 event log,
Domestic Declarations and International Declarations sub-logs:
<https://doi.org/10.4121/uuid:52fb97d4-4588-43c9-9d04-3604d4613b51>

Download both `.xes` files and place them in a `dataset/` directory beside this
repository, following the paths configured in `config.py`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Pipeline

Run the stages in order; each writes to `results/`.

| Stage | Script | What it does |
|---|---|---|
| 1 | `01_data_loading.py` | Loads both XES sub-logs via pm4py and merges them |
| 2 | `02_preprocessing.py` | 14-feature schema, type-aware imputation, log-transform of temporal features, label encoding, min-max scaling fitted on training data only |
| 3 | `03_sequences.py` | Prefix enumeration, zero-padding to 26 timesteps, case-level 70/10/20 split with verified disjointness |
| 4 | `04_models.py` | Architecture definitions (imported, not run directly) |
| 5 | `05_train.py` | Trains LSTM, GRU and Transformer, three seeded runs each, with early stopping and best-weight restoration |
| 6 | `05b_autoencoder.py` | Trains the dedicated GRU sequence-to-sequence autoencoder on clean sequences |
| 7 | `06_anomaly_injection.py` | Injects the five anomaly types into the test partition only |
| 8 | `07_scoring.py` | Confidence, reconstruction and composite scoring with threshold selection |
| 9 | `08_evaluate.py` | Metrics, per-type coverage, latency and throughput |
| 10 | `generate_plots.py` | Regenerates every figure reported in the thesis |

`check_results.py`, `debug_recon.py` and `extract_training_summary.py` are small
inspection utilities used during development.

## Anomaly taxonomy

Five types are injected into the test partition, one per selected case, so that
ground truth remains unambiguous:

| Type | Injection mechanism |
|---|---|
| Control-flow | Swap the timestamps of *Payment Handled* and *Request Payment* |
| Data | Scale `case:Amount` by 10× on an `EMPLOYEE`-role event |
| Resource | Replace the role of an approval event with `EMPLOYEE` |
| Temporal | Shift a middle event 25 hours backward, inducing a negative inter-event gap |
| Cross-entity | Scale `case:Amount` by 5× on international traces, exceeding the permit budget |

## Figures

Filenames follow the generation script; the thesis numbers figures
continuously. The mapping is:

| File in `results/figures/` | Thesis figure | Content |
|---|---|---|
| `fig7_1_val_loss` | Figure 4 | Validation loss convergence, three runs per architecture |
| `fig7_2_lr_decay` | Figure 5 | Learning-rate decay trajectories |
| `fig7_3_pred_performance` | Figure 6 | Next-activity prediction comparison |
| `fig7_4_roc_curves` | Figure 7 | ROC curves across scoring mechanisms |
| `fig7_5_pr_curves` | Figure 8 | Precision-recall curves under class imbalance |
| `fig7_6_detection_heatmap` | Figure 9 | Per-type detection coverage |
| `fig7_7_alpha_beta` | Figure 10 | Composite weighting sensitivity |
| `fig7_8_latency_accuracy` | Figure 11 | Latency and throughput trade-off |

## Reproducibility

* Seeds 42, 43 and 44 govern weight initialisation and batch ordering for the
  three classifier runs; anomaly injection uses its own fixed seed, so every
  model is evaluated against an identical injected test partition.
* All preprocessing estimators are fitted on the training partition only.
* Splitting is performed at case level; partition disjointness is verified.
* Trained weights (`results/models/`) and the metric files underlying every
  reported table (`results/metrics/`) are included here.
* Large intermediate preprocessing artifacts are excluded from version control;
  stages 1–3 regenerate them deterministically.
* All experiments were executed on CPU (AMD Ryzen 5 PRO 7530U, 16 GB RAM)
  without GPU acceleration.

## Limitations

The evaluation rests on a single institutional dataset; anomaly ground truth is
generated by controlled injection rather than audited fraud cases; detection
thresholds and the composite weighting are selected on the injected test
partition, so the reported operating points are optimistic; and each
architecture was trained three times, which bounds the confidence of the
architectural ranking. These constraints are discussed in the thesis.

## Author

Mustafa Wasif Allvi — University of Potsdam
