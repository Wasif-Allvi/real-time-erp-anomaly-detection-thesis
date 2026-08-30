# Deep Sequential Models for Real-Time Anomaly Detection in Heterogeneous ERP Process Flows

Implementation accompanying the Master's thesis *"A Comparative Analysis of Deep
Sequential Models for Real-Time Anomaly Detection in Heterogeneous ERP Process
Flows"* (University of Potsdam, Chair of Business Informatics, Processes and
Systems).

The repository contains the full pipeline behind every table and figure reported
in Chapters 6 and 7 of the thesis: preprocessing of the BPI Challenge 2020 event
log, three deep sequential architectures (LSTM, GRU, Transformer) with a dual
classification and reconstruction head, a dedicated sequence-to-sequence
autoencoder, a five-type anomaly injection framework, three anomaly scoring
mechanisms, and the evaluation and plotting scripts.

## Data

The experiments use the publicly available BPI Challenge 2020 event log
(Domestic Declarations and International Declarations sub-logs):
https://doi.org/10.4121/uuid:52fb97d4-4588-43c9-9d04-3604d4613b51

Download both `.xes` files and place them under a `dataset/` directory next to
this repository, following the paths configured in `config.py`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the results

Run the scripts in order; each writes its output to `results/`:

```bash
python 01_data_loading.py        # load and merge the XES sub-logs
python 02_preprocessing.py       # feature schema, imputation, encoding, scaling
python 03_sequences.py           # prefix enumeration and case-level split
python 05_train.py               # train LSTM, GRU, Transformer (3 seeded runs each)
python 05b_autoencoder.py        # train the dedicated GRU autoencoder
python 06_anomaly_injection.py   # inject the five anomaly types into the test set
python 07_scoring.py             # confidence, reconstruction and composite scoring
python 08_evaluate.py            # metrics, per-type coverage, latency
python generate_plots.py         # figures reported in Chapter 7
```

`04_models.py` defines the architectures and is imported by the training scripts.

## Reproducibility

* Random seeds are fixed in `config.py`; the three classifier runs use seeds
  42, 43 and 44, and anomaly injection uses its own fixed seed, so that every
  model is evaluated against the identical injected test partition.
* All preprocessing estimators are fitted on the training partition only.
* Splitting is performed at case level, with verified disjointness between the
  training, validation and test partitions.
* Trained model weights (`results/models/`) and the metric files underlying the
  reported tables (`results/metrics/`) are included in this repository.
* The large intermediate preprocessing artifacts are not included; they are
  regenerated deterministically by `01_data_loading.py` and `02_preprocessing.py`.

## Repository structure

```
01_data_loading.py … 08_evaluate.py   pipeline stages, executed in order
04_models.py                          LSTM, GRU and Transformer definitions
05b_autoencoder.py                    dedicated sequence-to-sequence autoencoder
config.py                             paths, feature schema, hyperparameters, seeds
generate_plots.py                     figures for Chapter 7
requirements.txt                      Python dependencies
results/metrics/                      evaluation outputs underlying the tables
results/models/                       trained weights (3 runs per architecture)
results/figures/                      figures as PDF and PNG
```

## Hardware

All experiments were executed on CPU (AMD Ryzen 5 PRO 7530U, 16 GB RAM) without
GPU acceleration.

## Author

Mustafa Wasif Allvi — University of Potsdam
