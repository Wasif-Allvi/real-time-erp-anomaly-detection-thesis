#!/usr/bin/env python3
"""
generate_plots.py
Chapter 7 Figure Generator
Thesis: Deep Sequential Models for ERP Anomaly Detection
Mustafa Wasif Allvi — University of Potsdam, M.Sc. Data Science, 2026

Figures produced (all white background, 300 DPI):
  fig7_1_val_loss.pdf/png          — Validation loss convergence curves
  fig7_2_lr_decay.pdf/png          — Learning rate decay trajectories
  fig7_3_pred_performance.pdf/png  — Next-activity prediction performance
  fig7_4_roc_curves.pdf/png        — ROC curves (requires model inference)
  fig7_5_pr_curves.pdf/png         — Precision-recall curves (requires model inference)
  fig7_6_detection_heatmap.pdf/png — Per-type anomaly detection heatmap (7 methods)
  fig7_7_alpha_beta.pdf/png        — Composite scoring weight sensitivity
  fig7_8_latency_accuracy.pdf/png  — Latency vs. accuracy deployment tradeoff

Figures 7.4 and 7.5 require model inference on the first run.
Results are cached to results/figures/score_cache.npz for all subsequent runs.

Run:
  cd "C:\\Users\\wasif\\Desktop\\Thesis\\Thesis Code"
  python generate_plots.py
"""

# ══════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════
import os
import sys
import json
import warnings
import importlib

import numpy as np
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════
ROOT_DIR = r'C:\Users\wasif\Desktop\Thesis\Thesis Code'
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

MET_DIR    = os.path.join(ROOT_DIR, 'results', 'metrics')
MOD_DIR    = os.path.join(ROOT_DIR, 'results', 'models')
PREP_DIR   = os.path.join(ROOT_DIR, 'results', 'preprocessed')
FIG_DIR    = os.path.join(ROOT_DIR, 'results', 'figures')
CACHE_FILE = os.path.join(FIG_DIR, 'score_cache.npz')
os.makedirs(FIG_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# PROJECT IMPORTS
# ══════════════════════════════════════════════════════════════
import config
from config import (
    DEVICE, BATCH_SIZE, MODEL_INPUT_DIM,
    N_CATEGORICAL, N_NUMERICAL,
    FINANCIAL_INDICES_IN_NUMERICAL, FINANCIAL_RECON_WEIGHT,
    NUM_RUNS,
)

_models_module             = importlib.import_module('04_models')
build_model                = _models_module.build_model

_ae_module                 = importlib.import_module('05b_autoencoder')
GRUAutoencoder             = _ae_module.GRUAutoencoder
compute_event_level_scores = _ae_module.compute_event_level_scores
build_numerical_weights    = _ae_module.build_numerical_weights
AE_HIDDEN_DIM              = _ae_module.AE_HIDDEN_DIM
AE_NUM_LAYERS              = _ae_module.AE_NUM_LAYERS
AE_DROPOUT                 = _ae_module.AE_DROPOUT
AE_RUNS                    = _ae_module.AE_RUNS
N_NUMERICAL_AE             = _ae_module.N_NUMERICAL

# ══════════════════════════════════════════════════════════════
# ACADEMIC PLOT STYLE
# White background, DejaVu Serif font, clean grid
# ══════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family'        : 'DejaVu Serif',
    'font.size'          : 11,
    'axes.labelsize'     : 12,
    'axes.titlesize'     : 12,
    'axes.titleweight'   : 'bold',
    'axes.titlepad'      : 10,
    'xtick.labelsize'    : 10,
    'ytick.labelsize'    : 10,
    'legend.fontsize'    : 9.5,
    'legend.framealpha'  : 0.95,
    'legend.edgecolor'   : '#cccccc',
    'figure.facecolor'   : 'white',
    'axes.facecolor'     : 'white',
    'savefig.facecolor'  : 'white',
    'axes.spines.top'    : False,
    'axes.spines.right'  : False,
    'axes.grid'          : True,
    'grid.alpha'         : 0.22,
    'grid.linestyle'     : '--',
    'axes.linewidth'     : 0.8,
})

COL = {
    'lstm'        : '#1565C0',
    'gru'         : '#2E7D32',
    'transformer' : '#E65100',
    'autoencoder' : '#6A1B9A',
    'joint_recon' : '#AD1457',
    'baseline'    : '#757575',
}
DISP = {'lstm': 'LSTM', 'gru': 'GRU', 'transformer': 'Transformer'}


def save_fig(fig, stem):
    for ext in ('pdf', 'png'):
        fig.savefig(
            os.path.join(FIG_DIR, f'{stem}.{ext}'),
            bbox_inches='tight',
            dpi=300 if ext == 'png' else None,
        )
    plt.close(fig)
    print(f'  Saved: {stem}.pdf / .png')


# ══════════════════════════════════════════════════════════════
# SCORE COMPUTATION (for Figures 7.4 and 7.5)
# ══════════════════════════════════════════════════════════════

class SequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.seq = torch.tensor(sequences, dtype=torch.float32)
        self.tgt = torch.tensor(targets,   dtype=torch.long)
    def __len__(self): return len(self.seq)
    def __getitem__(self, i): return self.seq[i], self.tgt[i]


def mean_numerical_target(sequences):
    """Mean numerical features over real (non-padded) timesteps only."""
    num_slice = sequences[:, :, N_CATEGORICAL:]
    mask      = (sequences.abs().sum(dim=2) > 0).float()
    expanded  = mask.unsqueeze(2)
    return (num_slice * expanded).sum(dim=1) / expanded.sum(dim=1).clamp(min=1)


def build_recon_weights():
    w = torch.ones(N_NUMERICAL)
    for i in FINANCIAL_INDICES_IN_NUMERICAL:
        if i < N_NUMERICAL:
            w[i] = FINANCIAL_RECON_WEIGHT
    return w


def compute_and_cache_scores():
    """
    Runs model inference on the injected test set once and caches results.
    On subsequent calls, loads directly from score_cache.npz.
    """
    if os.path.exists(CACHE_FILE):
        print('  Loading cached score arrays...')
        d = np.load(CACHE_FILE, allow_pickle=True)
        return dict(d)

    print('  Computing scores from model weights (approx. 15-25 min on CPU)...')

    inj         = joblib.load(os.path.join(PREP_DIR, 'test_injected.joblib'))
    seqs        = inj['sequences']
    tgts        = inj['targets']
    true_binary = inj['labels']

    enc         = joblib.load(os.path.join(PREP_DIR, 'label_encoders.joblib'))
    num_classes = len(enc['concept:name'].classes_)

    dataset = SequenceDataset(seqs, tgts)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    rw      = build_recon_weights()
    mse_fn  = nn.MSELoss(reduction='none')
    out     = {'true_binary': true_binary}

    for model_name in ['lstm', 'gru', 'transformer']:
        run_conf, run_recon = [], []
        for run_idx in range(1, NUM_RUNS + 1):
            wp = os.path.join(MOD_DIR, f'{model_name}_run{run_idx}.pth')
            if not os.path.exists(wp):
                print(f'  WARNING: not found: {wp}')
                continue
            model = build_model(model_name, num_classes)
            model.load_state_dict(torch.load(wp, map_location=DEVICE))
            model.eval()
            c_list, r_list = [], []
            with torch.no_grad():
                for batch, _ in loader:
                    batch  = batch.to(DEVICE)
                    logits, recon = model(batch)
                    probs  = torch.softmax(logits, dim=1)
                    conf   = (1 - probs.max(dim=1).values).cpu().numpy()
                    tgt_n  = mean_numerical_target(batch)
                    sq_err = mse_fn(recon, tgt_n)
                    wt_err = (sq_err * rw.to(DEVICE)).mean(dim=1).cpu().numpy()
                    c_list.extend(conf)
                    r_list.extend(wt_err)
            run_conf.append(np.array(c_list))
            run_recon.append(np.array(r_list))
            print(f'    {DISP[model_name]} run {run_idx} complete')

        mc   = np.mean(run_conf,  axis=0)
        mr   = np.mean(run_recon, axis=0)
        p99  = np.percentile(mr, 99)
        norm = np.clip(mr / p99, 0, 1) if p99 > 0 else mr
        out[f'{model_name}_confidence'] = mc
        out[f'{model_name}_recon_norm'] = norm

    nw        = build_numerical_weights()
    ae_scores = []
    for run_idx in range(1, AE_RUNS + 1):
        ap = os.path.join(MOD_DIR, f'autoencoder_run{run_idx}.pth')
        if not os.path.exists(ap):
            print(f'  WARNING: not found: {ap}')
            continue
        ae = GRUAutoencoder(
            input_dim=MODEL_INPUT_DIM, n_numerical=N_NUMERICAL_AE,
            hidden_dim=AE_HIDDEN_DIM, num_layers=AE_NUM_LAYERS,
            dropout=AE_DROPOUT,
        ).to(DEVICE)
        ae.load_state_dict(torch.load(ap, map_location=DEVICE))
        ae_scores.append(compute_event_level_scores(ae, seqs, nw))
        print(f'    Autoencoder run {run_idx} complete')

    out['autoencoder'] = np.mean(ae_scores, axis=0)
    np.savez(CACHE_FILE, **out)
    print('  Scores cached to disk.')
    return out


# ══════════════════════════════════════════════════════════════
# FIGURE 7.1 — Validation Loss Convergence Curves
# ══════════════════════════════════════════════════════════════
def fig_val_loss():
    print('\nFig 7.1: Validation Loss Convergence Curves')
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8),
                              gridspec_kw={'wspace': 0.30})

    for ax, mname, color in zip(
        axes,
        ['lstm', 'gru', 'transformer'],
        [COL['lstm'], COL['gru'], COL['transformer']]
    ):
        with open(os.path.join(MET_DIR, f'{mname}_train_history.json')) as f:
            runs = json.load(f)

        best_epochs  = [r['best_epoch'] for r in runs]
        all_val_loss = [r['history']['val_loss'] for r in runs]
        all_lr       = [r['history']['learning_rates'] for r in runs]
        max_len      = max(len(v) for v in all_val_loss)

        for v in all_val_loss:
            ax.plot(range(1, len(v) + 1), v, color=color, alpha=0.18, lw=0.9)

        mean_v = np.array([
            np.mean([v[ep] for v in all_val_loss if ep < len(v)])
            for ep in range(max_len)
        ])
        std_v = np.array([
            np.std([v[ep] for v in all_val_loss if ep < len(v)])
            for ep in range(max_len)
        ])
        xs = np.arange(1, max_len + 1)
        ax.plot(xs, mean_v, color=color, lw=2.2, label='Mean val. loss')
        ax.fill_between(xs, mean_v - std_v, mean_v + std_v,
                         color=color, alpha=0.13)

        lr_run1 = all_lr[0]
        for i in range(1, len(lr_run1)):
            if lr_run1[i] < lr_run1[i - 1] - 1e-9:
                ax.axvline(x=i + 1, color=color, ls=':', lw=0.9, alpha=0.40)

        mean_best = int(round(np.mean(best_epochs)))
        ax.axvline(x=mean_best, color='black', ls='--', lw=1.3, alpha=0.58,
                    label=f'Best epoch ({mean_best})')

        ax.set_xlabel('Epoch', labelpad=5)
        ax.set_ylabel('Validation Loss')
        ax.set_title(DISP[mname])
        ax.set_xlim(1, max_len)
        ax.legend(loc='upper right', fontsize=9)

    fig.suptitle('Figure 1 — Validation Loss Convergence Curves',
                  fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_fig(fig, 'fig7_1_val_loss')


# ══════════════════════════════════════════════════════════════
# FIGURE 7.2 — Learning Rate Decay Trajectories (all 3 runs)
# Three solid lines in progressively lighter color intensity
# ══════════════════════════════════════════════════════════════
def fig_lr_decay():
    print('Fig 7.2: Learning Rate Decay Trajectories')
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2),
                              gridspec_kw={'wspace': 0.36})

    # Three color-intensity tiers: dark (Run 1) → medium → light (Run 3)
    # All solid lines for clean visual comparison
    run_palette = {
        'lstm'        : ['#0D47A1', '#1976D2', '#64B5F6'],  # dark to light blue
        'gru'         : ['#1B5E20', '#388E3C', '#81C784'],  # dark to light green
        'transformer' : ['#BF360C', '#E64A19', '#FFAB91'],  # dark to light orange
    }

    for ax, mname in zip(axes, ['lstm', 'gru', 'transformer']):
        with open(os.path.join(MET_DIR, f'{mname}_train_history.json')) as f:
            runs = json.load(f)

        palette  = run_palette[mname]
        max_len  = max(len(r['history']['learning_rates']) for r in runs)

        for run_idx, (run_data, c) in enumerate(zip(runs, palette), start=1):
            lr_list = run_data['history']['learning_rates']
            ax.step(range(1, len(lr_list) + 1), lr_list,
                     where='post', color=c, lw=2.0, ls='-',
                     label=f'Run {run_idx}',
                     zorder=4 - run_idx)  # Run 1 drawn on top

        ax.set_yscale('log')
        ax.set_xlabel('Epoch', labelpad=5)
        ax.set_ylabel('Learning Rate (log scale)')
        ax.set_title(DISP[mname])
        ax.set_xlim(1, max_len)

        def _fmt(v, _):
            if v <= 0:
                return '0'
            exp = int(round(np.log10(v)))
            return f'$10^{{{exp}}}$'

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.legend(loc='upper right', fontsize=9)

    fig.suptitle('Figure 2 — Learning Rate Decay Trajectories',
                  fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_fig(fig, 'fig7_2_lr_decay')


# ══════════════════════════════════════════════════════════════
# FIGURE 7.3 — Next-Activity Prediction Performance
# Both panels now include prototype LSTM baseline reference
# ══════════════════════════════════════════════════════════════
def fig_pred_perf():
    print('Fig 7.3: Predictive Performance')
    with open(os.path.join(MET_DIR, 'final_evaluation.json')) as f:
        ev = json.load(f)['prediction']

    models = ['lstm', 'gru', 'transformer']
    labels = ['LSTM', 'GRU', 'Transformer']
    colors = [COL['lstm'], COL['gru'], COL['transformer']]

    acc_m = [ev[m]['test_acc_mean'] for m in models]
    acc_s = [ev[m]['test_acc_std']  for m in models]
    f1_m  = [ev[m]['test_f1m_mean'] for m in models]
    f1_s  = [ev[m]['test_f1m_std']  for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                    gridspec_kw={'wspace': 0.30})

    # Accuracy panel
    bars = ax1.bar(labels, acc_m, color=colors, alpha=0.85,
                    yerr=acc_s, capsize=7, ecolor='#333333',
                    edgecolor='white', linewidth=0.8)
    ax1.axhline(81.22, color=COL['baseline'], ls='--', lw=1.4, alpha=0.75,
                 label='First-cycle LSTM baseline (81.22%)')
    for b, v, s in zip(bars, acc_m, acc_s):
        ax1.text(b.get_x() + b.get_width() / 2, v + s + 0.35,
                  f'{v:.2f}%', ha='center', va='bottom',
                  fontsize=11, fontweight='bold')
    ax1.set_ylim(77, 92)
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Test Accuracy')
    ax1.legend(fontsize=9, loc='upper left')

    # F1-Macro panel — also includes prototype baseline for consistency
    bars = ax2.bar(labels, f1_m, color=colors, alpha=0.85,
                    yerr=f1_s, capsize=7, ecolor='#333333',
                    edgecolor='white', linewidth=0.8)
    ax2.axhline(0.2890, color=COL['baseline'], ls='--', lw=1.4, alpha=0.75,
                 label='First-cycle LSTM baseline (0.2890)')
    for b, v, s in zip(bars, f1_m, f1_s):
        ax2.text(b.get_x() + b.get_width() / 2, v + s + 0.004,
                  f'{v:.4f}', ha='center', va='bottom',
                  fontsize=11, fontweight='bold')
    ax2.set_ylim(0.22, 0.50)
    ax2.set_ylabel('F1-Macro Score')
    ax2.set_title('F1-Macro (37 activity classes)')
    ax2.legend(fontsize=9, loc='upper left')

    fig.suptitle('Figure 3 — Next-Activity Prediction Performance on Test Set',
                  fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_fig(fig, 'fig7_3_pred_performance')

# ══════════════════════════════════════════════════════════════
# FIGURE 7.4 — ROC Curves
# ══════════════════════════════════════════════════════════════
def fig_roc_curves(sc):
    print('Fig 7.4: ROC Curves')
    tb = sc['true_binary']

    entries = [
        ('lstm_confidence',        'LSTM — Confidence',          COL['lstm'],        '--', 1.8),
        ('gru_confidence',         'GRU — Confidence',           COL['gru'],         '--', 1.8),
        ('transformer_confidence', 'Transformer — Confidence',   COL['transformer'], '--', 1.8),
        ('gru_recon_norm',         'Joint Reconstruction (GRU)', COL['joint_recon'], '-',  2.2),
        ('autoencoder',            'Separate Autoencoder',       COL['autoencoder'], '-',  2.5),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.plot([0, 1], [0, 1], '--', color=COL['baseline'], lw=1.3, alpha=0.75,
             label='Random baseline (AUC = 0.50)', zorder=1)

    for key, base_label, color, ls, lw in entries:
        fpr, tpr, _ = roc_curve(tb, sc[key])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linestyle=ls, color=color, lw=lw, zorder=3,
                 label=f'{base_label} (AUC = {roc_auc:.4f})')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.legend(loc='lower right', fontsize=9.5)
    ax.set_title('Figure 4 — ROC Curves: Anomaly Detection Performance', pad=12)
    plt.tight_layout()
    save_fig(fig, 'fig7_4_roc_curves')


# ══════════════════════════════════════════════════════════════
# FIGURE 7.5 — Precision-Recall Curves
# ══════════════════════════════════════════════════════════════
def fig_pr_curves(sc):
    print('Fig 7.5: Precision-Recall Curves')
    tb         = sc['true_binary']
    rand_auprc = float(tb.mean())

    entries = [
        ('lstm_confidence',        'LSTM — Confidence',          COL['lstm'],        '--', 1.8),
        ('gru_confidence',         'GRU — Confidence',           COL['gru'],         '--', 1.8),
        ('transformer_confidence', 'Transformer — Confidence',   COL['transformer'], '--', 1.8),
        ('gru_recon_norm',         'Joint Reconstruction (GRU)', COL['joint_recon'], '-',  2.2),
        ('autoencoder',            'Separate Autoencoder',       COL['autoencoder'], '-',  2.5),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.axhline(rand_auprc, ls=':', color=COL['baseline'], lw=1.4, alpha=0.75,
                label=f'Random baseline (AUPRC = {rand_auprc:.3f})')

    for key, base_label, color, ls, lw in entries:
        ap           = average_precision_score(tb, sc[key])
        prec, rec, _ = precision_recall_curve(tb, sc[key])
        ax.plot(rec, prec, linestyle=ls, color=color, lw=lw,
                 label=f'{base_label} (AUPRC = {ap:.4f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.05)
    ax.legend(loc='upper right', fontsize=9.5)
    ax.set_title('Figure 5 — Precision-Recall Curves: AUPRC under Class Imbalance',
                  pad=12)
    plt.tight_layout()
    save_fig(fig, 'fig7_5_pr_curves')


# ══════════════════════════════════════════════════════════════
# FIGURE 7.6 — Per-Type Anomaly Detection Coverage Heatmap
# All 7 scoring methods; group separators between sections
# ══════════════════════════════════════════════════════════════
def fig_detection_heatmap():
    print('Fig 7.6: Per-Type Detection Heatmap')

    ROW_LABELS = [
        'LSTM Confidence',
        'GRU Confidence',
        'Transformer Confidence',
        'LSTM Joint Recon.',
        'GRU Joint Recon.',
        'Transformer Joint Recon.',
        'Separate Autoencoder',
    ]
    COL_LABELS = ['Control-Flow', 'Data', 'Resource', 'Temporal', 'Cross-Entity']

    # Detection rates at F1-optimal threshold from 07_scoring.py output
    DATA = np.array([
        [76.0, 59.5, 66.1, 62.1,  70.1],
        [92.4, 77.9, 87.0, 81.4,  90.6],
        [92.9, 80.1, 90.6, 82.8,  99.2],
        [42.2, 39.6, 40.2, 33.4,  95.5],
        [36.8, 37.6, 40.4, 27.0,  93.0],
        [43.3, 46.4, 45.8, 37.0,  98.0],
        [ 2.8, 50.1,  1.9,  4.5,  79.1],
    ])

    fig, ax = plt.subplots(figsize=(12, 5.8))
    im   = ax.imshow(DATA, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label('Detection Rate (%)', fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xticks(np.arange(len(COL_LABELS)))
    ax.set_yticks(np.arange(len(ROW_LABELS)))
    ax.set_xticklabels(COL_LABELS, fontsize=11)
    ax.set_yticklabels(ROW_LABELS, fontsize=11)
    ax.set_xlabel('Injected Anomaly Type', fontsize=12, labelpad=8)

    for i in range(len(ROW_LABELS)):
        for j in range(len(COL_LABELS)):
            v  = DATA[i, j]
            tc = 'white' if (v < 25 or v > 78) else 'black'
            ax.text(j, i, f'{v:.1f}%', ha='center', va='center',
                     fontsize=12, fontweight='bold', color=tc)

    # White separator lines between the three method groups
    for y in (2.5, 5.5):
        ax.axhline(y=y, color='white', linewidth=3.0, zorder=5)

    ax.set_title('Figure 6 — Per-Type Anomaly Detection Coverage Heatmap', pad=12)
    plt.tight_layout()
    save_fig(fig, 'fig7_6_detection_heatmap')


# ══════════════════════════════════════════════════════════════
# FIGURE 7.7 — Composite Scoring Weight Sensitivity Analysis
# ══════════════════════════════════════════════════════════════
def fig_alpha_beta():
    print('Fig 7.7: Alpha-Beta Weight Sensitivity')
    with open(os.path.join(MET_DIR, 'scoring_results.json')) as f:
        sr = json.load(f)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for mname, color in [
        ('lstm',        COL['lstm']),
        ('gru',         COL['gru']),
        ('transformer', COL['transformer']),
    ]:
        grid_d     = sr[mname]['grid_search']['grid']
        best_alpha = float(sr[mname]['grid_search']['best_alpha'])
        best_roc   = float(sr[mname]['grid_search']['best_roc'])
        alphas     = sorted(float(k) for k in grid_d.keys())
        roc_vals   = [float(grid_d[str(a)]) for a in alphas]

        ax.plot(alphas, roc_vals, 'o-', color=color, lw=2.0, ms=7,
                 label=DISP[mname])
        ax.plot(best_alpha, best_roc, '*', color=color, ms=14, zorder=6,
                 markeredgecolor='white', markeredgewidth=1.0)

    ax.axvspan(0.15, 0.25, alpha=0.07, color='black')
    ax.axvline(0.2, color='black', ls='--', alpha=0.35, lw=1.2)

    alphas_all = sorted(float(k) for k in sr['lstm']['grid_search']['grid'].keys())
    ax.set_xticks(alphas_all)
    ax.set_xticklabels([f'{a:.1f}' for a in alphas_all])
    ax.set_xlabel('Alpha  (weight of confidence score)', fontsize=12)
    ax.set_ylabel('ROC-AUC', fontsize=12)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(alphas_all)
    ax2.set_xticklabels([f'{1 - a:.1f}' for a in alphas_all], fontsize=10)
    ax2.set_xlabel('Beta  (weight of reconstruction score)', fontsize=12)
    ax2.spines['top'].set_visible(True)

    ax.legend(fontsize=10, loc='upper right')
    ax.set_title('Figure 7 — Composite Scoring Weight Sensitivity Analysis', pad=14)
    plt.tight_layout()
    save_fig(fig, 'fig7_7_alpha_beta')


# ══════════════════════════════════════════════════════════════
# FIGURE 7.8 — Latency vs. Accuracy Deployment Tradeoff
# Autoencoder bubble shown in dedicated "detection only" zone
# below a visual separator line. All four models have bubbles.
# Bubble legend at upper right with actual model colors.
# ══════════════════════════════════════════════════════════════
def fig_latency_accuracy():
    print('Fig 7.8: Latency vs. Accuracy Tradeoff')
    with open(os.path.join(MET_DIR, 'final_evaluation.json')) as f:
        ev = json.load(f)
    pred_d = ev['prediction']
    lat_d  = ev['latency']

    fig, ax = plt.subplots(figsize=(10, 7.2))

    # Zone boundaries
    AE_Y  = 83.35   # y-position for autoencoder bubble (no accuracy metric here)
    Y_SEP = 83.72   # separator between classifier zone and detection-only zone

    offsets = {
        'lstm'        : ( 0.14,  0.24),
        'gru'         : ( 0.14, -0.34),
        'transformer' : ( 0.14, -0.38),
    }

    # Plot the three classifiers
    for mname in ['lstm', 'gru', 'transformer']:
        acc   = pred_d[mname]['test_acc_mean']
        astd  = pred_d[mname]['test_acc_std']
        lat   = lat_d[mname]['latency_mean_ms']
        lstd  = lat_d[mname]['latency_std_ms']
        pars  = lat_d[mname]['parameters']
        thr   = lat_d[mname]['throughput_events_per_sec']
        color = COL[mname]
        sz    = pars / 750

        ax.scatter(lat, acc, s=sz, c=color, alpha=0.88,
                    edgecolors='white', linewidths=2.0, zorder=5)
        ax.errorbar(lat, acc, xerr=lstd, yerr=astd,
                     fmt='none', color=color, alpha=0.45,
                     capsize=4, linewidth=1.2, zorder=4)

        dx, dy = offsets[mname]
        ax.annotate(
            f'{DISP[mname]}\n{pars:,} params\n{thr:,.0f} ev/s',
            xy=(lat, acc),
            xytext=(lat + dx, acc + dy),
            fontsize=9.5, color=color, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=color, lw=1.1),
        )

    # Separator line between classifier zone and autoencoder zone
    ax.axhline(Y_SEP, color='#bbbbbb', ls='--', lw=1.0, alpha=0.90, zorder=2)
    ax.text(0.50, Y_SEP + 0.06,
            'Next-activity classifiers  (y-axis = test accuracy)',
            fontsize=7.5, color='#888888', va='bottom', style='italic')
    ax.text(0.50, Y_SEP - 0.07,
            'Anomaly detector  (no accuracy axis)',
            fontsize=7.5, color='#888888', va='top', style='italic')

    # Autoencoder: bubble in detection zone + vertical dashed line
    ae_lat = lat_d['autoencoder']['latency_mean_ms']
    ae_par = lat_d['autoencoder']['parameters']
    ae_thr = lat_d['autoencoder']['throughput_events_per_sec']
    ae_sz  = ae_par / 750

    ax.axvline(ae_lat, color=COL['autoencoder'], ls='--', lw=1.4,
                alpha=0.40, zorder=3)
    ax.scatter(ae_lat, AE_Y, s=ae_sz, c=COL['autoencoder'], alpha=0.88,
                edgecolors='white', linewidths=2.0, zorder=5)
    ax.annotate(
        f'Autoencoder\n{ae_par:,} params  |  {ae_thr:,.0f} ev/s\n(detection only)',
        xy=(ae_lat, AE_Y),
        xytext=(ae_lat - 0.25, AE_Y),
        fontsize=8.5, color=COL['autoencoder'], fontweight='bold',
        ha='right', va='center',
        arrowprops=dict(arrowstyle='->', color=COL['autoencoder'], lw=1.0),
    )

    ax.text(0.03, 0.97, 'Ideal: low latency, high accuracy',
             transform=ax.transAxes, fontsize=9, color='#555555', va='top',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', alpha=0.85))

    ax.set_xlabel('Mean Inference Latency  (ms per sequence)', fontsize=12)
    ax.set_ylabel('Test Accuracy  (%)', fontsize=12)
    ax.set_ylim(83.0, 88.2)
    ax.set_xlim(0.3, 7.8)

    # Only label the classifier accuracy range on y-axis
    # The 83.x zone is intentionally unlabeled (autoencoder visual zone)
    ax.set_yticks([84, 85, 86, 87, 88])

    # Bubble legend at UPPER RIGHT — all four models with actual colors
    for pk, pl, c in [
        (74924,  'Transformer  (75 k)',  COL['transformer']),
        (90823,  'Autoencoder  (91 k)',  COL['autoencoder']),
        (167852, 'GRU  (168 k)',         COL['gru']),
        (219308, 'LSTM  (219 k)',         COL['lstm']),
    ]:
        ax.scatter([], [], s=pk / 750, color=c, alpha=0.78, label=pl,
                    edgecolors='white', linewidths=1.5)

    ax.legend(title='Parameter count (bubble area)',
               title_fontsize=9, fontsize=8.5,
               loc='upper right', borderpad=0.9)

    ax.set_title('Figure 8 — Latency vs. Accuracy Deployment Tradeoff', pad=12)
    plt.tight_layout()
    save_fig(fig, 'fig7_8_latency_accuracy')

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('=' * 62)
    print('Chapter 7 Figure Generator')
    print(f'Output: {FIG_DIR}')
    print('=' * 62)

    # Figures using only JSON result files (fast, no model inference)
    fig_val_loss()
    fig_lr_decay()
    fig_pred_perf()
    fig_detection_heatmap()
    fig_alpha_beta()
    fig_latency_accuracy()

    # Figures requiring model inference
    # First run: 15-25 minutes. Subsequent runs: loads from cache.
    print('\nComputing score arrays for ROC and PR curves...')
    sc = compute_and_cache_scores()
    fig_roc_curves(sc)
    fig_pr_curves(sc)

    print('\n' + '=' * 62)
    print('All 8 figures saved to:', FIG_DIR)
    print('=' * 62)