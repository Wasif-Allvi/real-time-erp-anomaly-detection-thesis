# ============================================================
# generate_plots_v2.py
# Result figures for the thesis (Chapter 7), print-ready version
#
# Differences to generate_plots.py:
#   * runs WITHOUT PyTorch: reads only the metric JSON files, the
#     training histories, results/figures/raw_scores.npz and the
#     event-level labels written by 13_event_level_evaluation.py
#   * Times-like font (STIX) to match the thesis text, sizes set
#     for the printed text width (figures are 6.3 in wide)
#   * vector PDF output (plus PNG), written to results/figures_v2/
#   * Fig 7.1: mean curve only while all three runs are active
#   * Fig 7.3: per-run dot plot instead of truncated bars
#   * Fig 7.4/7.5: compact layout, legend outside the plot area
#   * Fig 7.4/7.5: event-level labels (a prefix is anomalous only once
#              it contains an altered event, see 13_...)
#   * Fig 7.6: threshold-free per-type ROC-AUC heatmap, each type against
#              normal prefixes of the same length (from 13_...)
#   * Fig 7.7: composite weight grid alpha = 0, 0.1, ..., 1 (from 13_...)
#   * Fig 7.8: latency of all four models against the 10 ms real-time
#              bound (DR4); accuracy in the labels, no accuracy axis
#              for the autoencoder
#
# Requires: 10_significance.py and 13_event_level_evaluation.py to have run.
# ============================================================

import os
import json
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

ROOT = os.path.dirname(os.path.abspath(__file__))
MET = os.path.join(ROOT, "results", "metrics")
PREP = os.path.join(ROOT, "results", "preprocessed")
RAW = os.path.join(ROOT, "results", "figures", "raw_scores.npz")
LABELS = os.path.join(MET, "event_labels.npz")
OUT = os.path.join(ROOT, "results", "figures_v2")
os.makedirs(OUT, exist_ok=True)

W = 6.3  # figure width in inches (thesis text width is about 5.9 in)

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#cccccc",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.4,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
})

COL = {"lstm": "#1565C0", "gru": "#2E7D32", "transformer": "#E65100",
       "autoencoder": "#6A1B9A", "joint": "#AD1457", "iforest": "#424242",
       "base": "#9E9E9E"}
SHADES = {"lstm": ["#0D47A1", "#1E88E5", "#90CAF9"],
          "gru": ["#1B5E20", "#43A047", "#A5D6A7"],
          "transformer": ["#BF360C", "#F4511E", "#FFAB91"]}
DISP = {"lstm": "LSTM", "gru": "GRU", "transformer": "Transformer"}
MODELS = ["lstm", "gru", "transformer"]


def load(name):
    with open(os.path.join(MET, name)) as f:
        return json.load(f)


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{stem}_v2.{ext}"))
    plt.close(fig)
    print(f"  saved {stem}_v2")


# ------------------------------------------------------------ 7.1
def fig_val_loss():
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.35), sharey=True)
    for ax, m in zip(axes, MODELS):
        runs = load(f"{m}_train_history.json")
        curves = [np.array(r["history"]["val_loss"]) for r in runs]
        for i, (r, v) in enumerate(zip(runs, curves)):
            ax.plot(np.arange(1, len(v) + 1), v, color=SHADES[m][i], lw=0.8,
                    alpha=0.9, label=f"Run {i + 1}")
            b = r["best_epoch"]
            ax.plot(b, v[b - 1], "o", ms=3.2, color=SHADES[m][i],
                    markeredgecolor="black", markeredgewidth=0.4, zorder=5)
        common = min(len(v) for v in curves)
        stack = np.vstack([v[:common] for v in curves])
        xs = np.arange(1, common + 1)
        ax.plot(xs, stack.mean(0), color="black", lw=1.1, label="Mean (all runs active)")
        ax.set_title(DISP[m])
        ax.set_xlabel("Epoch")
        ax.set_ylim(0.36, 1.0)
        ax.set_xlim(1, max(len(v) for v in curves))
    axes[0].set_ylabel("Validation loss")
    for ax in axes:
        ax.legend(loc="upper right", fontsize=6, handlelength=1.4,
                  borderpad=0.3, labelspacing=0.25)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig7_1_val_loss")


# ------------------------------------------------------------ 7.2
def fig_lr_decay():
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.0), sharey=True)
    for ax, m in zip(axes, MODELS):
        runs = load(f"{m}_train_history.json")
        for i, r in enumerate(runs):
            lr = r["history"]["learning_rates"]
            ax.step(np.arange(1, len(lr) + 1), lr, where="post",
                    color=SHADES[m][i], lw=1.2, label=f"Run {i + 1}", zorder=4 - i)
        ax.set_yscale("log")
        ax.set_title(DISP[m])
        ax.set_xlabel("Epoch")
        ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.legend(loc="upper right", fontsize=6.5)
    axes[0].set_ylabel("Learning rate")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig7_2_lr_decay")


# ------------------------------------------------------------ 7.3
def fig_pred_performance():
    sig = load("significance_results.json")["per_run"]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5))
    panels = [("accuracy_pct", "Test accuracy (%)", 81.22, "{:.2f}"),
              ("f1_macro", "F1-macro (37 classes)", 0.2890, "{:.4f}")]
    rng = np.random.default_rng(0)
    for ax, (key, label, first, fmt) in zip(axes, panels):
        for x, m in enumerate(MODELS):
            vals = np.array(sig[m][key])
            jit = rng.uniform(-0.08, 0.08, len(vals))
            ax.scatter(x + jit, vals, s=16, color=COL[m], alpha=0.55,
                       edgecolor="none", zorder=3)
            mu, sd = vals.mean(), vals.std(ddof=0)
            ax.errorbar(x + 0.22, mu, yerr=sd, fmt="D", ms=4.5, color=COL[m],
                        ecolor="black", elinewidth=0.8, capsize=3, zorder=4)
            ax.annotate(fmt.format(mu), (x + 0.32, mu), va="center", fontsize=7.5)
        ax.axhline(first, color=COL["base"], ls="--", lw=0.9)
        ax.annotate(f"first-cycle LSTM ({fmt.format(first)})", (-0.35, first),
                    xytext=(0, 3), textcoords="offset points", fontsize=7,
                    color="#616161")
        ax.set_xticks(range(3))
        ax.set_xticklabels([DISP[m] for m in MODELS])
        ax.set_xlim(-0.45, 2.75)
        ax.set_ylabel(label)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig7_3_pred_performance")


def detection_scores():
    """Run-averaged scores of 12_export_raw_scores.py and the event-level labels."""
    raw = np.load(RAW)
    sc = {"true_binary": np.load(LABELS)["event_labels"]}
    for m in MODELS:
        sc[f"{m}_confidence"] = raw[f"test_conf_{m}"].mean(0)
        sc[f"{m}_joint_recon"] = raw[f"test_err_{m}"].mean(0)
    sc["autoencoder"] = raw["test_ae"].mean(0)
    return sc


def curve_entries(sc, iso):
    e = [("lstm_confidence", "LSTM confidence", COL["lstm"], "--", 1.0),
         ("gru_confidence", "GRU confidence", COL["gru"], "--", 1.0),
         ("transformer_confidence", "Transformer confidence", COL["transformer"], "--", 1.0),
         ("gru_joint_recon", "GRU joint reconstruction", COL["joint"], "-", 1.3),
         ("autoencoder", "Dedicated autoencoder", COL["autoencoder"], "-", 1.7)]
    return [(sc[k], lab, c, ls, lw) for k, lab, c, ls, lw in e]


# ------------------------------------------------------------ 7.4 / 7.5
def fig_roc(sc, iso):
    y = sc["true_binary"]
    fig, ax = plt.subplots(figsize=(W, 3.1))
    ax.plot([0, 1], [0, 1], ":", color=COL["base"], lw=1.0, label="Random (0.50)")
    for s, lab, c, ls, lw in curve_entries(sc, iso):
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, ls=ls, color=c, lw=lw, label=f"{lab} ({auc(fpr, tpr):.4f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_aspect("equal")
    ax.legend(loc="center left", bbox_to_anchor=(1.03, 0.5), title="ROC-AUC",
              title_fontsize=7.5)
    save(fig, "fig7_4_roc_curves")


def fig_pr(sc, iso):
    y = sc["true_binary"]
    base = float(y.mean())
    fig, ax = plt.subplots(figsize=(W, 3.1))
    ax.axhline(base, ls=":", color=COL["base"], lw=1.0, label=f"Random ({base:.4f})")
    for s, lab, c, ls, lw in curve_entries(sc, iso):
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, ls=ls, color=c, lw=lw,
                label=f"{lab} ({average_precision_score(y, s):.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal")
    ax.legend(loc="center left", bbox_to_anchor=(1.03, 0.5), title="AUPRC",
              title_fontsize=7.5)
    save(fig, "fig7_5_pr_curves")


# ------------------------------------------------------------ 7.6
def fig_per_type_auc():
    pt = load("event_level_results.json")["scorers"]
    rows = [("lstm_confidence", "LSTM confidence"), ("gru_confidence", "GRU confidence"),
            ("transformer_confidence", "Transformer confidence"),
            ("lstm_joint_recon", "LSTM joint recon."), ("gru_joint_recon", "GRU joint recon."),
            ("transformer_joint_recon", "Transformer joint recon."),
            ("autoencoder", "Dedicated autoencoder")]
    types = ["control_flow", "data", "resource", "temporal", "cross_entity"]
    cols = ["Control-flow", "Data", "Resource", "Temporal", "Cross-entity"]
    M = np.array([[pt[k]["per_type_length_matched"][t]["auc"] for t in types] for k, _ in rows])
    LO = np.array([[pt[k]["per_type_length_matched"][t]["ci95"][0] for t in types] for k, _ in rows])
    HI = np.array([[pt[k]["per_type_length_matched"][t]["ci95"][1] for t in types] for k, _ in rows])

    fig, ax = plt.subplots(figsize=(W, 2.9))
    norm = TwoSlopeNorm(vmin=0.3, vcenter=0.5, vmax=1.0)
    im = ax.imshow(M, cmap="RdBu", norm=norm, aspect="auto")
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            sig = LO[i, j] > 0.5 or HI[i, j] < 0.5      # interval excludes 0.5
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color=("white" if (v > 0.8 or v < 0.36) else "black") if sig else "#8a8a8a",
                    fontweight="bold" if sig else "normal")
    for yline in (2.5, 5.5):
        ax.axhline(yline, color="white", lw=2.5)
    ax.set_xticks(range(5))
    ax.set_xticklabels(cols)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab for _, lab in rows])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, ticks=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    cb.set_label("Per-type ROC-AUC (0.5 = random)")
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(0.5)
    save(fig, "fig7_6_per_type_auc")


# ------------------------------------------------------------ 7.7
def fig_alpha_beta():
    comp = load("event_level_results.json")["composite"]
    fig, ax = plt.subplots(figsize=(W, 2.5))
    for m in MODELS:
        g = comp[m]["grid_roc_auc"]
        a = sorted(float(k) for k in g)
        v = [g[str(x)] for x in a]
        ax.plot(a, v, "o-", color=COL[m], ms=3.5, lw=1.2, label=DISP[m])
        best = int(np.argmax(v))
        ax.plot(a[best], v[best], "*", ms=9, color=COL[m], markeredgecolor="black",
                markeredgewidth=0.4, zorder=5)
    ax.set_xlabel(r"$\alpha$ (weight of the confidence score); $\beta = 1-\alpha$")
    ax.set_ylabel("ROC-AUC of composite score")
    ax.set_xticks([round(x, 1) for x in np.arange(0, 1.01, 0.1)])
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "fig7_7_alpha_beta")


# ------------------------------------------------------------ 7.8
def fig_latency():
    lat = load("final_evaluation.json")["latency"]
    pred = load("final_evaluation.json")["prediction"]
    names = ["lstm", "gru", "transformer", "autoencoder"]
    labels = [f"LSTM\n({pred['lstm']['test_acc_mean']:.2f}% acc.)",
              f"GRU\n({pred['gru']['test_acc_mean']:.2f}% acc.)",
              f"Transformer\n({pred['transformer']['test_acc_mean']:.2f}% acc.)",
              f"Autoencoder\n(AUPRC {load('event_level_results.json')['scorers']['autoencoder']['auprc']:.3f})"]
    colors = [COL["lstm"], COL["gru"], COL["transformer"], COL["autoencoder"]]
    means = [lat[n]["latency_mean_ms"] for n in names]
    p95 = [lat[n]["latency_p95_ms"] for n in names]
    yy = np.arange(len(names))[::-1].astype(float)
    fig, ax = plt.subplots(figsize=(W, 2.6))
    ax.barh(yy, means, color=colors, alpha=0.85, height=0.42, label="Mean latency")
    ax.scatter(p95, yy, marker="|", s=110, color="black", lw=1.3,
               label="95th percentile", zorder=4)
    for yv, mv, pv, n in zip(yy, means, p95, names):
        ax.annotate(f"mean {mv:.2f} ms, p95 {pv:.2f} ms, "
                    f"{lat[n]['throughput_events_per_sec']:,.0f} events/s",
                    (0.05, yv - 0.36), fontsize=6.8, va="center", color="#333333")
    ax.axvline(10, color="#C62828", ls="--", lw=1.0)
    ax.annotate("10 ms bound (DR4)", (10, yy[0] + 0.35), xytext=(-3, 0),
                textcoords="offset points", ha="right", va="center", fontsize=7,
                color="#C62828")
    ax.set_yticks(yy)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_ylim(-0.75, len(names) - 0.4)
    ax.set_xlim(0, 10.5)
    ax.set_xlabel("Inference latency (ms per sequence, CPU only)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="center right", fontsize=7)
    fig.tight_layout()
    save(fig, "fig7_8_latency")


def main():
    print("generate_plots_v2.py")
    sc = detection_scores()
    iso = None
    fig_val_loss()
    fig_lr_decay()
    fig_pred_performance()
    fig_roc(sc, iso)
    fig_pr(sc, iso)
    fig_per_type_auc()
    fig_alpha_beta()
    fig_latency()
    print(f"Figures written to {OUT}")


if __name__ == "__main__":
    main()
