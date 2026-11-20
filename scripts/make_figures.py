"""Generate the three evaluation figures from results/results.json.

Outputs (1600 x 1000 px, 200 dpi):
  results/fig_roc.png
  results/fig_capacity_dist.png
  results/fig_robustness.png

The script reads only the artifacts written by ``capacity_tunnel.cli eval``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from capacity_tunnel import detector  # noqa: E402

COLORS = {
    "b32": "#2b6cb0",
    "timing": "#c53030",
    "hardened": "#2f855a",
    "doh_browse": "#6b46c1",
    "bulk": "#b7791f",
    "interactive": "#4a5568",
    "overall": "#1a202c",
}
LABELS = {
    "b32": "b32 (5-bit labels)",
    "timing": "timing (1 bit/packet)",
    "hardened": "hardened (6-ary, padded)",
    "doh_browse": "DoH browsing",
    "bulk": "bulk download",
    "interactive": "interactive",
    "overall": "all tunnels",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.7,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "savefig.facecolor": "white",
        }
    )


def fig_roc(results: Dict, flows_path: str, out_path: str) -> None:
    data = np.load(flows_path, allow_pickle=False)
    y = data["y"].astype(int)
    scores = data["oof_scores"]
    labels = np.array([str(v) for v in data["class_labels"]])

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for mode in ("b32", "timing", "hardened"):
        mask = (labels == mode) | (y == 0)
        fpr, tpr, _ = detector.roc_curve(y[mask], scores[mask])
        auc = results["per_mode"][mode]["auc"]
        ax.plot(fpr, tpr, color=COLORS[mode], lw=1.6, label="%s, AUC %.4f" % (LABELS[mode], auc))
    fpr, tpr, _ = detector.roc_curve(y, scores)
    ax.plot(
        fpr,
        tpr,
        color=COLORS["overall"],
        lw=1.2,
        ls="--",
        label="%s, AUC %.4f" % (LABELS["overall"], results["detector"]["auc"]),
    )
    if "oof_scores_entropy_only" in data:
        ent = data["oof_scores_entropy_only"]
        ent_auc = results["baselines"]["entropy_only"]["auc"]
        fpr, tpr, _ = detector.roc_curve(y, ent)
        ax.plot(fpr, tpr, color="#dd6b20", lw=1.4, ls="-.", label="entropy-only baseline, AUC %.4f" % ent_auc)
    ax.plot([0, 1], [0, 1], color="#a0aec0", lw=0.8, ls=":")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.005)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Out-of-fold ROC, 5-fold stratified CV (n = %d flows)" % y.size)
    ax.legend(loc="lower right")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_capacity_dist(flows_path: str, out_path: str) -> None:
    data = np.load(flows_path, allow_pickle=False)
    X = data["X"]
    names = [str(v) for v in data["feature_names"]]
    labels = np.array([str(v) for v in data["class_labels"]])
    idx = names.index("log_capacity_bps")
    values = X[:, idx]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    bins = np.linspace(0.0, 3.2, 46)
    order = ("hardened", "timing", "b32", "doh_browse", "bulk", "interactive")
    for cls in order:
        mask = labels == cls
        ax.hist(
            values[mask],
            bins=bins,
            density=True,
            histtype="step",
            lw=1.4,
            color=COLORS[cls],
            label="%s (median %.0f bps)" % (LABELS[cls], float(np.median(10.0 ** values[mask] - 1.0))),
        )
    ax.set_xlabel("log10(1 + estimated capacity [bit/s])")
    ax.set_ylabel("density")
    ax.set_title("Per-flow capacity estimate by class (n = %d per class)" % int((labels == "b32").sum()))
    ax.set_ylim(0.0, 18.5)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_robustness(results: Dict, out_path: str) -> None:
    padding = results["robustness"]["padding_sweep"]
    jitter = results["robustness"]["timing_jitter_sweep"]

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8.0, 5.0), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.15]}
    )
    xp = [p["scale"] for p in padding]
    xj = [p["scale"] for p in jitter]

    top.plot(xp, [p["auc"] for p in padding], color=COLORS["hardened"], lw=1.6, marker="o", ms=4.0, label="size padding")
    top.plot(xj, [p["auc"] for p in jitter], color="#c53030", lw=1.6, marker="s", ms=4.0, label="timing jitter")
    top.axhline(0.95, color="#a0aec0", lw=0.9, ls=":")
    top.set_ylim(0.94, 1.005)
    top.set_ylabel("AUC vs held-out benign")
    top.set_title("(a) detection AUC under evasion")
    top.legend(loc="lower left", ncol=2)

    bottom.plot(xp, [p["median_capacity_bps"] for p in padding], color=COLORS["hardened"], lw=1.6, marker="o", ms=4.0, label="size padding")
    bottom.plot(xj, [p["median_capacity_bps"] for p in jitter], color="#c53030", lw=1.6, marker="s", ms=4.0, label="timing jitter")
    bottom.set_ylabel("median estimated capacity [bit/s]")
    bottom.set_xlabel("Evasion strength (multiplier on the default envelope)")
    bottom.set_title("(b) capacity cost of the same evasion")
    bottom.set_ylim(0.0, 150.0)
    bottom.legend(loc="upper right", ncol=2)

    for ax in (top, bottom):
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(set(xp + xj)))
        ax.set_xticklabels(["%gx" % v for v in sorted(set(xp + xj))])

    fig.suptitle("Hardened tunnel under evasion; detector frozen at training-time parameters", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=os.path.join(ROOT, "results", "results.json"))
    parser.add_argument("--flows", default=os.path.join(ROOT, "results", "flow_scores.npz"))
    parser.add_argument("--out", default=os.path.join(ROOT, "results"))
    args = parser.parse_args()

    with open(args.results, "r", encoding="utf-8") as fh:
        results = json.load(fh)

    _style()
    os.makedirs(args.out, exist_ok=True)
    fig_roc(results, args.flows, os.path.join(args.out, "fig_roc.png"))
    fig_capacity_dist(args.flows, os.path.join(args.out, "fig_capacity_dist.png"))
    fig_robustness(results, os.path.join(args.out, "fig_robustness.png"))
    print("wrote fig_roc.png, fig_capacity_dist.png, fig_robustness.png to %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
