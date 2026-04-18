"""Generate PR and ROC curves from saved per-split probabilities.

Reads figures/probs/*.npz (produced by run.py during eval) and writes:
  figures/curves_rp_to_xview.png    — PR + ROC for direction A
  figures/curves_xview_to_rp.png    — PR + ROC for direction B

Each figure has two panels (PR on the left, ROC on the right) and overlays
the within-val and cross-test curves so the within-source / cross-domain
delta is visible at a glance.

Run with: python plot_curves.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve

HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent
PROBS = PROJECT_ROOT / "figures" / "probs"
OUT = PROJECT_ROOT / "figures"

NAVY = "#142454"
NAVY_SOFT = "#52618A"
BRASS = "#B6892C"
GRID = "#D9D2BF"


def load_split(key: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(PROBS / f"{key}_{split}.npz")
    return data["probs_pos"], data["labels_true"]


def make_figure(pretty_name: str, key: str) -> tuple[plt.Figure, dict]:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    aucs: dict[str, dict[str, float]] = {"within_val": {}, "cross_test": {}}

    panel_specs = [
        ("within_val", NAVY_SOFT, "within-val"),
        ("cross_test", BRASS, "cross-test"),
    ]

    # PR
    ax = axes[0]
    for split, color, label in panel_specs:
        probs, labels = load_split(key, split)
        precision, recall, _ = precision_recall_curve(labels, probs)
        pr_auc = auc(recall, precision)
        aucs[split]["pr_auc"] = float(pr_auc)
        ax.plot(recall, precision, color=color, linewidth=2.2,
                label=f"{label}  (AP = {pr_auc:.3f})")
    ax.set_xlabel("Recall (aircraft)")
    ax.set_ylabel("Precision (aircraft)")
    ax.set_title(f"{pretty_name}  ·  Precision–Recall")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower left", framealpha=0.95)
    ax.grid(color=GRID, alpha=0.6, linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_color(NAVY_SOFT)

    # ROC
    ax = axes[1]
    for split, color, label in panel_specs:
        probs, labels = load_split(key, split)
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        aucs[split]["roc_auc"] = float(roc_auc)
        ax.plot(fpr, tpr, color=color, linewidth=2.2,
                label=f"{label}  (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color=NAVY_SOFT, alpha=0.5, linewidth=1.0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{pretty_name}  ·  ROC")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(color=GRID, alpha=0.6, linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_color(NAVY_SOFT)

    fig.tight_layout()
    return fig, aucs


def main() -> None:
    summary: dict[str, dict] = {}

    fig_a, auc_a = make_figure("Direction A · RP → xView", "rareplanes_to_xview")
    out_a = OUT / "curves_rp_to_xview.png"
    fig_a.savefig(out_a, dpi=140, bbox_inches="tight")
    plt.close(fig_a)
    summary["rareplanes_to_xview"] = auc_a
    print(f"saved {out_a.relative_to(PROJECT_ROOT)}")

    fig_b, auc_b = make_figure("Direction B · xView → RP", "xview_to_rareplanes")
    out_b = OUT / "curves_xview_to_rp.png"
    fig_b.savefig(out_b, dpi=140, bbox_inches="tight")
    plt.close(fig_b)
    summary["xview_to_rareplanes"] = auc_b
    print(f"saved {out_b.relative_to(PROJECT_ROOT)}")

    import json
    summary_path = OUT / "curves_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"saved {summary_path.relative_to(PROJECT_ROOT)}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
