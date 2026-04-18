"""Evaluation harness. Reports accuracy, precision, recall, macro-F1,
per-class F1, confusion matrix, and a trivial majority-class baseline.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader


@dataclass
class EvalReport:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_per_class: list[float]
    confusion: list[list[int]]
    baseline_accuracy: float
    baseline_f1_macro: float
    n_samples: int
    probs_pos: np.ndarray | None = None  # P(aircraft) per sample; not serialized
    labels_true: np.ndarray | None = None  # ground-truth labels; not serialized

    def as_dict(self) -> dict:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision_macro": round(self.precision_macro, 4),
            "recall_macro": round(self.recall_macro, 4),
            "f1_macro": round(self.f1_macro, 4),
            "f1_per_class": [round(v, 4) for v in self.f1_per_class],
            "confusion": self.confusion,
            "baseline_accuracy": round(self.baseline_accuracy, 4),
            "baseline_f1_macro": round(self.baseline_f1_macro, 4),
            "n_samples": self.n_samples,
        }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, loader: DataLoader, device: str
) -> EvalReport:
    model.eval()
    model.to(device)
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[float] = []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
        all_probs.extend(probs)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    accuracy = float((y_true == y_pred).mean())
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, f1_per, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    # trivial majority-class baseline
    counter = Counter(y_true.tolist())
    majority = max(counter, key=counter.get)
    baseline_preds = np.full_like(y_true, majority)
    baseline_acc = float((y_true == baseline_preds).mean())
    baseline_f1 = float(f1_score(y_true, baseline_preds, average="macro", zero_division=0))

    return EvalReport(
        accuracy=accuracy,
        precision_macro=float(prec),
        recall_macro=float(rec),
        f1_macro=float(f1),
        f1_per_class=[float(v) for v in f1_per],
        confusion=cm,
        baseline_accuracy=baseline_acc,
        baseline_f1_macro=baseline_f1,
        n_samples=int(len(y_true)),
        probs_pos=y_prob,
        labels_true=y_true,
    )


def save_confusion_plot(cm: list[list[int]], title: str, out_path: Path) -> None:
    arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(
        arr,
        annot=True,
        fmt="d",
        cmap=["#EBECF5", "#C89B3C"],
        cbar=False,
        linewidths=0.5,
        linecolor="#142454",
        xticklabels=["non-aircraft", "aircraft"],
        yticklabels=["non-aircraft", "aircraft"],
        ax=ax,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_comparison_bar(
    reports: dict[str, EvalReport], out_path: Path
) -> None:
    directions = list(reports.keys())
    f1_vals = [reports[d].f1_macro for d in directions]
    baseline_vals = [reports[d].baseline_f1_macro for d in directions]

    x = np.arange(len(directions))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(x - width / 2, baseline_vals, width, label="majority baseline", color="#142454")
    ax.bar(x + width / 2, f1_vals, width, label="trained model", color="#C89B3C")
    ax.set_xticks(x)
    ax.set_xticklabels(directions)
    ax.set_ylabel("macro F1")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
