"""Phase 5 — cross-client evaluation helpers.

Everything here operates on already-computed per-client results (see
`src.training.local_trainer.LocalTrainingResult`); it does not train or
load models itself except for producing confusion-matrix plots from a
saved checkpoint's predictions.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ALL_CATEGORIES = ("normal", "dos", "probe", "r2l", "u2r")

METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate")


@dataclass
class ClientSummary:
    client_id: str
    total_samples: int
    anomaly_percentage: float
    categories_present: list[str]
    categories_absent: list[str]
    local_train_metrics: dict
    global_val_metrics: dict


def categories_present_absent(category_counts: dict[str, int]) -> tuple[list[str], list[str]]:
    present = [c for c in ALL_CATEGORIES if category_counts.get(c, 0) > 0]
    absent = [c for c in ALL_CATEGORIES if category_counts.get(c, 0) == 0]
    return present, absent


def build_client_summary(
    client_id: str,
    client_stats: dict,
    local_train_metrics: dict,
    global_val_metrics: dict,
) -> ClientSummary:
    present, absent = categories_present_absent(client_stats["category_counts"])
    return ClientSummary(
        client_id=client_id,
        total_samples=client_stats["total_samples"],
        anomaly_percentage=client_stats["anomaly_percentage"],
        categories_present=present,
        categories_absent=absent,
        local_train_metrics=local_train_metrics,
        global_val_metrics=global_val_metrics,
    )


@dataclass
class AggregateStat:
    mean: float
    median: float
    minimum: float
    maximum: float
    stdev: float

    def to_dict(self) -> dict:
        return {
            "mean": self.mean,
            "median": self.median,
            "min": self.minimum,
            "max": self.maximum,
            "stdev": self.stdev,
        }


def _aggregate_one(values: list[float]) -> AggregateStat:
    return AggregateStat(
        mean=float(statistics.mean(values)),
        median=float(statistics.median(values)),
        minimum=float(min(values)),
        maximum=float(max(values)),
        stdev=float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    )


def compute_aggregate_statistics(client_summaries: list[ClientSummary]) -> dict[str, dict]:
    """Aggregate global-validation metrics across all clients.

    ROC-AUC values that are None (not available) are excluded from that
    metric's aggregation rather than treated as zero.
    """
    aggregates: dict[str, dict] = {}
    for key in METRIC_KEYS:
        values = [
            s.global_val_metrics[key]
            for s in client_summaries
            if s.global_val_metrics.get(key) is not None
        ]
        if values:
            aggregates[key] = _aggregate_one(values).to_dict()
        else:
            aggregates[key] = None
    return aggregates


def rank_clients_by_metric(client_summaries: list[ClientSummary], metric: str = "f1") -> list[ClientSummary]:
    ranked = [s for s in client_summaries if s.global_val_metrics.get(metric) is not None]
    return sorted(ranked, key=lambda s: s.global_val_metrics[metric], reverse=True)


def plot_confusion_matrix(cm: list[list[int]], title: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    im = ax.imshow(cm_arr, cmap="Blues")
    labels = ["Normal (0)", "Anomaly (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=11, pad=12)
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm_arr[i, j]), ha="center", va="center",
                color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black", fontsize=13,
            )
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_curves(client_id: str, epochs_data: list[dict], best_epoch: int, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [e["epoch"] for e in epochs_data]
    train_loss = [e["train_loss"] for e in epochs_data]
    val_loss = [e["val_loss"] for e in epochs_data]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(epochs, train_loss, label="Local training loss", color="#3b82f6")
    ax.plot(epochs, val_loss, label="Global validation loss", color="#ef4444")
    ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.6, label="Best epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title(f"{client_id} — Training / Global Validation Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
