"""Binary classification metrics for anomaly detection.

Accuracy alone is misleading for imbalanced/near-balanced anomaly
detection, so this module always reports precision/recall/F1/FPR
alongside it, plus ROC-AUC and the confusion matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    false_positive_rate: float
    confusion_matrix: list[list[int]]  # [[TN, FP], [FN, TP]]

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "false_positive_rate": self.false_positive_rate,
            "confusion_matrix": self.confusion_matrix,
        }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> ClassificationMetrics:
    """Compute standard binary metrics.

    y_true, y_pred: 0/1 integer arrays.
    y_prob: predicted probability of the positive (anomaly) class, used
    for ROC-AUC. If None (or only one class present in y_true), ROC-AUC
    is reported as None rather than fabricated.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    roc_auc = None
    if y_prob is not None and len(np.unique(y_true)) == 2:
        roc_auc = float(roc_auc_score(y_true, y_prob))

    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc_auc,
        false_positive_rate=fpr,
        confusion_matrix=cm.tolist(),
    )
