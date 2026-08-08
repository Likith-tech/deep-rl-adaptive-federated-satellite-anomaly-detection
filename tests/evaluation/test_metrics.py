from __future__ import annotations

import numpy as np

from src.evaluation.metrics import compute_metrics


def test_compute_metrics_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.9, 0.8])

    metrics = compute_metrics(y_true, y_pred, y_prob)

    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.false_positive_rate == 0.0
    assert metrics.roc_auc == 1.0
    assert metrics.confusion_matrix == [[2, 0], [0, 2]]


def test_compute_metrics_all_wrong_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])

    metrics = compute_metrics(y_true, y_pred)

    assert metrics.accuracy == 0.0
    assert metrics.recall == 0.0
    assert metrics.false_positive_rate == 1.0


def test_compute_metrics_confusion_matrix_layout():
    # 1 TN, 1 FP, 1 FN, 1 TP
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])

    metrics = compute_metrics(y_true, y_pred)

    tn, fp, fn, tp = np.array(metrics.confusion_matrix).ravel()
    assert (tn, fp, fn, tp) == (1, 1, 1, 1)
    assert metrics.false_positive_rate == 0.5


def test_compute_metrics_roc_auc_none_without_probabilities():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])

    metrics = compute_metrics(y_true, y_pred, y_prob=None)

    assert metrics.roc_auc is None
