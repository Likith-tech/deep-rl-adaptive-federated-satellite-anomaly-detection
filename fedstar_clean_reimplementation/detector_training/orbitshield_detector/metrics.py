"""Fixed global class space, explicit undefined values and deterministic selection."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score)


def classification_metrics(labels, probabilities, classes):
    y, p = np.asarray(labels), np.asarray(probabilities)
    if (y.ndim != 1 or y.dtype.kind not in 'iu' or len(y) == 0
            or p.shape != (len(y), classes) or not np.isfinite(p).all()
            or np.any(y < 0) or np.any(y >= classes) or np.any(p < 0)
            or not np.allclose(p.sum(1), 1, atol=1e-5)):
        raise ValueError('Invalid metric inputs/global label space')
    pred = p.argmax(1)  # ties go to lowest class index; binary threshold is strictly > .5
    labels_all = np.arange(classes)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=labels_all, zero_division=0)
    matrix = confusion_matrix(y, pred, labels=labels_all)
    predicted = matrix.sum(0)
    result = {'accuracy': float(accuracy_score(y, pred)), 'confusion_matrix': matrix.tolist(),
              'balanced_accuracy': float(recall[support > 0].mean()),
              'zero_division_policy': '0 in aggregate F-scores; null for undefined per-class precision/recall',
              'per_class': {str(i): {'precision': float(precision[i]) if predicted[i] else None,
                  'recall': float(recall[i]) if support[i] else None,
                  'f1': float(f1[i]) if support[i] + predicted[i] else None,
                  'support': int(support[i]), 'predicted': int(predicted[i])} for i in labels_all},
              'undefined_reasons': []}
    for average in ('macro', 'weighted'):
        pr, re, fs, _ = precision_recall_fscore_support(y, pred, labels=labels_all,
                                                       average=average, zero_division=0)
        result.update({f'{average}_precision': float(pr), f'{average}_recall': float(re), f'{average}_f1': float(fs)})
    if classes == 2:
        result['positive_class'] = result['per_class']['1']
        both = len(np.unique(y)) == 2
        result['roc_auc'] = float(roc_auc_score(y, p[:, 1])) if both else None
        result['average_precision'] = float(average_precision_score(y, p[:, 1])) if both else None
        if not both:
            result['undefined_reasons'].append('ROC-AUC/AP require both classes under this evaluation policy')
    return result


def better(candidate, incumbent, metric):
    """Maximize score, minimize loss, retain earlier checkpoint on exact ties."""
    score = candidate[metric]
    if score is None or not np.isfinite(score) or not np.isfinite(candidate['loss']):
        raise ValueError('Selection metric/loss is undefined')
    return incumbent is None or (score, -candidate['loss']) > (incumbent[metric], -incumbent['loss'])
