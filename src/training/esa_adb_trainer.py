"""Training loop for the ESA-ADB centralized baseline (Experiment 1).

Isolated from src/training/{baseline_trainer,temporal_trainer,opssat_trainer}.py
— none of those files are imported or modified here. Reuses two
already-generic, dataset-agnostic pieces directly: the model
(src/models/temporal_gru_attention.py, unmodified, config-driven
input_dim) and the metrics module (src/evaluation/metrics.py,
unmodified) — extended locally with PR-AUC (average_precision_score),
since PR-AUC is not part of the shared metrics module and this file
does not modify that module to add it.

    ESA-ADB centralized (all 58 channels pooled) train/validation/test
    SequenceDataset objects (src/satellite/esa_adb/pipeline.py)
        -> DataLoader -> TemporalGRUAttention(input_dim=1)
        -> BCEWithLogitsLoss(pos_weight=<train-only class ratio>)
        -> Adam -> validation loss each epoch (at a fixed 0.5 threshold,
           for monitoring/checkpoint-selection only) -> best checkpoint
           (by validation loss) -> early stopping
        -> AFTER training: decision threshold selected by scanning
           [0.01, 0.99] and maximizing F1 on the VALIDATION set only,
           using the frozen best-checkpoint's probabilities
        -> test set evaluated exactly once, at the frozen threshold

Class imbalance handling: BCEWithLogitsLoss's `pos_weight` is set to
(train negative count / train positive count), computed from the TRAIN
split only — this is the simplest standard technique for severe class
imbalance (train anomaly rate here is ~3.2%) and is documented
explicitly in every saved config so it is never silently assumed.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import ClassificationMetrics, compute_metrics
from src.models.temporal_gru_attention import TemporalGRUAttention
from src.preprocessing.sequences import SequenceDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op if no GPU present


def make_dataloader(dataset: SequenceDataset, batch_size: int, shuffle: bool) -> DataLoader:
    tensor_dataset = TensorDataset(
        torch.from_numpy(dataset.sequences.copy()),
        torch.from_numpy(dataset.labels.astype("float32").copy()),
    )
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)


def compute_pos_weight(labels: np.ndarray) -> float:
    """(negative count / positive count) from the given labels. Callers
    must pass ONLY the train split's labels — using validation/test
    labels here would leak class-balance information into training."""
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0:
        raise ValueError("Cannot compute pos_weight: zero positive labels in the given split.")
    return n_neg / n_pos


@dataclass
class ExtendedMetrics:
    classification: ClassificationMetrics
    pr_auc: float | None

    def to_dict(self) -> dict:
        d = self.classification.to_dict()
        d["pr_auc"] = self.pr_auc
        return d


def compute_extended_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> ExtendedMetrics:
    base = compute_metrics(y_true, y_pred, y_prob)
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else None
    return ExtendedMetrics(classification=base, pr_auc=pr_auc)


def run_inference(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Forward pass only (no loss, no labels needed for the loss) —
    returns (y_prob, y_true) as plain numpy arrays, in loader order."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits, _ = model(X_batch)
            logits = logits.squeeze(-1)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(y_batch.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels).astype(int)


def run_validation_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, ClassificationMetrics]:
    """Per-epoch monitoring only — uses a FIXED 0.5 threshold, never the
    validation-selected decision threshold (that selection happens once,
    after training, in select_threshold_by_f1 below)."""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits, _ = model(X_batch)
            logits = logits.squeeze(-1)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            n_samples += len(y_batch)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels).astype(int)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    avg_loss = total_loss / n_samples
    return avg_loss, metrics


def select_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray | None = None) -> dict:
    """Threshold-selection method: grid search over `thresholds`
    (default 0.01 to 0.99 step 0.01), scoring each by F1 on the given
    (VALIDATION-ONLY) labels/probabilities. Ties broken by the LOWEST
    threshold among those achieving the max F1 (deterministic,
    reproducible). Callers must never pass test labels/probabilities
    here."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.0, 0.01), 2)

    from sklearn.metrics import f1_score

    scores = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        scores.append((float(t), float(f1)))

    best_threshold, best_f1 = max(scores, key=lambda ts: (ts[1], -ts[0]))
    return {
        "method": "grid search over threshold in [0.01, 0.99] step 0.01, "
                  "maximizing F1 on the validation set only; ties broken by lowest threshold",
        "selected_threshold": best_threshold,
        "selected_validation_f1": best_f1,
        "all_scores": scores,
    }


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_precision: float
    val_recall: float
    val_f1: float


@dataclass
class TrainingHistory:
    epochs: list[EpochRecord] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    stopped_early: bool = False
    training_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "epochs": [vars(e) for e in self.epochs],
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "stopped_early": self.stopped_early,
            "training_seconds": self.training_seconds,
        }


def train_esa_adb_baseline(
    train_dataset: SequenceDataset,
    validation_dataset: SequenceDataset,
    config: dict,
    checkpoint_dir: Path,
    checkpoint_name: str = "best_model.pt",
) -> TrainingHistory:
    seed = config["training"]["seed"]
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = make_dataloader(train_dataset, config["training"]["batch_size"], shuffle=True)
    val_loader = make_dataloader(validation_dataset, config["training"]["batch_size"], shuffle=False)

    model = TemporalGRUAttention(
        input_dim=config["input_dim"],
        projection_dim=config["projection_dim"],
        hidden_dim=config["hidden_dim"],
        num_gru_layers=config["num_gru_layers"],
        dropout=config["dropout"],
        output_dim=config["output_dim"],
    ).to(device)

    pos_weight_value = compute_pos_weight(train_dataset.labels)
    config["class_weighting"] = {
        "used": True,
        "method": "BCEWithLogitsLoss(pos_weight=train_negative_count/train_positive_count)",
        "pos_weight": pos_weight_value,
        "computed_from": "train split only",
    }
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history = TrainingHistory()
    patience = config["training"]["early_stopping_patience"]
    epochs_without_improvement = 0

    start_time = time.time()

    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        total_train_loss = 0.0
        n_train = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits, _ = model(X_batch)
            logits = logits.squeeze(-1)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(y_batch)
            n_train += len(y_batch)

        train_loss = total_train_loss / n_train
        val_loss, val_metrics = run_validation_loss(model, val_loader, criterion, device)

        history.epochs.append(
            EpochRecord(
                epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                val_accuracy=val_metrics.accuracy, val_precision=val_metrics.precision,
                val_recall=val_metrics.recall, val_f1=val_metrics.f1,
            )
        )

        improved = val_loss < history.best_val_loss
        if improved:
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_metrics_at_0.5": val_metrics.to_dict(),
                },
                checkpoint_dir / checkpoint_name,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:3d}/{config['training']['epochs']} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_f1@0.5={val_metrics.f1:.4f} val_recall@0.5={val_metrics.recall:.4f} "
            f"{'(best)' if improved else ''}"
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch} (no val_loss improvement for {patience} epochs).")
            history.stopped_early = True
            break

    history.training_seconds = time.time() - start_time
    (checkpoint_dir / f"{Path(checkpoint_name).stem}_history.json").write_text(json.dumps(history.to_dict(), indent=2))
    return history


def curve_data(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
    return {
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": roc_thresholds.tolist()},
        "pr_curve": {"precision": precision.tolist(), "recall": recall.tolist(), "thresholds": pr_thresholds.tolist()},
    }
