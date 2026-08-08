"""Training loop for the baseline MLP.

    Processed training data -> DataLoader -> MLP -> BCEWithLogitsLoss
        -> Adam -> validation each epoch -> best checkpoint (by
        validation loss) -> early stopping if no improvement for
        `early_stopping_patience` epochs.

KDDTest+ (the held-out test set) is never touched here — only train/
validation. Final test evaluation happens separately, once, in
scripts/evaluate_baseline.py.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import compute_metrics
from src.models.baseline_mlp import BaselineMLP

NON_FEATURE_COLUMNS = ["attack", "difficulty", "label_original", "label_binary"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op if no GPU present


def load_features_and_labels(parquet_path: Path, feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(parquet_path)
    X = df[feature_columns].to_numpy(dtype="float32").copy()
    y = df["label_binary"].to_numpy(dtype="float32").copy()
    return X, y


def make_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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


def run_validation(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch).squeeze(-1)
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


def train_baseline(
    train_parquet: Path,
    validation_parquet: Path,
    feature_columns: list[str],
    config: dict,
    checkpoint_dir: Path,
) -> TrainingHistory:
    seed = config["training"]["seed"]
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train = load_features_and_labels(train_parquet, feature_columns)
    X_val, y_val = load_features_and_labels(validation_parquet, feature_columns)

    train_loader = make_dataloader(X_train, y_train, config["training"]["batch_size"], shuffle=True)
    val_loader = make_dataloader(X_val, y_val, config["training"]["batch_size"], shuffle=False)

    model = BaselineMLP(
        input_dim=config["input_dim"],
        hidden_dimensions=config["hidden_dimensions"],
        dropout=config["dropout"],
        output_dim=config["output_dim"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
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
            logits = model(X_batch).squeeze(-1)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(y_batch)
            n_train += len(y_batch)

        train_loss = total_train_loss / n_train
        val_loss, val_metrics = run_validation(model, val_loader, criterion, device)

        history.epochs.append(
            EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_accuracy=val_metrics.accuracy,
                val_precision=val_metrics.precision,
                val_recall=val_metrics.recall,
                val_f1=val_metrics.f1,
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
                    "feature_columns": feature_columns,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_metrics": val_metrics.to_dict(),
                },
                checkpoint_dir / "best_model.pt",
            )
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:3d}/{config['training']['epochs']} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_f1={val_metrics.f1:.4f} val_recall={val_metrics.recall:.4f} "
            f"{'(best)' if improved else ''}"
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch} (no val_loss improvement for {patience} epochs).")
            history.stopped_early = True
            break

    history.training_seconds = time.time() - start_time
    (checkpoint_dir / "training_history.json").write_text(json.dumps(history.to_dict(), indent=2))
    return history
