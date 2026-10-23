"""Phase 5 — local (per-satellite) training loop.

    One satellite's local partition -> DataLoader -> MLP (same
        architecture as the Phase 2 baseline, but a fresh copy per
        client) -> BCEWithLogitsLoss -> Adam -> evaluated each epoch
        against the GLOBAL validation set -> best checkpoint (by
        global validation loss) -> early stopping.

Every client starts from the SAME initial weights (see
`build_initial_state_dict`) so that differences between clients'
resulting local models can be attributed to their local data, not to
different random initialization. There is no communication or
aggregation between clients here — each call to `train_local_client`
is fully independent.

KDDTest+ is never touched by this module.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from src.evaluation.metrics import ClassificationMetrics
from src.models.baseline_mlp import BaselineMLP
from src.training.baseline_trainer import (
    load_features_and_labels,
    make_dataloader,
    run_validation,
    set_seed,
)


def build_initial_state_dict(config: dict, seed: int) -> dict:
    """Build one randomly-initialized model and return its state_dict.

    Every client clones this same state_dict as its starting point, so
    that model initialization is not a confounding factor in local
    training differences.
    """
    set_seed(seed)
    model = BaselineMLP(
        input_dim=config["input_dim"],
        hidden_dimensions=config["hidden_dimensions"],
        dropout=config["dropout"],
        output_dim=config["output_dim"],
    )
    return {k: v.clone() for k, v in model.state_dict().items()}


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
class LocalTrainingResult:
    client_id: str
    epochs: list[EpochRecord] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    stopped_early: bool = False
    training_seconds: float = 0.0
    train_samples: int = 0
    local_train_metrics: ClassificationMetrics | None = None
    global_val_metrics: ClassificationMetrics | None = None

    def to_dict(self) -> dict:
        return {
            "client_id": self.client_id,
            "epochs": [vars(e) for e in self.epochs],
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "stopped_early": self.stopped_early,
            "training_seconds": self.training_seconds,
            "train_samples": self.train_samples,
            "local_train_metrics": self.local_train_metrics.to_dict() if self.local_train_metrics else None,
            "global_val_metrics": self.global_val_metrics.to_dict() if self.global_val_metrics else None,
        }


def train_local_client(
    client_id: str,
    client_train_parquet: Path,
    global_validation_parquet: Path,
    feature_columns: list[str],
    config: dict,
    initial_state_dict: dict,
    checkpoint_dir: Path,
    training_seed: int,
) -> LocalTrainingResult:
    """Train one satellite client's local model.

    `client_train_parquet` must be that client's own partition only
    (data/partitions/<client_id>/train.parquet) — this function does
    not know about, and never loads, any other client's data.
    """
    set_seed(training_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train = load_features_and_labels(client_train_parquet, feature_columns)
    X_val, y_val = load_features_and_labels(global_validation_parquet, feature_columns)

    train_loader = make_dataloader(X_train, y_train, config["training"]["batch_size"], shuffle=True)
    val_loader = make_dataloader(X_val, y_val, config["training"]["batch_size"], shuffle=False)
    # Non-shuffled loader over the client's own training data, used only
    # for reporting local-training metrics (not for gradient updates).
    train_eval_loader = make_dataloader(X_train, y_train, config["training"]["batch_size"], shuffle=False)

    model = BaselineMLP(
        input_dim=config["model"]["input_dim"],
        hidden_dimensions=config["model"]["hidden_dimensions"],
        dropout=config["model"]["dropout"],
        output_dim=config["model"]["output_dim"],
    ).to(device)
    model.load_state_dict({k: v.clone() for k, v in initial_state_dict.items()})

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result = LocalTrainingResult(client_id=client_id, train_samples=len(y_train))
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

        result.epochs.append(
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

        improved = val_loss < result.best_val_loss
        if improved:
            result.best_val_loss = val_loss
            result.best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "client_id": client_id,
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

        if epochs_without_improvement >= patience:
            result.stopped_early = True
            break

    result.training_seconds = time.time() - start_time

    # Reload the best checkpoint (not necessarily the last epoch) before
    # computing the final reported local-train / global-val metrics.
    best_checkpoint = torch.load(checkpoint_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    _, local_train_metrics = run_validation(model, train_eval_loader, criterion, device)
    _, global_val_metrics = run_validation(model, val_loader, criterion, device)
    result.local_train_metrics = local_train_metrics
    result.global_val_metrics = global_val_metrics

    (checkpoint_dir / "training_history.json").write_text(json.dumps(result.to_dict(), indent=2))
    return result
