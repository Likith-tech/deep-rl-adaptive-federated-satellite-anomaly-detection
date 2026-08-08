"""Training loop for the temporal GRU + attention model.

Data preparation differs from the MLP baseline (src/training/baseline_trainer.py)
because sequences need ordered, non-shuffled rows:

    data/interim/kdd_train_cleaned.parquet (order-preserved, cleaned+labeled)
        -> split_ordered_train_validation (contiguous, NOT shuffled —
           different from Phase 1's split_train_validation used for the MLP)
        -> transform with Phase 1's ALREADY-FIT scaler/encoder
           (results/models/preprocessing/*.joblib — fit on Phase 1's
           training data only, reused here unchanged; no new fitting,
           so no new leakage is introduced)
        -> create_sequences per split (never crosses the split boundary)
        -> DataLoader -> TemporalGRUAttention -> BCEWithLogitsLoss
        -> Adam -> validation each epoch -> best checkpoint (by
           validation loss) -> early stopping

KDDTest+ is never touched here — only train/validation. Final test
evaluation happens separately, once, in scripts/evaluate_temporal.py.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import compute_metrics
from src.models.temporal_gru_attention import TemporalGRUAttention
from src.preprocessing.encoding import apply_categorical_encoder
from src.preprocessing.scaling import apply_scaler
from src.preprocessing.sequences import SequenceDataset, create_sequences, split_ordered_train_validation


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op if no GPU present


def prepare_temporal_train_validation(
    interim_train_parquet: Path,
    scaler_path: Path,
    encoder_path: Path,
    numerical_columns: list[str],
    categorical_columns: list[str],
    feature_columns: list[str],
    label_column: str,
    validation_size: float,
    sequence_length: int,
    stride: int | None,
    label_strategy: str = "any",
    label_threshold: float | None = None,
) -> tuple[SequenceDataset, SequenceDataset]:
    """Build order-preserving train/validation SequenceDatasets from the
    Phase 1 interim (cleaned, labeled, NOT yet split/encoded/scaled) data,
    reusing the already-fit Phase 1 scaler/encoder."""
    df = pd.read_parquet(interim_train_parquet)
    train_block, validation_block = split_ordered_train_validation(df, validation_size)

    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)

    def transform(block: pd.DataFrame) -> pd.DataFrame:
        block = apply_scaler(block, scaler, numerical_columns)
        block = apply_categorical_encoder(block, encoder, categorical_columns)
        return block

    train_transformed = transform(train_block)
    validation_transformed = transform(validation_block)

    train_sequences = create_sequences(
        train_transformed, feature_columns, label_column, sequence_length, stride, label_strategy, label_threshold
    )
    validation_sequences = create_sequences(
        validation_transformed, feature_columns, label_column, sequence_length, stride, label_strategy, label_threshold
    )
    return train_sequences, validation_sequences


def make_dataloader(dataset: SequenceDataset, batch_size: int, shuffle: bool) -> DataLoader:
    tensor_dataset = TensorDataset(
        torch.from_numpy(dataset.sequences.copy()),
        torch.from_numpy(dataset.labels.astype("float32").copy()),
    )
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)


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


def train_temporal(
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
            logits, _ = model(X_batch)
            logits = logits.squeeze(-1)
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
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_metrics": val_metrics.to_dict(),
                },
                checkpoint_dir / checkpoint_name,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"[seq_len={config['sequence_length']}] Epoch {epoch:3d}/{config['training']['epochs']} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_f1={val_metrics.f1:.4f} val_recall={val_metrics.recall:.4f} "
            f"{'(best)' if improved else ''}"
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch} (no val_loss improvement for {patience} epochs).")
            history.stopped_early = True
            break

    history.training_seconds = time.time() - start_time
    (checkpoint_dir / f"{Path(checkpoint_name).stem}_history.json").write_text(json.dumps(history.to_dict(), indent=2))
    return history
