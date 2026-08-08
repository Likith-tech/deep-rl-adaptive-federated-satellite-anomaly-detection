"""Tests for the temporal training loop, using tiny synthetic sequence
data — never the real NSL-KDD dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.encoding import fit_categorical_encoder
from src.preprocessing.scaling import fit_scaler
from src.preprocessing.sequences import SequenceDataset, create_sequences
from src.training.temporal_trainer import prepare_temporal_train_validation, set_seed, train_temporal

FEATURE_COLUMNS = [f"f{i}" for i in range(6)]


def _make_synthetic_sequence_dataset(n_sequences: int, sequence_length: int, n_features: int, seed: int) -> SequenceDataset:
    rng = np.random.default_rng(seed)
    sequences = rng.normal(size=(n_sequences, sequence_length, n_features)).astype("float32")
    # Label depends on mean of feature 0 across the sequence, so there's a learnable signal.
    labels = (sequences[:, :, 0].mean(axis=1) > 0).astype("int64")
    record_labels = [np.zeros(sequence_length, dtype="int64") for _ in range(n_sequences)]
    return SequenceDataset(sequences, labels, record_labels, sequence_length, sequence_length)


@pytest.fixture
def tiny_config() -> dict:
    return {
        "input_dim": 6,
        "projection_dim": 8,
        "hidden_dim": 8,
        "num_gru_layers": 1,
        "dropout": 0.1,
        "output_dim": 1,
        "sequence_length": 5,
        "training": {
            "batch_size": 16,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "bce_with_logits",
            "epochs": 3,
            "early_stopping_patience": 10,
            "seed": 42,
        },
    }


def test_training_step_reduces_loss_over_epochs(tmp_path, tiny_config):
    train_ds = _make_synthetic_sequence_dataset(100, 5, 6, seed=1)
    val_ds = _make_synthetic_sequence_dataset(30, 5, 6, seed=2)

    history = train_temporal(train_ds, val_ds, tiny_config, tmp_path)

    assert len(history.epochs) == tiny_config["training"]["epochs"]
    assert history.epochs[-1].train_loss < history.epochs[0].train_loss


def test_training_saves_best_checkpoint(tmp_path, tiny_config):
    train_ds = _make_synthetic_sequence_dataset(100, 5, 6, seed=1)
    val_ds = _make_synthetic_sequence_dataset(30, 5, 6, seed=2)

    train_temporal(train_ds, val_ds, tiny_config, tmp_path)

    assert (tmp_path / "best_model.pt").exists()
    assert (tmp_path / "best_model_history.json").exists()


def test_training_is_reproducible_with_fixed_seed(tmp_path, tiny_config):
    train_ds = _make_synthetic_sequence_dataset(100, 5, 6, seed=1)
    val_ds = _make_synthetic_sequence_dataset(30, 5, 6, seed=2)

    history_a = train_temporal(train_ds, val_ds, tiny_config, tmp_path / "a")
    history_b = train_temporal(train_ds, val_ds, tiny_config, tmp_path / "b")

    losses_a = [round(e.train_loss, 6) for e in history_a.epochs]
    losses_b = [round(e.train_loss, 6) for e in history_b.epochs]
    assert losses_a == losses_b


def test_set_seed_is_callable_without_error():
    set_seed(42)


# --- prepare_temporal_train_validation integration test (small synthetic files) ---

CATEGORICAL_COLUMNS = ["proto"]
NUMERICAL_COLUMNS = [f"num{i}" for i in range(3)]
ALL_FEATURE_COLUMNS = NUMERICAL_COLUMNS  # + one-hot columns added after encoding


def _make_interim_like_df(n_rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({col: rng.normal(size=n_rows) for col in NUMERICAL_COLUMNS})
    df["proto"] = rng.choice(["tcp", "udp"], size=n_rows)
    df["label_binary"] = rng.integers(0, 2, size=n_rows)
    return df


def test_prepare_temporal_train_validation_builds_sequences_without_leakage(tmp_path: Path):
    df = _make_interim_like_df(200)
    interim_path = tmp_path / "kdd_train_cleaned.parquet"
    df.to_parquet(interim_path, index=False)

    # Fit a tiny scaler/encoder on the *whole* synthetic df here purely as
    # fixture setup (mirrors reusing Phase 1's already-fit artifacts) —
    # not testing Phase 1's own fit-on-train-only guarantee, which is
    # covered by tests/data/test_encoding_scaling.py.
    scaler = fit_scaler(df, NUMERICAL_COLUMNS)
    encoder = fit_categorical_encoder(df, CATEGORICAL_COLUMNS)
    import joblib
    scaler_path = tmp_path / "scaler.joblib"
    encoder_path = tmp_path / "encoder.joblib"
    joblib.dump(scaler, scaler_path)
    joblib.dump(encoder, encoder_path)

    encoded_feature_columns = NUMERICAL_COLUMNS + list(encoder.get_feature_names_out(CATEGORICAL_COLUMNS))

    train_seqs, val_seqs = prepare_temporal_train_validation(
        interim_train_parquet=interim_path,
        scaler_path=scaler_path,
        encoder_path=encoder_path,
        numerical_columns=NUMERICAL_COLUMNS,
        categorical_columns=CATEGORICAL_COLUMNS,
        feature_columns=encoded_feature_columns,
        label_column="label_binary",
        validation_size=0.2,
        sequence_length=5,
        stride=None,
    )

    assert train_seqs.num_sequences > 0
    assert val_seqs.num_sequences > 0
    assert train_seqs.sequences.shape[1:] == (5, len(encoded_feature_columns))
    # 200 rows, 20% validation -> 160 train / 40 validation (order-preserving split)
    # sequences of length 5, non-overlapping -> 32 train seqs, 8 val seqs
    assert train_seqs.num_sequences == 32
    assert val_seqs.num_sequences == 8
