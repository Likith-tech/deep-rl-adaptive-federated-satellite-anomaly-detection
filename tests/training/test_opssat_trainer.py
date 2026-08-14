"""Tests for the OPSSAT-AD training loop, using tiny synthetic feature
matrices — never the real downloaded dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.training.opssat_trainer import set_seed, train_opssat

FEATURE_DIM = 10


def _make_synthetic_features(n_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, FEATURE_DIM)).astype("float32")
    y = (X[:, 0] + rng.normal(scale=0.5, size=n_rows) > 0).astype("float32")
    return X, y


@pytest.fixture
def synthetic_train_val() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, y_train = _make_synthetic_features(150, seed=1)
    X_val, y_val = _make_synthetic_features(50, seed=2)
    return X_train, y_train, X_val, y_val


@pytest.fixture
def tiny_config() -> dict:
    return {
        "model": {
            "input_dim": FEATURE_DIM,
            "projection_dim": 8,
            "hidden_dim": 8,
            "num_gru_layers": 1,
            "dropout": 0.1,
            "output_dim": 1,
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "bce_with_logits",
            "epochs": 3,
            "early_stopping_patience": 10,
            "seed": 42,
        },
    }


def test_training_reduces_loss_over_epochs(tmp_path: Path, synthetic_train_val, tiny_config):
    X_train, y_train, X_val, y_val = synthetic_train_val
    history = train_opssat(X_train, y_train, X_val, y_val, tiny_config, tmp_path / "checkpoints")

    assert len(history.epochs) == tiny_config["training"]["epochs"]
    assert history.epochs[-1].train_loss < history.epochs[0].train_loss


def test_training_saves_checkpoint_and_history(tmp_path: Path, synthetic_train_val, tiny_config):
    X_train, y_train, X_val, y_val = synthetic_train_val
    checkpoint_dir = tmp_path / "checkpoints"
    train_opssat(X_train, y_train, X_val, y_val, tiny_config, checkpoint_dir)

    assert (checkpoint_dir / "best_model.pt").exists()
    assert (checkpoint_dir / "training_history.json").exists()


def test_training_is_reproducible_with_fixed_seed(tmp_path: Path, synthetic_train_val, tiny_config):
    X_train, y_train, X_val, y_val = synthetic_train_val

    history_a = train_opssat(X_train, y_train, X_val, y_val, tiny_config, tmp_path / "run_a")
    history_b = train_opssat(X_train, y_train, X_val, y_val, tiny_config, tmp_path / "run_b")

    losses_a = [round(e.train_loss, 6) for e in history_a.epochs]
    losses_b = [round(e.train_loss, 6) for e in history_b.epochs]
    assert losses_a == losses_b


def test_set_seed_is_callable_without_error():
    set_seed(42)
