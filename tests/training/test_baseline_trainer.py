"""Tests for the baseline training loop, using a tiny synthetic parquet
dataset written to a temp dir — never the real ~22MB NSL-KDD data, so
these tests run in a couple seconds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.training.baseline_trainer import set_seed, train_baseline

FEATURE_COLUMNS = [f"f{i}" for i in range(6)]


def _make_synthetic_parquet(path: Path, n_rows: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, len(FEATURE_COLUMNS)))
    # Make label weakly separable from f0 so the model has something to learn.
    y = (X[:, 0] + rng.normal(scale=0.5, size=n_rows) > 0).astype(int)
    df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    df["label_binary"] = y
    df["label_original"] = np.where(y == 1, "synthetic_attack", "normal")
    df["attack"] = df["label_original"]
    df["difficulty"] = 10
    df.to_parquet(path, index=False)


@pytest.fixture
def synthetic_train_val(tmp_path: Path) -> tuple[Path, Path]:
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "validation.parquet"
    _make_synthetic_parquet(train_path, n_rows=200, seed=1)
    _make_synthetic_parquet(val_path, n_rows=60, seed=2)
    return train_path, val_path


@pytest.fixture
def tiny_config() -> dict:
    return {
        "input_dim": len(FEATURE_COLUMNS),
        "hidden_dimensions": [8, 4],
        "dropout": 0.1,
        "output_dim": 1,
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


def test_training_step_reduces_loss_over_epochs(tmp_path, synthetic_train_val, tiny_config):
    train_path, val_path = synthetic_train_val
    checkpoint_dir = tmp_path / "checkpoints"

    history = train_baseline(train_path, val_path, FEATURE_COLUMNS, tiny_config, checkpoint_dir)

    assert len(history.epochs) == tiny_config["training"]["epochs"]
    # Loss should generally trend downward on this easy separable synthetic task.
    assert history.epochs[-1].train_loss < history.epochs[0].train_loss


def test_training_saves_best_checkpoint(tmp_path, synthetic_train_val, tiny_config):
    train_path, val_path = synthetic_train_val
    checkpoint_dir = tmp_path / "checkpoints"

    train_baseline(train_path, val_path, FEATURE_COLUMNS, tiny_config, checkpoint_dir)

    assert (checkpoint_dir / "best_model.pt").exists()
    assert (checkpoint_dir / "training_history.json").exists()


def test_training_is_reproducible_with_fixed_seed(tmp_path, synthetic_train_val, tiny_config):
    train_path, val_path = synthetic_train_val

    history_a = train_baseline(train_path, val_path, FEATURE_COLUMNS, tiny_config, tmp_path / "run_a")
    history_b = train_baseline(train_path, val_path, FEATURE_COLUMNS, tiny_config, tmp_path / "run_b")

    losses_a = [round(e.train_loss, 6) for e in history_a.epochs]
    losses_b = [round(e.train_loss, 6) for e in history_b.epochs]
    assert losses_a == losses_b


def test_set_seed_is_callable_without_error():
    set_seed(42)
