"""Tests for Phase 5 local (per-satellite) training, using tiny synthetic
parquet fixtures — never the real Phase 4 partitions, so these run in a
couple of seconds and don't require the full data pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.local_evaluation import (
    build_client_summary,
    categories_present_absent,
    compute_aggregate_statistics,
    rank_clients_by_metric,
)
from src.evaluation.metrics import compute_metrics
from src.training.local_trainer import build_initial_state_dict, train_local_client

FEATURE_COLUMNS = [f"f{i}" for i in range(6)]


def _make_synthetic_parquet(path: Path, n_rows: int, seed: int, anomaly_bias: float = 0.0) -> None:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, len(FEATURE_COLUMNS)))
    y = (X[:, 0] + anomaly_bias + rng.normal(scale=0.5, size=n_rows) > 0).astype(int)
    df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    df["label_binary"] = y
    df["label_original"] = np.where(y == 1, "synthetic_attack", "normal")
    df["attack"] = df["label_original"]
    df["difficulty"] = 10
    df.to_parquet(path, index=False)


@pytest.fixture
def tiny_config() -> dict:
    return {
        "model": {
            "input_dim": len(FEATURE_COLUMNS),
            "hidden_dimensions": [8, 4],
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
        "initialization": {"shared_init_seed": 42},
    }


@pytest.fixture
def two_clients_and_global_val(tmp_path: Path) -> dict:
    """Two distinctly-seeded 'client' partitions plus one shared global
    validation set, mirroring the real SAT-XX / validation.parquet split."""
    sat_a = tmp_path / "SAT-A" / "train.parquet"
    sat_b = tmp_path / "SAT-B" / "train.parquet"
    global_val = tmp_path / "validation.parquet"
    sat_a.parent.mkdir(parents=True)
    sat_b.parent.mkdir(parents=True)
    _make_synthetic_parquet(sat_a, n_rows=150, seed=1)
    _make_synthetic_parquet(sat_b, n_rows=90, seed=2, anomaly_bias=1.0)
    _make_synthetic_parquet(global_val, n_rows=80, seed=3)
    return {"SAT-A": sat_a, "SAT-B": sat_b, "global_val": global_val}


# --- Local dataset loading -------------------------------------------------


def test_client_dataset_has_expected_feature_count_and_valid_labels(two_clients_and_global_val):
    df = pd.read_parquet(two_clients_and_global_val["SAT-A"])
    for col in FEATURE_COLUMNS:
        assert col in df.columns
    assert set(df["label_binary"].unique()).issubset({0, 1})


# --- Client isolation --------------------------------------------------


def test_client_isolation_sat_a_and_sat_b_are_disjoint_and_distinct(two_clients_and_global_val):
    df_a = pd.read_parquet(two_clients_and_global_val["SAT-A"])
    df_b = pd.read_parquet(two_clients_and_global_val["SAT-B"])
    # Different sizes/seeds -> not accidentally the same file/content.
    assert len(df_a) != len(df_b)
    assert not df_a.equals(df_b)


def test_train_local_client_only_reads_its_own_partition(tmp_path, two_clients_and_global_val, tiny_config):
    """Training SAT-A must not be affected by SAT-B's data existing on disk."""
    initial_state = build_initial_state_dict(tiny_config["model"], tiny_config["initialization"]["shared_init_seed"])

    result_a = train_local_client(
        client_id="SAT-A",
        client_train_parquet=two_clients_and_global_val["SAT-A"],
        global_validation_parquet=two_clients_and_global_val["global_val"],
        feature_columns=FEATURE_COLUMNS,
        config=tiny_config,
        initial_state_dict=initial_state,
        checkpoint_dir=tmp_path / "ckpt_a",
        training_seed=tiny_config["training"]["seed"],
    )
    assert result_a.train_samples == 150  # SAT-A's own row count, not SAT-B's (90)


# --- Model initialization ------------------------------------------------


def test_all_clients_start_from_identical_initial_weights(tiny_config):
    state_1 = build_initial_state_dict(tiny_config["model"], seed=42)
    state_2 = build_initial_state_dict(tiny_config["model"], seed=42)
    for key in state_1:
        assert torch.allclose(state_1[key], state_2[key])


def test_initialization_is_reproducible_across_calls(tiny_config):
    state_a = build_initial_state_dict(tiny_config["model"], seed=7)
    state_b = build_initial_state_dict(tiny_config["model"], seed=7)
    state_c = build_initial_state_dict(tiny_config["model"], seed=8)
    for key in state_a:
        assert torch.equal(state_a[key], state_b[key])
    # Different seed should (almost certainly) produce different weights.
    any_diff = any(not torch.equal(state_a[key], state_c[key]) for key in state_a)
    assert any_diff


# --- Training --------------------------------------------------------------


def test_training_one_client_reduces_loss_and_saves_checkpoint(tmp_path, two_clients_and_global_val, tiny_config):
    initial_state = build_initial_state_dict(tiny_config["model"], tiny_config["initialization"]["shared_init_seed"])
    checkpoint_dir = tmp_path / "checkpoints" / "SAT-A"

    result = train_local_client(
        client_id="SAT-A",
        client_train_parquet=two_clients_and_global_val["SAT-A"],
        global_validation_parquet=two_clients_and_global_val["global_val"],
        feature_columns=FEATURE_COLUMNS,
        config=tiny_config,
        initial_state_dict=initial_state,
        checkpoint_dir=checkpoint_dir,
        training_seed=42,
    )

    assert len(result.epochs) == tiny_config["training"]["epochs"]
    assert result.epochs[-1].train_loss < result.epochs[0].train_loss
    assert (checkpoint_dir / "best_model.pt").exists()
    assert (checkpoint_dir / "training_history.json").exists()
    assert result.local_train_metrics is not None
    assert result.global_val_metrics is not None


def test_training_is_reproducible_with_fixed_seed(tmp_path, two_clients_and_global_val, tiny_config):
    initial_state = build_initial_state_dict(tiny_config["model"], tiny_config["initialization"]["shared_init_seed"])

    result_a = train_local_client(
        "SAT-A", two_clients_and_global_val["SAT-A"], two_clients_and_global_val["global_val"],
        FEATURE_COLUMNS, tiny_config, initial_state, tmp_path / "run_a", training_seed=42,
    )
    result_b = train_local_client(
        "SAT-A", two_clients_and_global_val["SAT-A"], two_clients_and_global_val["global_val"],
        FEATURE_COLUMNS, tiny_config, initial_state, tmp_path / "run_b", training_seed=42,
    )

    losses_a = [round(e.train_loss, 6) for e in result_a.epochs]
    losses_b = [round(e.train_loss, 6) for e in result_b.epochs]
    assert losses_a == losses_b


# --- Evaluation --------------------------------------------------------------


def test_roc_auc_not_available_for_single_class_predictions():
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3, 0.1])
    metrics = compute_metrics(y_true, y_pred, y_prob)
    assert metrics.roc_auc is None


def test_roc_auc_available_when_both_classes_present():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])
    metrics = compute_metrics(y_true, y_pred, y_prob)
    assert metrics.roc_auc is not None


# --- Cross-client evaluation helpers ----------------------------------------


def test_categories_present_absent_splits_correctly():
    counts = {"normal": 10, "dos": 5, "probe": 0, "r2l": 0, "u2r": 2}
    present, absent = categories_present_absent(counts)
    assert set(present) == {"normal", "dos", "u2r"}
    assert set(absent) == {"probe", "r2l"}


def test_build_client_summary_and_aggregate_statistics():
    client_stats = {
        "total_samples": 100,
        "anomaly_percentage": 40.0,
        "category_counts": {"normal": 60, "dos": 40, "probe": 0, "r2l": 0, "u2r": 0},
    }
    local_metrics = {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1": 0.9,
                      "roc_auc": 0.9, "false_positive_rate": 0.1}
    val_metrics_a = {"accuracy": 0.8, "precision": 0.8, "recall": 0.8, "f1": 0.8,
                      "roc_auc": 0.8, "false_positive_rate": 0.2}
    val_metrics_b = {"accuracy": 0.6, "precision": 0.6, "recall": 0.6, "f1": 0.6,
                      "roc_auc": None, "false_positive_rate": 0.4}

    summary_a = build_client_summary("SAT-A", client_stats, local_metrics, val_metrics_a)
    summary_b = build_client_summary("SAT-B", client_stats, local_metrics, val_metrics_b)

    assert summary_a.categories_absent == ["probe", "r2l", "u2r"]

    aggregates = compute_aggregate_statistics([summary_a, summary_b])
    assert aggregates["f1"]["mean"] == pytest.approx(0.7)
    assert aggregates["f1"]["min"] == pytest.approx(0.6)
    assert aggregates["f1"]["max"] == pytest.approx(0.8)
    # roc_auc: SAT-B's is None, so aggregate should only reflect SAT-A's 0.8.
    assert aggregates["roc_auc"]["mean"] == pytest.approx(0.8)

    ranked = rank_clients_by_metric([summary_a, summary_b], metric="f1")
    assert ranked[0].client_id == "SAT-A"
    assert ranked[-1].client_id == "SAT-B"


def test_aggregate_statistics_all_none_when_no_roc_auc_available():
    client_stats = {
        "total_samples": 10, "anomaly_percentage": 0.0,
        "category_counts": {"normal": 10, "dos": 0, "probe": 0, "r2l": 0, "u2r": 0},
    }
    metrics_no_roc = {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0,
                       "roc_auc": None, "false_positive_rate": 0.0}
    summary = build_client_summary("SAT-X", client_stats, metrics_no_roc, metrics_no_roc)
    aggregates = compute_aggregate_statistics([summary])
    assert aggregates["roc_auc"] is None
