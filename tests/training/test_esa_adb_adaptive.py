"""Tests for src/training/esa_adb_adaptive.py — ESA-ADB rule-based
adaptive FL (Step B). Tiny synthetic data, mirrors
tests/federated/test_esa_adb_federated.py's fixtures/style exactly so
the two aggregation strategies are tested under identical conditions."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.sequences import SequenceDataset
from src.satellite.esa_adb.clients import CLIENT_CHANNELS
from src.training.esa_adb_adaptive import run_esa_adb_adaptive_fl

REAL_CLIENT_IDS = list(CLIENT_CHANNELS.keys())  # client_1..client_8 — build_esa_adb_resource_signal requires real IDs

SMALL_MODEL_CONFIG = {"input_dim": 6, "projection_dim": 4, "hidden_dim": 4, "num_gru_layers": 1, "dropout": 0.0, "output_dim": 1}
SMALL_TRAINING_CONFIG = {"batch_size": 8, "learning_rate": 0.01, "local_epochs": 1, "seed": 42}
RULE_WEIGHTS = {"performance": 0.40, "data": 0.25, "resource": 0.15, "fairness": 0.20}


def _make_dataset(n: int, n_anomaly: int, input_dim: int, seed: int) -> SequenceDataset:
    rng = np.random.default_rng(seed)
    sequences = rng.normal(size=(n, 10, input_dim)).astype("float32")
    labels = np.array([1] * n_anomaly + [0] * (n - n_anomaly), dtype="int64")
    rng.shuffle(labels)
    record_labels = [np.zeros(10, dtype="int64") for _ in range(n)]
    return SequenceDataset(sequences=sequences, labels=labels, record_labels=record_labels, sequence_length=10, stride=10, label_strategy="last")


def test_run_esa_adb_adaptive_fl_completes_and_records_all_rounds(tmp_path):
    client_train = {cid: _make_dataset(30, 6, 6, seed=i) for i, cid in enumerate(REAL_CLIENT_IDS[:3])}
    client_val = {cid: _make_dataset(10, 2, 6, seed=100 + i) for i, cid in enumerate(REAL_CLIENT_IDS[:3])}
    pooled_val = _make_dataset(30, 6, 6, seed=999)

    result = run_esa_adb_adaptive_fl(
        client_train_datasets=client_train, client_validation_datasets=client_val, pooled_validation_dataset=pooled_val,
        model_config=SMALL_MODEL_CONFIG, training_config=SMALL_TRAINING_CONFIG, num_rounds=2,
        shared_init_seed=42, checkpoint_dir=tmp_path, rule_weights=RULE_WEIGHTS,
    )

    assert len(result.round_history) == 2
    assert (tmp_path / "best_global_model.pt").exists()
    assert (tmp_path / "round_history.json").exists()
    assert (tmp_path / "round_checkpoints" / "round_01.pt").exists()
    assert (tmp_path / "round_checkpoints" / "round_02.pt").exists()


def test_round_record_includes_scores_weights_and_fairness_gap(tmp_path):
    client_train = {cid: _make_dataset(30, 6, 6, seed=i) for i, cid in enumerate(REAL_CLIENT_IDS[:3])}
    client_val = {cid: _make_dataset(10, 2, 6, seed=100 + i) for i, cid in enumerate(REAL_CLIENT_IDS[:3])}
    pooled_val = _make_dataset(30, 6, 6, seed=999)

    result = run_esa_adb_adaptive_fl(
        client_train_datasets=client_train, client_validation_datasets=client_val, pooled_validation_dataset=pooled_val,
        model_config=SMALL_MODEL_CONFIG, training_config=SMALL_TRAINING_CONFIG, num_rounds=1,
        shared_init_seed=42, checkpoint_dir=tmp_path, rule_weights=RULE_WEIGHTS,
    )

    record = result.round_history[0]
    assert set(record.client_scores.keys()) == set(REAL_CLIENT_IDS[:3])
    assert set(record.aggregation_weights.keys()) == set(REAL_CLIENT_IDS[:3])
    assert pytest.approx(sum(record.aggregation_weights.values()), abs=1e-6) == 1.0
    client_f1s = [m["f1"] for m in record.per_client_val.values()]
    assert record.fairness_gap_f1 == pytest.approx(max(client_f1s) - min(client_f1s))


def test_aggregation_weights_differ_from_uniform_when_clients_are_heterogeneous(tmp_path):
    """Sanity check that the rule actually produces non-uniform weights
    (as opposed to accidentally degenerating to plain FedAvg) when
    client sample counts are very different."""
    client_train = {
        REAL_CLIENT_IDS[0]: _make_dataset(100, 20, 6, seed=0),
        REAL_CLIENT_IDS[1]: _make_dataset(10, 2, 6, seed=1),
    }
    client_val = {cid: _make_dataset(10, 2, 6, seed=200 + i) for i, cid in enumerate(client_train)}
    pooled_val = _make_dataset(20, 4, 6, seed=999)

    result = run_esa_adb_adaptive_fl(
        client_train_datasets=client_train, client_validation_datasets=client_val, pooled_validation_dataset=pooled_val,
        model_config=SMALL_MODEL_CONFIG, training_config=SMALL_TRAINING_CONFIG, num_rounds=1,
        shared_init_seed=42, checkpoint_dir=tmp_path, rule_weights=RULE_WEIGHTS,
    )
    weights = result.round_history[0].aggregation_weights
    assert weights[REAL_CLIENT_IDS[0]] != pytest.approx(weights[REAL_CLIENT_IDS[1]])
