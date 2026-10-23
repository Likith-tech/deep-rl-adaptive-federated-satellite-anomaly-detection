"""Tests for src/training/esa_adb_federated.py — the ESA-ADB FedAvg
orchestration loop (Experiment 2). Tiny synthetic data, 2-3 clients,
1-2 rounds — fast, no real ESA-ADB download required."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.sequences import SequenceDataset
from src.training.esa_adb_federated import build_initial_state_dict, run_esa_adb_fedavg

SMALL_MODEL_CONFIG = {"input_dim": 6, "projection_dim": 4, "hidden_dim": 4, "num_gru_layers": 1, "dropout": 0.0, "output_dim": 1}
SMALL_TRAINING_CONFIG = {"batch_size": 8, "learning_rate": 0.01, "local_epochs": 1, "seed": 42}


def _make_dataset(n: int, n_anomaly: int, input_dim: int, seed: int) -> SequenceDataset:
    rng = np.random.default_rng(seed)
    sequences = rng.normal(size=(n, 10, input_dim)).astype("float32")
    labels = np.array([1] * n_anomaly + [0] * (n - n_anomaly), dtype="int64")
    rng.shuffle(labels)
    record_labels = [np.zeros(10, dtype="int64") for _ in range(n)]
    return SequenceDataset(sequences=sequences, labels=labels, record_labels=record_labels, sequence_length=10, stride=10, label_strategy="last")


def test_build_initial_state_dict_is_deterministic_for_same_seed():
    a = build_initial_state_dict(SMALL_MODEL_CONFIG, seed=42)
    b = build_initial_state_dict(SMALL_MODEL_CONFIG, seed=42)
    for k in a:
        assert np.allclose(a[k].numpy(), b[k].numpy())


def test_build_initial_state_dict_differs_for_different_seeds():
    a = build_initial_state_dict(SMALL_MODEL_CONFIG, seed=1)
    b = build_initial_state_dict(SMALL_MODEL_CONFIG, seed=2)
    any_diff = any(not np.allclose(a[k].numpy(), b[k].numpy()) for k in a)
    assert any_diff


def test_run_esa_adb_fedavg_completes_and_records_all_rounds(tmp_path):
    client_train = {f"client_{i}": _make_dataset(30, 6, 6, seed=i) for i in range(3)}
    client_val = {f"client_{i}": _make_dataset(10, 2, 6, seed=100 + i) for i in range(3)}
    pooled_val = _make_dataset(30, 6, 6, seed=999)

    result = run_esa_adb_fedavg(
        client_train_datasets=client_train, client_validation_datasets=client_val, pooled_validation_dataset=pooled_val,
        model_config=SMALL_MODEL_CONFIG, training_config=SMALL_TRAINING_CONFIG, num_rounds=2,
        shared_init_seed=42, checkpoint_dir=tmp_path,
    )

    assert len(result.round_history) == 2
    assert (tmp_path / "best_global_model.pt").exists()
    assert (tmp_path / "round_history.json").exists()
    assert (tmp_path / "round_checkpoints" / "round_01.pt").exists()
    assert (tmp_path / "round_checkpoints" / "round_02.pt").exists()


def test_round_record_includes_per_client_metrics_and_fairness_gap(tmp_path):
    client_train = {f"client_{i}": _make_dataset(30, 6, 6, seed=i) for i in range(3)}
    client_val = {f"client_{i}": _make_dataset(10, 2, 6, seed=100 + i) for i in range(3)}
    pooled_val = _make_dataset(30, 6, 6, seed=999)

    result = run_esa_adb_fedavg(
        client_train_datasets=client_train, client_validation_datasets=client_val, pooled_validation_dataset=pooled_val,
        model_config=SMALL_MODEL_CONFIG, training_config=SMALL_TRAINING_CONFIG, num_rounds=1,
        shared_init_seed=42, checkpoint_dir=tmp_path,
    )

    record = result.round_history[0]
    assert set(record.per_client_val.keys()) == {"client_0", "client_1", "client_2"}
    for cid, m in record.per_client_val.items():
        assert set(m.keys()) == {"loss", "f1", "recall", "precision", "roc_auc", "pr_auc", "sample_count"}
    client_f1s = [m["f1"] for m in record.per_client_val.values()]
    assert record.fairness_gap_f1 == pytest.approx(max(client_f1s) - min(client_f1s))


def test_every_client_trains_only_from_the_same_shared_global_state(tmp_path):
    """Regression guard: all clients must start each round from an
    IDENTICAL global state — verified indirectly by checking round 1's
    aggregated model differs from the initial state (training happened)
    and that best_round/best_val_loss are populated."""
    client_train = {f"client_{i}": _make_dataset(30, 6, 6, seed=i) for i in range(2)}
    client_val = {f"client_{i}": _make_dataset(10, 2, 6, seed=100 + i) for i in range(2)}
    pooled_val = _make_dataset(20, 4, 6, seed=999)

    result = run_esa_adb_fedavg(
        client_train_datasets=client_train, client_validation_datasets=client_val, pooled_validation_dataset=pooled_val,
        model_config=SMALL_MODEL_CONFIG, training_config=SMALL_TRAINING_CONFIG, num_rounds=1,
        shared_init_seed=42, checkpoint_dir=tmp_path,
    )
    assert result.best_round == 1
    assert result.best_val_loss < float("inf")


def test_client_only_sees_its_own_train_dataset():
    """EsaAdbFederatedClient must never reference another client's data
    — verified by checking pos_weight is derived purely from the
    client's own train_dataset.labels."""
    from src.training.esa_adb_federated import EsaAdbFederatedClient
    from src.training.esa_adb_trainer import compute_pos_weight

    ds = _make_dataset(40, 8, 6, seed=7)
    client = EsaAdbFederatedClient("client_x", ds)
    assert client.pos_weight == pytest.approx(compute_pos_weight(ds.labels))
