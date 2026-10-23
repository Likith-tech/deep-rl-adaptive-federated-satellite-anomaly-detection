"""Tests for src/federated/esa_adb_drl_round_provider.py — the real
ESA-ADB DQN round provider (Step C). Tiny synthetic data, exercises the
full DRLFederatedEnvironment integration end-to-end without downloading
real ESA-ADB data or running real training-scale rounds."""

from __future__ import annotations

import numpy as np
import pytest

from src.drl.action_space import NUM_ACTIONS, action_to_aggregation_weights
from src.federated.drl_environment import DRLFederatedEnvironment
from src.federated.esa_adb_drl_round_provider import EsaAdbRoundProvider
from src.preprocessing.sequences import SequenceDataset
from src.satellite.esa_adb.clients import CLIENT_CHANNELS

REAL_CLIENT_IDS = list(CLIENT_CHANNELS.keys())  # client_1..client_8 — build_esa_adb_resource_signal requires real IDs

SMALL_MODEL_CONFIG = {"input_dim": 6, "projection_dim": 4, "hidden_dim": 4, "num_gru_layers": 1, "dropout": 0.0, "output_dim": 1}
SMALL_TRAINING_CONFIG = {"batch_size": 8, "learning_rate": 0.01, "local_epochs": 1, "seed": 42}


def _make_dataset(n: int, n_anomaly: int, input_dim: int, seed: int) -> SequenceDataset:
    rng = np.random.default_rng(seed)
    sequences = rng.normal(size=(n, 10, input_dim)).astype("float32")
    labels = np.array([1] * n_anomaly + [0] * (n - n_anomaly), dtype="int64")
    rng.shuffle(labels)
    record_labels = [np.zeros(10, dtype="int64") for _ in range(n)]
    return SequenceDataset(sequences=sequences, labels=labels, record_labels=record_labels, sequence_length=10, stride=10, label_strategy="last")


def _make_provider(num_rounds: int) -> EsaAdbRoundProvider:
    client_train = {cid: _make_dataset(30, 6, 6, seed=i) for i, cid in enumerate(REAL_CLIENT_IDS)}  # 8 real clients, matches src/drl/state.py's REQUIRED_NUM_CLIENTS
    pooled_val = _make_dataset(30, 6, 6, seed=999)
    return EsaAdbRoundProvider(
        client_train_datasets=client_train, pooled_validation_dataset=pooled_val,
        model_config=SMALL_MODEL_CONFIG, training_config=SMALL_TRAINING_CONFIG,
        shared_init_seed=42, num_rounds=num_rounds,
    )


def test_initial_signals_provider_returns_eight_signals_and_a_float_f1():
    provider = _make_provider(num_rounds=2)
    signals, val_f1 = provider.initial_signals_provider()
    assert len(signals) == 8
    assert isinstance(val_f1, float)
    assert 0.0 <= val_f1 <= 1.0


def test_round_provider_aggregates_and_records_round_state_dict():
    provider = _make_provider(num_rounds=2)
    signals, val_f1 = provider.initial_signals_provider()
    weights = action_to_aggregation_weights(0, signals)
    next_signals, new_val_f1 = provider.round_provider(1, weights, signals)

    assert len(next_signals) == 8
    assert 1 in provider.round_state_dicts
    assert 1 in provider.round_eval_history
    assert isinstance(new_val_f1, float)


def test_full_drl_environment_episode_runs_end_to_end_with_real_esa_adb_round_provider():
    """Integration test: drives a full 2-round episode through
    DRLFederatedEnvironment (the same environment class used for both
    Phase 9B/NSL-KDD and this ESA-ADB adaptation) with the real
    ESA-ADB provider (synthetic tiny data, not real ESA-ADB)."""
    provider = _make_provider(num_rounds=2)
    env = DRLFederatedEnvironment(
        initial_signals_provider=provider.initial_signals_provider, round_provider=provider.round_provider,
        num_rounds=2, lambda_fair=0.1, lambda_stab=0.05,
    )
    state = env.reset()
    assert state.shape == (44,)

    rng = np.random.default_rng(0)
    done = False
    steps = 0
    while not done:
        action = int(rng.integers(0, NUM_ACTIONS))
        result = env.step(action)
        assert isinstance(result.reward, float)
        state = result.observation
        assert state.shape == (44,)
        done = result.done
        steps += 1
    assert steps == 2
    assert set(provider.round_state_dicts.keys()) == {1, 2}


def test_no_test_dataset_reference_anywhere_in_module():
    """Structural privacy-boundary check (same style as
    tests/drl/test_privacy_boundary.py): this module must never
    reference test-split naming anywhere in its source."""
    import inspect

    from src.federated import esa_adb_drl_round_provider as mod

    source = inspect.getsource(mod)
    for forbidden in ("test_dataset", "pooled_test", "client_test_ds"):
        assert forbidden not in source, f"found forbidden reference to {forbidden!r} in esa_adb_drl_round_provider.py"
