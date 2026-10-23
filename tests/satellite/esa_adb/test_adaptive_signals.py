"""Tests for src/satellite/esa_adb/adaptive_signals.py — the ESA-ADB
real (non-simulated) resource/fairness signals used by the rule-based
and DQN adaptive FL experiments."""

from __future__ import annotations

import numpy as np
import pytest

from src.satellite.esa_adb.adaptive_signals import (
    binary_shannon_entropy,
    build_esa_adb_fairness_signals,
    build_esa_adb_resource_signal,
    build_esa_adb_resource_signals,
)
from src.satellite.esa_adb.clients import CLIENT_CHANNELS


def test_binary_shannon_entropy_is_zero_for_all_one_class():
    assert binary_shannon_entropy(np.zeros(10, dtype="int64")) == 0.0
    assert binary_shannon_entropy(np.ones(10, dtype="int64")) == 0.0


def test_binary_shannon_entropy_is_zero_for_empty_array():
    assert binary_shannon_entropy(np.array([], dtype="int64")) == 0.0


def test_binary_shannon_entropy_is_maximal_at_perfect_balance():
    balanced = np.array([0, 1] * 50, dtype="int64")
    skewed = np.array([0] * 95 + [1] * 5, dtype="int64")
    assert binary_shannon_entropy(balanced) == pytest.approx(1.0, abs=1e-9)
    assert binary_shannon_entropy(balanced) > binary_shannon_entropy(skewed)


def test_build_esa_adb_resource_signal_matches_real_channel_partition():
    for client_id, owned in CLIENT_CHANNELS.items():
        assert build_esa_adb_resource_signal(client_id) == pytest.approx(1.0 / len(owned))


def test_build_esa_adb_resource_signal_unknown_client_raises_keyerror():
    with pytest.raises(KeyError):
        build_esa_adb_resource_signal("not_a_real_client")


def test_build_esa_adb_resource_signals_covers_all_eight_real_clients():
    signals = build_esa_adb_resource_signals(list(CLIENT_CHANNELS.keys()))
    assert set(signals.keys()) == set(CLIENT_CHANNELS.keys())
    assert all(v > 0 for v in signals.values())


def test_build_esa_adb_fairness_signals_computed_per_client_from_own_labels_only():
    labels = {
        "client_a": np.array([0, 0, 0, 1], dtype="int64"),
        "client_b": np.array([0, 1], dtype="int64"),
    }
    signals = build_esa_adb_fairness_signals(labels)
    assert signals["client_a"] == pytest.approx(binary_shannon_entropy(labels["client_a"]))
    assert signals["client_b"] == pytest.approx(binary_shannon_entropy(labels["client_b"]))
    assert signals["client_b"] > signals["client_a"]  # more balanced -> higher entropy
