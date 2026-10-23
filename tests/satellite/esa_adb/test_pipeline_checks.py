"""Pipeline-level pure-logic tests: temporal split correctness (category
6), anomaly-window/split-boundary leakage detection (category 7), and
sequence pooling used to build category-11's no-duplicate-across-clients
guarantee. Synthetic data only, no real download required."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.sequences import SequenceDataset
from src.satellite.esa_adb import loader, pipeline


def test_split_boundaries_ordered_and_disjoint_passes_for_real_config():
    assert pipeline.check_split_boundaries_ordered_and_disjoint() == []


def test_split_boundaries_check_detects_inverted_boundary(monkeypatch):
    monkeypatch.setattr(loader, "VALIDATION_START", pd.Timestamp("1999-01-01"))  # before TRAIN_START
    violations = pipeline.check_split_boundaries_ordered_and_disjoint()
    assert len(violations) == 1


def test_no_anomaly_interval_straddles_split_boundary_passes_for_clean_data():
    label_rows_by_channel = {
        "channel_synth": [
            {"ID": "id_1", "Channel": "channel_synth", "StartTime": "2000-03-01T00:00:00.000Z", "EndTime": "2000-03-02T00:00:00.000Z"},
        ]
    }
    channel_results = {"channel_synth": None}  # only keys are used by this check
    violations = pipeline.check_no_anomaly_interval_straddles_split_boundary(channel_results, label_rows_by_channel)
    assert violations == []


def test_no_anomaly_interval_straddles_split_boundary_detects_a_real_violation():
    """An interval spanning 2000-06-25 to 2000-07-05 straddles the real
    train/validation boundary (2000-07-01, per the Option B validation
    split — see loader.py's VALIDATION_START docstring) — must be caught."""
    label_rows_by_channel = {
        "channel_synth": [
            {"ID": "id_1", "Channel": "channel_synth", "StartTime": "2000-06-25T00:00:00.000Z", "EndTime": "2000-07-05T00:00:00.000Z"},
        ]
    }
    channel_results = {"channel_synth": None}
    violations = pipeline.check_no_anomaly_interval_straddles_split_boundary(channel_results, label_rows_by_channel)
    assert len(violations) == 1
    assert "channel_synth" in violations[0]


def test_no_anomaly_interval_straddles_validation_test_boundary_detects_a_real_violation():
    """An interval spanning 2000-09-25 to 2000-10-05 straddles the real
    validation/test boundary (2000-10-01, unchanged by Option B) —
    must be caught."""
    label_rows_by_channel = {
        "channel_synth": [
            {"ID": "id_1", "Channel": "channel_synth", "StartTime": "2000-09-25T00:00:00.000Z", "EndTime": "2000-10-05T00:00:00.000Z"},
        ]
    }
    channel_results = {"channel_synth": None}
    violations = pipeline.check_no_anomaly_interval_straddles_split_boundary(channel_results, label_rows_by_channel)
    assert len(violations) == 1
    assert "channel_synth" in violations[0]


def _make_seq_dataset(n: int, n_anomaly: int, seq_len: int = 5) -> SequenceDataset:
    sequences = np.zeros((n, seq_len, 1), dtype="float32")
    labels = np.array([1] * n_anomaly + [0] * (n - n_anomaly), dtype="int64")
    record_labels = [np.zeros(seq_len, dtype="int64") for _ in range(n)]
    return SequenceDataset(sequences=sequences, labels=labels, record_labels=record_labels, sequence_length=seq_len, stride=seq_len, label_strategy="last")


def test_pool_sequence_datasets_concatenates_correctly():
    a = _make_seq_dataset(3, 1)
    b = _make_seq_dataset(2, 2)
    pooled = pipeline.pool_sequence_datasets([a, b], sequence_length=5, stride=5, label_strategy="last")
    assert pooled.num_sequences == 5
    assert pooled.anomaly_count == 3


def test_pool_sequence_datasets_handles_all_empty():
    empty = SequenceDataset(
        sequences=np.empty((0, 5, 1), dtype="float32"), labels=np.empty((0,), dtype="int64"),
        record_labels=[], sequence_length=5, stride=5, label_strategy="last",
    )
    pooled = pipeline.pool_sequence_datasets([empty, empty], sequence_length=5, stride=5, label_strategy="last")
    assert pooled.num_sequences == 0
    assert pooled.sequences.shape == (0, 5, 1)


def test_pooled_client_sequences_have_no_cross_client_duplicates_when_channels_disjoint():
    """Since sequences are built strictly per-channel and clients own
    disjoint channel sets (enforced by clients.validate_partition), the
    pooled arrays for two different clients can never share a row that
    came from the same source channel+window. Verified here by giving
    each channel a unique constant value baked into every timestep, then
    checking no pooled sequence array value appears in more than one
    client's pool."""
    channel_a = _make_seq_dataset(2, 0)
    channel_a.sequences[:] = 1.0
    channel_b = _make_seq_dataset(2, 0)
    channel_b.sequences[:] = 2.0

    client_1 = pipeline.pool_sequence_datasets([channel_a], sequence_length=5, stride=5, label_strategy="last")
    client_2 = pipeline.pool_sequence_datasets([channel_b], sequence_length=5, stride=5, label_strategy="last")

    assert not np.isin(client_1.sequences, client_2.sequences).any()
