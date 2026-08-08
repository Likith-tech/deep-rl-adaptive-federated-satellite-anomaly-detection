from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.sequences import (
    compute_sequence_label,
    create_sequences,
    split_ordered_train_validation,
)


def _make_ordered_df(n_rows: int, feature_cols: list[str], label_pattern: list[int]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {col: rng.normal(size=n_rows) for col in feature_cols}
    labels = [label_pattern[i % len(label_pattern)] for i in range(n_rows)]
    data["label_binary"] = labels
    # a "position" column makes it easy to assert order was preserved
    data["position"] = np.arange(n_rows)
    return pd.DataFrame(data)


FEATURE_COLS = ["f0", "f1", "f2"]


def test_create_sequences_shape_and_count_non_overlapping():
    df = _make_ordered_df(20, FEATURE_COLS, [0])
    result = create_sequences(df, FEATURE_COLS, "label_binary", sequence_length=5)

    # 20 rows / window 5, stride 5 (default) -> 4 complete sequences, no remainder
    assert result.num_sequences == 4
    assert result.sequences.shape == (4, 5, 3)
    assert result.sequence_length == 5
    assert result.stride == 5


def test_create_sequences_drops_incomplete_trailing_window():
    df = _make_ordered_df(22, FEATURE_COLS, [0])  # 22 / 5 -> 4 full windows, 2 leftover rows dropped
    result = create_sequences(df, FEATURE_COLS, "label_binary", sequence_length=5)
    assert result.num_sequences == 4


def test_create_sequences_label_is_any_anomaly_in_window():
    # Pattern repeats every 5: normal, normal, normal, ANOMALY, normal
    df = _make_ordered_df(15, FEATURE_COLS, [0, 0, 0, 1, 0])
    result = create_sequences(df, FEATURE_COLS, "label_binary", sequence_length=5)

    assert result.num_sequences == 3
    assert list(result.labels) == [1, 1, 1]  # every window contains exactly one "1"


def test_create_sequences_all_normal_window_labeled_normal():
    df = _make_ordered_df(10, FEATURE_COLS, [0])
    result = create_sequences(df, FEATURE_COLS, "label_binary", sequence_length=5)
    assert list(result.labels) == [0, 0]


def test_create_sequences_preserves_record_level_labels():
    df = _make_ordered_df(10, FEATURE_COLS, [0, 1, 0, 0, 0])
    result = create_sequences(df, FEATURE_COLS, "label_binary", sequence_length=5)

    assert len(result.record_labels) == result.num_sequences
    assert list(result.record_labels[0]) == [0, 1, 0, 0, 0]
    assert list(result.record_labels[1]) == [0, 1, 0, 0, 0]


def test_create_sequences_preserves_row_order_not_shuffled():
    df = _make_ordered_df(10, FEATURE_COLS, [0])
    result = create_sequences(df, ["position"], "label_binary", sequence_length=5)
    assert result.sequences[0, :, 0].tolist() == [0, 1, 2, 3, 4]
    assert result.sequences[1, :, 0].tolist() == [5, 6, 7, 8, 9]


def test_create_sequences_custom_stride_allows_overlap():
    df = _make_ordered_df(10, FEATURE_COLS, [0])
    result = create_sequences(df, FEATURE_COLS, "label_binary", sequence_length=5, stride=2)
    # windows start at 0, 2 -> only 2 windows fit fully within 10 rows with stride 2 and length 5
    # starts: 0,2,4 -> end=9 (fits, since 4+5=9<=10); next start=6 -> end=11 > 10, stop.
    assert result.num_sequences == 3
    assert result.stride == 2


def test_create_sequences_empty_dataframe_returns_no_sequences():
    df = _make_ordered_df(0, FEATURE_COLS, [0])
    result = create_sequences(df, FEATURE_COLS, "label_binary", sequence_length=5)
    assert result.num_sequences == 0
    assert result.sequences.shape == (0, 5, 3)


def test_create_sequences_rejects_invalid_sequence_length():
    df = _make_ordered_df(10, FEATURE_COLS, [0])
    with pytest.raises(ValueError):
        create_sequences(df, FEATURE_COLS, "label_binary", sequence_length=0)


def test_split_ordered_train_validation_preserves_order_and_no_shuffle():
    df = _make_ordered_df(100, FEATURE_COLS, [0])
    train_df, val_df = split_ordered_train_validation(df, validation_size=0.2)

    assert len(train_df) == 80
    assert len(val_df) == 20
    # train is strictly the first 80 rows, in original order
    assert train_df["position"].tolist() == list(range(80))
    # validation is strictly the last 20 rows, in original order
    assert val_df["position"].tolist() == list(range(80, 100))


def test_split_ordered_train_validation_no_overlap():
    df = _make_ordered_df(50, FEATURE_COLS, [0])
    train_df, val_df = split_ordered_train_validation(df, validation_size=0.3)
    assert set(train_df["position"]).isdisjoint(set(val_df["position"]))


def test_split_ordered_train_validation_rejects_invalid_fraction():
    df = _make_ordered_df(10, FEATURE_COLS, [0])
    with pytest.raises(ValueError):
        split_ordered_train_validation(df, validation_size=1.5)


def test_sequences_never_cross_split_boundary():
    """End-to-end guard: sequencing each split independently means no
    sequence can span the train/validation boundary."""
    df = _make_ordered_df(50, FEATURE_COLS, [0])
    train_df, val_df = split_ordered_train_validation(df, validation_size=0.2)

    train_seqs = create_sequences(train_df, ["position"], "label_binary", sequence_length=5)
    val_seqs = create_sequences(val_df, ["position"], "label_binary", sequence_length=5)

    train_positions = set(train_seqs.sequences.flatten().astype(int).tolist())
    val_positions = set(val_seqs.sequences.flatten().astype(int).tolist())
    assert train_positions.isdisjoint(val_positions)
    assert max(train_positions) < min(val_positions)


# --- compute_sequence_label: each strategy in isolation ---


def test_compute_sequence_label_any_strategy():
    assert compute_sequence_label(np.array([0, 0, 0, 0]), "any") == 0
    assert compute_sequence_label(np.array([0, 1, 0, 0]), "any") == 1
    assert compute_sequence_label(np.array([1, 1, 1, 1]), "any") == 1


def test_compute_sequence_label_majority_strategy():
    assert compute_sequence_label(np.array([0, 0, 0, 1]), "majority") == 0  # 1/4, not majority
    assert compute_sequence_label(np.array([1, 1, 1, 0]), "majority") == 1  # 3/4, majority
    assert compute_sequence_label(np.array([1, 1, 0, 0]), "majority") == 0  # exactly half -> normal (tie)


def test_compute_sequence_label_last_strategy():
    assert compute_sequence_label(np.array([1, 1, 1, 0]), "last") == 0
    assert compute_sequence_label(np.array([0, 0, 0, 1]), "last") == 1


def test_compute_sequence_label_ratio_strategy():
    window = np.array([1, 1, 0, 0])  # ratio = 0.5
    assert compute_sequence_label(window, "ratio", threshold=0.5) == 1  # >= threshold
    assert compute_sequence_label(window, "ratio", threshold=0.51) == 0
    assert compute_sequence_label(window, "ratio", threshold=0.0) == 1  # everything qualifies


def test_compute_sequence_label_ratio_requires_threshold():
    with pytest.raises(ValueError):
        compute_sequence_label(np.array([1, 0]), "ratio", threshold=None)


def test_compute_sequence_label_ratio_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        compute_sequence_label(np.array([1, 0]), "ratio", threshold=1.5)


def test_compute_sequence_label_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        compute_sequence_label(np.array([1, 0]), "bogus_strategy")


# --- create_sequences: strategy propagation and per-strategy class balance ---


def test_create_sequences_majority_strategy_reduces_anomaly_rate_vs_any():
    # Pattern: 1 anomaly per 5 records (any=anomaly for every window,
    # majority=never anomaly since 1/5 is not a majority).
    df = _make_ordered_df(20, FEATURE_COLS, [1, 0, 0, 0, 0])

    any_result = create_sequences(df, FEATURE_COLS, "label_binary", sequence_length=5, label_strategy="any")
    majority_result = create_sequences(
        df, FEATURE_COLS, "label_binary", sequence_length=5, label_strategy="majority"
    )

    assert any_result.anomaly_rate == 1.0
    assert majority_result.anomaly_rate == 0.0


def test_create_sequences_last_strategy_uses_final_record_only():
    df = _make_ordered_df(10, FEATURE_COLS, [1, 0, 0, 0, 0])  # last of each 5-window is index 4 -> label 0
    result = create_sequences(df, FEATURE_COLS, "label_binary", sequence_length=5, label_strategy="last")
    assert list(result.labels) == [0, 0]


def test_create_sequences_ratio_strategy_with_threshold():
    df = _make_ordered_df(10, FEATURE_COLS, [1, 1, 0, 0, 0])  # ratio = 0.4 per window
    below = create_sequences(
        df, FEATURE_COLS, "label_binary", sequence_length=5, label_strategy="ratio", label_threshold=0.5
    )
    at_or_below = create_sequences(
        df, FEATURE_COLS, "label_binary", sequence_length=5, label_strategy="ratio", label_threshold=0.4
    )
    assert list(below.labels) == [0, 0]
    assert list(at_or_below.labels) == [1, 1]


def test_sequence_dataset_reports_class_distribution():
    df = _make_ordered_df(20, FEATURE_COLS, [1, 0, 0, 0, 0])
    result = create_sequences(df, FEATURE_COLS, "label_binary", sequence_length=5, label_strategy="any")
    assert result.anomaly_count == 4
    assert result.normal_count == 0
    assert result.anomaly_rate == 1.0


def test_create_sequences_records_label_strategy_metadata():
    df = _make_ordered_df(10, FEATURE_COLS, [0])
    result = create_sequences(
        df, FEATURE_COLS, "label_binary", sequence_length=5, label_strategy="ratio", label_threshold=0.3
    )
    assert result.label_strategy == "ratio"
    assert result.label_threshold == 0.3
