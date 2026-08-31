"""Label-correctness tests (category 4: label correctness) — synthetic
data only, no real download required."""

from __future__ import annotations

import pandas as pd

from src.satellite.esa_adb.labeling import channel_anomaly_intervals, group_label_rows_by_channel, label_timestamps


def test_channel_anomaly_intervals_filters_to_own_channel_only(synthetic_label_rows):
    intervals = channel_anomaly_intervals("channel_synth", synthetic_label_rows)
    # Two rows belong to channel_synth (id_1's mapping and id_2); channel_other's row must be excluded.
    assert len(intervals) == 2


def test_channel_anomaly_intervals_ignores_other_channels(synthetic_label_rows):
    intervals = channel_anomaly_intervals("channel_other", synthetic_label_rows)
    assert len(intervals) == 1


def test_label_timestamps_marks_only_inside_interval(synthetic_channel_series, synthetic_label_rows):
    intervals = channel_anomaly_intervals("channel_synth", synthetic_label_rows)
    labels = label_timestamps(synthetic_channel_series.index, intervals)

    idx = synthetic_channel_series.index
    inside_subsequence = (idx >= pd.Timestamp("2000-01-01 00:10:00")) & (idx <= pd.Timestamp("2000-01-01 00:20:00"))
    outside = ~inside_subsequence & (idx != pd.Timestamp("2000-01-01 00:50:00"))

    assert (labels[inside_subsequence] == 1).all()
    assert (labels[outside] == 0).all()


def test_label_timestamps_handles_point_event(synthetic_channel_series, synthetic_label_rows):
    """A Length=Point event has StartTime == EndTime; the single matching
    timestamp must still be labeled 1 via the same <=/<= rule (no special case needed)."""
    intervals = channel_anomaly_intervals("channel_synth", synthetic_label_rows)
    labels = label_timestamps(synthetic_channel_series.index, intervals)

    point_ts_position = synthetic_channel_series.index.get_loc(pd.Timestamp("2000-01-01 00:50:00"))
    assert labels[point_ts_position] == 1


def test_label_timestamps_empty_intervals_returns_all_zero(synthetic_channel_series):
    labels = label_timestamps(synthetic_channel_series.index, [])
    assert (labels == 0).all()
    assert labels.dtype.name == "int64"


def test_group_label_rows_by_channel_groups_correctly(synthetic_label_rows):
    grouped = group_label_rows_by_channel(synthetic_label_rows)
    assert set(grouped.keys()) == {"channel_synth", "channel_other"}
    assert len(grouped["channel_synth"]) == 2
    assert len(grouped["channel_other"]) == 1


def test_multivariate_event_applies_independently_per_channel(synthetic_label_rows):
    """id_1 affects both channel_synth and channel_other (multivariate) —
    each channel's own interval must be derived only from its own row,
    never from the other channel's row."""
    synth_intervals = channel_anomaly_intervals("channel_synth", synthetic_label_rows)
    other_intervals = channel_anomaly_intervals("channel_other", synthetic_label_rows)
    # Same real-time span for id_1 in this fixture, but each list is independently built.
    assert synth_intervals[0] == other_intervals[0]
    assert len(synth_intervals) != len(other_intervals)  # channel_synth also has id_2 (point event)
