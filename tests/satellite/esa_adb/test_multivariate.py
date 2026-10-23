"""Tests for src/satellite/esa_adb/multivariate.py — timestamp
alignment detection/intersection join, joint-label OR logic,
deterministic construction. Synthetic data only, no real download
required (real-data alignment numbers are verified separately via
scripts/train_esa_adb_baseline_multivariate.py's own printed report)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.satellite.esa_adb import multivariate as mv
from src.satellite.esa_adb import preprocessing as prep


def _make_channel_df(index: pd.DatetimeIndex, values: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({prep.VALUE_COLUMN: values.astype("float32"), prep.LABEL_COLUMN: labels.astype("int64")}, index=index)


@pytest.fixture
def aligned_channel_frames() -> dict[str, pd.DataFrame]:
    """Two channels sharing the exact same timestamp grid (fully aligned)."""
    index = pd.date_range("2000-01-01 00:00:00", periods=10, freq="300s")
    return {
        "channel_1": _make_channel_df(index, np.arange(10), np.zeros(10)),
        "channel_2": _make_channel_df(index, np.arange(10) * 2, np.zeros(10)),
    }


@pytest.fixture
def misaligned_channel_frames() -> dict[str, pd.DataFrame]:
    """channel_3 starts ONE bin later than channel_1/channel_2 — mirrors
    the real subsystem_1 (channel_61/62/63) finding."""
    full_index = pd.date_range("2000-01-01 00:00:00", periods=10, freq="300s")
    short_index = full_index[1:]  # missing the very first bin
    return {
        "channel_1": _make_channel_df(full_index, np.arange(10), np.zeros(10)),
        "channel_2": _make_channel_df(full_index, np.arange(10) * 2, np.zeros(10)),
        "channel_3": _make_channel_df(short_index, np.arange(9), np.zeros(9)),
    }


def test_verify_alignment_detects_no_misalignment_when_fully_aligned(aligned_channel_frames):
    report = mv.verify_alignment(aligned_channel_frames)
    assert report.common_row_count == 10
    assert report.misaligned_channels == {}


def test_verify_alignment_detects_the_misaligned_channel(misaligned_channel_frames):
    """channel_3 is the outlier (missing the 00:00:00 bin that
    channel_1/channel_2 both have) — it must be the one flagged, not
    the two well-covered channels. This mirrors the real finding:
    channels 61/62/63 are flagged, not the other 55."""
    report = mv.verify_alignment(misaligned_channel_frames)
    assert report.common_row_count == 9  # intersection excludes the one unmatched leading bin
    assert set(report.misaligned_channels.keys()) == {"channel_3"}
    assert report.misaligned_channels["channel_3"]["num_missing_timestamps"] == 1
    assert report.misaligned_channels["channel_3"]["missing_timestamps_sample"] == ["2000-01-01 00:00:00"]
    assert report.channel_row_counts["channel_1"] == 10
    assert report.channel_row_counts["channel_3"] == 9


def test_build_aligned_multivariate_frame_drops_only_the_unmatched_row(misaligned_channel_frames):
    wide = mv.build_aligned_multivariate_frame(misaligned_channel_frames)
    assert len(wide) == 9
    assert wide.index.min() == pd.Timestamp("2000-01-01 00:05:00")  # the dropped bin was 00:00:00
    assert list(wide.columns) == ["channel_1", "channel_2", "channel_3", mv.JOINT_LABEL_COLUMN]


def test_build_aligned_multivariate_frame_preserves_real_values_no_fabrication(aligned_channel_frames):
    wide = mv.build_aligned_multivariate_frame(aligned_channel_frames)
    assert list(wide["channel_1"]) == list(range(10))
    assert list(wide["channel_2"]) == [i * 2 for i in range(10)]


def test_joint_label_is_or_across_channels():
    index = pd.date_range("2000-01-01", periods=5, freq="300s")
    frames = {
        "channel_1": _make_channel_df(index, np.zeros(5), np.array([0, 1, 0, 0, 0])),
        "channel_2": _make_channel_df(index, np.zeros(5), np.array([0, 0, 0, 1, 0])),
        "channel_3": _make_channel_df(index, np.zeros(5), np.array([0, 0, 0, 0, 0])),
    }
    wide = mv.build_aligned_multivariate_frame(frames)
    assert list(wide[mv.JOINT_LABEL_COLUMN]) == [0, 1, 0, 1, 0]


def test_joint_label_all_zero_when_no_channel_anomalous():
    index = pd.date_range("2000-01-01", periods=4, freq="300s")
    frames = {
        "channel_1": _make_channel_df(index, np.zeros(4), np.zeros(4)),
        "channel_2": _make_channel_df(index, np.zeros(4), np.zeros(4)),
    }
    wide = mv.build_aligned_multivariate_frame(frames)
    assert (wide[mv.JOINT_LABEL_COLUMN] == 0).all()


def test_split_multivariate_frame_uses_the_same_approved_boundaries():
    from src.satellite.esa_adb import loader

    index = pd.date_range("2000-01-01", "2000-12-31 23:55:00", freq="300s")
    df = pd.DataFrame({"channel_1": np.arange(len(index)), mv.JOINT_LABEL_COLUMN: np.zeros(len(index), dtype="int64")}, index=index)

    splits = mv.split_multivariate_frame(df)
    assert splits["train"].index.max() <= loader.TRAIN_END
    assert splits["validation"].index.min() >= loader.VALIDATION_START
    assert splits["test"].index.min() >= loader.TEST_START


def test_fit_multivariate_scalers_uses_train_only():
    train_df = pd.DataFrame({"channel_1": [1.0, 2.0, 3.0], "channel_2": [10.0, 20.0, 30.0]})
    scalers = mv.fit_multivariate_scalers(train_df, ["channel_1", "channel_2"])
    assert scalers["channel_1"].mean == pytest.approx(2.0)
    assert scalers["channel_2"].mean == pytest.approx(20.0)


def test_build_multivariate_sequences_shape():
    index = pd.date_range("2000-01-01", periods=120, freq="300s")
    df = pd.DataFrame(
        {f"channel_{i}": np.random.default_rng(i).normal(size=120) for i in range(1, 4)},
        index=index,
    )
    df[mv.JOINT_LABEL_COLUMN] = 0

    seq = mv.build_multivariate_sequences(df, ["channel_1", "channel_2", "channel_3"])
    assert seq.sequences.shape[1] == prep.SEQUENCE_LENGTH
    assert seq.sequences.shape[2] == 3  # input_dim = number of channels


def test_build_aligned_multivariate_frame_is_deterministic(misaligned_channel_frames):
    a = mv.build_aligned_multivariate_frame(misaligned_channel_frames)
    b = mv.build_aligned_multivariate_frame(misaligned_channel_frames)
    pd.testing.assert_frame_equal(a, b)
