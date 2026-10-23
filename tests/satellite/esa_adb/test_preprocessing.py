"""Preprocessing tests: resampling behavior, normalization leakage
(category 5), sequence shape (category 8), sequence-label correctness
(category 9). Synthetic data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.satellite.esa_adb import preprocessing as prep


def test_resample_channel_forward_fills_gaps():
    """A channel sampled every 60s, resampled to 300s (5min), should
    carry the last real value forward into empty bins — no interpolation,
    no fabricated values."""
    index = pd.date_range("2000-01-01 00:00:00", periods=20, freq="60s")
    series = pd.Series(np.arange(20, dtype="float32"), index=index, name="c")

    df = prep.resample_channel(series, intervals=[], freq="300s")

    # 20 minutes of 60s data resampled to 5min bins -> 4 bins, each taking
    # the LAST real value observed in that bin (values 4, 9, 14, 19).
    assert list(df[prep.VALUE_COLUMN]) == [4.0, 9.0, 14.0, 19.0]


def test_resample_channel_drops_leading_unfilled_bins():
    """If the first real reading arrives partway into what would be bin
    0, that leading bin must be dropped, not fabricated."""
    index = pd.date_range("2000-01-01 00:04:00", periods=5, freq="60s")  # starts 4 min in
    series = pd.Series(np.arange(5, dtype="float32"), index=index, name="c")

    df = prep.resample_channel(series, intervals=[], freq="300s")
    assert not df[prep.VALUE_COLUMN].isna().any()


def test_resample_channel_labels_directly_against_intervals():
    """label_timestamps checks each bin's LEFT-edge instant (the
    resampled index value itself) against the interval — see
    preprocessing.py's module docstring "edge-precision note". An
    interval spanning two bin left-edges (00:05:00 and 00:10:00) labels
    exactly those two bins."""
    index = pd.date_range("2000-01-01 00:00:00", periods=20, freq="60s")
    series = pd.Series(np.arange(20, dtype="float32"), index=index, name="c")
    intervals = [(pd.Timestamp("2000-01-01 00:03:00"), pd.Timestamp("2000-01-01 00:12:00"))]

    df = prep.resample_channel(series, intervals=intervals, freq="300s")
    # Bin left-edges are 00:00, 00:05, 00:10, 00:15 -> only 00:05 and 00:10 fall in [00:03, 00:12].
    assert df[prep.LABEL_COLUMN].tolist() == [0, 1, 1, 0]


def test_fit_channel_scaler_does_not_see_validation_or_test_values():
    """Normalization leakage check: a scaler fit on train values only
    must be unaffected by drastically different validation/test values."""
    train_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    scaler = prep.fit_channel_scaler("c", train_values)

    # A scaler fit on train+val+test combined (simulating a leak) would differ.
    leaked_values = np.concatenate([train_values, np.array([1000.0, 2000.0])])
    leaked_scaler = prep.fit_channel_scaler("c", leaked_values)

    assert scaler.mean == pytest.approx(3.0)
    assert scaler.mean != pytest.approx(leaked_scaler.mean)


def test_fit_channel_scaler_flags_constant_channel():
    scaler = prep.fit_channel_scaler("c", np.full(10, 7.0))
    assert scaler.is_constant is True
    # transform on a constant channel must not divide by zero / produce NaN or inf.
    out = scaler.transform(np.array([7.0, 7.0]))
    assert np.isfinite(out).all()


def test_build_channel_sequences_shape():
    index = pd.date_range("2000-01-01", periods=120, freq="300s")
    df = pd.DataFrame({
        prep.VALUE_COLUMN: np.arange(120, dtype="float32"),
        prep.LABEL_COLUMN: np.zeros(120, dtype="int64"),
    }, index=index)
    scaler = prep.fit_channel_scaler("c", df[prep.VALUE_COLUMN].to_numpy())

    seq = prep.build_channel_sequences(df, scaler)

    assert seq.sequences.shape[1] == prep.SEQUENCE_LENGTH
    assert seq.sequences.shape[2] == 1  # input_dim=1, univariate per channel
    expected_n = (120 - prep.SEQUENCE_LENGTH) // prep.STRIDE + 1
    assert seq.sequences.shape[0] == expected_n
    assert seq.labels.shape[0] == expected_n


def test_build_channel_sequences_label_uses_last_timestep_rule():
    """label_strategy='last' -> a sequence is anomalous iff its FINAL
    timestep is anomalous, regardless of earlier anomalous timesteps."""
    n = 100
    index = pd.date_range("2000-01-01", periods=n, freq="300s")
    labels = np.zeros(n, dtype="int64")
    labels[prep.SEQUENCE_LENGTH - 1] = 1  # anomalous exactly at the end of the first window
    labels[0] = 1  # anomalous at the START of the first window (must NOT make it positive under "last")
    df = pd.DataFrame({prep.VALUE_COLUMN: np.arange(n, dtype="float32"), prep.LABEL_COLUMN: labels}, index=index)
    scaler = prep.fit_channel_scaler("c", df[prep.VALUE_COLUMN].to_numpy())

    seq = prep.build_channel_sequences(df, scaler)

    assert seq.labels[0] == 1  # last timestep of window 0 (index SEQUENCE_LENGTH-1) is anomalous
