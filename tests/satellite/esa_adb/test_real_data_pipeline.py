"""Tests that require the real, downloaded ESA-ADB Mission 1 archive
(data/raw/esa_adb/Mission1/ESA-Mission1/) — automatically skipped when
that archive is absent (e.g. on CI without the 3.5GB download), so the
rest of the suite never depends on it.

Covers: loader determinism (category 1), timestamp ordering (category
2), channel selection (category 3), reproducibility (category 12)."""

from __future__ import annotations

import pytest

from src.satellite.esa_adb import loader, pipeline

pytestmark = pytest.mark.skipif(not loader.MISSION_DIR.exists(), reason="real ESA-ADB Mission 1 archive not downloaded")


def test_target_yes_channels_returns_exactly_58_channels():
    channels = loader.target_yes_channels()
    assert len(channels) == 58


def test_target_yes_channels_is_deterministic():
    a = loader.target_yes_channels()
    b = loader.target_yes_channels()
    assert a == b  # same order, same content, every call


def test_target_yes_channels_excludes_target_no_channels():
    meta = loader.load_channel_metadata()
    channels = set(loader.target_yes_channels())
    target_no = {c for c, m in meta.items() if not m.target}
    assert channels.isdisjoint(target_no)


def test_load_channel_window_is_deterministic():
    a = loader.load_channel_window("channel_41")
    b = loader.load_channel_window("channel_41")
    assert a.equals(b)


def test_load_channel_window_timestamps_strictly_within_approved_window():
    series = loader.load_channel_window("channel_61")
    assert series.index.min() >= loader.WINDOW_START
    assert series.index.max() <= loader.WINDOW_END


def test_load_channel_window_timestamps_sorted_ascending():
    series = loader.load_channel_window("channel_70")
    assert series.index.is_monotonic_increasing


def test_load_channel_window_has_no_duplicate_timestamps():
    series = loader.load_channel_window("channel_12")
    assert not series.index.duplicated().any()


def test_load_channel_window_raises_for_unknown_channel():
    with pytest.raises(loader.EsaAdbDataNotFoundError):
        loader.load_channel_window("channel_9999")


@pytest.mark.slow
def test_single_channel_pipeline_reproducible():
    meta = loader.load_channel_metadata()
    label_rows = loader.load_label_rows()
    from src.satellite.esa_adb.labeling import group_label_rows_by_channel

    label_rows_by_channel = group_label_rows_by_channel(label_rows)

    result_a = pipeline.build_channel_pipeline("channel_61", meta["channel_61"], label_rows_by_channel)
    result_b = pipeline.build_channel_pipeline("channel_61", meta["channel_61"], label_rows_by_channel)

    assert result_a.resampled_rows == result_b.resampled_rows
    assert result_a.scaler.mean == pytest.approx(result_b.scaler.mean)
    assert result_a.scaler.scale == pytest.approx(result_b.scaler.scale)
    for split in pipeline.SPLITS:
        assert result_a.sequences[split].num_sequences == result_b.sequences[split].num_sequences
        assert result_a.sequences[split].anomaly_count == result_b.sequences[split].anomaly_count
