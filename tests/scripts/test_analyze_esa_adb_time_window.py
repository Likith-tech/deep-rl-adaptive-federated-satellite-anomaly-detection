"""Focused tests for scripts/analyze_esa_adb_time_window.py's pure
logic (month_range, build_events, window_stats) — synthetic data only,
no real ESA-ADB download required."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import analyze_esa_adb_time_window as m  # noqa: E402


def dt(y, mo, d=1, h=0):
    return datetime(y, mo, d, h, tzinfo=timezone.utc)


def test_month_range_single_month():
    assert m.month_range(dt(2005, 3, 10), dt(2005, 3, 20)) == ["2005-03"]


def test_month_range_spans_multiple_months():
    keys = m.month_range(dt(2005, 11, 20), dt(2006, 2, 5))
    assert keys == ["2005-11", "2005-12", "2006-01", "2006-02"]


def test_build_events_groups_by_id_and_computes_min_max_span():
    channels = {"channel_1": {"Subsystem": "subsystem_1"}, "channel_2": {"Subsystem": "subsystem_1"}}
    label_rows = [
        {"ID": "id_1", "Channel": "channel_1", "StartTime": "2005-01-01T00:00:00.000Z", "EndTime": "2005-01-02T00:00:00.000Z"},
        {"ID": "id_1", "Channel": "channel_2", "StartTime": "2005-01-01T12:00:00.000Z", "EndTime": "2005-01-03T00:00:00.000Z"},
    ]
    types = {"id_1": {"Category": "Anomaly", "Dimensionality": "Multivariate", "Locality": "Global", "Length": "Subsequence"}}

    events = m.build_events(channels, label_rows, types)
    assert len(events) == 1
    e = events[0]
    assert e["id"] == "id_1"
    assert e["start"] == dt(2005, 1, 1)  # earliest of the two StartTimes
    assert e["end"] == datetime(2005, 1, 3, tzinfo=timezone.utc)  # latest of the two EndTimes
    assert e["channels"] == ["channel_1", "channel_2"]
    assert e["subsystems"] == ["subsystem_1"]
    assert e["n_mappings"] == 2
    assert e["category"] == "Anomaly"


def test_build_events_multiple_ids_stay_separate():
    channels = {"channel_1": {"Subsystem": "subsystem_1"}, "channel_9": {"Subsystem": "subsystem_5"}}
    label_rows = [
        {"ID": "id_1", "Channel": "channel_1", "StartTime": "2005-01-01T00:00:00.000Z", "EndTime": "2005-01-02T00:00:00.000Z"},
        {"ID": "id_2", "Channel": "channel_9", "StartTime": "2007-06-01T00:00:00.000Z", "EndTime": "2007-06-01T01:00:00.000Z"},
    ]
    types = {"id_1": {"Category": "Anomaly", "Dimensionality": "Univariate", "Locality": "Local", "Length": "Point"},
             "id_2": {"Category": "Rare Event", "Dimensionality": "Univariate", "Locality": "Local", "Length": "Point"}}
    events = m.build_events(channels, label_rows, types)
    assert {e["id"] for e in events} == {"id_1", "id_2"}


def test_window_stats_counts_only_overlapping_events():
    events = [
        {"id": "in", "start": dt(2005, 6, 1), "end": dt(2005, 6, 15), "channels": ["c1"], "subsystems": ["s1"],
         "category": "Anomaly", "dimensionality": "Univariate", "locality": "Local", "length": "Point", "n_mappings": 1},
        {"id": "outside", "start": dt(2010, 1, 1), "end": dt(2010, 1, 5), "channels": ["c2"], "subsystems": ["s2"],
         "category": "Anomaly", "dimensionality": "Univariate", "locality": "Local", "length": "Point", "n_mappings": 1},
        {"id": "straddle_start", "start": dt(2005, 5, 20), "end": dt(2005, 6, 5), "channels": ["c3"], "subsystems": ["s1"],
         "category": "Rare Event", "dimensionality": "Univariate", "locality": "Local", "length": "Point", "n_mappings": 1},
    ]
    win_start, win_end = dt(2005, 6, 1), dt(2005, 6, 30)
    stats = m.window_stats(events, win_start, win_end)

    assert stats["n_events"] == 2  # "in" and "straddle_start"; "outside" excluded
    assert stats["n_channels_affected"] == 2  # c1, c3
    assert stats["n_subsystems_affected"] == 1  # s1 only
    assert stats["events_fully_inside"] == 1  # "in"
    assert stats["events_straddling_start"] == 1  # "straddle_start" began before window, active at window start


def test_window_stats_empty_window_returns_zero_events():
    events = [{"id": "e1", "start": dt(2001, 1, 1), "end": dt(2001, 1, 2), "channels": ["c1"], "subsystems": ["s1"],
               "category": "Anomaly", "dimensionality": "Univariate", "locality": "Local", "length": "Point", "n_mappings": 1}]
    stats = m.window_stats(events, dt(2012, 1, 1), dt(2012, 1, 31))
    assert stats["n_events"] == 0
    assert stats["n_channels_affected"] == 0
