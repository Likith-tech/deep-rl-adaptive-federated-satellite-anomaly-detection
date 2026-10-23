"""ESA-ADB per-timestamp label construction.

Labeling rule (exact, documented — never invented):

For a given channel C, a real telemetry timestamp t is labeled
ANOMALOUS (1) if and only if there exists at least one row in the
official `labels.csv` such that:

    row.Channel == C  AND  row.StartTime <= t <= row.EndTime

using the row's own per-channel StartTime/EndTime bounds (never the
cross-channel event-level min/max span used only for the earlier
time-window audit in scripts/analyze_esa_adb_time_window.py — that
aggregation would incorrectly widen a channel's labeled range to match
another channel's window). All other timestamps are labeled NORMAL (0).

`labels.csv` timestamps are timezone-aware (ISO8601, 'Z'/UTC). The raw
per-channel telemetry series loaded by loader.load_channel_window() has
a timezone-NAIVE datetime64 index. Both are real UTC clock times (the
archive does not mix timezones), so StartTime/EndTime are converted to
naive UTC (tzinfo stripped after conversion, not before) for comparison
— this is a real, disclosed alignment step, not a labeling invention.

Anomaly events with `Length=Point` are represented in labels.csv with
StartTime == EndTime (a single-instant row) — the same `<=`/`<=` rule
handles this without a special case, since a single matching timestamp
still satisfies `StartTime <= t <= EndTime`. `Length=Subsequence`
events (the dominant case, 184/200) work identically via the interval
check. Multivariate events are handled naturally: a multivariate event
contributes one row per affected channel, and each row is applied only
to its own channel.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from src.satellite.esa_adb.loader import timestamp_to_utc


def channel_anomaly_intervals(channel: str, label_rows: list[dict[str, str]]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Real [StartTime, EndTime] intervals (naive UTC) for one channel,
    taken directly and only from labels.csv rows where Channel==channel."""
    intervals = []
    for r in label_rows:
        if r["Channel"] != channel:
            continue
        start = pd.Timestamp(timestamp_to_utc(r["StartTime"])).tz_localize(None)
        end = pd.Timestamp(timestamp_to_utc(r["EndTime"])).tz_localize(None)
        intervals.append((start, end))
    return intervals


def label_timestamps(index: pd.DatetimeIndex, intervals: list[tuple[pd.Timestamp, pd.Timestamp]]) -> np.ndarray:
    """Binary 0/1 array, one entry per timestamp in `index`, under the
    rule documented in this module's docstring. O(n_intervals *
    n_timestamps) via boolean OR over interval masks — intervals per
    channel are few (labels.csv has 3,589 rows across 58 channels, a
    handful per channel), so this stays fast without needing an
    interval tree."""
    labels = np.zeros(len(index), dtype="int64")
    if not intervals:
        return labels
    for start, end in intervals:
        mask = (index >= start) & (index <= end)
        labels |= np.asarray(mask)
    return labels.astype("int64")


def group_label_rows_by_channel(label_rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    by_channel: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in label_rows:
        by_channel[r["Channel"]].append(r)
    return dict(by_channel)
