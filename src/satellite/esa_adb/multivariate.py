"""ESA-ADB multivariate (cross-channel) sequence construction — Experiment 1A.

Isolated addition alongside pipeline.py: does NOT modify loader.py,
labeling.py, preprocessing.py, clients.py, or pipeline.py. Reuses their
functions unchanged (load_channel_window, channel_anomaly_intervals,
resample_channel, fit_channel_scaler) plus
src/preprocessing/sequences.py:create_sequences (already
dataset-agnostic and multi-feature-column-capable — no changes needed
there either).

**Timestamp-alignment finding (verified, not assumed):** each channel is
resampled independently to a 300s grid (preprocessing.py). 55 of the 58
Target=YES channels resample to 105,408 rows starting 2000-01-01
00:00:00. The 3 subsystem_1 channels (61, 62, 63 — client_3's channels,
native ~15-minute cadence) resample to 105,407 rows starting one bin
later, 2000-01-01 00:05:00 (their first real reading in the approved
window falls after the 00:00:00-00:05:00 bin, so that leading bin has
no real data to forward-fill from and is correctly dropped by
resample_channel). All 58 channels agree on every OTHER timestamp,
including the final one (2000-12-31 23:55:00).

Concatenating the 58 independently-resampled series positionally would
silently misalign channels 61/62/63 by one 5-minute step relative to
the other 55 — exactly the risk flagged before this module was written.
The fix used here is an INNER JOIN on the timestamp index (the
intersection of all 58 channels' real timestamps) — this drops exactly
the single 2000-01-01 00:00:00 row globally (1 row out of 105,408, real
data only, nothing fabricated or interpolated) and guarantees every
remaining row is a genuine shared instant across all 58 channels.

**Joint label rule (disclosed, not invented from scratch — a direct,
minimal extension of the already-approved per-channel rule):** at each
aligned timestamp, the joint (multivariate) label is 1 if ANY of the 58
channels' own official per-timestamp label (labeling.label_timestamps,
unchanged) is 1 at that timestamp. This is the natural cross-channel
reading of the same official ESA anomaly intervals — a real
multivariate anomaly event affects specific channels at specific times,
so the joint state vector at a timestamp is anomalous exactly when at
least one of its channels is under an official anomaly interval. The
sequence-level label reuses the SAME "last"-timestep strategy already
approved for the univariate baseline (preprocessing.LABEL_STRATEGY) —
sequence label = joint label of the final aligned timestep in the
window. No new sequence-labeling strategy is introduced.

Per-channel normalization is preserved unchanged (one StandardScaler
per channel, fit on that channel's own TRAIN-split values only) — this
experiment isolates ONLY the input_dim change (1 -> 58), so every other
design choice from the univariate baseline is kept identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.preprocessing.sequences import SequenceDataset, create_sequences
from src.satellite.esa_adb import loader
from src.satellite.esa_adb import preprocessing as prep
from src.satellite.esa_adb.labeling import channel_anomaly_intervals, group_label_rows_by_channel, label_timestamps

JOINT_LABEL_COLUMN = "joint_label"


@dataclass
class AlignmentReport:
    channel_row_counts: dict[str, int]
    common_row_count: int
    misaligned_channels: dict[str, dict]  # channel -> {rows, first_ts, dropped_leading, dropped_trailing}
    common_first_ts: str
    common_last_ts: str

    def to_dict(self) -> dict:
        return {
            "channel_row_counts_distribution": {str(k): v for k, v in _count_distribution(self.channel_row_counts).items()},
            "common_row_count": self.common_row_count,
            "num_misaligned_channels": len(self.misaligned_channels),
            "misaligned_channels": self.misaligned_channels,
            "common_first_ts": self.common_first_ts,
            "common_last_ts": self.common_last_ts,
        }


def _count_distribution(row_counts: dict[str, int]) -> dict[int, int]:
    dist: dict[int, int] = {}
    for v in row_counts.values():
        dist[v] = dist.get(v, 0) + 1
    return dist


def load_all_resampled_channels() -> dict[str, pd.DataFrame]:
    """Load + resample every one of the 58 Target=YES channels
    independently (same real per-channel loading as pipeline.py's
    build_all_channels, but keeping each channel's full resampled
    [value, label] DataFrame rather than pre-sequencing it — needed
    here to align on timestamps BEFORE sequencing)."""
    label_rows_by_channel = group_label_rows_by_channel(loader.load_label_rows())
    out = {}
    for channel in loader.target_yes_channels():
        raw = loader.load_channel_window(channel)
        intervals = channel_anomaly_intervals(channel, label_rows_by_channel.get(channel, []))
        out[channel] = prep.resample_channel(raw, intervals)
    return out


def verify_alignment(channel_frames: dict[str, pd.DataFrame]) -> AlignmentReport:
    """Real, measured alignment check — never assumed.

    A channel is reported as "misaligned" if it is MISSING one or more
    timestamps that at least one other channel has (i.e., its own real
    index is not a superset of the union) — NOT simply "has fewer rows
    than the intersection size," which would backwards-flag the
    majority of well-covered channels instead of the actual outliers
    (e.g. in the real data, channels 61/62/63 are the ones missing the
    2000-01-01 00:00:00 bin that the other 55 channels have; those 55
    are the ones correctly NOT flagged)."""
    row_counts = {c: len(df) for c, df in channel_frames.items()}

    union_index = None
    intersection_index = None
    for df in channel_frames.values():
        union_index = df.index if union_index is None else union_index.union(df.index)
        intersection_index = df.index if intersection_index is None else intersection_index.intersection(df.index)
    union_index = union_index.sort_values()
    intersection_index = intersection_index.sort_values()

    misaligned = {}
    for c, df in channel_frames.items():
        missing = union_index.difference(df.index)
        if len(missing) > 0:
            missing_sorted = missing.sort_values()
            misaligned[c] = {
                "rows": len(df), "first_ts": str(df.index.min()), "last_ts": str(df.index.max()),
                "num_missing_timestamps": len(missing),
                "missing_timestamps_sample": [str(t) for t in missing_sorted[:5]],
            }

    return AlignmentReport(
        channel_row_counts=row_counts,
        common_row_count=len(intersection_index),
        misaligned_channels=misaligned,
        common_first_ts=str(intersection_index.min()),
        common_last_ts=str(intersection_index.max()),
    )


def build_aligned_multivariate_frame(channel_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Inner-join all 58 channels onto their common (real, not
    fabricated) timestamp intersection. Returns one wide DataFrame:
    columns = the 58 channel names (raw, not-yet-normalized values) +
    JOINT_LABEL_COLUMN (OR across all 58 channels' own official
    per-timestamp labels), indexed by the shared timestamp grid,
    sorted ascending."""
    common_index = None
    for df in channel_frames.values():
        common_index = df.index if common_index is None else common_index.intersection(df.index)
    common_index = common_index.sort_values()

    channels = sorted(channel_frames.keys(), key=lambda c: int(c.split("_")[1]))
    value_columns = {c: channel_frames[c].loc[common_index, prep.VALUE_COLUMN] for c in channels}
    wide = pd.DataFrame(value_columns, index=common_index)

    label_stack = np.stack([channel_frames[c].loc[common_index, prep.LABEL_COLUMN].to_numpy() for c in channels], axis=1)
    wide[JOINT_LABEL_COLUMN] = (label_stack.sum(axis=1) > 0).astype("int64")

    return wide


def split_multivariate_frame(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    bounds = {
        "train": (loader.TRAIN_START, loader.TRAIN_END),
        "validation": (loader.VALIDATION_START, loader.VALIDATION_END),
        "test": (loader.TEST_START, loader.TEST_END),
    }
    return {name: df[(df.index >= s) & (df.index <= e)] for name, (s, e) in bounds.items()}


def fit_multivariate_scalers(train_df: pd.DataFrame, channel_columns: list[str]) -> dict[str, prep.ChannelScaler]:
    """One StandardScaler PER CHANNEL, fit on that channel's own
    TRAIN-split values only — same normalization philosophy as the
    univariate baseline, unchanged."""
    return {c: prep.fit_channel_scaler(c, train_df[c].to_numpy(dtype="float64")) for c in channel_columns}


def transform_multivariate_frame(df: pd.DataFrame, scalers: dict[str, prep.ChannelScaler], channel_columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in channel_columns:
        out[c] = scalers[c].transform(df[c].to_numpy(dtype="float64"))
    return out


def build_multivariate_sequences(df: pd.DataFrame, channel_columns: list[str]) -> SequenceDataset:
    """Reuses src/preprocessing/sequences.py:create_sequences UNCHANGED
    — it is already dataset-agnostic and accepts an arbitrary list of
    feature columns, so no multivariate-specific sequencing code was
    needed beyond building the right input DataFrame."""
    return create_sequences(
        df, feature_columns=channel_columns, record_label_column=JOINT_LABEL_COLUMN,
        sequence_length=prep.SEQUENCE_LENGTH, stride=prep.STRIDE, label_strategy=prep.LABEL_STRATEGY,
    )
