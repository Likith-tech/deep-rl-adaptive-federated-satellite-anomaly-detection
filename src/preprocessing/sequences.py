"""Sequence construction for temporal anomaly detection.

NSL-KDD has no reliable timestamp field representing genuine
chronological network traffic (see docs/datasets/dataset_selection.md
and docs/project-progress/04-phase-3-spatio-temporal.md). What we build
here is a sequence of ORDERED/SEQUENTIAL records — the dataset's own
provided row ordering — not a true time-series. This is documented
deliberately and must not be described as genuine satellite orbital
time-series data.

Three pure, independently-testable pieces:

1. `split_ordered_train_validation` — a contiguous, order-preserving
   train/validation split (first block = train, last block =
   validation). This is DIFFERENT from Phase 1's shuffled, stratified
   `split_train_validation` (src/preprocessing/splitting.py), which is
   correct for the row-independent MLP baseline but would destroy
   sequence order if reused here. Using a fresh split keeps sequences
   from ever crossing the train/validation boundary.

2. `compute_sequence_label` — turns a window of record-level labels
   into a single sequence-level label, under a choice of strategies
   (see LabelStrategy below). Kept separate from `create_sequences` so
   the labeling logic itself is trivially unit-testable.

3. `create_sequences` — turns an ordered DataFrame into fixed-length,
   non-overlapping (by default) sequences, using `compute_sequence_label`
   for the sequence-level label while preserving the original
   record-level labels for future experiments.

Sequences are never constructed across split boundaries: each split
(train/validation/test) is windowed independently, so no sequence can
contain records from two different splits.

## Sequence-labeling strategies

An initial experiment (see docs/project-progress/04-phase-3-spatio-temporal.md
and results/reports/temporal_results.md, "Initial sequence-labeling
experiment") used ANY-anomaly labeling and found it produced a
near-single-class sequence dataset on NSL-KDD (records are anomalous
~46-57% of the time, so a 16-record window almost always contains at
least one anomaly). That result was rejected as an invalid benchmark,
not used as evidence of model quality, and this module was reworked to
support multiple strategies so a defensible one could be chosen based
on actual measured class distributions (see
scripts/analyze_sequence_labeling.py):

- "any": sequence = anomaly if ANY record in the window is anomalous.
  Simple, but degenerate on NSL-KDD's per-record base rate (see above).
- "majority": sequence = anomaly if MORE THAN HALF the records in the
  window are anomalous (ties, i.e. exactly half, count as normal).
- "last": sequence label = the label of the final record in the window
  (treats the sequence as "context leading up to this record").
- "ratio": sequence = anomaly if the window's anomaly ratio is >=
  `threshold` (a required, configurable float in [0, 1] for this
  strategy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

LabelStrategy = Literal["any", "majority", "last", "ratio"]
VALID_LABEL_STRATEGIES: tuple[str, ...] = ("any", "majority", "last", "ratio")


@dataclass
class SequenceDataset:
    """A batch of fixed-length sequences built from one dataset split.

    sequences: float32 array, shape (num_sequences, sequence_length, num_features)
    labels: int array, shape (num_sequences,) — sequence-level label under `label_strategy`
    record_labels: list of int arrays, one per sequence, shape (sequence_length,) each
        — the original per-record labels, preserved (never discarded) so
        the sequence-level label can always be recomputed under a
        different strategy without rebuilding sequences from scratch.
    sequence_length: the configured window size
    stride: the configured step between window starts
    label_strategy: which strategy was used to compute `labels`
    label_threshold: the threshold used, if label_strategy == "ratio" (else None)
    """

    sequences: np.ndarray
    labels: np.ndarray
    record_labels: list[np.ndarray]
    sequence_length: int
    stride: int
    label_strategy: str = "any"
    label_threshold: float | None = None

    @property
    def num_sequences(self) -> int:
        return len(self.labels)

    @property
    def num_features(self) -> int:
        return self.sequences.shape[-1] if len(self.sequences) else 0

    @property
    def anomaly_count(self) -> int:
        return int(self.labels.sum())

    @property
    def normal_count(self) -> int:
        return int(len(self.labels) - self.labels.sum())

    @property
    def anomaly_rate(self) -> float:
        return float(self.labels.mean()) if len(self.labels) else 0.0


def split_ordered_train_validation(
    df: pd.DataFrame, validation_size: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Order-preserving, contiguous train/validation split.

    Unlike src/preprocessing/splitting.py's split_train_validation (which
    shuffles), this keeps row order intact — required so that sequences
    built from each half are genuinely ordered. The first
    (1 - validation_size) fraction of rows becomes train; the remaining
    rows become validation. No shuffling, no stratification.
    """
    if not 0.0 < validation_size < 1.0:
        raise ValueError(f"validation_size must be in (0, 1), got {validation_size}")

    n = len(df)
    split_index = int(round(n * (1 - validation_size)))
    train_df = df.iloc[:split_index].reset_index(drop=True)
    validation_df = df.iloc[split_index:].reset_index(drop=True)
    return train_df, validation_df


def compute_sequence_label(
    window_labels: np.ndarray,
    strategy: LabelStrategy = "any",
    threshold: float | None = None,
) -> int:
    """Reduce a window of record-level 0/1 labels to a single sequence label.

    See module docstring for what each strategy means. Raises ValueError
    for an unknown strategy, or for strategy="ratio" without a threshold.
    """
    if strategy not in VALID_LABEL_STRATEGIES:
        raise ValueError(f"Unknown label_strategy {strategy!r}; expected one of {VALID_LABEL_STRATEGIES}")

    if strategy == "any":
        return int(window_labels.any())
    if strategy == "majority":
        return int(window_labels.mean() > 0.5)
    if strategy == "last":
        return int(window_labels[-1])
    # strategy == "ratio"
    if threshold is None:
        raise ValueError("threshold is required when label_strategy='ratio'")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    return int(window_labels.mean() >= threshold)


def create_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    record_label_column: str,
    sequence_length: int,
    stride: int | None = None,
    label_strategy: LabelStrategy = "any",
    label_threshold: float | None = None,
) -> SequenceDataset:
    """Build fixed-length sequences from an ORDERED DataFrame.

    Windows are taken in row order (`df` must already be in the order
    intended for sequencing — this function does not sort or shuffle).
    `stride` defaults to `sequence_length` (non-overlapping windows).
    Any trailing rows that don't fill a complete window are dropped
    (documented, not silently misleading — see SequenceDataset.num_sequences
    vs len(df)).

    The sequence-level label is computed by `compute_sequence_label`
    under `label_strategy` (default "any", kept for backward
    compatibility — see module docstring for why this default alone is
    NOT recommended on NSL-KDD without checking the resulting class
    distribution first).
    """
    if sequence_length < 1:
        raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")
    stride = stride or sequence_length
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    features = df[feature_columns].to_numpy(dtype="float32")
    record_labels_all = df[record_label_column].to_numpy(dtype="int64")

    n = len(df)
    sequences: list[np.ndarray] = []
    seq_labels: list[int] = []
    record_labels: list[np.ndarray] = []

    start = 0
    while start + sequence_length <= n:
        end = start + sequence_length
        window_features = features[start:end]
        window_labels = record_labels_all[start:end]

        sequences.append(window_features)
        seq_labels.append(compute_sequence_label(window_labels, label_strategy, label_threshold))
        record_labels.append(window_labels)

        start += stride

    if sequences:
        sequences_array = np.stack(sequences).astype("float32")
    else:
        sequences_array = np.empty((0, sequence_length, len(feature_columns)), dtype="float32")

    return SequenceDataset(
        sequences=sequences_array,
        labels=np.array(seq_labels, dtype="int64"),
        record_labels=record_labels,
        sequence_length=sequence_length,
        stride=stride,
        label_strategy=label_strategy,
        label_threshold=label_threshold,
    )
