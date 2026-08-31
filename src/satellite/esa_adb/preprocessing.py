"""ESA-ADB preprocessing: resampling, train-only normalization, sequence
construction.

Real per-channel sampling rates measured directly inside the approved
2000 window (see docs/research/experiments/esa-adb/README.md Entry 3)
vary by more than an order of magnitude across subsystems: ~30s
(subsystem_3/subsystem_5), ~90s (subsystem_6), ~900s/15min
(subsystem_1). A single fixed-length sequence cannot be built directly
from four different native cadences, so every channel is resampled to
one common grid before sequencing.

**Resampling design decision (disclosed, not tuned against any
downstream metric — made before any model is trained):**

    RESAMPLE_FREQ = "300s" (5 minutes)

- Chosen because it is coarser than every fast channel's native rate
  (~30s/~90s), so those channels are genuinely downsampled (aggregated,
  not just relabeled), and finer than the slowest channel's native rate
  (~900s/15min, subsystem_1), so subsystem_1 is forward-filled between
  its real readings rather than upsampled by interpolation/fabrication.
- Per bin: take the LAST real reading observed in that 5-minute bin
  (`resample(...).last()`), representing the most recent real sensor
  state — then forward-fill any bin with no reading at all (this
  happens routinely for subsystem_1's ~15-minute native cadence). No
  interpolation, no fabricated values: every resampled value is either
  a real reading or a real reading carried forward.
- Leading bins before a channel's FIRST real reading (no prior value to
  forward-fill) are dropped, never filled with a fabricated default.

**Label alignment**: anomaly intervals ([StartTime, EndTime]) are
continuous real-time spans, not tied to the original discrete readings,
so resampled timestamps are labeled directly against those intervals
(src/satellite/esa_adb/labeling.py:label_timestamps) rather than by
propagating/aggregating the pre-resample per-reading labels — this
avoids any resampling-induced label distortion.

Disclosed edge-precision note: `resample(...).last()` labels each bin
with its LEFT-edge instant (pandas' default `label="left"`), and
label_timestamps checks only that single instant against each interval
— it does not check whether the interval merely overlaps the bin's
5-minute span. This can miss labeling the single bin in which an
anomaly interval *starts* (if the interval begins strictly after that
bin's left edge) or the single bin in which it *ends*. Given real
anomaly durations are dominated by `Subsequence`-length events spanning
hours to days (see esa_adb_dataset_audit.md Section 5), this affects at
most the first/last 5-minute bin of a small minority of events and is
judged acceptable; it is not silently ignored — this note exists so the
behavior is never mistaken for a bug later.

**Model input shape decision (disclosed):** each channel's resampled,
normalized value is a SINGLE scalar per timestep (input_dim=1).
Sequences are built per channel (never combining multiple channels into
one multivariate vector) so that every one of the 8 FL clients trains
the SAME architecture (TemporalGRUAttention with input_dim=1), which
FedAvg requires for cross-client parameter averaging. A client's local
training pool is the UNION of the univariate sequences from every
channel assigned to that client (src/satellite/esa_adb/clients.py) —
this preserves the real subsystem/group partition as the source of
non-IID structure while avoiding the input-dimension-mismatch problem
already documented for the SMAP/MSL audit (Track B, rejected candidate
for the same reason).

**Sequence length/stride/label rule (disclosed, chosen before any
metric is computed):**

    SEQUENCE_LENGTH = 50   (50 x 5min = ~4.2 hours of context)
    STRIDE = 25            (50% overlap)
    LABEL_STRATEGY = "last"

The "last" rule (sequence label = label of the final timestep) is
adopted directly from this project's own prior, evidence-based decision
for NSL-KDD (src/preprocessing/sequences.py's module docstring): an
earlier experiment found "any"-anomaly labeling degenerate (near-100%
positive) on a dataset where anomalous records are common inside a
window. The same risk applies here (anomaly events are frequently
`Subsequence`-length, i.e. many consecutive resampled timesteps), so
"any" is rejected for the same, already-documented reason, without
re-litigating it. The resulting TRAIN-split label balance is checked
empirically after sequencing (see pipeline.py) as a sanity/degeneracy
check only — never used to retune the rule against held-out
performance.

Normalization: one StandardScaler PER CHANNEL, fit on that channel's
own TRAIN-split resampled values only (never validation/test) — pooling
raw channels with different absolute physical scales into one
univariate stream without per-channel fitting would let one channel's
magnitude dominate purely because of its units, not its dynamics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.preprocessing.sequences import SequenceDataset, create_sequences

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "results" / "models" / "esa_adb_preprocessing"

RESAMPLE_FREQ = "300s"
SEQUENCE_LENGTH = 50
STRIDE = 25
LABEL_STRATEGY = "last"

VALUE_COLUMN = "value"
LABEL_COLUMN = "label"


def resample_channel(series: pd.Series, intervals: list[tuple[pd.Timestamp, pd.Timestamp]], freq: str = RESAMPLE_FREQ) -> pd.DataFrame:
    """Resample one real channel series to `freq` (last-value-per-bin +
    forward-fill; see module docstring), drop leading bins with no real
    reading yet, and label the resulting grid directly against the
    channel's real anomaly intervals.

    Returns a DataFrame indexed by the resampled timestamp with columns
    [VALUE_COLUMN, LABEL_COLUMN].
    """
    from src.satellite.esa_adb.labeling import label_timestamps

    resampled = series.resample(freq).last()
    resampled = resampled.ffill()
    resampled = resampled.dropna()  # leading bins before the first real reading

    labels = label_timestamps(resampled.index, intervals)

    return pd.DataFrame({VALUE_COLUMN: resampled.to_numpy(dtype="float32"), LABEL_COLUMN: labels}, index=resampled.index)


@dataclass
class ChannelScaler:
    channel: str
    mean: float
    scale: float
    is_constant: bool  # True if the train-split values had ~zero variance

    def transform(self, values: np.ndarray) -> np.ndarray:
        scale = self.scale if self.scale > 1e-12 else 1.0
        return ((values - self.mean) / scale).astype("float32")


def fit_channel_scaler(channel: str, train_values: np.ndarray) -> ChannelScaler:
    """Fit a StandardScaler on ONE channel's TRAIN-split values only.

    Constant-channel detection uses the RAW train-value variance, not
    scaler.scale_: sklearn's StandardScaler internally substitutes any
    zero variance with scale_=1.0 (to avoid a division by zero in
    transform), so scale_ alone can never reveal that a channel was
    constant — checking the raw values directly is required."""
    is_constant = bool(np.std(train_values) < 1e-12)

    scaler = StandardScaler()
    scaler.fit(train_values.reshape(-1, 1).astype("float64"))
    mean = float(scaler.mean_[0])
    scale = float(scaler.scale_[0])
    return ChannelScaler(channel=channel, mean=mean, scale=scale, is_constant=is_constant)


def build_channel_sequences(df: pd.DataFrame, scaler: ChannelScaler) -> SequenceDataset:
    """Normalize (using an ALREADY-FIT scaler) and build fixed-length
    sequences for one channel-split DataFrame (must already be sorted,
    resampled, and restricted to a single split — see pipeline.py)."""
    normalized = df.copy()
    normalized[VALUE_COLUMN] = scaler.transform(df[VALUE_COLUMN].to_numpy(dtype="float64"))
    return create_sequences(
        normalized,
        feature_columns=[VALUE_COLUMN],
        record_label_column=LABEL_COLUMN,
        sequence_length=SEQUENCE_LENGTH,
        stride=STRIDE,
        label_strategy=LABEL_STRATEGY,
    )


def save_scalers(scalers: dict[str, ChannelScaler], artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scalers, artifacts_dir / "channel_scalers.joblib")
    summary = {
        c: {"mean": s.mean, "scale": s.scale, "is_constant": s.is_constant}
        for c, s in scalers.items()
    }
    (artifacts_dir / "channel_scalers_summary.json").write_text(json.dumps(summary, indent=2))


def load_scalers(artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR) -> dict[str, ChannelScaler]:
    return joblib.load(artifacts_dir / "channel_scalers.joblib")
