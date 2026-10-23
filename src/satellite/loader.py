"""Phase 9A — reusable OPSSAT-AD dataset loader.

Mirrors src/data/loader.py's role for NSL-KDD: discovers the raw file,
loads it, and reports basic structure — nothing NSL-KDD-specific is
reused here, since OPSSAT-AD's schema is entirely different (see
results/reports/opssat_ad_inspection.md for the full inspection this
module's constants are based on).

`dataset.csv` (the file this module reads) is the pre-extracted,
fixed-width, one-row-per-segment feature table — NOT the raw
variable-length telemetry (that's `segments.csv`, unused in this phase;
see the Phase 9A pre-integration audit for why `dataset.csv` was chosen
as the starting representation).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw" / "opssat_ad"
DATASET_FILENAME = "dataset.csv"

# Columns confirmed directly from the downloaded file (see
# results/reports/opssat_ad_inspection.md) — NOT copied from the paper's
# abstract-level description, which used different (incorrect, for this
# file) channel names.
SEGMENT_ID_COLUMN = "segment"
SPLIT_COLUMN = "train"  # 1 = author's training segment, 0 = author's test segment
LABEL_COLUMN = "anomaly"  # 0 = nominal, 1 = anomalous (operational anomaly, NOT a cyberattack label)
CATEGORICAL_COLUMN = "channel"

# The 19 numeric feature columns — everything in dataset.csv except the
# segment ID, split flag, label, and the categorical channel column.
NUMERIC_FEATURE_COLUMNS = [
    "sampling", "duration", "len", "mean", "var", "std", "kurtosis", "skew",
    "n_peaks", "smooth10_n_peaks", "smooth20_n_peaks", "diff_peaks", "diff2_peaks",
    "diff_var", "diff2_var", "gaps_squared", "len_weighted", "var_div_duration", "var_div_len",
]


class OpsSatDatasetNotFoundError(FileNotFoundError):
    """Raised when data/raw/opssat_ad/dataset.csv is missing."""


def load_opssat_dataset(raw_dir: str | Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Load dataset.csv from `raw_dir`.

    Raises OpsSatDatasetNotFoundError with a clear message (including how
    to obtain the file) if it's missing — never silently returns an
    empty/partial result.
    """
    raw_dir = Path(raw_dir)
    dataset_path = raw_dir / DATASET_FILENAME

    if not dataset_path.is_file():
        raise OpsSatDatasetNotFoundError(
            f"OPSSAT-AD dataset.csv not found at {dataset_path}.\n"
            "Download it with:\n"
            "  python scripts/download_opssat_ad.py\n"
            "See docs/research/datasets/dataset_decision.md for provenance."
        )

    df = pd.read_csv(dataset_path)

    missing_columns = set(NUMERIC_FEATURE_COLUMNS + [SEGMENT_ID_COLUMN, SPLIT_COLUMN, LABEL_COLUMN, CATEGORICAL_COLUMN]) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"dataset.csv is missing expected columns: {sorted(missing_columns)}. "
            "The downloaded file may not match the version this loader was built against — "
            "re-run scripts/inspect_opssat_ad.py and compare against results/reports/opssat_ad_inspection.md."
        )

    return df


def split_by_author_train_flag(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (author_train_segments, author_test_segments) using the
    dataset's own `train` column. The returned test set must be touched
    exactly once, for final evaluation only — see src/satellite/preprocessing.py
    for how the validation set is carved out of the training portion only."""
    train_df = df[df[SPLIT_COLUMN] == 1].reset_index(drop=True)
    test_df = df[df[SPLIT_COLUMN] == 0].reset_index(drop=True)
    return train_df, test_df
