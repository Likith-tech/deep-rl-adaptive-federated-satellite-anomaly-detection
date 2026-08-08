"""Reusable NSL-KDD dataset loader.

Discovers the raw dataset files, loads them into pandas DataFrames, and
reports basic quality signals (shape, dtypes, missing/duplicate/infinite
values, label distribution). Paths are resolved relative to a configurable
raw-data directory — never hardcoded to an absolute path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset_info import ColumnProfile, DatasetInfo
from src.data.schema import (
    ALL_COLUMNS,
    CATEGORICAL_COLUMNS,
    LABEL_COLUMN,
    RAW_TEST_FILENAME,
    RAW_TRAIN_FILENAME,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the expected raw dataset files are missing."""


@dataclass
class NslKddDataset:
    train: pd.DataFrame
    test: pd.DataFrame

    @property
    def combined(self) -> pd.DataFrame:
        return pd.concat([self.train, self.test], ignore_index=True)


def _expected_files(raw_dir: Path) -> dict[str, Path]:
    return {
        "train": raw_dir / RAW_TRAIN_FILENAME,
        "test": raw_dir / RAW_TEST_FILENAME,
    }


def _load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=ALL_COLUMNS)
    return df


def load_nsl_kdd(raw_dir: str | Path = DEFAULT_RAW_DIR) -> NslKddDataset:
    """Load the NSL-KDD train and test splits from `raw_dir`.

    Raises DatasetNotFoundError with a clear message (including the exact
    expected filenames and how to obtain them) if the files are missing.
    """
    raw_dir = Path(raw_dir)
    files = _expected_files(raw_dir)
    missing = [str(path) for path in files.values() if not path.is_file()]

    if missing:
        raise DatasetNotFoundError(
            "NSL-KDD raw files not found. Expected:\n"
            f"  {files['train']}\n"
            f"  {files['test']}\n"
            f"Missing: {missing}\n\n"
            "Download them with:\n"
            '  curl -o data/raw/KDDTrain+.txt '
            '"https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt"\n'
            '  curl -o data/raw/KDDTest+.txt '
            '"https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt"\n'
            "See docs/datasets/dataset_selection.md for details."
        )

    return NslKddDataset(
        train=_load_split(files["train"]),
        test=_load_split(files["test"]),
    )


def profile_dataframe(
    df: pd.DataFrame,
    source_files: list[str],
    label_column: str = LABEL_COLUMN,
) -> DatasetInfo:
    """Build a DatasetInfo summary for `df` (missing/duplicate/infinite values, dtypes, label distribution)."""
    numeric_df = df.select_dtypes(include=[np.number])
    infinite_count = int(np.isinf(numeric_df.to_numpy()).sum()) if not numeric_df.empty else 0

    columns = [
        ColumnProfile(
            name=col,
            dtype=str(df[col].dtype),
            missing_count=int(df[col].isna().sum()),
            is_categorical=col in CATEGORICAL_COLUMNS,
        )
        for col in df.columns
    ]

    class_distribution = (
        df[label_column].value_counts().to_dict() if label_column in df.columns else {}
    )

    return DatasetInfo(
        source_files=source_files,
        num_rows=len(df),
        num_columns=len(df.columns),
        columns=columns,
        duplicate_rows=int(df.duplicated().sum()),
        infinite_value_count=infinite_count,
        label_column=label_column,
        class_distribution={str(k): int(v) for k, v in class_distribution.items()},
        memory_usage_mb=float(df.memory_usage(deep=True).sum()) / (1024 * 1024),
    )
