from __future__ import annotations

from pathlib import Path

import pytest

from src.satellite.loader import (
    LABEL_COLUMN,
    NUMERIC_FEATURE_COLUMNS,
    OpsSatDatasetNotFoundError,
    load_opssat_dataset,
    split_by_author_train_flag,
)


def test_load_opssat_dataset_raises_clear_error_when_missing(tmp_path: Path):
    with pytest.raises(OpsSatDatasetNotFoundError, match="download_opssat_ad"):
        load_opssat_dataset(tmp_path / "does_not_exist")


def test_load_opssat_dataset_reads_expected_columns(synthetic_opssat_csv):
    df = load_opssat_dataset(synthetic_opssat_csv)
    for col in NUMERIC_FEATURE_COLUMNS:
        assert col in df.columns
    assert LABEL_COLUMN in df.columns
    assert set(df[LABEL_COLUMN].unique()).issubset({0, 1})


def test_load_opssat_dataset_rejects_missing_columns(tmp_path: Path):
    import pandas as pd

    raw_dir = tmp_path / "opssat_ad"
    raw_dir.mkdir()
    pd.DataFrame({"segment": [1, 2], "anomaly": [0, 1]}).to_csv(raw_dir / "dataset.csv", index=False)

    with pytest.raises(ValueError, match="missing expected columns"):
        load_opssat_dataset(raw_dir)


def test_split_by_author_train_flag_partitions_disjointly(synthetic_opssat_csv):
    df = load_opssat_dataset(synthetic_opssat_csv)
    train_df, test_df = split_by_author_train_flag(df)

    assert len(train_df) + len(test_df) == len(df)
    assert (train_df["train"] == 1).all()
    assert (test_df["train"] == 0).all()
