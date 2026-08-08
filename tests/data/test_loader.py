from __future__ import annotations

from pathlib import Path

import pytest

from src.data.loader import DatasetNotFoundError, load_nsl_kdd, profile_dataframe


def test_load_nsl_kdd_raises_clear_error_when_missing(tmp_path: Path):
    with pytest.raises(DatasetNotFoundError) as exc_info:
        load_nsl_kdd(raw_dir=tmp_path)

    message = str(exc_info.value)
    assert "KDDTrain+.txt" in message
    assert "KDDTest+.txt" in message


def test_profile_dataframe_reports_shape_and_label_distribution(synthetic_nsl_kdd_df):
    profile = profile_dataframe(synthetic_nsl_kdd_df, source_files=["synthetic"])

    assert profile.num_rows == 6
    assert profile.num_columns == len(synthetic_nsl_kdd_df.columns)
    assert profile.class_distribution["normal"] == 3
    assert profile.class_distribution["neptune"] == 2
    assert profile.class_distribution["smurf"] == 1
    assert profile.duplicate_rows == 0
    assert profile.infinite_value_count == 0


def test_profile_dataframe_identifies_categorical_columns(synthetic_nsl_kdd_df, categorical_columns):
    profile = profile_dataframe(synthetic_nsl_kdd_df, source_files=["synthetic"])
    assert set(profile.categorical_columns) == set(categorical_columns)
