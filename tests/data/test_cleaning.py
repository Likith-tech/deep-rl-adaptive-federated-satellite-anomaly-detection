from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing.cleaning import clean_dataset


def test_clean_dataset_removes_duplicates(synthetic_nsl_kdd_df, numerical_columns, categorical_columns):
    df_with_dupe = pd.concat([synthetic_nsl_kdd_df, synthetic_nsl_kdd_df.iloc[[0]]], ignore_index=True)

    cleaned, report = clean_dataset(df_with_dupe, numerical_columns, categorical_columns)

    assert report.duplicate_rows_found == 1
    assert report.duplicate_rows_removed == 1
    assert len(cleaned) == len(synthetic_nsl_kdd_df)


def test_clean_dataset_replaces_infinite_values(synthetic_nsl_kdd_df, numerical_columns, categorical_columns):
    df = synthetic_nsl_kdd_df.copy()
    df["src_bytes"] = df["src_bytes"].astype("float64")
    df.loc[0, "src_bytes"] = np.inf

    cleaned, report = clean_dataset(df, numerical_columns, categorical_columns)

    assert report.infinite_values_found == 1
    assert not np.isinf(cleaned["src_bytes"]).any()


def test_clean_dataset_fills_missing_numerical_with_median(synthetic_nsl_kdd_df, numerical_columns, categorical_columns):
    df = synthetic_nsl_kdd_df.copy()
    df.loc[0, "duration"] = None

    cleaned, report = clean_dataset(df, numerical_columns, categorical_columns)

    assert cleaned["duration"].isna().sum() == 0
    assert "duration" in report.missing_values_filled


def test_clean_dataset_reports_output_row_count(synthetic_nsl_kdd_df, numerical_columns, categorical_columns):
    cleaned, report = clean_dataset(synthetic_nsl_kdd_df, numerical_columns, categorical_columns)
    assert report.output_rows == len(cleaned) == len(synthetic_nsl_kdd_df)
