"""Data cleaning steps: missing values, infinite values, duplicates.

Each function reports what it changed rather than silently mutating data,
so the pipeline can produce an honest data-quality report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CleaningReport:
    input_rows: int
    duplicate_rows_found: int
    duplicate_rows_removed: int
    missing_values_found: dict[str, int] = field(default_factory=dict)
    missing_values_filled: dict[str, int] = field(default_factory=dict)
    infinite_values_found: int = 0
    infinite_values_replaced: int = 0
    output_rows: int = 0

    def to_dict(self) -> dict:
        return {
            "input_rows": self.input_rows,
            "duplicate_rows_found": self.duplicate_rows_found,
            "duplicate_rows_removed": self.duplicate_rows_removed,
            "missing_values_found": self.missing_values_found,
            "missing_values_filled": self.missing_values_filled,
            "infinite_values_found": self.infinite_values_found,
            "infinite_values_replaced": self.infinite_values_replaced,
            "output_rows": self.output_rows,
        }


def clean_dataset(
    df: pd.DataFrame,
    numerical_columns: list[str],
    categorical_columns: list[str],
    drop_duplicates: bool = True,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean `df` in place-safe fashion (returns a new DataFrame) and report what changed.

    Strategy (explicit, not silent):
    - Duplicate rows: measured, then dropped (kept if drop_duplicates=False).
    - Infinite values in numerical columns: replaced with NaN, then imputed
      with the column median (robust to outliers/attack spikes).
    - Missing numerical values: imputed with column median.
    - Missing categorical values: imputed with the literal string "unknown"
      rather than dropped, to avoid losing rows.
    """
    df = df.copy()
    input_rows = len(df)

    duplicate_count = int(df.duplicated().sum())
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)

    missing_found = {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().any()}

    numeric_subset = df[numerical_columns]
    infinite_mask = np.isinf(numeric_subset.to_numpy(dtype="float64", na_value=0.0))
    infinite_count = int(infinite_mask.sum())
    if infinite_count:
        df[numerical_columns] = numeric_subset.replace([np.inf, -np.inf], np.nan)

    missing_filled: dict[str, int] = {}
    for col in numerical_columns:
        n_missing = int(df[col].isna().sum())
        if n_missing:
            median = df[col].median()
            df[col] = df[col].fillna(median)
            missing_filled[col] = n_missing

    for col in categorical_columns:
        n_missing = int(df[col].isna().sum())
        if n_missing:
            df[col] = df[col].fillna("unknown")
            missing_filled[col] = n_missing

    report = CleaningReport(
        input_rows=input_rows,
        duplicate_rows_found=duplicate_count,
        duplicate_rows_removed=duplicate_count if drop_duplicates else 0,
        missing_values_found=missing_found,
        missing_values_filled=missing_filled,
        infinite_values_found=infinite_count,
        infinite_values_replaced=infinite_count,
        output_rows=len(df),
    )
    return df, report
