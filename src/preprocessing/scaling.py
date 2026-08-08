"""Numerical feature scaling.

The scaler is fit ONLY on the training split (never on validation/test,
and never on the full dataset before splitting) to avoid leakage:

    training data -> fit scaler -> transform train, validation, test
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit_scaler(train_df: pd.DataFrame, numerical_columns: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[numerical_columns])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler, numerical_columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    df[numerical_columns] = scaler.transform(df[numerical_columns])
    return df
