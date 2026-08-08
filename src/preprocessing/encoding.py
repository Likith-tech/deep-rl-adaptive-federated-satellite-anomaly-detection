"""Categorical feature encoding.

One-hot encoding is used for NSL-KDD's three nominal categorical columns
(`protocol_type`, `service`, `flag`) — none are ordinal, and cardinality
is low enough (3 / ~70 / ~11 unique values) that one-hot is practical.

The encoder is fit ONLY on the training split, then reused to transform
validation/test to guarantee train/test consistency and prevent unseen
categories from silently producing different columns per split.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def fit_categorical_encoder(train_df: pd.DataFrame, categorical_columns: list[str]) -> OneHotEncoder:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype="float64")
    encoder.fit(train_df[categorical_columns])
    return encoder


def apply_categorical_encoder(
    df: pd.DataFrame, encoder: OneHotEncoder, categorical_columns: list[str]
) -> pd.DataFrame:
    encoded = encoder.transform(df[categorical_columns])
    encoded_columns = encoder.get_feature_names_out(categorical_columns)
    encoded_df = pd.DataFrame(encoded, columns=encoded_columns, index=df.index)

    remainder = df.drop(columns=categorical_columns)
    return pd.concat([remainder, encoded_df], axis=1)
