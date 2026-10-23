"""TON_IoT-specific preprocessing. Mirrors src/satellite/preprocessing.py's
role for OPSSAT-AD: stratified train/validation/test split, then
StandardScaler (numeric) + OneHotEncoder (categorical) fit on the TRAIN
split only, applied to all splits.

    full dataset (211,043 rows)
        -> stratified 70/15/15 train/validation/test split (label_binary)
        -> fit StandardScaler + OneHotEncoder on TRAIN only
        -> transform train/validation/test with the SAME fitted objects
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.satellite.ton_iot_loader import CATEGORICAL_FEATURE_COLUMNS, LABEL_COLUMN, NUMERIC_FEATURE_COLUMNS, TYPE_COLUMN


@dataclass
class TonIotPreprocessingArtifacts:
    scaler: StandardScaler
    encoder: OneHotEncoder
    feature_columns: list[str]


def three_way_split(df: pd.DataFrame, seed: int, val_size: float = 0.15, test_size: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[LABEL_COLUMN], shuffle=True)
    relative_val = val_size / (1.0 - test_size)
    train, val = train_test_split(train_val, test_size=relative_val, random_state=seed, stratify=train_val[LABEL_COLUMN], shuffle=True)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def fit_preprocessing(train_df: pd.DataFrame) -> TonIotPreprocessingArtifacts:
    scaler = StandardScaler()
    scaler.fit(train_df[NUMERIC_FEATURE_COLUMNS].to_numpy(dtype="float64"))

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(train_df[CATEGORICAL_FEATURE_COLUMNS].astype(str))

    cat_feature_names = list(encoder.get_feature_names_out(CATEGORICAL_FEATURE_COLUMNS))
    feature_columns = list(NUMERIC_FEATURE_COLUMNS) + cat_feature_names
    return TonIotPreprocessingArtifacts(scaler=scaler, encoder=encoder, feature_columns=feature_columns)


def transform(df: pd.DataFrame, artifacts: TonIotPreprocessingArtifacts) -> pd.DataFrame:
    numeric_scaled = artifacts.scaler.transform(df[NUMERIC_FEATURE_COLUMNS].to_numpy(dtype="float64"))
    categorical_encoded = artifacts.encoder.transform(df[CATEGORICAL_FEATURE_COLUMNS].astype(str))
    features = np.concatenate([numeric_scaled, categorical_encoded], axis=1).astype("float32")

    out = pd.DataFrame(features, columns=artifacts.feature_columns)
    out["label_binary"] = df[LABEL_COLUMN].to_numpy(dtype="float32")
    out["type"] = df[TYPE_COLUMN].to_numpy()
    return out
