"""Bot-IoT-specific preprocessing. Mirrors src/satellite/ton_iot_preprocessing.py.
All features are already numeric (see bot_iot_loader.py), so only a
stratified split + StandardScaler (fit on train only) are needed — no
one-hot encoding.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.satellite.bot_iot_loader import CATEGORY_COLUMN, LABEL_COLUMN, NUMERIC_FEATURE_COLUMNS


@dataclass
class BotIotPreprocessingArtifacts:
    scaler: StandardScaler
    feature_columns: list[str]


def three_way_split(df: pd.DataFrame, seed: int, val_size: float = 0.15, test_size: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[LABEL_COLUMN], shuffle=True)
    relative_val = val_size / (1.0 - test_size)
    train, val = train_test_split(train_val, test_size=relative_val, random_state=seed, stratify=train_val[LABEL_COLUMN], shuffle=True)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def fit_preprocessing(train_df: pd.DataFrame) -> BotIotPreprocessingArtifacts:
    scaler = StandardScaler()
    scaler.fit(train_df[NUMERIC_FEATURE_COLUMNS].to_numpy(dtype="float64"))
    return BotIotPreprocessingArtifacts(scaler=scaler, feature_columns=list(NUMERIC_FEATURE_COLUMNS))


def transform(df: pd.DataFrame, artifacts: BotIotPreprocessingArtifacts) -> pd.DataFrame:
    numeric_scaled = artifacts.scaler.transform(df[NUMERIC_FEATURE_COLUMNS].to_numpy(dtype="float64")).astype("float32")
    out = pd.DataFrame(numeric_scaled, columns=artifacts.feature_columns)
    out["label_binary"] = df[LABEL_COLUMN].to_numpy(dtype="float32")
    out["category"] = df[CATEGORY_COLUMN].to_numpy()
    return out
