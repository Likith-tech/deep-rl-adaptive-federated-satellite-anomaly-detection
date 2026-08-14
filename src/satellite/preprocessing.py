"""Phase 9A — OPSSAT-AD-specific preprocessing.

Deliberately independent of src/preprocessing/ (Phase 1/3's NSL-KDD
pipeline) — no scaler, encoder, or sequence-construction logic is
shared, since OPSSAT-AD's schema, feature meaning, and label semantics
are entirely different (segment-level operational-anomaly labels vs.
NSL-KDD's per-record/per-sequence cyberattack labels). Phase 1/3's
files (src/preprocessing/sequences.py, results/models/preprocessing/)
are never imported or modified here.

    author's train segments (train=1)
        -> stratified split (this project's own, seeded) into
           train / validation (validation is carved from train ONLY —
           the author's test segments, train=0, are never touched here)
        -> fit StandardScaler + OneHotEncoder on the TRAIN portion only
        -> transform train, validation, and (later, once) test with the
           SAME fitted objects
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.satellite.loader import CATEGORICAL_COLUMN, LABEL_COLUMN, NUMERIC_FEATURE_COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "results" / "models" / "opssat_preprocessing"


@dataclass
class OpsSatPreprocessingArtifacts:
    scaler: StandardScaler
    encoder: OneHotEncoder
    numeric_columns: list[str]
    categorical_column: str
    feature_columns: list[str]  # final, ordered feature names after encoding

    def save(self, artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR) -> None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, artifacts_dir / "scaler.joblib")
        joblib.dump(self.encoder, artifacts_dir / "encoder.joblib")
        (artifacts_dir / "feature_columns.json").write_text(json.dumps(self.feature_columns, indent=2))
        (artifacts_dir / "metadata.json").write_text(json.dumps({
            "numeric_columns": self.numeric_columns,
            "categorical_column": self.categorical_column,
        }, indent=2))

    @classmethod
    def load(cls, artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR) -> "OpsSatPreprocessingArtifacts":
        scaler = joblib.load(artifacts_dir / "scaler.joblib")
        encoder = joblib.load(artifacts_dir / "encoder.joblib")
        feature_columns = json.loads((artifacts_dir / "feature_columns.json").read_text())
        metadata = json.loads((artifacts_dir / "metadata.json").read_text())
        return cls(
            scaler=scaler, encoder=encoder,
            numeric_columns=metadata["numeric_columns"],
            categorical_column=metadata["categorical_column"],
            feature_columns=feature_columns,
        )


def stratified_train_validation_split(
    train_df: pd.DataFrame, validation_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic stratified split of the AUTHOR'S training segments
    only (train_df must already be filtered to train=1 — see
    src/satellite/loader.py:split_by_author_train_flag). The author's
    test segments are never passed into this function."""
    if not 0.0 < validation_size < 1.0:
        raise ValueError(f"validation_size must be in (0, 1), got {validation_size}")

    train_split, val_split = train_test_split(
        train_df,
        test_size=validation_size,
        random_state=seed,
        stratify=train_df[LABEL_COLUMN],
        shuffle=True,
    )
    return train_split.reset_index(drop=True), val_split.reset_index(drop=True)


def fit_preprocessing(train_df: pd.DataFrame) -> OpsSatPreprocessingArtifacts:
    """Fit a StandardScaler (numeric columns) and OneHotEncoder (channel)
    on `train_df` ONLY. Never call this with validation or test data."""
    scaler = StandardScaler()
    scaler.fit(train_df[NUMERIC_FEATURE_COLUMNS].to_numpy(dtype="float64"))

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(train_df[[CATEGORICAL_COLUMN]])

    categorical_feature_names = [f"channel_{c}" for c in encoder.categories_[0]]
    feature_columns = list(NUMERIC_FEATURE_COLUMNS) + categorical_feature_names

    return OpsSatPreprocessingArtifacts(
        scaler=scaler,
        encoder=encoder,
        numeric_columns=list(NUMERIC_FEATURE_COLUMNS),
        categorical_column=CATEGORICAL_COLUMN,
        feature_columns=feature_columns,
    )


def transform(df: pd.DataFrame, artifacts: OpsSatPreprocessingArtifacts) -> np.ndarray:
    """Apply an ALREADY-FIT scaler/encoder to `df`. Never re-fits."""
    numeric_scaled = artifacts.scaler.transform(df[artifacts.numeric_columns].to_numpy(dtype="float64"))
    categorical_encoded = artifacts.encoder.transform(df[[artifacts.categorical_column]])
    return np.concatenate([numeric_scaled, categorical_encoded], axis=1).astype("float32")


def extract_labels(df: pd.DataFrame) -> np.ndarray:
    return df[LABEL_COLUMN].to_numpy(dtype="float32")
