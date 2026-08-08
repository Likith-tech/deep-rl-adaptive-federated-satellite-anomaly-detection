"""End-to-end NSL-KDD preprocessing pipeline.

    raw (KDDTrain+/KDDTest+)
        -> clean (duplicates, missing, infinite)
        -> label (label_binary / label_original)
        -> split (train / validation from KDDTrain+; test = KDDTest+)
        -> fit encoder + scaler on TRAIN ONLY
        -> transform all splits
        -> write data/processed/{train,validation,test}.parquet + metadata.json
        -> write preprocessing artifacts (encoder, scaler, feature list)

Every number in the resulting metadata.json is measured from the actual
run — nothing here is a placeholder or fabricated statistic.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import joblib
import pandas as pd

from src.data.loader import load_nsl_kdd, profile_dataframe
from src.data.schema import CATEGORICAL_COLUMNS, DIFFICULTY_COLUMN, LABEL_COLUMN, NUMERICAL_COLUMNS
from src.preprocessing.cleaning import clean_dataset
from src.preprocessing.encoding import apply_categorical_encoder, fit_categorical_encoder
from src.preprocessing.labels import add_binary_label
from src.preprocessing.scaling import apply_scaler, fit_scaler
from src.preprocessing.splitting import split_train_validation

REPO_ROOT = Path(__file__).resolve().parents[2]

NON_FEATURE_COLUMNS = [LABEL_COLUMN, DIFFICULTY_COLUMN, "label_original", "label_binary"]


def run_pipeline(
    raw_dir: str | Path = REPO_ROOT / "data" / "raw",
    processed_dir: str | Path = REPO_ROOT / "data" / "processed",
    interim_dir: str | Path = REPO_ROOT / "data" / "interim",
    artifacts_dir: str | Path = REPO_ROOT / "results" / "models" / "preprocessing",
    validation_size: float = 0.15,
    random_seed: int = 42,
    dataset_name: str = "NSL-KDD",
) -> dict:
    processed_dir = Path(processed_dir)
    interim_dir = Path(interim_dir)
    artifacts_dir = Path(artifacts_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    raw = load_nsl_kdd(raw_dir)
    raw_train_profile = profile_dataframe(raw.train, source_files=["KDDTrain+.txt"])
    raw_test_profile = profile_dataframe(raw.test, source_files=["KDDTest+.txt"])

    # 2. Clean (train and test cleaned identically/independently — no cross-split info used)
    clean_train, train_cleaning_report = clean_dataset(raw.train, NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS)
    clean_test, test_cleaning_report = clean_dataset(raw.test, NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS)

    # Interim = cleaned, labeled, but not yet split/encoded/scaled.
    clean_train = add_binary_label(clean_train, LABEL_COLUMN)
    clean_test = add_binary_label(clean_test, LABEL_COLUMN)
    clean_train.to_parquet(interim_dir / "kdd_train_cleaned.parquet", index=False)
    clean_test.to_parquet(interim_dir / "kdd_test_cleaned.parquet", index=False)

    # 3. Split KDDTrain+ into train/validation. KDDTest+ is the held-out test set as-is.
    train_split, validation_split = split_train_validation(
        clean_train, validation_size=validation_size, random_seed=random_seed
    )
    test_split = clean_test

    # 4. Fit encoder + scaler on TRAIN ONLY, then transform all splits.
    encoder = fit_categorical_encoder(train_split, CATEGORICAL_COLUMNS)
    scaler = fit_scaler(train_split, NUMERICAL_COLUMNS)

    def transform(df: pd.DataFrame) -> pd.DataFrame:
        df = apply_scaler(df, scaler, NUMERICAL_COLUMNS)
        df = apply_categorical_encoder(df, encoder, CATEGORICAL_COLUMNS)
        return df

    train_final = transform(train_split)
    validation_final = transform(validation_split)
    test_final = transform(test_split)

    feature_columns = [c for c in train_final.columns if c not in NON_FEATURE_COLUMNS]

    # 5. Persist processed splits
    train_final.to_parquet(processed_dir / "train.parquet", index=False)
    validation_final.to_parquet(processed_dir / "validation.parquet", index=False)
    test_final.to_parquet(processed_dir / "test.parquet", index=False)

    # 6. Persist preprocessing artifacts
    joblib.dump(scaler, artifacts_dir / "scaler.joblib")
    joblib.dump(encoder, artifacts_dir / "encoder.joblib")
    (artifacts_dir / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2))

    # 7. Metadata
    metadata = {
        "dataset_name": dataset_name,
        "dataset_version": "NSL-KDD (KDDTrain+/KDDTest+)",
        "num_samples": len(raw.train) + len(raw.test),
        "num_features": len(feature_columns),
        "label_column": "label_binary",
        "binary_labels": {"normal": 0, "anomaly": 1},
        "train_samples": len(train_final),
        "validation_samples": len(validation_final),
        "test_samples": len(test_final),
        "random_seed": random_seed,
        "categorical_columns_encoded": CATEGORICAL_COLUMNS,
        "numerical_columns_scaled": NUMERICAL_COLUMNS,
        "class_distribution": {
            "train": {str(k): int(v) for k, v in train_final["label_binary"].value_counts().items()},
            "validation": {str(k): int(v) for k, v in validation_final["label_binary"].value_counts().items()},
            "test": {str(k): int(v) for k, v in test_final["label_binary"].value_counts().items()},
        },
    }
    (processed_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return {
        "metadata": metadata,
        "raw_train_profile": raw_train_profile.to_dict(),
        "raw_test_profile": raw_test_profile.to_dict(),
        "train_cleaning_report": asdict(train_cleaning_report),
        "test_cleaning_report": asdict(test_cleaning_report),
    }


if __name__ == "__main__":
    result = run_pipeline()
    print(json.dumps(result["metadata"], indent=2))
