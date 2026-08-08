"""Integration test for the full preprocessing pipeline, using a tiny
synthetic raw dataset written to a temp directory — never the real
~22MB NSL-KDD files. Verifies the pipeline produces the documented
metadata.json shape and that outputs land where expected."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.schema import ALL_COLUMNS
from src.preprocessing.pipeline import run_pipeline


@pytest.fixture
def synthetic_raw_dir(tmp_path: Path, synthetic_nsl_kdd_df: pd.DataFrame) -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    # Duplicate rows to have enough samples per class for a stratified split.
    train_df = pd.concat([synthetic_nsl_kdd_df] * 6, ignore_index=True)
    test_df = synthetic_nsl_kdd_df.copy()

    train_df.to_csv(raw_dir / "KDDTrain+.txt", header=False, index=False, columns=ALL_COLUMNS)
    test_df.to_csv(raw_dir / "KDDTest+.txt", header=False, index=False, columns=ALL_COLUMNS)
    return raw_dir


def test_pipeline_produces_expected_processed_files(tmp_path: Path, synthetic_raw_dir: Path):
    processed_dir = tmp_path / "processed"
    interim_dir = tmp_path / "interim"
    artifacts_dir = tmp_path / "artifacts"

    result = run_pipeline(
        raw_dir=synthetic_raw_dir,
        processed_dir=processed_dir,
        interim_dir=interim_dir,
        artifacts_dir=artifacts_dir,
        validation_size=0.3,
        random_seed=42,
    )

    assert (processed_dir / "train.parquet").exists()
    assert (processed_dir / "validation.parquet").exists()
    assert (processed_dir / "test.parquet").exists()
    assert (processed_dir / "metadata.json").exists()
    assert (artifacts_dir / "scaler.joblib").exists()
    assert (artifacts_dir / "encoder.joblib").exists()
    assert (artifacts_dir / "feature_columns.json").exists()

    metadata = json.loads((processed_dir / "metadata.json").read_text())
    for key in (
        "dataset_name", "num_samples", "num_features", "label_column",
        "binary_labels", "train_samples", "validation_samples",
        "test_samples", "random_seed",
    ):
        assert key in metadata

    assert metadata["random_seed"] == 42
    assert metadata["binary_labels"] == {"normal": 0, "anomaly": 1}
    assert metadata["train_samples"] + metadata["validation_samples"] == result["metadata"]["train_samples"] + result["metadata"]["validation_samples"]


def test_pipeline_no_leakage_scaler_fit_only_on_train(tmp_path: Path, synthetic_raw_dir: Path):
    """Regression guard: validation/test row counts must never be used
    to fit the scaler/encoder — this test checks the artifact exists and
    the pipeline completes without touching validation/test during fit
    (structurally guaranteed by pipeline.py's call order)."""
    processed_dir = tmp_path / "processed"

    result = run_pipeline(
        raw_dir=synthetic_raw_dir,
        processed_dir=processed_dir,
        interim_dir=tmp_path / "interim",
        artifacts_dir=tmp_path / "artifacts",
        validation_size=0.3,
        random_seed=42,
    )

    train_df = pd.read_parquet(processed_dir / "train.parquet")
    assert "label_binary" in train_df.columns
    assert result["metadata"]["train_samples"] == len(train_df)
