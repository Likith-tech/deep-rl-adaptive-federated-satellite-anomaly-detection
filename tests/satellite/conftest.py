"""Synthetic OPSSAT-AD-shaped fixtures — never the real downloaded
data/raw/opssat_ad/dataset.csv, so these tests run in milliseconds and
never depend on scripts/download_opssat_ad.py having been run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.satellite.loader import (
    CATEGORICAL_COLUMN,
    LABEL_COLUMN,
    NUMERIC_FEATURE_COLUMNS,
    SEGMENT_ID_COLUMN,
    SPLIT_COLUMN,
)

CHANNELS = ["CADC0872", "CADC0873", "CADC0888"]


def _make_synthetic_dataset(n_rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {SEGMENT_ID_COLUMN: np.arange(1, n_rows + 1)}
    for col in NUMERIC_FEATURE_COLUMNS:
        data[col] = rng.normal(loc=0.0, scale=1.0, size=n_rows)
    data[CATEGORICAL_COLUMN] = rng.choice(CHANNELS, size=n_rows)
    # Weakly separable label from one feature, so training has something to learn.
    data[LABEL_COLUMN] = (data["mean"] + rng.normal(scale=0.5, size=n_rows) > 0).astype(int)
    # 75% train / 25% test author split, roughly.
    data[SPLIT_COLUMN] = (rng.uniform(size=n_rows) < 0.75).astype(int)
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_opssat_csv(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "opssat_ad"
    raw_dir.mkdir()
    df = _make_synthetic_dataset(n_rows=200, seed=1)
    csv_path = raw_dir / "dataset.csv"
    df.to_csv(csv_path, index=False)
    return raw_dir


@pytest.fixture
def synthetic_opssat_df() -> pd.DataFrame:
    return _make_synthetic_dataset(n_rows=200, seed=1)
