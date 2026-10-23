from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FEATURE_COLUMNS = [f"f{i}" for i in range(6)]


def _make_synthetic_parquet(path: Path, n_rows: int, seed: int, anomaly_bias: float = 0.0) -> None:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, len(FEATURE_COLUMNS)))
    y = (X[:, 0] + anomaly_bias + rng.normal(scale=0.5, size=n_rows) > 0).astype(int)
    df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    df["label_binary"] = y
    df["label_original"] = np.where(y == 1, "synthetic_attack", "normal")
    df["attack"] = df["label_original"]
    df["difficulty"] = 10
    df.to_parquet(path, index=False)


@pytest.fixture
def tiny_model_config() -> dict:
    return {
        "input_dim": len(FEATURE_COLUMNS),
        "hidden_dimensions": [8, 4],
        "dropout": 0.1,
        "output_dim": 1,
    }


@pytest.fixture
def tiny_training_config() -> dict:
    return {
        "local_epochs": 1,
        "batch_size": 32,
        "learning_rate": 0.01,
        "optimizer": "adam",
        "loss": "bce_with_logits",
        "seed": 42,
    }


@pytest.fixture
def three_clients_and_global_val(tmp_path: Path) -> dict:
    """Three distinctly-seeded 'client' partitions plus one shared
    global validation set, mirroring SAT-XX / validation.parquet."""
    paths = {}
    specs = [("SAT-A", 120, 1, 0.0), ("SAT-B", 60, 2, 1.0), ("SAT-C", 30, 3, -1.0)]
    for client_id, n_rows, seed, bias in specs:
        p = tmp_path / client_id / "train.parquet"
        p.parent.mkdir(parents=True)
        _make_synthetic_parquet(p, n_rows=n_rows, seed=seed, anomaly_bias=bias)
        paths[client_id] = p

    global_val = tmp_path / "validation.parquet"
    _make_synthetic_parquet(global_val, n_rows=80, seed=99)
    paths["global_val"] = global_val
    paths["partitions_dir"] = tmp_path
    paths["client_ids"] = ["SAT-A", "SAT-B", "SAT-C"]
    return paths
