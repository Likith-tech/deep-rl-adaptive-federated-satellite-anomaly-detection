"""Phase 7 — smoke test: does the existing FedAvg trainer (Phase 6)
run correctly across MULTIPLE differently-partitioned (different
Dirichlet alpha) client sets? Uses tiny synthetic data — never the
real ~107k-row NSL-KDD data or the real 8-client partitions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.federated.trainer import run_federated_training
from src.simulation.partitioner import dirichlet_partition, verify_partition_integrity
from tests.federated.conftest import FEATURE_COLUMNS


def _make_synthetic_df(n_rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, len(FEATURE_COLUMNS)))
    y = (X[:, 0] + rng.normal(scale=0.5, size=n_rows) > 0).astype(int)
    df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    df["label_binary"] = y
    df["category"] = np.where(y == 1, "synthetic_attack", "normal")
    return df


@pytest.mark.parametrize("alpha", [0.1, 5.0])
def test_federated_training_runs_for_different_alpha_partitions(tmp_path, tiny_model_config, tiny_training_config, alpha):
    df = _make_synthetic_df(n_rows=300, seed=1)
    client_ids = ["SAT-A", "SAT-B", "SAT-C"]

    partition_result = dirichlet_partition(
        df=df, client_ids=client_ids, category_column="category", alpha=alpha, seed=42, max_attempts=20,
    )
    verify_partition_integrity(df, partition_result.client_indices)

    partitions_dir = tmp_path / "partitions"
    for cid, indices in partition_result.client_indices.items():
        client_dir = partitions_dir / cid
        client_dir.mkdir(parents=True)
        df.loc[indices].drop(columns=["category"]).reset_index(drop=True).to_parquet(
            client_dir / "train.parquet", index=False
        )

    global_val = tmp_path / "validation.parquet"
    _make_synthetic_df(n_rows=60, seed=2).drop(columns=["category"]).to_parquet(global_val, index=False)

    result = run_federated_training(
        client_ids=client_ids,
        partitions_dir=partitions_dir,
        global_validation_parquet=global_val,
        feature_columns=FEATURE_COLUMNS,
        model_config=tiny_model_config,
        training_config=tiny_training_config,
        num_rounds=2,
        shared_init_seed=42,
        checkpoint_dir=tmp_path / "checkpoints",
    )

    assert len(result.round_history) == 2
    for round_record in result.round_history:
        assert {u["client_id"] for u in round_record.client_updates} == set(client_ids)
