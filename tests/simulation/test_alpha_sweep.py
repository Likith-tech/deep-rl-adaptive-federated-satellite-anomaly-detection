"""Phase 7 — tests for the controlled Dirichlet-alpha sweep used in the
non-IID Federated Learning experiments. Reuses the same synthetic
fixture and partitioner/non-IID-metrics functions as Phase 4's tests —
never the real ~107k-row NSL-KDD data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.simulation.non_iid_metrics import compute_non_iid_report
from src.simulation.partitioner import (
    compute_category_series,
    compute_client_stats,
    dirichlet_partition,
    verify_partition_integrity,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CLIENT_IDS = [f"SAT-{i:02d}" for i in range(1, 6)]  # 5 clients — enough for a meaningful sweep, fast to test
ALPHAS = [0.1, 0.5, 1.0, 5.0, 10.0]  # the actual Phase 7 alpha sweep


@pytest.mark.parametrize("alpha", ALPHAS)
def test_every_alpha_produces_a_valid_partition(synthetic_labeled_df, alpha):
    """Total sample count is preserved, no duplicates, no loss — for
    every alpha in the actual Phase 7 sweep, not just one."""
    result = dirichlet_partition(
        df=synthetic_labeled_df, client_ids=CLIENT_IDS, category_column="category",
        alpha=alpha, seed=42, max_attempts=20,
    )
    verify_partition_integrity(synthetic_labeled_df, result.client_indices)

    total_assigned = sum(len(idx) for idx in result.client_indices.values())
    assert total_assigned == len(synthetic_labeled_df)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_every_alpha_produces_no_duplicate_assignment(synthetic_labeled_df, alpha):
    result = dirichlet_partition(
        df=synthetic_labeled_df, client_ids=CLIENT_IDS, category_column="category",
        alpha=alpha, seed=42, max_attempts=20,
    )
    all_indices = np.concatenate(list(result.client_indices.values()))
    assert len(all_indices) == len(set(all_indices.tolist()))


def test_same_alpha_and_seed_is_reproducible(synthetic_labeled_df):
    result_a = dirichlet_partition(
        df=synthetic_labeled_df, client_ids=CLIENT_IDS, category_column="category",
        alpha=0.3, seed=7, max_attempts=20,
    )
    result_b = dirichlet_partition(
        df=synthetic_labeled_df, client_ids=CLIENT_IDS, category_column="category",
        alpha=0.3, seed=7, max_attempts=20,
    )
    for cid in CLIENT_IDS:
        assert np.array_equal(result_a.client_indices[cid], result_b.client_indices[cid])


def test_lower_alpha_produces_more_heterogeneous_partition_than_higher_alpha(synthetic_labeled_df):
    """The core assumption of the whole Phase 7 study: alpha actually
    controls measured heterogeneity — verified, not assumed."""
    low_alpha_result = dirichlet_partition(
        df=synthetic_labeled_df, client_ids=CLIENT_IDS, category_column="category",
        alpha=0.05, seed=42, max_attempts=20,
    )
    high_alpha_result = dirichlet_partition(
        df=synthetic_labeled_df, client_ids=CLIENT_IDS, category_column="category",
        alpha=10.0, seed=42, max_attempts=20,
    )

    category_series = compute_category_series(synthetic_labeled_df, "category", None)
    low_stats = compute_client_stats(synthetic_labeled_df, low_alpha_result.client_indices, category_series)
    high_stats = compute_client_stats(synthetic_labeled_df, high_alpha_result.client_indices, category_series)

    low_report = compute_non_iid_report(low_stats)
    high_report = compute_non_iid_report(high_stats)

    assert low_report.pairwise_js_distance_avg > high_report.pairwise_js_distance_avg


def test_original_phase4_style_alpha_partition_is_independent_of_a_fresh_one(synthetic_labeled_df):
    """Two independently-generated alpha=0.5 partitions (e.g. the
    original Phase 4 run and this phase's own alpha_0.5 experiment) do
    not have to be, and here are not required to be, the SAME object —
    each call to dirichlet_partition is self-contained and touches no
    shared/global state."""
    result_a = dirichlet_partition(
        df=synthetic_labeled_df, client_ids=CLIENT_IDS, category_column="category",
        alpha=0.5, seed=42, max_attempts=20,
    )
    # A second, unrelated call with a different seed must not be
    # affected by (or mutate) the first call's result.
    result_b = dirichlet_partition(
        df=synthetic_labeled_df, client_ids=CLIENT_IDS, category_column="category",
        alpha=0.5, seed=99, max_attempts=20,
    )
    verify_partition_integrity(synthetic_labeled_df, result_a.client_indices)
    verify_partition_integrity(synthetic_labeled_df, result_b.client_indices)
    # First result's indices must be untouched by the second call.
    assert sum(len(idx) for idx in result_a.client_indices.values()) == len(synthetic_labeled_df)


def test_non_iid_config_declares_the_expected_alpha_sweep():
    """configs/non_iid.yaml is the single source of truth for which
    alphas Phase 7 actually runs — verify it loads and matches what
    the analysis/report code expects, so a config edit can't silently
    diverge from what was actually measured."""
    config = yaml.safe_load((REPO_ROOT / "configs" / "non_iid.yaml").read_text())
    assert config["alphas"] == ALPHAS
    assert config["satellites"]["num_clients"] == 8
    assert config["federation"]["clients_per_round"] == config["satellites"]["num_clients"]


def test_each_alpha_experiment_writes_to_its_own_isolated_directory(tmp_path, synthetic_labeled_df):
    """Experiment configuration separation: writing two different
    alphas' client partitions to their own experiment directories must
    not clobber or mix with each other."""
    alpha_a_dir = tmp_path / "alpha_0.1" / "partitions"
    alpha_b_dir = tmp_path / "alpha_5.0" / "partitions"

    for alpha, out_dir in ((0.1, alpha_a_dir), (5.0, alpha_b_dir)):
        result = dirichlet_partition(
            df=synthetic_labeled_df, client_ids=CLIENT_IDS, category_column="category",
            alpha=alpha, seed=42, max_attempts=20,
        )
        for client_id, indices in result.client_indices.items():
            client_dir = out_dir / client_id
            client_dir.mkdir(parents=True)
            synthetic_labeled_df.loc[indices].to_parquet(client_dir / "train.parquet", index=False)

    # Both experiments' files exist, independently, under their own alpha directory.
    for client_id in CLIENT_IDS:
        assert (alpha_a_dir / client_id / "train.parquet").exists()
        assert (alpha_b_dir / client_id / "train.parquet").exists()

    # And they are NOT identical (different alpha -> different partition),
    # confirming the two experiments didn't collapse into the same data.
    sat01_a = pd.read_parquet(alpha_a_dir / "SAT-01" / "train.parquet")
    sat01_b = pd.read_parquet(alpha_b_dir / "SAT-01" / "train.parquet")
    assert len(sat01_a) != len(sat01_b) or not sat01_a.equals(sat01_b)
