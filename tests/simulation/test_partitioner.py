from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.simulation.partitioner import (
    ClientStats,
    _allocate_counts,
    compute_category_series,
    compute_client_stats,
    dirichlet_partition,
    verify_partition_integrity,
)

CLIENT_IDS = ["SAT-01", "SAT-02", "SAT-03", "SAT-04"]


def test_allocate_counts_sums_exactly_to_n():
    proportions = np.array([0.1, 0.2, 0.3, 0.4])
    counts = _allocate_counts(37, proportions)
    assert sum(counts) == 37
    assert len(counts) == 4


def test_allocate_counts_handles_zero_n():
    counts = _allocate_counts(0, np.array([0.5, 0.5]))
    assert counts == [0, 0]


def test_dirichlet_partition_assigns_every_sample_exactly_once(synthetic_labeled_df):
    result = dirichlet_partition(
        df=synthetic_labeled_df,
        client_ids=CLIENT_IDS,
        category_column="category",
        alpha=0.5,
        seed=42,
    )
    verify_partition_integrity(synthetic_labeled_df, result.client_indices)  # raises if violated
    total_assigned = sum(len(idx) for idx in result.client_indices.values())
    assert total_assigned == len(synthetic_labeled_df)


def test_dirichlet_partition_no_duplicate_assignment(synthetic_labeled_df):
    result = dirichlet_partition(
        df=synthetic_labeled_df, client_ids=CLIENT_IDS, category_column="category", alpha=0.5, seed=42
    )
    all_indices = np.concatenate(list(result.client_indices.values()))
    assert len(all_indices) == len(set(all_indices.tolist()))


def test_dirichlet_partition_deterministic_with_same_seed(synthetic_labeled_df):
    result_a = dirichlet_partition(
        df=synthetic_labeled_df, client_ids=CLIENT_IDS, category_column="category", alpha=0.5, seed=42
    )
    result_b = dirichlet_partition(
        df=synthetic_labeled_df, client_ids=CLIENT_IDS, category_column="category", alpha=0.5, seed=42
    )
    for cid in CLIENT_IDS:
        assert list(result_a.client_indices[cid]) == list(result_b.client_indices[cid])


def test_dirichlet_partition_different_seed_changes_partition(synthetic_labeled_df):
    result_a = dirichlet_partition(
        df=synthetic_labeled_df, client_ids=CLIENT_IDS, category_column="category", alpha=0.5, seed=42
    )
    result_b = dirichlet_partition(
        df=synthetic_labeled_df, client_ids=CLIENT_IDS, category_column="category", alpha=0.5, seed=7
    )
    assert any(
        list(result_a.client_indices[cid]) != list(result_b.client_indices[cid]) for cid in CLIENT_IDS
    )


def test_dirichlet_partition_no_test_or_validation_columns_needed(synthetic_labeled_df):
    # Guard against accidentally requiring/using any split-identifying
    # column other than what's passed in — the function must only ever
    # see the training DataFrame it's given.
    result = dirichlet_partition(
        df=synthetic_labeled_df, client_ids=CLIENT_IDS, category_column="category", alpha=0.5, seed=42
    )
    assert set(result.client_indices.keys()) == set(CLIENT_IDS)


def test_dirichlet_partition_respects_satisfiable_constraints(synthetic_labeled_df):
    constraints = {
        "min_total_samples": 50,
        "min_normal_samples": 5,
        "min_anomaly_samples": 5,
        "min_categories_represented": 2,
    }
    result = dirichlet_partition(
        df=synthetic_labeled_df,
        client_ids=CLIENT_IDS,
        category_column="category",
        alpha=1.0,
        seed=42,
        constraints=constraints,
        max_attempts=30,
    )
    assert result.constraints_satisfied is True
    assert result.constraint_violations == {}


def test_dirichlet_partition_reports_unsatisfiable_constraints_honestly(synthetic_labeled_df):
    impossible_constraints = {"min_total_samples": 10_000_000}
    result = dirichlet_partition(
        df=synthetic_labeled_df,
        client_ids=CLIENT_IDS,
        category_column="category",
        alpha=0.5,
        seed=42,
        constraints=impossible_constraints,
        max_attempts=3,
    )
    assert result.constraints_satisfied is False
    assert result.attempts_used == 3
    assert len(result.constraint_violations) > 0


def test_verify_partition_integrity_raises_on_duplicate():
    df = pd.DataFrame({"x": [0, 0, 0, 0]}, index=[0, 1, 2, 3])
    bad_indices = {"SAT-01": np.array([0, 1]), "SAT-02": np.array([1, 2, 3])}  # 1 duplicated
    with pytest.raises(AssertionError):
        verify_partition_integrity(df, bad_indices)


def test_verify_partition_integrity_raises_on_missing_sample():
    df = pd.DataFrame({"x": [0, 0, 0, 0]}, index=[0, 1, 2, 3])
    incomplete_indices = {"SAT-01": np.array([0, 1]), "SAT-02": np.array([2])}  # 3 missing
    with pytest.raises(AssertionError):
        verify_partition_integrity(df, incomplete_indices)


def test_compute_client_stats_matches_actual_data(synthetic_labeled_df):
    result = dirichlet_partition(
        df=synthetic_labeled_df, client_ids=CLIENT_IDS, category_column="category", alpha=0.5, seed=42
    )
    category_series = compute_category_series(synthetic_labeled_df, "category", None)
    stats = compute_client_stats(synthetic_labeled_df, result.client_indices, category_series)

    for cid in CLIENT_IDS:
        indices = result.client_indices[cid]
        expected_total = len(indices)
        expected_anomaly = int(synthetic_labeled_df.loc[indices, "label_binary"].sum())
        assert stats[cid].total_samples == expected_total
        assert stats[cid].anomaly_samples == expected_anomaly
        assert stats[cid].normal_samples == expected_total - expected_anomaly


def test_client_stats_anomaly_percentage_and_categories_represented():
    stats = ClientStats(
        client_id="SAT-01",
        total_samples=100,
        normal_samples=60,
        anomaly_samples=40,
        category_counts={"normal": 60, "dos": 30, "probe": 10, "r2l": 0, "u2r": 0},
    )
    assert stats.anomaly_percentage == 40.0
    assert stats.categories_represented == 3
