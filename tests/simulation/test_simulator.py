from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.simulation.simulator import (
    build_satellite_environment,
    save_client_datasets,
    save_manifest,
    save_satellite_metadata,
)

TINY_CONFIG = {
    "satellites": {"num_clients": 4, "id_prefix": "SAT", "seed": 42},
    "partitioning": {
        "category_column": "category",
        "dirichlet_alpha": 1.0,
        "max_resample_attempts": 10,
        "constraints": {
            "min_total_samples": 10,
            "min_normal_samples": 1,
            "min_anomaly_samples": 1,
            "min_categories_represented": 1,
        },
    },
    "resources": {
        "bandwidth_mbps": [5, 100],
        "latency_ms": [50, 500],
        "compute_score": [0.2, 1.0],
        "availability_probability": [0.6, 1.0],
        "connectivity_quality": [0.3, 1.0],
    },
}


def test_build_satellite_environment_creates_expected_client_count(synthetic_labeled_df):
    env = build_satellite_environment(synthetic_labeled_df, TINY_CONFIG)
    assert len(env.clients) == 4
    assert set(env.clients.keys()) == {"SAT-01", "SAT-02", "SAT-03", "SAT-04"}


def test_build_satellite_environment_data_sizes_sum_to_total(synthetic_labeled_df):
    env = build_satellite_environment(synthetic_labeled_df, TINY_CONFIG)
    total_assigned = sum(client.data_size for client in env.clients.values())
    assert total_assigned == len(synthetic_labeled_df)


def test_build_satellite_environment_client_data_size_matches_stats(synthetic_labeled_df):
    env = build_satellite_environment(synthetic_labeled_df, TINY_CONFIG)
    for cid, client in env.clients.items():
        assert client.data_size == env.client_stats[cid].total_samples


def test_build_satellite_environment_resource_values_within_bounds(synthetic_labeled_df):
    env = build_satellite_environment(synthetic_labeled_df, TINY_CONFIG)
    for client in env.clients.values():
        assert 5 <= client.bandwidth_mbps <= 100
        assert 50 <= client.latency_ms <= 500
        assert 0.2 <= client.compute_score <= 1.0
        assert 0.6 <= client.availability_probability <= 1.0
        assert 0.3 <= client.connectivity_quality <= 1.0


def test_save_client_datasets_writes_correct_row_counts(tmp_path: Path, synthetic_labeled_df):
    env = build_satellite_environment(synthetic_labeled_df, TINY_CONFIG)
    paths = save_client_datasets(synthetic_labeled_df, env, tmp_path)

    assert set(paths.keys()) == set(env.clients.keys())
    for cid, path in paths.items():
        assert path.exists()
        client_df = pd.read_parquet(path)
        assert len(client_df) == env.clients[cid].data_size


def test_save_manifest_contains_correct_client_count_and_totals(tmp_path: Path, synthetic_labeled_df):
    env = build_satellite_environment(synthetic_labeled_df, TINY_CONFIG)
    manifest_path = tmp_path / "manifest.json"
    save_manifest(env, source_dataset="synthetic_test", output_path=manifest_path)

    manifest = json.loads(manifest_path.read_text())
    assert manifest["num_clients"] == 4
    assert sum(manifest["client_sample_counts"].values()) == len(synthetic_labeled_df)
    assert manifest["partition_strategy"] == "dirichlet"
    assert manifest["seed"] == 42


def test_save_satellite_metadata_marks_values_as_simulated(tmp_path: Path, synthetic_labeled_df):
    env = build_satellite_environment(synthetic_labeled_df, TINY_CONFIG)
    metadata_path = tmp_path / "satellite_metadata.json"
    save_satellite_metadata(env, output_path=metadata_path)

    metadata = json.loads(metadata_path.read_text())
    assert "simulated" in metadata["note"].lower()
    assert len(metadata["clients"]) == 4
    for client_data in metadata["clients"].values():
        assert "bandwidth_mbps" in client_data
        assert "client_id" in client_data
