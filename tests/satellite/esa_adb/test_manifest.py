"""Manifest-correctness tests (category 14) — synthetic
ChannelPipelineResult/SequenceDataset stand-ins, no real download
required, plus a real-file hash sanity check."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.sequences import SequenceDataset
from src.satellite.esa_adb import clients as clients_mod
from src.satellite.esa_adb import loader, pipeline, preprocessing as prep
from src.satellite.esa_adb.pipeline import ChannelPipelineResult


def _make_seq_dataset(n: int, n_anomaly: int) -> SequenceDataset:
    sequences = np.zeros((n, prep.SEQUENCE_LENGTH, 1), dtype="float32")
    labels = np.array([1] * n_anomaly + [0] * (n - n_anomaly), dtype="int64")
    record_labels = [np.zeros(prep.SEQUENCE_LENGTH, dtype="int64") for _ in range(n)]
    return SequenceDataset(
        sequences=sequences, labels=labels, record_labels=record_labels,
        sequence_length=prep.SEQUENCE_LENGTH, stride=prep.STRIDE, label_strategy=prep.LABEL_STRATEGY,
    )


@pytest.fixture
def synthetic_channel_results() -> dict[str, ChannelPipelineResult]:
    results = {}
    for channel in ["channel_61", "channel_62", "channel_63"]:  # matches client_3's real channel list
        scaler = prep.fit_channel_scaler(channel, np.array([1.0, 2.0, 3.0]))
        results[channel] = ChannelPipelineResult(
            channel=channel, subsystem="subsystem_1", group="14", physical_unit="physical_unit_1",
            resampled_rows=100, scaler=scaler,
            split_frames={"train": pd.DataFrame(), "validation": pd.DataFrame(), "test": pd.DataFrame()},
            sequences={"train": _make_seq_dataset(10, 1), "validation": _make_seq_dataset(5, 0), "test": _make_seq_dataset(3, 1)},
        )
    return results


@pytest.fixture
def synthetic_client_datasets(synthetic_channel_results) -> dict[str, dict[str, SequenceDataset]]:
    client_datasets = {}
    for client, channel_list in clients_mod.CLIENT_CHANNELS.items():
        client_datasets[client] = {}
        for split in pipeline.SPLITS:
            if client == "client_3":
                per_channel = [synthetic_channel_results[c].sequences[split] for c in channel_list]
            else:
                per_channel = [_make_seq_dataset(0, 0)]
            client_datasets[client][split] = pipeline.pool_sequence_datasets(per_channel, prep.SEQUENCE_LENGTH, prep.STRIDE, prep.LABEL_STRATEGY)
    return client_datasets


def test_build_dataset_summary_has_expected_fields(synthetic_channel_results):
    summary = pipeline.build_dataset_summary(synthetic_channel_results)
    assert summary["num_channels"] == 3
    assert summary["resampled_records_total"] == 300
    assert summary["sequence"]["length"] == prep.SEQUENCE_LENGTH
    for split in pipeline.SPLITS:
        assert split in summary["splits"]
        assert "num_sequences" in summary["splits"][split]


def test_build_dataset_summary_anomaly_counts_match_pooled_data(synthetic_channel_results):
    summary = pipeline.build_dataset_summary(synthetic_channel_results)
    # 3 channels, each with 1 train anomaly -> 3 pooled train anomalies.
    assert summary["splits"]["train"]["anomaly"] == 3
    assert summary["splits"]["train"]["num_sequences"] == 30


def test_build_client_distribution_covers_all_eight_clients(synthetic_client_datasets):
    dist = pipeline.build_client_distribution(synthetic_client_datasets)
    assert set(dist.keys()) == set(clients_mod.CLIENT_CHANNELS.keys())
    assert dist["client_3"]["num_channels"] == 3
    assert dist["client_3"]["train"]["anomaly"] == 3


def test_build_preprocessing_summary_flags_no_constant_channels_for_varying_data(synthetic_channel_results):
    summary = pipeline.build_preprocessing_summary(synthetic_channel_results)
    assert summary["num_constant_channels"] == 0
    assert summary["constant_channels"] == []


def test_build_preprocessing_summary_flags_constant_channel():
    scaler = prep.fit_channel_scaler("channel_x", np.full(5, 3.0))
    result = ChannelPipelineResult(
        channel="channel_x", subsystem="subsystem_1", group="14", physical_unit="physical_unit_1",
        resampled_rows=5, scaler=scaler,
        split_frames={"train": pd.DataFrame(), "validation": pd.DataFrame(), "test": pd.DataFrame()},
        sequences={"train": _make_seq_dataset(0, 0), "validation": _make_seq_dataset(0, 0), "test": _make_seq_dataset(0, 0)},
    )
    summary = pipeline.build_preprocessing_summary({"channel_x": result})
    assert summary["constant_channels"] == ["channel_x"]


def test_sha256_of_file_is_deterministic_and_matches_known_algorithm(tmp_path):
    import hashlib

    p = tmp_path / "sample.csv"
    p.write_text("Channel,Subsystem\nchannel_1,subsystem_1\n")

    computed = pipeline.sha256_of_file(p)
    expected = hashlib.sha256(p.read_bytes()).hexdigest()
    assert computed == expected

    # Determinism: hashing the same unchanged file twice gives the same digest.
    assert pipeline.sha256_of_file(p) == computed


@pytest.mark.skipif(not loader.MISSION_DIR.exists(), reason="real ESA-ADB Mission 1 archive not downloaded")
def test_build_manifest_includes_source_file_hashes(synthetic_channel_results, synthetic_client_datasets):
    """build_manifest hashes the real source CSVs (loader.CHANNELS_CSV
    etc.), so this specific test needs the real archive even though the
    channel_results/client_datasets it's given are synthetic."""
    manifest = pipeline.build_manifest(synthetic_channel_results, synthetic_client_datasets, checks={}, seed=42)
    assert "source_file_sha256" in manifest
    assert len(manifest["source_file_sha256"]) == 3
    for digest in manifest["source_file_sha256"].values():
        assert len(digest) == 64  # sha256 hex length
