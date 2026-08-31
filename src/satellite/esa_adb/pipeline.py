"""ESA-ADB data-preparation pipeline orchestration.

Ties together loader -> labeling -> preprocessing -> clients into one
reproducible, deterministic run. Builds per-channel resampled/labeled/
normalized/sequenced data, pools it per FL client (per the real
subsystem/group partition in clients.py), runs leakage/integrity
checks, and returns everything needed to write the manifest and
inspection artifacts (src/satellite/esa_adb writes NO files itself
except via the explicit save_* helpers below — pure functions
otherwise, for testability).

DATA PREPARATION ONLY. No model is trained here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.sequences import SequenceDataset
from src.satellite.esa_adb import clients as clients_mod
from src.satellite.esa_adb import loader
from src.satellite.esa_adb import preprocessing as prep
from src.satellite.esa_adb.labeling import channel_anomaly_intervals, group_label_rows_by_channel

SPLITS = ("train", "validation", "test")

SPLIT_BOUNDS = {
    "train": (loader.TRAIN_START, loader.TRAIN_END),
    "validation": (loader.VALIDATION_START, loader.VALIDATION_END),
    "test": (loader.TEST_START, loader.TEST_END),
}


@dataclass
class ChannelPipelineResult:
    channel: str
    subsystem: str
    group: str
    physical_unit: str
    resampled_rows: int
    scaler: prep.ChannelScaler
    split_frames: dict[str, pd.DataFrame] = field(repr=False)  # split -> resampled+labeled df (raw, pre-normalization)
    sequences: dict[str, SequenceDataset] = field(repr=False)  # split -> per-channel SequenceDataset


def _split_resampled(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for split, (start, end) in SPLIT_BOUNDS.items():
        out[split] = df[(df.index >= start) & (df.index <= end)]
    return out


def build_channel_pipeline(channel: str, meta: loader.ChannelMeta, label_rows_by_channel: dict[str, list[dict[str, str]]]) -> ChannelPipelineResult:
    """Full per-channel pipeline: load real window -> resample+label ->
    split -> fit scaler on TRAIN ONLY -> normalize+sequence each split."""
    raw_series = loader.load_channel_window(channel)
    intervals = channel_anomaly_intervals(channel, label_rows_by_channel.get(channel, []))
    resampled = prep.resample_channel(raw_series, intervals)

    split_frames = _split_resampled(resampled)

    train_values = split_frames["train"][prep.VALUE_COLUMN].to_numpy(dtype="float64")
    scaler = prep.fit_channel_scaler(channel, train_values)

    sequences = {}
    for split in SPLITS:
        df = split_frames[split]
        if len(df) >= prep.SEQUENCE_LENGTH:
            sequences[split] = prep.build_channel_sequences(df, scaler)
        else:
            sequences[split] = SequenceDataset(
                sequences=np.empty((0, prep.SEQUENCE_LENGTH, 1), dtype="float32"),
                labels=np.empty((0,), dtype="int64"),
                record_labels=[],
                sequence_length=prep.SEQUENCE_LENGTH,
                stride=prep.STRIDE,
                label_strategy=prep.LABEL_STRATEGY,
            )

    return ChannelPipelineResult(
        channel=channel,
        subsystem=meta.subsystem,
        group=meta.group,
        physical_unit=meta.physical_unit,
        resampled_rows=len(resampled),
        scaler=scaler,
        split_frames=split_frames,
        sequences=sequences,
    )


def build_all_channels() -> dict[str, ChannelPipelineResult]:
    """Run build_channel_pipeline for all 58 approved Target=YES
    channels, deterministically ordered."""
    meta_by_channel = loader.load_channel_metadata()
    label_rows = loader.load_label_rows()
    label_rows_by_channel = group_label_rows_by_channel(label_rows)

    channels = loader.target_yes_channels()
    results = {}
    for channel in channels:
        results[channel] = build_channel_pipeline(channel, meta_by_channel[channel], label_rows_by_channel)
    return results


def pool_sequence_datasets(datasets: list[SequenceDataset], sequence_length: int, stride: int, label_strategy: str) -> SequenceDataset:
    """Concatenate multiple per-channel SequenceDatasets (same split)
    into one pooled dataset for a client. Order: channels in the order
    given (deterministic)."""
    if not datasets or all(d.num_sequences == 0 for d in datasets):
        return SequenceDataset(
            sequences=np.empty((0, sequence_length, 1), dtype="float32"),
            labels=np.empty((0,), dtype="int64"),
            record_labels=[],
            sequence_length=sequence_length,
            stride=stride,
            label_strategy=label_strategy,
        )
    sequences = np.concatenate([d.sequences for d in datasets if d.num_sequences > 0], axis=0)
    labels = np.concatenate([d.labels for d in datasets if d.num_sequences > 0], axis=0)
    record_labels = [rl for d in datasets for rl in d.record_labels]
    return SequenceDataset(
        sequences=sequences,
        labels=labels,
        record_labels=record_labels,
        sequence_length=sequence_length,
        stride=stride,
        label_strategy=label_strategy,
    )


def build_client_datasets(channel_results: dict[str, ChannelPipelineResult]) -> dict[str, dict[str, SequenceDataset]]:
    """client -> split -> pooled SequenceDataset, using the approved
    real subsystem/group partition (clients.CLIENT_CHANNELS)."""
    clients_mod.validate_partition(list(channel_results.keys()))

    client_datasets: dict[str, dict[str, SequenceDataset]] = {}
    for client, channel_list in clients_mod.CLIENT_CHANNELS.items():
        client_datasets[client] = {}
        for split in SPLITS:
            per_channel = [channel_results[c].sequences[split] for c in channel_list]
            client_datasets[client][split] = pool_sequence_datasets(
                per_channel, prep.SEQUENCE_LENGTH, prep.STRIDE, prep.LABEL_STRATEGY
            )
    return client_datasets


# ---------------------------------------------------------------------
# Leakage / integrity checks
# ---------------------------------------------------------------------


def check_no_anomaly_interval_straddles_split_boundary(channel_results: dict[str, ChannelPipelineResult], label_rows_by_channel: dict[str, list[dict[str, str]]]) -> list[str]:
    """Return a list of violation messages (empty == all clear). Checks,
    per channel and per real anomaly interval, whether the interval
    straddles the train/validation or validation/test boundary."""
    violations = []
    boundaries = [
        ("train/validation", loader.VALIDATION_START),
        ("validation/test", loader.TEST_START),
    ]
    for channel in channel_results:
        intervals = channel_anomaly_intervals(channel, label_rows_by_channel.get(channel, []))
        for start, end in intervals:
            for name, boundary in boundaries:
                if start < boundary <= end:
                    violations.append(f"{channel}: anomaly interval [{start}, {end}] straddles {name} boundary ({boundary})")
    return violations


def check_no_channel_duplication() -> list[str]:
    try:
        clients_mod.validate_partition(loader.target_yes_channels())
    except AssertionError as e:
        return [str(e)]
    return []


def check_split_boundaries_ordered_and_disjoint() -> list[str]:
    violations = []
    if not (loader.TRAIN_START < loader.TRAIN_END < loader.VALIDATION_START < loader.VALIDATION_END < loader.TEST_START < loader.TEST_END):
        violations.append("Split boundaries are not strictly increasing/disjoint.")
    return violations


def check_scalers_fit_train_only(channel_results: dict[str, ChannelPipelineResult]) -> list[str]:
    """Sanity check: each channel's fitted scaler mean/scale must be
    computable purely from that channel's train-split values (re-derive
    and compare) — guards against an accidental future-data leak."""
    violations = []
    for channel, result in channel_results.items():
        train_values = result.split_frames["train"][prep.VALUE_COLUMN].to_numpy(dtype="float64")
        recomputed = prep.fit_channel_scaler(channel, train_values)
        if not np.isclose(recomputed.mean, result.scaler.mean) or not np.isclose(recomputed.scale, result.scaler.scale):
            violations.append(f"{channel}: fitted scaler does not match a fresh fit on its own train split (possible leakage).")
    return violations


def check_no_duplicate_sequences_across_clients(client_datasets: dict[str, dict[str, SequenceDataset]]) -> list[str]:
    """Channels are disjoint by construction (validate_partition), so
    sequences cannot overlap across clients unless the same channel was
    (incorrectly) assigned twice. This check re-verifies at the channel
    level, which is sufficient given sequences are built per channel."""
    return check_no_channel_duplication()


def run_all_checks(channel_results: dict[str, ChannelPipelineResult], client_datasets: dict[str, dict[str, SequenceDataset]], label_rows_by_channel: dict[str, list[dict[str, str]]]) -> dict[str, list[str]]:
    return {
        "no_anomaly_interval_straddles_split_boundary": check_no_anomaly_interval_straddles_split_boundary(channel_results, label_rows_by_channel),
        "no_channel_duplication": check_no_channel_duplication(),
        "split_boundaries_ordered_and_disjoint": check_split_boundaries_ordered_and_disjoint(),
        "scalers_fit_train_only": check_scalers_fit_train_only(channel_results),
        "no_duplicate_sequences_across_clients": check_no_duplicate_sequences_across_clients(client_datasets),
    }


# ---------------------------------------------------------------------
# Manifest / artifact writing
# ---------------------------------------------------------------------

DEFAULT_PROCESSED_DIR = loader.REPO_ROOT / "data" / "processed" / "esa_adb"


def sha256_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Streamed SHA-256 (reads in 1MB chunks, never loads the whole file
    into memory) — used only on the small metadata CSVs (channels.csv,
    labels.csv, anomaly_types.csv), never on the multi-GB channel
    archives."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(channel_results: dict[str, ChannelPipelineResult], client_datasets: dict[str, dict[str, SequenceDataset]], checks: dict[str, list[str]], seed: int) -> dict:
    source_paths = [loader.CHANNELS_CSV, loader.LABELS_CSV, loader.ANOMALY_TYPES_CSV]
    return {
        "dataset": "ESA-ADB",
        "mission": 1,
        "source_files": [str(p.relative_to(loader.REPO_ROOT)) for p in source_paths],
        "source_file_sha256": {str(p.relative_to(loader.REPO_ROOT)): sha256_of_file(p) for p in source_paths},
        "primary_window": {"start": str(loader.WINDOW_START), "end": str(loader.WINDOW_END)},
        "split_ranges": {
            "train": {"start": str(loader.TRAIN_START), "end": str(loader.TRAIN_END)},
            "validation": {"start": str(loader.VALIDATION_START), "end": str(loader.VALIDATION_END)},
            "test": {"start": str(loader.TEST_START), "end": str(loader.TEST_END)},
        },
        "channels": {
            "count": len(channel_results),
            "selection_rule": "Target == YES (official ESA-ADB benchmark channels)",
            "list": sorted(channel_results.keys(), key=lambda c: int(c.split("_")[1])),
        },
        "resampling": {
            "freq": prep.RESAMPLE_FREQ,
            "method": "last-value-per-bin, forward-filled, leading unfilled bins dropped",
        },
        "sequence": {
            "length": prep.SEQUENCE_LENGTH,
            "stride": prep.STRIDE,
            "label_strategy": prep.LABEL_STRATEGY,
            "input_dim": 1,
        },
        "labeling_rule": "timestamp t labeled anomalous iff exists a labels.csv row with row.Channel==channel and row.StartTime <= t <= row.EndTime",
        "clients": {
            client: {
                "channels": chans,
                "subsystem_groups": clients_mod.CLIENT_SUBSYSTEM_GROUPS[client],
            }
            for client, chans in clients_mod.CLIENT_CHANNELS.items()
        },
        "preprocessing": {
            "normalization": "StandardScaler per channel, fit on TRAIN split only",
            "artifacts_dir": str(prep.DEFAULT_ARTIFACTS_DIR.relative_to(loader.REPO_ROOT)),
        },
        "seed": seed,
        "checks": {name: {"passed": len(v) == 0, "violations": v} for name, v in checks.items()},
    }


def build_client_metadata(channel_results: dict[str, ChannelPipelineResult], client_datasets: dict[str, dict[str, SequenceDataset]]) -> dict:
    out = {}
    for client, chans in clients_mod.CLIENT_CHANNELS.items():
        per_split = {}
        for split in SPLITS:
            ds = client_datasets[client][split]
            per_split[split] = {
                "num_sequences": ds.num_sequences,
                "normal": ds.normal_count,
                "anomaly": ds.anomaly_count,
                "anomaly_rate": ds.anomaly_rate,
            }
        out[client] = {
            "subsystem_groups": clients_mod.CLIENT_SUBSYSTEM_GROUPS[client],
            "channels": chans,
            "num_channels": len(chans),
            "splits": per_split,
        }
    return out


def build_preprocessing_manifest(channel_results: dict[str, ChannelPipelineResult]) -> dict:
    return {
        "resample_freq": prep.RESAMPLE_FREQ,
        "sequence_length": prep.SEQUENCE_LENGTH,
        "stride": prep.STRIDE,
        "label_strategy": prep.LABEL_STRATEGY,
        "channel_scalers": {
            c: {"mean": r.scaler.mean, "scale": r.scaler.scale, "is_constant": r.scaler.is_constant, "resampled_rows": r.resampled_rows}
            for c, r in channel_results.items()
        },
    }


def write_manifests(manifest: dict, client_metadata: dict, preprocessing_manifest: dict, out_dir: Path = DEFAULT_PROCESSED_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "client_metadata.json").write_text(json.dumps(client_metadata, indent=2))
    (out_dir / "preprocessing.json").write_text(json.dumps(preprocessing_manifest, indent=2))


# ---------------------------------------------------------------------
# Lightweight inspection artifacts (counts/distributions only)
# ---------------------------------------------------------------------

DEFAULT_INSPECTION_DIR = loader.REPO_ROOT / "results" / "reports" / "esa_adb"


def build_dataset_summary(channel_results: dict[str, ChannelPipelineResult]) -> dict:
    raw_rows_by_channel = {c: r.resampled_rows for c, r in channel_results.items()}
    pooled = {
        split: pool_sequence_datasets(
            [channel_results[c].sequences[split] for c in channel_results], prep.SEQUENCE_LENGTH, prep.STRIDE, prep.LABEL_STRATEGY
        )
        for split in SPLITS
    }
    return {
        "dataset": "ESA-ADB Mission 1",
        "window": {"start": str(loader.WINDOW_START), "end": str(loader.WINDOW_END)},
        "num_channels": len(channel_results),
        "resampled_records_total": sum(raw_rows_by_channel.values()),
        "sequence": {"length": prep.SEQUENCE_LENGTH, "stride": prep.STRIDE, "label_strategy": prep.LABEL_STRATEGY},
        "splits": {
            split: {
                "num_sequences": pooled[split].num_sequences,
                "normal": pooled[split].normal_count,
                "anomaly": pooled[split].anomaly_count,
                "anomaly_rate": pooled[split].anomaly_rate,
            }
            for split in SPLITS
        },
    }


def build_client_distribution(client_datasets: dict[str, dict[str, SequenceDataset]]) -> dict:
    distribution = {}
    for client in clients_mod.CLIENT_CHANNELS:
        entry = {
            "num_channels": len(clients_mod.CLIENT_CHANNELS[client]),
            "subsystem_groups": clients_mod.CLIENT_SUBSYSTEM_GROUPS[client],
        }
        for split in SPLITS:
            ds = client_datasets[client][split]
            entry[split] = {
                "num_sequences": ds.num_sequences,
                "normal": ds.normal_count,
                "anomaly": ds.anomaly_count,
                "anomaly_rate": ds.anomaly_rate,
            }
        distribution[client] = entry
    return distribution


def build_preprocessing_summary(channel_results: dict[str, ChannelPipelineResult]) -> dict:
    constant_channels = [c for c, r in channel_results.items() if r.scaler.is_constant]
    return {
        "resample_freq": prep.RESAMPLE_FREQ,
        "resample_method": "last-value-per-bin, forward-filled, leading unfilled bins dropped",
        "normalization": "StandardScaler per channel, fit on TRAIN split only",
        "num_channels": len(channel_results),
        "num_constant_channels": len(constant_channels),
        "constant_channels": constant_channels,
    }


def write_inspection_artifacts(channel_results: dict[str, ChannelPipelineResult], client_datasets: dict[str, dict[str, SequenceDataset]], out_dir: Path = DEFAULT_INSPECTION_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_summary.json").write_text(json.dumps(build_dataset_summary(channel_results), indent=2))
    (out_dir / "client_distribution.json").write_text(json.dumps(build_client_distribution(client_datasets), indent=2))
    (out_dir / "preprocessing_summary.json").write_text(json.dumps(build_preprocessing_summary(channel_results), indent=2))
