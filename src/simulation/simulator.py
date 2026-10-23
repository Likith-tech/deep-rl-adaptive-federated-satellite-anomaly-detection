"""Orchestrates the simulated satellite environment: combines the
(pure, independently-testable) partitioner and network-conditions
generator into a full set of SatelliteClient objects, and handles the
I/O (writing per-client parquet files, manifest, metadata).

    Dataset (data/processed/train.parquet)
        |
        v
    Partition generator (src/simulation/partitioner.py)   -- WHO gets WHICH real rows
        |
        v
    Client datasets (data/partitions/SAT-xx/train.parquet)

    Simulation configuration (configs/satellite_simulation.yaml)
        |
        v
    Satellite metadata (src/simulation/network_conditions.py)  -- SIMULATED conditions
        |
        v
    SatelliteClient (client_id + data_size + simulated resource conditions)

Kept deliberately separate (per Phase 4 architecture requirement): which
rows a client gets and what simulated network conditions it has are
independent concerns, combined only here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.schema import ATTACK_CATEGORY_MAP
from src.simulation.network_conditions import generate_network_conditions
from src.simulation.partitioner import (
    ClientStats,
    PartitionResult,
    compute_category_series,
    compute_client_stats,
    dirichlet_partition,
    verify_partition_integrity,
)
from src.simulation.satellite_client import SatelliteClient, make_client_ids


@dataclass
class SatelliteEnvironment:
    clients: dict[str, SatelliteClient]
    partition_result: PartitionResult
    client_stats: dict[str, ClientStats]
    category_column: str
    category_map_applied: bool


def build_satellite_environment(train_df: pd.DataFrame, config: dict) -> SatelliteEnvironment:
    """Build the full simulated environment from the real training
    DataFrame and the satellite_simulation.yaml config dict (already
    parsed). Performs the partition, computes real per-client stats,
    generates simulated network conditions, and verifies integrity."""
    sat_cfg = config["satellites"]
    part_cfg = config["partitioning"]
    resource_cfg = config["resources"]

    client_ids = make_client_ids(sat_cfg["num_clients"], sat_cfg.get("id_prefix", "SAT"))

    category_map = ATTACK_CATEGORY_MAP if part_cfg["category_column"] == "label_original" else None

    partition_result = dirichlet_partition(
        df=train_df,
        client_ids=client_ids,
        category_column=part_cfg["category_column"],
        alpha=part_cfg["dirichlet_alpha"],
        seed=sat_cfg["seed"],
        category_map=category_map,
        constraints=part_cfg.get("constraints"),
        max_attempts=part_cfg.get("max_resample_attempts", 20),
    )

    verify_partition_integrity(train_df, partition_result.client_indices)

    category_series = compute_category_series(train_df, part_cfg["category_column"], category_map)
    client_stats = compute_client_stats(train_df, partition_result.client_indices, category_series)

    network_conditions = generate_network_conditions(client_ids, resource_cfg, seed=sat_cfg["seed"])

    clients = {
        cid: SatelliteClient(
            client_id=cid,
            data_size=client_stats[cid].total_samples,
            **network_conditions[cid],
        )
        for cid in client_ids
    }

    return SatelliteEnvironment(
        clients=clients,
        partition_result=partition_result,
        client_stats=client_stats,
        category_column=part_cfg["category_column"],
        category_map_applied=category_map is not None,
    )


def save_client_datasets(train_df: pd.DataFrame, env: SatelliteEnvironment, output_dir: Path) -> dict[str, Path]:
    """Write each client's real rows to data/partitions/<client_id>/train.parquet."""
    output_dir = Path(output_dir)
    paths = {}
    for client_id, indices in env.partition_result.client_indices.items():
        client_dir = output_dir / client_id
        client_dir.mkdir(parents=True, exist_ok=True)
        client_df = train_df.loc[indices].reset_index(drop=True)
        path = client_dir / "train.parquet"
        client_df.to_parquet(path, index=False)
        paths[client_id] = path
    return paths


def save_manifest(env: SatelliteEnvironment, source_dataset: str, output_path: Path) -> None:
    manifest = {
        "source_dataset": source_dataset,
        "num_clients": len(env.clients),
        "partition_strategy": "dirichlet",
        **env.partition_result.to_summary_dict(),
        "client_sample_counts": {cid: s.total_samples for cid, s in env.client_stats.items()},
        "client_stats": {cid: s.to_dict() for cid, s in env.client_stats.items()},
    }
    Path(output_path).write_text(json.dumps(manifest, indent=2))


def save_satellite_metadata(env: SatelliteEnvironment, output_path: Path) -> None:
    metadata = {
        "note": "All fields below except client_id and data_size are SIMULATED "
        "parameters, not real satellite measurements.",
        "clients": {cid: client.to_dict() for cid, client in env.clients.items()},
    }
    Path(output_path).write_text(json.dumps(metadata, indent=2))
