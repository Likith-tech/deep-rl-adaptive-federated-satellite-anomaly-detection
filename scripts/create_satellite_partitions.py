"""Generate the simulated satellite client environment from the real,
processed NSL-KDD training data.

Run from the repository root, after the Phase 1 pipeline has produced
data/processed/train.parquet:

    python scripts/create_satellite_partitions.py

ONLY the training split is partitioned — validation stays a single
global held-out set, and KDDTest+ is never touched (see
docs/project-progress/05-phase-4-satellite-simulation.md).

Writes:
    data/partitions/SAT-XX/train.parquet   (per client, gitignored)
    data/partitions/manifest.json           (small, committed)
    data/partitions/satellite_metadata.json (small, committed)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.simulation.simulator import (  # noqa: E402
    build_satellite_environment,
    save_client_datasets,
    save_manifest,
    save_satellite_metadata,
)

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PARTITIONS_DIR = REPO_ROOT / "data" / "partitions"
CONFIG_PATH = REPO_ROOT / "configs" / "satellite_simulation.yaml"


def main() -> None:
    train_path = PROCESSED_DIR / "train.parquet"
    if not train_path.exists():
        print(
            f"ERROR: {train_path} not found. Run the Phase 1 preprocessing pipeline first: "
            "python -m src.preprocessing.pipeline",
            file=sys.stderr,
        )
        sys.exit(1)

    config = yaml.safe_load(CONFIG_PATH.read_text())

    train_df = pd.read_parquet(train_path)
    print(f"Loaded training data: {len(train_df)} records "
          f"(this is the ONLY split partitioned — validation/test are untouched)")

    env = build_satellite_environment(train_df, config)

    print(f"\n=== Partition summary ({len(env.clients)} clients, "
          f"alpha={env.partition_result.alpha}, seed={env.partition_result.seed}) ===")
    print(f"Constraints satisfied: {env.partition_result.constraints_satisfied} "
          f"(attempts used: {env.partition_result.attempts_used})")
    if not env.partition_result.constraints_satisfied:
        print(f"Constraint violations: {env.partition_result.constraint_violations}", file=sys.stderr)

    for cid, stats in env.client_stats.items():
        print(f"  {cid}: {stats.total_samples} samples "
              f"({stats.normal_samples} normal / {stats.anomaly_samples} anomaly, "
              f"{stats.anomaly_percentage:.1f}% anomaly), "
              f"{stats.categories_represented} categories: {stats.category_counts}")

    total_assigned = sum(s.total_samples for s in env.client_stats.values())
    assert total_assigned == len(train_df), (
        f"Sample count mismatch: {total_assigned} assigned vs {len(train_df)} available"
    )
    print(f"\nData integrity check passed: {total_assigned} / {len(train_df)} training "
          f"samples assigned, no loss, no duplication (verified in build_satellite_environment).")

    PARTITIONS_DIR.mkdir(parents=True, exist_ok=True)
    paths = save_client_datasets(train_df, env, PARTITIONS_DIR)
    for cid, path in paths.items():
        print(f"Wrote {path} ({env.clients[cid].data_size} rows)")

    save_manifest(env, source_dataset="data/processed/train.parquet", output_path=PARTITIONS_DIR / "manifest.json")
    save_satellite_metadata(env, output_path=PARTITIONS_DIR / "satellite_metadata.json")
    print(f"\nWrote {PARTITIONS_DIR / 'manifest.json'}")
    print(f"Wrote {PARTITIONS_DIR / 'satellite_metadata.json'}")


if __name__ == "__main__":
    main()
