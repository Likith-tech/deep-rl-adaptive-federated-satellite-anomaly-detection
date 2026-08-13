"""Phase 7 — controlled non-IID Federated Learning experiments.

Run from the repository root (requires data/processed/train.parquet
from the Phase 1 pipeline):

    python scripts/run_non_iid_experiments.py

For each Dirichlet alpha in configs/non_iid.yaml, this script:
    1. builds a FRESH 8-client partition of the real training data
       (never touching data/partitions/, the original Phase 4 output)
    2. verifies partition integrity (no loss/duplication)
    3. measures non-IID-ness (JS distance, entropy, per-category variance)
    4. writes each client's local parquet under
       experiments/non_iid/alpha_<X>/partitions/<client>/train.parquet
    5. runs the SAME FedAvg algorithm as Phase 6 (src/federated/trainer.py)
       for the SAME number of rounds, selecting the best round by
       GLOBAL VALIDATION loss only (KDDTest+ is never touched here)

Writes, per alpha, under experiments/non_iid/alpha_<X>/:
    partition_metadata.json   (partition stats + non-IID report)
    run_config.json           (experiment config + best round summary)
    round_history.json        (round-by-round FedAvg metrics, written by the trainer)
    best_global_model.pt, round_checkpoints/round_NN.pt   (gitignored)

The original Phase 4 partition (data/partitions/) and Phase 6 baseline
(results/models/federated/) are never modified by this script.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.schema import ATTACK_CATEGORY_MAP  # noqa: E402
from src.federated.trainer import run_federated_training  # noqa: E402
from src.simulation.non_iid_metrics import compute_non_iid_report  # noqa: E402
from src.simulation.partitioner import (  # noqa: E402
    compute_category_series,
    compute_client_stats,
    dirichlet_partition,
    verify_partition_integrity,
)
from src.simulation.satellite_client import make_client_ids  # noqa: E402

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
ARTIFACTS_DIR = REPO_ROOT / "results" / "models" / "preprocessing"
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "non_iid"
CONFIG_PATH = REPO_ROOT / "configs" / "non_iid.yaml"


def build_partition_for_alpha(train_df: pd.DataFrame, client_ids: list[str], alpha: float, config: dict) -> dict:
    part_cfg = config["partitioning"]
    seed = config["satellites"]["seed"]

    partition_result = dirichlet_partition(
        df=train_df,
        client_ids=client_ids,
        category_column=part_cfg["category_column"],
        alpha=alpha,
        seed=seed,  # SAME seed for every alpha — isolates alpha as the only variable
        category_map=ATTACK_CATEGORY_MAP if part_cfg["category_column"] == "label_original" else None,
        constraints=part_cfg["constraints"],
        max_attempts=part_cfg["max_resample_attempts"],
    )
    verify_partition_integrity(train_df, partition_result.client_indices)

    category_series = compute_category_series(
        train_df, part_cfg["category_column"],
        ATTACK_CATEGORY_MAP if part_cfg["category_column"] == "label_original" else None,
    )
    client_stats = compute_client_stats(train_df, partition_result.client_indices, category_series)

    total_assigned = sum(s.total_samples for s in client_stats.values())
    assert total_assigned == len(train_df), (
        f"alpha={alpha}: sample count mismatch: {total_assigned} assigned vs {len(train_df)} available"
    )

    non_iid_report = compute_non_iid_report(client_stats)

    return {
        "partition_result": partition_result,
        "client_stats": client_stats,
        "non_iid_report": non_iid_report,
        "total_assigned": total_assigned,
    }


def save_client_partitions(train_df: pd.DataFrame, partition_result, output_dir: Path) -> None:
    for client_id, indices in partition_result.client_indices.items():
        client_dir = output_dir / client_id
        client_dir.mkdir(parents=True, exist_ok=True)
        train_df.loc[indices].reset_index(drop=True).to_parquet(client_dir / "train.parquet", index=False)


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    alphas = config["alphas"]

    train_path = PROCESSED_DIR / "train.parquet"
    if not train_path.exists():
        print(f"ERROR: {train_path} not found. Run the Phase 1 pipeline first.", file=sys.stderr)
        sys.exit(1)
    train_df = pd.read_parquet(train_path)
    print(f"Loaded training data: {len(train_df)} records (source for every alpha experiment)")

    feature_columns = json.loads((ARTIFACTS_DIR / "feature_columns.json").read_text())
    global_validation_parquet = PROCESSED_DIR / "validation.parquet"

    client_ids = make_client_ids(config["satellites"]["num_clients"], config["satellites"].get("id_prefix", "SAT"))
    model_config = config["model"]
    training_config = config["training"]
    num_rounds = config["federation"]["num_rounds"]
    shared_init_seed = config["initialization"]["shared_init_seed"]

    for alpha in alphas:
        alpha_dir = EXPERIMENT_ROOT / f"alpha_{alpha}"
        partitions_dir = alpha_dir / "partitions"
        print(f"\n{'=' * 70}\nAlpha = {alpha}\n{'=' * 70}")

        start = time.time()
        build_result = build_partition_for_alpha(train_df, client_ids, alpha, config)
        partition_result = build_result["partition_result"]
        client_stats = build_result["client_stats"]
        non_iid_report = build_result["non_iid_report"]

        print(f"Partition built (attempts_used={partition_result.attempts_used}, "
              f"constraints_satisfied={partition_result.constraints_satisfied})")
        if not partition_result.constraints_satisfied:
            print(f"  Constraint violations: {partition_result.constraint_violations}", file=sys.stderr)
        for cid, stats in client_stats.items():
            print(f"  {cid}: {stats.total_samples} samples, {stats.anomaly_percentage:.1f}% anomaly, "
                  f"{stats.categories_represented} categories: {stats.category_counts}")
        print(f"Non-IID: mean_js={non_iid_report.pairwise_js_distance_avg:.4f} "
              f"min_js={non_iid_report.pairwise_js_distance_min:.4f} "
              f"max_js={non_iid_report.pairwise_js_distance_max:.4f}")

        save_client_partitions(train_df, partition_result, partitions_dir)

        partition_metadata = {
            "alpha": alpha,
            "seed": config["satellites"]["seed"],
            "num_clients": len(client_ids),
            "total_samples": build_result["total_assigned"],
            "client_sample_counts": {cid: s.total_samples for cid, s in client_stats.items()},
            "client_stats": {cid: s.to_dict() for cid, s in client_stats.items()},
            "partition_result": partition_result.to_summary_dict(),
            "non_iid_report": non_iid_report.to_dict(),
        }
        alpha_dir.mkdir(parents=True, exist_ok=True)
        (alpha_dir / "partition_metadata.json").write_text(json.dumps(partition_metadata, indent=2))

        print(f"Running FedAvg: {len(client_ids)} clients, {num_rounds} rounds, "
              f"local_epochs={training_config['local_epochs']}")
        fed_result = run_federated_training(
            client_ids=client_ids,
            partitions_dir=partitions_dir,
            global_validation_parquet=global_validation_parquet,
            feature_columns=feature_columns,
            model_config=model_config,
            training_config=training_config,
            num_rounds=num_rounds,
            shared_init_seed=shared_init_seed,
            checkpoint_dir=alpha_dir,
        )

        print(f"Best round: {fed_result.best_round} (val_loss={fed_result.best_val_loss:.4f}, "
              f"val_f1={fed_result.best_val_metrics.f1:.4f})")

        run_config = {
            "alpha": alpha,
            "config": config,
            "client_ids": client_ids,
            "num_rounds": num_rounds,
            "best_round": fed_result.best_round,
            "best_val_loss": fed_result.best_val_loss,
            "best_val_metrics": fed_result.best_val_metrics.to_dict(),
            "total_seconds": fed_result.total_seconds,
            "alpha_wall_seconds": time.time() - start,
        }
        (alpha_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
        print(f"Alpha={alpha} complete in {time.time() - start:.1f}s. Artifacts under {alpha_dir}")

    print(f"\nAll {len(alphas)} alpha experiments complete. Run scripts/analyze_non_iid_results.py next.")


if __name__ == "__main__":
    main()
