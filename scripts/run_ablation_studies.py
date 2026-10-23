"""Phase 15 — run the two ablation sweeps (see configs/ablation_studies.yaml
for the full rationale of each).

Run from the repository root (requires Phase 4's ORIGINAL partition —
data/partitions/, the same one Phase 6/8/9 used):

    python scripts/run_ablation_studies.py

Sweep 1 (resource_weight_sweep) reuses Phase 8's UNMODIFIED
`run_adaptive_federated_training`; sweep 2 (staleness_decay_sweep)
reuses Phase 9's UNMODIFIED `run_staleness_federated_training`. Neither
src/ module is changed by this phase — only new experiment
configurations are run through existing, already-tested code paths.

Writes, under experiments/ablation_studies/:
    resource_weight_sweep/w_resource_<value>/{run_config.json,round_history.json,...}
    staleness_decay_sweep/decay_rate_<value>/{run_config.json,round_history.json,...}

Phase 6/8/9's own files are never read for writing and never modified.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.federated.adaptive_trainer import run_adaptive_federated_training  # noqa: E402
from src.simulation.non_iid_metrics import client_category_distribution, shannon_entropy  # noqa: E402
from src.simulation.partitioner import ClientStats  # noqa: E402
from src.training.staleness_trainer import run_staleness_federated_training  # noqa: E402

PARTITIONS_DIR = REPO_ROOT / "data" / "partitions"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
ARTIFACTS_DIR = REPO_ROOT / "results" / "models" / "preprocessing"
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "ablation_studies"
CONFIG_PATH = REPO_ROOT / "configs" / "ablation_studies.yaml"


def build_fairness_entropy(manifest: dict) -> dict[str, float]:
    """Identical to scripts/run_adaptive_fl_experiments.py's helper —
    duplicated rather than imported to keep each ablation-study script
    self-contained and because the function is short and stable."""
    categories = sorted({cat for s in manifest["client_stats"].values() for cat in s["category_counts"]})
    entropy = {}
    for client_id, stats_dict in manifest["client_stats"].items():
        stats = ClientStats(
            client_id=client_id,
            total_samples=stats_dict["total_samples"],
            normal_samples=stats_dict["normal_samples"],
            anomaly_samples=stats_dict["anomaly_samples"],
            category_counts=stats_dict["category_counts"],
        )
        distribution = client_category_distribution(stats, categories)
        entropy[client_id] = shannon_entropy(distribution)
    return entropy


def run_resource_weight_sweep(config: dict, client_ids: list[str], manifest: dict,
                               feature_columns: list[str], global_validation_parquet: Path) -> None:
    sweep_cfg = config["resource_weight_sweep"]
    ratio = sweep_cfg["non_resource_ratio"]
    model_config = config["model"]
    training_config = config["training"]
    num_rounds = config["federation"]["num_rounds"]
    shared_init_seed = config["initialization"]["shared_init_seed"]

    satellite_metadata = json.loads((REPO_ROOT / config["partitions"]["satellite_metadata"]).read_text())
    client_resource_metadata = {
        cid: {k: v for k, v in meta.items() if k not in ("client_id", "data_size")}
        for cid, meta in satellite_metadata["clients"].items()
    }
    client_fairness_entropy = build_fairness_entropy(manifest)

    sweep_root = EXPERIMENT_ROOT / "resource_weight_sweep"
    results_summary = []

    for w_resource in sweep_cfg["w_resource_values"]:
        remaining = 1.0 - w_resource
        rule_weights = {
            "performance": ratio["performance"] * remaining,
            "data": ratio["data"] * remaining,
            "resource": w_resource,
            "fairness": ratio["fairness"] * remaining,
        }
        weight_sum = sum(rule_weights.values())
        assert abs(weight_sum - 1.0) < 1e-6, f"w_resource={w_resource}: weights sum to {weight_sum}"

        run_name = f"w_resource_{w_resource:.2f}"
        run_dir = sweep_root / run_name
        print(f"\n{'=' * 70}\nresource_weight_sweep: {run_name} — {rule_weights}\n{'=' * 70}")

        start = time.time()
        result = run_adaptive_federated_training(
            client_ids=client_ids,
            partitions_dir=PARTITIONS_DIR,
            global_validation_parquet=global_validation_parquet,
            feature_columns=feature_columns,
            model_config=model_config,
            training_config=training_config,
            num_rounds=num_rounds,
            shared_init_seed=shared_init_seed,
            checkpoint_dir=run_dir,
            rule_weights=rule_weights,
            client_resource_metadata=client_resource_metadata,
            client_fairness_entropy=client_fairness_entropy,
        )
        print(f"Best round: {result.best_round} (val_loss={result.best_val_loss:.4f}, "
              f"val_f1={result.best_val_metrics.f1:.4f})")

        run_config = {
            "run_name": run_name, "w_resource": w_resource, "rule_weights": rule_weights,
            "num_rounds": num_rounds, "best_round": result.best_round,
            "best_val_loss": result.best_val_loss, "best_val_metrics": result.best_val_metrics.to_dict(),
            "total_seconds": result.total_seconds, "wall_seconds": time.time() - start,
        }
        (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
        results_summary.append(run_name)
        print(f"'{run_name}' complete in {time.time() - start:.1f}s. Artifacts under {run_dir}")

    print(f"\nresource_weight_sweep complete: {len(results_summary)} runs.")


def run_staleness_decay_sweep(config: dict, client_ids: list[str], manifest: dict,
                               feature_columns: list[str], global_validation_parquet: Path) -> None:
    sweep_cfg = config["staleness_decay_sweep"]
    model_config = config["model"]
    training_config = config["training"]
    num_rounds = config["federation"]["num_rounds"]
    shared_init_seed = config["initialization"]["shared_init_seed"]
    participation_seed = sweep_cfg["participation_seed"]

    satellite_metadata = json.loads((REPO_ROOT / config["partitions"]["satellite_metadata"]).read_text())
    client_availability = {
        cid: meta["availability_probability"] for cid, meta in satellite_metadata["clients"].items()
    }

    sweep_root = EXPERIMENT_ROOT / "staleness_decay_sweep"
    results_summary = []

    for decay_rate in sweep_cfg["decay_rate_values"]:
        run_name = f"decay_rate_{decay_rate:.2f}"
        run_dir = sweep_root / run_name
        print(f"\n{'=' * 70}\nstaleness_decay_sweep: {run_name}\n{'=' * 70}")

        start = time.time()
        result = run_staleness_federated_training(
            client_ids=client_ids,
            partitions_dir=PARTITIONS_DIR,
            global_validation_parquet=global_validation_parquet,
            feature_columns=feature_columns,
            model_config=model_config,
            training_config=training_config,
            num_rounds=num_rounds,
            shared_init_seed=shared_init_seed,
            checkpoint_dir=run_dir,
            client_availability=client_availability,
            staleness_mode="exponential",
            staleness_decay_rate=decay_rate,
            participation_seed=participation_seed,
        )
        print(f"Best round: {result.best_round} (val_loss={result.best_val_loss:.4f}, "
              f"val_f1={result.best_val_metrics.f1:.4f})")

        max_staleness_seen = max(s for r in result.round_history for s in r.client_staleness.values())
        run_config = {
            "run_name": run_name, "decay_rate": decay_rate, "num_rounds": num_rounds,
            "max_staleness_seen": max_staleness_seen, "best_round": result.best_round,
            "best_val_loss": result.best_val_loss, "best_val_metrics": result.best_val_metrics.to_dict(),
            "total_seconds": result.total_seconds, "wall_seconds": time.time() - start,
        }
        (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
        results_summary.append(run_name)
        print(f"'{run_name}' complete in {time.time() - start:.1f}s. Artifacts under {run_dir}")

    print(f"\nstaleness_decay_sweep complete: {len(results_summary)} runs.")


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())

    manifest_path = Path(config["partitions"]["manifest"])
    if not (REPO_ROOT / manifest_path).exists():
        print(f"ERROR: {manifest_path} not found. Run Phase 4 first: "
              "python scripts/create_satellite_partitions.py", file=sys.stderr)
        sys.exit(1)
    manifest = json.loads((REPO_ROOT / manifest_path).read_text())
    client_ids = sorted(manifest["client_sample_counts"].keys())

    feature_columns = json.loads((ARTIFACTS_DIR / "feature_columns.json").read_text())
    global_validation_parquet = PROCESSED_DIR / "validation.parquet"

    print(f"Ablation studies: clients={client_ids}, partition={config['partitions']['source']} "
          f"(Phase 4/6/8/9 original, alpha={manifest['alpha']})")

    run_resource_weight_sweep(config, client_ids, manifest, feature_columns, global_validation_parquet)
    run_staleness_decay_sweep(config, client_ids, manifest, feature_columns, global_validation_parquet)

    print("\nBoth ablation sweeps complete. Run scripts/evaluate_ablation_results.py next.")


if __name__ == "__main__":
    main()
