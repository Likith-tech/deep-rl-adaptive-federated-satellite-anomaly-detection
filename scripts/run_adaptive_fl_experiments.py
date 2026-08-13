"""Phase 8 — run the rule-based adaptive FedAvg ablation study.

Run from the repository root (requires Phase 4's ORIGINAL partition —
data/partitions/, the same one Phase 6 used):

    python scripts/run_adaptive_fl_experiments.py

For every rule config in configs/adaptive_fl.yaml (performance_only,
resource_only, data_only, combined), runs the SAME FedAvg setup as
Phase 6 (8 clients, 10 rounds, 1 local epoch, same partition, same
seeds) with ONLY the aggregation rule changed — deterministic,
rule-based weighting (src/federated/adaptive.py), NOT reinforcement
learning.

Writes, per rule config, under experiments/adaptive_fl/<rule_name>/:
    run_config.json, round_history.json, best_global_model.pt (gitignored),
    round_checkpoints/round_NN.pt (gitignored)

Phase 6's own files (results/reports/federated_results.md,
results/models/federated/, data/partitions/) are never read for
writing and never modified — this script only READS the Phase 4/6
partition, and writes exclusively under experiments/adaptive_fl/.
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

PARTITIONS_DIR = REPO_ROOT / "data" / "partitions"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
ARTIFACTS_DIR = REPO_ROOT / "results" / "models" / "preprocessing"
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "adaptive_fl"
CONFIG_PATH = REPO_ROOT / "configs" / "adaptive_fl.yaml"


def build_fairness_entropy(manifest: dict) -> dict[str, float]:
    """Shannon entropy of each client's local attack-category
    distribution — same function Phase 7 used
    (src/simulation/non_iid_metrics.py), computed here from the
    ALREADY-COMMITTED Phase 4 manifest.json, not recomputed from raw
    data."""
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


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    model_config = config["model"]
    training_config = config["training"]
    num_rounds = config["federation"]["num_rounds"]
    shared_init_seed = config["initialization"]["shared_init_seed"]

    manifest_path = Path(config["partitions"]["manifest"])
    if not (REPO_ROOT / manifest_path).exists():
        print(f"ERROR: {manifest_path} not found. Run Phase 4 first: "
              "python scripts/create_satellite_partitions.py", file=sys.stderr)
        sys.exit(1)
    manifest = json.loads((REPO_ROOT / manifest_path).read_text())
    client_ids = sorted(manifest["client_sample_counts"].keys())

    satellite_metadata = json.loads((REPO_ROOT / config["partitions"]["satellite_metadata"]).read_text())
    client_resource_metadata = {
        cid: {k: v for k, v in meta.items() if k not in ("client_id", "data_size")}
        for cid, meta in satellite_metadata["clients"].items()
    }

    client_fairness_entropy = build_fairness_entropy(manifest)

    feature_columns = json.loads((ARTIFACTS_DIR / "feature_columns.json").read_text())
    global_validation_parquet = PROCESSED_DIR / "validation.parquet"

    print(f"Adaptive FL ablation study: clients={client_ids}, rounds={num_rounds}, "
          f"partition={config['partitions']['source']} (Phase 4/6 original, alpha={manifest['alpha']})")
    print(f"Client fairness entropy (bits): { {k: round(v, 3) for k, v in client_fairness_entropy.items()} }")

    for rule_name, rule_cfg in config["rule_configs"].items():
        rule_weights = {k: rule_cfg[k] for k in ("performance", "data", "resource", "fairness")}
        weight_sum = sum(rule_weights.values())
        assert abs(weight_sum - 1.0) < 1e-6, f"{rule_name}: rule weights must sum to 1.0, got {weight_sum}"

        rule_dir = EXPERIMENT_ROOT / rule_name
        print(f"\n{'=' * 70}\nRule config: {rule_name} — {rule_weights}\n{'=' * 70}")

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
            checkpoint_dir=rule_dir,
            rule_weights=rule_weights,
            client_resource_metadata=client_resource_metadata,
            client_fairness_entropy=client_fairness_entropy,
        )

        for r in result.round_history:
            print(f"Round {r.round_number:2d}/{num_rounds} | val_loss={r.val_loss:.4f} "
                  f"val_f1={r.val_metrics['f1']:.4f}")
        print(f"Best round: {result.best_round} (val_loss={result.best_val_loss:.4f}, "
              f"val_f1={result.best_val_metrics.f1:.4f})")

        run_config = {
            "rule_name": rule_name,
            "rule_weights": rule_weights,
            "rule_rationale": rule_cfg.get("rationale", "").strip(),
            "config": config,
            "client_ids": client_ids,
            "client_sample_counts": {cid: manifest["client_sample_counts"][cid] for cid in client_ids},
            "client_fairness_entropy": client_fairness_entropy,
            "num_rounds": num_rounds,
            "best_round": result.best_round,
            "best_val_loss": result.best_val_loss,
            "best_val_metrics": result.best_val_metrics.to_dict(),
            "total_seconds": result.total_seconds,
            "wall_seconds": time.time() - start,
        }
        (rule_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
        print(f"Rule '{rule_name}' complete in {time.time() - start:.1f}s. Artifacts under {rule_dir}")

    print(f"\nAll {len(config['rule_configs'])} rule configs complete. "
          "Run scripts/evaluate_adaptive_fl_results.py next.")


if __name__ == "__main__":
    main()
