"""Phase 5 — train one independent local model per simulated satellite.

Run from the repository root (requires Phase 4's partitions to exist —
see data/partitions/README.md):

    python scripts/train_local_models.py

Reads:
    configs/local_training.yaml
    data/partitions/manifest.json               (client list + stats, from Phase 4)
    data/partitions/<client>/train.parquet       (each client's OWN local data, real records)
    data/processed/validation.parquet            (single GLOBAL validation set)
    results/models/preprocessing/feature_columns.json

Writes, per client:
    results/models/local/<client>/best_model.pt
    results/models/local/<client>/training_history.json
    results/models/local/<client>/metadata.json

Plus one combined:
    experiments/local_training/run_config.json

There is NO communication or aggregation between clients — each client
is trained fully independently, in sequence, starting from the same
shared initial weights (see configs/local_training.yaml ->
initialization.shared_init_seed). KDDTest+ is never touched here.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.training.local_trainer import build_initial_state_dict, train_local_client  # noqa: E402

PARTITIONS_DIR = REPO_ROOT / "data" / "partitions"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
ARTIFACTS_DIR = REPO_ROOT / "results" / "models" / "preprocessing"
LOCAL_MODELS_DIR = REPO_ROOT / "results" / "models" / "local"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "local_training"


def main() -> None:
    config = yaml.safe_load((REPO_ROOT / "configs" / "local_training.yaml").read_text())

    manifest_path = PARTITIONS_DIR / "manifest.json"
    if not manifest_path.exists():
        print(
            f"ERROR: {manifest_path} not found. Run Phase 4 first: "
            "python scripts/create_satellite_partitions.py",
            file=sys.stderr,
        )
        sys.exit(1)
    manifest = json.loads(manifest_path.read_text())
    client_ids = sorted(manifest["client_sample_counts"].keys())

    feature_columns_path = ARTIFACTS_DIR / "feature_columns.json"
    feature_columns = json.loads(feature_columns_path.read_text())

    global_validation_parquet = PROCESSED_DIR / "validation.parquet"
    if not global_validation_parquet.exists():
        print(f"ERROR: {global_validation_parquet} not found. Run the Phase 1 pipeline first.", file=sys.stderr)
        sys.exit(1)

    print(
        f"Local training config: input_dim={config['model']['input_dim']}, "
        f"hidden={config['model']['hidden_dimensions']}, "
        f"epochs={config['training']['epochs']}, "
        f"batch_size={config['training']['batch_size']}, "
        f"lr={config['training']['learning_rate']}, "
        f"clients={client_ids}"
    )

    initial_state_dict = build_initial_state_dict(config["model"], config["initialization"]["shared_init_seed"])
    print(f"Built shared initial weights (seed={config['initialization']['shared_init_seed']}).")

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    run_summary = {"clients": {}, "total_wall_seconds": 0.0}
    overall_start = time.time()

    for client_id in client_ids:
        client_train_parquet = PARTITIONS_DIR / client_id / "train.parquet"
        if not client_train_parquet.exists():
            print(f"ERROR: {client_train_parquet} not found — skipping {client_id}.", file=sys.stderr)
            continue

        checkpoint_dir = LOCAL_MODELS_DIR / client_id
        print(f"\n=== Training {client_id} "
              f"({manifest['client_sample_counts'][client_id]} local samples) ===")

        result = train_local_client(
            client_id=client_id,
            client_train_parquet=client_train_parquet,
            global_validation_parquet=global_validation_parquet,
            feature_columns=feature_columns,
            config=config,
            initial_state_dict=initial_state_dict,
            checkpoint_dir=checkpoint_dir,
            training_seed=config["training"]["seed"],
        )

        print(
            f"{client_id}: best_epoch={result.best_epoch}, "
            f"best_val_loss={result.best_val_loss:.4f}, "
            f"stopped_early={result.stopped_early}, "
            f"training_seconds={result.training_seconds:.1f}"
        )
        print(f"  local train  -> f1={result.local_train_metrics.f1:.4f} acc={result.local_train_metrics.accuracy:.4f}")
        print(f"  global val   -> f1={result.global_val_metrics.f1:.4f} acc={result.global_val_metrics.accuracy:.4f} "
              f"roc_auc={result.global_val_metrics.roc_auc}")

        client_metadata = {
            "client_id": client_id,
            "seed": config["training"]["seed"],
            "shared_init_seed": config["initialization"]["shared_init_seed"],
            "model_config": config["model"],
            "training_config": config["training"],
            "train_samples": result.train_samples,
            "training_seconds": result.training_seconds,
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "stopped_early": result.stopped_early,
            "local_train_metrics": result.local_train_metrics.to_dict(),
            "global_val_metrics": result.global_val_metrics.to_dict(),
        }
        (checkpoint_dir / "metadata.json").write_text(json.dumps(client_metadata, indent=2))

        run_summary["clients"][client_id] = client_metadata

    run_summary["total_wall_seconds"] = time.time() - overall_start
    run_summary["shared_init_seed"] = config["initialization"]["shared_init_seed"]
    run_summary["config"] = config
    (EXPERIMENT_DIR / "run_config.json").write_text(json.dumps(run_summary, indent=2))
    print(f"\nAll clients trained. Total wall time: {run_summary['total_wall_seconds']:.1f}s")
    print(f"Experiment record saved to: {EXPERIMENT_DIR / 'run_config.json'}")


if __name__ == "__main__":
    main()
