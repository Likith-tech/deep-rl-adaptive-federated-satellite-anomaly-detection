"""Train the baseline MLP anomaly detector on the Phase 1 processed data.

Run from the repository root:

    python scripts/train_baseline.py

Reads:
    configs/model.yaml                          (hyperparameters)
    data/processed/{train,validation}.parquet   (Phase 1 output)
    results/models/preprocessing/feature_columns.json

Writes:
    results/models/baseline/best_model.pt
    results/models/baseline/training_history.json
    experiments/baseline/run_config.json

Trains on TRAIN, selects the best checkpoint by VALIDATION loss.
KDDTest+ is never touched by this script — see scripts/evaluate_baseline.py
for the final, one-time test evaluation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.training.baseline_trainer import train_baseline  # noqa: E402

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
ARTIFACTS_DIR = REPO_ROOT / "results" / "models" / "preprocessing"
CHECKPOINT_DIR = REPO_ROOT / "results" / "models" / "baseline"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "baseline"


def main() -> None:
    model_config = yaml.safe_load((REPO_ROOT / "configs" / "model.yaml").read_text())["baseline"]

    feature_columns_path = ARTIFACTS_DIR / "feature_columns.json"
    if not feature_columns_path.exists():
        print(
            f"ERROR: {feature_columns_path} not found. Run the Phase 1 preprocessing "
            "pipeline first: python -m src.preprocessing.pipeline",
            file=sys.stderr,
        )
        sys.exit(1)
    feature_columns = json.loads(feature_columns_path.read_text())

    train_parquet = PROCESSED_DIR / "train.parquet"
    validation_parquet = PROCESSED_DIR / "validation.parquet"
    if not train_parquet.exists() or not validation_parquet.exists():
        print(
            f"ERROR: processed train/validation parquet not found under {PROCESSED_DIR}. "
            "Run: python -m src.preprocessing.pipeline",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Training baseline MLP: input_dim={model_config['input_dim']}, "
          f"hidden={model_config['hidden_dimensions']}, "
          f"epochs={model_config['training']['epochs']}, "
          f"batch_size={model_config['training']['batch_size']}, "
          f"lr={model_config['training']['learning_rate']}")

    history = train_baseline(
        train_parquet=train_parquet,
        validation_parquet=validation_parquet,
        feature_columns=feature_columns,
        config=model_config,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    print(f"\nBest epoch: {history.best_epoch}, best val_loss: {history.best_val_loss:.4f}")
    print(f"Training time: {history.training_seconds:.1f}s")
    print(f"Checkpoint saved to: {CHECKPOINT_DIR / 'best_model.pt'}")

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    run_record = {
        "model_config": model_config,
        "feature_count": len(feature_columns),
        "train_samples": None,  # filled below
        "validation_samples": None,
        "best_epoch": history.best_epoch,
        "best_val_loss": history.best_val_loss,
        "stopped_early": history.stopped_early,
        "training_seconds": history.training_seconds,
        "checkpoint_path": str((CHECKPOINT_DIR / "best_model.pt").relative_to(REPO_ROOT)),
    }
    metadata = json.loads((PROCESSED_DIR / "metadata.json").read_text())
    run_record["train_samples"] = metadata["train_samples"]
    run_record["validation_samples"] = metadata["validation_samples"]
    (EXPERIMENT_DIR / "run_config.json").write_text(json.dumps(run_record, indent=2))
    print(f"Experiment record saved to: {EXPERIMENT_DIR / 'run_config.json'}")


if __name__ == "__main__":
    main()
