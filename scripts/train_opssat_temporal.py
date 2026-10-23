"""Phase 9A — train the temporal model on OPSSAT-AD (real ESA spacecraft
telemetry), independently of the NSL-KDD/FedAvg/adaptive-FL pipeline.

Run from the repository root, after scripts/download_opssat_ad.py:

    python scripts/train_opssat_temporal.py

Reads:
    configs/opssat_ad.yaml
    data/raw/opssat_ad/dataset.csv

Writes:
    results/models/opssat_preprocessing/{scaler,encoder}.joblib, feature_columns.json, metadata.json
    results/models/opssat_temporal/best_model.pt, training_history.json
    experiments/opssat_ad/run_config.json

Only the AUTHOR'S training segments (train=1) are used here — the 529
author's test segments (train=0) are never loaded by this script. See
scripts/evaluate_opssat_temporal.py for the final, one-time test
evaluation. This script does not read, write, or import anything from
the NSL-KDD pipeline (src/preprocessing/, src/training/temporal_trainer.py,
configs/temporal.yaml).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.satellite.loader import load_opssat_dataset, split_by_author_train_flag  # noqa: E402
from src.satellite.preprocessing import (  # noqa: E402
    extract_labels,
    fit_preprocessing,
    stratified_train_validation_split,
    transform,
)
from src.training.opssat_trainer import train_opssat  # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs" / "opssat_ad.yaml"
CHECKPOINT_DIR = REPO_ROOT / "results" / "models" / "opssat_temporal"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "opssat_ad"


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())

    raw_dir = REPO_ROOT / config["data"]["raw_dir"]
    df = load_opssat_dataset(raw_dir)
    author_train_df, author_test_df = split_by_author_train_flag(df)
    print(f"Loaded OPSSAT-AD: {len(df)} total segments "
          f"({len(author_train_df)} author-train, {len(author_test_df)} author-test — test NOT loaded further here)")

    val_cfg = config["validation"]
    train_df, val_df = stratified_train_validation_split(
        author_train_df, validation_size=val_cfg["validation_size"], seed=val_cfg["seed"]
    )
    print(f"Train/validation split (from author-train ONLY, seed={val_cfg['seed']}, "
          f"stratified={val_cfg['stratify']}): {len(train_df)} train / {len(val_df)} validation")

    artifacts = fit_preprocessing(train_df)  # fit on TRAIN portion only
    artifacts_dir = REPO_ROOT / config["data"]["preprocessing_artifacts_dir"]
    artifacts.save(artifacts_dir)
    print(f"Fit preprocessing on {len(train_df)} training segments only. "
          f"Feature dim: {len(artifacts.feature_columns)}. Saved to {artifacts_dir}")

    X_train = transform(train_df, artifacts)
    y_train = extract_labels(train_df)
    X_val = transform(val_df, artifacts)
    y_val = extract_labels(val_df)

    print(f"Training temporal model (input_dim={config['model']['input_dim']}, "
          f"seq_len={config['model']['sequence_length']} — see configs/opssat_ad.yaml notes)")

    start = time.time()
    history = train_opssat(X_train, y_train, X_val, y_val, config, CHECKPOINT_DIR)
    wall_seconds = time.time() - start

    print(f"\nBest epoch: {history.best_epoch}, best val_loss: {history.best_val_loss:.4f}")
    print(f"Training time: {history.training_seconds:.1f}s")
    print(f"Checkpoint saved to: {CHECKPOINT_DIR / 'best_model.pt'}")

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    run_record = {
        "dataset": "OPSSAT-AD",
        "dataset_source": "https://zenodo.org/records/12588358 (CC-BY-4.0)",
        "config": config,
        "author_train_segments": len(author_train_df),
        "author_test_segments": len(author_test_df),
        "train_segments": len(train_df),
        "validation_segments": len(val_df),
        "feature_dim": len(artifacts.feature_columns),
        "feature_columns": artifacts.feature_columns,
        "best_epoch": history.best_epoch,
        "best_val_loss": history.best_val_loss,
        "stopped_early": history.stopped_early,
        "training_seconds": history.training_seconds,
        "wall_seconds": wall_seconds,
        "checkpoint_path": str((CHECKPOINT_DIR / "best_model.pt").relative_to(REPO_ROOT)),
    }
    (EXPERIMENT_DIR / "run_config.json").write_text(json.dumps(run_record, indent=2))
    print(f"Experiment record saved to: {EXPERIMENT_DIR / 'run_config.json'}")


if __name__ == "__main__":
    main()
