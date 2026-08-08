"""Train the temporal GRU + attention anomaly detector.

Run from the repository root:

    python scripts/train_temporal.py

For each candidate sequence length in configs/temporal.yaml
(`temporal.sequence_length_candidates`), trains a model on an
order-preserving train/validation split built from the Phase 1 interim
data (reusing Phase 1's already-fit scaler/encoder — see
src/training/temporal_trainer.py for why this differs from the MLP
baseline's data prep). The sequence length whose model achieves the
best VALIDATION loss is selected as the final model — KDDTest+ is never
touched here.

Writes:
    results/models/temporal/candidate_seq{N}/best_model.pt   (per candidate)
    results/models/temporal/best_model.pt                    (selected final model)
    results/models/temporal/selection_summary.json
    experiments/temporal/run_config.json
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.training.temporal_trainer import prepare_temporal_train_validation, train_temporal  # noqa: E402
from src.data.schema import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS  # noqa: E402

INTERIM_DIR = REPO_ROOT / "data" / "interim"
ARTIFACTS_DIR = REPO_ROOT / "results" / "models" / "preprocessing"
CHECKPOINT_DIR = REPO_ROOT / "results" / "models" / "temporal"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "temporal"


def main() -> None:
    temporal_config = yaml.safe_load((REPO_ROOT / "configs" / "temporal.yaml").read_text())["temporal"]

    interim_train_parquet = INTERIM_DIR / "kdd_train_cleaned.parquet"
    scaler_path = ARTIFACTS_DIR / "scaler.joblib"
    encoder_path = ARTIFACTS_DIR / "encoder.joblib"
    feature_columns_path = ARTIFACTS_DIR / "feature_columns.json"

    for path in (interim_train_parquet, scaler_path, encoder_path, feature_columns_path):
        if not path.exists():
            print(
                f"ERROR: {path} not found. Run the Phase 1 preprocessing pipeline first: "
                "python -m src.preprocessing.pipeline",
                file=sys.stderr,
            )
            sys.exit(1)

    feature_columns = json.loads(feature_columns_path.read_text())

    candidates = temporal_config["sequence_length_candidates"]
    validation_size = temporal_config["validation_size"]
    stride = temporal_config["stride"]
    label_strategy = temporal_config["sequence_label_strategy"]
    label_threshold = temporal_config.get("sequence_label_threshold")

    print(f"Sequence label strategy: {label_strategy}"
          + (f" (threshold={label_threshold})" if label_strategy == "ratio" else ""))

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    candidate_results = []

    for seq_len in candidates:
        print(f"\n=== Sequence length candidate: {seq_len} ===")
        train_seqs, val_seqs = prepare_temporal_train_validation(
            interim_train_parquet=interim_train_parquet,
            scaler_path=scaler_path,
            encoder_path=encoder_path,
            numerical_columns=NUMERICAL_COLUMNS,
            categorical_columns=CATEGORICAL_COLUMNS,
            feature_columns=feature_columns,
            label_column="label_binary",
            validation_size=validation_size,
            sequence_length=seq_len,
            stride=stride,
            label_strategy=label_strategy,
            label_threshold=label_threshold,
        )
        print(f"Train sequences: {train_seqs.num_sequences} (anomaly rate {100 * train_seqs.anomaly_rate:.2f}%), "
              f"Validation sequences: {val_seqs.num_sequences} (anomaly rate {100 * val_seqs.anomaly_rate:.2f}%)")

        model_config = {
            "input_dim": temporal_config["input_dim"],
            "projection_dim": temporal_config["projection_dim"],
            "hidden_dim": temporal_config["hidden_dim"],
            "num_gru_layers": temporal_config["num_gru_layers"],
            "dropout": temporal_config["dropout"],
            "output_dim": temporal_config["output_dim"],
            "sequence_length": seq_len,
            "stride": stride or seq_len,
            "label_strategy": label_strategy,
            "label_threshold": label_threshold,
            "feature_columns": feature_columns,
            "training": temporal_config["training"],
        }

        candidate_dir = CHECKPOINT_DIR / f"candidate_seq{seq_len}"
        history = train_temporal(train_seqs, val_seqs, model_config, candidate_dir)

        candidate_results.append(
            {
                "sequence_length": seq_len,
                "train_sequences": train_seqs.num_sequences,
                "validation_sequences": val_seqs.num_sequences,
                "train_anomaly_rate": train_seqs.anomaly_rate,
                "validation_anomaly_rate": val_seqs.anomaly_rate,
                "best_epoch": history.best_epoch,
                "best_val_loss": history.best_val_loss,
                "stopped_early": history.stopped_early,
                "training_seconds": history.training_seconds,
                "checkpoint_dir": str(candidate_dir.relative_to(REPO_ROOT)),
            }
        )

    # Select the candidate with the lowest validation loss.
    best_candidate = min(candidate_results, key=lambda r: r["best_val_loss"])
    print(f"\n=== Selected sequence length: {best_candidate['sequence_length']} "
          f"(val_loss={best_candidate['best_val_loss']:.4f}) ===")

    best_candidate_checkpoint = REPO_ROOT / best_candidate["checkpoint_dir"] / "best_model.pt"
    final_checkpoint = CHECKPOINT_DIR / "best_model.pt"
    shutil.copyfile(best_candidate_checkpoint, final_checkpoint)
    best_history_path = REPO_ROOT / best_candidate["checkpoint_dir"] / "best_model_history.json"
    shutil.copyfile(best_history_path, CHECKPOINT_DIR / "training_history.json")

    selection_summary = {
        "label_strategy": label_strategy,
        "label_threshold": label_threshold,
        "candidates": candidate_results,
        "selected_sequence_length": best_candidate["sequence_length"],
        "selection_criterion": "lowest validation loss",
        "final_checkpoint": str(final_checkpoint.relative_to(REPO_ROOT)),
    }
    (CHECKPOINT_DIR / "selection_summary.json").write_text(json.dumps(selection_summary, indent=2))
    print(f"Selection summary saved to: {CHECKPOINT_DIR / 'selection_summary.json'}")
    print(f"Final checkpoint: {final_checkpoint}")

    metadata = json.loads((REPO_ROOT / "data" / "processed" / "metadata.json").read_text())
    run_record = {
        "temporal_config": temporal_config,
        "selection_summary": selection_summary,
        "dataset_metadata": metadata,
    }
    (EXPERIMENT_DIR / "run_config.json").write_text(json.dumps(run_record, indent=2))
    print(f"Experiment record saved to: {EXPERIMENT_DIR / 'run_config.json'}")


if __name__ == "__main__":
    main()
