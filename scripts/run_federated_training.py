"""Phase 6 — run the baseline synchronous FedAvg experiment.

Run from the repository root (requires Phase 4's partitions — see
data/partitions/README.md):

    python scripts/run_federated_training.py

Reads:
    configs/federated.yaml
    data/partitions/manifest.json                (client list, from Phase 4)
    data/partitions/<client>/train.parquet        (each client's OWN local data)
    data/processed/validation.parquet             (single GLOBAL validation set)
    results/models/preprocessing/feature_columns.json

Writes:
    results/models/federated/best_global_model.pt
    results/models/federated/round_checkpoints/round_NN.pt
    results/models/federated/round_history.json
    experiments/federated/run_config.json

Standard FedAvg only: ALL clients participate every round, sample-count
weighted averaging (src/federated/fedavg.py), no adaptive client
selection, no staleness handling, no DRL. KDDTest+ is never touched
here — see scripts/evaluate_federated_model.py for the final, one-time
test evaluation of the selected best-validation round.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.federated.trainer import run_federated_training  # noqa: E402
from src.models.baseline_mlp import BaselineMLP  # noqa: E402

PARTITIONS_DIR = REPO_ROOT / "data" / "partitions"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
ARTIFACTS_DIR = REPO_ROOT / "results" / "models" / "preprocessing"
CHECKPOINT_DIR = REPO_ROOT / "results" / "models" / "federated"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "federated"


def count_parameters(model_config: dict) -> int:
    model = BaselineMLP(
        input_dim=model_config["input_dim"],
        hidden_dimensions=model_config["hidden_dimensions"],
        dropout=model_config["dropout"],
        output_dim=model_config["output_dim"],
    )
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    config = yaml.safe_load((REPO_ROOT / "configs" / "federated.yaml").read_text())
    model_config = config["model"]
    training_config = config["training"]
    federation_config = config["federation"]

    manifest_path = PARTITIONS_DIR / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found. Run Phase 4 first: "
              "python scripts/create_satellite_partitions.py", file=sys.stderr)
        sys.exit(1)
    manifest = json.loads(manifest_path.read_text())
    client_ids = sorted(manifest["client_sample_counts"].keys())

    if federation_config["num_clients"] != len(client_ids):
        print(f"WARNING: configs/federated.yaml num_clients={federation_config['num_clients']} "
              f"but manifest has {len(client_ids)} clients ({client_ids}). Using manifest's client list.",
              file=sys.stderr)

    feature_columns = json.loads((ARTIFACTS_DIR / "feature_columns.json").read_text())
    global_validation_parquet = PROCESSED_DIR / "validation.parquet"
    if not global_validation_parquet.exists():
        print(f"ERROR: {global_validation_parquet} not found. Run the Phase 1 pipeline first.", file=sys.stderr)
        sys.exit(1)

    num_rounds = federation_config["num_rounds"]
    shared_init_seed = config["initialization"]["shared_init_seed"]

    # Simulated per-client resource metadata (Phase 4) — recorded in
    # round logs for later adaptive-FL phases only; NOT used here to
    # select, exclude, or weight clients (see configs/federated.yaml).
    satellite_metadata_path = PARTITIONS_DIR / "satellite_metadata.json"
    client_resource_metadata = None
    if satellite_metadata_path.exists():
        satellite_metadata = json.loads(satellite_metadata_path.read_text())
        client_resource_metadata = {
            cid: {k: v for k, v in meta.items() if k not in ("client_id", "data_size")}
            for cid, meta in satellite_metadata["clients"].items()
        }

    print(
        f"Federated training (baseline FedAvg): clients={client_ids}, "
        f"rounds={num_rounds}, local_epochs={training_config['local_epochs']}, "
        f"batch_size={training_config['batch_size']}, lr={training_config['learning_rate']}, "
        f"aggregation={federation_config['aggregation_strategy']} ({federation_config['weighting']})"
    )

    result = run_federated_training(
        client_ids=client_ids,
        partitions_dir=PARTITIONS_DIR,
        global_validation_parquet=global_validation_parquet,
        feature_columns=feature_columns,
        model_config=model_config,
        training_config=training_config,
        num_rounds=num_rounds,
        shared_init_seed=shared_init_seed,
        checkpoint_dir=CHECKPOINT_DIR,
        client_resource_metadata=client_resource_metadata,
    )

    for round_record in result.round_history:
        m = round_record.val_metrics
        print(
            f"Round {round_record.round_number:2d}/{num_rounds} | "
            f"val_loss={round_record.val_loss:.4f} val_f1={m['f1']:.4f} "
            f"val_acc={m['accuracy']:.4f} round_seconds={round_record.round_seconds:.1f}"
        )

    print(f"\nBest round (by validation loss): {result.best_round}, "
          f"best_val_loss={result.best_val_loss:.4f}, "
          f"best_val_f1={result.best_val_metrics.f1:.4f}")
    print(f"Total training time: {result.total_seconds:.1f}s")
    print(f"Checkpoint saved to: {CHECKPOINT_DIR / 'best_global_model.pt'}")

    num_params = count_parameters(model_config)
    param_bytes = num_params * 4  # float32
    upload_per_round_bytes = param_bytes * len(client_ids)   # clients -> server
    download_per_round_bytes = param_bytes * len(client_ids)  # server -> clients
    total_communication_bytes = (upload_per_round_bytes + download_per_round_bytes) * num_rounds

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    run_record = {
        "config": config,
        "client_ids": client_ids,
        "client_sample_counts": {cid: manifest["client_sample_counts"][cid] for cid in client_ids},
        "num_rounds": num_rounds,
        "best_round": result.best_round,
        "best_val_loss": result.best_val_loss,
        "best_val_metrics": result.best_val_metrics.to_dict(),
        "total_seconds": result.total_seconds,
        "checkpoint_path": str((CHECKPOINT_DIR / "best_global_model.pt").relative_to(REPO_ROOT)),
        "communication_estimate": {
            "note": "Simulation estimate of parameter transfer size — NOT measured satellite network traffic.",
            "num_model_parameters": num_params,
            "bytes_per_model_transfer": param_bytes,
            "upload_bytes_per_round": upload_per_round_bytes,
            "download_bytes_per_round": download_per_round_bytes,
            "total_communication_bytes_all_rounds": total_communication_bytes,
        },
    }
    (EXPERIMENT_DIR / "run_config.json").write_text(json.dumps(run_record, indent=2))
    print(f"Experiment record saved to: {EXPERIMENT_DIR / 'run_config.json'}")


if __name__ == "__main__":
    main()
