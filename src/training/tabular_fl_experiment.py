"""Shared FedAvg experiment runner for the TON_IoT / Bot-IoT tracks.

Reuses, unmodified: src.simulation.partitioner.dirichlet_partition
(IID cell uses a large alpha to approximate a uniform/IID split — the
same "Dirichlet(alpha->large) ~ IID" convention widely used in the FL
literature — rather than inventing a second, parallel partitioning
code path), src.federated.trainer.run_federated_training,
src.federated.client.FederatedClient, src.federated.server.FederatedServer,
src.federated.fedavg.federated_average, src.models.baseline_mlp.BaselineMLP.

Only NEW logic here: writing per-client parquet partitions to a scratch
directory, and final held-out test evaluation with the project's
standard metrics (src.evaluation.metrics.compute_metrics) PLUS PR-AUC
(this project's own standard extra metric, not present in
compute_metrics, added locally here rather than editing the shared
metrics module for one extra number).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from src.evaluation.metrics import ClassificationMetrics, compute_metrics
from src.federated.trainer import run_federated_training
from src.models.baseline_mlp import BaselineMLP
from src.simulation.partitioner import dirichlet_partition, verify_partition_integrity
from src.training.baseline_trainer import load_features_and_labels

CLIENT_IDS = [f"SAT-{i:02d}" for i in range(1, 9)]
IID_ALPHA = 1000.0  # large alpha -> Dirichlet proportions concentrate near uniform -> approx. IID
NON_IID_ALPHA = 0.5  # same convention as the existing NSL-KDD non-IID track (experiments/non_iid/alpha_0.5)


def write_client_partitions(
    train_df: pd.DataFrame,
    category_column: str,
    alpha: float,
    seed: int,
    partitions_dir: Path,
) -> dict:
    partition = dirichlet_partition(
        df=train_df, client_ids=CLIENT_IDS, category_column=category_column,
        alpha=alpha, seed=seed, label_binary_column="label_binary",
    )
    verify_partition_integrity(train_df, partition.client_indices)

    if partitions_dir.exists():
        shutil.rmtree(partitions_dir)
    partitions_dir.mkdir(parents=True)

    client_sizes = {}
    for client_id, indices in partition.client_indices.items():
        client_dir = partitions_dir / client_id
        client_dir.mkdir(parents=True)
        subset = train_df.loc[indices].reset_index(drop=True)
        subset.to_parquet(client_dir / "train.parquet", index=False)
        client_sizes[client_id] = len(subset)

    return {"alpha": alpha, "seed": seed, "client_sizes": client_sizes, "attempts_used": partition.attempts_used}


def evaluate_test_set(
    model_config: dict,
    state_dict: dict[str, torch.Tensor],
    test_parquet: Path,
    feature_columns: list[str],
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineMLP(
        input_dim=model_config["input_dim"], hidden_dimensions=model_config["hidden_dimensions"],
        dropout=model_config["dropout"], output_dim=model_config["output_dim"],
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    X_test, y_test = load_features_and_labels(test_parquet, feature_columns)
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).to(device)).squeeze(-1)
        y_prob = torch.sigmoid(logits).cpu().numpy()
    y_true = y_test.astype(int)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics: ClassificationMetrics = compute_metrics(y_true, y_pred, y_prob)
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else None

    result = metrics.to_dict()
    result["pr_auc"] = pr_auc
    result["test_samples"] = int(len(y_true))
    result["test_anomaly_count"] = int(y_true.sum())
    return result


def run_fedavg_cell(
    dataset_name: str,
    processed_dir: Path,
    category_column: str,
    partition_type: str,  # "iid" or "non_iid"
    seed: int,
    num_rounds: int,
    model_config: dict,
    training_config: dict,
    feature_columns: list[str],
    experiment_scratch_dir: Path,
) -> dict:
    """Runs one (dataset, partition_type, seed) FedAvg cell end to end and
    returns the held-out test metrics. Scratch partitions/checkpoints are
    written under experiment_scratch_dir and are safe to delete afterward
    (only the final metrics/summary JSON written by the caller is kept)."""
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    alpha = IID_ALPHA if partition_type == "iid" else NON_IID_ALPHA

    cell_dir = experiment_scratch_dir / f"{partition_type}_seed{seed}"
    partitions_dir = cell_dir / "partitions"
    partition_info = write_client_partitions(train_df, category_column, alpha, seed, partitions_dir)

    training_config = dict(training_config, seed=seed)
    result = run_federated_training(
        client_ids=CLIENT_IDS, partitions_dir=partitions_dir,
        global_validation_parquet=processed_dir / "validation.parquet",
        feature_columns=feature_columns, model_config=model_config, training_config=training_config,
        num_rounds=num_rounds, shared_init_seed=seed, checkpoint_dir=cell_dir,
    )

    best_checkpoint = torch.load(cell_dir / "best_global_model.pt", weights_only=False)
    test_metrics = evaluate_test_set(model_config, best_checkpoint["model_state_dict"], processed_dir / "test.parquet", feature_columns)

    return {
        "dataset": dataset_name, "partition_type": partition_type, "seed": seed, "alpha": alpha,
        "partition_info": partition_info, "best_round": result.best_round, "best_val_loss": result.best_val_loss,
        "test_metrics": test_metrics,
    }
