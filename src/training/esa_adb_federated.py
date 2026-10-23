"""ESA-ADB Experiment 2 — standard FedAvg baseline under the approved
Option A ([X;M], input_dim=116) representation.

Reuses, UNCHANGED: `src.federated.server.FederatedServer` (standard
sample-count-weighted `federated_average`) and
`src.federated.protocol.ClientUpdate`/`RoundRecord` — both are already
dataset-agnostic (they operate on state_dicts and metadata only, never
on raw data or a specific model class), so no modification was needed
or made to `src/federated/`.

NOT reused: `src.federated.client.FederatedClient` and
`src.federated.trainer.run_federated_training` — both hardcode
`BaselineMLP` and read from `data/partitions/<client>/train.parquet`,
neither of which applies here (ESA-ADB clients hold in-memory
[X;M]-masked `SequenceDataset` objects, not MLP-shaped parquet
partitions). This module mirrors their structure and conventions
(Phase 6's `configs/federated.yaml`: local_epochs=1, batch_size=128,
Adam lr=0.001, seed=42, all clients participate every round, best round
selected by validation loss) rather than copying them, and the ONE
deliberate deviation — per-client `pos_weight` in `BCEWithLogitsLoss`,
absent from Phase 6 — is required because ESA-ADB's anomaly prevalence
is far more severe than NSL-KDD's, exactly as already established in
Experiments 1 and 1A.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from src.federated.protocol import ClientUpdate
from src.federated.server import FederatedServer
from src.models.temporal_gru_attention import TemporalGRUAttention
from src.preprocessing.sequences import SequenceDataset
from src.training.esa_adb_trainer import compute_extended_metrics, compute_pos_weight, make_dataloader, run_inference, set_seed


def build_initial_state_dict(model_config: dict, seed: int) -> dict[str, torch.Tensor]:
    """One randomly-initialized TemporalGRUAttention(input_dim=116),
    cloned as every client's shared round-1 starting point — mirrors
    src/training/local_trainer.py:build_initial_state_dict's role for
    the MLP, unchanged in spirit."""
    set_seed(seed)
    model = TemporalGRUAttention(
        input_dim=model_config["input_dim"], projection_dim=model_config["projection_dim"],
        hidden_dim=model_config["hidden_dim"], num_gru_layers=model_config["num_gru_layers"],
        dropout=model_config["dropout"], output_dim=model_config["output_dim"],
    )
    return {k: v.clone() for k, v in model.state_dict().items()}


class EsaAdbFederatedClient:
    """One real ESA-ADB client. Holds ONLY its own in-memory, already
    [X;M]-masked train SequenceDataset — never another client's data,
    never a reference to the pooled/global dataset."""

    def __init__(self, client_id: str, train_dataset: SequenceDataset) -> None:
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.pos_weight = compute_pos_weight(train_dataset.labels)  # from THIS client's train labels only

    def train_round(self, global_state_dict: dict[str, torch.Tensor], model_config: dict, training_config: dict, seed: int) -> ClientUpdate:
        set_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_loader = make_dataloader(self.train_dataset, training_config["batch_size"], shuffle=True)

        model = TemporalGRUAttention(
            input_dim=model_config["input_dim"], projection_dim=model_config["projection_dim"],
            hidden_dim=model_config["hidden_dim"], num_gru_layers=model_config["num_gru_layers"],
            dropout=model_config["dropout"], output_dim=model_config["output_dim"],
        ).to(device)
        model.load_state_dict({k: v.clone() for k, v in global_state_dict.items()})

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight, device=device))
        optimizer = torch.optim.Adam(model.parameters(), lr=training_config["learning_rate"])

        start_time = time.time()
        local_epochs = training_config["local_epochs"]
        total_loss, n_seen = 0.0, 0

        for _ in range(local_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                logits, _ = model(X_batch)
                logits = logits.squeeze(-1)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(y_batch)
                n_seen += len(y_batch)

        training_seconds = time.time() - start_time
        avg_loss = total_loss / n_seen

        return ClientUpdate(
            client_id=self.client_id,
            state_dict={k: v.detach().clone().cpu() for k, v in model.state_dict().items()},
            num_samples=self.train_dataset.num_sequences,
            local_epochs=local_epochs,
            train_loss=avg_loss,
            training_seconds=training_seconds,
        )


def _evaluate(model_config: dict, state_dict: dict, dataset: SequenceDataset, batch_size: int, pos_weight: float, device: torch.device):
    """Evaluate one dataset with an aggregated state_dict: returns
    (loss, ExtendedMetrics) at a FIXED 0.5 threshold — round-by-round
    monitoring only, never the final frozen-threshold evaluation (that
    happens once, after all rounds, in the runner script)."""
    model = TemporalGRUAttention(
        input_dim=model_config["input_dim"], projection_dim=model_config["projection_dim"],
        hidden_dim=model_config["hidden_dim"], num_gru_layers=model_config["num_gru_layers"],
        dropout=model_config["dropout"], output_dim=model_config["output_dim"],
    ).to(device)
    model.load_state_dict(state_dict)

    loader = make_dataloader(dataset, batch_size, shuffle=False)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    y_prob, y_true = run_inference(model, loader, device)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = compute_extended_metrics(y_true, y_pred, y_prob)

    # Loss recomputed with THIS dataset's own pos_weight-weighted criterion (matches how it was trained/scored elsewhere).
    total_loss, n = 0.0, 0
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits, _ = model(X_batch)
            logits = logits.squeeze(-1)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            n += len(y_batch)
    return total_loss / n, metrics


@dataclass
class FedAvgRoundRecord:
    round_number: int
    global_val_loss: float
    global_val_metrics: dict
    per_client_val: dict  # client_id -> {loss, f1, recall, precision, roc_auc, pr_auc, sample_count}
    fairness_gap_f1: float  # max(client val F1) - min(client val F1), same formula as Phase 8/9B
    client_updates: list[dict] = field(default_factory=list)
    round_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "round_number": self.round_number, "global_val_loss": self.global_val_loss,
            "global_val_metrics": self.global_val_metrics, "per_client_val": self.per_client_val,
            "fairness_gap_f1": self.fairness_gap_f1, "client_updates": self.client_updates,
            "round_seconds": self.round_seconds,
        }


@dataclass
class FedAvgTrainingResult:
    round_history: list[FedAvgRoundRecord] = field(default_factory=list)
    best_round: int = 0
    best_val_loss: float = float("inf")
    total_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "round_history": [r.to_dict() for r in self.round_history],
            "best_round": self.best_round, "best_val_loss": self.best_val_loss,
            "total_seconds": self.total_seconds,
        }


def run_esa_adb_fedavg(
    client_train_datasets: dict[str, SequenceDataset],
    client_validation_datasets: dict[str, SequenceDataset],
    pooled_validation_dataset: SequenceDataset,
    model_config: dict,
    training_config: dict,
    num_rounds: int,
    shared_init_seed: int,
    checkpoint_dir: Path,
) -> FedAvgTrainingResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clients = {cid: EsaAdbFederatedClient(cid, ds) for cid, ds in client_train_datasets.items()}
    initial_state_dict = build_initial_state_dict(model_config, shared_init_seed)
    server = FederatedServer(initial_state_dict)

    pooled_pos_weight = compute_pos_weight(pooled_validation_dataset.labels)  # for global val loss only, informational

    round_checkpoints_dir = checkpoint_dir / "round_checkpoints"
    round_checkpoints_dir.mkdir(parents=True, exist_ok=True)

    result = FedAvgTrainingResult()
    overall_start = time.time()

    for round_number in range(1, num_rounds + 1):
        round_start = time.time()
        global_state = server.get_global_state()  # identical copy to every client, per Phase 6's own contract

        updates = [
            clients[cid].train_round(global_state, model_config, training_config, seed=training_config["seed"])
            for cid in client_train_datasets
        ]

        new_global_state = server.aggregate(updates)  # standard sample-count-weighted FedAvg, unmodified

        global_val_loss, global_val_metrics = _evaluate(model_config, new_global_state, pooled_validation_dataset,
                                                          training_config["batch_size"], pooled_pos_weight, device)

        per_client_val = {}
        client_f1s = []
        for cid, val_ds in client_validation_datasets.items():
            c_loss, c_metrics = _evaluate(model_config, new_global_state, val_ds, training_config["batch_size"],
                                           clients[cid].pos_weight, device)
            per_client_val[cid] = {
                "loss": c_loss, "f1": c_metrics.classification.f1, "recall": c_metrics.classification.recall,
                "precision": c_metrics.classification.precision, "roc_auc": c_metrics.classification.roc_auc,
                "pr_auc": c_metrics.pr_auc, "sample_count": val_ds.num_sequences,
            }
            client_f1s.append(c_metrics.classification.f1)
        fairness_gap = max(client_f1s) - min(client_f1s)

        round_record = FedAvgRoundRecord(
            round_number=round_number, global_val_loss=global_val_loss, global_val_metrics=global_val_metrics.to_dict(),
            per_client_val=per_client_val, fairness_gap_f1=fairness_gap,
            client_updates=[u.to_summary_dict() for u in updates], round_seconds=time.time() - round_start,
        )
        result.round_history.append(round_record)

        torch.save({"round_number": round_number, "model_state_dict": new_global_state,
                    "global_val_loss": global_val_loss, "global_val_metrics": global_val_metrics.to_dict()},
                   round_checkpoints_dir / f"round_{round_number:02d}.pt")

        if global_val_loss < result.best_val_loss:
            result.best_val_loss = global_val_loss
            result.best_round = round_number
            torch.save({"round_number": round_number, "model_state_dict": new_global_state, "model_config": model_config,
                        "training_config": training_config, "global_val_loss": global_val_loss,
                        "global_val_metrics": global_val_metrics.to_dict()}, checkpoint_dir / "best_global_model.pt")

        print(f"Round {round_number:2d}/{num_rounds} | global_val_loss={global_val_loss:.4f} "
              f"global_val_f1={global_val_metrics.classification.f1:.4f} "
              f"global_val_roc_auc={global_val_metrics.classification.roc_auc:.4f} "
              f"fairness_gap_f1={fairness_gap:.4f}")

    result.total_seconds = time.time() - overall_start
    import json

    (checkpoint_dir / "round_history.json").write_text(json.dumps(result.to_dict(), indent=2))
    return result
