"""ESA-ADB rule-based adaptive Federated Learning (analogous to Phase
8's `src.federated.adaptive_trainer`, adapted to ESA-ADB's real 8-client
Option A representation — see `src.training.esa_adb_federated` for the
FedAvg baseline this mirrors).

Reuses UNCHANGED: `EsaAdbFederatedClient` / `build_initial_state_dict`
(`src.training.esa_adb_federated`), `FederatedServer.aggregate_with_weights`
(`src.federated.server`), and the entire Phase 8 scoring pipeline
(`src.federated.adaptive.ClientSignals` / `compute_client_scores` /
`scores_to_aggregation_weights`). The only ESA-ADB-specific addition is
`src.satellite.esa_adb.adaptive_signals` (real, non-simulated
resource/fairness signals — see that module's docstring for why Phase
4's synthetic satellite simulation isn't reused here).

    for each round:
        global_state = server.get_global_state()
        updates = [client.train_round(...) for client in clients]     # UNCHANGED
        for each update: evaluate its state_dict on POOLED validation
                          (fixed 0.5 threshold) -> performance signal
        scores = compute_client_scores(signals, rule_weights)          # deterministic, no RL
        weights = scores_to_aggregation_weights(scores)
        server.aggregate_with_weights(updates, weights)                 # -> new global model
        evaluate new global model on pooled + per-client validation
        record round history (val metrics, scores, weights, signals)

Test data is never touched here — see scripts/train_esa_adb_adaptive_rule_baseline.py.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from src.federated.adaptive import ClientSignals, compute_client_scores, scores_to_aggregation_weights
from src.federated.protocol import ClientUpdate
from src.federated.server import FederatedServer
from src.preprocessing.sequences import SequenceDataset
from src.satellite.esa_adb.adaptive_signals import build_esa_adb_fairness_signals, build_esa_adb_resource_signals
from src.training.esa_adb_federated import EsaAdbFederatedClient, _evaluate, build_initial_state_dict
from src.training.esa_adb_trainer import compute_extended_metrics, compute_pos_weight, make_dataloader, run_inference


@dataclass
class AdaptiveRoundRecord:
    round_number: int
    global_val_loss: float
    global_val_metrics: dict
    per_client_val: dict
    fairness_gap_f1: float
    client_scores: dict[str, float] = field(default_factory=dict)
    aggregation_weights: dict[str, float] = field(default_factory=dict)
    client_updates: list[dict] = field(default_factory=list)
    round_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "round_number": self.round_number, "global_val_loss": self.global_val_loss,
            "global_val_metrics": self.global_val_metrics, "per_client_val": self.per_client_val,
            "fairness_gap_f1": self.fairness_gap_f1, "client_scores": self.client_scores,
            "aggregation_weights": self.aggregation_weights, "client_updates": self.client_updates,
            "round_seconds": self.round_seconds,
        }


@dataclass
class AdaptiveTrainingResult:
    round_history: list[AdaptiveRoundRecord] = field(default_factory=list)
    best_round: int = 0
    best_val_loss: float = float("inf")
    total_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "round_history": [r.to_dict() for r in self.round_history],
            "best_round": self.best_round, "best_val_loss": self.best_val_loss,
            "total_seconds": self.total_seconds,
        }


def _performance_signal(model_config: dict, state_dict: dict, pooled_val: SequenceDataset, batch_size: int, device: torch.device) -> float:
    """This round's local update's F1 on the POOLED global validation
    set, at a FIXED 0.5 threshold (round-by-round signal only, never the
    final frozen-threshold evaluation)."""
    from src.models.temporal_gru_attention import TemporalGRUAttention

    model = TemporalGRUAttention(**{k: model_config[k] for k in ("input_dim", "projection_dim", "hidden_dim", "num_gru_layers", "dropout", "output_dim")}).to(device)
    model.load_state_dict(state_dict)
    loader = make_dataloader(pooled_val, batch_size, shuffle=False)
    y_prob, y_true = run_inference(model, loader, device)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = compute_extended_metrics(y_true, y_pred, y_prob)
    return metrics.classification.f1


def run_esa_adb_adaptive_fl(
    client_train_datasets: dict[str, SequenceDataset],
    client_validation_datasets: dict[str, SequenceDataset],
    pooled_validation_dataset: SequenceDataset,
    model_config: dict,
    training_config: dict,
    num_rounds: int,
    shared_init_seed: int,
    checkpoint_dir: Path,
    rule_weights: dict[str, float],
) -> AdaptiveTrainingResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clients = {cid: EsaAdbFederatedClient(cid, ds) for cid, ds in client_train_datasets.items()}
    initial_state_dict = build_initial_state_dict(model_config, shared_init_seed)
    server = FederatedServer(initial_state_dict)

    pooled_pos_weight = compute_pos_weight(pooled_validation_dataset.labels)

    resource_signals = build_esa_adb_resource_signals(list(client_train_datasets.keys()))
    fairness_signals = build_esa_adb_fairness_signals({cid: ds.labels for cid, ds in client_train_datasets.items()})

    round_checkpoints_dir = checkpoint_dir / "round_checkpoints"
    round_checkpoints_dir.mkdir(parents=True, exist_ok=True)

    result = AdaptiveTrainingResult()
    overall_start = time.time()

    for round_number in range(1, num_rounds + 1):
        round_start = time.time()
        global_state = server.get_global_state()

        updates: list[ClientUpdate] = [
            clients[cid].train_round(global_state, model_config, training_config, seed=training_config["seed"])
            for cid in client_train_datasets
        ]

        signals = []
        for u in updates:
            performance = _performance_signal(model_config, u.state_dict, pooled_validation_dataset, training_config["batch_size"], device)
            signals.append(ClientSignals(
                client_id=u.client_id, performance=performance, data=float(u.num_samples),
                resource=resource_signals[u.client_id], fairness=fairness_signals[u.client_id],
            ))

        scores = compute_client_scores(signals, rule_weights)
        weights = scores_to_aggregation_weights(scores)

        new_global_state = server.aggregate_with_weights(updates, weights)

        global_val_loss, global_val_metrics = _evaluate(model_config, new_global_state, pooled_validation_dataset,
                                                          training_config["batch_size"], pooled_pos_weight, device)

        per_client_val = {}
        client_f1s = []
        for cid, val_ds in client_validation_datasets.items():
            c_loss, c_metrics = _evaluate(model_config, new_global_state, val_ds, training_config["batch_size"],
                                           compute_pos_weight(client_train_datasets[cid].labels), device)
            per_client_val[cid] = {
                "loss": c_loss, "f1": c_metrics.classification.f1, "recall": c_metrics.classification.recall,
                "precision": c_metrics.classification.precision, "roc_auc": c_metrics.classification.roc_auc,
                "pr_auc": c_metrics.pr_auc, "sample_count": val_ds.num_sequences,
            }
            client_f1s.append(c_metrics.classification.f1)
        fairness_gap = max(client_f1s) - min(client_f1s)

        round_record = AdaptiveRoundRecord(
            round_number=round_number, global_val_loss=global_val_loss, global_val_metrics=global_val_metrics.to_dict(),
            per_client_val=per_client_val, fairness_gap_f1=fairness_gap, client_scores=scores,
            aggregation_weights=weights, client_updates=[u.to_summary_dict() for u in updates],
            round_seconds=time.time() - round_start,
        )
        result.round_history.append(round_record)

        torch.save({"round_number": round_number, "model_state_dict": new_global_state,
                    "global_val_loss": global_val_loss, "global_val_metrics": global_val_metrics.to_dict()},
                   round_checkpoints_dir / f"round_{round_number:02d}.pt")

        if global_val_loss < result.best_val_loss:
            result.best_val_loss = global_val_loss
            result.best_round = round_number
            torch.save({"round_number": round_number, "model_state_dict": new_global_state, "model_config": model_config,
                        "training_config": training_config, "rule_weights": rule_weights,
                        "global_val_loss": global_val_loss, "global_val_metrics": global_val_metrics.to_dict()},
                       checkpoint_dir / "best_global_model.pt")

        print(f"Round {round_number:2d}/{num_rounds} | global_val_loss={global_val_loss:.4f} "
              f"global_val_f1={global_val_metrics.classification.f1:.4f} "
              f"global_val_roc_auc={global_val_metrics.classification.roc_auc:.4f} "
              f"fairness_gap_f1={fairness_gap:.4f} weights={ {k: round(v, 3) for k, v in weights.items()} }")

    result.total_seconds = time.time() - overall_start
    (checkpoint_dir / "round_history.json").write_text(json.dumps(result.to_dict(), indent=2))
    return result
