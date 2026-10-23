"""ESA-ADB DQN adaptive Federated Learning — the REAL round provider
(analogous to `src.federated.drl_round_provider.RealNSLKDDRoundProvider`,
adapted to ESA-ADB's real 8-client Option A representation).

Adapts the UNCHANGED, dataset-agnostic Phase 9B DQN infrastructure
(`src.drl.dqn_agent.DQNAgent`, `src.federated.dqn_controller.DQNController`,
`src.drl.action_space` (9 named weight profiles), `src.drl.reward`
(`r_t = delta_f1 - lambda_fair * perf_spread - lambda_stab *
switch_penalty`), `src.drl.state` (44-dim state, 8 clients x 4 signal
blocks + val_F1 trend + round progress + previous-action one-hot), and
`src.federated.drl_environment.DRLFederatedEnvironment`) to ESA-ADB by
implementing the same `(round_number, weights, signals) ->
(next_signals, new_val_f1)` `RoundProvider` contract against
`EsaAdbFederatedClient` / `FederatedServer.aggregate_with_weights`
(`src.training.esa_adb_federated`) instead of NSL-KDD's
`FederatedClient`.

PRIVACY BOUNDARY (same as the NSL-KDD provider): `performance` (the
per-update signal fed into state/reward) is computed ONLY against the
POOLED VALIDATION `SequenceDataset` — never test. `data`/`resource`/
`fairness` are static, precomputed per client from TRAIN labels and the
real channel partition only (`src.satellite.esa_adb.adaptive_signals`).
Test data is never referenced anywhere in this module.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.federated.adaptive import ClientSignals
from src.federated.protocol import ClientUpdate
from src.federated.server import FederatedServer
from src.preprocessing.sequences import SequenceDataset
from src.satellite.esa_adb.adaptive_signals import build_esa_adb_fairness_signals, build_esa_adb_resource_signals
from src.training.esa_adb_federated import EsaAdbFederatedClient, _evaluate, build_initial_state_dict
from src.training.esa_adb_trainer import compute_extended_metrics, compute_pos_weight, make_dataloader, run_inference


class EsaAdbRoundProvider:
    """Stateful adapter: one instance == one real ESA-ADB FL episode's
    worth of round-by-round training/aggregation/evaluation, driven
    externally by `DRLFederatedEnvironment`."""

    def __init__(
        self,
        client_train_datasets: dict[str, SequenceDataset],
        pooled_validation_dataset: SequenceDataset,
        model_config: dict,
        training_config: dict,
        shared_init_seed: int,
        num_rounds: int,
    ) -> None:
        self.client_train_datasets = client_train_datasets
        self.pooled_validation_dataset = pooled_validation_dataset
        self.model_config = model_config
        self.training_config = training_config
        self.num_rounds = num_rounds
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.clients = {cid: EsaAdbFederatedClient(cid, ds) for cid, ds in client_train_datasets.items()}
        self.client_ids = list(client_train_datasets.keys())

        initial_state_dict = build_initial_state_dict(model_config, shared_init_seed)
        self.server = FederatedServer(initial_state_dict)

        self.pooled_pos_weight = compute_pos_weight(pooled_validation_dataset.labels)
        self.resource_signals = build_esa_adb_resource_signals(self.client_ids)
        self.fairness_signals = build_esa_adb_fairness_signals({cid: ds.labels for cid, ds in client_train_datasets.items()})

        self._pending_updates: dict[int, list[ClientUpdate]] = {}
        self.round_eval_history: dict[int, dict] = {}
        self._last_aggregated_state: dict | None = None
        # Every round's aggregated global state_dict, keyed by round
        # number — retained so a caller can select and checkpoint the
        # best-by-validation round's actual weights WITHOUT re-running
        # the (expensive, real-data) rollout a second time.
        self.round_state_dicts: dict[int, dict] = {}

    def _performance_signal(self, state_dict: dict) -> float:
        """This local update's F1 on POOLED validation, fixed 0.5
        threshold — matches `src.training.esa_adb_adaptive`'s rule-based
        performance signal exactly, so DQN and rule-based baselines are
        directly comparable."""
        from src.models.temporal_gru_attention import TemporalGRUAttention

        model = TemporalGRUAttention(**{k: self.model_config[k] for k in ("input_dim", "projection_dim", "hidden_dim", "num_gru_layers", "dropout", "output_dim")}).to(self.device)
        model.load_state_dict(state_dict)
        loader = make_dataloader(self.pooled_validation_dataset, self.training_config["batch_size"], shuffle=False)
        y_prob, y_true = run_inference(model, loader, self.device)
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = compute_extended_metrics(y_true, y_pred, y_prob)
        return metrics.classification.f1

    def _evaluate_and_record_round(self, round_number: int, state_dict: dict) -> float:
        """Round-level evaluation of the AGGREGATED global model on
        POOLED validation — records full metrics for reporting and
        returns F1 (the only value the DQN state/reward are allowed to
        use)."""
        val_loss, val_metrics = _evaluate(self.model_config, state_dict, self.pooled_validation_dataset,
                                           self.training_config["batch_size"], self.pooled_pos_weight, self.device)
        self.round_eval_history[round_number] = {"val_loss": val_loss, "val_metrics": val_metrics.to_dict()}
        return val_metrics.classification.f1

    def _train_round(self, round_number: int) -> list[ClientUpdate]:
        global_state = self.server.get_global_state()
        updates = [
            self.clients[cid].train_round(global_state, self.model_config, self.training_config, seed=self.training_config["seed"])
            for cid in self.client_ids
        ]
        self._pending_updates[round_number] = updates
        return updates

    def _signals_from_updates(self, updates: list[ClientUpdate]) -> list[ClientSignals]:
        signals = []
        for u in updates:
            performance = self._performance_signal(u.state_dict)
            signals.append(ClientSignals(
                client_id=u.client_id, performance=performance, data=float(u.num_samples),
                resource=self.resource_signals[u.client_id], fairness=self.fairness_signals[u.client_id],
            ))
        return signals

    def initial_signals_provider(self) -> tuple[list[ClientSignals], float]:
        """`DRLFederatedEnvironment.reset()`'s injection point: evaluates
        the untrained initial global model as the baseline val_F1, then
        trains round 1's clients and returns round 1's signals."""
        initial_val_f1 = self._evaluate_and_record_round(0, self.server.get_global_state())
        updates = self._train_round(1)
        return self._signals_from_updates(updates), initial_val_f1

    def round_provider(self, round_number: int, weights: dict[str, float], signals: list[ClientSignals]) -> tuple[list[ClientSignals], float]:
        """`DRLFederatedEnvironment.step()`'s injection point: aggregates
        this round's already-trained updates via the UNCHANGED
        `FederatedServer.aggregate_with_weights`, evaluates the new
        global model, and (if the episode continues) trains the next
        round's clients so their signals are ready for the following
        state."""
        updates = self._pending_updates.pop(round_number)
        new_global_state = self.server.aggregate_with_weights(updates, weights)
        new_val_f1 = self._evaluate_and_record_round(round_number, new_global_state)
        self._last_aggregated_state = new_global_state
        self.round_state_dicts[round_number] = {k: v.clone() for k, v in new_global_state.items()}

        if round_number < self.num_rounds:
            next_updates = self._train_round(round_number + 1)
            next_signals = self._signals_from_updates(next_updates)
        else:
            next_signals = signals  # episode terminates this round; DRLFederatedEnvironment marks done=True, value never acted on

        return next_signals, new_val_f1
