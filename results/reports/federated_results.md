# Phase 6 — Federated Learning Baseline Results (FedAvg)

## 1. Purpose

Establish the first genuine Federated Learning result: can standard synchronous FedAvg, training the same Phase 2 MLP architecture across the 8 non-IID satellite clients without sharing raw data, match or beat (a) the Phase 2 centralized model on KDDTest+ and (b) what any single satellite could achieve training alone (Phase 5)?

## 2. FedAvg architecture

Standard McMahan et al. synchronous FedAvg: `w_global = sum_k (n_k / N) * w_k`, where `w_k` is client k's local model after local training and `n_k` its local sample count. Implemented in `src/federated/fedavg.py` (pure, unit-tested aggregation), `src/federated/client.py` (local training only), `src/federated/server.py` (aggregation only — never reads a parquet file, verified by a dedicated test), and `src/federated/trainer.py` (orchestrates the round loop). No raw client data is ever sent to the server — only model parameters and sample counts (`src/federated/protocol.py`).

## 3. Client setup

The 8 simulated satellites from Phase 4 (`data/partitions/`), each training on ONLY its own real NSL-KDD local partition. See `docs/project-progress/05-phase-4-satellite-simulation.md`.

## 4. Number of clients

8 clients, ALL participating every round (`clients_per_round: 8` — no adaptive selection in this phase).

## 5. Communication rounds

10 rounds.

## 6. Local epochs

1 local epoch(s) per client per round.

## 7. FedAvg weighting

Sample-count weighted (`sample_count`) — NOT equal weighting. Client k's contribution weight = its local sample count / total samples across all participating clients that round.

## 8. Training configuration

- Optimizer: adam (Adam)
- Learning rate: 0.001
- Loss: bce_with_logits (BCEWithLogitsLoss)
- Batch size: 128
- Seed: 42
- Shared initial-weights seed: 42 (Round 1's global model, and every client's starting point each round, comes from this single shared seed)
- Validation strategy: global validation set (`data/processed/validation.parquet`) evaluated after every round; used ONLY for round selection. KDDTest+ evaluated once, after the round was already chosen.

## 9. Round-by-round validation metrics

| Round | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |
|------:|---------:|----------:|-------:|---:|--------:|----:|
| 1 | 0.9712 | 0.9944 | 0.9435 | 0.9683 | 0.9966 | 0.0047 |
| 2 | 0.9750 | 0.9932 | 0.9527 | 0.9725 | 0.9983 | 0.0056 |
| 3 | 0.9827 | 0.9931 | 0.9695 | 0.9812 | 0.9987 | 0.0058 |
| 4 | 0.9849 | 0.9944 | 0.9731 | 0.9836 | 0.9989 | 0.0048 |
| 5 | 0.9853 | 0.9945 | 0.9738 | 0.9841 | 0.9990 | 0.0047 |
| 6 | 0.9853 | 0.9948 | 0.9735 | 0.9840 | 0.9990 | 0.0045 |
| 7 | 0.9862 | 0.9963 | 0.9740 | 0.9850 | 0.9990 | 0.0032 |
| 8 | 0.9874 | 0.9965 | 0.9764 | 0.9863 | 0.9990 | 0.0030 |
| 9 | 0.9879 | 0.9968 | 0.9773 | 0.9869 | 0.9990 | 0.0028 |
| 10 | 0.9888 | 0.9970 | 0.9790 | 0.9879 | 0.9990 | 0.0026 |

See `results/plots/federated/val_f1_per_round.png`, `val_loss_per_round.png`, and `val_accuracy_per_round.png` for the visual versions.

## 10. Best validation round

**Round 10** (selected by lowest global validation loss, 0.0346; validation F1 at that round: 0.9879). KDDTest+ was NOT consulted for this selection.

## 11. Final KDDTest+ metrics

Evaluated ONCE, after round selection, on the untouched KDDTest+ set:

| Metric | Value |
|---|---:|
| Accuracy | 0.7531 |
| Precision | 0.9219 |
| Recall | 0.6186 |
| F1 | 0.7404 |
| ROC-AUC | 0.8479 |
| False Positive Rate | 0.0693 |

## 12. Confusion matrix (KDDTest+)

```
                Predicted Normal   Predicted Anomaly
Actual Normal               9038                  673
Actual Anomaly              4894                 7939
```

See `results/plots/federated/confusion_matrix.png` for the visual version.

## 13. Comparison with centralized baseline

All three rows below are FINAL KDDTest+ results — a fair, like-for-like comparison (unlike Phase 5, which only had global-validation numbers available):

| Metric | Phase 2 centralized (KDDTest+) | Phase 6 FedAvg (KDDTest+) |
|---|---:|---:|
| Accuracy | 0.7832 | 0.7531 |
| Precision | 0.9278 | 0.9219 |
| Recall | 0.6713 | 0.6186 |
| F1 | 0.7790 | 0.7404 |
| ROC-AUC | 0.8972 | 0.8479 |
| FPR | 0.0690 | 0.0693 |

FedAvg did NOT beat the centralized baseline on KDDTest+ — its F1 was 0.0386 lower (0.7404 vs 0.7790). (Real, measured difference — not adjusted or rounded favorably.)

## 14. Comparison with local-only baseline

**Important dataset caveat:** Phase 5's numbers below are GLOBAL-VALIDATION results (KDDTest+ was intentionally not used in Phase 5). Phase 6's FedAvg number in this section is ALSO shown on global validation for a like-for-like comparison — its KDDTest+ number is in section 13 instead, compared only against the other KDDTest+ result (Phase 2).

| Metric (global validation) | Phase 5 local-only (mean across 8 clients) | Phase 5 local-only (range) | Phase 6 FedAvg (best round) |
|---|---:|---:|---:|
| F1 | 0.9721 | 0.9242 – 0.9918 | 0.9879 |

FedAvg's validation F1 (0.9879) exceeds the mean local-only client performance (0.9721), though not the single best local client (0.9918). This is a validation-set comparison only — see section 13 for the KDDTest+-only comparison against Phase 2.

## 15. Communication-cost estimate

- Model parameters: 23,937
- Bytes per model transfer (float32): 95,748 bytes (~93.5 KB)
- Upload (clients -> server) per round: 765,984 bytes (~748.0 KB)
- Download (server -> clients) per round: 765,984 bytes (~748.0 KB)
- Total estimated communication, all 10 rounds: 15,319,680 bytes (~14.61 MB)

**This is a simulation estimate of parameter transfer size only** (raw float32 parameter count × 4 bytes) — it is NOT measured real network traffic and does NOT account for the simulated per-client bandwidth/latency from Phase 4 (those are preserved in round logs for later phases but do not affect this phase's training or this estimate).

## 16. Limitations

- Standard synchronous FedAvg only: no adaptive client selection, no staleness-aware aggregation, no FedProx, no personalization, no secure aggregation, no differential privacy, no resource-based client weighting.
- ALL 8 clients participate every round — no partial participation or dropout modeling yet.
- The satellite/client environment is simulated (Phase 4); only the underlying NSL-KDD records are real.
- NSL-KDD remains a terrestrial dataset — not real satellite telemetry.
- No DRL/DQN/reinforcement learning of any kind is used to guide this phase's client selection or aggregation.
- Simulated resource metadata (bandwidth/latency/compute/availability/connectivity) is recorded per round but does not influence training in this phase.
- Round count and local-epoch count were chosen for a clean, interpretable baseline, not tuned via a large sweep.

## 17. What this enables next

Phase 7 (Adaptive Federated Learning) can now build on a real, honestly-measured FedAvg baseline: does adapting client selection or aggregation (e.g. using the simulated resource metadata already being logged) improve on this baseline's KDDTest+ result, its communication cost, or both? No adaptive selection, staleness handling, or DRL has been implemented in Phase 6 — standard FedAvg only.