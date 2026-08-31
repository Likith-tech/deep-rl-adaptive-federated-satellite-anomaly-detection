# Phase 9 — Staleness- and Resource-Aware Federated Learning Results

**All numbers in this report are real, measured values from scripts/run_staleness_fl_experiments.py, read directly from experiments/staleness_fl/*/run_config.json and round_history.json, plus one KDDTest+ evaluation per config computed by this script. Nothing here is estimated, interpolated, or fabricated. Where a result is unfavorable, it is reported exactly as measured.**

## 1. Research question

Phase 4's satellite simulation has always assigned every client a simulated `availability_probability`, but no phase before this one ever let it affect round-to-round PARTICIPATION — Phase 6/8/9B all had every client fetch and train against the exact current global model every round, so no run before this one ever contained real staleness. This phase asks: when availability is finally allowed to gate whether a client receives the latest global model each round (producing real, measured staleness), does discounting stale updates before aggregation help, hurt, or not matter relative to ignoring staleness entirely?

## 2. Mechanism

Each round, each client's participation is decided by a seeded coin flip weighted by its real Phase 4 `availability_probability` (`src/federated/staleness.py:decide_participation`). A client that doesn't sync this round keeps training against whatever global state it last synced to — a REAL, structurally-caused staleness (not simulated after the fact), since that update genuinely reflects an older base model. Every client still trains and submits an update every round (staleness WEIGHTING, not selection — the same philosophy Phase 8 already committed to, for the same reason: a dropped client's rare-category data contributes nothing that round). The resulting staleness is converted to a discount factor and combined with the client's sample count, then applied through the UNMODIFIED `FederatedServer.aggregate_with_weights` (Phase 8's real extension point).

```
no_correction:      w_stale = 1                    (staleness measured but ignored)
linear_decay:       w_stale = 1 / (1 + staleness)
exponential_decay:  w_stale = 0.5 ** staleness
```

All three configs share the exact same `participation_seed`, so the real staleness PATTERN each client experiences is identical across configs — only the aggregation response to that pattern differs. Full formulas in `src/federated/staleness.py`; full config in `configs/staleness_fl.yaml`.

## 3. Experimental configuration

Identical to Phase 6/8 in every respect except staleness handling: 8 clients, the SAME `data/partitions/` (Phase 4 original, alpha=0.5) partition, 10 rounds, 1 local epoch, batch size 128, learning rate 0.001, Adam, BCEWithLogitsLoss, seed 42, shared initial weights (seed 42).

## 4. Main comparison table

| Config | Val F1 (best round) | Test F1 | Test ROC-AUC | Client fairness gap | Max staleness seen |
|---|---:|---:|---:|---:|---:|
| no_correction | 0.9851 | 0.7371 | 0.8515 | 0.1964 | 3 |
| linear_decay | 0.9853 | 0.7354 | 0.8556 | 0.1964 | 3 |
| exponential_decay | 0.9854 | 0.7360 | 0.8549 | 0.1994 | 3 |
| **phase6_fedavg (reference)** | 0.9879 | 0.7404 | 0.8479 | not measured | 0 (synchronous) |
| **phase8_best_f1 (resource_only, reference)** | not comparable | 0.7547 | 0.8438 | see performance_only below | 0 (synchronous) |
| **phase9b_dqn (reference)** | not comparable | 0.7425 | 0.8361 | 0.1833 | 0 (synchronous) |

## 5. Honest reading

Best KDDTest+ F1 among the three staleness configs: **no_correction** (0.7371). Best client fairness gap: **linear_decay** (0.1964). 
The staleness-blind control (`no_correction`) scored best on F1 among the three configs — staleness-aware discounting did NOT improve on ignoring staleness in this single-seed run. Reported exactly as measured, not adjusted or re-run to seek a more favorable result.

Against the wider Phase 6/8/9B comparison, the best staleness config here (no_correction, F1 0.7371) did NOT beat Phase 6 FedAvg (0.7404) and did NOT beat Phase 8's best rule (resource_only, 0.7547).

## 6. Validation results (best round per config)

| Config | Best round | Val F1 | Val loss |
|---|---:|---:|---:|
| no_correction | 10 | 0.9851 | 0.0354 |
| linear_decay | 9 | 0.9853 | 0.0348 |
| exponential_decay | 9 | 0.9854 | 0.0348 |

## 7. Final KDDTest+ results

Evaluated ONCE per config, after round selection by validation loss only:

| Config | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---:|---:|---:|---:|---:|---:|
| no_correction | 0.7503 | 0.9199 | 0.6149 | 0.7371 | 0.8515 | 0.0707 |
| linear_decay | 0.7490 | 0.9192 | 0.6129 | 0.7354 | 0.8556 | 0.0712 |
| exponential_decay | 0.7495 | 0.9197 | 0.6134 | 0.7360 | 0.8549 | 0.0707 |

## 8. Fairness results

**Caveat (same as Phase 7/8):** these per-client numbers evaluate each config's final selected model on every client's OWN local training data (not a held-out split, since none exists per client) — a proxy for fairness, not a clean generalization test.

| Config | Mean local F1 | Min | Max | Gap (max-min) |
|---|---:|---:|---:|---:|
| no_correction | 0.9553 | 0.8017 | 0.9981 | 0.1964 |
| linear_decay | 0.9554 | 0.8017 | 0.9981 | 0.1964 |
| exponential_decay | 0.9550 | 0.7986 | 0.9981 | 0.1994 |

## 9. Limitations

- Single run, single seed per config — not a statistically powered comparison, same caveat every prior phase (6, 7, 8, 9B) already carried.
- `participation_seed` is fixed and shared across configs by design (so the staleness pattern itself is identical), but this means only one specific random staleness pattern was tested, not a distribution of possible patterns.
- `decay_rate=0.5` for `exponential_decay` was chosen as a round, interpretable default before any result was seen, not tuned.
- The client-fairness proxy (own local training data, not a held-out split) carries the same limitation Phase 7/8 already documented.

## 10. Exact artifact locations

```
experiments/staleness_fl/<config_name>/run_config.json
experiments/staleness_fl/<config_name>/round_history.json
results/plots/staleness_fl/*.png
results/reports/staleness_fl_results.md   (this file)
```