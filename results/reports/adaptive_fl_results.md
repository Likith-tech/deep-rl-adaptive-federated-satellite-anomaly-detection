# Phase 8 — Rule-Based Adaptive Federated Learning Results

## Main comparison table

| Rule | Val F1 (best round) | Test F1 | Test ROC-AUC | Client fairness gap |
|---|---:|---:|---:|---:|
| performance_only | 0.9890 | 0.7447 | 0.8497 | 0.1245 |
| resource_only | 0.9860 | 0.7547 | 0.8438 | 0.2259 |
| data_only | 0.9850 | 0.7397 | 0.8313 | 0.2113 |
| combined | 0.9883 | 0.7439 | 0.8536 | 0.1496 |
| **phase6_fedavg (reference)** | 0.9879 | 0.7404 | 0.8479 | not measured in Phase 6 |

## 1. Purpose

Establish a transparent, DETERMINISTIC (not reinforcement-learned) rule-based adaptive FedAvg baseline, sitting between Phase 6 (standard FedAvg) and the future Phase 9 (DRL-based adaptive FL). Every client still participates every round — this phase changes ONLY how much each client's update counts toward the aggregated global model, via a fixed, documented arithmetic rule with no learned parameters and no reward signal.

## 2. Why Phase 6 was insufficient

Phase 6 weights every client purely by local sample count. Phase 7 showed that non-IID heterogeneity barely moves the overall KDDTest+ score but devastates PER-CLIENT fairness (~41x gap between the best- and worst-served satellite across tested alphas). Sample-count weighting has no mechanism to respond to that — a large but unhelpful or narrow-category client gets the same automatic priority as a small, diverse, currently-useful one.

## 3. Adaptive rule / formula

```
client_score_k = w_perf * performance_k + w_data * data_k
               + w_resource * resource_k + w_fair * fairness_k
```

All four raw signals are min-max normalized to [0, 1] across the participating clients each round (fairness/data/resource are static per experiment; performance is recomputed every round from that round's local update, evaluated on global validation). Scores are renormalized to sum to 1 and used directly as FedAvg aggregation weights (`src/federated/server.py:aggregate_with_weights`, additive on top of the unmodified Phase 6 `aggregate` method). Full formula and component rationale in `src/federated/adaptive.py` and `configs/adaptive_fl.yaml`.

**Component rationale:**
- **performance**: this round's local update's F1 on global validation — rewards updates currently helping the shared model.
- **data**: local training sample count — the classic FedAvg intuition, now one signal among several.
- **resource**: simulated Phase 4 conditions (bandwidth, compute, availability, connectivity — higher better; latency inverted). SIMULATED, not real satellite telemetry.
- **fairness**: Shannon entropy of the client's local attack-category distribution (same function as Phase 7) — explicitly counteracts high performers drowning out clients holding rare categories (r2l/u2r).

## 4. Client-selection or weighting mechanism

AGGREGATION WEIGHTING, not selection — all 8 clients participate every round for every rule config, chosen deliberately as the safer, more defensible mechanism (hard selection risks silently dropping a rare-category client for an entire round). A top-k selection utility is implemented and unit-tested (`select_top_k_clients`) for later reuse but not separately benchmarked in this phase's experiments.

## 5. Experimental configuration

Identical to Phase 6 in every respect except the aggregation rule: 8 clients, the SAME `data/partitions/` (Phase 4 original, alpha=0.5) partition — not a fresh one — 10 rounds, 1 local epoch, batch size 128, learning rate 0.001, Adam, BCEWithLogitsLoss, seed 42, shared initial weights (seed 42). Rule weights (section 3) were fixed BEFORE any KDDTest+ evaluation ran.

## 6. Validation results (best round per rule)

| Rule | Best round | Val F1 | Val loss |
|---|---:|---:|---:|
| performance_only | 10 | 0.9890 | 0.0322 |
| resource_only | 8 | 0.9860 | 0.0403 |
| data_only | 9 | 0.9850 | 0.0408 |
| combined | 10 | 0.9883 | 0.0333 |

## 7. Final KDDTest+ results

Evaluated ONCE per rule config, after round selection by validation loss only:

| Rule | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---:|---:|---:|---:|---:|---:|
| performance_only | 0.7563 | 0.9224 | 0.6244 | 0.7447 | 0.8497 | 0.0694 |
| resource_only | 0.7631 | 0.9190 | 0.6402 | 0.7547 | 0.8438 | 0.0746 |
| data_only | 0.7526 | 0.9223 | 0.6175 | 0.7397 | 0.8313 | 0.0688 |
| combined | 0.7556 | 0.9217 | 0.6236 | 0.7439 | 0.8536 | 0.0700 |
| phase6_fedavg (reference) | 0.7531 | 0.9219 | 0.6186 | 0.7404 | 0.8479 | 0.0693 |

## 8. FedAvg vs. Adaptive FL comparison

The 'combined' rule improved on Phase 6 FedAvg's KDDTest+ F1 by +0.0035. Reported exactly as measured — the rule weights were not adjusted after seeing this number.

**Observed pattern, reported exactly as measured:** the rule with the highest KDDTest+ F1 was **resource_only** (0.7547), not `combined`. The rule with the SMALLEST fairness gap was **performance_only** (0.1245) — NOT the fairness-weighted `combined` rule, which is a genuinely counterintuitive result we are not smoothing over: explicitly weighting for local category diversity did not produce the best-measured fairness outcome in this single-seed run. A plausible explanation is that `performance` (evaluated on the shared global validation set) already implicitly favors clients whose updates generalize well, which can correlate with diversity in ways the static entropy signal does not fully capture round-to-round — but with one seed and four rule configs we cannot separate a genuine effect from noise here.

## 9. Fairness results

**Caveat (same as Phase 7):** these per-client numbers evaluate each rule's final selected model on every client's OWN local training data — a proxy for how well the model fits that client's distribution, not a held-out generalization test.

| Rule | Mean F1 | Min F1 | Max F1 | Std Dev | Max-Min Gap |
|---|---:|---:|---:|---:|---:|
| performance_only | 0.9689 | 0.8739 | 0.9985 | 0.0400 | 0.1245 |
| resource_only | 0.9533 | 0.7724 | 0.9983 | 0.0750 | 0.2259 |
| data_only | 0.9532 | 0.7869 | 0.9982 | 0.0693 | 0.2113 |
| combined | 0.9644 | 0.8486 | 0.9982 | 0.0486 | 0.1496 |

Smallest fairness gap: **performance_only** (0.1245). Largest: **resource_only** (0.2259). Since all clients participate every round in every rule (weighting, not selection), category coverage in the aggregate is 100% for every rule — no client's categories are ever entirely excluded; the fairness question here is about DEGREE of influence, not presence/absence.

## 10. Communication comparison

Identical to Phase 6 for every rule: 23,937 parameters, all 8 clients participate every round, 10 rounds — same ~14.61 MB total estimated transfer as Phase 6. Aggregation weighting changes HOW MUCH each update counts, not whether it is transmitted — communication cost would only drop if selection (not weighting) were used instead.

## 11. Ablation results

Four rule configs were run: performance_only, resource_only, data_only (each isolating one signal), and combined (the intended primary rule). See the main comparison table above and `results/plots/adaptive_fl/val_f1_comparison.png` / `client_contribution_range.png` for the visual comparison. Client SELECTION (as opposed to weighting) was implemented and unit-tested but not separately benchmarked — see section 4.

## 12. Limitations

- Rule-based only — NOT reinforcement learning; no learned parameters, no reward signal.
- Single seed (42) per rule config — not a statistically powered comparison.
- Fairness metric uses each client's own local training data as a proxy (no separate held-out per-client split exists).
- Resource signals are simulated (Phase 4), not real satellite telemetry.
- All clients participate every round in every rule — this phase did not benchmark hard client selection.
- NSL-KDD remains terrestrial network data; satellite framing is simulated throughout.

## 13. Why these results motivate adaptive Federated Learning

Whatever pattern emerges above (reported honestly, not cherry-picked), this phase's main contribution is the infrastructure and evidence needed before Phase 9: a working, tested, non-RL scoring/weighting mechanism plumbed through the exact same client/server/FedAvg code Phase 6 and Phase 9 both use, with real measured numbers for what simple, transparent rules can and cannot achieve. Phase 9's DRL controller will be compared against THIS baseline, not just against Phase 6.

## 14. What Phase 9 will do

Replace the fixed rule weights with a DRL (DQN) controller that learns client-weighting/selection decisions from a reward signal, using the same signals (performance, data, resource, fairness) this phase established as inputs — but LEARNED rather than hand-specified.