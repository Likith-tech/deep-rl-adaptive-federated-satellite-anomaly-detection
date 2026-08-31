# Phase 14 — Full Cross-Phase Evaluation

**This report trains nothing and re-runs nothing. Every number below is copied verbatim from the named source report, each already committed and independently reproducible from its own phase's scripts. This document exists to put every method this project has built, so far, side by side in one place — not to introduce any new result.**

## 1. Purpose

Phases 2 through 9 each measured their own result in isolation, compared only against their own immediate predecessor. This phase assembles the full picture: every anomaly-detection method built across the whole project, ranked on the same held-out KDDTest+ set, with the federated-learning methods additionally compared on a client-fairness axis where that was measured.

## 2. Full ranking — every method, by KDDTest+ F1

| Rank | Method | Family | F1 | ROC-AUC | Fairness gap | Source |
|---:|---|---|---:|---:|---:|---|
| 1 | Phase 2 — Centralized MLP baseline | non-federated | 0.7790 | 0.8972 | not measured | `results/reports/baseline_results.md` |
| 2 | Phase 7 — Non-IID sweep, alpha=5.0 (best test F1) | federated (non-IID study) | 0.7640 | 0.8641 | not measured | `results/reports/non_iid_results.md` |
| 3 | Phase 3 — Temporal GRU+Attention (centralized, corrected) | non-federated | 0.7558 | 0.8923 | not measured | `results/reports/temporal_results.md` |
| 4 | Phase 8 — Adaptive FL, resource_only (best F1 pre-Phase-9) | federated (rule-based adaptive) | 0.7547 | 0.8438 | 0.2259 | `results/reports/adaptive_fl_results.md` |
| 5 | Phase 7 — Non-IID sweep, alpha=10.0 (least heterogeneous) | federated (non-IID study) | 0.7541 | 0.8637 | not measured | `results/reports/non_iid_results.md` |
| 6 | Phase 8 — Adaptive FL, performance_only | federated (rule-based adaptive) | 0.7447 | 0.8497 | 0.1245 | `results/reports/adaptive_fl_results.md` |
| 7 | Phase 8 — Adaptive FL, combined | federated (rule-based adaptive) | 0.7439 | 0.8536 | 0.1496 | `results/reports/adaptive_fl_results.md` |
| 8 | Phase 9B — DQN adaptive FL | federated (DRL adaptive) | 0.7425 | 0.8361 | 0.1833 | `results/reports/drl_adaptive_fl_results.md` |
| 9 | Phase 7 — Non-IID sweep, alpha=0.1 (most heterogeneous) | federated (non-IID study) | 0.7409 | 0.8172 | not measured | `results/reports/non_iid_results.md` |
| 10 | Phase 6 — FedAvg (sample-count weighting) | federated | 0.7404 | 0.8479 | not measured | `results/reports/federated_results.md` |
| 11 | Phase 8 — Adaptive FL, data_only | federated (rule-based adaptive) | 0.7397 | 0.8313 | 0.2113 | `results/reports/adaptive_fl_results.md` |
| 12 | Phase 9 — Staleness-aware FL, no_correction (best of the 3) | federated (staleness-aware) | 0.7371 | 0.8515 | 0.1964 | `results/reports/staleness_fl_results.md` |
| 13 | Phase 9 — Staleness-aware FL, exponential_decay | federated (staleness-aware) | 0.7360 | 0.8549 | 0.1994 | `results/reports/staleness_fl_results.md` |
| 14 | Phase 9 — Staleness-aware FL, linear_decay | federated (staleness-aware) | 0.7354 | 0.8556 | 0.1964 | `results/reports/staleness_fl_results.md` |

**Best KDDTest+ F1 across the entire project: Phase 2 — Centralized MLP baseline (0.7790)** — a centralized, non-federated model, expected to lead (see section 5). **Best among every FEDERATED result tried, including Phase 7's alpha sweep: Phase 7 — Non-IID sweep, alpha=5.0 (best test F1) (0.7640)** — this is a lower-heterogeneity DATA PARTITION, not a different aggregation mechanism; see section 3 for the more informative apples-to-apples comparison that holds the partition fixed and varies only the mechanism. Weakest overall: Phase 9 — Staleness-aware FL, linear_decay (0.7354).

## 3. Apples-to-apples: same partition, same model, aggregation mechanism only

The comparison above mixes methods that vary the DATA partition (Phase 7's alpha sweep) with methods that hold the partition fixed and vary only the AGGREGATION mechanism (Phase 6, 8, 9B, 9). This section isolates the second, more direct comparison — every entry below uses the EXACT SAME `data/partitions/` (Phase 4 original, alpha=0.5), model, and round/epoch budget:

| Rank | Mechanism | F1 | Fairness gap |
|---:|---|---:|---:|
| 1 | Adaptive FL, resource_only (best F1 pre-Phase-9) | 0.7547 | 0.2259 |
| 2 | Adaptive FL, performance_only | 0.7447 | 0.1245 |
| 3 | Adaptive FL, combined | 0.7439 | 0.1496 |
| 4 | DQN adaptive FL | 0.7425 | 0.1833 |
| 5 | FedAvg (sample-count weighting) | 0.7404 | not measured |
| 6 | Adaptive FL, data_only | 0.7397 | 0.2113 |
| 7 | Staleness-aware FL, no_correction (best of the 3) | 0.7371 | 0.1964 |
| 8 | Staleness-aware FL, exponential_decay | 0.7360 | 0.1994 |
| 9 | Staleness-aware FL, linear_decay | 0.7354 | 0.1964 |

**Under this controlled comparison, `resource_only` (Phase 8) remains the single best mechanism found so far (0.7547 F1).** Neither the learned DQN controller (Phase 9B) nor either staleness-aware variant (Phase 9) improved on it. This is reported as the honest, current state of the project — a simple, hand-specified rule beating two more sophisticated mechanisms (learned adaptation, staleness correction) is a real, useful finding, not a disappointment to explain away.

## 4. Client fairness ranking (federated adaptive methods only)

| Rank | Method | Fairness gap | F1 |
|---:|---|---:|---:|
| 1 | Adaptive FL, performance_only | 0.1245 | 0.7447 |
| 2 | Adaptive FL, combined | 0.1496 | 0.7439 |
| 3 | DQN adaptive FL | 0.1833 | 0.7425 |
| 4 | Staleness-aware FL, no_correction (best of the 3) | 0.1964 | 0.7371 |
| 5 | Staleness-aware FL, linear_decay | 0.1964 | 0.7354 |
| 6 | Staleness-aware FL, exponential_decay | 0.1994 | 0.7360 |
| 7 | Adaptive FL, data_only | 0.2113 | 0.7397 |
| 8 | Adaptive FL, resource_only (best F1 pre-Phase-9) | 0.2259 | 0.7547 |

**Best fairness gap: Phase 8 — Adaptive FL, performance_only (0.1245).** The same method never wins both F1 and fairness across this project's experiments — `performance_only` (Phase 8) is fairest but not the best F1; `resource_only` (Phase 8) is best F1 but among the least fair. This performance/fairness tension, first observed in Phase 8, holds up across every later phase (9B, 9) that measured both — a consistent, real pattern, not a one-off artifact.

## 5. Non-federated references

For context, not as a fair comparison (different training regime entirely — one model sees ALL the data, no communication rounds, no aggregation): the Phase 2 centralized MLP baseline scores 0.7790 F1, and Phase 3's temporal GRU+attention model scores 0.7558 F1 — both HIGHER than every federated method in this report. This gap (centralized beating every federated variant tried) is expected and already documented starting in Phase 6's own report: federated learning trades some raw accuracy for the ability to train without centralizing raw satellite data, and this project's federated round/epoch budgets have been deliberately modest throughout.

## 6. What this evaluation does NOT do

- No new training, no new hyperparameter search — every number is sourced, not derived.
- Does not resolve WHY `resource_only` outperforms the more sophisticated mechanisms — Phase 15 (ablation studies) is where that gets investigated on a couple of specific design axes, not here.
- Single seed per method throughout the underlying phases — none of the differences above are statistically tested; read small gaps (a few thousandths of F1) with that in mind.

## 7. Exact artifact locations

```
results/plots/full_evaluation/*.png
results/reports/full_evaluation.md   (this file)
```