# Phase 5 — Local Satellite Training Results

## 1. Purpose

Establish how well each simulated satellite client can detect anomalies using ONLY its own local, non-IID training data, with no communication or aggregation between clients. This is the reference point Federated Learning (Phase 6+) will be compared against.

## 2. Local satellite setup

Satellite clients and their local data partitions come from Phase 4 (`data/partitions/`, Dirichlet non-IID partitioning, alpha=0.5, seed=42). Every local data record is a REAL NSL-KDD training record; the satellite/client framing itself is SIMULATED. See `docs/project-progress/05-phase-4-satellite-simulation.md`.

## 3. Number of clients

8 clients trained: SAT-01, SAT-02, SAT-03, SAT-04, SAT-05, SAT-06, SAT-07, SAT-08.

## 4. Local data sizes

| Client | Total samples | Anomaly % | Categories present | Categories absent |
|---|---:|---:|---|---|
| SAT-01 | 8,675 | 99.07% | normal, dos, probe, r2l, u2r | none |
| SAT-02 | 39,030 | 63.89% | normal, dos, probe, r2l, u2r | none |
| SAT-03 | 6,813 | 16.20% | normal, dos, probe, r2l, u2r | none |
| SAT-04 | 6,246 | 81.89% | normal, dos, probe, r2l, u2r | none |
| SAT-05 | 8,382 | 39.19% | normal, dos, probe, r2l, u2r | none |
| SAT-06 | 4,819 | 43.76% | normal, dos, probe, r2l, u2r | none |
| SAT-07 | 19,144 | 17.47% | normal, dos, probe, u2r | r2l |
| SAT-08 | 13,968 | 9.65% | normal, dos, probe, r2l, u2r | none |

## 5. Model architecture

Same architecture as the Phase 2 baseline MLP: input(121) -> Dense(128) -> ReLU -> Dropout(0.3) -> Dense(64) -> ReLU -> Dropout(0.3) -> Dense(1) -> logit (sigmoid at inference). Every client trains an independent copy, all starting from the SAME shared initial weights (see section 7 below).

## 6. Preprocessing

No preprocessing is refit per client. Every client's parquet file already contains the same one-hot-encoded, standard-scaled feature representation produced once by the Phase 1 pipeline (`results/models/preprocessing/{scaler,encoder}.joblib`) before Phase 4 partitioned the training data. Feature columns are identical and in the same order across all clients (`results/models/preprocessing/feature_columns.json`).

## 7. Training configuration

- Optimizer: adam (Adam)
- Learning rate: 0.001
- Loss: bce_with_logits (BCEWithLogitsLoss)
- Batch size: 128
- Max epochs: 25
- Early stopping patience: 5 epochs (on global validation loss)
- Training seed: 42
- Shared initial-weights seed: 42 (every client starts from an identical initial model state, so differences in outcomes reflect local data, not initialization)
- Validation strategy: global validation set (`data/processed/validation.parquet`) evaluated after every epoch, for checkpoint selection AND as the common cross-client reference. KDDTest+ untouched.

## 8. Local training metrics

Metrics computed on each client's OWN local training data (the data it was trained on) — reflects what the model learned about its own local distribution, not generalization.

| Client | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---:|---:|---:|---:|---:|---:|
| SAT-01 | 0.9997 | 0.9998 | 0.9999 | 0.9998 | 0.9997 | 0.0247 |
| SAT-02 | 0.9979 | 0.9979 | 0.9989 | 0.9984 | 1.0000 | 0.0038 |
| SAT-03 | 0.9937 | 0.9889 | 0.9719 | 0.9804 | 0.9995 | 0.0021 |
| SAT-04 | 0.9936 | 0.9971 | 0.9951 | 0.9961 | 0.9994 | 0.0133 |
| SAT-05 | 0.9932 | 0.9930 | 0.9896 | 0.9913 | 0.9994 | 0.0045 |
| SAT-06 | 0.9929 | 0.9948 | 0.9891 | 0.9919 | 0.9997 | 0.0041 |
| SAT-07 | 0.9941 | 0.9988 | 0.9677 | 0.9830 | 0.9990 | 0.0003 |
| SAT-08 | 0.9921 | 0.9578 | 0.9607 | 0.9593 | 0.9991 | 0.0045 |

## 9. Global validation metrics

Metrics computed on the SAME held-out global validation set for every client — this is the fair, common basis for comparing clients against each other and later against Federated Learning.

| Client | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---:|---:|---:|---:|---:|---:|
| SAT-01 | 0.9240 | 0.8623 | 0.9958 | 0.9242 | 0.9906 | 0.1385 |
| SAT-02 | 0.9924 | 0.9951 | 0.9885 | 0.9918 | 0.9991 | 0.0043 |
| SAT-03 | 0.9788 | 0.9952 | 0.9592 | 0.9768 | 0.9961 | 0.0041 |
| SAT-04 | 0.9732 | 0.9668 | 0.9760 | 0.9714 | 0.9908 | 0.0292 |
| SAT-05 | 0.9761 | 0.9921 | 0.9562 | 0.9738 | 0.9940 | 0.0066 |
| SAT-06 | 0.9919 | 0.9900 | 0.9926 | 0.9913 | 0.9991 | 0.0087 |
| SAT-07 | 0.9723 | 0.9995 | 0.9409 | 0.9693 | 0.9931 | 0.0004 |
| SAT-08 | 0.9799 | 0.9923 | 0.9643 | 0.9781 | 0.9976 | 0.0065 |

## 10. Per-client comparison

| Satellite | Samples | Anomaly % | Val Accuracy | Val Precision | Val Recall | Val F1 | Val ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| SAT-01 | 8,675 | 99.07% | 0.9240 | 0.8623 | 0.9958 | 0.9242 | 0.9906 |
| SAT-02 | 39,030 | 63.89% | 0.9924 | 0.9951 | 0.9885 | 0.9918 | 0.9991 |
| SAT-03 | 6,813 | 16.20% | 0.9788 | 0.9952 | 0.9592 | 0.9768 | 0.9961 |
| SAT-04 | 6,246 | 81.89% | 0.9732 | 0.9668 | 0.9760 | 0.9714 | 0.9908 |
| SAT-05 | 8,382 | 39.19% | 0.9761 | 0.9921 | 0.9562 | 0.9738 | 0.9940 |
| SAT-06 | 4,819 | 43.76% | 0.9919 | 0.9900 | 0.9926 | 0.9913 | 0.9991 |
| SAT-07 | 19,144 | 17.47% | 0.9723 | 0.9995 | 0.9409 | 0.9693 | 0.9931 |
| SAT-08 | 13,968 | 9.65% | 0.9799 | 0.9923 | 0.9643 | 0.9781 | 0.9976 |

See `results/plots/local_training/client_comparison.png` for the visual version.

## 11. Aggregate statistics (across all clients, global validation)

| Metric | Mean | Median | Min | Max | Std dev |
|---|---:|---:|---:|---:|---:|
| accuracy | 0.9736 | 0.9775 | 0.9240 | 0.9924 | 0.0215 |
| precision | 0.9741 | 0.9922 | 0.8623 | 0.9995 | 0.0463 |
| recall | 0.9717 | 0.9702 | 0.9409 | 0.9958 | 0.0197 |
| f1 | 0.9721 | 0.9753 | 0.9242 | 0.9918 | 0.0211 |
| roc_auc | 0.9951 | 0.9951 | 0.9906 | 0.9991 | 0.0034 |
| false_positive_rate | 0.0248 | 0.0066 | 0.0004 | 0.1385 | 0.0468 |

## 12. Non-IID observations

Best global-validation F1: **SAT-02** (0.9918), local anomaly rate 63.89%, 39,030 samples, missing categories: none.

Worst global-validation F1: **SAT-01** (0.9242), local anomaly rate 99.07%, 8,675 samples, missing categories: none.

Clients missing at least one attack category in their local training data: SAT-07 (missing: r2l).

Observed pattern (association, not proven causation — see caveat below): clients with extreme local anomaly rates (very high or very low) and/or missing attack categories tend to sit at the extremes of the global-validation F1 ranking above. This is consistent with, but does not by itself prove, that non-IID local data limits what a purely local model can learn.

**Caveat:** with only 8 clients, this is a small, observational sample — good for describing what happened in this run, not for establishing statistical causation. Language above is deliberately correlational, not causal.

## 13. Comparison to centralized baseline

**Important dataset caveat:** the Phase 2 numbers below are FINAL, one-time KDDTest+ results. The Phase 5 numbers are global-VALIDATION results (used for development/checkpoint selection, per the project's rule that KDDTest+ stays untouched until final federated evaluation). These are NOT a like-for-like test-set comparison — shown side by side only to give a rough sense of scale, not a rigorous benchmark.

| Metric | Phase 2 centralized (KDDTest+, final) | Phase 5 aggregate mean (global validation) |
|---|---:|---:|
| Accuracy | 0.7832 | 0.9736 |
| Precision | 0.9278 | 0.9741 |
| Recall | 0.6713 | 0.9717 |
| F1 | 0.7790 | 0.9721 |
| ROC-AUC | 0.8972 | 0.9951 |
| FPR | 0.0690 | 0.0248 |

## 14. Limitations

- NSL-KDD is a terrestrial network intrusion dataset — it is not, and has never been claimed to be, real satellite telemetry.
- The satellite/client environment (client IDs, and the fact that this is 'satellites' at all) is simulated; only the underlying network records are real.
- Local models never communicate with each other in this phase — no aggregation, no FedAvg, no global model. Any similarity between clients' models is coincidental (same architecture, same initial weights), not the result of information sharing.
- Phase 5 metrics use the global VALIDATION set, not KDDTest+ — see the explicit caveat in section 13.
- With only 8 clients, aggregate statistics (mean/std/etc.) are based on a small sample; treat spread numbers as descriptive, not as a statistically powered study.
- Some clients have very few or zero samples of rare attack categories (R2L, U2R) purely because those categories are rare in the underlying real dataset overall — not a partitioning bug.

## 15. What this enables next

Phase 6 (Federated Learning Baseline) can now be evaluated against a real, honestly-measured local-only reference: does aggregating clients' updates (e.g. via FedAvg) actually improve on what any single satellite could achieve alone, and does it close the gap to the Phase 2 centralized result? No Federated Learning, DRL, or model aggregation has been implemented in Phase 5 — environment (Phase 4) and local-only training (Phase 5) only.