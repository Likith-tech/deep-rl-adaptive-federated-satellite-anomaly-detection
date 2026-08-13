# Phase 7 — Non-IID Federated Learning Experiments

## Main comparison table

| Alpha | Mean JS | Val F1 | Test F1 | Test ROC-AUC | Client F1 Std |
|------:|--------:|-------:|--------:|-------------:|--------------:|
| 0.1 | 0.6892 | 0.9821 | 0.7409 | 0.8172 | 0.0904 |
| 0.5 | 0.5270 | 0.9879 | 0.7404 | 0.8479 | 0.0469 |
| 1.0 | 0.3419 | 0.9924 | 0.7555 | 0.8610 | 0.0076 |
| 5.0 | 0.2063 | 0.9932 | 0.7640 | 0.8641 | 0.0032 |
| 10.0 | 0.1452 | 0.9924 | 0.7541 | 0.8637 | 0.0024 |

## 1. Purpose

Measure how the DEGREE of client data heterogeneity affects standard FedAvg, using controlled experiments that vary ONLY the Dirichlet concentration parameter (alpha) while holding every other factor (model, clients, rounds, local epochs, batch size, learning rate, optimizer, loss, FedAvg weighting, validation set, test set, seeds) fixed.

## 2. What non-IID means

Each simulated satellite's local data comes from a different distribution of attack categories than the others — the opposite of every client holding a random, representative sample of everything. Lower Dirichlet alpha produces MORE skewed (non-IID) client distributions; higher alpha produces distributions closer to IID (identical across clients).

## 3. Experimental design

For each alpha, a FRESH 8-client Dirichlet partition of the real training data (`data/processed/train.parquet`, 107,077 records) was built using the same seed (42) and the same partitioning machinery as Phase 4 (`src/simulation/partitioner.py`). The ORIGINAL Phase 4/6 partition (`data/partitions/`, alpha=0.5) was never touched — every alpha in this study, including 0.5, uses its own fresh partition under `experiments/non_iid/alpha_<X>/partitions/`, so alpha=0.5 here is a separate but comparable run to Phase 6.

## 4. Alpha values

0.1, 0.5, 1.0, 5.0, 10.0

## 5. Partition statistics

| Alpha | Total samples | Constraints satisfied | Attempts used |
|---:|---:|---|---:|
| 0.1 | 107,077 | False | 20 |
| 0.5 | 107,077 | True | 1 |
| 1.0 | 107,077 | True | 1 |
| 5.0 | 107,077 | True | 1 |
| 10.0 | 107,077 | True | 1 |

All alphas accounted for every one of the 107,077 training records exactly once (verified by `verify_partition_integrity` — no loss, no duplication). Validation and KDDTest+ were never partitioned or touched by this phase.

## 6. JS-distance measurements

| Alpha | Mean JS | Min JS | Max JS | Mean Entropy (bits) |
|------:|--------:|-------:|-------:|---------------------:|
| 0.1 | 0.6892 | 0.1748 | 0.9986 | 0.8019 |
| 0.5 | 0.5270 | 0.0824 | 0.9589 | 0.8689 |
| 1.0 | 0.3419 | 0.0528 | 0.6344 | 1.1847 |
| 5.0 | 0.2063 | 0.0821 | 0.4327 | 1.3148 |
| 10.0 | 0.1452 | 0.0587 | 0.3073 | 1.3524 |

This confirms alpha actually changed measured heterogeneity — not assumed. See `results/plots/non_iid/js_distance_by_alpha.png`.

## 7. Client distributions

Per-satellite anomaly percentage and attack-category distribution for every alpha: `results/plots/non_iid/alpha_<X>_anomaly_pct.png` and `alpha_<X>_category_distribution.png`.

## 8. FedAvg configuration (identical across all alphas)

- Clients: 8 (ALL participate every round)
- Rounds: 10
- Local epochs: 1
- Batch size: 128
- Learning rate: 0.001
- Optimizer: adam (Adam)
- Loss: bce_with_logits (BCEWithLogitsLoss)
- FedAvg weighting: sample_count (sample-count weighted)
- Seed: 42, shared init seed: 42

## 9. Round-by-round validation results

**alpha=0.1**

| Round | Val F1 | Val Loss |
|------:|-------:|---------:|
| 1 | 0.9547 | 0.1760 |
| 2 | 0.9707 | 0.0800 |
| 3 | 0.9755 | 0.0750 |
| 4 | 0.9821 | 0.0704 |
| 5 | 0.9827 | 0.0759 |
| 6 | 0.9829 | 0.0843 |
| 7 | 0.9841 | 0.0880 |
| 8 | 0.9848 | 0.0899 |
| 9 | 0.9849 | 0.0933 |
| 10 | 0.9849 | 0.0995 |

**alpha=0.5**

| Round | Val F1 | Val Loss |
|------:|-------:|---------:|
| 1 | 0.9683 | 0.0954 |
| 2 | 0.9725 | 0.0575 |
| 3 | 0.9812 | 0.0438 |
| 4 | 0.9836 | 0.0391 |
| 5 | 0.9841 | 0.0371 |
| 6 | 0.9840 | 0.0361 |
| 7 | 0.9850 | 0.0354 |
| 8 | 0.9863 | 0.0351 |
| 9 | 0.9869 | 0.0350 |
| 10 | 0.9879 | 0.0346 |

**alpha=1.0**

| Round | Val F1 | Val Loss |
|------:|-------:|---------:|
| 1 | 0.9698 | 0.0893 |
| 2 | 0.9753 | 0.0577 |
| 3 | 0.9803 | 0.0406 |
| 4 | 0.9884 | 0.0313 |
| 5 | 0.9902 | 0.0282 |
| 6 | 0.9905 | 0.0267 |
| 7 | 0.9906 | 0.0257 |
| 8 | 0.9922 | 0.0250 |
| 9 | 0.9923 | 0.0241 |
| 10 | 0.9924 | 0.0237 |

**alpha=5.0**

| Round | Val F1 | Val Loss |
|------:|-------:|---------:|
| 1 | 0.9682 | 0.0872 |
| 2 | 0.9744 | 0.0591 |
| 3 | 0.9785 | 0.0445 |
| 4 | 0.9834 | 0.0363 |
| 5 | 0.9889 | 0.0317 |
| 6 | 0.9913 | 0.0282 |
| 7 | 0.9919 | 0.0265 |
| 8 | 0.9924 | 0.0255 |
| 9 | 0.9927 | 0.0249 |
| 10 | 0.9932 | 0.0245 |

**alpha=10.0**

| Round | Val F1 | Val Loss |
|------:|-------:|---------:|
| 1 | 0.9685 | 0.0856 |
| 2 | 0.9746 | 0.0588 |
| 3 | 0.9775 | 0.0443 |
| 4 | 0.9828 | 0.0360 |
| 5 | 0.9884 | 0.0315 |
| 6 | 0.9907 | 0.0284 |
| 7 | 0.9912 | 0.0262 |
| 8 | 0.9914 | 0.0250 |
| 9 | 0.9924 | 0.0244 |
| 10 | 0.9924 | 0.0239 |

See `results/plots/non_iid/alpha_<X>_val_f1_per_round.png` for the visual versions.

## 10. Final KDDTest+ results

Evaluated ONCE per alpha, after round selection by validation loss only:

| Alpha | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.7530 | 0.9192 | 0.6206 | 0.7409 | 0.8172 | 0.0721 |
| 0.5 | 0.7531 | 0.9219 | 0.6186 | 0.7404 | 0.8479 | 0.0693 |
| 1.0 | 0.7645 | 0.9235 | 0.6393 | 0.7555 | 0.8610 | 0.0700 |
| 5.0 | 0.7708 | 0.9231 | 0.6517 | 0.7640 | 0.8641 | 0.0718 |
| 10.0 | 0.7632 | 0.9221 | 0.6378 | 0.7541 | 0.8637 | 0.0712 |

## 11. Client fairness results

**Important caveat:** these per-client numbers evaluate the FINAL selected global model on each client's OWN local partition — not a separate held-out per-client split, since none exists. This is a proxy for how well the shared global model fits each client's local distribution, not a leakage-free generalization test (this data was part of what trained the global model via FedAvg).

| Alpha | Mean F1 | Median F1 | Min F1 | Max F1 | Std Dev | Max-Min Gap |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.9398 | 0.9784 | 0.7303 | 0.9927 | 0.0904 | 0.2623 |
| 0.5 | 0.9647 | 0.9738 | 0.8533 | 0.9983 | 0.0469 | 0.1450 |
| 1.0 | 0.9880 | 0.9883 | 0.9744 | 0.9980 | 0.0076 | 0.0236 |
| 5.0 | 0.9924 | 0.9920 | 0.9869 | 0.9968 | 0.0032 | 0.0099 |
| 10.0 | 0.9927 | 0.9921 | 0.9898 | 0.9962 | 0.0024 | 0.0064 |

See `results/plots/non_iid/client_f1_distribution.png`.

## 12. Rare-category observations

Clients with ZERO local examples of a rare attack category:

| Alpha | Clients missing r2l | Clients missing u2r |
|---:|---|---|
| 0.1 | SAT-01, SAT-02, SAT-04, SAT-08 | SAT-01, SAT-03, SAT-04 |
| 0.5 | SAT-07 | none |
| 1.0 | none | none |
| 5.0 | none | none |
| 10.0 | none | none |

No samples were invented or artificially balanced to fix these gaps — they reflect genuine rarity of r2l/u2r in the underlying real dataset combined with the Dirichlet split.

## 13. Comparison with Phase 6

Phase 6's original baseline (`data/partitions/`, alpha=0.5, 10 rounds) scored validation F1 0.9879 and final KDDTest+ F1 0.7404. This phase's OWN alpha=0.5 run (a freshly-generated partition, same seed/alpha/constraints as Phase 6) is included in the tables above for direct comparison under identical methodology.

This phase's alpha=0.5 run reproduced Phase 6's numbers exactly (validation F1 0.9879, KDDTest+ F1 0.7404) — since both use the identical seed, alpha, and partitioning constraints, the Dirichlet partitioner deterministically regenerates the same partition, and FedAvg deterministically regenerates the same training trajectory. This is a useful cross-phase reproducibility check, not a coincidence.

## 14. Limitations

- Only 4 alpha values and 8 clients — a small number of conditions; trends described below are observational, not a statistically powered study.
- NSL-KDD remains a terrestrial dataset; the satellite/client framing is simulated throughout.
- Standard synchronous FedAvg only — no adaptive client selection, no staleness handling, no DRL.
- Client fairness metrics (section 11) use each client's own local training data as a proxy, not a genuinely held-out per-client split.
- Very low alpha may fail the partitioning constraints (section 5) for some clients — reported honestly, not hidden or fixed.

## 15. What this tells us about the need for adaptive FL

Across the tested range, alpha=0.1 (most heterogeneous, mean JS distance 0.6892) scored KDDTest+ F1 0.7409 with a client fairness gap of 0.2623, while alpha=10.0 (closest to IID, mean JS distance 0.1452) scored KDDTest+ F1 0.7541 with a client fairness gap of 0.0064. As heterogeneity increased across the tested alphas, performance and fairness showed the pattern above — we deliberately avoid claiming causation from four data points. If this pattern holds up, it is exactly the kind of gap adaptive client selection or weighting (Phase 8) would aim to close.