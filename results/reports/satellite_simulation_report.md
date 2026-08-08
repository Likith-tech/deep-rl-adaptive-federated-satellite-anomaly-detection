# Satellite Simulation Report

## 1. Purpose

Simulation of a multi-satellite learning environment built on top of a real network intrusion dataset, so that later Federated Learning / Adaptive FL / DRL phases have a realistic, non-IID, multi-client environment to operate on. **This phase does not implement Federated Learning, DRL, or any training algorithm — it builds only the client environment.**

## 2. Dataset source

Real NSL-KDD training data (`data/processed/train.parquet`, produced by the Phase 1 pipeline). NSL-KDD is a terrestrial network intrusion-detection dataset — **not** satellite traffic. No real satellite measurements are used anywhere in this simulation. Only the training split is partitioned; validation remains a single global held-out set and KDDTest+ is completely untouched (see `configs/satellite_simulation.yaml` and `data/partitions/README.md`).

## 3. Number of satellites

**8** simulated clients (SAT-01, SAT-02, SAT-03, SAT-04, SAT-05, SAT-06, SAT-07, SAT-08), configured via `configs/satellite_simulation.yaml` (`satellites.num_clients`).

## 4. Partition strategy

Dirichlet label-skew partitioning (`configs/satellite_simulation.yaml` `partitioning.strategy: dirichlet`, alpha=0.5, seed=42). For each attack CATEGORY (normal/dos/probe/r2l/u2r — `src/data/schema.py:ATTACK_CATEGORY_MAP`), that category's real record indices are shuffled and split across clients according to a Dirichlet-sampled proportion vector. Every real record is assigned to exactly one client — see Data integrity below.

Constraints satisfied: **True** (took 1 attempt(s) out of a max of 20).

## 5. Why the data is non-IID

Because each attack category is independently split across clients using a *different* random Dirichlet proportion vector, clients end up with different mixes of attack categories — some dominated by one attack type, others mostly normal traffic — rather than each client getting a uniform random sample of the whole dataset. Section 10 below measures this directly rather than assuming it.

## 6. Client sample distributions

| Client | Samples | Normal | Anomaly | Anomaly % |
|---|---:|---:|---:|---:|
| SAT-01 | 8675 | 81 | 8594 | 99.1% |
| SAT-02 | 39030 | 14094 | 24936 | 63.9% |
| SAT-03 | 6813 | 5709 | 1104 | 16.2% |
| SAT-04 | 6246 | 1131 | 5115 | 81.9% |
| SAT-05 | 8382 | 5097 | 3285 | 39.2% |
| SAT-06 | 4819 | 2710 | 2109 | 43.8% |
| SAT-07 | 19144 | 15800 | 3344 | 17.5% |
| SAT-08 | 13968 | 12620 | 1348 | 9.7% |

Total training samples partitioned: **107077** (matches `data/processed/train.parquet` exactly — 8 clients, no loss, no duplication).

See `results/plots/satellite/client_data_sizes.png` and `anomaly_proportion.png`.

## 7. Attack distributions

| Client | dos | normal | probe | r2l | u2r | Categories present |
|---|---:|---:|---:|---:|---:|---:|
| SAT-01 | 8521 | 81 | 41 | 29 | 3 | 5 |
| SAT-02 | 23948 | 14094 | 953 | 25 | 10 | 5 |
| SAT-03 | 943 | 5709 | 79 | 80 | 2 | 5 |
| SAT-04 | 38 | 1131 | 5007 | 65 | 5 | 5 |
| SAT-05 | 565 | 5097 | 2701 | 4 | 15 | 5 |
| SAT-06 | 1299 | 2710 | 669 | 140 | 1 | 5 |
| SAT-07 | 3000 | 15800 | 340 | 0 | 4 | 4 |
| SAT-08 | 763 | 12620 | 90 | 489 | 6 | 5 |

See `results/plots/satellite/attack_category_distribution.png`.

## 8. Resource simulation

SIMULATED per-client operating conditions (`configs/satellite_simulation.yaml` -> `resources`), sampled uniformly at random within the configured ranges, seeded for reproducibility. These are controlled experiment parameters, **not** measurements from real satellite hardware.

| Client | Bandwidth (Mbps) | Latency (ms) | Compute score | Availability | Connectivity |
|---|---:|---:|---:|---:|---:|
| SAT-01 | 78.53 | 247.50 | 0.887 | 0.879 | 0.366 |
| SAT-02 | 97.68 | 392.51 | 0.829 | 0.651 | 0.615 |
| SAT-03 | 40.23 | 467.04 | 0.715 | 0.929 | 0.610 |
| SAT-04 | 26.59 | 299.56 | 0.251 | 0.931 | 0.742 |
| SAT-05 | 77.02 | 209.54 | 0.977 | 0.957 | 0.845 |
| SAT-06 | 23.49 | 260.02 | 0.235 | 0.662 | 0.778 |
| SAT-07 | 75.75 | 485.38 | 0.461 | 0.748 | 0.629 |
| SAT-08 | 23.00 | 108.46 | 0.581 | 0.691 | 0.769 |

See `results/plots/satellite/resource_distributions.png`.

## 9. Connectivity simulation

`availability_probability` (probability the client is reachable during a communication round) and `connectivity_quality` (link quality when it IS reachable) are modeled as two independent SIMULATED dimensions — see the table above for actual generated values per client.

## 10. Non-IID measurements

- Pairwise Jensen-Shannon distance between clients' category distributions (0 = identical, 1 = maximally different): **avg=0.5270, min=0.0824, max=0.9589**
- Per-client category-distribution entropy (bits; lower = more skewed toward few categories):

| Client | Entropy (bits) |
|---|---:|
| SAT-01 | 0.1563 |
| SAT-02 | 1.1037 |
| SAT-03 | 0.7619 |
| SAT-04 | 0.8237 |
| SAT-05 | 1.2467 |
| SAT-06 | 1.5231 |
| SAT-07 | 0.7534 |
| SAT-08 | 0.5824 |

(Maximum possible entropy for 5 categories, i.e. perfectly uniform, is log2(5) = 2.3219 bits — clients well below that are meaningfully skewed toward fewer categories.)

- Per-category variance of proportion across clients (higher = clients disagree more about how common that category is):

| Category | Variance across clients |
|---|---:|
| dos | 0.101314 |
| normal | 0.093720 |
| probe | 0.068313 |
| r2l | 0.000162 |
| u2r | 0.000000 |

## 11. Resource statistics

| Field | Min | Max | Mean |
|---|---:|---:|---:|
| bandwidth_mbps | 23.000 | 97.684 | 55.286 |
| latency_ms | 108.465 | 485.379 | 308.753 |
| compute_score | 0.235 | 0.977 | 0.617 |
| availability_probability | 0.651 | 0.957 | 0.806 |
| connectivity_quality | 0.366 | 0.845 | 0.669 |

## 12. Reproducibility

- Seed: 42
- Partition strategy: dirichlet (alpha=0.5, category_column=label_original)
- Same config + same seed reproduces the exact same partition and resource values (verified by `tests/simulation/test_partitioner.py::test_dirichlet_partition_deterministic_with_same_seed` and `test_network_conditions.py::test_generate_network_conditions_deterministic_with_same_seed`).
- Regenerate with: `python scripts/create_satellite_partitions.py` then `python scripts/analyze_satellite_partitions.py`.

## 13. Limitations

- **All satellite conditions (bandwidth, latency, compute, availability, connectivity) are simulated** — sampled uniformly at random within configured ranges, not derived from real orbital mechanics, real link budgets, or real hardware.
- **NSL-KDD remains a terrestrial dataset.** No claim is made that any record represents real satellite traffic.
- Dirichlet partitioning creates label-distribution skew (different mixes of attack categories per client); it does not simulate genuine spatial/geographic relationships between satellites, orbital coverage patterns, or inter-satellite links.
- Rare categories (`r2l`, `u2r`) have very few real records overall, so some clients legitimately receive zero or very few of them — this is a property of the real data, not a partition bug.
- No Federated Learning, DRL, or model training happens in this phase — this is environment construction only.

## 14. What this enables next

Phase 5 (Local Satellite Training) trains a local model independently on each client's real partitioned data, using the client datasets and metadata produced here — still without any federated aggregation, communication rounds, or DRL.