# Comparison Table — All 4 Papers

> Reminder: only **4** of the requested 10 papers were provided in this conversation (see the top-level `README.md`, "Honest note on paper count"). This table covers exactly those 4, accurately. Nothing below is invented.

## Main comparison table

| # | Paper (short name) | Year | Venue | Domain | "Anomaly" means | Dataset | Real satellite data? | DL model | FL? | RL? | Adaptive? | Best reported metric | Main limitation | Relevance to capstone |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | Hundman et al. — LSTM + Nonparametric Dynamic Thresholding | 2018 | KDD '18 (ACM) | Spacecraft telemetry | Physical/operational fault | NASA SMAP/MSL (real, ISA-labeled) | Yes (single spacecraft, real) | Plain LSTM (2×80 units), per-channel | No | No | Threshold is data-driven, not learned by an agent | Precision 87.5%, Recall 80.0%, F0.5 0.71 | No FL, no RL, no multi-satellite; single-channel models | High (foundational telemetry AD method + dataset origin) |
| 1 | Yang et al. — Improved DL Telemetry AD (Attention + Multi-scale) | 2021 | Microelectronics Reliability (Elsevier) | Spacecraft telemetry | Physical/operational fault | Same SMAP/MSL benchmark + real Beidou clock (synthetic anomaly injected) | Yes (single spacecraft + Beidou, real, but Beidou anomaly is injected) | Double-layer LSTM + Attention | No | No | Threshold combines local+global scales, not learned | Recall 86.54%, FAR 12.62% | No FL, no RL, no multi-satellite; compute cost not quantified | Medium (extends Paper 3, still single-spacecraft) |
| 2 | Maleki Sadr et al. — Monte Carlo Dropout | 2022 | IEEE TAES | Spacecraft telemetry | Physical/operational fault | Same SMAP/MSL benchmark | Yes (single spacecraft, real) | Bayesian LSTM via MC Dropout (3 LSTM + 2 dense) | No | No | Post-processing threshold tuned per-signal (semi-supervised), not learned by an agent | F1 0.84 (SMAP), 0.74 (MSL) | No FL, no RL, no multi-satellite; threshold tuning references true labels | Medium (adds uncertainty estimation; still single-spacecraft) |
| 4 | Chen et al. — FedSTAR | 2026 | Information Sciences (Elsevier) | Satellite **network traffic** | Malicious traffic / cyberattack | STIN (satellite-relevant benchmark) + TON_IoT + Bot-IoT (auxiliary, generic) | **No** — benchmark/IoT datasets, not operational satellite traces (paper states this itself) | SimVPv2 (pseudo-video) + gSTA/LKA spatio-temporal attention | **Yes** | **Yes (SAC)** | Yes — SAC-driven clustering + aggregation policy, hybrid discrete+continuous | Non-IID accuracy: 95.43% (STIN), 89.16% (TON_IoT), 99.88% (Bot-IoT) | No real satellite traffic; no Byzantine robustness (explicitly disclosed); heavier/slower than lightweight baselines | **Very High** — closest paper to capstone's title and design |

## Reading the table correctly

- **Papers 1–3 and Paper 4 are not directly comparable on accuracy/F1** — they solve different problems on different kinds of data (telemetry fault vs. network intrusion). Never present them side-by-side as if one "beats" another on the same task; that would be comparing apples to oranges.
- The **only genuinely apples-to-apples comparison** across your 4 papers is the lineage Paper 3 → Paper 1 → Paper 2, all solving the same telemetry-forecasting-AD problem on the same underlying SMAP/MSL benchmark. Even there, be careful: they report different metric emphases (Paper 3: Precision/Recall/F0.5; Paper 1: Recall/False-Alarm-Rate; Paper 2: F1/Accuracy/Precision/Recall) and different train/test protocols, so exact numeric comparison across the three papers should be described as "directionally similar/improving," not as a precisely controlled benchmark race.

## Where "N/A" or "Not reported" genuinely applies

| Paper | Field | Value |
|---|---|---|
| Paper 1 | Statistical significance testing | Not reported |
| Paper 1 | Exact compute-cost figures | Not reported (only qualitatively acknowledged) |
| Paper 2 | Explicit numbered future-work list | Not present (no dedicated Future Work section) |
| Paper 2 | Multi-seed / statistical significance testing | Not reported |
| Paper 3 | Statistical significance testing | Not reported |
| Paper 4 | Real operational satellite traffic evaluation | N/A — explicitly not attempted, stated as a limitation |

## Where your own project fits in this same table (for direct comparison — not one of the 4 source papers, included for context)

| Component | Your project (as actually implemented, Phases 0–9B) |
|---|---|
| Domain | Both: Track A = network intrusion (NSL-KDD, simulating 8 satellite clients); Track B = spacecraft telemetry (OPSSAT-AD, real, single spacecraft) |
| Real satellite data? | Track B: yes (real OPS-SAT CubeSat telemetry, F1 0.9315). Track A: no (terrestrial NSL-KDD, simulated as satellite clients) |
| DL model | BaselineMLP (Track A); TemporalGRUAttention with sequence_length=1 (Track B) |
| FL? | Yes — FedAvg (Phase 6), rule-based adaptive weighting (Phase 8), DQN-driven adaptive weighting (Phase 9B) — Track A only |
| RL? | Yes — DQN, 9 discrete weight-profiles, full 750-round pretraining run completed |
| Adaptive? | Yes, at 3 levels: fixed FedAvg → hand-written rule → learned DQN policy |
| Best reported metric (Track A, KDDTest+) | Phase 8 `resource_only` F1 0.7547 (best); Phase 9B DQN F1 0.7425 (did not beat it) |
| Main limitation (self-disclosed) | Track A and Track B never combined; DQN underperformed best rule-based baseline in its one completed run; single-seed reporting |

Both tracks are honestly disclosed as separate — your project does **not** claim to have run federated DRL on real, multi-satellite telemetry, exactly the same honest boundary FedSTAR (Paper 4) draws around its own use of benchmark (not real) satellite network data.
