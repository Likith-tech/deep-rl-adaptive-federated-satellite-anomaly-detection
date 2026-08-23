# Viva Prep — Simple Explanations

## Basic questions

**Q: What is satellite network anomaly detection?**
A: Automatically noticing when something is wrong — either a physical/operational fault on a spacecraft (temperature, power, attitude) or malicious network traffic between satellites/ground — instead of a human watching every signal manually.

**Q: Why is anomaly detection needed?**
A: Spacecraft have thousands of telemetry channels and, increasingly, network links exposed to cyber threats; no human team can watch all of it in real time, and missing a real fault or attack can risk the mission.

**Q: What is federated learning?**
A: Training one shared model across many separate clients (satellites, in our case), where each client trains on its own local data and only shares model updates — not raw data — with a central server.

**Q: Why use federated learning?**
A: Two reasons: privacy (raw data, whether telemetry or network traffic, may be sensitive and shouldn't leave the satellite) and bandwidth (satellite-ground links are limited and expensive; shipping model updates is cheaper than shipping raw data).

**Q: What is reinforcement learning?**
A: An agent learns what decision to make by trying actions, observing a reward, and improving over many attempts — rather than being shown labeled "correct answer" examples.

**Q: What is deep reinforcement learning?**
A: Reinforcement learning where a deep neural network represents the agent's decision function, needed because the situations the agent observes (its "state") are too large/continuous to represent as a simple lookup table.

**Q: What is adaptive learning, in our project's specific sense?**
A: Instead of combining federated clients with one fixed rule forever (like plain FedAvg), the system changes how it combines clients based on current conditions — either via a hand-written rule (Phase 8) or a learned policy (Phase 9B, DQN).

**Q: Why combine FL and DRL?**
A: Because plain FedAvg performs poorly when clients' data is Non-IID (very different from each other) — some way of deciding, round to round, which clients to trust more is needed; DRL is one way to *learn* that decision instead of hand-writing it.

**Q: What is the research gap?**
A: No existing paper combines real, multi-satellite federated learning with genuine physical-telemetry anomaly detection; the one paper that does combine FL + DRL for satellite anomaly detection (FedSTAR, 2026) targets network-traffic intrusion detection on benchmark (not real) data, doesn't test Byzantine-robustness, and doesn't model real orbital communication constraints — all explicitly acknowledged as open by that paper itself.

**Q: What is our contribution?**
A: A working, already-executed, honestly-reported three-stage comparison (fixed FedAvg → hand-written adaptive rule → DQN-learned adaptive policy) on a federated satellite-client simulation, using a small, fully-explainable discrete DQN action space, plus a separate, honest validation of a telemetry-style detector against one real spacecraft's actual data (OPSSAT-AD) — with the DQN's currently-measured underperformance against the best hand-written rule reported openly, along with a specific, testable hypothesis for why.

---

## Paper questions (one paragraph each — see the individual paper notes in `01-paper-notes/` for full depth)

**Paper 3 (Hundman et al., 2018):** Trained one LSTM per spacecraft telemetry channel to predict the next value, then used a nonparametric (no Gaussian assumption) statistical method to automatically decide the anomaly threshold, plus a pruning step to cut false positives. Real NASA SMAP/MSL data. Best result: 87.5% precision, 80.0% recall. Biggest limitation: single-spacecraft, no FL, no RL. Relation to us: the foundational method + dataset origin for the "telemetry forecasting AD" branch of the literature.

**Paper 1 (Yang et al., 2021):** Improved the same idea with a double-layer LSTM + attention mechanism, and combined local + global (multi-scale) detection to catch both sudden and slow-developing anomalies. Same SMAP/MSL benchmark + real Beidou satellite clock data (synthetic anomaly injected). Best result: 86.54% recall, 12.62% false alarm rate. Biggest limitation: same as Paper 3 (no FL/RL); Beidou anomaly is synthetic, not naturally occurring. Relation to us: shows how attention improves the same forecasting approach, similar in spirit to our Phase 3 GRU+attention architecture.

**Paper 2 (Maleki Sadr et al., 2022):** Added genuine uncertainty estimation to LSTM-based telemetry forecasting via Monte Carlo Dropout (dropout kept on at test time, run many times, spread of outputs = confidence interval). Same SMAP/MSL benchmark. Best result: F1 0.84 (SMAP), 0.74 (MSL), beating 6 other published methods while being cheaper to run. Biggest limitation: same no-FL/no-RL limitation; its post-processing threshold tuning references true labels. Relation to us: a strong candidate future-work direction — add confidence estimates to our own detectors.

**Paper 4 (Chen et al., 2026, FedSTAR):** The closest paper to our project's title — federated learning (with SAC-driven client clustering) + a heavy spatio-temporal deep learning model (SimVPv2/gSTA/LKA) for detecting cyberattacks in satellite network traffic. STIN/TON_IoT/Bot-IoT benchmark datasets, not real satellite traffic. Best result: 95.43%/89.16%/99.88% Non-IID accuracy across the 3 datasets, beating 7 baselines. Biggest limitation, self-stated: no real satellite data, no Byzantine robustness, no orbital-realistic networking, heavier/slower than lightweight alternatives. Relation to us: the essential "closest prior work" our project must be explicitly differentiated from — see the difficult-question answer below.

---

## Difficult viva questions

**Q: Why not use centralized learning?**
A: Centralized learning would require every satellite to ship its raw data to one place — expensive in bandwidth and a privacy/security risk. Federated learning avoids both by keeping raw data local and only sharing model updates.

**Q: Why not use normal (non-federated) deep learning, like Papers 1–3 do?**
A: Papers 1–3 solve a genuinely different, single-spacecraft problem — they never needed to combine information from multiple satellites. Our project's Track A specifically targets a multi-satellite setting, where federated learning is the relevant technique.

**Q: Why federated learning specifically, and not just training one model per satellite independently?**
A: Our own project directly measured this trade-off in Phase 5 (independent local training) vs. Phase 6 (FedAvg) — FedAvg's global-validation F1 (0.9879) beat the *average* independently-trained local client (0.9721), showing that combining knowledge across clients helps overall, even though it didn't beat the single best individual local client (0.9918).

**Q: Why reinforcement learning, and not just a fixed rule (like Phase 8)?**
A: Because Phase 8's own results showed no single fixed rule is best on every metric at once (`resource_only` won on F1, `performance_only` won on fairness, not `combined`) — this motivated testing whether a *learned* policy could do better by adapting round-to-round instead of committing to one fixed formula.

**Q: Why deep reinforcement learning, and not tabular RL?**
A: Our state is a 44-dimensional vector of continuous numbers (per-client performance/data/resource/fairness signals, validation trends, etc.) — far too large and continuous for a lookup table; a small neural network (DQN) is needed to generalize across similar-but-not-identical states.

**Q: What exactly is "adaptive" about our framework?**
A: The aggregation weighting given to each client changes every round, based on that round's observed signals — first via a hand-written rule (Phase 8), then via a DQN policy that itself decides which of 9 named weighting profiles to apply each round (Phase 9B) — rather than a single fixed rule (like plain FedAvg's sample-count-only weighting) used forever.

**Q: How does the RL agent make decisions?**
A: Each round, it observes a 44-dimensional state vector, passes it through a small neural network (44 → 64 → 64 → 9), and picks the action (one of 9 weight-profiles) with the highest estimated Q-value — with some randomness (epsilon-greedy) during training to keep exploring, and pure greedy selection during the final evaluation run.

**Q: What is the state?**
A: Per-client performance/data-volume/resource/fairness signals (4 families × 8 clients = 32 numbers), plus this round's and the previous round's global validation F1 (2 numbers), plus round progress (1 number), plus the previous action as a one-hot vector (9 numbers) — 44 total.

**Q: What is the action?**
A: A choice among 9 fixed, named aggregation weight-profiles (e.g. `performance_only`, `resource_only`, `combined`, `uniform`, plus several blends) — a discrete choice, not a continuous number, and not client selection or clustering.

**Q: What is the reward?**
A: `ΔF1 (this round's global validation F1 improvement) − λ_fair × (spread of per-client performance signals) − λ_stab × (1 if the chosen profile changed from last round, else 0)` — balancing "did accuracy improve" against "are clients being treated too unevenly" and "is the policy oscillating."

**Q: What happens if clients have non-IID data?**
A: Measured directly in Phase 7: heterogeneity (measured via Jensen-Shannon distance) shrinks the "client fairness gap" as it decreases, and affects KDDTest+ F1 non-monotonically (peaking at a moderate heterogeneity level, not the lowest). This non-IID condition is exactly what motivates Phase 8/9B's adaptive weighting in the first place.

**Q: How do you handle communication overhead?**
A: All 8 clients participate every round (weighting, not selection) — the same communication cost as plain FedAvg (~14.6 MB per 10 rounds for the current model size); the project deliberately chose weighting over selection specifically to avoid the fairness risk of dropping a client, at the cost of not reducing communication further.

**Q: How is privacy improved?**
A: The server (and the DQN) never reads a client's raw data file — enforced not just by design but by an automated, AST-based test that inspects the actual source code of the server and DRL modules and fails the build if pandas is ever imported or a parquet/csv file is ever read.

**Q: Why is satellite network data different from normal network data?**
A: Satellites have intermittent, high-latency, bandwidth-constrained links (unlike most terrestrial networks assumed by generic federated learning research), and, per Paper 4's own explicit disclosure, this orbital realism (delay, intermittent contact, staleness) is not yet modeled by any paper in our literature set, including our own project.

**Q: What dataset are we using?**
A: Track A: NSL-KDD (real, but terrestrial, network-intrusion data), Dirichlet-partitioned into 8 simulated satellite clients. Track B: OPSSAT-AD (real ESA OPS-SAT CubeSat telemetry, single spacecraft, no federated structure).

**Q: What are the limitations of our approach?**
A: (1) Track A's "satellites" are simulated from terrestrial data, not real satellite traffic; (2) Track B has no federated structure — the two tracks are never combined; (3) the DQN did not beat the best rule-based baseline in its one completed run, and the training regime used a single fixed partition across all pretraining episodes rather than the originally-intended diverse partitions; (4) no Byzantine/adversarial-client robustness is tested; (5) single-seed, single-run reporting throughout, consistent across every phase.

**Q: Isn't it a weakness that your DQN lost to a simple rule?**
A: It's an honestly-reported, informative result, not a weakness to hide. It shows that simply adding a DRL controller doesn't automatically help — and gives a specific, testable explanation (limited training-partition diversity) for a future experiment to investigate, which is a more scientifically useful outcome than reporting only favorable numbers.
