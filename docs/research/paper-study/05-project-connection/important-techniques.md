# Important Techniques Worth Knowing Cold (Drawn From the 4 Papers)

A short list of the specific *techniques* (not whole papers) most worth being able to explain precisely, because they are either directly relevant to your project or very likely to come up in questions.

## 1. Nonparametric Dynamic Thresholding (Paper 3)
Finds an anomaly-error cutoff by searching over candidate thresholds and picking the one that most shrinks the mean/std of the remaining "normal" data, penalized for flagging too much — no Gaussian assumption. **Why it matters:** the philosophical ancestor of "let the data decide the rule, don't hand-pick a number" — same spirit (different mechanism) as your project's Phase 9B letting a DQN decide the aggregation weighting instead of a hand-picked rule.

## 2. EWMA (Exponentially Weighted Moving Average) smoothing (Papers 1, 2, 3)
Used by all three telemetry papers to smooth prediction-error noise before thresholding, so ordinary jitter doesn't get mistaken for an anomaly. A simple, standard, easy-to-explain technique — worth knowing the one-line formula: recent errors count more, older errors count exponentially less.

## 3. Attention mechanism (Paper 1; conceptually also your project's Phase 3 `TemporalGRUAttention`)
Lets a model learn which past time steps matter most for the current prediction, instead of the network's built-in memory alone deciding. Worth being able to draw the simple version: a weighted sum of past hidden states, weights themselves learned.

## 4. Monte Carlo Dropout (Paper 2)
Keep dropout switched on at test time; run the same input through the network many times; treat the spread of outputs as an uncertainty estimate. Cheap, practical approximation to a full Bayesian Neural Network. **Why it matters for your project's future work:** it's a low-effort way to add "how confident is this detection?" to any existing model your project already has, without redesigning the architecture.

## 5. Pruning / consecutive-point confirmation (Papers 2 and 3, independently)
Both papers add a post-processing step that only confirms an anomaly if enough points *in a row* stay flagged, specifically to cut down false positives caused by ordinary noise. Two independent papers converging on the same idea is good evidence it's a real, useful pattern, not a one-off trick.

## 6. Dirichlet partitioning for simulating Non-IID clients (Paper 4 and your own project's Phase 4/7)
A single, tunable parameter controls how skewed the simulated client data distributions are. Worth knowing cold: lower α/η = more skewed = harder for FedAvg; both Paper 4 (η=0.5) and your project (α=0.5 main partition, full sweep in Phase 7) use this exact technique.

## 7. Classifier-head-only similarity for clustering clients (Paper 4)
Instead of comparing entire model weights (expensive, high-dimensional), Paper 4 clusters clients using only their final/classifier-layer parameters, citing empirical evidence that final layers reflect local-task differences better than early layers. **Relevant contrast:** your project's DQN state vector instead uses four *derived signal families* (performance, data, resource, fairness) per client, not raw model-parameter similarity at all — a different, arguably more interpretable, design choice worth being able to contrast explicitly.

## 8. Multi-term reward shaping with a stability/divergence penalty (Paper 4 and your project's Phase 9B, independently)
Both add a penalty term discouraging the agent from destabilizing the system (Paper 4: intra-cluster divergence penalty; your project: penalty for changing the chosen action/profile between consecutive rounds) alongside the main performance-gain term. Worth explaining why: an agent optimizing raw performance alone can learn to behave erratically round-to-round if nothing discourages instability.

## 9. Variance decomposition proof (Paper 4)
A short piece of algebra (not just an experiment) proving that plain FedAvg mathematically hides all client heterogeneity inside a single "within-group" term, while clustering reveals a genuine "between-cluster" variance component. Worth knowing as an example of *proving* why a method should help, not just observing that it did — a stronger form of evidence than an ablation table alone.

## 10. AST-based privacy-boundary testing (your project's own technique — not from any of the 4 papers, included here because it's directly comparable to what these papers only claim)
None of the 4 papers describe mechanically testing their privacy claims. Your project parses its own server/DRL module source code (via Python's `ast` module) to verify, automatically, that pandas is never imported and no parquet/csv file is ever read — turning "the server can't see raw client data" from a design intention into a continuously-enforced, testable invariant.
