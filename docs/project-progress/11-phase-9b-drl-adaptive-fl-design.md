# Phase 9B — DRL (DQN) Adaptive Federated Learning: Design Document (early draft)

**Superseded/consolidated note (added, not a rewrite):** this was the
first Phase 9B design pass. A second, more detailed design-and-audit
pass followed in the next session and produced
`11-phase-9b-drl-adaptive-federated-learning.md`, which is now the
canonical Phase 9B research log (per the project's documentation
policy, this earlier draft is kept as-is rather than deleted or
silently rewritten — see `docs/research/development-log/README.md`'s
"What this directory is NOT" section for the same preserve-don't-rewrite
convention applied here). The design conclusions in this draft
(discrete weight-profile action space, validation-only reward,
weighting-not-selection for the first version) were carried forward
unchanged into the canonical document; nothing here was found wrong,
it was just superseded by a fuller pass covering additional explicit
questions the user asked in the follow-up turn.

**Status: DESIGN ONLY. No DQN code has been implemented. This document
describes what will be built and why, so it can be reviewed before any
implementation begins.**

## Scope of this document

Phase 9B replaces Phase 8's fixed, hand-chosen client-scoring weights
(`performance=0.40, data=0.25, resource=0.15, fairness=0.20`) with a
DQN controller that *learns* how to weight clients from a reward
signal — using the same four signal families Phase 8 already wired up
(performance, data, resource, fairness), learned instead of manually
specified. This stays entirely within **Track A (NSL-KDD)**; OPSSAT-AD
(Track B) is not used here (see "Real spacecraft data boundary" below).

## What already exists (verified by reading the actual code, not docs)

- `src/federated/client.py` / `server.py` / `protocol.py` / `fedavg.py`
  — Phase 6's client/server/aggregation primitives. `server.py` is
  verified (by a real AST-based test,
  `test_server_module_never_imports_pandas_or_touches_parquet`) to
  never import pandas or read a parquet file.
- `src/federated/adaptive.py` — Phase 8's `ClientSignals` dataclass,
  `build_resource_signal`, `normalize` (min-max across clients,
  degenerate case → 0.5), `compute_client_scores` (the fixed linear
  rule), `scores_to_aggregation_weights` (renormalize to sum 1,
  degenerate case → uniform), `select_top_k_clients` (built and
  tested, unused so far).
- `src/federated/adaptive_trainer.py` — the round loop: train all 8
  clients (unchanged) → evaluate each update on global validation
  (→ performance signal) → score → aggregate via
  `FederatedServer.aggregate_with_weights` → evaluate → log. This is
  the exact loop Phase 9B intercepts.
- `src/simulation/non_iid_metrics.py` (`shannon_entropy`,
  `compute_non_iid_report`) and `network_conditions.py`
  (`generate_network_conditions`, 5 simulated resource fields) — the
  source of the fairness and resource signals.
- `configs/drl.yaml` — currently a **null placeholder** ("populated in
  Phases 12-15"). `src/drl/` currently contains only `__init__.py` and
  `.gitkeep` — an empty, already-scaffolded package. Both confirm no
  prior DRL work exists; both are where Phase 9B's real config/code go.
- Phase 8's own measured result table (`results/reports/adaptive_fl_results.md`):
  `resource_only` reached the best KDDTest+ F1 (0.7547 vs. Phase 6's
  0.7404); `performance_only` — not the fairness-weighted `combined`
  rule — reached the best fairness gap (0.1245). This is the real,
  disclosed result Phase 9B's learned controller must be compared
  against, not a synthetic target.

## State

**What the DQN observes, per round, for the fixed set of 8 clients:**

| Feature | Dimension | Source (reused, not reinvented) | Why included |
|---|---|---|---|
| `performance_k` | 8 | this round's local update's F1 on global validation (`adaptive_trainer.py`'s existing evaluation step) | Direct signal of whether each client's current update is helping — Phase 8's strongest signal by its own ablation results |
| `data_k` (min-max normalized) | 8 | `ClientUpdate.num_samples` | Reused FedAvg intuition; kept comparable in scale via the same `normalize()` Phase 8 uses |
| `resource_k` | 8 | `build_resource_signal` (Phase 4's simulated bandwidth/latency/compute/availability/connectivity) | Lets the agent learn to discount slow/unavailable clients, using signal already wired for exactly this purpose in Phase 8 |
| `fairness_k` | 8 | Shannon entropy of local category distribution (`non_iid_metrics.py`) | Lets the agent learn to protect rare-category (r2l/u2r) clients — the signal Phase 7 showed matters most under heterogeneity |
| `val_f1_t`, `val_f1_{t-1}` | 2 | server-side global validation eval | Recent-trend context so the agent can react to whether the last decision helped |
| `round_number / num_rounds` | 1 | loop counter | Lets the agent learn round-dependent behavior (e.g., more conservative near the end) without leaking future-round information |
| `previous_action` (one-hot) | `|A|` | last round's chosen profile | Lets the agent learn stability and avoid needless oscillation |

**Deliberately excluded**: raw per-client data (never available to the
server anyway — see leakage controls below); anything derived from
KDDTest+; any information from a future round. This state is a strict
scientific-defensibility choice: every feature already has a proven,
documented purpose from Phase 7/8's own measurement work — nothing new
is introduced "because it might help."

Total dimension = 8×4 + 2 + 1 + `|A|` (fixed once `|A|` is set below).

## Action

**Compared options:**

| Option | DQN-learnable | Compute | Explainable | Fits `aggregate_with_weights` | Scales to 8 clients |
|---|---|---|---|---|---|
| A. Aggregation weights (continuous, per-client) | No — DQN is a discrete-action method | N/A | Low | Yes | Yes |
| B. Client selection (binary per client, 2^8=256) | Yes, but sparse/combinatorial | High | Medium | Yes, but abandons Phase 8's deliberate "weighting not selection" fairness posture | Marginal |
| C. **Both / hybrid** | Yes, but compounds B's combinatorics | Highest | Medium | Yes | Marginal |
| **Discrete weighting profiles (chosen)** | **Yes — small fixed action set** | **Low** | **High — each action is a named, inspectable weight vector** | **Yes, directly reuses `compute_client_scores`/`scores_to_aggregation_weights` unchanged** | **Yes — independent of client count** |

**Decision: discrete weighting profiles**, the simplest scientifically
defensible option. Define a fixed catalogue of `|A|=9` candidate
`(w_perf, w_data, w_resource, w_fair)` vectors: Phase 8's own four
profiles (`performance_only`, `resource_only`, `data_only`, `combined`)
plus a uniform profile and four additional interpolated blends
spanning the simplex. The DQN's action each round is "pick one of
these 9 profiles"; the chosen profile flows through the **unmodified**
`compute_client_scores()` → `scores_to_aggregation_weights()` →
`aggregate_with_weights()` pipeline. All 8 clients keep participating
every round (preserves Phase 8's fairness-safety reasoning for
weighting over selection); every decision is logged as a named,
human-readable profile choice, not an opaque vector — directly
supporting the "genuinely learned, not a fake lookup table" requirement,
since the Q-function is learned over real distinct aggregation
behaviors, not indexed by round number or hard-coded per round.

## Reward

Balances global performance, fairness, communication/resource cost, and
round-to-round stability — never optimizes one metric alone, and never
touches KDDTest+.

```
r_t = ΔF1_t  -  λ_fair · fairness_gap_t  -  λ_stab · 1[action_t ≠ action_{t-1}]  -  λ_cost · comm_cost_t

where:
  ΔF1_t          = val_F1_t - val_F1_{t-1}        (global validation set only)
  fairness_gap_t = max_k(local_F1_k) - min_k(local_F1_k)   (same proxy Phase 8 reports)
  1[action_t ≠ action_{t-1}]  = 1 if the profile changed this round, else 0
  comm_cost_t    = 0 under the chosen action space (all 8 clients always participate,
                   same as Phase 6/8) — retained in the formula for honesty/extensibility,
                   not removed, since a later selection-based phase would make it nonzero
```

`λ_fair`, `λ_stab`, `λ_cost` are fixed constants in `configs/drl.yaml`,
chosen before any KDDTest+ evaluation — exactly the same discipline
Phase 8 applied to its rule weights. **KDDTest+ never enters this
formula or any part of training** — only the global validation set
(already how Phase 6/7/8 select checkpoints) and the same
own-local-data fairness proxy Phase 8 already reports.

## Environment

```
state_t --(DQN picks action_t: one of 9 profiles)-->
    compute_client_scores(profile) -> scores_to_aggregation_weights(scores)
    -> server.aggregate_with_weights(updates, weights)   [unchanged FederatedServer]
    -> evaluate new global model on global validation     [unchanged eval code]
    -> reward r_t (formula above)
    -> state_{t+1} (this round's signals + new val_F1 + action_t as previous_action)
```

One **episode** = one full simulated FL run (a fixed number of rounds
over one client partition). **Termination**: an episode ends when its
configured round budget is reached — there is no early-termination
condition (unlike, e.g., a divergence criterion), since FL rounds are
inexpensive relative to the value of a full, comparable-length episode.

## DQN

- Input dimension: state vector size from "State" above (≈43 with
  `|A|=9`).
- Hidden layers: 64 → 64, ReLU.
- Output dimension: `|A|=9` (one Q-value per profile).
- Optimizer: Adam, learning rate 1e-3.
- Discount factor γ: 0.95 (short, ≤20-round episodes — a high γ isn't
  needed and slows convergence at this scale).
- Replay buffer: fixed-capacity deque, ~5,000 transitions.
- Batch size: 32.
- Target network: yes, hard update every 10 learning steps.
- Epsilon-greedy: ε 1.0 → 0.05, decayed across the pretraining episode
  budget.
- Target update frequency: every 10 learning steps (paired with the
  target network above).
- Training episodes: ~50-80 (see "Experiment Plan" — the exact count
  is calibrated against this machine's actual per-round timing before
  being locked in, not guessed).
- Random seed: 42, consistent with the rest of the project.

These are deliberately conservative, capstone-scale settings — not
tuned for maximum performance, matching the project's stated
preference for a defensible, explainable baseline over a maximally
optimized one.

## Training strategy

One 10-round FL run (Phase 6/8's protocol) gives only ~10 transitions
— too few for meaningful DQN learning, and it would conflate "is the FL
model converging" with "is the RL agent converging" in a single
non-stationary run. Instead, the DQN trains **across many simulated FL
episodes**: fresh 8-client Dirichlet partitions at varied seeds/alphas
(reusing `src/simulation/partitioner.py`, the exact mechanism Phase 7's
alpha sweep used), each run for ~15-20 rounds, for on the order of
50-80 episodes — accumulating roughly 750-1,600 transitions with
experience replay. Every episode is a real federated training run
using the unmodified `FederatedClient`/`FederatedServer` — not a
simulated proxy standing in for it.

## Real spacecraft data boundary

Phase 9B stays entirely within **Track A (NSL-KDD)**. OPSSAT-AD (Track
B) is not used because it has no multi-client/federated structure (a
single spacecraft — Phase 9A's own "What this does NOT prove" section
already states the FL machinery was never applied to it), and building
an artificial multi-client split of OPSSAT-AD would require its own
separately-scoped design, not a Phase 9B side effect. Track B's
F1/MCC/AUCPR figures are never combined with Track A's F1 into one
number, in either direction.

**Also carried forward from the Phase 9A audit, unmodified**: the
current OPSSAT-AD implementation uses `sequence_length=1` and must be
described as "segment-level anomaly classification using the Phase 3
GRU+attention architecture with sequence_length=1" — not genuine
temporal sequence modeling. This document does not touch that
implementation or its description.

## Scientific experiment plan

| Method | FL rounds | Partition | Purpose |
|---|---|---|---|
| DQN pretraining | ~15-20/episode × ~50-80 episodes | fresh Dirichlet partitions, varied seed/alpha | learn the Q-function only |
| Phase 6 FedAvg (existing) | 10 | `data/partitions/` (Phase 4/6 exact) | baseline — already run, not rerun |
| Phase 8 `combined` (existing) | 10 | `data/partitions/` (Phase 4/6 exact) | rule-based reference — already run, not rerun |
| **Phase 9B final comparison run** | 10 | `data/partitions/` (Phase 4/6 exact) | frozen (ε=0) DQN policy, identical protocol to Phase 6/8 |

Held fixed across all three methods' *final comparison*: number of
clients (8), partition, model architecture, validation/test split,
preprocessing, local epochs (1), batch size (128), learning rate
(0.001), optimizer/loss, shared init seed (42) — no unjustified change
to any of these. The only intentionally varied factor in the final
comparison is the aggregation-weighting mechanism itself.

**Avoiding an unfair advantage for the DQN**: the anomaly-detection
model itself only ever sees data through the same 1-local-epoch/round
protocol under every method — the DQN's controller-pretraining budget
(disclosed explicitly, e.g. "~1,000 transitions across ~60 episodes")
is a distinct, reported cost of the *controller*, not extra data
exposure for the model being evaluated. This is reported as an honest
asymmetry, not hidden, the same way Phase 8 disclosed that its rule
weights were "chosen by reasoning" before any test run.

**KDDTest+** is evaluated exactly once per method's final comparison
run, after the round/checkpoint (or frozen policy) is already
selected — never used for reward, action selection, or hyperparameter
choice.

**Fairness metric**: same proxy as Phase 7/8 — max/min client-local-F1
gap.

**Communication estimate**: unchanged from Phase 6/8 under this action
space (all 8 clients transmit every round) — ~14.6 MB/10 rounds, same
23,937-parameter model.

## Baselines

Phase 6 FedAvg (`results/reports/federated_results.md`) and Phase 8
`combined` plus, optionally, each Phase 8 ablation
(`results/reports/adaptive_fl_results.md`) — all real, already-run,
committed results, not re-executed for this comparison.

## Leakage/privacy controls

- **KDDTest+ never enters DQN training** — reward uses only the global
  validation set and the local-partition fairness proxy.
- **Test data never influences reward, action selection, or
  hyperparameter choice** — by construction, per the formulas above.
- **No client receives another client's raw data** — unchanged
  `FederatedClient`, each still only opens its own partition file.
- **Server does not read client parquet files** — unchanged
  `FederatedServer`; the existing AST-based test keeps enforcing this.
- **No future-round information in the state** — only this-round and
  previous-round quantities are used.
- **RL experience never touches held-out test information** — every
  transition is built from training-partition local updates + global
  validation, across pretraining episodes that never load KDDTest+.

## Test plan

| Test | Expected behavior |
|---|---|
| Deterministic DQN initialization | Same seed → identical initial network weights |
| State construction | Correct shape/ordering/normalization; no NaN on degenerate (identical-client) input |
| Action validity | Every action index maps to a non-negative weight vector summing to 1 |
| Weight normalization | `scores_to_aggregation_weights` output always sums to 1, falls back to uniform on all-zero input (existing Phase 8 behavior, reused) |
| Aggregation correctness | `aggregate_with_weights` output matches a hand-computed weighted average for a small fixture |
| Reward calculation | Hand-computed example matches `reward()`'s output; no path from KDDTest+ into the function |
| Replay buffer | Push/sample respects capacity; deterministic sampling under a seeded RNG |
| Epsilon-greedy behavior | ε=0 always argmax; ε=1 uniform random over many draws; decay matches config |
| Target network updates | Target weights change only at the configured sync frequency |
| Client isolation | `FederatedClient` never reads another client's partition path |
| Server privacy boundary | AST-based check: `drl_trainer.py`/`environment.py` never import pandas or call `read_parquet`/`read_csv` |
| No test-set leakage | Dedicated test asserting no KDDTest+/`test.parquet` path appears anywhere in the training/environment code path |
| Reproducibility | Two identical-seed pretraining runs (tiny synthetic environment) produce identical loss curves |
| End-to-end FL + DQN loop | A handful of episodes on tiny synthetic partitions runs without error and never touches KDDTest+ |

## Risks / limitations (disclosed, not hidden)

- Pretraining compute cost (~50-80 episodes × 15-20 rounds × 8 clients)
  is a real multiple of Phase 6/8's single 10-round run — needs a
  timing check before the exact episode count is locked in.
- Single-seed capstone scale, like Phase 7/8 — not a statistically
  powered study.
- The discrete weight-profile action space approximates the true
  continuous weighting space — a DQN-compatibility tradeoff, not a
  hidden limitation.
- Reward-shaping constants (`λ_fair`, `λ_stab`, `λ_cost`) are a design
  choice made before seeing test results, same honesty obligation as
  Phase 8's rule weights.
- At only 8 clients, the learned policy may converge close to one of
  Phase 8's existing profiles rather than discovering new behavior — a
  plausible, reportable outcome.
- The fairness proxy inherits Phase 8's own documented limitation
  (evaluated on each client's own local data, not a genuine held-out
  per-client test).

## Reusable existing code (no duplication)

`src/federated/{client,server,protocol,fedavg}.py` (unchanged);
`src/federated/adaptive.py`'s `ClientSignals`/`build_resource_signal`/
`normalize`/`compute_client_scores`/`scores_to_aggregation_weights`;
`src/federated/adaptive_trainer.py` as the structural template for the
new round loop; `src/simulation/{partitioner,non_iid_metrics,network_conditions}.py`
unchanged; `src/training/local_trainer.py:build_initial_state_dict`,
`src/training/baseline_trainer.py:{load_features_and_labels,make_dataloader,run_validation}`
unchanged; `src/evaluation/metrics.py` unchanged; `src/drl/` (already
scaffolded, empty) as the new-code location.

## New files required (not created yet)

```
src/drl/replay_buffer.py
src/drl/dqn_agent.py
src/drl/action_space.py
src/drl/environment.py
src/federated/drl_trainer.py
configs/drl.yaml            (filled in, replacing the current null placeholder)
scripts/train_drl_controller.py
scripts/run_drl_adaptive_fl.py
scripts/evaluate_drl_fl_results.py
tests/drl/{test_replay_buffer,test_dqn_agent,test_environment,test_drl_trainer}.py
```

## Recommended implementation order

1. `configs/drl.yaml` filled in with the concrete spec above.
2. `src/drl/replay_buffer.py` + tests.
3. `src/drl/dqn_agent.py` + tests.
4. `src/drl/action_space.py` + `src/drl/environment.py` + tests.
5. `src/federated/drl_trainer.py` + smoke test.
6. `scripts/train_drl_controller.py` (multi-episode pretraining).
7. `scripts/run_drl_adaptive_fl.py` (frozen-policy final run, Phase
   6/8-matching protocol).
8. `scripts/evaluate_drl_fl_results.py` (one-time KDDTest+ eval,
   fairness, 3-way comparison report/plots).
9. Full test suite + privacy/leakage re-verification.
10. `docs/project-progress/12-phase-9b-drl-adaptive-fl-results.md`
    write-up, once real results exist.
11. Present for review — no auto-commit, no auto-push, same discipline
    as every prior phase.
