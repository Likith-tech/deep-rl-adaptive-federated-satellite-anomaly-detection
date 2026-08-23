# Phase 9B — DRL Pilot Implementation

**Status: PILOT SCOPE ONLY.** No DQN pretraining run has been executed.
No performance claim is made. This document records exactly what was
implemented and verified this turn, following the APPROVED FOR
IMPLEMENTATION decision in
`docs/research/experiments/phase-9b-drl/design-checkpoint-review-v2.md`.

## Purpose of the pilot

Prove that the DRL controller's basic mechanics — configuration
loading, state construction, the discrete action catalogue, action →
aggregation-weight conversion, reward calculation, and the DQN's
forward pass / action selection — are correct, deterministic, and
privacy-safe, BEFORE any expensive multi-episode pretraining run is
attempted. This is the "pilot" scope explicitly separated from the
full experiment in the canonical design document's section 25.

## Approved design being implemented

`docs/project-progress/11-phase-9b-drl-adaptive-federated-learning.md`
(post both correction passes — state dimension 44, the corrected
`perf_spread_t` reward, the 9-profile action catalogue, the OPSSAT-AD
separation statement, the artifact path convention), approved by
`docs/research/experiments/phase-9b-drl/design-checkpoint-review-v2.md`
("APPROVED FOR IMPLEMENTATION").

## Files created

```
configs/drl.yaml                          filled in (was a null placeholder)
src/drl/__init__.py                       module docstring, pilot-scope notice
src/drl/action_space.py                   9-profile catalogue + action -> weights
src/drl/state.py                          44-dim state construction
src/drl/reward.py                         corrected reward formula
src/drl/dqn_agent.py                      QNetwork + DQNAgent (pilot: no replay buffer, no learning step)
src/drl/config.py                         configs/drl.yaml loader + invariant validation
tests/drl/__init__.py
tests/drl/conftest.py                     synthetic 8-client fixtures
tests/drl/test_action_space.py
tests/drl/test_state.py
tests/drl/test_reward.py
tests/drl/test_dqn_agent.py
tests/drl/test_config.py
tests/drl/test_privacy_boundary.py        AST-based, same style as tests/federated/test_server.py
scripts/run_drl_pilot.py                  pilot verification script (synthetic data only)
experiments/drl_adaptive_fl/pilot/pilot_verification.json
results/plots/drl_adaptive_fl_pilot/action_profile_weights.png
results/reports/drl_adaptive_fl_pilot.md
docs/project-progress/12-phase-9b-drl-pilot.md   (this file)
```

No existing file was modified except the two research-log files listed
in "Documentation" below.

## State representation (implemented exactly as approved: 44-dim)

`src/drl/state.py:build_state` — built entirely from
`src.federated.adaptive.normalize`d `ClientSignals` (no new information
channel):

```
performance_k (8) + data_k (8) + resource_k (8) + fairness_k (8)
+ val_F1_t, val_F1_{t-1} (2)
+ round / num_rounds (1)
+ previous_action one-hot (9)
= 44
```

Client ordering is fixed (sorted `client_id`), verified deterministic
regardless of input list order (`test_state_component_ordering_is_deterministic`).
First-round handling (`val_f1_prev=None`, `previous_action=None`) is
explicit, not silent: the trend slot is filled with the current F1
(zero implied trend) and the previous-action one-hot is left all-zero.
`build_state` raises `ValueError` — never silently pads/truncates — if
given anything other than exactly 8 client signals.

## Action representation (9-profile discrete catalogue, weighting-only)

`src/drl/action_space.py` — 9 named profiles (Phase 8's exact
`performance_only`/`resource_only`/`data_only`/`combined`, plus
`uniform` and four additional blends spanning the simplex), each a
`(performance, data, resource, fairness)` weight vector validated at
import time to be non-negative and sum to 1. `action_to_aggregation_weights`
passes the chosen profile through the **exact, unmodified** Phase 8
pipeline (`src.federated.adaptive.compute_client_scores` →
`scores_to_aggregation_weights`) — no second FedAvg/weighting
implementation was created. Every action always returns weights for
all 8 clients (weighting only — no selection).

## Reward formula (implemented exactly as corrected)

```
r_t = ΔF1_t − λ_fair · perf_spread_t − λ_stab · 1[action_t ≠ action_{t-1}]
ΔF1_t         = val_F1_t − val_F1_{t-1}
perf_spread_t = max_k(performance_k,t) − min_k(performance_k,t)
```

`src/drl/reward.py:compute_reward` takes only plain floats and a
`{client_id: float}` performance-signal dict — no file path, no
DataFrame-shaped argument (enforced by a signature-inspection test),
and performs no I/O of any kind.

## Privacy boundary

`tests/drl/test_privacy_boundary.py`, same style as
`tests/federated/test_server.py`'s existing AST-based test:
`src/drl/{reward,state,action_space,dqn_agent}.py` are each verified,
by parsing their actual import statements and call sites (not
docstrings), to never import `pandas`/`pyarrow` and never call
`read_parquet`/`read_csv`; a further scan confirms no literal reference
to a client partition path appears in any of these four modules'
source; `compute_reward`'s own parameter names are checked to exclude
anything suggesting file/path/DataFrame access. The true Phase 8
own-local-data fairness gap remains reachable only from
`scripts/evaluate_adaptive_fl_results.py` (Phase 8's own, unmodified,
offline script) — never from any Phase 9B module.

## Test strategy

46 new tests (`tests/drl/`), all synthetic-fixture based (`conftest.py`'s
`synthetic_signals`/`degenerate_signals` — 8 fabricated clients,
never real NSL-KDD data), covering: state dimension/ordering/finiteness/
first-round handling/rejection of malformed input; all 9 actions'
validity, determinism, and distinctness; the reward formula against
hand-computed values, first-round handling, the stability penalty in
isolation, and the performance-spread penalty in isolation; the DQN's
input/output dimensions, forward-pass shape, seeded reproducibility
(both greedy and epsilon-exploring), and rejection of malformed input;
config loading and its invariant checks; and the full privacy-boundary
suite described above.

## Pilot results

- **`tests/drl/` — 46/46 passed.**
- **Full suite (`pytest tests`) — 237/237 passed** (191 pre-existing +
  46 new; zero regressions in any Phase 0-9A test).
- `scripts/run_drl_pilot.py` executed successfully against 8
  **synthetic** clients (seeded, fabricated — never real data),
  producing `experiments/drl_adaptive_fl/pilot/pilot_verification.json`,
  `results/plots/drl_adaptive_fl_pilot/action_profile_weights.png`, and
  `results/reports/drl_adaptive_fl_pilot.md`. Confirmed: all 9 actions
  produce valid weight vectors (non-negative, sum to 1) over the
  synthetic clients; a 44-dim state was built correctly for both a
  first-round and a second-round case; an example reward computed to
  `0.016053` for a hand-traceable case (ΔF1=0.04, λ_fair·spread and
  λ_stab·0 subtracted); the untrained DQN's forward pass produced a
  finite 9-value Q-vector and a valid greedy action index (`3`,
  `"combined"`) — explicitly labeled in the output as reflecting only
  random initialization, not a learned policy.

## Failures / problems encountered

One test-authoring bug was caught and fixed during this session, before
any result was reported: an early draft of
`test_privacy_boundary.py`'s "no literal partition-path string" check
flagged `src/drl/reward.py`'s own docstring (which explained, in
prose, why the Phase 8 fairness metric isn't reused) as containing
`"train.parquet"`. This was not a real privacy violation — the string
appeared only in a comment, never in executable code — but the
docstring was reworded to remove the literal filename anyway, so the
test's guard stays simple and meaningful rather than needing a
docstring-aware exception carved into it. No other failure occurred.

## Limitations (of this pilot; not final-experiment limitations)

- No replay buffer and no DQN learning-update (`.learn()`/gradient
  step) were implemented — explicitly out of scope for this turn (see
  "What was intentionally NOT implemented" below).
- `lambda_fair`/`lambda_stab` values in `configs/drl.yaml` (0.1, 0.05)
  are this pilot's own documented starting defaults, not calibrated
  against any real multi-round trajectory yet — the config file records
  this as an open decision, not a finished tuning result.
- All pilot verification used synthetic, fabricated client signals —
  no real NSL-KDD partition, no real FL round, no real validation-F1
  trajectory was exercised.
- The 9-profile catalogue's specific blend weights (profiles 5-8) are a
  reasonable, documented spanning of the simplex, not derived from any
  optimization — consistent with the canonical design's own framing of
  the catalogue as fixed, not learned.

## What was intentionally NOT implemented

Per this turn's explicit scope limit: no replay buffer; no DQN
learning-update loop; no `src/federated/drl_trainer.py` round-loop
integration; no multi-episode pretraining script; no final 3-way
comparison run; no real NSL-KDD data loaded anywhere in this turn's new
code; no model checkpoint of any kind was produced (the DQN's weights
here are its untrained, seeded random initialization only, never
saved). No Phase 1-9A file, config, or result was modified.

## Exact next step (as recorded at the end of the FIRST pilot pass)

A review checkpoint (matching the design-checkpoint pattern already
used twice for the design itself) before any full DQN pretraining
budget is executed — specifically: (1) calibrate `lambda_fair`/
`lambda_stab` against a short real (not synthetic) trajectory before
committing to the pretraining-scale values, (2) design and implement
the replay buffer and learning-update loop only once that calibration

---

## Update — extended pilot (same day, follow-up session)

The first pilot pass above deliberately stopped short of a replay
buffer, target network, learning step, or environment abstraction. A
follow-up request in the same day's session asked for exactly those
pieces, still at pilot (not full-experiment) scale. This section
records what was added — the content above is preserved unchanged as
the historical record of the first, smaller pass.

### Additional files created

```
src/drl/replay_buffer.py                 ReplayBuffer (push/sample, seeded)
src/federated/drl_environment.py         DRLFederatedEnvironment (reset()/step())
src/federated/dqn_controller.py          DQNController (agent + buffer + epsilon schedule)
scripts/run_drl_pilot_rollout.py         tiny deterministic end-to-end rollout script
tests/drl/test_replay_buffer.py
tests/federated/test_drl_environment.py
tests/federated/test_dqn_controller.py
experiments/drl_adaptive_fl/pilot_run/rollout_summary.json
results/plots/drl_adaptive_fl_pilot/pilot_rollout_rewards.png
```

### Files modified (this extension)

- `src/drl/dqn_agent.py` — added a `target_network` (hard-synced via
  `update_target_network()`) and a real `learn()` gradient step
  (standard DQN target: `r + gamma * (1-done) * max_a' Q_target(s',a')`,
  MSE loss, Adam). The pre-existing `q_values()`/`act()` interface is
  unchanged.
- `tests/drl/test_dqn_agent.py` — extended with target-sync and
  `learn()` tests (loss actually decreases over repeated steps on a
  fixed batch; rejects an empty batch).
- `tests/drl/test_privacy_boundary.py` — extended to also cover
  `src/drl/replay_buffer.py`, `src/federated/drl_environment.py`, and
  `src/federated/dqn_controller.py`.
- `configs/drl.yaml` — added a `pilot:` block (`episodes: 3`,
  `rounds_per_episode: 3`, `replay_warmup: 4`, `batch_size: 4`,
  `seed: 42`) and `dqn.epsilon_decay_steps: 20`; updated the `dqn:`
  block's and file-level `notes:` comments to reflect that a tiny
  rollout has now run (see below) — no design VALUE was changed, only
  what was previously "not yet implemented" commentary.

### DQN architecture (now includes)

Everything from the first pass, plus: a target `QNetwork`
(hard-synced, not soft/Polyak — matches the canonical design's "target
network with hard update every N steps"), a `ReplayBuffer` (fixed
capacity, uniform sampling via a caller-supplied seeded
`numpy.random.Generator`), and `DQNAgent.learn()` — a real gradient
step, not a stub.

### Environment abstraction

`src/federated.drl_environment.DRLFederatedEnvironment` implements
`reset() -> observation` and `step(action) -> StepResult(observation,
reward, done, info)`, matching the design's state → action →
(training+aggregation, delegated) → evaluation → reward → next-state
loop. Critically, it does NOT itself run `FederatedClient`/
`FederatedServer` — it delegates "what happens when this round's
chosen weights are applied" to an injected `round_provider` callable,
so unit tests (and this pilot's rollout script) can supply a tiny,
deterministic, fully synthetic function and never touch real data,
while a future (not-yet-built) real training run could supply a
`round_provider` that wraps the unmodified Phase 6 client/server code
without this module ever needing to change.

### Tests (this extension)

31 additional tests: `test_replay_buffer.py` (5 — capacity eviction,
sample size/reproducibility, oversized-batch rejection, nonpositive
capacity rejection), `test_dqn_agent.py` additions (4 — target-sync
starting state, target-sync-on-demand, loss decreasing over repeated
`learn()` calls, empty-batch rejection), `test_drl_environment.py` (7
— state shape/determinism on `reset()`, `StepResult` shape, step
determinism, episode termination at the configured round count,
step-before-reset rejection, weight validity), `test_dqn_controller.py`
(6 — action validity, choose_action reproducibility, epsilon decay
schedule, target-sync-at-configured-frequency through the controller,
`maybe_learn` warmup gating, `maybe_learn` returning a real loss once
warmup is reached), plus privacy-boundary parametrization growing from
4 to 7 modules (further 9 new parametrized cases).

**One test bug was found and fixed in this extension**: an initial
`test_remember_updates_target_network_at_configured_frequency` assumed
the main network would have diverged from the target BEFORE the
configured sync point, but the chosen `replay_warmup`/
`target_update_frequency` values meant `learn()` hadn't actually run
yet when the sync fired — so the test's own premise was wrong, not the
code. Fixed by explicitly diverging the network via a direct
`agent.learn()` call before testing the sync trigger in isolation,
rather than relying on `remember()`'s own gating to coincidentally
produce divergence before the sync. Caught before any result was
reported; no production code changed because of this.

**Test totals after this extension**: `tests/drl/` + the two new
`tests/federated/test_drl_environment.py` /
`test_dqn_controller.py` files — **77/77 passed**. **Full suite —
268/268 passed** (237 prior + 31 new; zero regressions in any Phase
0-9A test).

### Tiny deterministic pilot ROLLOUT (the actual "run the pilot" step)

`scripts/run_drl_pilot_rollout.py` ran the full environment/controller
loop for `configs/drl.yaml`'s `pilot:` budget (3 episodes × 3 rounds =
9 environment steps), using a deterministic synthetic `round_provider`
(never real data). Result: 9 environment steps, 5 `learn()` steps (once
the replay buffer reached `replay_warmup=4`), episode total rewards
-0.1686 / -0.1536 / -0.1650. **Determinism was explicitly verified**:
the script was run twice and the two `rollout_summary.json` outputs
were confirmed byte-for-byte identical via `diff`. These reward numbers
reflect a barely-trained network (5 gradient steps) against a
deliberately simple synthetic reward dynamic — **they are not, and are
not presented as, evidence of DQN quality**; see
`results/reports/drl_adaptive_fl_pilot.md`'s addendum section for the
full write-up.

### No scientific/design problem was found during this extension

Per the experimental-safety instruction to stop and document rather
than silently redesign if the pilot exposed a problem: none of the 268
tests failed against the approved design, the environment's `step()`
output satisfies all approved invariants (weight validity, state
shape, deterministic transitions), and the one bug found (described
above) was in test-authoring, not in the approved design or its
implementation. No design change was made.

### Updated "what was intentionally NOT implemented"

Replay buffer, target network, and `learn()` — now implemented (this
extension). Still NOT implemented, unchanged from the first pass' own
scope limit: any `round_provider` that wraps the REAL
`FederatedClient`/`FederatedServer` (only synthetic providers exist so
far); the full 50-80-episode pretraining budget; the final 3-way
comparison run; any model checkpoint saved to disk (the DQN's weights
after this pilot's 5 `learn()` steps were never persisted); any change
to Phase 1-9A.

### Exact next step (superseding the first pass' next-step note)

Still a review checkpoint before the full pretraining budget — now
narrowed to exactly one remaining piece: implement a real
`round_provider` wrapping the unmodified `FederatedClient`/
`FederatedServer` (so the environment can run against `data/partitions/`
instead of synthetic signals), calibrate `lambda_fair`/`lambda_stab`
against that short real trajectory, and only then schedule the full
pretraining run — each as its own reviewed step, not bundled together.

---

## Update — aggregation-correctness integration test (re-confirmation pass, same day)

A follow-up re-confirmation request checked the pilot against an
explicit 14-item checklist. 13 items already had dedicated test
coverage from the two passes above; one gap was found: nothing had
directly proven that a DQN action's aggregation weights, once handed to
the REAL, unmodified `FederatedServer.aggregate_with_weights`
(`src/federated/server.py`, Phase 6/8's own code), produce a
mathematically correct weighted average of actual client model
parameters — `action_to_aggregation_weights` and
`aggregate_with_weights` had each been tested correct in isolation, but
never together.

Closed with `tests/federated/test_drl_aggregation_integration.py` (3
new tests, no source module changed): for all 9 actions, DQN-chosen
weights fed through the real server produce a result matching a
hand-computed weighted average (`atol=1e-6`); every action's weights
cover all 8 client IDs; the server's round counter increments correctly
after a DRL-driven aggregation.

**Test totals after this pass**: Phase 9B pilot tests — **80/80
passed**. Full suite — **271/271 passed** (268 prior + 3 new, zero
regressions). No design/scientific problem found; no redesign occurred.
`git diff --stat` against every Phase 1-9A path remains empty. Nothing
committed or pushed.

### Exact next step (unchanged by this pass)

Same as above: implement a real `round_provider` wrapping the
unmodified `FederatedClient`/`FederatedServer`, calibrate
`lambda_fair`/`lambda_stab` against a short real trajectory, then
schedule the full pretraining run as its own separately-reviewed step.

---

## Update — REAL round provider implemented and validated (follow-up session)

The one remaining blocker named above — no code training the
environment against real `data/partitions/` — is now closed. Full
detail (implementation decisions, privacy checks, real-data pilot
configuration/results, timing, problems/fixes, and test counts) is
recorded in
`docs/research/experiments/phase-9b-drl/README.md`, Entry 11 — not
duplicated here to avoid two competing copies of the same record.

**Status summary:** `src/federated/drl_round_provider.py`
(`RealNSLKDDRoundProvider` + `build_real_round_provider`) wraps the
unmodified `FederatedClient`/`FederatedServer` and satisfies
`DRLFederatedEnvironment`'s `RoundProvider` contract. One real,
measured, 2-round execution against the actual 8-client NSL-KDD
partition succeeded (validation F1: 0.6190 initial → 0.9652 after
round 1 → 0.9725 after round 2, using a FIXED "combined" action, not a
DQN decision) in 11.276s total wall-clock time. 17 new focused tests
pass; the full repository suite is at **288/288 passed**, zero
regressions.

**Still NOT implemented / NOT done:** the exact full pretraining
episode/round budget (still only a range in `configs/drl.yaml`, by
explicit instruction not to pin it yet); `lambda_fair`/`lambda_stab`
calibration against real (rather than synthetic) round trajectories;
any DQN-driven (as opposed to fixed-action) real-data round; the final
3-way comparison experiment; any change to Phase 1-9A.

### Exact next step (superseding the note above)

Two remaining decisions, each requiring its own explicit approval
before the full pretraining run: (1) pin an exact episode count and
rounds-per-episode value (currently a range: 50-80 episodes, 15-20
rounds); (2) decide whether `lambda_fair`/`lambda_stab` should be
recalibrated using this session's real validation-F1 trajectory
(0.619 → 0.965 → 0.973 — a much larger jump than the pilot's synthetic
reward dynamics assumed) before committing to the full run.
