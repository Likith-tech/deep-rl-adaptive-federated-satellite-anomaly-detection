# Phase 9B � DRL Pilot Verification Report

**This is a PILOT MECHANICS verification, NOT a performance
result.** No DQN training occurred. No NSL-KDD data was loaded.
No claim is made here that DQN improves (or worsens) anomaly
detection � see
`docs/project-progress/12-phase-9b-drl-pilot.md` for the full
write-up and `docs/project-progress/11-phase-9b-drl-adaptive-federated-learning.md`
for the approved design this pilot verifies.

## What this verifies

- `configs/drl.yaml` loads and validates against the approved
  design invariants (state dimension 44, action count 9, 8
  clients).
- All 9 action profiles produce valid (non-negative, sum-to-1)
  per-client aggregation weights over 8 SYNTHETIC clients � via
  the exact, unmodified Phase 8 `compute_client_scores` /
  `scores_to_aggregation_weights` pipeline.
- State construction produces the exact 44-dimensional vector
  for both a first-round (no previous action/trend) and a
  second-round (with them) case.
- Reward calculation for one example round transition.
- The DQN's untrained forward pass produces a 9-value Q-vector
  and a valid greedy action index � reflecting only its random
  seeded initialization, not any learned behavior.

## Action profile catalogue (fixed, deterministic � not a result)

| Profile | performance | data | resource | fairness |
|---|---:|---:|---:|---:|
| performance_only | 1.00 | 0.00 | 0.00 | 0.00 |
| resource_only | 0.00 | 0.00 | 1.00 | 0.00 |
| data_only | 0.00 | 1.00 | 0.00 | 0.00 |
| combined | 0.40 | 0.25 | 0.15 | 0.20 |
| uniform | 0.25 | 0.25 | 0.25 | 0.25 |
| fairness_leaning | 0.20 | 0.20 | 0.20 | 0.40 |
| performance_leaning | 0.55 | 0.15 | 0.10 | 0.20 |
| resource_leaning | 0.20 | 0.20 | 0.45 | 0.15 |
| balanced_performance_fairness | 0.35 | 0.15 | 0.10 | 0.40 |

## Example reward computation

```json
{
  "val_f1_prev": 0.7,
  "val_f1": 0.74,
  "action": 3,
  "previous_action": null,
  "lambda_fair": 0.1,
  "lambda_stab": 0.05,
  "reward": 0.016053
}
```

## Untrained DQN forward pass (random init only � not a policy)

```json
{
  "seed": 42,
  "q_values": [
    0.147869,
    -0.040317,
    -0.057811,
    0.199883,
    -0.212874,
    -0.08018,
    0.007701,
    0.093069,
    0.070727
  ],
  "greedy_action": 3,
  "greedy_action_name": "combined",
  "note": "Untrained network \u2014 this action choice reflects random initialization only, NOT a learned policy. Not a performance claim."
}
```

## What this does NOT show

- Whether a trained DQN improves or worsens anomaly detection �
  no training was run.
- Any comparison against Phase 6 FedAvg or Phase 8 rule-based
  adaptive FL results � those remain untouched, real, committed
  numbers this pilot did not re-derive or approach.
- Performance on real NSL-KDD data � every signal here is
  synthetic.

See `results/plots/drl_adaptive_fl_pilot/action_profile_weights.png`
for the visualized action catalogue and
`experiments/drl_adaptive_fl/pilot/pilot_verification.json` for the
full machine-readable output of this script.

## Addendum — tiny deterministic pilot ROLLOUT (`scripts/run_drl_pilot_rollout.py`)

**Note:** this addendum is appended manually because
`scripts/run_drl_pilot.py` regenerates the rest of this file from
scratch on every run (it does not know about the rollout script's
output) — re-running the mechanics-verification script alone will wipe
this section, so it must be re-added if that happens (as it did once
during this project's own development, recorded honestly in
`docs/research/experiments/phase-9b-drl/README.md`).

A second, separate pilot artifact exercises the full
`DRLFederatedEnvironment.reset()/step()` + `DQNController.choose_action()/
remember()/maybe_learn()` loop end-to-end, using `configs/drl.yaml`'s
`pilot:` block (3 episodes x 3 rounds = 9 environment steps total) and
a deterministic, purely synthetic `round_provider` — still **NOT**
real NSL-KDD data, still **NOT** a performance claim.

- Total environment steps: 9. Total `learn()` steps: 5 (steps where
  the replay buffer had reached `pilot.replay_warmup=4` transitions).
- Episode total rewards: episode 1 = -0.1686, episode 2 = -0.1536,
  episode 3 = -0.1650 — small negative numbers by construction of this
  rollout's deliberately simple synthetic `round_provider`, **not
  evidence about DQN quality**.
- **Determinism verified repeatedly**: run twice in an earlier session
  and again this session (3 total runs across 2 sessions) — every
  `rollout_summary.json` output has been byte-for-byte identical.

See `experiments/drl_adaptive_fl/pilot_run/rollout_summary.json` for
the full per-round log and
`results/plots/drl_adaptive_fl_pilot/pilot_rollout_rewards.png` for the
per-step reward and learn-loss plot.
