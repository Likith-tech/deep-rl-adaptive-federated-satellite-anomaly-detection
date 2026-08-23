# Phase 9B — DQN Adaptive Federated Learning: Results

**All numbers in this report are real, measured values from one
executed run of `scripts/run_phase9b_full_experiment.py`
(2026-08-15), read directly from
`experiments/drl_adaptive_fl/full_run/*.json`. Nothing here is
estimated, interpolated, or fabricated. Where a result is
unfavorable to DQN, it is reported exactly as measured — a worse
result than Phase 6/8 is a valid scientific outcome, not a defect to
hide.**

## 1. Research question

Can a DQN controller, choosing among the same 9 discrete weighting
profiles Phase 8 already defined, learn — from a reward signal built
only from validation-set performance — an aggregation policy that
matches or improves on Phase 8's best rule-based result, measured
honestly on the untouched KDDTest+ set?

## 2. Hypothesis

A learned policy *might* adapt its weighting choice per round in ways
a static rule cannot, and so might improve on Phase 8's `combined`
rule. Stated in advance as a hypothesis to test, not a target to
engineer toward (per the project's standing rule).

## 3. Experimental setup

- Real 8-client NSL-KDD partition (`data/partitions/`, the exact
  Phase 4/6 partition — untouched, not regenerated).
- Real global validation split (`data/processed/validation.parquet`).
- KDDTest+ (`data/processed/test.parquet`) touched exactly once, at
  the very end, after the policy was already frozen and a round was
  already selected by validation loss alone.
- `RealNSLKDDRoundProvider` — an adapter wrapping the **unmodified**
  Phase 6/8 `FederatedClient`/`FederatedServer` (no second FL
  implementation).

## 4. DQN architecture

Small MLP Q-network: input 44, hidden [64, 64] (ReLU), output 9
(one Q-value per weighting profile). Target network (hard-synced every
10 learning steps). Replay buffer (capacity 5,000, uniform sampling).
Adam, learning rate 0.001, γ=0.95. Epsilon-greedy: 1.0 → 0.05, linearly
decayed across the full 750-step pretraining budget. Seed 42
throughout. All values taken directly from `configs/drl.yaml`, none
invented for this run.

## 5. State

Exactly 44 dimensions (`performance_k`×8, `data_k`×8, `resource_k`×8,
`fairness_k`×8, validation-F1 trend×2, round-progress×1,
previous-action one-hot×9), built by `src/drl/state.py`, verified by
its own shape assertion every round of this run.

## 6. Action space

The approved discrete 9-profile weighting catalogue
(`performance_only`, `resource_only`, `data_only`, `combined`,
`uniform`, `fairness_leaning`, `performance_leaning`,
`resource_leaning`, `balanced_performance_fairness`). Weighting only —
all 8 clients participated every single round of all 750 pretraining
rounds and all 10 final-run rounds; no client was ever dropped.

## 7. Reward

```
r_t = ΔF1_t − λ_fair · perf_spread_t − λ_stab · 1[action_t ≠ action_{t-1}]
```
`λ_fair=0.1`, `λ_stab=0.05` (unchanged from `configs/drl.yaml`, not
retuned for this run). Built only from the global-validation-only
`ΔF1_t` and the per-round `performance_k` spread — **never** from
Phase 8's local-F1 fairness gap, and KDDTest+ was never read by this
function (verified by the existing AST-based privacy tests, re-run
before this experiment).

## 8. FL configuration

8 clients, Phase 2/6/8's exact `BaselineMLP` (121→[128,64]→1, dropout
0.3), local_epochs=1, batch_size=128, lr=0.001, Adam,
BCEWithLogitsLoss, shared_init_seed=42 — identical to Phase 6/8, not
changed for this experiment.

## 9. Training procedure

**Budget used (documented choice, per explicit instruction to use the
lower bound of the already-approved range rather than pause for
another decision):** 50 episodes × 15 rounds/episode = 750 real FL
rounds. `replay_warmup = dqn.batch_size = 32`. `epsilon_decay_steps =
750` (the approved 1.0→0.05 schedule, mechanically scaled to the real
budget's step count — not a new parameter). **Disclosed limitation**:
every episode reused the SAME real partition and the SAME
`shared_init_seed=42` (per this turn's explicit "do not create another
partition" instruction) — so episode-to-episode variation came only
from the DQN's own action sequence (exploration + evolving Q-values),
not from varied data as the original design's "many fresh partitions"
strategy envisioned. This is a real, disclosed deviation from that
part of the original design, not a silent one.

After pretraining, ONE frozen-policy (ε=0) run of 10 rounds (matching
Phase 6/8's exact protocol) on the same real partition. The round with
the lowest validation loss was selected — **round 10** (val_loss =
0.04105) — before any KDDTest+ access.

## 10. Results

### 10.1 DQN training behavior (measured)

| Metric | Value |
|---|---:|
| Episodes | 50 |
| Rounds/episode | 15 |
| Total env steps | 750 |
| Episode 1 total reward | −0.3572 |
| Episode 50 total reward | +0.1435 |
| Mean episode reward (all 50) | −0.1414 |
| Mean episode reward (last 10) | +0.1329 |
| DQN loss, step 1 | 0.0480 |
| DQN loss, last step | 0.0001 |
| DQN loss, mean of last 50 steps | 0.0001 |
| Mean validation F1 across all 750 rounds | 0.9837 |
| Total learn() steps | 719 |
| Wall-clock: pretraining | 6088.3s (101.5 min) |
| Wall-clock: final run | 97.4s |
| Wall-clock: total | 6189.7s (103.2 min) |

Episode reward rose from strongly negative (episode 1: −0.3572) to
mildly positive on average over the last 10 episodes (+0.1329) — a
real, measured upward trend, consistent with SOME learning signal
being captured. DQN loss fell from 0.048 to ~0.0001 and stayed there —
see §15 (Interpretation) for why this is reported as an ambiguous
signal, not unambiguous evidence of a good policy.

**Action distribution, all 750 rounds:** `combined` 258 (34.4%),
`uniform` 98 (13.1%), `data_only` 82 (10.9%), `fairness_leaning` 64
(8.5%), `resource_leaning` 64 (8.5%), `balanced_performance_fairness`
53 (7.1%), `resource_only` 49 (6.5%), `performance_only` 45 (6.0%),
`performance_leaning` 37 (4.9%).

**Action distribution, last 10 episodes (150 rounds, mostly
exploiting, ε≈0.05-0.18):** `combined` 94 (62.7%), `uniform` 28
(18.7%), `data_only` 14 (9.3%), `fairness_leaning` 6 (4.0%),
`performance_only` 3 (2.0%), `resource_leaning` 2 (1.3%),
`resource_only` 2 (1.3%), `performance_leaning` 1 (0.7%). The policy
concentrated heavily on `combined` — Phase 8's own primary,
reasoned-not-learned rule — during near-greedy late-training rounds.

### 10.2 Final frozen-policy run (10 rounds, ε=0)

| Round | Action chosen | Val F1 | Val loss |
|---:|---|---:|---:|
| 1 | performance_leaning | 0.9643 | 0.1192 |
| 2 | data_only | 0.9710 | 0.0623 |
| 3 | data_only | 0.9826 | 0.0493 |
| 4 | data_only | 0.9831 | 0.0451 |
| 5 | data_only | 0.9837 | 0.0423 |
| 6 | data_only | 0.9845 | 0.0414 |
| 7 | data_only | 0.9846 | 0.0411 |
| 8 | data_only | 0.9849 | 0.0411 |
| 9 | data_only | 0.9853 | 0.0416 |
| **10 (selected)** | data_only | **0.9861** | **0.0411** |

**Honest, measured note**: the FROZEN GREEDY policy selected
`data_only` for 9 of 10 rounds — a different, more concentrated
pattern than the `combined`-dominant distribution seen during
near-greedy late pretraining (§10.1). This is reported exactly as
observed, not smoothed into a single consistent "the DQN learned X"
narrative — see §15.

## 11. Baseline comparison

| Method | KDDTest+ F1 | KDDTest+ ROC-AUC | Source |
|---|---:|---:|---|
| Phase 6 FedAvg | 0.7404 | 0.8479 | stored, not rerun |
| Phase 8 performance_only | 0.7447 | — (not reported per-rule) | stored, not rerun |
| Phase 8 resource_only (best F1) | **0.7547** | — | stored, not rerun |
| Phase 8 data_only | 0.7397 | — | stored, not rerun |
| Phase 8 combined | 0.7439 | — | stored, not rerun |
| **Phase 9B DQN (this run)** | **0.7425** | **0.8361** | measured, this run |

**Phase 9B DQN's exact KDDTest+ result (measured once, frozen round
10 model):**

| Metric | Value |
|---|---:|
| Accuracy | 0.7548 |
| Precision | 0.9231 |
| Recall | 0.6210 |
| F1 | 0.7425 |
| ROC-AUC | 0.8361 |
| False Positive Rate | 0.0684 |
| Confusion matrix | `[[9047, 664], [4864, 7969]]` |
| KDDTest+ loss | 2.2515 |

**Honest reading**: DQN's F1 (0.7425) is marginally above Phase 6
(+0.0021) and Phase 8's `data_only` (+0.0028), essentially a tie with
`combined` (−0.0014) and `performance_only` (−0.0022), and clearly
below Phase 8's best, `resource_only` (−0.0122). DQN's ROC-AUC
(0.8361) is below Phase 6's (0.8479). **DQN did NOT beat Phase 8's
best rule-based result on F1 or beat Phase 6 on ROC-AUC in this run.**
This is reported as a real, valid result — not adjusted, not rerun to
seek a better number.

## 12. Fairness comparison

Post-hoc only (Phase 8's own-local-data F1 gap metric) — **never used
in the DQN's reward, action selection, or model selection**, computed
once, after the frozen model was already selected by validation loss.

| Client | Local F1 (Phase 9B DQN) |
|---|---:|
| SAT-01 | 0.9982 |
| SAT-02 | 0.9974 |
| SAT-03 | 0.9594 |
| SAT-04 | 0.9659 |
| SAT-05 | 0.9753 |
| SAT-06 | 0.9625 |
| SAT-07 | 0.9909 |
| SAT-08 | 0.8148 |
| **Mean** | 0.9580 |
| **Gap (max−min)** | **0.1833** |

| Method | Fairness gap |
|---|---:|
| Phase 8 performance_only (best) | 0.1245 |
| Phase 8 combined | 0.1496 |
| **Phase 9B DQN** | **0.1833** |
| Phase 8 data_only | 0.2113 |
| Phase 8 resource_only | 0.2259 |

**DQN's fairness gap (0.1833) is worse than Phase 8's `combined`
(0.1496) and `performance_only` (0.1245)**, though better than
`data_only`/`resource_only`. DQN did not improve fairness relative to
Phase 8's best rule in this run — reported exactly as measured.

## 13. Ablation

As specified (minimum required): **A. DQN adaptive weighting vs. B.
best Phase 8 rule-based adaptive weighting (`resource_only`, F1
0.7547)**. DQN underperforms B by 0.0122 F1. No additional ablation
sweep was run, per the explicit instruction to avoid an open-ended
hyperparameter search — one controlled comparison, reported honestly.

## 14. Failure cases (reported, not hidden)

- **DQN did not beat Phase 8's best rule (`resource_only`) on F1**,
  and did not beat Phase 8's best rule (`performance_only`) on
  fairness. A negative-relative-to-best-baseline result.
- **The frozen greedy policy's action pattern (9/10 rounds `data_only`)
  diverges from the pretraining-time near-greedy action distribution
  (62.7% `combined`)** — the two do not tell one consistent "the DQN
  learned to prefer X" story; this inconsistency is itself a finding,
  not resolved or explained away here.
- **DQN loss collapsing to ~0.0001** is consistent with either genuine
  convergence OR a degenerate near-constant Q-function (a known DQN
  failure mode when reward magnitudes are small); this run's evidence
  does not distinguish the two — flagged as an open question, not
  claimed as clean convergence.
- **Heterogeneity did not clearly destabilize training** — no NaN/Inf,
  no reward divergence, rewards stayed in a bounded, interpretable
  range throughout.
- **Resource signals**: cannot be isolated as helping or hurting in
  this single run — no resource-only ablation of the DQN itself was
  run (only the fixed Phase 8 `resource_only` rule, reused from
  storage).
- **Whether the DQN "actually learned meaningful action differences"**:
  partially — the reward trend (§10.1) and the shift in the final
  run's action pattern versus round 1 (performance_leaning → sustained
  data_only) suggest SOME learned discrimination between profiles, but
  the single-fixed-partition training regime (§9's disclosed
  limitation) means this cannot be distinguished from the policy
  simply memorizing one specific trajectory's dynamics rather than a
  generalizable adaptive strategy.

## 15. Interpretation

The most defensible reading of this run: the DQN learned SOMETHING
from its 750-step, single-partition training regime — reward trended
upward, DQN loss fell, and the final run's behavior differs
meaningfully from round 1 — but it did not learn a policy that
generalizes to beat the best already-known rule-based baseline in this
one measured run, and its own late-pretraining exploitation behavior
(mostly `combined`) is not even fully consistent with its own final
frozen-greedy behavior (mostly `data_only`). The single-fixed-partition
training regime (§9) is the most likely structural explanation: without
varied partitions/seeds across episodes, the DQN had limited incentive
or opportunity to learn a policy that generalizes beyond the one
trajectory shape it was trained on repeatedly.

## 16. Limitations

- Single run, single seed — not a statistically powered comparison,
  same caveat every prior phase (6, 7, 8) already carried.
- Single fixed partition across all 50 pretraining episodes (§9),
  a disclosed deviation from the original multi-partition design.
- `λ_fair`/`λ_stab` were not recalibrated against real (as opposed to
  synthetic-pilot) reward dynamics before this run, per explicit
  instruction not to pause for that decision.
- KDDTest+ loss (2.25) is high relative to validation loss (0.041) —
  consistent with the same known validation-vs-test generalization gap
  documented since Phase 2 (KDDTest+ contains attack types absent from
  training), not a new or DQN-specific problem.
- The DQN loss-collapse ambiguity (§14) is unresolved by this run.

## 17. Reproducibility

Seed 42 throughout (DQN init, shared model init, final-run policy
selection). Full round-by-round log:
`experiments/drl_adaptive_fl/full_run/pretraining_log.jsonl` (50
lines, one per episode, each containing every round's action, weights,
reward, val_f1, val_loss, full val_metrics, performance_spread,
epsilon, dqn_loss, replay_buffer_size, total_steps). Final run:
`final_run.json`. `configs/drl.yaml` fully specifies every
hyperparameter used. GPU/CPU note: this run executed on CPU
(`torch.cuda.is_available()` was False on this machine) — no
GPU-specific nondeterminism applies; standard PyTorch CPU-op
determinism is assumed, not independently re-verified bit-for-bit
beyond the pilot-stage determinism tests already passing.

## 18. Exact artifact locations

```
experiments/drl_adaptive_fl/full_run/pretraining_log.jsonl
experiments/drl_adaptive_fl/full_run/final_run.json
experiments/drl_adaptive_fl/full_run/kddtest_evaluation.json
experiments/drl_adaptive_fl/full_run/fairness_evaluation.json
experiments/drl_adaptive_fl/full_run/timing.json
experiments/drl_adaptive_fl/full_run/analysis_summary.json
results/plots/drl_adaptive_fl/{dqn_reward_by_episode,validation_f1_by_episode,
  dqn_loss_vs_step,epsilon_by_episode,action_distribution,
  client_weights_final_run,performance_spread_by_round,
  drl_vs_baselines_f1,drl_vs_baselines_roc_auc,client_fairness_drl,
  confusion_matrix}.png
results/reports/drl_adaptive_fl_results.md   (this file)
```

## 19. Was the hypothesis supported?

**No, not in this run.** The hypothesis (§2) was that a learned policy
might match or improve on Phase 8's best result. Measured outcome: DQN
(F1 0.7425, fairness gap 0.1833) underperformed Phase 8's best F1
(`resource_only`, 0.7547) and best fairness (`performance_only`,
0.1245), while roughly matching (not exceeding) Phase 6 and Phase 8's
`combined`/`data_only`. This is reported as the honest result of one
controlled run, not adjusted to appear more favorable.

## 20. What should happen next

Per the project's standing rule that a worse result is valid and not
grounds for silent redesign: (1) this result stands as-is; (2) a
genuinely diverse multi-partition pretraining regime (the original
design's intent, deferred this run per explicit instruction) is the
most likely candidate to address the single-fixed-partition limitation
identified in §9/§15; (3) the DQN-loss-collapse ambiguity (§14) would
benefit from logging the Q-value spread (not just the TD loss) in a
future run to distinguish convergence from degeneracy; (4) any future
run should be treated as a NEW, separately-reported result, not a
replacement for this one — this report stands as the honest record of
this specific, real, completed experiment.
