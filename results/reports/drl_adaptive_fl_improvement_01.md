# Phase 9B — Improvement 01: Per-Episode Local-Training Seed Diversity

**Result: H1 REJECTED. The original Phase 9B result (F1=0.7425,
fairness gap=0.1833, `experiments/drl_adaptive_fl/full_run/`,
`results/reports/drl_adaptive_fl_results.md`) remains the project's
primary, unmodified Phase 9B result and is UNCHANGED by this
experiment.** This report documents a real, measured, negative
outcome — not adjusted, not rerun to seek a better number.

## 1. Hypothesis (pre-registered before this run)

**H1:** Varying the local-training seed per pretraining episode
(instead of a constant seed for every round of every episode) would
increase training-experience diversity and improve the frozen policy's
KDDTest+ generalization relative to the original run.

- **Independent variable:** `training_config["seed"] = 42 + episode_number`
  (episode-varied; constant within an episode's 15 rounds/8 clients).
- **Controlled (identical to the original run):** partition, `shared_init_seed=42`,
  model architecture, local_epochs/batch_size/lr/optimizer/loss, reward
  formula and λ values, state/action design, 50 episodes × 15 rounds,
  final-run protocol (10 rounds, seed=42 unchanged, ε=0, val-loss
  selection), one KDDTest+ evaluation.
- **Success criterion (pre-registered):** KDDTest+ F1 ≥ 0.7500 AND no
  core metric degrades >0.01 without offsetting benefit AND fairness
  gap ≤ 0.20.
- **Failure criterion (pre-registered):** F1 ≤ 0.7450 OR any core
  metric degrades >0.01 without benefit OR fairness gap worsens >0.02.

## 2. Result vs. pre-registered criteria

| Criterion | Threshold | Measured | Met? |
|---|---|---:|---|
| F1 success | ≥ 0.7500 | 0.7397 | **No** |
| F1 failure | ≤ 0.7450 | 0.7397 | **Yes — failure triggered** |
| Fairness success | ≤ 0.20 | 0.2113 | **No** |
| Fairness failure | worsens >0.02 vs. 0.1833 | 0.2113 (+0.0279) | **Yes — failure triggered** |
| Accuracy/Precision/Recall/ROC-AUC | no drop >0.01 | max drop 0.0048 (ROC-AUC) | met (no large single-metric collapse) |

**Both pre-registered failure conditions were triggered. H1 is
rejected.** This was not a marginal/ambiguous case requiring
judgment — it is a clean failure against thresholds fixed in advance.

## 3. Full metric comparison

| Metric | Original Phase 9B (frozen) | Improvement 01 | Delta |
|---|---:|---:|---:|
| Accuracy | 0.7548 | 0.7526 | −0.0022 |
| Precision | 0.9231 | 0.9223 | −0.0008 |
| Recall | 0.6210 | 0.6175 | −0.0035 |
| **F1** | **0.7425** | **0.7397** | **−0.0028** |
| ROC-AUC | 0.8361 | 0.8313 | −0.0048 |
| FPR | 0.0684 | 0.0688 | +0.0004 |
| **Fairness gap** | **0.1833** | **0.2113** | **+0.0279 (worse)** |

Every metric moved in the unfavorable direction, though most by small
margins — the fairness gap's regression is the largest and clearest
effect.

## 4. Training behavior

Reward still improved across phases (Early −0.316 → Mid −0.197 → Late
+0.066 — nearly identical trend shape to the original run), confirming
the seed-diversity manipulation did not destabilize training. DQN loss
fell from ~0.047 to ~0.0001 (same pattern as the original run). 645 of
750 rewards (86.0%) were negative (vs. 87.3% in the original) —
essentially unchanged reward structure.

**Seed diversity was verified to actually occur**: all 50 episodes
used distinct seeds (43-92), confirmed programmatically.

## 5. Action/policy behavior — a genuinely different pattern

| Phase | Improvement 01 dominant action | Original dominant action |
|---|---|---|
| Early | resource_leaning/combined/data_only (~14% each, no clear leader) | combined (13.8%, no clear leader) |
| Mid | fairness_leaning (27.8%) | combined (29.4%) |
| Late | **fairness_leaning (52.9%)** | **combined (58.8%)** |
| Final frozen run | **data_only, all 10/10 rounds** | data_only, 9/10 rounds |

The seed-diversity manipulation DID change which action the policy
converged toward during training (`fairness_leaning` here vs.
`combined` in the original) — real evidence the intervention had a
genuine effect on learning dynamics, not a no-op. **Notably, the
FROZEN final run converged to `data_only` in BOTH experiments**,
despite different late-training exploration patterns — this is
consistent with (though does not prove) the state-normalization-mismatch
hypothesis raised in the original run's analysis (Entry 13, Task 3):
the final run's 10-round protocol may draw the greedy policy toward
`data_only` somewhat independently of what dominated pretraining,
because the `round_progress` state feature differs between the
15-round training and 10-round evaluation contexts. This report does
not claim this is proven — only that it is now supported by a SECOND
independent observation.

## 6. Interpretation

The seed-diversity intervention changed training dynamics (different
dominant action during pretraining) but did **not** improve — and
somewhat worsened — generalization and fairness. This suggests the
original run's single-fixed-trajectory limitation, while real, was
**not the dominant factor** limiting KDDTest+ performance. A more
likely dominant factor (not tested by this experiment): the reward's
fundamentally weak signal-to-noise ratio once validation F1 is
near-saturated (Entry 13, Task 4 finding — evidence: 86-87% negative
rewards in BOTH experiments, unchanged by seed diversity), and/or the
`round_progress` protocol mismatch (Entry 13, Task 3's still-untested
diagnostic) — neither of which this experiment targeted.

## 7. Reproducibility

Seed formula: `42 + episode_number` (deterministic, documented in
`scripts/run_phase9b_improvement_01.py:episode_training_config`).
Full round-by-round log: `experiments/drl_adaptive_fl/improvement_01/pretraining_log.jsonl`.
Timing: pretraining 7982.1s (133.0 min), final run 80.4s, total 8065.0s
(134.4 min) — longer than the original run's 103.2 min; not
investigated further (plausibly ordinary machine-load variance across
sessions, not a property of the code change).

## 8. What this does NOT show

- That seed diversity is a bad idea in general — only that THIS
  specific, narrow form of it (local-training seed only, same
  partition, same 50×15 budget) did not help here.
- That the original Phase 9B result was wrong or should be replaced —
  it remains the project's valid, primary result.
- Anything about the `round_progress` protocol-mismatch hypothesis
  directly — this experiment did not test that variable.

## 9. Final classification

**B — Improvement is not justified; keep the original Phase 9B
result.** The original result (F1=0.7425, `results/reports/drl_adaptive_fl_results.md`)
remains the project's Phase 9B result. This experiment is preserved as
a documented, valid negative finding, not deleted or hidden.

## 10. Artifact locations

```
experiments/drl_adaptive_fl/improvement_01/{pretraining_log.jsonl,final_run.json,
  kddtest_evaluation.json,fairness_evaluation.json,timing.json,stdout.log}
results/reports/drl_adaptive_fl_improvement_01.md   (this file)
```
No plots were generated for this report (the negative result is fully
characterized by the tables above; per the standing "no excessive
plots" guidance, visualizations are not duplicated here — the
original run's plotting script pattern is available in
`scripts/analyze_drl_full_run.py` if visual comparison is wanted
later).
