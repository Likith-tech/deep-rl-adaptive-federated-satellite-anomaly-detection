# 009 — Phase 8: Rule-Based Adaptive Federated Learning

**Note on commit grouping:** committed together with Phase 7 in
`45c8d7d` — see the note at the top of `008-phase-7.md`.

**Reconstruction note:** Original chat prompt/response not preserved in
repository; reconstructed from `docs/project-progress/09-phase-8-rule-based-adaptive-fl.md`,
`results/reports/adaptive_fl_results.md`, `configs/adaptive_fl.yaml`,
`src/federated/adaptive.py`, `src/federated/adaptive_trainer.py`, and
Git history.

1. **Date/time:** 2026-08-14 00:44:38 +0530 (commit `45c8d7d`).
2. **Objective:** Replace Phase 6's sample-count-only FedAvg weighting
   with a transparent, deterministic client-scoring rule that responds
   to more than just data volume — explicitly the RULE-BASED (non-RL)
   half of adaptive FL, laying the groundwork Phase 9B's DQN will later
   learn instead of hand-specify.
3. **Original task/prompt:** Not preserved in repository.
4. **What Claude/AI was asked to do:** Not recoverable verbatim; scope
   inferred — build a 4-signal (performance/data/resource/fairness)
   scoring rule, wire it into FedAvg additively (no modification to
   Phase 6 code), run an ablation study.
5. **Important Claude response/report information:** Not recoverable
   verbatim.
6. **What was actually implemented:** `src/federated/adaptive.py`
   (`ClientSignals`, `build_resource_signal`, `normalize`,
   `compute_client_scores`, `scores_to_aggregation_weights`,
   `select_top_k_clients`); `src/federated/adaptive_trainer.py`
   (`run_adaptive_federated_training` — reuses `FederatedClient`/
   `FederatedServer` unchanged, evaluates each client's just-trained
   update on global validation for the "performance" signal, computes
   scores, aggregates via the newly-added `FederatedServer.aggregate_with_weights`);
   `weighted_average` added to `fedavg.py` (Phase 6's own
   `federated_average` re-expressed in terms of it, behavior unchanged,
   confirmed by Phase 6's own tests still passing byte-for-byte);
   `scripts/run_adaptive_fl_experiments.py`;
   `scripts/evaluate_adaptive_fl_results.py`.
7. **Files created:** `configs/adaptive_fl.yaml`,
   `results/reports/adaptive_fl_results.md`, `results/plots/adaptive_fl/*.png`,
   `results/models/federated/round_history.json`,
   `tests/federated/test_adaptive.py` (part of the combined commit `45c8d7d`).
8. **Files modified:** `src/federated/fedavg.py` (additive:
   `weighted_average` added), `src/federated/server.py` (additive:
   `aggregate_with_weights` added) — both confirmed additive, not
   destructive, since Phase 6's own `federated_average`/`aggregate`
   tests still pass unchanged.
9. **Dataset used:** Phase 6's EXACT partition (`data/partitions/`, NOT
   a fresh one) — deliberate, for a true apples-to-apples comparison
   against the committed Phase 6 baseline.
10. **Dataset status:** Real NSL-KDD, non-IID partitioned (identical to
    Phase 4/6).
11. **Model architecture:** Phase 2's `BaselineMLP`, unchanged.
12. **Hyperparameters:** Identical to Phase 6 (10 rounds, 1 local
    epoch, batch 128, lr 0.001, Adam, seed 42, shared init seed 42).
    Rule weights (the only intentional variable): `performance_only`
    (1/0/0/0), `resource_only` (0/0/1/0), `data_only` (0/1/0/0),
    `combined` (performance 0.40 / data 0.25 / resource 0.15 /
    fairness 0.20) — all fixed BEFORE any KDDTest+ evaluation.
13. **Experiments:** 4 rule configs run on Phase 6's exact partition,
    all 8 clients participating every round (weighting, not selection).
14. **Results (real, measured):**

    | Rule | Val F1 | KDDTest+ F1 | Fairness gap |
    |---|---:|---:|---:|
    | performance_only | 0.9890 | 0.7447 | **0.1245** (best fairness) |
    | resource_only | 0.9860 | **0.7547** (best test F1) | 0.2259 |
    | data_only | 0.9850 | 0.7397 | 0.2113 |
    | combined | 0.9883 | 0.7439 | 0.1496 |
    | Phase 6 FedAvg (reference) | 0.9879 | 0.7404 | not measured |

    `resource_only` beat Phase 6 by the widest margin (+0.0143 F1);
    `performance_only` — NOT the fairness-weighted `combined` rule —
    achieved the smallest fairness gap. Communication cost unchanged
    from Phase 6 (~14.6 MB / 10 rounds, all 8 clients every round).
15. **Problems/errors:** The counterintuitive fairness result
    (`combined`, despite a real 0.20 fairness weight, was NOT the
    fairness winner) was investigated and reported honestly rather than
    re-run or reweighted to look tidier.
16. **How problems were solved:** N/A — reported as a real, plausible
    (performance-on-shared-validation may already implicitly reward
    diversity) but statistically unconfirmed finding (single seed per
    rule), explicitly flagged as a question for later, more systematic
    search — directly motivating Phase 9B's premise of learning the
    weights instead of guessing them.
17. **Important decisions:** Weighting chosen over hard selection
    (selection risks silently dropping a rare-category client for a
    whole round — directly conflicts with the fairness goal); rule
    weights fixed before any test evaluation, never tuned against
    KDDTest+; built strictly additively on Phase 6's code.
18. **Rejected alternatives:** Hard top-k client selection was built
    and unit-tested (`select_top_k_clients`) but not used for these
    experiments — judged the higher-risk mechanism and out of this
    phase's scope.
19. **Limitations:** Rule-based only — genuinely NOT reinforcement
    learning (no reward signal, no learned parameters); single seed per
    rule; fairness metric is a proxy (each rule's final model evaluated
    on each client's own local training data, not a genuine held-out
    per-client test); resource signals remain simulated.
20. **Tests:** `tests/federated/test_adaptive.py` — signal
    normalization, scoring formula, weight renormalization, degenerate
    cases (all-equal signals, all-zero scores), plus Phase 6's own
    `test_fedavg.py`/`test_server.py` re-verified to still pass
    unchanged.
21. **Test count:** Cumulative repository test count at commit `45c8d7d`
    is 168 (shared with Phase 7 — see note in `008-phase-7.md`; the two
    phases' new tests were not separately counted since they were
    committed together).
22. **Git branch:** `project-development-3`.
23. **Commit hash:** `45c8d7d9642d5544053a0760e56db2b94b4c54d4`
24. **Push status:** Pushed — `origin/project-development-3` pointed
    exactly at this commit before Phase 9A moved local
    `project-development-3` one commit ahead.
25. **What was intentionally NOT done:** No DRL/DQN/reinforcement
    learning of any kind — every weight here is fixed and hand-chosen,
    not learned; no staleness-aware aggregation, FedProx,
    personalization, secure aggregation, or differential privacy; no
    hard client selection experimentally benchmarked.
26. **Lessons learned:** Not recoverable verbatim. The four wired-up
    signal families (performance/data/resource/fairness) and their
    honest, counterintuitive result (best-fairness rule ≠
    fairness-weighted rule) are the explicit, stated baseline Phase 9B
    must build on and try to beat — see
    `docs/project-progress/09-phase-8-rule-based-adaptive-fl.md`'s "What
    Phase 9 will do" section.
27. **Next step:** Originally slated as "Phase 9" (staleness-aware FL)
    in the early 20-phase plan; in practice, the next work done was
    Phase 9A (real spacecraft data validation, a separate track), with
    Phase 9B (DQN-learned adaptive FL, replacing this phase's fixed
    weights) following as the FL-track's next milestone.
