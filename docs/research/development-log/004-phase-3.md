# 004 — Phase 3: Temporal Anomaly Detection Foundation

**Reconstruction note:** Original chat prompt/response not preserved in
repository; reconstructed from `docs/project-progress/04-phase-3-spatio-temporal.md`,
`docs/project-progress/00-project-plan.md` (Phase 3 caveat section),
`results/reports/temporal_results.md`,
`results/reports/sequence_labeling_analysis.md`,
`experiments/temporal/initial_any_anomaly_temporal_results_ARCHIVE.md`,
and Git history.

1. **Date/time:** 2026-08-08 19:10:12 +0530 (commit `adf1be4`).
2. **Objective:** Build a GRU + learned temporal-attention model reading
   sequences of records instead of single records — the TEMPORAL
   component/foundation of the eventual spatio-temporal model (no
   spatial/multi-satellite component yet).
3. **Original task/prompt:** Not preserved in repository.
4. **What Claude/AI was asked to do:** Not recoverable verbatim; scope
   inferred — build sequence construction, a GRU+attention model,
   determine a sound sequence-labeling strategy, train and evaluate
   honestly on KDDTest+.
5. **Important Claude response/report information:** Not recoverable
   verbatim.
6. **What was actually implemented:** Two experiments, both preserved:
   (a) a first attempt labeling a sequence "anomaly" if ANY record in
   it was anomalous — found degenerate and **rejected**, archived in
   `experiments/temporal/initial_any_anomaly_temporal_results_ARCHIVE.md`
   rather than reported as a success; (b) a corrected attempt, after
   measuring four candidate labeling strategies against real data
   (`results/reports/sequence_labeling_analysis.md`), selecting
   "label by the sequence's last record" on that evidence, retraining,
   and evaluating once on KDDTest+.
7. **Files created:** 24 files changed (`git show --stat adf1be4`),
   including the temporal model code, `scripts/analyze_sequence_labeling.py`,
   `scripts/train_temporal.py`, `scripts/evaluate_temporal.py`,
   `results/reports/temporal_results.md`,
   `results/reports/sequence_labeling_analysis.md`,
   `experiments/temporal/initial_any_anomaly_temporal_results_ARCHIVE.md`.
8. **Files modified:** Preprocessing/loading code extended to produce
   sequences rather than single records.
9. **Dataset used:** NSL-KDD (same Phase 1 processed split).
10. **Dataset status:** Real, terrestrial (unchanged).
11. **Model architecture:** GRU + learned temporal attention, reading
    sequences of NSL-KDD records (`TemporalGRUAttention`, later reused
    unmodified in Phase 9A for OPSSAT-AD).
12. **Hyperparameters:** Sequence-length candidates `[8, 16, 32]`
    evaluated, best selected by validation loss (per README's "Temporal
    model" section); other training hyperparameters not individually
    re-verified for this log entry — see `results/reports/temporal_results.md`.
13. **Experiments:** (1) Rejected: ANY-anomaly sequence labeling —
    produced ~100% anomaly-labeled sequences given NSL-KDD's ~46-57%
    per-record anomaly rate, a degenerate near-single-class task,
    correctly identified as an invalid benchmark rather than reported.
    (2) Accepted: four labeling strategies measured, LAST-record
    labeling selected on evidence, retrained, evaluated once on
    KDDTest+.
14. **Results (real, measured):** Corrected model: 75.8% accuracy,
    92.2% precision, 64.0% recall, 75.6%/75.58% F1 (both figures appear
    across README/project-plan; treated as the same real measured
    result), 89.2% ROC-AUC on KDDTest+ — honestly slightly below the
    Phase 2 baseline's 77.9% F1.
15. **Problems/errors:** The ANY-anomaly labeling choice produced a
    degenerate 100%-score result — not a leakage bug (verified by
    tests), but an invalid benchmark design.
16. **How problems were solved:** Caught, investigated, and **rejected**
    rather than reported as a success; archived for research integrity;
    four alternative labeling strategies were then measured against
    real data before selecting a replacement, rather than guessing.
17. **Important decisions:** LAST-record labeling selected specifically
    because it was non-degenerate on every split, had no arbitrary
    threshold, and gave the fairest possible basis for comparison
    against the Phase 2 baseline.
18. **Rejected alternatives:** ANY-anomaly sequence labeling (rejected,
    see above); the other 3 of the 4 measured labeling strategies were
    not selected (see `results/reports/sequence_labeling_analysis.md`
    for the comparison).
19. **Limitations:** Establishes only the TEMPORAL
    component/foundation — no genuine spatial or multi-satellite
    modeling yet (that is Phase 4); final result honestly slightly
    below the Phase 2 baseline.
20. **Tests:** Sequence-construction and temporal-model tests.
21. **Test count:** Cumulative repository test count at this commit
    (`git grep "def test_" adf1be4 -- '*.py'`): 69 (net +39 from Phase 2's 30).
22. **Git branch:** `project-development`.
23. **Commit hash:** `adf1be43f5dd19dc2f1144b2d60ecfa60f905d42`
24. **Push status:** `origin/project-development` currently points
    exactly at this commit — pushed and up to date.
25. **What was intentionally NOT done:** No spatial/multi-satellite
    modeling, no federated learning, no DRL.
26. **Lessons learned:** Not recoverable verbatim. The rejected
    ANY-anomaly experiment established the project's standing practice
    (repeated explicitly in the Phase 9A audit) of investigating
    suspiciously perfect results rather than reporting them, and of
    preserving rejected experiments rather than deleting them.
27. **Next step:** Phase 4 — build the simulated multi-satellite client
    environment, which gives meaning to the spatial half of
    "spatio-temporal."
