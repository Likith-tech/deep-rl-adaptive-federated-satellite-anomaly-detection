# 003 — Phase 2: Baseline Anomaly Detection Model

**Reconstruction note:** Original chat prompt/response not preserved in
repository; reconstructed from `docs/project-progress/03-phase-2-baseline.md`,
`results/reports/baseline_results.md`, and Git history.

1. **Date/time:** 2026-08-08 15:50:09 +0530 (commit `661ae74`).
2. **Objective:** Train a simple, trustworthy first model as a reference
   point every later (spatio-temporal/federated/DRL) result gets
   compared against.
3. **Original task/prompt:** Not preserved in repository.
4. **What Claude/AI was asked to do:** Not recoverable verbatim; scope
   inferred — implement an MLP baseline, train with early stopping,
   evaluate once on KDDTest+, report honestly.
5. **Important Claude response/report information:** Not recoverable
   verbatim.
6. **What was actually implemented:** `BaselineMLP` model
   (121 → 128 → 64 → 1, ReLU, Dropout); `scripts/train_baseline.py`;
   `scripts/evaluate_baseline.py`; `results/reports/baseline_results.md`;
   `results/plots/baseline/` (confusion matrix, training curves).
7. **Files created:** 22 files changed (`git show --stat 661ae74`).
8. **Files modified:** Wiring into existing `src/models`/`src/training`
   packages from the Phase 0 skeleton.
9. **Dataset used:** NSL-KDD, Phase 1's processed split (107,077 train /
   18,896 validation / 22,544 test).
10. **Dataset status:** Real, terrestrial (unchanged from Phase 1).
11. **Model architecture:** MLP, 121 → Dense(128, ReLU, Dropout) →
    Dense(64, ReLU, Dropout) → Dense(1) → sigmoid probability.
12. **Hyperparameters:** BCE loss, Adam, batch size 256, learning rate
    0.001, seed 42, max 30 epochs with early stopping (patience 5 on
    validation loss).
13. **Experiments:** Single training run, best checkpoint selected by
    validation loss, one final KDDTest+ evaluation.
14. **Results (real, measured):** Training stopped early at epoch 18
    (best checkpoint epoch 13), ~48 seconds on CPU. Validation: 99.61%
    accuracy / 99.49% precision / 99.68% recall / 99.59% F1. **KDDTest+
    (final, honest result):** 78.32% accuracy, 92.78% precision, 67.13%
    recall, 77.90% F1, 89.72% ROC-AUC, 6.90% FPR. Confusion matrix:
    9,041 TN / 670 FP / 4,218 FN / 8,615 TP.
15. **Problems/errors:** None recorded as bugs. The large
    validation-vs-test gap was investigated and attributed (not
    hidden) to KDDTest+'s deliberately-included unseen attack types —
    documented as an expected, known property of this benchmark.
16. **How problems were solved:** N/A — this was an explained
    characteristic of the benchmark, not a defect requiring a fix.
17. **Important decisions:** Deliberately simple architecture (no
    aggressive hyperparameter tuning) to keep the baseline a
    trustworthy, easy-to-interpret reference rather than a
    best-possible score; best-checkpoint selection strictly by
    validation loss, never by test performance.
18. **Rejected alternatives:** Not recorded in repository artifacts for
    this phase.
19. **Limitations:** No time/sequence modeling (records judged fully
    independently); ~1/3 of test attacks missed, mostly novel ones;
    terrestrial data only; centralized (no FL yet); hyperparameters not
    aggressively tuned (intentional).
20. **Tests:** Model architecture and training-loop tests
    (`tests/models/`, `tests/training/`).
21. **Test count:** Cumulative repository test count at this commit
    (`git grep "def test_" 661ae74 -- '*.py'`): 30 (net +12 from Phase 1's 18).
22. **Git branch:** `project-development`.
23. **Commit hash:** `661ae74bb684fdfb512975f09848b64d98cd7e90`
24. **Push status:** Ancestor of `origin/project-development` (up to
    date with local `project-development` at `adf1be4`).
25. **What was intentionally NOT done:** No temporal/sequence modeling,
    no satellite simulation, no federated learning, no hyperparameter
    search.
26. **Lessons learned:** Not recoverable verbatim. The
    validation-vs-test gap investigation established the project's
    recurring practice (repeated in every later phase) of reporting and
    explaining honest gaps rather than smoothing them over.
27. **Next step:** Phase 3 — build a temporal (GRU+attention) model that
    reads sequences instead of single records, and compare its result
    directly against this baseline's real numbers.
