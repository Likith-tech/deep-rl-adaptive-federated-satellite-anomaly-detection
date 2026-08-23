# 006 — Phase 5: Independent Local Satellite Training

**Reconstruction note:** Original chat prompt/response not preserved in
repository; reconstructed from `docs/project-progress/06-phase-5-local-training.md`,
`results/reports/local_training_results.md`, and Git history.

1. **Date/time:** 2026-08-08 22:00:30 +0530 (commit `47728e6`).
2. **Objective:** Establish what happens if every satellite trains
   completely alone (no communication), as the honest baseline
   Federated Learning must beat.
3. **Original task/prompt:** Not preserved in repository.
4. **What Claude/AI was asked to do:** Not recoverable verbatim; scope
   inferred — train an independent copy of the Phase 2 MLP per
   satellite client, evaluate all 8 on a shared global validation set.
5. **Important Claude response/report information:** Not recoverable
   verbatim.
6. **What was actually implemented:** `scripts/train_local_models.py`
   (trains 8 independent `BaselineMLP` copies, one per `SAT-01..SAT-08`,
   from identical shared initial weights, seed 42, no aggregation);
   `scripts/evaluate_local_models.py` (evaluates every client on the
   same global validation set, generates confusion matrices/training
   curves/report).
7. **Files created:** 45 files changed (`git show --stat 47728e6`),
   including `results/reports/local_training_results.md`,
   `results/plots/local_training/`, `results/models/local/SAT-*/`
   (metadata + training history — checkpoints themselves gitignored).
8. **Files modified:** None to prior phases' core logic (additive/reuse
   of the Phase 2 `BaselineMLP` architecture).
9. **Dataset used:** Each satellite's own Phase 4 local partition
   (`data/partitions/<client>/train.parquet`) — real NSL-KDD records,
   no data shared between clients.
10. **Dataset status:** Real NSL-KDD, non-IID partitioned (unchanged
    from Phase 4).
11. **Model architecture:** Phase 2's `BaselineMLP` (121 → 128 → 64 → 1),
    one independent copy per client.
12. **Hyperparameters:** Shared initial weights, seed 42 for all 8
    clients (so outcome differences reflect local data, not
    initialization); other training hyperparameters match Phase 2's.
13. **Experiments:** 8 independent local training runs, each evaluated
    on the same global validation set; KDDTest+ untouched.
14. **Results (real, measured):** Global-validation F1 ranged from
    **92.4%** (`SAT-01`, 99.1% locally anomalous) to **99.2%**
    (`SAT-02`, largest and most balanced partition, 39,030 samples). An
    observed (not causally proven) pattern: clients with extreme local
    anomaly rates or a missing category (`SAT-07` has zero local `r2l`
    records) tended toward the lower end.
15. **Problems/errors:** None recorded.
16. **How problems were solved:** Not applicable.
17. **Important decisions:** All 8 clients evaluated on the identical
    global validation set (not their own local data) for a fair
    cross-client comparison; shared initialization to isolate data
    effects from initialization luck.
18. **Rejected alternatives:** Not recorded in repository artifacts for
    this phase.
19. **Limitations:** No communication/aggregation — this phase
    deliberately measures the "alone" case only; local data volume and
    category coverage varies a great deal by design (Phase 4's
    non-IID partition).
20. **Tests:** Local-training-loop tests (`tests/training/`).
21. **Test count:** Cumulative repository test count at this commit
    (`git grep "def test_" 47728e6 -- '*.py'`): 121 (net +12 from Phase 4's 109).
22. **Git branch:** `project-development-2` (this commit is the tip of
    `origin/project-development-2`).
23. **Commit hash:** `47728e670b9936a9d9fc367384c88e0de58e9341`
24. **Push status:** Pushed — `origin/project-development-2` points
    exactly at this commit.
25. **What was intentionally NOT done:** No Federated Learning, no
    communication or aggregation between clients, no DRL.
26. **Lessons learned:** Not recoverable verbatim. The resulting
    92.4%-99.2% local F1 spread became the direct point of comparison
    Phase 6's FedAvg result is measured against in
    `docs/project-progress/00-project-plan.md`'s Phase 6 summary.
27. **Next step:** Phase 6 — let the 8 satellites cooperate via
    standard synchronous FedAvg.
