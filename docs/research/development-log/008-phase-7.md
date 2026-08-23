# 008 — Phase 7: Non-IID Federated Learning Experiments

**Note on commit grouping:** Phase 7 and Phase 8 were implemented and
committed together in a single commit, `45c8d7d` ("Phase 7-8: non-IID
and rule-based adaptive FL"). This entry and `009-phase-8.md` describe
the two phases separately (as they are logically and scientifically
distinct — a measurement study vs. an intervention), but they share the
same commit hash, date, branch, and push status.

**Reconstruction note:** Original chat prompt/response not preserved in
repository; reconstructed from `docs/project-progress/08-phase-7-non-iid-federated-learning.md`,
`results/reports/non_iid_results.md`, `configs/non_iid.yaml`,
`src/simulation/`, and Git history.

1. **Date/time:** 2026-08-14 00:44:38 +0530 (commit `45c8d7d`).
2. **Objective:** Isolate client data heterogeneity as the SOLE
   experimental variable, using the exact same FedAvg pipeline as
   Phase 6, across a controlled Dirichlet-alpha sweep.
3. **Original task/prompt:** Not preserved in repository.
4. **What Claude/AI was asked to do:** Not recoverable verbatim; scope
   inferred — re-run Phase 6's FedAvg on five fresh Dirichlet
   partitions (alpha = 0.1, 0.5, 1.0, 5.0, 10.0), holding every other
   factor fixed, and measure heterogeneity and fairness, not just F1.
5. **Important Claude response/report information:** Not recoverable
   verbatim.
6. **What was actually implemented:** `scripts/run_non_iid_experiments.py`
   (builds a fresh partition per alpha under `experiments/non_iid/alpha_<X>/`,
   runs the unmodified Phase 6 FedAvg pipeline on each);
   `scripts/analyze_non_iid_results.py` (KDDTest+ evaluation per alpha,
   client fairness gap, plots, report); `configs/non_iid.yaml`.
7. **Files created:** `configs/non_iid.yaml`,
   `results/reports/non_iid_results.md`, `results/plots/non_iid/*.png`,
   `experiments/non_iid/alpha_*/{partition_metadata,run_config,round_history}.json`,
   `tests/simulation/test_alpha_sweep.py`, `tests/federated/test_non_iid_smoke.py`
   (part of the combined 84-file, 15,521-insertion commit `45c8d7d`).
8. **Files modified:** None to Phase 6's FedAvg code (pipeline reused
   unmodified — that reuse is itself the point of this controlled
   study). The original Phase 4/6 partition (`data/partitions/`) was
   never touched.
9. **Dataset used:** NSL-KDD training data, five FRESH Dirichlet
   partitions (alpha = 0.1, 0.5, 1.0, 5.0, 10.0) under
   `experiments/non_iid/`, distinct from the original Phase 4/6
   partition.
10. **Dataset status:** Real NSL-KDD, freshly non-IID-partitioned per
    alpha (simulated client assignment, as in Phase 4).
11. **Model architecture:** Phase 2's `BaselineMLP`, unchanged.
12. **Hyperparameters:** Identical to Phase 6 for every alpha — model,
    10 rounds, 1 local epoch, batch size 128, learning rate 0.001,
    Adam, BCEWithLogitsLoss, seed 42 — alpha is the sole variable.
13. **Experiments:** 5 full FedAvg runs, one per Dirichlet alpha, each
    evaluated once on KDDTest+ after round selection by validation loss.
14. **Results (real, measured):** Mean pairwise JS distance fell
    monotonically from **0.6892** (alpha=0.1) to **0.1452** (alpha=10.0).
    Client fairness gap (best- minus worst-served satellite, own local
    data) shrank monotonically at every step, from **0.2623** to
    **0.0064** (~41x). KDDTest+ F1 was **non-monotonic**: rose to a
    peak of **0.7640 at alpha=5.0**, then fell to 0.7541 at alpha=10.0
    — reported exactly as measured. At alpha=0.1, several clients had
    zero local examples of rare categories (r2l/u2r) and the partition
    failed its own fairness constraints — reported honestly. The fresh
    alpha=0.5 run reproduced Phase 6's exact numbers (cross-phase
    reproducibility check).
15. **Problems/errors:** Non-monotonic KDDTest+ F1 vs. alpha was
    investigated, not smoothed over; explicitly flagged as possibly
    genuine saturation or run-to-run noise, undetermined with one run
    per alpha.
16. **How problems were solved:** N/A — reported as an open, honestly
    unresolved question for future multi-seed work, not papered over.
17. **Important decisions:** Every hyperparameter except alpha held
    fixed, so alpha is provably the sole variable; the original Phase
    4/6 partition was deliberately never touched, so this study could
    not contaminate Phase 6's committed baseline result.
18. **Rejected alternatives:** Not recorded in repository artifacts for
    this phase.
19. **Limitations:** Single seed per alpha — not a statistically
    powered comparison; the non-monotonic F1 finding cannot yet be
    distinguished from noise with this design.
20. **Tests:** `tests/simulation/test_alpha_sweep.py`,
    `tests/federated/test_non_iid_smoke.py`.
21. **Test count:** Cumulative repository test count at commit `45c8d7d`
    (`git grep "def test_" 45c8d7d -- '*.py'`): 168 (net +25 from Phase
    6's 143 — this net figure covers Phase 7 AND Phase 8 combined,
    since both were committed together).
22. **Git branch:** `project-development-3`.
23. **Commit hash:** `45c8d7d9642d5544053a0760e56db2b94b4c54d4`
24. **Push status:** Pushed — `origin/project-development-3` points
    exactly at this commit (as of before Phase 9A; Phase 9A later moved
    local `project-development-3` one commit ahead).
25. **What was intentionally NOT done:** No adaptive client selection
    or DRL — this is a measurement study, not an intervention.
26. **Lessons learned:** Not recoverable verbatim. The measured
    fairness-gap-vs-heterogeneity relationship became the direct
    motivation stated for Phase 8 ("Phase 7 then showed that... a
    client fairness gap that grew about 41x").
27. **Next step:** Phase 8 — replace sample-count-only weighting with a
    transparent rule that responds to more than just data volume.
