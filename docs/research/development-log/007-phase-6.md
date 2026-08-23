# 007 — Phase 6: Synchronous FedAvg Federated Learning Baseline

**Reconstruction note:** Original chat prompt/response not preserved in
repository; reconstructed from `docs/project-progress/07-phase-6-federated-learning.md`,
`results/reports/federated_results.md`, `configs/federated.yaml`,
`src/federated/`, and Git history.

1. **Date/time:** 2026-08-08 22:56:40 +0530 (commit `8f03d62`).
2. **Objective:** Implement the project's first genuine Federated
   Learning system — all 8 satellites jointly train one shared model
   via standard synchronous FedAvg, with no raw data ever leaving a
   client.
3. **Original task/prompt:** Not preserved in repository.
4. **What Claude/AI was asked to do:** Not recoverable verbatim; scope
   inferred — implement client/server/protocol/FedAvg modules with an
   explicit, tested privacy boundary; run 10 communication rounds; do
   one final KDDTest+ evaluation.
5. **Important Claude response/report information:** Not recoverable
   verbatim.
6. **What was actually implemented:** `src/federated/protocol.py`
   (`ClientUpdate`, `RoundRecord` — the only typed shapes exchanged);
   `src/federated/client.py` (`FederatedClient.train_round`, loads only
   its own partition); `src/federated/server.py` (`FederatedServer`,
   never imports pandas/reads parquet — enforced by a dedicated AST-based
   test); `src/federated/fedavg.py` (`federated_average`,
   `weighted_average`); `src/federated/trainer.py`
   (`run_federated_training`, the round-loop orchestrator);
   `scripts/run_federated_training.py`; `scripts/evaluate_federated_model.py`.
7. **Files created:** 25 files changed (`git show --stat 8f03d62`),
   including `configs/federated.yaml`, `results/reports/federated_results.md`,
   `results/plots/federated/`, `results/models/federated/round_history.json`.
8. **Files modified:** None to Phase 5's local-training code (additive
   — federated code reuses `BaselineMLP` and `build_initial_state_dict`).
9. **Dataset used:** Phase 4's exact partition (`data/partitions/`,
   alpha=0.5), Phase 1's global validation set, KDDTest+ (once, final).
10. **Dataset status:** Real NSL-KDD, non-IID partitioned (unchanged).
11. **Model architecture:** Phase 2's `BaselineMLP`, same shared
    initialization (seed 42) as Phase 5.
12. **Hyperparameters:** 10 communication rounds, 1 local epoch/round,
    batch size 128, learning rate 0.001, Adam, BCEWithLogitsLoss, seed
    42, sample-count-weighted averaging (`w_global = sum_k (n_k/N) w_k`).
13. **Experiments:** All 8 clients participate every round (no adaptive
    selection); best round selected by global validation loss alone
    (KDDTest+ never consulted for selection); one final KDDTest+
    evaluation.
14. **Results (real, measured):** Global validation F1 improved every
    round, 0.9683 (round 1) → **0.9879** (round 10, selected). Final,
    one-time KDDTest+: **F1 0.7404** — honestly slightly below Phase
    2's centralized baseline (0.7790), attributed to this baseline's
    deliberately small round/epoch budget. On global validation, FedAvg
    (0.9879) beat the Phase 5 local-only mean (0.9721) but not the
    single best local client (0.9918).
15. **Problems/errors:** None recorded as bugs. The below-baseline
    KDDTest+ F1 was reported and explained, not hidden or re-run to
    look better.
16. **How problems were solved:** N/A — explained honestly as a
    consequence of the deliberately small round/epoch budget, not a
    defect.
17. **Important decisions:** Server deliberately kept ignorant of raw
    data — enforced with a real test
    (`test_server_module_never_imports_pandas_or_touches_parquet`), not
    just a design intention; best-checkpoint selection strictly by
    validation loss.
18. **Rejected alternatives:** Not recorded in repository artifacts for
    this phase.
19. **Limitations:** No adaptive client selection, no staleness-aware
    aggregation, no resource-based weighting, no DRL — all later
    phases. Simulated per-client resource metadata recorded in round
    logs but not yet used to influence anything.
20. **Tests:** `tests/federated/test_fedavg.py`, `test_client.py`,
    `test_server.py` (including the privacy-boundary AST test),
    `test_trainer.py`.
21. **Test count:** Cumulative repository test count at this commit
    (`git grep "def test_" 8f03d62 -- '*.py'`): 143 (net +22 from Phase 5's 121).
22. **Git branch:** `project-development-2`.
23. **Commit hash:** `8f03d6223cd1d1e7c0a6052cd4d1414a146ec862`
24. **Push status:** Local `project-development-2` is 1 commit ahead of
    `origin/project-development-2` (which sits at `47728e6`, Phase 5) —
    **this Phase 6 commit is NOT pushed to the remote**, as of this
    log's writing.
25. **What was intentionally NOT done:** No adaptive client selection,
    staleness-aware aggregation, or DRL.
26. **Lessons learned:** Not recoverable verbatim. The
    `federated_average`/`weighted_average` split in `fedavg.py` (a
    general primitive underneath the standard rule) was structured so
    Phase 8 could add rule-based weighting without touching this
    phase's code — later confirmed additive when Phase 8 was built.
27. **Next step:** Phase 7 — isolate client data heterogeneity as a
    controlled variable using the same FedAvg pipeline.
