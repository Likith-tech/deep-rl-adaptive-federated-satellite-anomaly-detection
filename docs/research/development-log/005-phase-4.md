# 005 — Phase 4: Simulated Satellite Client Environment

**Reconstruction note:** Original chat prompt/response not preserved in
repository; reconstructed from `docs/project-progress/05-phase-4-satellite-simulation.md`,
`results/reports/satellite_simulation_report.md`, `data/partitions/README.md`,
and Git history.

1. **Date/time:** 2026-08-08 20:57:50 +0530 (commit `20f934b`).
2. **Objective:** Turn the single NSL-KDD dataset into 8 simulated
   satellite clients with non-IID data and simulated resource
   conditions, so Federated Learning has something realistic to operate
   on. No FL, DRL, or model training in this phase — environment
   construction only.
3. **Original task/prompt:** Not preserved in repository.
4. **What Claude/AI was asked to do:** Not recoverable verbatim; scope
   inferred — build client representations, Dirichlet non-IID
   partitioning, measured (not assumed) heterogeneity metrics, and
   simulated per-client resource conditions.
5. **Important Claude response/report information:** Not recoverable
   verbatim.
6. **What was actually implemented:** `src/simulation/satellite_client.py`
   (`SatelliteClient` representation); `src/simulation/partitioner.py`
   (Dirichlet non-IID partitioning); `src/simulation/non_iid_metrics.py`
   (JS divergence, Shannon entropy, per-category variance — measured,
   not assumed); `src/simulation/network_conditions.py` (simulated
   bandwidth/latency/compute/availability/connectivity, seeded uniform
   sampling); `src/simulation/simulator.py` + `scripts/create_satellite_partitions.py`
   + `scripts/analyze_satellite_partitions.py` orchestration.
7. **Files created:** 29 files changed (`git show --stat 20f934b`),
   including `results/reports/satellite_simulation_report.md`,
   `data/partitions/README.md`, `data/partitions/manifest.json`,
   `data/partitions/satellite_metadata.json`.
8. **Files modified:** None to prior phases' core logic (additive).
9. **Dataset used:** NSL-KDD training split (`data/processed/train.parquet`,
   107,077 records) — partitioned across 8 simulated clients. Validation
   stays a single global set; KDDTest+ untouched.
10. **Dataset status:** Real NSL-KDD records, **simulated** client
    assignment/resource conditions layered on top — explicitly labeled
    as simulation, not real satellite telemetry, throughout.
11. **Model architecture:** None — environment construction only.
12. **Hyperparameters:** Dirichlet concentration alpha = 0.5, seed = 42.
13. **Experiments:** Partition 8 clients (`SAT-01`..`SAT-08`) via
    Dirichlet(alpha=0.5) over attack category; generate simulated
    resource conditions; measure heterogeneity.
14. **Results (real, measured):** Average pairwise Jensen-Shannon
    distance between clients' category distributions = **0.527**
    (0=identical, 1=maximally different). All 107,077 training records
    assigned to exactly one client — zero loss, zero duplication
    (verified).
15. **Problems/errors:** None recorded.
16. **How problems were solved:** Not applicable.
17. **Important decisions:** Only the training split is partitioned;
    validation stays global and KDDTest+ untouched — preserving a fair,
    consistent evaluation basis for every later FL phase. Non-IID-ness
    was measured, never merely asserted.
18. **Rejected alternatives:** Not recorded in repository artifacts for
    this phase.
19. **Limitations:** Simulated satellite conditions, not real hardware
    or link measurements; no genuine orbital mechanics or inter-satellite
    geometry — a non-IID label distribution plus simulated resource
    heterogeneity, nothing more.
20. **Tests:** `tests/simulation/` — partitioner integrity (no data
    loss/duplication), non-IID metric correctness, network-condition
    generation determinism.
21. **Test count:** Cumulative repository test count at this commit
    (`git grep "def test_" 20f934b -- '*.py'`): 109 (net +40 from Phase 3's 69).
22. **Git branch:** `project-development`.
23. **Commit hash:** `20f934b8e917ebfeea21b02663172933d1bd6f05`
24. **Push status:** Local-only ancestor at the time of this log's
    writing — `origin/project-development` sits at `adf1be4` (Phase 3);
    this commit and everything after it up to `47728e6` exist on the
    remote as `origin/project-development-2`'s base history (that
    branch currently sits at `47728e6`, Phase 5) — i.e. this commit
    IS present on `origin/project-development-2`.
25. **What was intentionally NOT done:** No Federated Learning, no DRL,
    no model training — environment construction only.
26. **Lessons learned:** Not recoverable verbatim. The "measure, don't
    assume" approach to non-IID-ness established here (JS distance,
    entropy) becomes the fairness/heterogeneity measurement basis
    reused unchanged through Phases 7, 8, and the Phase 9B design.
27. **Next step:** Phase 5 — train an independent local model on each
    satellite's own partition, no communication yet.
