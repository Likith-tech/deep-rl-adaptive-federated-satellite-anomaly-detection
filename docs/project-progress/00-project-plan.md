# Project Plan & Progress Tracker

**Project:** Deep Reinforcement Learning-Driven Adaptive Federated
Framework for Satellite Network Anomaly Detection
**Platform:** OrbitShield

This is the master tracker for the whole project. Each phase has (or
will have) its own simple-language write-up in this folder
(`docs/project-progress/`), meant to help explain the project to
faculty and during viva — not just to record what was technically done.

## Status

```
Phase 0  — Project Foundation          COMPLETE
Phase 1  — Dataset & Preprocessing     COMPLETE
Phase 2  — Baseline Detection          COMPLETE
Phase 3  — Temporal Model (GRU+Attn)   COMPLETE (see caveat below)
Phase 4  — Satellite Simulation        COMPLETE
Phase 5  — Local Satellite Training    IN PROGRESS
Phase 6  — Federated Learning          NOT STARTED
Phase 7  — Non-IID FL                  NOT STARTED
Phase 8  — Adaptive FL                 NOT STARTED
Phase 9  — Staleness-Aware FL          NOT STARTED
Phase 10 — DRL Environment             NOT STARTED
Phase 11 — DQN Controller              NOT STARTED
Phase 12 — DRL Adaptive FL             NOT STARTED
Phase 13 — Resource-Aware Adaptation   NOT STARTED
Phase 14 — Full Evaluation             NOT STARTED
Phase 15 — Ablation Study              NOT STARTED
Phase 16 — OrbitShield Integration     NOT STARTED
Phase 17 — Final System & Documentation NOT STARTED
```

A phase is only marked COMPLETE once its code has actually been run
successfully and produced real, measured results — never based on plans
or partial work.

## Phase write-ups

| File | Covers |
|---|---|
| `01-phase-0-foundation.md` | Repository, backend, frontend skeleton |
| `02-phase-1-dataset.md` | NSL-KDD dataset selection, cleaning, preprocessing |
| `03-phase-2-baseline.md` | First real anomaly-detection model (MLP baseline) — F1 77.9% on KDDTest+ |
| `04-phase-3-spatio-temporal.md` | Temporal GRU+attention model — first attempt (ANY-anomaly labeling) was rejected as degenerate (100% F1, near-single-class task); corrected (LAST-record labeling, evidence-based) attempt scores F1 75.58% on KDDTest+, honestly slightly below the Phase 2 baseline's 77.90% (see caveat below) |
| `05-phase-4-satellite-simulation.md` | Simulated 8-client satellite environment via Dirichlet non-IID partitioning of the real training data (measured avg pairwise JS distance 0.527) + simulated per-client resource conditions — no FL/DRL yet |
| `06-phase-5-local-training.md` | Independent local training of the Phase 2 MLP on each satellite's own partition (no communication/aggregation); global-validation F1 ranged 0.9242 (SAT-01) to 0.9918 (SAT-02) across 8 clients |
| *(more added as each phase completes)* | |

## Relationship to the original 20-phase plan

Earlier project planning documents (see `README.md`) used a slightly
finer-grained 20-phase breakdown (e.g. splitting "dataset acquisition"
and "preprocessing" into two separate phases). This tracker uses the
phase grouping actually requested and executed session-by-session,
which combined some of those steps. No completed phase was reordered or
skipped — this is a renumbering for clarity, not a change in what was
built. See individual phase write-ups for exactly what each phase
covered.

## Phase 3 caveat (read before citing its results)

**Scope reminder:** Phase 3 establishes only the TEMPORAL
component/foundation of the eventual spatio-temporal model. No genuine
spatial or multi-satellite modeling has been implemented — that
requires the satellite-client/network simulation environment, which is
Phase 4 above (the next milestone, not a skipped step).

Phase 3 went through two experiments, both preserved for research
integrity:

1. **Rejected first attempt:** "sequence = anomaly if ANY record in the
   window is anomalous." Combined with NSL-KDD's ~46-57% per-record
   anomaly rate, this made ~100% of sequences carry the anomaly label —
   a degenerate, near-single-class task that trivially scored 100% on
   every metric. Not a leakage bug (verified by tests), not a
   fabricated result — a real measured outcome that was correctly
   identified as an invalid benchmark and rejected rather than reported
   as a success. Fully preserved in
   `experiments/temporal/initial_any_anomaly_temporal_results_ARCHIVE.md`
   and summarized in `results/reports/temporal_results.md`.
2. **Corrected second attempt:** four labeling strategies were measured
   against real data (`results/reports/sequence_labeling_analysis.md`)
   and "sequence label = the last record's own label" was selected on
   that evidence (non-degenerate on every split, no arbitrary
   threshold, fairest possible basis for comparison against the
   baseline). Retrained and evaluated once on KDDTest+: **F1 75.58%**,
   honestly slightly below the Phase 2 baseline's 77.90% — a real,
   trustworthy, non-degenerate result. Full detail and interpretation
   (including an attention-weight-based explanation for the gap) in
   `results/reports/temporal_results.md` and
   `docs/project-progress/04-phase-3-spatio-temporal.md`.

Phase 3 was reviewed and approved and is now marked **COMPLETE**.

## Phase 4 summary (satellite simulation)

Phase 4 builds a **simulation of a multi-satellite learning environment
using the real NSL-KDD dataset** — NSL-KDD remains terrestrial traffic;
nothing here claims otherwise. 8 simulated clients (`SAT-01`..`SAT-08`)
were created by Dirichlet-partitioning the real training data
(`data/processed/train.parquet`, 107,077 records) by attack category,
seed=42, alpha=0.5. Non-IID-ness was measured, not assumed: average
pairwise Jensen-Shannon distance between clients' category distributions
= **0.527** (0=identical, 1=maximally different). Each client also
received SIMULATED resource conditions (bandwidth, latency, compute,
availability, connectivity) sampled within configured ranges. Data
integrity was verified: all 107,077 training records assigned to
exactly one client, zero loss, zero duplication; validation stays a
single global set and KDDTest+ was never touched. Full detail in
`docs/project-progress/05-phase-4-satellite-simulation.md` and
`results/reports/satellite_simulation_report.md`. No Federated
Learning, DRL, or model training happens in this phase — environment
construction only.

## Phase 5 summary (local satellite training)

Phase 5 trains one independent copy of the Phase 2 baseline MLP per
satellite client (`SAT-01`..`SAT-08`), using ONLY that client's own
Phase 4 partition (`data/partitions/<client>/train.parquet`) — real
NSL-KDD records, no data shared between clients, no aggregation, no
communication. All 8 clients start from identical shared initial
weights (seed=42) so outcome differences reflect local data, not
initialization. Every client is evaluated on the SAME global validation
set (`data/processed/validation.parquet`) as the fair cross-client
comparison; KDDTest+ is untouched. Measured global-validation F1 ranged
from **0.9242** (`SAT-01`, 99.07% locally anomalous, dominated by dos)
to **0.9918** (`SAT-02`, 63.89% locally anomalous, the largest and most
balanced partition, 39,030 samples). An observed (not causally proven)
pattern: clients with extreme local anomaly rates or a missing attack
category (`SAT-07` has zero local `r2l` records) tended toward the
lower end of the ranking. Full detail in
`docs/project-progress/06-phase-5-local-training.md` and
`results/reports/local_training_results.md`. No Federated Learning,
DRL, or model aggregation happens in this phase.

## What's explicitly NOT done yet (as of Phase 5)

- No federated learning (no Flower, no FedAvg, no aggregation, no
  communication rounds, no server/client training loop) — local models
  from Phase 5 remain fully independent of each other.
- No DRL/DQN/reinforcement learning of any kind.
- The OrbitShield website is not connected to any real model or
  simulation data — it still shows "Awaiting live data."
- No spatial/geographic/orbital realism — the satellite simulation
  creates non-IID label distributions and simulated resource
  heterogeneity, not real orbital mechanics or inter-satellite links.
- Temporal context (Phase 3, last-record-label form) has not yet been
  shown to improve on the baseline — a real, useful finding for guiding
  later phases, not a blocker.
- No KDDTest+ evaluation of any Phase 5 model — that happens only once
  a federated methodology is established (Phase 6+).

These are all planned for later phases and are intentionally out of
scope until their turn.
