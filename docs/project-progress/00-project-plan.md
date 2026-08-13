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
Phase 5  — Local Satellite Training    COMPLETE
Phase 6  — Federated Learning          COMPLETE
Phase 7  — Non-IID FL                  COMPLETE
Phase 8  — Adaptive FL (rule-based)    COMPLETE (see caveat below)
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
| `07-phase-6-federated-learning.md` | First real Federated Learning: standard synchronous FedAvg across all 8 clients, 10 rounds, sample-count-weighted averaging; final KDDTest+ F1 0.7404 (honestly below Phase 2's 0.7790, real measured result) |
| `08-phase-7-non-iid-federated-learning.md` | Controlled Dirichlet-alpha sweep (0.1/0.5/1.0/5.0/10.0) isolating client heterogeneity as the sole variable in the same FedAvg pipeline; measured JS distance fell from 0.6892 (alpha=0.1) to 0.1452 (alpha=10.0) and the client fairness gap shrank monotonically from 0.2623 to 0.0064 (~41x), while KDDTest+ F1 peaked at alpha=5.0 (0.7640) non-monotonically |
| `09-phase-8-rule-based-adaptive-fl.md` | Deterministic (non-RL) client-scoring rule replacing Phase 6's sample-count-only FedAvg weighting; 4 ablations (performance/resource/data-only, combined) on Phase 6's exact partition — `resource_only` reached the highest KDDTest+ F1 (0.7547 vs Phase 6's 0.7404) and `performance_only` (not the fairness-weighted `combined` rule) reached the smallest client fairness gap (0.1245), an honest, counterintuitive result reported as measured |
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

## Phase 6 summary (federated learning baseline — FedAvg)

Phase 6 implements the first genuine Federated Learning experiment:
standard, synchronous FedAvg (McMahan et al.) across all 8 satellite
clients, sample-count-weighted averaging
(`w_global = sum_k (n_k/N) * w_k`, `src/federated/fedavg.py`), with an
explicit, tested client/server privacy boundary — the server
(`src/federated/server.py`) never imports pandas or reads a parquet
file; it only ever receives model parameters and sample counts
(`src/federated/protocol.py`). Same Phase 2 MLP architecture, same
shared initial weights (seed=42) as Phase 5. **10 communication
rounds**, 1 local epoch per client per round, ALL 8 clients
participating every round (no adaptive selection yet). Global
validation F1 improved every round, from 0.9683 (round 1) to **0.9879**
(round 10, selected as best by validation loss — KDDTest+ never
consulted for selection). Final, one-time KDDTest+ evaluation: **F1
0.7404**, honestly slightly below the Phase 2 centralized baseline's
0.7790 — a real, measured result attributed to the deliberately small
round/epoch budget of this baseline, not adjusted or hidden. On global
validation, FedAvg (0.9879) beat the Phase 5 local-only mean (0.9721)
but not the single best local client (0.9918) — compared on matching
dataset splits throughout. Full detail in
`docs/project-progress/07-phase-6-federated-learning.md` and
`results/reports/federated_results.md`. No adaptive client selection,
staleness-aware aggregation, or DRL happens in this phase.

## Phase 7 summary (non-IID Federated Learning experiments)

Phase 7 isolates client data heterogeneity as the SOLE experimental
variable: five fresh 8-client Dirichlet partitions (alpha=0.1, 0.5,
1.0, 5.0, 10.0) of the same real training data, with the exact same
FedAvg pipeline, model, rounds (10), local epochs (1), and all other
hyperparameters held fixed across every alpha (`configs/non_iid.yaml`,
`scripts/run_non_iid_experiments.py`). The original Phase 4/6 partition
(`data/partitions/`, alpha=0.5) was never touched — every alpha here,
including a fresh 0.5, has its own partition under
`experiments/non_iid/alpha_<X>/`. Measured (not assumed) heterogeneity
fell monotonically as alpha increased: mean pairwise JS distance
**0.6892** (alpha=0.1) down to **0.1452** (alpha=10.0). The **client
fairness gap** (best-served minus worst-served satellite, each
evaluated on its own local data) shrank monotonically and
substantially at every step, from **0.2623** (alpha=0.1) to **0.0064**
(alpha=10.0), a ~41x difference. KDDTest+ F1, in contrast, was
**non-monotonic**: it rose from 0.7409 (alpha=0.1) to a peak of
**0.7640 at alpha=5.0**, then fell back to 0.7541 at alpha=10.0 —
reported exactly as measured, not smoothed over; with one run per
alpha this could be genuine saturation or run-to-run noise, and more
trials would be needed to distinguish the two. At alpha=0.1 several
clients had zero local examples of rare attack categories (r2l/u2r)
and the partition failed its own fairness constraints — reported
honestly, not hidden. The freshly-generated alpha=0.5 run reproduced
Phase 6's exact numbers (same seed, alpha, constraints), a useful
cross-phase reproducibility check. Full
detail in `docs/project-progress/08-phase-7-non-iid-federated-learning.md`
and `results/reports/non_iid_results.md`. No adaptive client selection
or DRL happens in this phase — this is a measurement study, not an
intervention.

## Phase 8 summary (rule-based adaptive Federated Learning)

**Scope reminder:** this is the RULE-BASED (deterministic, non-RL) half
of adaptive FL — no reward signal, no learned parameters, no policy
network. DRL-based adaptation is a separate, later phase (this
tracker's own Phase 9 skeleton predates a finer session-level split
into "rule-based" then "DRL-based" adaptive FL; both fall under this
tracker's Phase 8 "Adaptive FL" heading above).

Phase 8 replaces Phase 6's sample-count-only FedAvg weighting with a
transparent client score — `w_perf*performance + w_data*data +
w_resource*resource + w_fair*fairness`, each component min-max
normalized across clients, weights fixed BEFORE any KDDTest+
evaluation (`src/federated/adaptive.py`, `configs/adaptive_fl.yaml`).
All 8 clients still participate every round (weighting, not selection
— selection risks silently dropping a rare-category client for a whole
round). Built entirely on top of Phase 6's unmodified client/server/
FedAvg code (`weighted_average` added to `fedavg.py`,
`aggregate_with_weights` added to `server.py`, both purely additive —
Phase 6's own `federated_average`/`aggregate` behavior is byte-for-byte
unchanged, confirmed by the pre-existing Phase 6 tests still passing).

Four rule configs were run on Phase 6's EXACT partition
(`data/partitions/`, not a fresh one): `performance_only`,
`resource_only`, `data_only` (each isolating one signal), and
`combined` (performance 0.40 / data 0.25 / resource 0.15 / fairness
0.20). Real, measured, slightly counterintuitive result reported
honestly: **`resource_only` reached the highest KDDTest+ F1 (0.7547)**,
beating Phase 6's 0.7404 by the widest margin; **`performance_only`
(not the fairness-weighted `combined` rule) reached the smallest
client-fairness gap (0.1245)** — `combined` was only second-best on
fairness (0.1496). This was not adjusted or re-run to look tidier. Full
detail in `docs/project-progress/09-phase-8-rule-based-adaptive-fl.md`
and `results/reports/adaptive_fl_results.md`. No DRL, no learned
weights, in this phase.

## What's explicitly NOT done yet (as of Phase 8)

- No DRL/DQN/reinforcement learning of any kind — Phase 8's adaptive
  weights are fixed, hand-specified numbers, not learned.
- No staleness-aware aggregation, FedProx, personalization, secure
  aggregation, or differential privacy.
- No hard client SELECTION (as opposed to weighting) was
  experimentally benchmarked, though a selection utility exists and is
  tested (`src/federated/adaptive.py:select_top_k_clients`).
- The OrbitShield website is not connected to any real model or
  simulation data — it still shows "Awaiting live data."
- No spatial/geographic/orbital realism — the satellite simulation
  creates non-IID label distributions and simulated resource
  heterogeneity, not real orbital mechanics or inter-satellite links.
- Simulated per-client resource metadata (bandwidth, latency, compute,
  availability, connectivity) now DOES influence aggregation (Phase 8's
  `resource` signal) — but only via a fixed rule, not a learned policy.
- Temporal context (Phase 3, last-record-label form) has not yet been
  shown to improve on the baseline — a real, useful finding for guiding
  later phases, not a blocker.

These are all planned for later phases and are intentionally out of
scope until their turn.
