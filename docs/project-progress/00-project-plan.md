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
Phase 9B — DRL (DQN) Adaptive FL       COMPLETE (see caveat below) — subsumes tracker Phases 10-12 below
Phase 9  — Staleness-Aware FL          COMPLETE (see caveat below) — combined with Resource-Aware Adaptation, Phase 13
Phase 10 — DRL Environment             COMPLETE — see Phase 9B
Phase 11 — DQN Controller              COMPLETE — see Phase 9B
Phase 12 — DRL Adaptive FL             COMPLETE — see Phase 9B
Phase 13 — Resource-Aware Adaptation   COMPLETE — see Phase 9 above
Phase 14 — Full Evaluation             COMPLETE (see caveat below)
Phase 15 — Ablation Study              COMPLETE (see caveat below)
Phase 16 — OrbitShield Integration     COMPLETE (see caveat below)
Phase 17 — Final System & Documentation COMPLETE (see caveat below) — merge to project-development awaits explicit review
```

*Numbering note: Phase 9B (DQN adaptive FL) was executed as one combined
design+pilot+full-experiment track under this project's own
session-level numbering, and its scope covers what the phase list above
originally split into three rows (9/10/11 — DRL environment, DQN
controller, DRL-driven adaptive FL). Nothing was skipped — see
`docs/project-progress/11-phase-9b-drl-adaptive-federated-learning.md`
and `docs/project-progress/12-phase-9b-drl-pilot.md` for the full
implementation record, and `results/reports/drl_adaptive_fl_results.md`
for the real, measured result (KDDTest+ F1 0.7425, ROC-AUC 0.8361 — did
not beat Phase 8's best rule-based result, an honest negative finding).
A separate, exploratory ESA-ADB dataset track was also investigated in
parallel (real ESA spacecraft telemetry; audited, piloted, one real
FedAvg baseline run under a masked federated representation, weak
result — ROC-AUC 0.4855) and is now parked, not part of this numbered
roadmap — see `docs/research/experiments/esa-adb/README.md`.*

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
| `12-phase-9b-drl-pilot.md` | DQN pilot implementation (state/action/reward/agent mechanics, synthetic data only, no pretraining run) |
| `11-phase-9b-drl-adaptive-federated-learning.md` | Full DQN adaptive FL design + implementation + real 750-round experiment; KDDTest+ F1 0.7425, ROC-AUC 0.8361 — did not beat Phase 8's best rule-based result (`resource_only`, F1 0.7547), an honest negative finding |
| `13-stin-feasibility-audit.md`, `14-smap-msl-track-b-audit.md` | Dataset research: STIN audited and rejected; SMAP/MSL audited as a Track B candidate (not adopted as the active roadmap dataset) |
| `15-phase-9-staleness-resource-fl.md` | Staleness- and resource-aware FL: a client's simulated Phase 4 availability genuinely gates round-to-round sync (real, seeded staleness), and 3 staleness-discount rules are ablated; the staleness-blind control (`no_correction`) scored best (KDDTest+ F1 0.7371) — an honest negative finding for the two staleness-aware rules tried |
| `16-full-evaluation.md` | Full cross-phase evaluation (no new training): every method's real KDDTest+ number in one ranked table; Phase 8's `resource_only` remains the single best aggregation mechanism, beating both the learned DQN (Phase 9B) and staleness-aware discounting (Phase 9) |
| `17-ablation-studies.md` | Two real, motivated sweeps: resource-weight (confirms w_resource=1.0 is a genuine peak, not a coarse-grid artifact) and staleness decay-rate (confirms no tested rate beats the staleness-blind control) — both sweeps exactly reproduce their Phase 8/9 reference points as a built-in honesty check |
| `18-orbitshield-integration.md` | Read-only FastAPI + React integration: real experiment/model registry, federated overview/rounds/adaptive-control, and the Phase 9B DQN controller/decisions now render on the actual OrbitShield website, verified with a real headless-browser pass against the running dev stack — no training happens through the API |
| `19-final-report.md` | Capstone document tying every phase (0-16) together for faculty/viva review — introduces no new experiment, cites every number back to its measuring phase |

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

## Phase 9B summary (DRL/DQN adaptive Federated Learning)

Phase 9B implements and runs a **DQN controller** that chooses, each
federated round, among the same 9 discrete weighting profiles Phase 8
already defined (`performance_only`, `resource_only`, `data_only`,
`combined`, `uniform`, and four blends) — the DQN decides *which*
already-validated rule to apply this round, not a new weighting
mechanism. State: 44-dim, built entirely from signals Phase 8 already
computes (per-client performance/data/resource/fairness, validation-F1
trend, previous action, round progress). Reward:
`ΔF1_t − λ_fair·perf_spread_t − λ_stab·1[action changed]`, built only
from global-validation signals — KDDTest+ is never read by it, verified
by AST-based privacy tests. Trained via 750 real FL rounds (50 episodes
× 15 rounds) against the real Phase 6 partition (single fixed
partition/seed across all episodes — a disclosed deviation from the
original multi-partition design), then evaluated with one frozen-policy
(ε=0) 10-round run, round 10 selected by validation loss alone.

**Real, measured, honestly-reported result:** KDDTest+ **F1 0.7425,
ROC-AUC 0.8361** — a narrow improvement over Phase 6 FedAvg (F1 0.7404)
and Phase 8's `data_only` (0.7397), roughly matching `combined`
(0.7439), but **did not beat Phase 8's best F1** (`resource_only`,
0.7547) **or best fairness gap** (`performance_only`, 0.1245; DQN's own
gap was 0.1833, worse than `combined` and `performance_only`). Episode
reward trended upward (−0.357 → +0.133 mean of last 10 episodes) and DQN
loss fell to ~0.0001, but the frozen greedy policy's action pattern
(9/10 rounds `data_only`) diverges from the late-pretraining exploitation
pattern (62.7% `combined`) — an unresolved inconsistency, reported as a
finding, not smoothed over. Full detail in
`docs/project-progress/11-phase-9b-drl-adaptive-federated-learning.md`
and `results/reports/drl_adaptive_fl_results.md`.

Phase 9B was reviewed and approved and is now marked **COMPLETE**. It
subsumes the phase list's original separate rows for "DRL environment"
(10), "DQN controller" (11), and "DRL-driven adaptive FL" (12).

## Phase 9 summary (staleness- and resource-aware Federated Learning)

Phase 9 is the first phase to let Phase 4's simulated per-client
`availability_probability` actually gate round-to-round PARTICIPATION —
every prior phase (6, 7, 8, 9B) had all 8 clients fetch and train
against the exact current global model every round, so no run before
this one ever contained real staleness. Each round, each client's
sync/no-sync outcome is a seeded coin flip weighted by its own real
availability; a client that doesn't sync keeps training against
whatever global state it last synced to, producing a real, measured
staleness (not simulated after the fact). Every client still trains and
submits an update every round (weighting, not selection — the same
philosophy Phase 8 already committed to). Three staleness-discount
rules were ablated on Phase 6's exact partition, sharing the identical
seeded sync/no-sync pattern so only the discount response differs:
`no_correction` (w=1, staleness measured but ignored), `linear_decay`
(w=1/(1+staleness)), `exponential_decay` (w=0.5^staleness). Combines,
in this project's own numbering, what the master tracker originally
listed as two separate rows — "Staleness-Aware FL" (9) and
"Resource-Aware Adaptation" (13) — since Phase 4's availability field is
the same underlying mechanism driving both.

**Real, measured, honestly-reported result:** the staleness-blind
control, `no_correction`, scored the highest KDDTest+ F1 (**0.7371**) of
the three — BOTH staleness-aware discount rules scored slightly lower
(`exponential_decay` 0.7360, `linear_decay` 0.7354). None of the three
beat Phase 6 FedAvg (0.7404) or Phase 8's best rule (`resource_only`,
0.7547). Real staleness in this run topped out at 3 rounds (measured,
not assumed). A plausible, disclosed explanation: with only 8 clients,
10 rounds, and a small maximum real staleness, there wasn't enough
staleness in this run for correcting it to clearly pay off, while the
discount itself slightly reduced otherwise-useful updates' weight — a
hypothesis, not a proven cause; not re-run to chase a better number.
Fairness gaps (0.1964-0.1994 across all three configs) were also
roughly unaffected by which discount rule was used. Full detail in
`docs/project-progress/15-phase-9-staleness-resource-fl.md` and
`results/reports/staleness_fl_results.md`.

Phase 9 was reviewed and approved and is now marked **COMPLETE**. It
subsumes the phase list's originally separate "Resource-Aware
Adaptation" row (13).

## Phase 14 summary (full cross-phase evaluation)

Phase 14 runs no new experiment — it reads the already-committed final
KDDTest+ number from every prior phase's own report (2, 3, 6, 7, 8, 9B,
9; 14 methods total) into one master ranking, cited by source so every
number can be checked against where it was originally measured
(`scripts/generate_full_evaluation_report.py`,
`results/reports/full_evaluation.md`).

**Headline, honest finding:** holding the data partition fixed and
varying only the aggregation mechanism, Phase 8's `resource_only` rule
(KDDTest+ F1 0.7547) remains the single best result across every
mechanism tried, including the learned DQN controller (Phase 9B,
0.7425) and both staleness-aware discount rules (Phase 9, best 0.7371).
A simple, hand-specified rule beats every more sophisticated mechanism
built so far — reported as the honest current state of the evidence,
not smoothed into a tidier narrative. The performance/fairness tension
first observed in Phase 8 (the most-accurate method is never the
fairest one) holds across every later phase that measured both. Full
detail in `docs/project-progress/16-full-evaluation.md`.

Phase 14 was reviewed and approved and is now marked **COMPLETE**.

## Phase 15 summary (ablation studies)

Phase 15 runs two real, cheap, narrowly-motivated sweeps — not a broad
hyperparameter search — each targeting an open question a prior phase's
own report explicitly left unresolved: (1) a resource-weight sweep
(`w_resource` from 0.0 to 1.0, five points) asking whether Phase 8's
`resource_only` win is a genuine peak or a coarse-4-option artifact; (2)
a staleness exponential-decay-rate sweep (0.1 to 0.9, five points)
asking whether Phase 9's untuned `decay_rate=0.5` default was the
reason staleness-aware discounting didn't help. Both reuse Phase 8/9's
UNMODIFIED training code — only the swept parameter changes — and each
includes one point that exactly reproduces an already-committed result
(`w_resource=1.0` = Phase 8's `resource_only`; `decay_rate=0.5` = Phase
9's `exponential_decay`), both matching bit-for-bit (F1 0.7547 and
0.7360 respectively), confirming the sweep harness is faithful.

**Real, measured findings:** the resource-weight sweep climbs
monotonically from 0.7426 (w_resource=0.0) to 0.7547 (w_resource=1.0) —
Phase 8's `resource_only` result is a genuine optimum, not a lucky
coarse-grid artifact. The staleness decay-rate sweep's best point
(decay_rate=0.90, F1 0.7369) still falls short of Phase 9's
staleness-blind control (0.7371) — no decay_rate tested changes Phase
9's negative finding. Both results make the earlier phases' conclusions
MORE trustworthy, not less. Full detail in
`docs/project-progress/17-ablation-studies.md` and
`results/reports/ablation_study.md`.

Phase 15 was reviewed and approved and is now marked **COMPLETE**.

## Phase 16 summary (OrbitShield integration)

Phase 16 wires the real, already-committed results from every prior
phase into the actual OrbitShield web platform — read-only, no
training happens through the API. New backend
(`backend/app/api/{experiments,models,federated,drl}.py`,
`backend/app/services/results_registry.py`) reads the same committed
JSON artifacts every phase's own scripts wrote (round histories, the
Phase 9B DQN's pretraining/final-run logs) and Phase 14's fixed
reference numbers into a small set of REST endpoints. New frontend
pages (`frontend/src/pages/{experiments,federated,drl,system}/`)
replace the placeholder shell for: Experiments, Model Registry,
Experiment/Model Details, Analytics, Federated Overview, Training
Rounds, Adaptive Control, DRL Controller, Decision History, and System
Health. Routes with no real backing data (satellite network, threats,
per-client fairness, activity, settings) deliberately stay on
"Awaiting live data" — not filled with placeholder numbers.

**Verification, not just a build check:** both servers were actually
started and driven with headless Chromium (Playwright) against the
running dev stack, navigating and screenshotting every newly-wired
route plus a detail page, confirming zero console errors and the real
KDDTest+ numbers (e.g. `resource_only`'s 0.7547) genuinely render — not
just that the code compiles. The still-placeholder routes were
re-checked too, confirming nothing regressed. 18 new backend tests
assert against these same real numbers (419/419 full suite passing);
`npm run build`/`npm run lint` pass clean. Full detail in
`docs/project-progress/18-orbitshield-integration.md`.

Phase 16 was reviewed and approved and is now marked **COMPLETE**.

## Phase 17 summary (final system & documentation)

Phase 17 is a documentation and reconciliation pass, not new
implementation: `docs/project-progress/19-final-report.md` was written
as the single capstone document tying every phase (0 through 16)
together for faculty/viva review, citing every real number back to the
phase report that actually measured it — no new experiment was run to
produce it. `README.md` and this tracker were reconciled to their true
final state across every phase's completion (a running, incremental
process throughout this engagement, not a one-time fix at the end).

**What "final" means here, precisely:** every phase in the numbered
roadmap (0-16) is real, measured, and committed on branch
`project-development-7-staleness-fl`. The one deliberately withheld
step is merging that branch history back into `project-development` —
a shared-branch action this project's own standing rule requires
explicit review for, not something to do automatically at the end of a
documentation pass.

Phase 17 was reviewed and approved and is now marked **COMPLETE**
(merge pending explicit go-ahead).

## ESA-ADB (parked research track, not part of the numbered roadmap)

A separate, exploratory investigation of the ESA-ADB dataset (real ESA
spacecraft telemetry, 3 missions) ran in parallel on its own branch:
acquisition + structural/leakage audit, time-window selection, a full
data/FL pipeline (8 real-metadata-derived clients), two centralized
baselines (single-channel failed, ROC-AUC 0.4734; 58-channel
multivariate succeeded, ROC-AUC 0.9085), a federated architecture design
for heterogeneous per-client channel ownership (Option A: masked global
input, zero FedAvg code changes required), and one real FedAvg baseline
run under that design (ROC-AUC 0.4855 — a genuine, honestly-reported
weak result). Parked per explicit decision rather than debugged further,
so the primary NSL-KDD roadmap below could resume. Full record:
`docs/research/experiments/esa-adb/README.md`.

## What's explicitly NOT done yet (as of Phase 9)

- No FedProx, personalization, secure aggregation, or differential
  privacy — not part of this project's approved roadmap.
- No hard client SELECTION (as opposed to weighting) was
  experimentally benchmarked, though a selection utility exists and is
  tested (`src/federated/adaptive.py:select_top_k_clients`).
- The OrbitShield website is not connected to any real model or
  simulation data — it still shows "Awaiting live data" (Phase 16).
- No spatial/geographic/orbital realism — the satellite simulation
  creates non-IID label distributions and simulated resource
  heterogeneity, not real orbital mechanics or inter-satellite links.
- Phase 9's staleness simulation is real but single-seed and
  small-scale (max real staleness observed: 3 rounds across 10 total
  rounds) — a longer run or harsher availability distribution was
  identified as the natural next experiment to distinguish "the
  technique doesn't help" from "there wasn't enough staleness to
  matter," but not attempted, per the standing rule against re-running
  to chase a better number.
- Temporal context (Phase 3, last-record-label form) has not yet been
  shown to improve on the baseline — a real, useful finding for guiding
  later phases, not a blocker.
- No full cross-phase evaluation report or ablation study yet (Phases
  14/15) — comparisons so far are pairwise, phase-by-phase.

These are all planned for later phases and are intentionally out of
scope until their turn.
