# Deep Reinforcement Learning-Driven Adaptive Federated Framework for Satellite Network Anomaly Detection

**Platform name:** OrbitShield

B.Tech capstone project implementing a practical adaptive framework for
satellite network anomaly detection, combining spatio-temporal anomaly
detection, federated learning across non-IID satellite clients, and a
DRL (DQN) controller that adapts client selection, aggregation, and
resource decisions under staleness and connectivity constraints.

This is not presented as fully novel research — non-IID federated
learning, staleness-aware aggregation, and DRL-driven adaptive FL are
active research areas. The contribution here is a practical, integrated
implementation and evaluation of these techniques applied to simulated
satellite network traffic.

## OrbitShield

OrbitShield is the operational web platform used to visualize and
interact with the system: satellite network state, detected threats,
federated learning rounds, DRL controller decisions, model registry,
and experiment results. It is built as a real command-center style
application, not a documentation site — architecture, methodology, and
research background live in the project report, not as app pages.

## Project status

```
Phase 0  — Project Foundation          COMPLETE
Phase 1  — Dataset Acquisition         COMPLETE
Phase 2  — Data Preprocessing          COMPLETE
Phase 3  — Baseline Model              COMPLETE
Phase 4  — Temporal Model (GRU+Attn)   COMPLETE
Phase 5  — Satellite Simulation        COMPLETE
Phase 6  — Local Satellite Training    COMPLETE
Phase 7  — Federated Learning (FedAvg) COMPLETE
Phase 8  — Non-IID FL Experiments      COMPLETE
Phase 9  — Rule-Based Adaptive FL      COMPLETE
Phase 10+ — (see Development phases)   NOT STARTED
```

Rule-based (non-reinforcement-learned) adaptive Federated Learning is
implemented — a transparent, fixed client-scoring rule now determines
each satellite's aggregation weight, replacing Phase 6/7's sample-
count-only weighting. DRL-driven adaptive aggregation has not yet been
implemented.

See `docs/project-progress/` for a plain-language, viva-ready write-up
of every completed phase (`00-project-plan.md` is the master tracker,
including the full history).

Phase 1 produced a real, reproducible NSL-KDD dataset pipeline (load →
clean → label → split → encode/scale → processed dataset + metadata).
Phase 2 trained a baseline MLP anomaly detector on that data — a simple
feed-forward network (121 → 128 → 64 → 1), evaluated once on the
held-out KDDTest+ set: **78.3% accuracy, 92.8% precision, 67.1% recall,
77.9% F1, 89.7% ROC-AUC** (see `results/reports/baseline_results.md`).
Phase 3 built a temporal GRU + attention model reading sequences instead
of single records. A first attempt (label a sequence "anomaly" if ANY
record in it was) was caught, investigated, and **rejected**: it made
~100% of sequences carry the anomaly label given NSL-KDD's per-record
attack rate, so its 100% test score was meaningless — preserved as a
documented failed experiment, not hidden. A corrected attempt (label a
sequence by its *last* record, selected by measuring four candidate
strategies against real data) produced a trustworthy, non-degenerate
result: **75.8% accuracy, 92.2% precision, 64.0% recall, 75.6% F1, 89.2%
ROC-AUC** on KDDTest+ — honestly slightly below the baseline. See
`results/reports/temporal_results.md` for both experiments in full.
Phase 4 (this project's tracker numbering — see Development phases
table below for the 20-phase numbering) built a **simulation of a
multi-satellite learning environment using the real NSL-KDD dataset**:
8 simulated clients, non-IID partitioned by Dirichlet distribution
(measured average pairwise Jensen-Shannon distance between clients'
category distributions: **0.527**, 0=identical/1=maximally different),
each with simulated (clearly labeled as such) bandwidth/latency/compute/
availability/connectivity — see
`results/reports/satellite_simulation_report.md`. Phase 5 (this
project's tracker numbering; row 7 in the Development phases table
below) then trained an independent local model on each of the 8
satellites using ONLY that satellite's own local data — no
communication or aggregation between clients. Measured global-
validation F1 ranged from **92.4%** (`SAT-01`, whose local data is
99.1% anomalous) to **99.2%** (`SAT-02`, the largest and most balanced
local partition) — see
`results/reports/local_training_results.md`. Phase 6 (this project's
tracker numbering; row 8 in the Development phases table below) then
implemented the first real Federated Learning: standard synchronous
FedAvg across all 8 clients, sample-count-weighted averaging, 10
communication rounds, with a tested privacy boundary (the server never
reads client data — only model parameters and sample counts). Global
validation F1 improved every round to **98.8%** (round 10, selected by
validation loss alone); the final, one-time KDDTest+ result was **F1
74.0%**, honestly slightly below the Phase 2 centralized baseline's
77.9% — a real, measured result attributed to this baseline's
deliberately small round/epoch budget — see
`results/reports/federated_results.md`. Phase 7 (this project's tracker
numbering; row 9 in the Development phases table below) then ran a
controlled study isolating client data heterogeneity as the sole
variable: the same FedAvg pipeline was re-run on five fresh Dirichlet
partitions (alpha = 0.1, 0.5, 1.0, 5.0, 10.0), with everything else
held fixed. Measured heterogeneity fell monotonically as alpha
increased (mean pairwise JS distance **0.6892** at alpha=0.1 down to
**0.1452** at alpha=10.0), and the **client fairness gap** (best-served
vs. worst-served satellite) shrank monotonically at every step, from
**0.2623** to **0.0064** (~41x). KDDTest+ F1 was **non-monotonic**: it
peaked at alpha=5.0 (0.7640) and was slightly lower at alpha=10.0
(0.7541) despite alpha=10.0 being closer to IID — reported exactly as
measured, not smoothed over — see
`results/reports/non_iid_results.md`. Phase 8 (this project's tracker
numbering; row 10 in the Development phases table below) then
implemented **rule-based (NOT reinforcement-learned) adaptive FedAvg**:
a transparent client score (`performance + data + resource + fairness`,
each min-max normalized, fixed weights chosen before any test
evaluation) replaces Phase 6/7's sample-count-only weighting, built
additively on Phase 6's unmodified client/server/FedAvg code. Four rule
configs were run on Phase 6's exact partition; the honest, measured
result: `resource_only` reached the highest KDDTest+ F1 (**0.7547**,
beating Phase 6's 0.7404), while `performance_only` — not the
fairness-weighted `combined` rule — reached the smallest client
fairness gap (**0.1245** vs. `combined`'s 0.1496), reported exactly as
measured rather than smoothed over — see
`results/reports/adaptive_fl_results.md`. No DRL/learned weights, and
none of the models are connected to the frontend. The frontend
dashboard intentionally shows "Awaiting live data" states rather than
fabricated metrics.

## Development phases

| Phase | Scope |
|---|---|
| 0 | Repository + project foundation ✅ complete |
| 1 | Dataset acquisition and analysis ✅ complete (preprocessing pipeline also implemented — see note below) |
| 2 | Data preprocessing ✅ substantially complete as part of Phase 1 (see note below) |
| 3 | Baseline anomaly detection ✅ complete |
| 4 | Temporal sequence construction ✅ complete, including a corrected relabel after an initial degenerate attempt (see note below) |
| 5 | Spatio-temporal anomaly model ⚠️ TEMPORAL component/foundation complete (did not beat baseline — honest result); genuine SPATIAL/satellite modeling is not implemented and requires row 6 below (see note below) |
| 6 | Satellite client simulation ✅ complete — 8 simulated clients, measured non-IID partition, simulated resource conditions generated (see note below) |
| 7 | Local satellite training ✅ complete — 8 independent local models trained, no communication/aggregation (see note below) |
| 8 | Standard FedAvg ✅ complete — 10 rounds, all 8 clients, sample-count-weighted averaging, tested privacy boundary (see note below) |
| 9 | Non-IID experiments ✅ complete — controlled Dirichlet-alpha sweep isolating heterogeneity as the sole variable (see note below) |
| 10 | Adaptive federated learning ⏳ in progress, awaiting review — rule-based (non-RL) client scoring/weighting; DRL-based adaptation still to come (see note below) |
| 11 | Staleness-aware FL *(next, after Phase 10 review)* |
| 12 | DRL environment |
| 13 | DQN controller |
| 14 | DRL-driven adaptive FL |
| 15 | Resource-aware adaptation |
| 16 | Optional robustness/security extension |
| 17 | Full evaluation |
| 18 | Ablation studies |
| 19 | OrbitShield professional web platform |
| 20 | Final integration, documentation and presentation |

*Note: Phases 1 and 2 were executed together in a single work session
(dataset selection/acquisition/analysis plus a full clean → label →
split → encode/scale pipeline), since building a trustworthy processed
dataset required all of it. Phase 3 (baseline anomaly detection) added
the first real model — a simple MLP. Phases 4 and 5 were also executed
together as "Phase 3"/"Phase 4" of this project's own session-by-session
tracker (see `docs/project-progress/00-project-plan.md`): sequence
construction plus a GRU+attention model that reads those sequences.
**This establishes only the TEMPORAL component/foundation of the
eventual spatio-temporal model** — it does not implement genuine spatial
or multi-satellite modeling of any kind. An initial sequence-labeling
choice (ANY-anomaly) was found to produce a degenerate, near-single-class
task and was rejected rather than reported; a corrected, evidence-selected
labeling strategy (LAST-record) produced a trustworthy but
honestly-below-baseline result — see `results/reports/temporal_results.md`
for both experiments.
The simulated satellite-client/network environment — row 6
("Satellite client simulation") in this table's numbering, and Phase 4
in `docs/project-progress/00-project-plan.md`'s numbering (the two
files use different phase-numbering granularity — see "Relationship to
the original 20-phase plan" in that tracker) — has now been built: 8
simulated clients, a measured (not assumed) non-IID partition of the
real training data, and simulated per-client resource conditions. This
is what finally gives meaning to the SPATIAL half of "spatio-temporal."
It is a **simulation using the real NSL-KDD dataset**, not real
satellite telemetry — see `docs/project-progress/05-phase-4-satellite-simulation.md`
for the full, explicit real-vs-simulated breakdown.
Row 7 ("Local satellite training") — Phase 5 in
`docs/project-progress/00-project-plan.md`'s numbering — has now been
built: each of the 8 satellites trains an independent copy of the
Phase 2 baseline MLP on ONLY its own local data, starting from
identical shared initial weights, with no communication or aggregation
between clients. All 8 are evaluated on the same global validation set
for a fair comparison; KDDTest+ remains untouched. See
`docs/project-progress/06-phase-5-local-training.md` and
`results/reports/local_training_results.md`.
Row 8 ("Standard FedAvg") — Phase 6 in
`docs/project-progress/00-project-plan.md`'s numbering — has now been
built: all 8 satellites train together via standard synchronous FedAvg
(sample-count-weighted averaging), with an explicit, tested privacy
boundary between client-side local training and server-side
aggregation (the server never touches raw data). 10 communication
rounds; global validation F1 reached 98.8% (round 10); final KDDTest+
F1 was 74.0%, honestly slightly below the Phase 2 centralized
baseline's 77.9%. See
`docs/project-progress/07-phase-6-federated-learning.md` and
`results/reports/federated_results.md`.
Row 9 ("Non-IID experiments") — Phase 7 in
`docs/project-progress/00-project-plan.md`'s numbering — has now been
built: a controlled study re-running the SAME FedAvg pipeline across
five fresh Dirichlet partitions (alpha = 0.1, 0.5, 1.0, 5.0, 10.0) of
the same real training data, holding every other factor (model,
clients, rounds, local epochs, batch size, learning rate, optimizer,
loss, FedAvg weighting, validation set, test set, seeds) fixed so
alpha is the only variable. Measured heterogeneity (mean pairwise JS
distance) fell monotonically from 0.6892 (alpha=0.1) to 0.1452
(alpha=10.0), and the client fairness gap (best-served vs.
worst-served satellite) shrank monotonically at every step, from
0.2623 to 0.0064 (~41x). KDDTest+ F1 was non-monotonic — it peaked at
alpha=5.0 (0.7640) and dipped slightly at alpha=10.0 (0.7541), reported
exactly as measured. The original Phase 4/6 partition
(`data/partitions/`) was never touched; the freshly-generated alpha=0.5
run reproduced Phase 6's exact numbers, a useful cross-phase
reproducibility check.
See `docs/project-progress/08-phase-7-non-iid-federated-learning.md`
and `results/reports/non_iid_results.md`.
Row 10 ("Adaptive federated learning") — Phase 8 in
`docs/project-progress/00-project-plan.md`'s numbering — has now been
built, RULE-BASED ONLY (no reinforcement learning): a transparent
client score (`w_perf*performance + w_data*data + w_resource*resource
+ w_fair*fairness`, all min-max normalized, weights fixed before any
KDDTest+ evaluation) replaces Phase 6/7's sample-count-only weighting.
Built additively on Phase 6's unmodified client/server/FedAvg code —
`weighted_average`/`aggregate_with_weights` were ADDED, not changed,
and Phase 6's own `federated_average`/`aggregate` behavior is
unchanged (its tests still pass byte-for-byte). All 8 clients still
participate every round (weighting, not selection). Four rule configs
were run on Phase 6's EXACT partition: `resource_only` reached the
highest KDDTest+ F1 (0.7547, beating Phase 6's 0.7404), while
`performance_only` — not the fairness-weighted `combined` rule —
reached the smallest client fairness gap (0.1245 vs. `combined`'s
0.1496), an honest, counterintuitive result reported exactly as
measured. See `docs/project-progress/09-phase-8-rule-based-adaptive-fl.md`
and `results/reports/adaptive_fl_results.md`. No DRL/learned weights,
no staleness-aware aggregation, or other advanced FL components exist
yet, and remain out of scope until those later milestones are built.*

## Dataset

**Selected: NSL-KDD** (a terrestrial network intrusion-detection
dataset — not satellite-specific; see
`docs/datasets/dataset_selection.md` for the full reasoning and
`docs/datasets/feature_decisions.md` for the per-feature leakage
analysis). Real dataset statistics are in
`results/reports/dataset_quality.md`.

To reproduce the dataset and pipeline locally:

```bash
# 1. Download the raw dataset (~22.5 MB, not committed to this repo)
mkdir -p data/raw
curl -o data/raw/KDDTrain+.txt "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt"
curl -o data/raw/KDDTest+.txt "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt"

# 2. Inspect it
python scripts/inspect_dataset.py

# 3. Run the full preprocessing pipeline (writes data/processed/, data/interim/,
#    and results/models/preprocessing/ artifacts — all git-ignored/regenerable)
python -m src.preprocessing.pipeline

# 4. Regenerate the dataset quality report + plots (committed, small)
python scripts/generate_dataset_report.py
```

## Baseline model

A simple feed-forward MLP (121 → 128 → 64 → 1) trained on the Phase 1
processed data as a reference point for later spatio-temporal/federated/
DRL comparisons. Full results, hyperparameters, and confusion matrix in
`results/reports/baseline_results.md`; plain-language write-up in
`docs/project-progress/03-phase-2-baseline.md`.

```bash
# Requires the processed dataset from the steps above, plus PyTorch:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 1. Train (selects the best checkpoint by validation loss)
python scripts/train_baseline.py

# 2. Evaluate once, finally, on KDDTest+ (writes plots + results report)
python scripts/evaluate_baseline.py
```

## Temporal model

A GRU + learned temporal attention model that reads sequences of
records instead of single ones — the temporal half of the eventual
spatio-temporal model (no spatial/multi-satellite component yet).

An initial sequence-labeling choice ("anomaly if ANY record in the
sequence is anomalous") was found to make ~100% of sequences carry the
anomaly label on NSL-KDD, making its 100% test score meaningless — this
was caught, investigated, and rejected rather than reported (preserved
in `experiments/temporal/initial_any_anomaly_temporal_results_ARCHIVE.md`
for research integrity). Four candidate labeling strategies were then
measured against real data (`results/reports/sequence_labeling_analysis.md`)
and "label by the sequence's last record" was selected on that
evidence. The corrected model scores **75.6% F1 / 75.8% accuracy / 89.2%
ROC-AUC** on KDDTest+ — a real, non-degenerate result, honestly slightly
below the Phase 2 baseline. Full detail (both experiments) in
`results/reports/temporal_results.md`; plain-language write-up in
`docs/project-progress/04-phase-3-spatio-temporal.md`.

```bash
# Requires the processed dataset + PyTorch (see Baseline model above)

# 1. (Optional) Investigate sequence-labeling strategies against real
#    data before training (writes results/reports/sequence_labeling_analysis.md)
python scripts/analyze_sequence_labeling.py

# 2. Train across sequence-length candidates [8, 16, 32], selecting
#    the best by validation loss (writes results/models/temporal/)
python scripts/train_temporal.py

# 3. Evaluate once, finally, on KDDTest+ (writes plots + results report)
python scripts/evaluate_temporal.py
```

## Satellite simulation

A **simulation of a multi-satellite learning environment using the real
NSL-KDD dataset** — not real satellite telemetry. 8 simulated clients
(`SAT-01`..`SAT-08`) are built by Dirichlet-partitioning the real
training data by attack category (non-IID, measured — average pairwise
Jensen-Shannon distance between clients' distributions: 0.527), each
with simulated (explicitly labeled as such) bandwidth, latency, compute
capacity, availability, and connectivity. Only the training split is
partitioned; validation stays global and KDDTest+ is untouched. Full
detail in `results/reports/satellite_simulation_report.md` and
`data/partitions/README.md`; plain-language write-up in
`docs/project-progress/05-phase-4-satellite-simulation.md`. No
Federated Learning, DRL, or model training happens here — this builds
the client environment only.

```bash
# Requires the processed dataset (see Dataset above)

# 1. Partition the real training data across simulated satellite
#    clients + generate simulated resource conditions (writes
#    data/partitions/SAT-*/train.parquet, gitignored, + manifest.json
#    and satellite_metadata.json, committed)
python scripts/create_satellite_partitions.py

# 2. Measure non-IID quality + resource heterogeneity, generate plots
#    and the full report
python scripts/analyze_satellite_partitions.py
```

## Local satellite training

Each simulated satellite (`SAT-01`..`SAT-08`) trains an **independent**
copy of the Phase 2 baseline MLP architecture on ONLY its own local
partition — no communication or aggregation between clients. All 8
start from identical shared initial weights (seed=42) so outcome
differences reflect local (non-IID) data, not initialization. Every
client is evaluated on the same global validation set for a fair
cross-client comparison; KDDTest+ is untouched. Measured global-
validation F1 ranged **92.4%** (`SAT-01`) to **99.2%** (`SAT-02`) across
the 8 clients. Full detail in
`results/reports/local_training_results.md`; plain-language write-up in
`docs/project-progress/06-phase-5-local-training.md`. No Federated
Learning, DRL, or model aggregation happens here.

```bash
# Requires the satellite partitions (see Satellite simulation above)
# plus PyTorch (see Baseline model above)

# 1. Train all 8 clients independently (writes results/models/local/SAT-*/)
python scripts/train_local_models.py

# 2. Evaluate every client on the shared global validation set, generate
#    confusion matrices, training curves, and the full results report
python scripts/evaluate_local_models.py
```

## Federated learning (FedAvg baseline)

The first real Federated Learning experiment: all 8 simulated
satellites train the same Phase 2 MLP architecture together via
standard **synchronous FedAvg** — sample-count-weighted averaging,
`w_global = sum_k (n_k/N) * w_k` — with NO raw data ever sent to the
server (only model parameters + sample counts;
`src/federated/server.py` never imports pandas or reads a parquet
file, verified by a dedicated test). All 8 clients participate every
round; no adaptive selection yet. 10 communication rounds, 1 local
epoch/round; the best round was selected using global validation loss
alone (never KDDTest+). Final, one-time KDDTest+ result: **74.0% F1**
(honestly slightly below the Phase 2 centralized baseline's 77.9% — a
real result attributed to this baseline's deliberately small
round/epoch budget). Full detail in
`results/reports/federated_results.md`; plain-language write-up in
`docs/project-progress/07-phase-6-federated-learning.md`. No adaptive
client selection, staleness-aware aggregation, or DRL happens here.

```bash
# Requires the satellite partitions (see Satellite simulation above)
# plus PyTorch (see Baseline model above)

# 1. Run the federated training loop — all 8 clients, 10 rounds
#    (writes results/models/federated/, experiments/federated/)
python scripts/run_federated_training.py

# 2. Evaluate the selected best-validation-round global model ONCE on
#    KDDTest+, generate round-by-round plots and the full results report
python scripts/evaluate_federated_model.py
```

## Non-IID Federated Learning experiments

A controlled study of how client data heterogeneity affects standard
FedAvg. The SAME FedAvg pipeline (Phase 6) is re-run across five fresh
8-client Dirichlet partitions (alpha = 0.1, 0.5, 1.0, 5.0, 10.0) of the
same real training data — everything except alpha (model, clients,
rounds, local epochs, batch size, learning rate, optimizer, loss,
FedAvg weighting, validation set, test set, seeds) is held fixed, so
alpha is the sole experimental variable. The original Phase 4/6
partition (`data/partitions/`) is never touched — every alpha here,
including a fresh 0.5, gets its own partition under
`experiments/non_iid/`. Measured heterogeneity (mean pairwise JS
distance) fell monotonically from **0.6892** (alpha=0.1) to **0.1452**
(alpha=10.0), and the **client fairness gap** — the difference between
the best- and worst-served satellite, each evaluated on its own local
data — shrank monotonically at every step, roughly **41x** (0.2623 →
0.0064). Final KDDTest+ F1 was **non-monotonic**: it peaked at
alpha=5.0 (0.7640) and dipped slightly at alpha=10.0 (0.7541), reported
exactly as measured rather than smoothed over. Full detail in
`results/reports/non_iid_results.md`; plain-language write-up in
`docs/project-progress/08-phase-7-non-iid-federated-learning.md`. No
adaptive client selection or DRL happens here — this is a measurement
study.

```bash
# Requires the processed dataset (see Dataset above) plus PyTorch

# 1. Build a fresh partition + run FedAvg for every configured alpha
#    (writes experiments/non_iid/alpha_<X>/)
python scripts/run_non_iid_experiments.py

# 2. Evaluate every alpha's selected model on KDDTest+, compute client
#    fairness, generate plots and the full results report
python scripts/analyze_non_iid_results.py
```

## Rule-based adaptive Federated Learning

A transparent, DETERMINISTIC (explicitly NOT reinforcement-learned)
alternative to Phase 6's sample-count-only FedAvg weighting. Every
client's aggregation weight now comes from a documented rule:
`w_perf*performance + w_data*data + w_resource*resource +
w_fair*fairness` (all four signals min-max normalized across clients;
weights fixed before any KDDTest+ evaluation — see
`src/federated/adaptive.py` and `configs/adaptive_fl.yaml`). Built
additively on Phase 6's unmodified `FederatedClient`/`FederatedServer`
code (`weighted_average`/`aggregate_with_weights` were ADDED, not
changed). All 8 clients still participate every round — this changes
HOW MUCH each update counts, not whether it's sent. Four rule configs
were run on Phase 6's exact partition (`data/partitions/`): the honest,
measured result is that `resource_only` reached the highest KDDTest+ F1
(**0.7547**, beating Phase 6's 0.7404), while `performance_only` — not
the fairness-weighted `combined` rule — reached the smallest client
fairness gap (**0.1245** vs. `combined`'s 0.1496), reported exactly as
measured. Full detail in `results/reports/adaptive_fl_results.md`;
plain-language write-up in
`docs/project-progress/09-phase-8-rule-based-adaptive-fl.md`.

```bash
# Requires the Phase 4 partitions (see Satellite simulation above)
# plus PyTorch

# 1. Run all 4 rule configs (performance/resource/data-only, combined)
#    on Phase 6's exact partition (writes experiments/adaptive_fl/<rule>/)
python scripts/run_adaptive_fl_experiments.py

# 2. Evaluate each rule's selected model on KDDTest+, compute client
#    fairness, generate plots and the full results report
python scripts/evaluate_adaptive_fl_results.py
```

## Repository structure

```
configs/        Experiment/system configuration (YAML) — no hardcoded params
data/           raw / interim / processed / partitions (large files git-ignored)
src/            ML/FL/DRL source: data, preprocessing, models, training,
                federated, adaptive, drl, simulation, satellite,
                evaluation, utils
backend/        FastAPI application (API, services, schemas, core) + tests
frontend/       React + TypeScript + Vite application (OrbitShield UI)
experiments/    Experiment run configs/scripts, separate from src/
results/        Generated metrics, models, plots, logs, reports
                (mostly git-ignored/regenerable — e.g. model checkpoints;
                small reviewable deliverables like results/reports/*.md,
                results/plots/{dataset,baseline,temporal,satellite,local_training,federated,non_iid,adaptive_fl}/,
                results/models/local/*/{metadata,training_history}.json,
                results/models/federated/round_history.json,
                experiments/non_iid/alpha_*/{partition_metadata,run_config,
                round_history}.json, and experiments/adaptive_fl/<rule>/
                {run_config,round_history}.json are committed)
notebooks/      Exploratory analysis
scripts/        Standalone utility scripts
tests/          Tests for src/ (data, models, training, evaluation,
                simulation, federated, drl, integration)
docs/           Architecture, methodology, dataset and API documentation
```

## Running the backend

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python -m uvicorn backend.app.main:app --reload
```

The API will be available at `http://localhost:8000`, with a health
check at `http://localhost:8000/health`.

## Running the frontend

```bash
cd frontend
npm install
npm run dev
```

The app will be available at `http://localhost:5173`. It expects the
backend at `http://localhost:8000` by default (see `frontend`'s
`VITE_API_BASE_URL`, sourced from `.env.example`).

## Running tests

```bash
# Backend + data/preprocessing (uses a tiny synthetic fixture, not the real dataset)
.venv\Scripts\activate
pytest

# Frontend
cd frontend
npm run lint
npm run build
```

## Environment variables

Copy `.env.example` to `.env` and adjust as needed. No secrets are
committed to this repository.

## Data & result integrity

This project never fabricates experimental results. Any metric not yet
backed by a real experiment is shown as "Not available" / "Awaiting
experiment", and simulated data is always clearly labeled as such.
