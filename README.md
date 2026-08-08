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
Phase 5  — Satellite Simulation        IN PROGRESS
Phase 6+ — (see Development phases)    NOT STARTED
```

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
`results/reports/satellite_simulation_report.md`. No federated learning
or DRL logic has been implemented yet, and none of the models are
connected to the frontend. The frontend dashboard intentionally shows
"Awaiting live data" states rather than fabricated metrics.

## Development phases

| Phase | Scope |
|---|---|
| 0 | Repository + project foundation ✅ complete |
| 1 | Dataset acquisition and analysis ✅ complete (preprocessing pipeline also implemented — see note below) |
| 2 | Data preprocessing ✅ substantially complete as part of Phase 1 (see note below) |
| 3 | Baseline anomaly detection ✅ complete |
| 4 | Temporal sequence construction ✅ complete, including a corrected relabel after an initial degenerate attempt (see note below) |
| 5 | Spatio-temporal anomaly model ⚠️ TEMPORAL component/foundation complete (did not beat baseline — honest result); genuine SPATIAL/satellite modeling is not implemented and requires row 6 below (see note below) |
| 6 | Satellite client simulation ⏳ in progress, awaiting review — 8 simulated clients, measured non-IID partition, simulated resource conditions generated (see note below) |
| 7 | Local satellite training *(next, after Phase 6 review)* |
| 8 | Standard FedAvg |
| 9 | Non-IID experiments |
| 10 | Adaptive federated learning |
| 11 | Staleness-aware FL |
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
for the full, explicit real-vs-simulated breakdown. No federated or DRL
components exist yet, and remain out of scope until local per-client
training (the next milestone) and then actual Federated Learning are
built.*

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
                small reviewable deliverables like results/reports/*.md
                and results/plots/{dataset,baseline,temporal,satellite}/
                are committed)
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
