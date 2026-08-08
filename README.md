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
Phase 3  — Baseline Model              NOT STARTED
Phase 4+ — (see Development phases)    NOT STARTED
```

Phase 1 produced a real, reproducible NSL-KDD dataset pipeline (load →
clean → label → split → encode/scale → processed dataset + metadata),
run successfully end-to-end on the real dataset. See
`docs/datasets/dataset_selection.md` for the dataset decision and
`results/reports/dataset_quality.md` for measured statistics. No models
have been trained, and no federated learning or DRL logic has been
implemented yet. The frontend dashboard intentionally shows "Awaiting
live data" states rather than fabricated metrics.

## Development phases

| Phase | Scope |
|---|---|
| 0 | Repository + project foundation ✅ complete |
| 1 | Dataset acquisition and analysis ✅ complete (preprocessing pipeline also implemented — see note below) |
| 2 | Data preprocessing ✅ substantially complete as part of Phase 1 (see note below) |
| 3 | Baseline anomaly detection *(next)* |
| 4 | Temporal sequence construction |
| 5 | Spatio-temporal anomaly model |
| 6 | Satellite client simulation |
| 7 | Local satellite training |
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
dataset required all of it. No model training, feature learning, or
FL/DRL logic was introduced — that remains entirely out of scope until
Phase 3+.*

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

## Repository structure

```
configs/        Experiment/system configuration (YAML) — no hardcoded params
data/           raw / interim / processed / partitions (large files git-ignored)
src/            ML/FL/DRL source: data, preprocessing, models, training,
                federated, adaptive, drl, satellite, evaluation, utils
backend/        FastAPI application (API, services, schemas, core) + tests
frontend/       React + TypeScript + Vite application (OrbitShield UI)
experiments/    Experiment run configs/scripts, separate from src/
results/        Generated metrics, models, plots, logs, reports
                (mostly git-ignored; small reviewable deliverables like
                results/reports/dataset_quality.md and
                results/plots/dataset/ are committed)
notebooks/      Exploratory analysis
scripts/        Standalone utility scripts
tests/          Tests for src/ (data, models, federated, drl, integration)
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
