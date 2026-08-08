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

This repository is currently at **Phase 0 — project foundation**. No
dataset has been downloaded, no models have been trained, and no
federated learning or DRL logic has been implemented yet. The frontend
dashboard intentionally shows "Awaiting live data" states rather than
fabricated metrics.

## Development phases

| Phase | Scope |
|---|---|
| 0 | Repository + project foundation *(current)* |
| 1 | Dataset acquisition and analysis |
| 2 | Data preprocessing |
| 3 | Baseline anomaly detection |
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

## Repository structure

```
configs/        Experiment/system configuration (YAML) — no hardcoded params
data/           raw / interim / processed / partitions (large files git-ignored)
src/            ML/FL/DRL source: data, preprocessing, models, training,
                federated, adaptive, drl, satellite, evaluation, utils
backend/        FastAPI application (API, services, schemas, core) + tests
frontend/       React + TypeScript + Vite application (OrbitShield UI)
experiments/    Experiment run configs/scripts, separate from src/
results/        Generated metrics, models, plots, logs, reports (git-ignored)
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
# Backend
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

## Datasets

No dataset is bundled with this repository. Phase 1 will document
dataset selection (e.g. CICIDS2017, UNSW-NB15, NSL-KDD, or a
satellite-specific dataset where available) and acquisition steps,
including disk space requirements, before anything is downloaded.
Terrestrial datasets, if used, will be explicitly partitioned to
simulate non-IID satellite clients with connectivity, latency,
bandwidth, and resource constraints — they are not satellite datasets
by nature.

## Data & result integrity

This project never fabricates experimental results. Any metric not yet
backed by a real experiment is shown as "Not available" / "Awaiting
experiment", and simulated data is always clearly labeled as such.
