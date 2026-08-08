# Phase 0 — Project Foundation

## What was the goal?

Before writing any AI/ML code, we needed a clean, organized project
folder — so that everything we build later (dataset code, models,
federated learning, DRL, and the website) has a proper place to live.
This phase was about building that skeleton, not about doing any actual
machine learning yet.

## What did we do?

- Created the full folder structure for the whole project (data, source
  code, backend, frontend, experiments, results, tests, docs).
- Set up a Python backend (FastAPI) with a simple "health check" page
  that just says "I'm running."
- Set up the frontend website skeleton (React + TypeScript) called
  **OrbitShield** — with a sidebar, top bar, and a dashboard page that
  says "Awaiting live data" everywhere, instead of showing fake numbers.
- Wrote a `README.md` explaining the whole project and how to run it.
- Set up Git (version control) so all our work is tracked and safely
  backed up on GitHub, on a branch called `project-development`.
- Made sure things like passwords, downloaded datasets, and temporary
  files never get accidentally saved to GitHub (`.gitignore`).

## Why did we do it?

A capstone project like this eventually has a LOT of moving parts:
dataset code, multiple AI models, a simulated satellite network, a
federated learning system, a reinforcement learning agent, and a
professional website. If we don't organize the folders and tooling
properly from day one, the project turns into a mess that's hard to
explain, hard to grade, and hard to build on. Phase 0 is the
foundation everything else stands on.

## What did we create?

- `backend/` — the Python API server (FastAPI)
- `frontend/` — the OrbitShield website (React + TypeScript)
- `src/` — where all the real ML/FL/DRL code will live, organized into
  subfolders (`data`, `preprocessing`, `models`, `training`,
  `federated`, `adaptive`, `drl`, `satellite`, `evaluation`, `utils`)
- `configs/` — settings files (so we don't hardcode numbers everywhere)
- `data/`, `results/`, `experiments/`, `docs/`, `tests/` — folders for
  datasets, outputs, experiment records, documentation, and automated
  tests
- `README.md` — the main explanation of the whole project

## What did we actually achieve?

- The backend runs and responds to a health check
  (`http://localhost:8000/health`).
- The frontend runs and shows the OrbitShield dashboard shell in a
  browser.
- Both were tested and confirmed working (2 automated tests passed).
- Everything was committed to GitHub on the `project-development`
  branch.
- No AI model, no dataset, and no fake data existed yet — intentionally.

## How do I explain this to my mam?

"Before building the AI part, I first set up the project properly —
like laying the foundation of a building. I created the folder
structure, a basic backend server, and a basic website shell, all
connected to GitHub for version control. Nothing intelligent happens
yet — it's just the skeleton the AI system will be built on top of."

## Important technical terms

- **Repository (repo):** the project's folder, tracked by Git so every
  change is saved and recoverable.
- **Backend:** the server-side program that will eventually run our AI
  models and answer requests from the website.
- **Frontend:** the website itself — what a user sees and clicks on.
- **FastAPI:** a Python tool for building backend servers.
- **React + TypeScript:** tools for building the website.
- **`.gitignore`:** a file that tells Git "never save these files" —
  used to keep secrets and huge datasets out of GitHub.

## Problems/limitations

- Nothing is connected to real data yet — the dashboard is a shell.
- No AI/ML logic exists yet.
- This phase, by itself, doesn't prove anything about the research —
  it's purely engineering setup.

## What comes next?

Phase 1 — get an actual dataset, clean it, and prepare it so a model
can be trained on it later.
