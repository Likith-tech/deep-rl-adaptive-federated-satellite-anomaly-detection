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
Phase 3  — Spatio-Temporal Model       NOT STARTED
Phase 4  — Satellite Simulation        NOT STARTED
Phase 5  — Local Satellite Training    NOT STARTED
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

## What's explicitly NOT done yet (as of Phase 2)

- No satellite simulation of any kind exists yet.
- No federated learning (no Flower, no FedAvg, no clients).
- No DRL/DQN/reinforcement learning of any kind.
- The OrbitShield website is not connected to any real model — it still
  shows "Awaiting live data."
- No spatio-temporal / sequence model exists yet.

These are all planned for later phases and are intentionally out of
scope until their turn.
