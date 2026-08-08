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
Phase 3  — Temporal Model (GRU+Attn)   IN PROGRESS (corrected experiment run, awaiting review — see caveat below)
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
| `04-phase-3-spatio-temporal.md` | Temporal GRU+attention model — first attempt (ANY-anomaly labeling) was rejected as degenerate (100% F1, near-single-class task); corrected (LAST-record labeling, evidence-based) attempt scores F1 75.58% on KDDTest+, honestly slightly below the Phase 2 baseline's 77.90% (see caveat below) |
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

Phase 3 remains **IN PROGRESS** (not COMPLETE) pending review of this
corrected experiment.

## What's explicitly NOT done yet (as of Phase 3)

- No satellite simulation of any kind exists yet.
- No federated learning (no Flower, no FedAvg, no clients).
- No DRL/DQN/reinforcement learning of any kind.
- The OrbitShield website is not connected to any real model — it still
  shows "Awaiting live data."
- No spatial/multi-satellite component exists yet — Phase 3 only
  covered the temporal half of "spatio-temporal."
- Temporal context (in this simple last-record-label form) has not yet
  been shown to improve on the baseline — a real, useful finding for
  guiding later phases, not a blocker.

These are all planned for later phases and are intentionally out of
scope until their turn.
