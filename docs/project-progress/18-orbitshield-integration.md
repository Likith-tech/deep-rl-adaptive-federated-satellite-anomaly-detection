# Phase 16 — OrbitShield Integration

## Why Phase 16 was needed

Every phase so far produced real, measured results — but they only
ever lived in markdown reports and JSON files. OrbitShield, the actual
web platform this project is named after, still showed nothing but
"Awaiting live data" on every page. Phase 16 wires the real backend
(FastAPI) and frontend (React) together so the platform finally shows
this project's own real numbers — not a redesign, not new experiments,
just connecting what already exists.

## What "integration" means here — read-only, not live

The backend never trains anything and never re-derives a metric. Every
endpoint reads straight from the same committed JSON files each
phase's own scripts already wrote (`results/models/federated/round_history.json`,
`experiments/adaptive_fl/*/round_history.json`,
`experiments/staleness_fl/*/round_history.json`,
`experiments/drl_adaptive_fl/full_run/{final_run,analysis_summary}.json`)
plus the same fixed reference numbers Phase 14's full-evaluation report
already used. If a number on the website ever disagrees with its source
report, the source report is right — the website is a window onto the
same facts, not a second source of truth.

## What got connected

**Backend** (`backend/app/api/{experiments,models,federated,drl}.py`,
`backend/app/services/results_registry.py`):
- `/experiments`, `/experiments/{id}` — all 14 real, completed methods
  (Phases 2, 3, 6, 7, 8, 9B, 9), with round-by-round history where one
  exists.
- `/models`, `/models/{id}` — the same registry, framed as trained
  model versions.
- `/federated/overview` — best/fairest mechanism, real comparison.
- `/federated/rounds/{id}`, `/federated/adaptive-control` — round
  history and Phase 8's real rule weights.
- `/drl/controller`, `/drl/decisions` — Phase 9B's real pretraining
  scale and its frozen-policy final run, round by round.

**Frontend** (`frontend/src/pages/{experiments,federated,drl,system}/`):
the corresponding pages — Experiments, Model Registry, Experiment/Model
Details, Analytics, Federated Overview, Training Rounds, Adaptive
Control, DRL Controller, Decision History, System Health — now fetch
and render this real data instead of showing the placeholder shell.

## What deliberately stayed a placeholder

Satellite Network, Traffic, Threats, Threat Investigation, Clients,
Aggregation, Activity, and Settings still show "Awaiting live data" —
honestly, on purpose. No real per-client held-out evaluation data, live
satellite telemetry, or system event log exists yet to back those
pages, and this project's own data-integrity rule says a page shows
"Awaiting live data" rather than fabricated numbers when nothing real
backs it.

## Verification

Both servers were actually started and driven with a headless browser
(Playwright/Chromium) against the running dev stack — not just built
or unit-tested. Every newly-wired route was navigated, screenshotted,
and checked for zero console errors and zero "Could not load"/"Awaiting
live data" fallback text: `/experiments`, `/models`,
`/federated/overview`, `/federated/rounds`, `/federated/adaptive-control`,
`/drl/controller`, `/drl/decisions`, `/system/health`, and a detail page
(`/experiments/phase8_resource_only`, confirmed showing the real F1
0.7547). The four still-placeholder routes were also re-checked to
confirm they honestly still say "Awaiting live data" — connecting new
pages didn't accidentally regress the ones that should stay
placeholders.

18 new backend tests (`backend/tests/test_{experiments,federated,drl}.py`)
assert against the same real numbers, e.g. that `/federated/overview`
identifies `resource_only` as the best mechanism and `performance_only`
as the fairest — not synthetic fixtures. Full repository suite: 419/419
passing. `npm run build` and `npm run lint` both pass clean.

## Limitations

- Read-only. No endpoint triggers training, and none will until a
  later phase explicitly adds that (out of scope here — Phase 16 is
  about surfacing results, not running new ones from the UI).
- The "Clients"/"Aggregation" pages could in principle show real
  per-round `aggregation_weights` data (already returned by
  `/federated/rounds/{id}`) — left as placeholders this phase because
  no dedicated per-client endpoint/page was built; a natural, easy
  follow-up rather than a blocker.
- No auth, no multi-user concerns — matches this project's scope (a
  capstone demo platform, not a production multi-tenant system).
