# Development Log

This directory is the permanent, chronological record of how this
capstone project ("Deep Reinforcement Learning-Driven Adaptive
Federated Framework for Satellite Network Anomaly Detection",
platform name OrbitShield) was actually built — phase by phase, in the
order it really happened, for use in the final report, an eventual
IEEE-style paper, viva defense, and project review.

## Why this exists

Every other document in this repository (`docs/project-progress/`,
`results/reports/`) explains **what** each phase produced. This log
exists to also capture **how the work was actually carried out
session-by-session** — objectives, what was asked of the AI assistant,
what was implemented, problems hit and how they were resolved,
decisions made (and alternatives rejected), and exact git/test state
at the time — in one place, ordered by when it happened.

## Source of truth and honesty policy

Every entry in this log was reconstructed from artifacts that already
exist in this repository: `docs/project-progress/`, `docs/research/`,
`results/reports/`, `experiments/`, `configs/`, and real Git history
(`git log`, `git show`). **No conversation transcript from before this
documentation system existed was preserved outside this repository.**
Where an entry needed the *original chat prompt* or the *AI assistant's
original response text* and neither survived in any repository file,
the entry says so explicitly:

> "Original chat prompt/response not preserved in repository;
> reconstructed from available project artifacts."

Nothing in this log is invented. Metrics, dates, commit hashes, and
file lists are all taken directly from the sources above. Test counts
for historical phases were computed by counting `def test_` occurrences
in the actual source tree at each phase's commit (`git grep "def test_" <commit>`)
— a real, reproducible count, not a memory or estimate — and are
labeled as such rather than presented as "tests passed at the time"
(which was not independently re-verified for phases before Phase 9A).

One genuine exception exists: `docs/research/claude-research/phase-9-real-spacecraft-dataset-search-prompt.md`
and `...-response.md` are real, preserved prompt/response artifacts
from the Phase 9 dataset-search research step, dated 2026-08-13. These
are referenced directly (not reconstructed) in `010-phase-9a-opssat.md`.

## Files in this directory

| File | Contents |
|---|---|
| `README.md` | This file |
| `000-project-start.md` | Project inception — name, scope, repository creation |
| `001-phase-0.md` through `010-phase-9a-opssat.md` | One chronological entry per completed phase |
| `MASTER_PROJECT_JOURNAL.md` | One summary table, Phase 0 through Phase 9A |
| `CLAUDE_PROMPT_RESPONSE_INDEX.md` | The policy for recording all future (Phase 9B onward) AI-assisted work |

## What this directory is NOT

- Not a replacement for `docs/project-progress/` (the plain-language,
  viva-ready write-ups) or `results/reports/` (the measured-results
  reports) — this log points to them rather than duplicating their
  numbers.
- Not a place where historical conclusions get revised. If a later
  phase's audit corrects an earlier claim (e.g. Phase 9A's
  sequence-length-1 terminology correction), that correction is
  recorded as a **new, dated note referencing the old entry** — the
  original entry is never silently rewritten. This matches the
  research-preservation convention already established in
  `docs/research/decisions/research_decision_log.md`.
- Not modified retroactively to make past work look tidier than it
  was. Rejected experiments, transcription errors, and honest
  below-baseline results are recorded here exactly as they are
  recorded elsewhere in the repository.
