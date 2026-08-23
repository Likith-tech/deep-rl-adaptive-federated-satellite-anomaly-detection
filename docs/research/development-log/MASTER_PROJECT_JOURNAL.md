# Master Project Journal — Phase 0 through Phase 9A

One-row-per-phase summary. Full detail, including all 27 documented
fields per phase, lives in the individual `0XX-*.md` entries in this
directory. Every number below is taken directly from repository
artifacts (`docs/project-progress/`, `results/reports/`, Git history)
— see the corresponding entry file for sources.

**Addendum, 2026-08-15 — Phase 9B status (not yet a numbered entry
here; no phase-completion commit exists yet):** Phase 9B (DQN adaptive
FL) design was approved for implementation
(`docs/research/experiments/phase-9b-drl/design-checkpoint-review-v2.md`)
and its PILOT SCOPE was implemented and verified the same day
(`docs/project-progress/12-phase-9b-drl-pilot.md`) — `configs/drl.yaml`,
`src/drl/{action_space,state,reward,dqn_agent,config}.py`, 46 new
synthetic-fixture tests (all passing, full suite 237/237, zero Phase
0-9A regressions), and a pilot-verification script/report using
fabricated client signals only. **No DQN training, no real-data
experiment, no performance claim, and no commit exist yet** — a full
`010A-phase-9b-*.md` entry will be added to this log once Phase 9B
actually completes (a real pretraining run + 3-way comparison), matching
the existing convention of only logging phases once their code has
actually run and produced real, measured results.

| Phase | Objective | Dataset | Model | Main Result | Problems | Decision | Branch | Commit | Status | Next Step |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 — Foundation | Repository/backend/frontend skeleton | None | None | Backend health check + frontend shell working, 2 tests passed | None recorded | Platform name OrbitShield; monorepo; "Awaiting live data" policy | `project-development` | `8e57494` | Ancestor of `origin/project-development` | Phase 1 |
| 1 — Dataset | Acquire, clean, split NSL-KDD | NSL-KDD (real, terrestrial) | None | 107,077/18,896/22,544 train/val/test; 0 missing/duplicate/infinite values; 16 tests passed | None recorded | NSL-KDD selected over alternatives; strict leakage avoidance | `project-development` | `c0214ee` | Ancestor of `origin/project-development` | Phase 2 |
| 2 — Baseline | First real anomaly detector (MLP) | NSL-KDD | MLP 121→128→64→1 | KDDTest+ F1 **0.7790** (Acc 78.3%, ROC-AUC 89.7%) | Val-vs-test gap investigated, explained (unseen attack types) | Simple architecture, no aggressive tuning, honest reporting | `project-development` | `661ae74` | Ancestor of `origin/project-development` | Phase 3 |
| 3 — Temporal | GRU+attention sequence model | NSL-KDD | GRU+Attention (`TemporalGRUAttention`) | KDDTest+ F1 **0.7558-0.756** (slightly below Phase 2) | ANY-anomaly labeling found degenerate (100% score, near-single-class) | Rejected ANY-anomaly; selected LAST-record labeling on evidence | `project-development` | `adf1be4` | **Pushed** — `origin/project-development` tip | Phase 4 |
| 4 — Satellite Simulation | 8-client non-IID environment + simulated resources | NSL-KDD (simulated client split) | None (environment only) | Mean pairwise JS distance **0.527**; 0 data loss/duplication | None recorded | Only training split partitioned; validation/test stay global | `project-development` → `-2` | `20f934b` | Present on `origin/project-development-2` | Phase 5 |
| 5 — Local Training | Independent per-client training, no communication | NSL-KDD (Phase 4 partition) | Phase 2 MLP, 8 independent copies | Global-val F1 **0.9242**–**0.9918** across 8 clients | None recorded | Shared init (seed 42) to isolate data effects | `project-development-2` | `47728e6` | **Pushed** — `origin/project-development-2` tip | Phase 6 |
| 6 — FedAvg | Standard synchronous Federated Learning | NSL-KDD (Phase 4 partition) | Phase 2 MLP, FedAvg | Val F1 **0.9879** (round 10); KDDTest+ F1 **0.7404** (honest, below baseline) | Below-baseline test F1 explained (small round/epoch budget), not hidden | Server never touches raw data (tested); selection by val loss only | `project-development-2` | `8f03d62` | **Not pushed** (local ahead by 1) | Phase 7 |
| 7 — Non-IID Study | Isolate heterogeneity as sole variable (5 alphas) | NSL-KDD (fresh partitions per alpha) | Phase 2 MLP, FedAvg (unchanged) | JS distance 0.6892→0.1452; fairness gap 0.2623→0.0064 (~41x); KDDTest+ F1 peaked non-monotonically at alpha=5.0 (0.7640) | Non-monotonic F1 not fully explained — flagged as noise-vs-real-effect, undetermined | Every factor but alpha held fixed; original Phase 4/6 partition untouched | `project-development-3` | `45c8d7d` | **Pushed** at the time (later superseded locally by Phase 9A) | Phase 8 |
| 8 — Rule-Based Adaptive FL | Fixed 4-signal scoring rule replaces sample-count weighting | NSL-KDD (Phase 6 exact partition) | Phase 2 MLP, weighted FedAvg | `resource_only` best KDDTest+ F1 (**0.7547**); `performance_only` (not `combined`) best fairness gap (**0.1245**) | Fairness-weighted rule did NOT win fairness — reported honestly, unexplained with certainty | Weighting (not selection) chosen for fairness safety; weights fixed pre-test | `project-development-3` | `45c8d7d` | Same as Phase 7 (shared commit) | Phase 9A / 9B |
| 9A — OPSSAT-AD Validation | Real spacecraft telemetry validation (Track B) | OPSSAT-AD (real ESA OPS-SAT, operational-fault labels) | Phase 3 `TemporalGRUAttention`, unmodified, seq_len=1 | Test F1 **0.9315**, MCC 0.9144, AUCPR 0.9759 — but audited and re-described as **segment-level classification**, not genuine temporal modeling | `sequence_length=1` overclaiming caught by a dedicated audit; transcription/channel-name/license discrepancies also found and documented | Kept existing implementation (Option A), corrected terminology only; deferred genuine sequence modeling (Option B) to a future phase | `project-development-3` | `c58a8a2` | **Not pushed** (local ahead by 1) | Phase 9B |

## Cross-phase test count (cumulative, via `git grep "def test_" <commit>`)

| Commit | Phase(s) | Cumulative test functions |
|---|---|---:|
| `8e57494` | 0 | 2 |
| `c0214ee` | 1 | 18 |
| `661ae74` | 2 | 30 |
| `adf1be4` | 3 | 69 |
| `20f934b` | 4 | 109 |
| `47728e6` | 5 | 121 |
| `8f03d62` | 6 | 143 |
| `45c8d7d` | 7 + 8 | 168 |
| `c58a8a2` | 9A | 184 |

This count method (grepping `def test_` in the actual tree at each
commit) is a real, reproducible measurement done for this journal — not
a memory or an estimate. It is not identical to "tests passed," since
it was not feasible to re-execute every historical commit's test suite
for this journal; Phase 9A's figure (184, i.e. +16 over Phase 7-8) was
cross-checked by actually running `pytest` on the current tree, which
independently confirmed 16 passing tests in the exact modules Phase 9A
added.

## Branch topology (as of this journal's writing)

```
main                    -> 8e57494 (Phase 0 only)
project-development     -> adf1be4 (Phase 0-3), pushed, == origin
project-development-2   -> 8f03d62 (Phase 0-6), 1 commit ahead of origin (Phase 6 unpushed)
project-development-3   -> c58a8a2 (Phase 0-9A), 1 commit ahead of origin (Phase 9A unpushed)
project-development-4-drl -> c58a8a2 (identical to project-development-3), new, unpushed, isolated for Phase 9B
```

## Track separation (stated once here, applies to every phase above)

- **Track A** (Phases 1-3, 4-8, and the future Phase 9B): NSL-KDD —
  real, terrestrial network intrusion data; satellite layer simulated
  on top of it.
- **Track B** (Phase 9A only, so far): OPSSAT-AD — real ESA spacecraft
  telemetry, operational/physical-fault labels, no cyberattack labels,
  no federated structure. Never merged into Track A's metrics.
