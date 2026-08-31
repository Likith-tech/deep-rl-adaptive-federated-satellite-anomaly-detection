# Final Report — Deep Reinforcement Learning-Driven Adaptive Federated
Framework for Satellite Network Anomaly Detection (OrbitShield)

**Status: every phase in this project's roadmap (0 through 16) is real,
measured, and committed.** This document is the single entry point
tying all of them together — it introduces no new experiment; every
number below is cited back to the phase report that actually measured
it. Written for faculty review and viva preparation, alongside the more
detailed materials in `docs/research/paper-study/`.

## 1. What this project is

A B.Tech capstone implementing a practical, integrated pipeline for
satellite network anomaly detection: a temporal anomaly-detection model,
non-IID federated learning across simulated satellite clients, and two
different adaptive-aggregation mechanisms (a deterministic rule and a
learned DQN controller), evaluated honestly at every step — including
where the more sophisticated mechanism did **not** win. This is
explicitly **not** presented as novel research; non-IID FL, staleness-
aware aggregation, and DRL-driven adaptive FL are active research areas.
The contribution is a practical, working, honestly-evaluated
implementation and comparison of these techniques.

## 2. The dataset, stated plainly

**NSL-KDD** — a real, terrestrial network intrusion-detection dataset,
not satellite-specific. It stands in for satellite network traffic
throughout this project (see `docs/datasets/dataset_selection.md` for
the full reasoning). The "satellite" framing comes from Phase 4's
simulation layer (Dirichlet-partitioned clients + simulated per-client
resource conditions), built on top of this real terrestrial data — not
from any real spacecraft telemetry. A separate, real-spacecraft-data
research track (ESA-ADB) was investigated in parallel and is discussed
in section 7.

## 3. The whole pipeline, phase by phase

| Phase | What was built | Headline real result | Full detail |
|---|---|---|---|
| 0 | Repository, backend/frontend skeleton | — | `01-phase-0-foundation.md` |
| 1 | NSL-KDD acquisition, cleaning, preprocessing | 107,077 train records, verified integrity | `02-phase-1-dataset.md` |
| 2 | Baseline MLP anomaly detector | KDDTest+ F1 **0.7790**, ROC-AUC 0.8972 | `03-phase-2-baseline.md` |
| 3 | Temporal GRU+attention model | First attempt rejected as degenerate (100% "success" on an invalid benchmark); corrected model F1 0.7558, honestly below baseline | `04-phase-3-spatio-temporal.md` |
| 4 | Simulated 8-satellite non-IID environment | Measured JS distance 0.527; simulated resource conditions | `05-phase-4-satellite-simulation.md` |
| 5 | Independent local training per satellite | Global-val F1 ranged 0.9242-0.9918 across clients | `06-phase-5-local-training.md` |
| 6 | Standard FedAvg (sample-count weighting) | KDDTest+ F1 **0.7404** | `07-phase-6-federated-learning.md` |
| 7 | Non-IID alpha sweep (0.1-10.0) | Fairness gap shrank ~41x with alpha; F1 peaked non-monotonically at alpha=5.0 (0.7640) | `08-phase-7-non-iid-federated-learning.md` |
| 8 | Rule-based adaptive FL (4 configs) | `resource_only` best F1 (**0.7547**); `performance_only` best fairness (0.1245) — not the fairness-weighted `combined` rule | `09-phase-8-rule-based-adaptive-fl.md` |
| 9B | DQN-learned adaptive FL | F1 0.7425 — did **not** beat Phase 8's best rule, an honest negative result | `11-phase-9b-drl-adaptive-federated-learning.md`, `12-phase-9b-drl-pilot.md` |
| 9 | Staleness- and resource-aware FL | Staleness-blind control beat both staleness-aware discount rules (F1 0.7371 vs. 0.7360/0.7354) | `15-phase-9-staleness-resource-fl.md` |
| 14 | Full cross-phase evaluation | Confirms `resource_only` (Phase 8) as the best mechanism found, across all of Phase 6/8/9B/9 | `16-full-evaluation.md` |
| 15 | Ablation studies (2 sweeps) | Confirms `resource_only`'s win is a genuine peak, not a coarse-grid artifact; confirms no staleness decay_rate beats the staleness-blind control | `17-ablation-studies.md` |
| 16 | OrbitShield web platform integration | Read-only FastAPI + React now render every result above on the live site | `18-orbitshield-integration.md` |

*(Also completed but not part of the primary comparison chain: Phase 9A,
real spacecraft validation on the small public OPSSAT-AD dataset — see
`10-phase-9-opssat-validation.md` and `results/reports/opssat_ad_results.md`.)*

## 4. The project's single most important honest finding

Across every aggregation mechanism tried on the identical Phase 4/6
partition — plain FedAvg, four rule-based adaptive configs, a trained
DQN controller, and three staleness-aware discount rules — **the
simplest, fixed, hand-specified rule (Phase 8's `resource_only`) still
wins**, and two independent Phase 15 ablation sweeps confirmed this is
a genuine, robust result, not an artifact of a coarse comparison. Making
the aggregation mechanism "smarter" (learned, or staleness-corrected)
did not improve on a simple weighting rule in this project's specific,
small-scale, single-seed setting. This is reported as the honest,
central result of the whole adaptive-FL investigation — not adjusted,
not hidden, and not re-run to chase a better headline number.

A second consistent pattern: **no method ever won on both accuracy and
fairness simultaneously.** The most-accurate mechanism (`resource_only`,
fairness gap 0.2259) is never the fairest one (`performance_only`,
0.1245) — this performance/fairness tension held across every phase (8,
9B, 9) that measured both.

## 5. What "adaptive" and "DRL-driven" actually mean in this project

- **Phase 8 (rule-based):** a fixed arithmetic formula combining four
  signals (performance, data, resource, fairness) into a per-client
  aggregation weight — no learning, no reward signal.
- **Phase 9B (DRL/DQN):** a small Q-network observes the same four
  signal families and picks among 9 discrete weighting profiles each
  round, trained via 750 real FL rounds of pretraining against a reward
  built only from global-validation performance (never KDDTest+,
  verified by AST-based privacy tests).
- **Phase 9 (staleness-aware):** Phase 4's simulated per-client
  availability genuinely gates round-to-round participation for the
  first time in this project, producing real (not assumed) staleness,
  which a fixed discount rule then weights into aggregation.

All three sit on top of the same underlying FedAvg/client/server code
from Phase 6, extended additively — none of Phase 6's original,
tested behavior was ever modified to build the later phases.

## 6. Data & result integrity — the rule this whole project followed

No experiment result in this project was ever fabricated, adjusted
after the fact, or hidden for being unfavorable. Where a phase's own
hypothesis wasn't supported by its measured result (Phase 3's temporal
model below baseline, Phase 9B's DQN below Phase 8's best rule, Phase
9's staleness correction below its own control), that result stands as
reported. Every phase's report states its real configuration, its real
measured numbers, and its real limitations — including single-seed
caveats that apply throughout. The frontend (Phase 16) carries the same
rule into the live platform: routes without real backing data show
"Awaiting live data," never a fabricated number.

## 7. Parked research: the ESA-ADB track

A separate, exploratory investigation of ESA-ADB (real ESA spacecraft
telemetry across 3 missions) ran in parallel on its own branch history,
outside this numbered roadmap: acquisition, a structural/leakage audit,
a federated architecture design for heterogeneous per-client channel
ownership, and one real FedAvg baseline run — a genuine, honestly
reported weak result (KDDTest+-analog ROC-AUC 0.4855). This track is
parked, not abandoned silently — full record in
`docs/research/experiments/esa-adb/README.md`. It remains a candidate
for future work but is not part of this project's completed, evaluated
roadmap.

## 8. Reproducing this project

Every phase's exact commands are documented in `README.md`'s
per-section "```bash ... ```" blocks, from raw dataset download through
the OrbitShield backend/frontend. All seeds (42 throughout), configs
(`configs/*.yaml`), and committed artifacts (`results/reports/*.md`,
`experiments/*/run_config.json`) are real and checked in — nothing
required to reproduce a reported number is hidden or gitignored, except
large regenerable intermediates (raw data, model checkpoints) explicitly
called out in `.gitignore`.

## 9. Limitations that apply across the whole project

- **Single seed throughout.** No phase ran a statistically powered
  multi-seed comparison; small differences (a few thousandths of F1)
  between methods should not be over-interpreted.
- **NSL-KDD is terrestrial, not satellite data.** The "satellite" framing
  is a simulation layer (Phase 4) built on real network-traffic data,
  not real spacecraft telemetry, and is disclosed as such everywhere it
  appears.
- **Simulated resource/availability fields** (Phase 4) drive Phase 8/9B's
  `resource` signal and Phase 9's staleness mechanism — real numbers
  sampled within configured ranges, not measured satellite hardware.
- **No spatial/orbital realism** — non-IID label distributions and
  simulated resource heterogeneity stand in for genuine orbital
  mechanics or inter-satellite links.
- **OrbitShield's backend is read-only** — it surfaces real results; it
  does not (yet) trigger training from the UI.

## 10. Viva preparation

See `docs/research/paper-study/06-viva-preparation/` for prepared
likely-question responses, paper-based questions, and plain-language
explanations of every technique used — all written against the same
real results cited throughout this report.

## 11. What remains

Per the master tracker (`docs/project-progress/00-project-plan.md`),
every phase through 16 is complete. The only remaining item is final
integration — merging this roadmap's branch history back into
`project-development` — which happens with explicit review, not
automatically, since it is a shared-branch action.
