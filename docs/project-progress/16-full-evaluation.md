# Phase 14 — Full Cross-Phase Evaluation

## Why Phase 14 was needed

Every phase from 2 through 9 measured its own result and compared it
only to the phase immediately before it — a reasonable way to work
through the project step by step, but it means no single document ever
put every method side by side. By this point the project has built two
centralized models, a plain FedAvg baseline, a heterogeneity sweep, four
rule-based adaptive-FL configs, a learned DQN controller, and three
staleness-aware configs — fourteen real, measured results in total.
Phase 14 doesn't run anything new; it reads every one of those already-
committed numbers into one place so the whole project's progress (and
its honest limits) can be seen at a glance.

## What this phase is NOT

Not a new experiment. `scripts/generate_full_evaluation_report.py`
trains nothing, touches no partition, and doesn't re-evaluate any model
— every number in its output is copied, by hand, straight from the
report that originally measured it (cited next to each row so it can be
checked). If any number here ever disagrees with its source report, the
source report is right.

## The two comparisons that matter

**All methods, ranked** — mixes different data partitions (Phase 7's
heterogeneity sweep) with different aggregation mechanisms (Phase 6, 8,
9B, 9) on the SAME partition. Useful for a full overview, but not a
clean like-for-like comparison, since Phase 7's alpha sweep changes the
data itself, not the algorithm.

**Aggregation mechanism only** — the more informative comparison: every
entry here uses the exact same Phase 4/6 partition, model, and round
budget, varying only HOW client updates are combined. This isolates the
real question: does a smarter aggregation rule beat a simpler one, all
else equal?

## Results

### Full ranking (KDDTest+ F1)

| Method | F1 |
|---|---:|
| Phase 2 — Centralized MLP baseline | 0.7790 |
| Phase 7 — alpha=5.0 (non-IID sweep) | 0.7640 |
| Phase 3 — Temporal GRU+Attention | 0.7558 |
| Phase 8 — resource_only | **0.7547** (best federated mechanism) |
| Phase 7 — alpha=10.0 | 0.7541 |
| Phase 8 — performance_only | 0.7447 |
| Phase 8 — combined | 0.7439 |
| Phase 9B — DQN adaptive FL | 0.7425 |
| Phase 7 — alpha=0.1 | 0.7409 |
| Phase 6 — FedAvg | 0.7404 |
| Phase 8 — data_only | 0.7397 |
| Phase 9 — no_correction (staleness) | 0.7371 |
| Phase 9 — exponential_decay | 0.7360 |
| Phase 9 — linear_decay | 0.7354 |

### Aggregation mechanism only (same partition throughout)

Ranked, best to worst: `resource_only` (0.7547) > `performance_only`
(0.7447) > `combined` (0.7439) > DQN (0.7425) > FedAvg (0.7404) >
`data_only` (0.7397) > staleness `no_correction` (0.7371) > staleness
`exponential_decay` (0.7360) > staleness `linear_decay` (0.7354).

## The honest headline finding

Across every mechanism this project has tried on the exact same data,
**a simple, hand-specified rule (Phase 8's `resource_only`) still wins**
— it beats the learned DQN controller (Phase 9B) and both staleness-
aware discount rules (Phase 9). Neither "make it learned" nor "make it
staleness-aware" improved on "weight by simulated resource desirability,
fixed and by hand." That's a real, useful, slightly humbling finding for
a project built specifically to explore adaptive and learned FL — and we
are reporting it as the honest current state of the evidence, not
smoothing it into a tidier story.

## The fairness/accuracy trade-off, confirmed again

No single method wins on both KDDTest+ F1 and client fairness. The
fairest method (`performance_only`, gap 0.1245) is not the most accurate
one; the most accurate method (`resource_only`, gap 0.2259) is among the
least fair. This tension first showed up in Phase 8's own results and
holds across every later phase (9B, 9) that measured both — a consistent
pattern across five different aggregation mechanisms now, not a one-off.

## Why centralized models still lead

Phase 2 (centralized MLP) and Phase 3 (centralized temporal model) both
score higher than every federated method — expected, and already
explained starting in Phase 6's own report: a centralized model sees all
the data at once with no communication constraint, while every federated
method here trades some raw accuracy for training without centralizing
raw satellite data, on a deliberately modest round/epoch budget
throughout.

## Limitations

- No new experiments — every number is a citation, not a fresh
  measurement, so this phase inherits every limitation already disclosed
  in its 8 source reports (single seed throughout, simulated resource/
  availability fields, NSL-KDD as a terrestrial proxy, etc.).
- Small F1 differences (a few thousandths) between adjacent-ranked
  methods should not be read as statistically meaningful — none of the
  underlying phases ran multiple seeds.

## What Phase 15 will do

Phase 14 identifies WHAT the best-known result is (`resource_only`) but
not WHY it beats the more sophisticated mechanisms. Phase 15 (ablation
studies) investigates that on a couple of specific design axes — real
runs, real numbers, honest reporting either way.
