# Phase 15 — Ablation Studies

## Why Phase 15 was needed

Phase 14 lined up every method this project has built and found
something worth digging into further: Phase 8's `resource_only` rule —
a fixed, hand-specified weighting, not anything learned — still beats
every more sophisticated mechanism tried, including the DQN controller
and both staleness-aware discount rules. But Phase 8 only ever tested
`resource_only` as one of four coarse options (0% or 100% resource
weight, nothing in between). And Phase 9's own report admitted its
staleness `decay_rate=0.5` was "a round, interpretable default... not
tuned." Phase 15 turns both of those loose threads into real, measured
answers instead of leaving them as guesses.

## Two sweeps, not a big search

We picked exactly two questions, both directly motivated by a specific
gap a prior phase's own report already named — not a broad, unfocused
hyperparameter search:

1. **Does weighting resource more keep helping, or does it peak
   somewhere before 100%?** Swept `w_resource` from 0.0 to 1.0 in five
   steps, keeping the remaining weight split across performance/data/
   fairness in Phase 8's own `combined` proportions.
2. **Would a different staleness decay_rate have changed Phase 9's
   result?** Swept `decay_rate` from 0.1 (aggressive discount) to 0.9
   (gentle discount) in five steps, reusing the exact same simulated
   sync/no-sync pattern Phase 9 used, so only the discount curve
   changes.

Both sweeps reuse Phase 8 and Phase 9's training code completely
unchanged — no new algorithm was written for this phase, only new
experiment configurations run through code that was already tested.

## A built-in honesty check

Each sweep includes one point that should exactly reproduce an
already-committed result: `w_resource=1.0` is mathematically identical
to Phase 8's `resource_only`, and `decay_rate=0.5` is identical to
Phase 9's `exponential_decay`. Both reproduced their original numbers
**exactly** (F1 0.7547 and 0.7360, bit for bit) — solid evidence this
sweep's code is faithful to the phases it's extending, not a subtly
different re-implementation.

## Results

### Resource-weight sweep

| w_resource | KDDTest+ F1 |
|---:|---:|
| 0.00 | 0.7426 |
| 0.25 | 0.7443 |
| 0.50 | 0.7464 |
| 0.75 | 0.7465 |
| 1.00 | **0.7547** |

**Finding:** the peak really is at 100% resource weight — the sweep
climbs steadily as resource weight increases, with no dip near the top.
Phase 8's counterintuitive `resource_only` result wasn't a lucky
edge-case of a coarse 4-option test; it's the genuine optimum across
this finer 5-point search too.

### Staleness decay-rate sweep

| decay_rate | KDDTest+ F1 |
|---:|---:|
| 0.10 | 0.7361 |
| 0.30 | 0.7359 |
| 0.50 | 0.7360 |
| 0.70 | 0.7358 |
| 0.90 | 0.7369 (best of the five) |

**Finding:** even the best decay_rate tested (0.90, gentlest discount)
scores 0.7369 — still below Phase 9's staleness-blind control
(`no_correction`, 0.7371). No decay_rate in this sweep changes Phase
9's conclusion: under this project's real, small-scale staleness (never
more than 3 rounds behind), discounting stale updates doesn't pay off,
no matter how gently or aggressively you discount them.

## What this adds to the project's evidence

Both findings make the earlier phases' results MORE trustworthy, not
less — Phase 8's surprising `resource_only` win was real and robust, and
Phase 9's negative staleness-correction result wasn't just bad luck with
an untuned parameter. Neither sweep changes which method sits at the top
of Phase 14's ranking.

## Limitations

- Single seed per sweep point, same caveat every phase before it
  carried.
- Each sweep varies one axis at a time; interactions between resource
  weight and staleness decay (a combined 2D sweep) weren't explored.
- The resource-weight sweep's split among performance/data/fairness was
  fixed at Phase 8's `combined` ratio, not itself swept — a further
  nested question, left for a future extension if ever needed.

## What Phase 16 will do

With every core FL/DRL/adaptive mechanism now evaluated and stress-
tested, Phase 16 turns to wiring OrbitShield's actual web platform
(FastAPI backend + React frontend) to these real, already-committed
results — currently the site shows nothing but "Awaiting live data"
placeholders.
