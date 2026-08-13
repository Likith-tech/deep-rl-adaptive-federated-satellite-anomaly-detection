# Phase 8 — Rule-Based Adaptive Federated Learning

## Why Phase 8 was needed

Phase 6 gave every satellite influence over the shared model strictly
proportional to how much local data it had. Phase 7 then showed that
when satellites' data is very different from each other (non-IID),
that simple rule leaves some satellites much worse served than others
— a "client fairness gap" that grew about 41x between the mildest and
strongest heterogeneity we tested. Sample-count weighting has no way to
notice or respond to that. Phase 8 asks: what if we weighted clients
using more than just "how much data do you have"?

## What problem from Phase 6/7 motivated it

Phase 6's rule (`weight = local samples / total samples`) treats a
large, unhelpful, narrow-category client exactly the same as a small,
diverse, currently-useful one. Phase 7 proved the fairness cost of
that is real and substantial. Phase 8 builds a transparent alternative
that a project guide (or anyone) can inspect line by line — no hidden
learned behavior yet.

## What adaptive FL means

Instead of a fixed rule, "adaptive" means the server can change how it
weighs each satellite's contribution based on current conditions. In
Phase 8, that adaptation is a small, fixed arithmetic formula — not
something that learns. Learning-based adaptation (via reinforcement
learning) is Phase 9, deliberately kept separate.

## What rules were used

Every client's contribution weight comes from a **client score**,
combining four ingredients, each scaled to a common 0-1 range so
they're comparable:

```
client_score = w_perf   x performance
             + w_data   x data
             + w_resource x resource
             + w_fair   x fairness
```

- **performance** — how well the satellite's just-trained update does
  on the shared validation set, right now, this round.
- **data** — how many local training samples the satellite has (the
  same idea Phase 6 used alone).
- **resource** — the satellite's simulated bandwidth, compute power,
  availability, and connectivity (from Phase 4) — clearly simulated,
  not real satellite measurements.
- **fairness** — how diverse the satellite's local mix of attack types
  is (measured the same way Phase 7 measured non-IID-ness). A
  satellite holding rare attack types gets a boost here even if its
  raw performance is mediocre.

We tested four weight combinations: three "ablations" that isolate one
ingredient at a time (`performance_only`, `resource_only`,
`data_only`), and one **combined** rule using all four
(`performance 0.40, data 0.25, resource 0.15, fairness 0.20`) — chosen
by reasoning about what should matter, before looking at any test
result.

## How client weights/selection work

Every satellite still trains and still counts toward the final model
every single round — nothing is dropped. What changes is HOW MUCH each
satellite's update counts. This is called **weighting**, as opposed to
**selection** (where some clients would be excluded some rounds). We
chose weighting deliberately: selection risks silently dropping a
satellite that happens to hold a rare attack category for an entire
round, which directly conflicts with the fairness goal. A tool for
selection exists in the code and is tested, but wasn't used for these
experiments.

## Why those rules were chosen

- Performance was weighted highest (0.40) in the combined rule because
  it most directly reflects whether an update is currently helping.
- Data comes second (0.25) — it's still a meaningful signal, just not
  the only one anymore.
- Fairness got a real, deliberate share (0.20) — specifically to stop
  high performers from drowning out satellites with unusual data.
- Resource got the smallest share (0.15) since it's simulated and the
  least directly tied to model quality — its main purpose here is to
  wire up the signal for later phases.

These weights were fixed BEFORE we ran a single KDDTest+ evaluation.

## Experimental configuration

Everything Phase 6 used, unchanged: the same 8 satellites, the exact
same Phase 4 partition (not a fresh one), the same MLP architecture,
10 communication rounds, 1 local epoch per round, batch size 128,
learning rate 0.001, Adam, BCEWithLogitsLoss, seed 42. The ONLY thing
that changed across our 4 experiments was the rule weights above.

## Results

| Rule | Validation F1 | KDDTest+ F1 | Client fairness gap |
|---|---:|---:|---:|
| performance_only | 0.9890 | 0.7447 | **0.1245** (best fairness) |
| resource_only | 0.9860 | **0.7547** (best test score) | 0.2259 |
| data_only | 0.9850 | 0.7397 | 0.2113 |
| combined | 0.9883 | 0.7439 | 0.1496 |
| Phase 6 FedAvg (reference) | 0.9879 | 0.7404 | not measured |

## FedAvg vs Adaptive FL comparison

All four adaptive rules except `data_only` beat Phase 6's KDDTest+ F1
of 0.7404 — `resource_only` by the widest margin (+0.0143), `combined`
by a smaller but real margin (+0.0035). `data_only` (0.7397) came in
almost exactly where Phase 6 sits, which makes sense: it's the closest
analog to Phase 6's own sample-count-based rule within this framework.

## Fairness results

Here's the honest, slightly surprising part: the rule specifically
designed to protect fairness (`combined`, with a real 0.20 weight on
the fairness signal) did **not** produce the best fairness outcome.
That distinction went to `performance_only` (gap 0.1245) — combined
was second-best (0.1496). We are reporting this exactly as measured,
not smoothing it into a tidier story. A plausible explanation:
evaluating "performance" against the shared validation set may already
implicitly reward satellites whose local update generalizes well,
which can overlap with diversity in ways our simple, static entropy
signal doesn't fully capture round to round. With one experiment per
rule, we can't yet tell whether this is a real pattern or noise — a
question the next phase's more systematic search could help answer.

## Communication results

No change from Phase 6 — every rule still has all 8 satellites
participate and transmit an update every round, so total estimated
communication (~14.6 MB across 10 rounds, same 23,937-parameter model)
is identical. Adaptive weighting affects how much an update counts
after arriving, not whether it's sent.

## Limitations

- Rule-based only — genuinely NOT reinforcement learning. No learned
  parameters, no reward signal, nothing trained.
- Single seed (42) per rule — not a statistically powered comparison;
  differences of a few hundredths of an F1 point should be read with
  that in mind.
- The fairness metric evaluates each rule's final model on every
  satellite's own local training data — a proxy, not a genuine
  held-out per-client test.
- Resource signals remain simulated (Phase 4), not real satellite
  telemetry.
- Client selection (dropping low-scoring clients) was built and tested
  but not experimentally compared here — only weighting was.
- NSL-KDD remains terrestrial network data throughout.

## What Phase 9 will do

Phase 9 replaces these fixed, hand-chosen weights with a DRL (DQN)
controller that *learns* how to weight or select clients from a reward
signal — using the same four signal types (performance, data,
resource, fairness) this phase already wired up, but learned instead
of manually specified. This phase's results become the baseline Phase
9 needs to actually beat.
