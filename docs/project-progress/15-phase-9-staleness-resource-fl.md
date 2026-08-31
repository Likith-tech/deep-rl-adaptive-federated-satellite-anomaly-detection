# Phase 9 — Staleness- and Resource-Aware Federated Learning

## Why Phase 9 was needed

Phase 4 gave every simulated satellite a set of simulated resource
conditions — bandwidth, latency, compute power, connectivity, and an
`availability_probability`. Phase 8 and Phase 9B both READ that
availability field (as one input to a scoring rule or a DQN's state),
but neither ever let it actually determine whether a satellite receives
the newest shared model each round. In every experiment before this
one, all 8 satellites fetched the exact current global model and
trained on it every single round — a synchronous, staleness-free world
that doesn't really test whether "resource-aware" claims mean anything
operationally. Phase 9 asks: what happens once a satellite's own
simulated availability genuinely determines whether it's working from
the latest model or an older one — and does correcting for that
staleness during aggregation actually help?

## What "staleness" means here

If a satellite misses a round (didn't sync in time — modeled here as a
probability tied to its simulated availability), it keeps training
locally on whatever global model it last received, not the current
one. When it finally submits its update, that update reflects an older
starting point than everyone else's. **Staleness** is simply "how many
rounds behind" that starting point is. A satellite that syncs every
round has staleness 0 always; one that's been out of sync for 3 rounds
has staleness 3 when it finally does sync again.

This is built as a real, seeded simulation — not an assumption bolted
onto the results afterward. Each round, each satellite's sync/no-sync
outcome is decided by a coin flip weighted by its own real Phase 4
`availability_probability`, so a satellite with 93% availability
(SAT-04) syncs almost every round, while one with 65% (SAT-02) misses
more often — exactly the intuition "resource-aware" is supposed to
capture.

## What changed vs. every phase before it

Every satellite still trains and submits an update every single
round — nothing is dropped, matching Phase 8/9B's own weighting-not-
selection philosophy. What's new is *which* global model a satellite's
update is trained against, and a discount applied to a stale update's
contribution before aggregation:

```
w_stale(staleness) =
    1                        (no_correction — staleness measured, ignored)
    1 / (1 + staleness)      (linear_decay)
    0.5 ^ staleness          (exponential_decay)
```

The discount is multiplied by the satellite's sample count, then
renormalized — the same style of transparent, fixed arithmetic rule
Phase 8 used, applied through the exact same, unmodified
`FederatedServer.aggregate_with_weights` Phase 8 already built.

## Experimental configuration

Everything Phase 6/8 used, unchanged: the same 8 satellites, the exact
same Phase 4 partition (not a fresh one), the same MLP architecture, 10
communication rounds, 1 local epoch per round, batch size 128, learning
rate 0.001, Adam, BCEWithLogitsLoss, seed 42. All three staleness
configs share the exact same seeded sync/no-sync pattern, so only the
aggregation response to that one real pattern differs between them.

## Results

| Config | Validation F1 | KDDTest+ F1 | Client fairness gap | Max staleness seen |
|---|---:|---:|---:|---:|
| no_correction | 0.9851 | **0.7371** (best of the three) | 0.1964 | 3 |
| linear_decay | 0.9853 | 0.7354 | 0.1964 | 3 |
| exponential_decay | 0.9854 | 0.7360 | 0.1994 | 3 |
| Phase 6 FedAvg (reference) | 0.9879 | 0.7404 | not measured | 0 (synchronous) |
| Phase 8 best (resource_only, reference) | not comparable | 0.7547 | see performance_only | 0 (synchronous) |
| Phase 9B DQN (reference) | not comparable | 0.7425 | 0.1833 | 0 (synchronous) |

Real, maximum staleness actually observed under this simulation: 3
rounds. That is a genuine, measured number from the seeded coin flips —
not a design target.

## Honest reading

This is a real negative result and we're reporting it exactly as
measured: **staleness-aware discounting did not beat the staleness-
blind control.** `no_correction` — the config that measures staleness
but ignores it when weighting — scored the highest KDDTest+ F1 (0.7371)
of the three, slightly ahead of `exponential_decay` (0.7360) and
`linear_decay` (0.7354). None of the three staleness configs beat Phase
6's plain FedAvg (0.7404), and none came close to Phase 8's best rule
(`resource_only`, 0.7547).

A plausible explanation, offered honestly as a hypothesis rather than a
proven cause: with only 8 clients, 10 rounds, and real staleness
topping out at 3 rounds, the amount of actual staleness in this run was
small enough that discounting it bought little, while the discount
itself slightly reduced how much weight otherwise-useful updates
received. This does not mean staleness-aware aggregation is a bad idea
in general — only that, under this specific real, measured, small-scale
simulation, it did not help. A larger round budget or a harsher
availability distribution (more real staleness to correct for) would be
the natural next experiment to distinguish "the technique doesn't work
here" from "there wasn't enough staleness in this run to matter" — not
attempted in this phase, per the standing rule against re-running an
experiment just to chase a better number.

## Fairness results

All three configs land in a similar, real range (gap 0.1964-0.1994),
worse than Phase 8's best fairness result (`performance_only`, 0.1245)
and Phase 9B's DQN (0.1833). Staleness-aware discounting did not
noticeably help or hurt fairness relative to the staleness-blind
control here — another honest, unremarkable finding, not smoothed into
a bigger claim than the data supports.

## Limitations

- Single run, single seed per config — not a statistically powered
  comparison, same caveat every prior phase (6, 7, 8, 9B) already
  carried.
- The seeded sync/no-sync pattern is shared across all three configs by
  design (so only the discount rule varies), meaning only one specific
  random staleness pattern was tested, not a distribution of them.
- Real staleness in this run topped out at 3 rounds — a small-scale
  result; a longer run or lower-availability simulation might tell a
  different story.
- `decay_rate=0.5` for `exponential_decay` was picked as a round,
  interpretable default before any result was seen, not tuned.
- The client-fairness proxy (own local training data, not a held-out
  split) carries the same limitation Phase 7/8/9B already documented.
- NSL-KDD remains terrestrial network data throughout; resource/
  availability fields remain simulated (Phase 4), not real satellite
  telemetry.

## What this means for later phases

Phase 8's `resource_only` rule remains this project's single best
KDDTest+ result across every adaptive-FL variant tried so far (Phase 8,
9B, and now Phase 9). Phase 9's contribution is not a new
state-of-the-art number — it's the first real, measured evidence about
whether naive staleness correction helps in this project's specific
small-scale simulated setting, and the honest answer this run gives is
"not noticeably." This closes out the master tracker's originally
separate "Staleness-Aware FL" (Phase 9) and "Resource-Aware Adaptation"
(Phase 13) rows, combined here since Phase 4's availability field is
the same underlying mechanism driving both.
