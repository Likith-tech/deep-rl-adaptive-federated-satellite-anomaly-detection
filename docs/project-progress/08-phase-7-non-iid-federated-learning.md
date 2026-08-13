# Phase 7 — Non-IID Federated Learning Experiments

## What was the goal?

Phase 6 gave us one Federated Learning result, using one specific level
of data heterogeneity across our 8 satellites (alpha=0.5). But how
sensitive is FedAvg to *how different* the satellites' data actually
is? Phase 7 answers that by deliberately controlling and measuring the
degree of heterogeneity, instead of just accepting whatever the
original partition happened to produce.

## What is non-IID?

Non-IID means the satellites do not see the same distribution of
network traffic or attacks. One satellite might see mostly "dos"
attacks, another mostly normal traffic, another a rare mix of
everything. This is the realistic, hard case for Federated Learning —
the opposite of every device holding a random, representative sample of
the whole dataset.

## What did we change?

Only the **Dirichlet alpha** parameter, which controls how skewed the
per-satellite data split is. We tested five values: **0.1** (very
skewed/heterogeneous), **0.5** (Phase 6's original level), **1.0**,
**5.0**, and **10.0** (closest to evenly balanced). For each alpha, we
built a completely fresh 8-satellite partition of the same real
training data.

## What did we keep the same?

Everything else: the model (Phase 2's MLP), all 8 clients, 10
communication rounds, 1 local epoch per round, batch size 128, learning
rate 0.001, Adam optimizer, BCEWithLogitsLoss, sample-count-weighted
FedAvg, the same shared initial weights, the same global validation
set, and the same final KDDTest+ set. Alpha was the ONLY thing that
changed between experiments — this is what makes the comparison
scientifically meaningful.

## What happened?

| Alpha | Heterogeneity (mean JS distance) | Validation F1 | KDDTest+ F1 | Client fairness gap |
|---:|---:|---:|---:|---:|
| 0.1 | 0.6892 (most heterogeneous) | 0.9821 | 0.7409 | 0.2623 |
| 0.5 | 0.5270 | 0.9879 | 0.7404 | 0.1450 |
| 1.0 | 0.3419 | 0.9924 | 0.7555 | 0.0236 |
| 5.0 | 0.2063 | 0.9932 | 0.7640 | 0.0099 |
| 10.0 | 0.1452 (closest to IID) | 0.9924 | 0.7541 | 0.0064 |

As alpha increased (data became more balanced across satellites),
validation F1 and KDDTest+ F1 both rose — up to a point — and the gap
between the best-served and worst-served satellite ("client fairness
gap") shrank steadily and consistently across every single alpha step,
from 0.26 down to 0.006. Full numbers, round-by-round history, and
every plot are in `results/reports/non_iid_results.md`.

## Which alpha was most heterogeneous?

**Alpha=0.1**, measured (not assumed): mean pairwise Jensen-Shannon
distance of 0.6892 between satellites' local category distributions —
more than 4.7x higher than alpha=10.0's 0.1452. At this level, several
satellites ended up with zero examples of the rare "r2l" or "u2r"
attack categories, and the partition even failed our normal fairness
constraints (e.g. one satellite had zero "normal" traffic at all) —
reported honestly rather than hidden or artificially fixed.

## Which performed best/worst?

**Alpha=10.0 was the most heterogeneity-reducing setting** and gave
the smallest client fairness gap (0.0064) — every satellite was served
almost equally well by the shared model. But **alpha=5.0 actually
scored the highest validation F1 (0.9932) and the highest KDDTest+ F1
(0.7640)** — slightly better than alpha=10.0's KDDTest+ F1 of 0.7541.
We report this exactly as measured rather than picking whichever
looks tidiest: going from alpha=5.0 to alpha=10.0, heterogeneity kept
falling and fairness kept improving, but the overall KDDTest+ score
went down slightly. With only one run per alpha, we can't tell whether
this is a genuine saturation effect (diminishing/reversing returns
once data is already fairly balanced) or ordinary run-to-run noise —
more repeated trials would be needed to tell those apart. **Alpha=0.1
performed worst** on every measure: lowest validation F1, tied-lowest
KDDTest+ F1, and by far the worst fairness gap (0.2623).

## Did non-IID hurt FedAvg?

Based on what we actually measured: yes, in a specific and important
way. The overall KDDTest+ score changed only modestly and
non-monotonically across the tested range (0.7409 at alpha=0.1 up to a
peak of 0.7640 at alpha=5.0, then back down slightly to 0.7541 at
alpha=10.0) — but the **client fairness gap shrank steadily and
substantially at every step**, a ~41x difference between the most
heterogeneous (0.2623) and least heterogeneous (0.0064) settings
tested. In other words, standard FedAvg mostly still produces a
workable *average* model even under heavy heterogeneity, but some
individual satellites get left behind much more than others, and that
effect is far more consistent and dramatic than heterogeneity's effect
on the overall score. With only 5 tested alpha values, we describe this
as an observed pattern, not a proven causal law — but it's a real,
honest, useful finding.

## How do I explain this to my mam?

"In Phase 7, we studied how different data distributions across
satellites affect Federated Learning. We created five controlled
non-IID levels using Dirichlet partitioning while keeping the model and
FedAvg algorithm completely unchanged. We found that when satellites'
data is very different from each other, the shared model still mostly
works overall, but some satellites end up much worse served than
others — the gap between the best and worst satellite was about 41
times larger under heavy heterogeneity than under the most balanced
setting we tested. This tells us that just averaging everyone equally
isn't enough when satellites are very different, which is exactly the
problem Adaptive Federated Learning is meant to solve."

## Important terms

- **IID (Independent and Identically Distributed):** every client's
  data looks like a random, representative sample of the whole
  dataset — the easy, unrealistic case.
- **Non-IID:** clients' data comes from meaningfully different
  distributions — the realistic, harder case, and the whole point of
  this phase.
- **Dirichlet distribution:** the statistical tool used to control how
  skewed each satellite's data split is; its "alpha" (concentration)
  parameter is what we varied here.
- **Label skew:** the specific kind of non-IID-ness used here — clients
  differ in which attack categories they see, and how often.
- **Jensen-Shannon (JS) distance:** a number from 0 (identical
  distributions) to 1 (maximally different) that measures how different
  two satellites' data distributions are — used here to confirm alpha
  actually changed heterogeneity, not just assume it.
- **Client heterogeneity:** how different clients' local data is from
  each other — the core variable of this whole phase.
- **Client fairness gap:** the difference between the best-performing
  and worst-performing satellite's F1 score, evaluated using the same
  shared model — a measure of whether some satellites are being left
  behind.

## Limitations

- Only 8 clients and 5 alpha values were tested — a small, controlled
  study, not a large statistical sweep.
- NSL-KDD remains a terrestrial network intrusion dataset — not real
  satellite telemetry.
- The satellite/client environment is simulated, as established in
  Phase 4.
- The Dirichlet partitioning used to create different heterogeneity
  levels is itself a synthetic, controlled construction, not naturally
  occurring satellite data.
- Only standard, synchronous FedAvg was used — no adaptive client
  selection, no staleness handling.
- No DRL, DQN, or reinforcement learning of any kind was used in this
  phase.
- "Client fairness" here evaluates the shared model on each client's
  own local training data (there's no separate held-out per-client
  set) — a distributional-fit proxy, not a leakage-free generalization
  test.

## What comes next?

Adaptive Federated Learning: use what we learned here — that
heterogeneity mostly hurts *fairness* between satellites more than it
hurts the *overall* score — to explore smarter approaches than treating
every satellite identically every round.
