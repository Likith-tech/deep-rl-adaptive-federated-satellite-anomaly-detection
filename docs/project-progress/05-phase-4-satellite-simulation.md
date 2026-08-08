# Phase 4 — Simulated Satellite Network & Non-IID Client Environment

**Scope of this phase, stated plainly:** this phase builds a
**simulation of a multi-satellite learning environment using a real
network intrusion dataset.** NSL-KDD is not, and has never been claimed
to be, satellite traffic. The satellite IDs, bandwidth, latency,
compute capacity, availability, and connectivity introduced here are
all **simulated** experiment parameters, clearly labeled as such
everywhere they appear. No Federated Learning, DRL, or model training
happens in this phase — this is environment construction only.

## What was the goal?

We need to turn our single network dataset into multiple simulated
satellite clients so that Federated Learning will have something
realistic to operate on. Up to now, every model (baseline, temporal)
trained on one big pile of data in one place. Federated Learning is
about training across many separate clients that each only see their
own local data — so first we need those clients to actually exist.

## What did we build?

- **Satellite clients** (`src/simulation/satellite_client.py`): a
  reusable `SatelliteClient` representation — an ID (`SAT-01`, `SAT-02`,
  ...) plus simulated resource conditions plus how much real data it
  holds.
- **Data partitioning** (`src/simulation/partitioner.py`): splits the
  real training data across clients using a **Dirichlet non-IID
  strategy** — see "Important terms" below. Every real record ends up
  on exactly one client; nothing is lost, duplicated, or invented.
- **Non-IID measurement** (`src/simulation/non_iid_metrics.py`): rather
  than just asserting the partition is non-IID, we actually measured it
  (Jensen-Shannon divergence, entropy, variance — see real numbers
  below).
- **Satellite conditions** (`src/simulation/network_conditions.py`):
  simulated bandwidth, latency, compute capacity, availability, and
  connectivity quality for each client, randomly sampled within
  configured ranges, reproducibly (fixed seed).
- **Orchestration** (`src/simulation/simulator.py` +
  `scripts/create_satellite_partitions.py` +
  `scripts/analyze_satellite_partitions.py`): ties the above together,
  writes the per-client datasets, a manifest, satellite metadata, plots,
  and a full report.

## Why is this important?

A single centralized model (like our Phase 2/3 models) can't
demonstrate anything about Federated Learning — there's only one
"client," so there's nothing to federate. To honestly build and later
evaluate FL, Adaptive FL, and DRL-based client selection, we first need
a believable multi-client environment where clients:

- hold different amounts of data,
- see different mixes of traffic (non-IID — this is what makes FL hard
  and interesting in practice, not a simplified toy case), and
- have different, imperfect operating conditions (some slower, some
  less available) — the actual thing DRL-based client selection will
  eventually have to reason about.

## What is simulated?

- Satellite IDs (`SAT-01`...`SAT-08`) — labels we assigned, not real
  satellite designations.
- Bandwidth, latency, compute capacity, availability, connectivity —
  all randomly generated within configured ranges
  (`configs/satellite_simulation.yaml`), not measured from any real
  hardware or orbital link.
- The very fact that this data represents "satellites" at all — it's a
  simulation layer on top of ordinary network intrusion data.

## What is real?

- Every row of data each client holds is a **real** NSL-KDD training
  record with its real label — nothing invented.
- The imbalance in how attack types are naturally distributed in
  NSL-KDD (e.g. very few R2L/U2R records overall) is real and carries
  through into the simulation.
- The measured non-IID statistics (divergence, entropy, variance) are
  computed from the actual generated partition, not assumed.

## Actual results

**8 simulated clients**, Dirichlet partitioning (alpha=0.5, seed=42) of
**107,077** real training records:

| Client | Samples | Anomaly % | Dominant category |
|---|---:|---:|---|
| SAT-01 | 8,675 | 99.1% | dos |
| SAT-02 | 39,030 | 63.9% | dos + normal |
| SAT-03 | 6,813 | 16.2% | normal |
| SAT-04 | 6,246 | 81.9% | probe |
| SAT-05 | 8,382 | 39.2% | normal + probe |
| SAT-06 | 4,819 | 43.8% | mixed |
| SAT-07 | 19,144 | 17.5% | normal |
| SAT-08 | 13,968 | 9.7% | normal |

Non-IID, measured directly (not assumed): average pairwise
Jensen-Shannon distance between clients' category distributions =
**0.527** (0 = identical, 1 = maximally different), with per-client
entropy ranging from **0.16 bits** (SAT-01, heavily dos-dominated) to
**1.52 bits** (SAT-06, most balanced). Full numbers, resource
statistics, and every client's individual values are in
`results/reports/satellite_simulation_report.md`.

## How do I explain this to my mam?

"We took the real network intrusion dataset we prepared earlier and
divided the training data among 8 simulated satellites, using a
statistical method (Dirichlet partitioning) so each satellite sees a
genuinely different mix of traffic — some see mostly one kind of
attack, others see mostly normal traffic. We didn't just assume this
was non-IID — we measured it. We also assigned each satellite simulated
bandwidth, latency, and computing conditions, clearly labeled as
simulated, not real hardware measurements. We're doing this so that in
the next phase we can train models locally on each satellite, and
eventually use Federated Learning to combine what they learn without
sharing raw data."

## Important terms

- **Client:** one simulated satellite — an ID, its own slice of real
  data, and its own simulated operating conditions.
- **Non-IID (Not Independent and Identically Distributed):** each
  client's data comes from a different distribution than the others —
  the opposite of every client getting a random, representative sample
  of everything. This is the normal, expected case for real distributed
  devices/satellites, and it's what makes Federated Learning genuinely
  hard.
- **Partition:** dividing the dataset into disjoint pieces, one per
  client.
- **Dirichlet distribution:** a probability distribution over
  "proportions that add up to 1." We use it to randomly decide, for
  each attack category separately, what fraction of that category's
  records go to each client — a small "concentration" parameter
  (alpha) makes the split more skewed/non-IID; we used alpha=0.5.
- **Bandwidth:** simulated data-transfer capacity (megabits/second).
- **Latency:** simulated one-way communication delay (milliseconds).
- **Availability:** simulated probability a client is reachable during
  a given round of communication.
- **Compute capacity:** a simulated, normalized (0-1) score for how
  much local processing power a client has.

## Limitations

- Every satellite condition (bandwidth, latency, compute, availability,
  connectivity) is a simulated number sampled from a configured range —
  not derived from real orbital mechanics, real link budgets, or real
  satellite hardware.
- NSL-KDD remains a terrestrial dataset; nothing here changes that.
- Dirichlet partitioning creates realistic *label* skew, but doesn't
  simulate genuine geography, orbital coverage, or inter-satellite
  links.
- Rare attack categories (R2L, U2R) have very few real records in the
  whole dataset, so some clients legitimately end up with zero or very
  few of them — a property of the real data, not a bug in the
  partitioning.
- No Federated Learning, DRL, or model training happens yet — this
  phase only built the environment those will run on.

## What comes next?

Phase 5 (Local Satellite Training): train a model independently on each
satellite's own local, real data — still no communication between
clients, no aggregation, no global model. That establishes what each
client can learn on its own, which later Federated Learning results
will be compared against.
