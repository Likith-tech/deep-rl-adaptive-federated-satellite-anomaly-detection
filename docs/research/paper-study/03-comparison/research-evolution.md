# Research Evolution — How These 4 Papers Actually Relate

## The honest picture: two separate lineages, not one straight line

A tempting but **incorrect** way to describe your 4 papers is a single, tidy progression like "Traditional ML → Deep Learning → Federated Learning → DRL." That is not what actually happened. Your 4 papers form **two separate, mostly disconnected research lineages** that happen to share the word "satellite" and "anomaly detection":

```text
LINEAGE A — Single-spacecraft telemetry forecasting-based anomaly detection
(no FL, no RL, in any of these three papers)

Hundman et al. 2018 (Paper 3)
  "Predict the next telemetry value with an LSTM; decide the anomaly
   threshold automatically using a nonparametric statistical method,
   not a fixed number."
        │
        │  (same SMAP/MSL benchmark reused directly)
        ▼
Yang et al. 2021 (Paper 1)
  "Improve the LSTM predictor with attention (better use of long-range
   history) and add a SECOND detection time-scale (global, not just
   local) to catch slow-developing anomalies the first paper's local
   window would miss."
        │
        │  (same SMAP/MSL benchmark reused again)
        ▼
Maleki Sadr et al. 2022 (Paper 2)
  "Instead of (or alongside) improving the point-prediction accuracy,
   give the model a genuine UNCERTAINTY estimate (via Monte Carlo
   Dropout), so detection is based on 'did the real value fall outside
   my confidence region' rather than just 'is the error big'."


LINEAGE B — Satellite-network intrusion detection with FL + DRL
(a completely separate research thread; does not cite or build on
 Lineage A's dataset, method, or authors)

Chen et al. 2026 (Paper 4, FedSTAR)
  "Detect cyberattacks in satellite network traffic using federated
   learning across many satellite clients (never share raw traffic),
   with an SAC reinforcement-learning agent on the server deciding how
   to cluster and combine clients, because plain averaging fails under
   Non-IID client data."
```

**What actually evolved within Lineage A:** the *prediction* model got progressively fancier (LSTM → LSTM+attention → Bayesian LSTM with uncertainty), and the *detection* logic got progressively smarter about thresholds (fixed-window statistical threshold → multi-scale threshold → confidence-region + consecutive-point post-processing). What **never appeared** anywhere in Lineage A: multiple satellites, federated training, or any reinforcement learning.

**What Lineage B introduces that Lineage A never touches at all:** a multi-client (federated) setting, Non-IID data handling, and a reinforcement-learning agent making structural decisions about *how training itself proceeds* (not about detection thresholds).

## Why this matters for how you frame your literature review

If you present these 4 papers as one smooth evolutionary line, a sharp examiner will notice the jump from "single spacecraft, no FL, no RL" straight to "federated + DRL, but a totally different kind of anomaly and no real satellite data" is not a natural progression — it's a **field-level gap**, and that gap is genuinely where your capstone's contribution needs to be positioned (see `04-research-gap/identified-gaps.md`).

The honest, defensible framing is:

> "The literature splits into two streams that don't talk to each other: mature, single-spacecraft telemetry forecasting-based detection with no distributed-learning structure (Papers 1–3), and an emerging, much newer stream (Paper 4, published the same year as this capstone) that combines federated learning and DRL, but for network-traffic intrusion detection rather than physical telemetry, and without real satellite data. This capstone sits between the two: it builds a working federated + DRL pipeline (closer to Lineage B in mechanism) while also, separately and honestly, validating a telemetry-style detector against one real spacecraft's data (closer to Lineage A's domain, via OPSSAT-AD) — without yet merging the two tracks into one system."

## A note on timing

Paper 4 (FedSTAR) was received in April 2026 and published online in June 2026 — essentially concurrent with, not predating by years, work of this kind. This means "nobody has combined FL + DRL + satellite anomaly detection before" is **not a defensible novelty claim** for your capstone (FedSTAR already did it, in the same year). What *is* defensible is a precise, narrower differentiation — see `04-research-gap/our-project-positioning.md`.

## What did NOT evolve across any of these 4 papers

- **No paper in your set uses real, multi-satellite federated data of any kind.** Every multi-client setting in Paper 4 (and in your own project's Track A) is a *simulated* partitioning of a centralized, non-satellite-specific-in-origin dataset. This is a persistent, field-wide gap, not something later papers "solved" — it's simply not solvable yet without new, currently-nonexistent public data.
- **No paper addresses adversarial/malicious clients.** Paper 4 explicitly defers this to future work; Papers 1–3 have no client concept for this to even apply to.
