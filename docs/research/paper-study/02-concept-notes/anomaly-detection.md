# Concept — Anomaly Detection (and Intrusion Detection)

## 1. What is it?

**Anomaly detection:** finding data points (or time periods) that don't match the pattern of "normal" behavior a system has learned or been told to expect.

> **Technical term → simple meaning → example**
> **Anomaly detection:** teaching a model what "normal" looks like, then flagging anything that deviates from it by enough.
> *Analogy:* if you know exactly how your car normally sounds, a new rattling noise stands out immediately — you don't need to have heard that exact rattle before to know something's wrong.

**Intrusion Detection System (IDS):** the network-security specific flavor of anomaly detection — looking for signs of an attacker in network traffic rather than a physical fault in a machine. Paper 4 (FedSTAR) is an IDS-style system; Papers 1–3 are telemetry-fault-style systems (see `satellite-networks.md` for the full distinction).

## 2. Why does it exist?

You usually don't have labeled examples of every possible failure or attack in advance (new failures and new attacks happen), so instead of training a classifier on "here are all the bad things," you train a model on "here is what normal looks like," and treat large deviations as suspicious. This is why 3 of your 4 papers use a **forecasting-based** approach (predict what should happen, compare to what did happen) rather than a plain classifier.

## 3. Simple example

A telemetry channel normally sits between 18–22°C. A forecasting model predicts the next value should be ~20°C; the real reading comes in at 35°C — a large residual (predicted minus actual) triggers an anomaly flag.

## 4. How it works — the two main strategies your papers use

- **Forecasting-based (Papers 1, 2, 3):** predict the next value from recent history; large prediction error = anomaly. Requires deciding a *threshold* for "how large is too large" — this is the hard, genuinely research-worthy part (Paper 3's whole contribution is a better way to choose this threshold automatically).
- **Classification-based (Paper 4, and your project's own NSL-KDD-based models):** directly classify each sample as normal or one of several attack categories, using labeled training data (supervised learning) — this works when labeled attack data of the relevant type is actually available, which is more realistic for network intrusion detection than for rare physical spacecraft faults.

## 5. Why the papers use their chosen strategy

Telemetry papers (1–3) prefer forecasting-based detection because physical/operational anomalies are rare and hard to label exhaustively in advance — you can't train a classifier on failure modes you've never seen. Paper 4 uses classification because network-attack datasets (STIN/TON_IoT/Bot-IoT) *do* provide labeled attack categories, making supervised classification practical and more precise (you get to know *which* attack type, not just "something's wrong").

## 6. Advantages

- Forecasting-based: works even without labeled anomalies; interpretable (you can literally show "predicted vs. actual" and the gap).
- Classification-based: more precise (tells you *what kind* of anomaly), usually higher accuracy when labels are genuinely available.

## 7. Limitations

- Forecasting-based: choosing the threshold is genuinely hard (this is most of what Papers 1–3 improve on, one after another); can still miss anomalies that happen to look like normal variation.
- Classification-based: only as good as its labeled training data — cannot detect a genuinely novel attack type it has never seen labeled examples of (a real limitation of Paper 4's approach, and of your project's own NSL-KDD-based Track A).

## 8. Which papers use it

All 4 — Papers 1–3 forecasting-based; Paper 4 classification-based.

## 9. How it connects to our project

Your project's Track A (NSL-KDD) is classification-based, matching Paper 4's style. Your project's Track B (OPSSAT-AD, Phase 9A) is *also* classification-based in its current implementation (segment-level classification with `sequence_length=1`, per your project's own honest documentation), even though its architecture (`TemporalGRUAttention`) was originally built with forecasting/sequence-modeling in mind — worth knowing precisely for viva, since it's a subtle but important detail your own project's documentation is careful about.
