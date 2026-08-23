# Concept — Adaptive Learning / Adaptive Federated Learning

## 1. What is it?

> **Technical term → simple meaning → example**
> **Adaptive (federated) learning:** instead of combining every client's update with a fixed, unchanging rule (like plain FedAvg's "weight by data volume, always"), the system changes *how* it combines clients based on current conditions — who's currently performing well, who has scarce/valuable data, who has good connectivity, etc.
> *Analogy:* a fixed rule is like a teacher who always grades group projects by "average of everyone's score, no exceptions." An adaptive rule is a teacher who, this week, gives more weight to the group members who contributed the most useful ideas — and might weight differently next week if circumstances change.

**Important distinction your own project draws explicitly (and you should too): rule-based adaptive vs. learned (RL-based) adaptive.**

| | Rule-based adaptive | Learned (RL-based) adaptive |
|---|---|---|
| How the weighting is decided | A human writes a formula/rule ahead of time (e.g., `0.40*performance + 0.25*data + 0.15*resource + 0.20*fairness`) | An agent learns, from a reward signal over many rounds, which weighting tends to work best |
| Example in your project | Phase 8 (`src/federated/adaptive.py`) | Phase 9B (`src/drl/`, DQN) |
| Example in your papers | Not present in any of your 4 papers as a *separate* stage — Paper 4 goes straight to RL-based | Paper 4 (SAC) |

## 2. Why does it exist?

Because plain FedAvg (fixed, data-volume-only weighting) performs poorly under Non-IID conditions (see `non-iid-data.md`) — some signal about *which clients to trust more right now* needs to enter the aggregation decision, and a single fixed formula can't react to changing round-to-round circumstances the way a rule that's periodically re-evaluated (or fully learned) can.

## 3. Simple example

Round 5: client SAT-03 has been training on unusually noisy/corrupted-looking data this round (low local validation performance) — an adaptive system reduces its aggregation weight this round. Round 12: SAT-03's data quality recovers — an adaptive system restores its weight. A fixed FedAvg rule would treat SAT-03 identically in both rounds (only ever considering its sample count).

## 4. How it works (your project's actual two-stage approach)

```text
Phase 6  (FedAvg, fixed):         weight_k = n_k / N               (sample count only)
Phase 8  (rule-based adaptive):   weight_k = f(performance_k, data_k, resource_k, fairness_k)
                                   with FIXED, hand-chosen coefficients (0.40/0.25/0.15/0.20)
Phase 9B (learned adaptive):      weight_k = f(...) where the COMBINATION RULE ITSELF
                                   (which of 9 named profiles to apply) is chosen by a
                                   trained DQN agent, round by round
```

## 5. Why the papers/project use it

Your own project's Phase 8 ablation results are the clearest possible motivation for *why* adaptive weighting (of any kind) matters here: `resource_only` reached the best F1 (0.7547, beating plain FedAvg's 0.7404), while `performance_only` — not the "balanced" `combined` rule — reached the best fairness gap (0.1245). **A single fixed rule cannot be best at everything simultaneously** — this observation is exactly what motivated trying a *learned* (Phase 9B) approach instead of just picking one more fixed rule.

## 6. Advantages

- Rule-based: fully transparent, cheap, no training cost, immediately explainable ("we weighted resource availability 15%, because...").
- Learned (RL-based): can in principle discover better, context-sensitive combinations no human explicitly wrote down.

## 7. Limitations

- Rule-based: the weighting coefficients are still a human guess, chosen *before* seeing results — Phase 8's own result (fairness-weighted `combined` rule losing on fairness to `performance_only`) shows this guess can be wrong in surprising ways.
- Learned (RL-based): needs a lot of training experience, isn't guaranteed to beat a good fixed rule (your project's own Phase 9B result confirms this directly), and its behavior can be harder to fully explain after the fact (though your project's design explicitly keeps this explainable by using named, inspectable weight-profiles rather than an opaque continuous vector).

## 8. Which papers use it

Only Paper 4 uses adaptive federated learning (going straight to the RL-based/SAC stage, no separate rule-based baseline reported in the paper). None of Papers 1–3 have any federated structure to adapt.

## 9. How it connects to our project

This is arguably the **core intellectual arc of your entire capstone**: Phase 6 (fixed) → Phase 8 (rule-based adaptive) → Phase 9B (learned/DRL adaptive), each one built directly on top of the previous phase's unmodified code, with each phase's result honestly reported — including the fact that the most "advanced" stage (Phase 9B) did not outperform the simpler Phase 8 rule in its one completed run. This progression, and being able to explain *why* each step was taken and what each step's real, measured result was, is likely to be one of the most-asked topics in your viva.
