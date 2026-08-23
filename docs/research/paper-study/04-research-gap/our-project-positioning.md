# Our Project's Positioning in the Literature

## The single most important thing to understand

Your capstone's title — **"Deep Reinforcement Learning-Driven Adaptive Federated Framework for Satellite Network Anomaly Detection"** — is *very* close to Paper 4's title: **"Anomaly detection method for satellite networks based on adaptive federated learning driven by deep reinforcement learning" (FedSTAR)**, published in 2026, the same year as your capstone. This is not a coincidence to be nervous about — it's a sign your project is working on a genuinely active, current research direction. But it means **"we combine FL + DRL + satellite anomaly detection" is not, by itself, a defensible novelty claim** — FedSTAR already did that. Your differentiation has to be more precise than the title-level combination.

## Breaking your project into components, honestly

```text
Satellite Network  +  Anomaly Detection  +  Deep Learning  +  Federated Learning  +  Adaptive Mechanism  +  Deep Reinforcement Learning
```

| Component | What existing papers (in your set) already do | What they don't do | What your project actually does (verified from real code/docs, not assumed) |
|---|---|---|---|
| **Satellite network** | Papers 1–3: single real spacecraft telemetry. Paper 4: satellite-relevant *benchmark* traffic (STIN), not real captures | No paper has real, multi-satellite, federated data of either kind | Track A: 8 simulated satellite clients via Dirichlet-partitioned NSL-KDD (terrestrial data, explicitly disclosed as a simulation, not real satellite traffic). Track B: 1 real spacecraft (OPS-SAT), no FL |
| **Anomaly detection** | Papers 1–3: telemetry/physical-fault forecasting-based detection. Paper 4: network-traffic classification | No cross-domain paper (telemetry method + network-traffic-style FL) | Track A is classification-based network intrusion detection (like Paper 4's domain). Track B is classification-based telemetry detection (like Papers 1–3's domain, though with `sequence_length=1`, so not genuinely temporal in its current form — disclosed by your own project) |
| **Deep learning** | Papers 1–3: LSTM variants (plain, attention, Bayesian/MC-dropout). Paper 4: SimVPv2 + gSTA/LKA (heavy spatio-temporal) | — | BaselineMLP (Track A, 121→128→64→1); TemporalGRUAttention (Track B) — deliberately smaller/simpler than Paper 4's architecture |
| **Federated learning** | Paper 4: SAC-driven clustering + weighted aggregation across 20 clients | No telemetry-domain paper has FL at all | FedAvg (Phase 6) → rule-based adaptive weighting (Phase 8) → DQN-driven adaptive weighting (Phase 9B), across 8 clients, all real and already run |
| **Adaptive mechanism** | Paper 4: SAC picks clustering algorithm + continuous knobs every round | No paper explicitly compares a *fixed rule* against a *learned* adaptive mechanism on the identical pipeline, reporting both honestly | Your project explicitly ran and reported **both stages** (Phase 8 rule-based, Phase 9B learned) on the *same* underlying FedAvg pipeline and the *same* partition — a controlled, apples-to-apples comparison that isn't present in Paper 4 (which reports SAC directly, without also reporting a fixed-rule intermediate stage on the same setup) |
| **Deep reinforcement learning** | Paper 4: SAC (hybrid discrete+continuous action: clustering algorithm + cluster count + 3 continuous knobs) | No paper uses a small, purely-discrete, fully-named/explainable action catalogue | DQN, 9 fixed named weight-profiles (no clustering, no continuous outputs) — a deliberately narrower, more explainable, more capstone-appropriate action space |

## Novel idea vs. Engineering combination vs. Potential research contribution

Per the explicit instruction not to claim automatic novelty from combining existing pieces, here is the honest breakdown:

- **Not a novel idea:** "combine FL and DRL for satellite anomaly detection" — Paper 4 already published this combination, in the same year.
- **An engineering combination, done well:** wiring DQN-driven adaptive weighting on top of an already-working, tested FedAvg + rule-based-adaptive pipeline, with real privacy-boundary tests (AST-based) extended to every new module — solid, disciplined engineering, but "we built the pipeline correctly" is not, by itself, a research contribution.
- **A genuine, potential research contribution (small but real, and it needs to stay honestly scoped):** a **controlled, three-way, same-pipeline comparison** of fixed FedAvg vs. hand-written rule-based adaptive weighting vs. DQN-learned adaptive weighting, on the identical client partition and protocol, with the DQN's result reported honestly even though it did not win — including a specific, falsifiable hypothesis (training-partition diversity) for *why* it didn't win, which a follow-up experiment (varying partitions across pretraining episodes, as the original Phase 9B design intended but the completed run deliberately did not do) could directly test. This is a real, if modest, empirical contribution: most papers that introduce a DRL-based FL controller (including Paper 4) report it as a winning result — a rigorously reported *negative* result, with a specific testable explanation, is scientifically useful and comparatively rare in this space.

## What must still be experimentally validated (not yet proven)

Be explicit about these in any report/viva — do not present them as settled:

1. Whether a DQN-driven policy trained on **genuinely diverse** partitions (the original Phase 9B design intent, not yet executed) would outperform Phase 8's best rule — **untested**, only hypothesized.
2. Whether the DQN's loss collapsing to near-zero (~0.0001) reflects real convergence or a degenerate, near-constant Q-function — **explicitly flagged as unresolved** in your own project's own results report.
3. Whether any of this generalizes beyond NSL-KDD-as-simulated-satellite-clients to real satellite network traffic (like Paper 4's STIN) or to the telemetry domain (Track B) — **not tested**, and would require new experiments, not just new claims.
4. Robustness to Non-IID severity beyond what Phase 7's α-sweep already covered, specifically *for the DQN* (Phase 7's sweep was done for plain FedAvg, before Phase 9B existed) — **not yet run for the DQN specifically**.

## The one-sentence answer to "how is your project different from FedSTAR?"

> "FedSTAR uses a heavier SAC agent to choose a clustering algorithm and continuous aggregation hyperparameters for satellite *network-traffic* intrusion detection on benchmark (not real) satellite data. Our project uses a smaller, fully-explainable DQN to choose among 9 fixed, named aggregation-weighting profiles — no clustering — validated with a controlled three-stage comparison (fixed → rule-based → learned) on the same pipeline, and separately, honestly, validates a telemetry-style detector against one real spacecraft's actual data — a track FedSTAR has no equivalent of, though our two tracks are not yet combined into one federated real-satellite system either."
