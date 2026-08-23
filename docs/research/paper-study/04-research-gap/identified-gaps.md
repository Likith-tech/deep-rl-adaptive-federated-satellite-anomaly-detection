# Identified Research Gaps (Evidence-Based, Across All 4 Papers)

**Method:** for every gap below, we require (1) evidence from the papers themselves, (2) why it matters, (3) how it connects to satellites/FL/RL/privacy/Non-IID specifically, and (4) whether your own project already addresses it, partially addresses it, or shares the same open gap. We do not invent a gap just to make the capstone sound novel — several gaps below are gaps your own project *also* has, and that is reported honestly.

---

## Gap 1 — No paper combines telemetry-style physical-fault detection with federated learning across multiple satellites

**Evidence:** Papers 1, 2, 3 are all single-spacecraft, centrally-trained (no client/server split at all). Paper 4 has federated learning, but for network-traffic intrusion detection, never telemetry.

**Why it matters:** a real satellite constellation has both concerns (physical health *and* network security) and, in principle, could benefit from sharing "what does a healthy/anomalous state look like" knowledge across satellites without shipping raw telemetry to the ground — but no paper in this set builds or evaluates that.

**Satellite/FL/RL/privacy/Non-IID relevance:** this is fundamentally a federated-learning-for-telemetry gap.

**Does your project address it?** **No — and this is honestly disclosed by your own project's documentation.** Your project's Track A (federated + DRL) uses NSL-KDD (terrestrial network intrusion data, not telemetry). Your project's Track B (real spacecraft telemetry, OPSSAT-AD) has no federated structure — a single spacecraft, explicitly stated in your own Phase 9A document: *"That this project's Federated Learning / adaptive-FL machinery (Phases 6-8) works on real multi-satellite data — only one real spacecraft's data was used, with no federated structure attempted here"* is listed under "What this does NOT prove." **This remains an open gap, not something to claim as solved.**

---

## Gap 2 — No paper uses reinforcement learning for telemetry-based (physical-fault) anomaly detection

**Evidence:** Papers 1–3 use no RL of any kind. Paper 4's RL (SAC) controls federated aggregation policy, not detection thresholds, and is applied to network-traffic classification, not telemetry forecasting.

**Why it matters:** Papers 1–3 all wrestle, in different ways, with the same core hard problem — "how do you decide the anomaly threshold without a human hand-tuning it" (nonparametric statistics in Paper 3, multi-scale combination in Paper 1, uncertainty regions in Paper 2). None of them frame this as something an RL agent could *learn* from a reward signal (e.g., a reward based on how well the chosen threshold trades off false positives vs. false negatives over time) — an unexplored angle.

**Does your project address it?** Partially, and in a different way than the gap describes: your project's DRL (Phase 9B) controls **federated aggregation weighting**, not a detection threshold — the same scope choice as Paper 4, not a novel extension into telemetry-threshold learning. This specific gap (RL for telemetry thresholding itself) remains genuinely open, in both the literature and your own project.

---

## Gap 3 — No paper validates federated + DRL machinery against real, operational satellite data of any kind (telemetry or network traffic)

**Evidence:** Paper 4 explicitly states in its own Conclusion: *"STIN is closer to satellite-terrestrial traffic than TON_IoT and Bot-IoT, but it still cannot replace operational satellite traces."* Papers 1–3 do use real telemetry, but never in a federated setting (no multi-client structure exists for real data to be tested against in a distributed way).

**Why it matters:** this is arguably the single largest practical gap in the entire field, not specific to any one paper — genuinely realistic, federated, multi-satellite operational data essentially does not exist as a public research resource yet.

**Does your project address it?** Partially, and honestly disclosed: Track B validates against real spacecraft telemetry (OPSSAT-AD) but with no FL; Track A runs the full FL+DRL pipeline but on simulated (Dirichlet-partitioned NSL-KDD) clients, not real satellite data of either kind. **This is a shared, field-wide limitation your project inherits rather than solves — a defensible thing to say plainly in your report, not something to paper over.**

---

## Gap 4 — Robustness to malicious/compromised (Byzantine) clients is essentially untouched

**Evidence:** Paper 4 states directly, twice: *"The adversarial robustness of the federated optimization process itself is left for future investigation."* Papers 1–3 have no client concept for this to apply to.

**Why it matters:** in any real federated deployment (satellite or otherwise), you must assume some clients could be compromised, faulty, or actively malicious — an attacker who controls one satellite's transmitted model update could otherwise poison the shared global model.

**Does your project address it?** No — no phase of your project tests against an adversarial or Byzantine client. This is a legitimate, currently-open direction shared with Paper 4, not a gap your project currently fills.

---

## Gap 5 — Orbital-realistic networking conditions (delay, intermittent contact, staleness) are absent from every DRL/FL satellite paper found so far

**Evidence:** Paper 4's own Conclusion: *"The current setting also simplifies orbital networking conditions. Factors such as delayed updates, intermittent links, stale client states, and time varying link availability are not fully captured."*

**Why it matters:** real satellites don't have an always-on connection to a central server the way most FL research (including Paper 4 and your project) assumes — this is a genuine, satellite-specific gap that generic FL research (built for phones/hospitals) doesn't need to solve.

**Does your project address it?** No. Your project's federated rounds (Phases 6–9B) assume synchronous, always-available client participation (all 8 clients every round, by design, for fairness reasons) — a simplification your own project shares with Paper 4, not something either has solved.

---

## Gap 6 — A single, fixed-rule-based aggregation weighting is not simultaneously optimal on every metric (F1 and fairness) — but whether a *learned* policy reliably does better remains unresolved

**Evidence:** this is your **own project's own measured finding**, not something taken from any of the 4 papers: Phase 8's ablations showed `resource_only` (not `combined`) won on F1, while `performance_only` (not `combined`) won on fairness — no single fixed rule is best at everything. Then Phase 9B's DQN, tested precisely to see if a *learned* policy could do better than the best fixed rule, did **not** beat it on either F1 or fairness in the one completed run.

**Why it matters:** this is a genuinely valuable, honest, negative-but-informative result — it suggests that simply adding a DRL controller is not automatically an improvement, and that the *amount and diversity* of training experience (your project's own disclosed limitation: all 750 pretraining rounds used the same single fixed data partition, not the varied partitions the original design intended) likely matters as much as the choice of algorithm.

**Does your project address it?** This gap essentially **is** your project's own central, already-executed experiment — and its answer, honestly reported, is "not yet, and here's the most likely reason why (limited training diversity)." This is the most scientifically mature, most defensible finding your project currently has, and should be framed as exactly that in a report or viva — not hidden, not spun as a success.

---

## Summary — the strongest, most defensible gaps to build a "why this capstone" argument around

1. **Gap 1 + Gap 3 together**: no existing work bridges federated telemetry-style physical-fault detection with real spacecraft data — a field-wide, not paper-specific, absence. Your project's two-track structure (Track A federated+DRL / Track B real-telemetry validation) is a reasonable, honestly-scoped step *toward* this, explicitly not a claim of having closed the gap.
2. **Gap 6**: a rigorously executed, honestly-reported experiment testing whether a *narrow, explainable* DQN-based adaptive aggregation policy can beat a well-tuned fixed rule — with a real, disclosed negative result and a specific, testable hypothesis for why (training-data diversity) that a future run could investigate.
3. **Gap 4 + Gap 5**: both genuinely open, both explicitly named as future work by the most relevant prior paper (Paper 4) — legitimate to name as shared, unsolved ground rather than differentiators.
