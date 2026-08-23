# Paper 4 — Anomaly Detection Method for Satellite Networks Based on Adaptive Federated Learning Driven by Deep Reinforcement Learning (FedSTAR)

> ⚠️ **This is the single most important paper in your set.** Its title is almost the same as your capstone's title, and it is the only one of the four papers that actually combines satellite anomaly detection + federated learning + deep reinforcement learning in one system. Read Section 14 and Section 16 of this note very carefully before your viva — you need a clear, honest answer ready for "how is your project different from this paper?"

**Authors:** Hui Chen, Kun Zhang, Yangtao Chen, Xiaohe Wu, Chenxi Cai, Dianying Chen (Nanjing University of Science and Technology / Nankai University, China)
**Year:** 2026 (received April 2026, accepted June 2026, published online 19 June 2026)
**Venue:** *Information Sciences*, Vol. 755 (Elsevier journal, article 123808)
**Source file:** `Anomaly detection method for satellite networks based on.pdf`

---

## 1. One-Minute Explanation

This paper (called **FedSTAR**) builds a system to detect **cyberattacks** in satellite network traffic — things like DDoS attacks, botnets, and port scans aimed at satellite communication links — while satellites train locally (federated learning) instead of sending raw traffic to a central server. Two things make FedSTAR different from plain federated learning: **(1)** it treats each satellite/client's 1D network-traffic data as if it were a tiny video (a "pseudo-video"), so a spatio-temporal deep learning model (based on an architecture called SimVPv2, using a technique called Large Kernel Attention) can find patterns in both the shape of the traffic and how it changes over time, together — instead of processing space and time separately, like most prior work does. **(2)** Instead of just averaging all clients' models together every round (plain FedAvg), a **Soft Actor-Critic (SAC) reinforcement learning agent** sits on the server and, every communication round, decides: which clustering algorithm to group similar clients into, how many clusters to use, what fraction of clients to sample, how to weight the aggregation, and how much to blend cluster-specific models back into the global model. This matters because real satellite clients have very different (non-IID) traffic patterns — plain averaging blurs these differences away and hurts performance; SAC learns, from a reward based on detection accuracy, how to group and combine clients more intelligently instead of using one fixed rule.

---

## 2. The Problem

- Satellite networks (as part of 6G-era Low Earth Orbit constellations) are exposed to real cyber threats: DDoS, botnets, malware, port scanning, protocol-specific exploits — the paper specifically mentions satellite-specific DDoS attacks that exploit the Delay-Tolerant Networking (DTN) protocol used for inter-satellite communication.
- Sending all raw traffic to a central server for analysis is impractical: satellite-ground links have limited, expensive bandwidth, and raw traffic could contain sensitive information (privacy risk).
- **Federated Learning (FL)** solves the "don't send raw data" problem by having each satellite train locally and only share model updates — but plain FedAvg (simple averaging) breaks down when clients' data distributions are very different (**Non-IID**), which is realistic here because different satellites see different traffic patterns depending on orbit, mission, and region.
- Prior spatio-temporal deep learning approaches (e.g., combining a CNN with an LSTM) usually process space and time **one after another** ("serial" processing) — the paper argues this loses subtle interactions between spatial and temporal patterns that a more coupled model could catch.

**Simple example:** Imagine 20 satellites, each seeing different mixes of normal vs. attack traffic (some see mostly DDoS attempts, others see mostly port scans). If you just average all 20 satellites' models equally every round, you get a "blurry compromise" model that isn't great at detecting *any* specific attack type well. FedSTAR instead groups similar satellites together (clustering) and combines them more thoughtfully.

---

## 3. What Is the Main Idea?

```text
CLIENT SIDE (each satellite):
1D network traffic features
        ↓
Pseudo-video projection (reshape into T=4 time-steps × 64 channels,
        treating traffic like a tiny video)
        ↓
Encoder (ConvSC: Conv2D + GroupNorm + SiLU) — extracts spatial patterns
        ↓
Translator (gSTA blocks using Large Kernel Attention) — extracts
        coupled spatio-temporal patterns, both dimensions together
        ↓
Classifier head (pooling + MLP) → attack label (which category, or normal)


SERVER SIDE (once per communication round):
Classifier-head parameters uploaded by every client
        ↓
Compute pairwise similarity matrix between clients (how alike are
        their models right now?)
        ↓
SAC (Soft Actor-Critic) agent observes this state, decides:
  - which clustering algorithm to use (FedAvg / K-Means / Spectral /
    Agglomerative / DBSCAN)
  - how many clusters
  - what % of clients to sample per cluster
  - how to weight aggregation (data-volume-weighted vs. uniform)
  - how much to blend the new cluster models into the old global model
        ↓
Group similar clients → aggregate within each cluster → fuse into
        one updated global model
        ↓
Reward = (accuracy gain) − (performance-drop penalty) − (cluster
        divergence penalty) → used to keep improving the SAC agent
```

---

## 4. Dataset

Three datasets, used for different purposes:

1. **STIN** — a flow-based network security benchmark **specifically built for Satellite-Terrestrial Integrated Networks**. 32 features, 209,564 instances, 9 attack categories (7 general network attacks like Botnets/Web Attacks/Backdoors/DDoS, plus **2 satellite-specific DDoS attacks** that target the DTN protocol used between satellites). This is the paper's primary, most satellite-relevant dataset.
2. **TON_IoT** — a general IoT/edge-computing cybersecurity benchmark (44 features, 211,043 instances, 10 attack categories: DoS/DDoS, scanning, backdoors, MITM, ransomware, etc.). Used as an **auxiliary** benchmark to test whether FedSTAR handles Non-IID heterogeneity well in general, not because it's satellite data.
3. **Bot-IoT** — a large-scale botnet/IoT intrusion dataset (46 features, over 73 million instances, 5 attack categories). Also used as an auxiliary benchmark, chosen for its scale and its flooding/probing/botnet attack styles.

**Important, explicitly stated by the paper itself:** none of these are real, operational satellite traffic captures. STIN is the closest (purpose-built for satellite-terrestrial scenarios) but is still a benchmark dataset, not live orbital traffic — the paper's own Conclusion admits this directly: *"STIN is closer to satellite-terrestrial traffic than TON_IoT and Bot-IoT, but it still cannot replace operational satellite traces."*

**What one "sample" represents:** one network traffic flow record (like a row in a network intrusion dataset), with numeric/categorical features describing that flow, labeled as normal or one of several attack categories — **not** a spacecraft telemetry reading. This is a fundamentally different kind of "anomaly" from Papers 1–3 in your set (malicious traffic vs. physical/operational fault).

**Client simulation:** 20 clients, data split either IID (random uniform) or Non-IID (Dirichlet distribution, concentration parameter η=0.5) — a synthetic partitioning of a centralized dataset into simulated clients, methodologically similar to how your own project's Track A (NSL-KDD) simulates 8 satellite clients via Dirichlet partitioning.

---

## 5. Features / Input Data

| Feature | What it represents | Why it matters |
|---|---|---|
| Network flow features (32–46 depending on dataset) | Statistics of a network connection/flow (protocol, size, timing, etc.) | The raw signal used to distinguish normal vs. attack traffic |
| Classifier-head parameters (uploaded to server) | A compressed summary of what each client's model has learned recently | Used to measure client similarity — cheaper and more targeted than comparing entire models, since the paper found final/classifier layers reflect data heterogeneity better than early layers |
| Cosine-similarity matrix (server state) | Pairwise "how alike are these two clients' models right now" | This is literally what the SAC agent "sees" (its state) each round |

---

## 6. Methodology

```text
Step 1 → Each client trains its local spatio-temporal detector for
          several local epochs (SGD)
Step 2 → Clients upload their classifier-head parameters (not the
          whole model) to the server
Step 3 → Server computes a pairwise similarity matrix between all
          clients' classifier heads
Step 4 → This matrix + recent loss/accuracy history + previous
          action becomes the SAC agent's STATE
Step 5 → SAC picks an ACTION: {clustering algorithm, #clusters,
          client sampling ratio, aggregation weight factor,
          global-local fusion coefficient}
Step 6 → Clients are grouped into that many clusters using the
          chosen algorithm
Step 7 → Within each cluster: sample a subset of clients, aggregate
          their models with the chosen weighting rule
Step 8 → Fuse each cluster's model back into ONE global model using
          the fusion coefficient
Step 9 → Evaluate the new global model's accuracy/loss → compute
          REWARD → SAC agent updates itself (learns) using this
          reward, over many communication rounds
Step 10 → Repeat for the fixed number of rounds
```

---

## 7. Model / Algorithm

### Soft Actor-Critic (SAC)

**Technical term → simple meaning → example**
> **Soft Actor-Critic (SAC):** A deep reinforcement learning algorithm that learns a policy (what action to take) by balancing "get a high reward" against "keep exploring, don't lock into one behavior too early" — it works with both discrete choices (like picking an algorithm from a list) and continuous values (like a sampling ratio between 0 and 1), which is why it fits this paper's *hybrid* action space (algorithm choice + numeric knobs, together).
> *Analogy:* Imagine a manager deciding every week how to organize employees into project teams. SAC is like a manager who both tries what's worked well before (exploit) and occasionally experiments with a new grouping (explore), gradually getting better at the decision over many weeks — rather than a manager who fixes the same grouping rule forever.

**Why the authors used SAC specifically (and not, e.g., a simpler DQN):** the action includes both a *discrete* choice (which clustering algorithm — 5 options) **and** *continuous* values (sampling ratio, aggregation weight, fusion coefficient — each a real number between 0 and 1). SAC natively supports this kind of mixed/hybrid action space; a plain DQN (like your own project uses) is a **discrete-only** method and cannot output continuous numbers directly.

**What goes in / what comes out:** in — the client-similarity matrix + training-history features (a numeric state vector); out — a clustering-algorithm choice, a cluster count, and three continuous control values.

### SimVPv2 + gSTA + Large Kernel Attention (LKA)

**Simple explanation:** the "pseudo-video" trick turns each traffic sample into a small stack of feature maps across a few fake "time frames," so a video-style deep learning model (originally built for predicting how video frames evolve) can be reused for network traffic. Inside this model, LKA approximates the effect of a very large convolution filter (a filter that can "see" far across the whole feature map) using a cheaper combination of smaller convolutions — giving the model long-range pattern recognition without the huge computational cost a literally-large filter would need.

**Why it matters for your project:** this is a genuinely more complex architecture (multiple stacked ConvSC + gSTA/LKA blocks) than anything used in Papers 1–3 or in your own current model architectures (BaselineMLP, TemporalGRUAttention) — useful to understand as "what a heavier, more specialized spatio-temporal architecture for network traffic looks like," even though your project intentionally does not need to match this complexity, since it uses simpler models by design for explainability and capstone-appropriate scale.

### DRL-CFL (the paper's overall mechanism name)

The paper's own name for combining SAC-driven clustering + aggregation-policy learning into the federated loop — Algorithm 1 in the paper.

---

## 8. Architecture

```text
   Client 1 ──┐
   Client 2 ──┤   (each: pseudo-video → ConvSC encoder →
   Client 3 ──┼──  gSTA/LKA translator → classifier head)
     ...      │
   Client 20 ─┘
        │  (classifier-head params only, every round)
        ▼
 ┌───────────────────────────────────────────┐
 │              SERVER (SAC agent)             │
 │  similarity matrix → SAC state              │
 │  SAC action → clustering algo + #clusters +  │
 │  sampling ratio + agg. weight + fusion coef. │
 │  → cluster clients → intra-cluster weighted  │
 │  aggregation → global-local fusion           │
 └───────────────────────────────────────────┘
        │
        ▼
   Updated global model → sent back to all clients
```

---

## 9. Results

**Non-IID accuracy (Table 3), η=0.5 Dirichlet, best result per dataset:**

| Dataset | FedSTAR | Best baseline (name) |
|---|---:|---|
| STIN | **95.43%** | FLOGA-AD: 94.87% |
| TON_IoT | **89.16%** | FLOGA-AD: 88.98% |
| Bot-IoT | **99.88%** | FLOGA-AD: 98.01% |

- FedSTAR beat all 7 baselines (CNN-Bi-LSTM, DeepFed, PFL-IDS, Fed-IDS, KAN-Conv, DFSat, FLOGA-AD) on mean accuracy across all 6 dataset × distribution combinations (3 datasets × IID/Non-IID).
- **Statistical significance (Table 4, paired t-test):** significant (p < 0.05) vs. the strongest IID baseline (DeepFed) on all 3 datasets. Under Non-IID, significant vs. the strongest baseline (FLOGA-AD) **only on Bot-IoT** (p = 6.00×10⁻⁴); **not significant** on STIN (p = 0.0802) or TON_IoT (p = 0.519) — the paper reports this honestly rather than hiding it.
- **Ablation (Fig. 5):** under Non-IID, removing the SAC-based clustering, the clustering-algorithm pool, the divergence penalty, or the classifier-head-based similarity measure all hurt accuracy — confirming these components genuinely contribute, especially under heterogeneity (effects were much smaller under IID).
- **Variance decomposition (Table 2, a mathematical proof, not just an experiment):** shows that FedAvg mathematically forces *all* client-update variance into a single "within-group" bucket (hiding heterogeneity), while FedSTAR's clustering reveals a genuine "between-cluster" variance component (R_B = 0.12, meaning 12% of total variance is attributable to differences *between* clusters) — a clean, defensible piece of evidence for *why* clustering should help, not just *that* it happened to help in this run.
- **SAC policy behavior (Fig. 6):** the agent explored all 5 clustering algorithms in the first ~20 rounds, then converged to mostly choosing DBSCAN (34 of 50 rounds) and mostly K=2 clusters (29 of 50 rounds) — evidence the policy is genuinely adapting, not hard-coded.
- **Cost (Table 6):** FedSTAR is **not the cheapest** method — it needs more training time (102.82 min vs. FLOGA-AD's 87.83 min) and higher inference latency (4.78 ms/batch vs. 3.08 ms/batch) than the closest competitor, in exchange for its accuracy gain and lower communication cost (372.02 MiB vs. 445.99 MiB) and lower FLOPs (96.2M vs. 170.0M).

---

## 10. What Did This Paper Do Well?

- Honestly reports **non-significant** results where they occurred (Table 4) instead of only showing favorable comparisons — a good scientific practice.
- Provides a genuine **mathematical proof** (variance decomposition, Section 3.3.3) for *why* clustering should reveal hidden heterogeneity, not just an empirical claim.
- A thorough ablation study that specifically separates IID and Non-IID effects, showing the proposed components matter *more* exactly where they should (under heterogeneity) — a well-designed experiment, not a coincidence.
- Directly examines the SAC agent's actual round-by-round behavior (which algorithm, which cluster count, how the continuous knobs evolved) instead of treating the RL agent as a black box.
- Reports both accuracy **and** deployment cost (communication, training time, FLOPs, latency) together, giving an honest accuracy-vs-cost tradeoff picture.
- Uses a satellite-specific benchmark (STIN) with genuinely satellite-relevant attack types (DTN-protocol DDoS), not just generic IoT/network data.

---

## 11. What Are Its Limitations?

- **Explicitly stated by the authors (Conclusion, "This study still has several limitations"):**
  - No real operational satellite traffic was used — STIN is the closest available benchmark, but is not a replacement for real orbital traces.
  - Orbital networking realism is simplified — no delayed updates, intermittent links, stale client states, or time-varying link availability are modeled.
  - FedSTAR is **heavier** than lightweight federated baselines, which may restrict deployment on satellites with tight power or latency budgets.
  - Robustness under extreme label skew, unseen attacks, distribution shift, sparse local samples, or **adversarial/Byzantine federated learning** (compromised clients injecting poisoned updates) is an **open, unaddressed issue** — the paper states this directly: *"The adversarial robustness of the federated optimization process itself is left for future investigation."* (this sentence appears twice in the paper — in the Introduction and again in the Conclusion, so it's clearly a deliberate, acknowledged scope boundary, not an oversight.)
- **No telemetry/physical-fault anomaly detection at all** — "anomaly" in this paper always means malicious network traffic, never a spacecraft hardware/operational fault. *(observed, by comparing directly against Papers 1–3)*
- No genuinely real spacecraft data of any kind is used anywhere in the paper. *(observed)*
- Single random-seed reporting per configuration is not explicitly stated as multi-seed, though 5 random seeds (2024–2028) were used for the main comparison table (Table 3), which is actually a methodological strength worth noting (better than Papers 1–3, which appear single-run).

---

## 12. Future Work

The authors explicitly state (Conclusion):
1. Develop **lightweight variants** to reduce local computation and communication cost.
2. Extend evaluation beyond public benchmarks by incorporating **operational satellite traces and orbital network simulations** that account for mobility, intermittent contact windows, delay-tolerant networking forwarding, and time-varying channels.
3. Investigate **federated optimization in the presence of compromised clients** — Byzantine updates, poisoning attacks, coordinated manipulations — and design more robust aggregation protocols for these settings.

**Why this matters for your project:** item 3 (robustness to compromised/malicious clients) is a gap that is *also* untouched by your own project — worth explicitly acknowledging as shared, still-open ground rather than claiming your project already solves it.

---

## 13. Research Gap

### Explicit gap (stated by the authors)
- No real satellite traffic validation.
- No robustness to Byzantine/poisoning clients.
- No orbital-realistic networking conditions (delay, intermittency, staleness).
- Heavier compute/latency cost than lightweight alternatives.

### Observed gap
- The paper's SAC agent controls *clustering and aggregation policy* — it does **not** control anything about *detection* itself (the detection architecture is fixed; SAC only decides how clients are grouped and combined). This is an important, precise distinction: DRL here optimizes the *federated learning process*, not the anomaly detector's own decisions. *(this is also true of your own project's DQN — a genuinely shared design philosophy worth naming explicitly, see Section 14)*
- No comparison against a non-clustering DRL baseline that instead learns continuous per-client aggregation *weights* directly (skipping the discrete clustering step entirely) — the paper compares against fixed-clustering and no-RL ablations, but not against "DRL without clustering."

---

## 14. Connection to Our Capstone

| Our component | Does this paper address it? | How? |
|---|---|---|
| Satellite network | Partially | Uses a satellite-relevant benchmark (STIN) with satellite-specific attack types, but not real operational satellite traffic (same honest caveat your own project applies to its own NSL-KDD-based Track A) |
| Anomaly detection | Yes, but a **different kind** | Malicious network-traffic detection (cyberattacks), not physical/telemetry-fault detection — same domain framing as your project's Track A (NSL-KDD-based), *different* from Track B (OPSSAT-AD real telemetry) |
| Deep learning | Yes | A specialized spatio-temporal architecture (SimVPv2 + gSTA/LKA), substantially heavier than your project's BaselineMLP/TemporalGRUAttention |
| Federated learning | Yes | Full FL loop with client clustering + intra-cluster + global aggregation — more elaborate than your project's FedAvg + weighted-aggregation approach |
| Reinforcement learning | Yes | SAC (continuous + discrete hybrid action) — your project uses DQN (discrete-only, smaller action set of 9 fixed profiles) |
| Adaptive behavior | Yes | SAC dynamically picks clustering algorithm + hyperparameters every round — directly comparable to (and more elaborate than) your project's Phase 9B DQN, which picks among 9 fixed weight-profiles every round |
| Privacy | Partially | Standard FL privacy framing (raw data stays local); does not address Byzantine/malicious clients, explicitly left as future work |
| Non-IID clients | Yes | Central focus of the paper — Dirichlet-based Non-IID partitioning, exactly the same partitioning technique (Dirichlet distribution) your project's Phase 4/7 uses for its own 8 simulated satellite clients |
| Communication efficiency | Yes | Explicitly measured (MiB per round) and compared against baselines |
| Real-time detection | Not clearly specified in the paper | Reports inference latency (ms/batch) but does not explicitly frame the system as running in real time on an actual satellite |

---

## 15. What We Can Learn From This Paper

- **This is your project's closest "sibling" in the literature** — read it as a picture of what a much larger, better-resourced version of a similar idea looks like, not as a template to copy line-for-line. Your project's Phase 9B design documents already independently arrived at several of the same good practices this paper uses: comparing against a clear rule-based baseline, disclosing negative/marginal results honestly, using Dirichlet Non-IID partitioning, and keeping the DRL agent's job narrowly scoped to the *aggregation/weighting decision*, not the detector itself.
- The **variance-decomposition proof** (Table 2) is a genuinely portable idea: if your project ever wants to argue *why* Phase 8/9B's weighted aggregation should help under Non-IID conditions (beyond just reporting the resulting F1 number), a similar "how much of the total client-update variance is FedAvg hiding" analysis would strengthen that argument mathematically, not just empirically.
- The explicit SAC-behavior analysis (which algorithm/cluster-count was chosen, and when) is a good template for how your own project already reports Phase 9B's DQN action distribution (e.g., which of the 9 weight-profiles were chosen, and how that changed between exploration and the final frozen-policy run) — you are already doing something methodologically similar; this paper shows what a more mature, larger-scale version of that same reporting style looks like.
- Confirms, from an independent and much larger project, that **DQN vs. SAC** is a real, meaningful design choice, not a detail — SAC's ability to output continuous actions is *why* this paper's action space (clustering algorithm + 3 continuous knobs) is richer than your project's action space (9 fixed discrete profiles). If your project's report ever needs to justify choosing DQN over SAC, this paper is the natural point of comparison: DQN was chosen for a **smaller, more explainable, more capstone-appropriate** action space, at the cost of not being able to output continuous aggregation weights directly.

---

## 16. What We Should NOT Copy

- **Do not claim your project is the first to combine federated learning, deep reinforcement learning, and satellite anomaly detection** — this paper already did that, in 2026, with a title extremely close to your capstone's title. Any framing of your project's novelty needs to explicitly distinguish itself from FedSTAR (see "the critical positioning question" in Part 4/Part 5 of the knowledge base for how to phrase this honestly).
- Do not copy the "spatio-temporal cyberattack detection on network traffic" framing wholesale as if it were the same problem your project solves — your project's title says "satellite network anomaly detection," and your actual implementation currently targets **network intrusion patterns (NSL-KDD, terrestrial data used to simulate satellite clients)**, which is architecturally closer to this paper's problem framing than to Papers 1–3's telemetry-fault framing — but your project's own OPSSAT-AD track (Track B) is telemetry-based and is **never combined** with the FL/DRL machinery. Be precise about which of your two tracks you're discussing at any point in a report or viva answer.
- Do not present FedSTAR's SAC-based clustering as strictly "better" than your project's DQN-based weighting without qualification — FedSTAR is also heavier, slower to train, and higher-latency than a lighter baseline (its own Table 6 says so). A smaller, more explainable action space (as your project chose) is a legitimate, defensible design decision, not an inferior one by default.

---

## 17. Simple Example

**Illustrative example — not from the paper.**

Imagine 20 satellites. Satellites 1–7 mostly see DDoS-style traffic; satellites 8–14 mostly see port-scanning traffic; satellites 15–20 see a healthy mix of both. Plain FedAvg averages all 20 equally every round, producing one "average" model that's mediocre at both attack types. FedSTAR's SAC agent instead notices (from the similarity matrix) that satellites 1–7 look alike, 8–14 look alike, and 15–20 look different from both — it picks DBSCAN with roughly 2–3 clusters, aggregates each group separately, and only then blends the group models into one global model, producing a model that handles both attack types better than the blurry, plain-averaged version would.

---

## 18. Explain Like I'm Preparing for Viva

**Q: What problem does this paper solve?**
A: Detecting cyberattacks in satellite network traffic using federated learning (so raw traffic never leaves each satellite), while handling the reality that different satellites see very different (Non-IID) traffic patterns — solved by using a reinforcement learning agent (SAC) on the server to intelligently cluster and combine clients, instead of a fixed averaging rule.

**Q: Why did they use SAC instead of a simpler RL algorithm like DQN?**
A: Their action space is a mix of a discrete choice (which of 5 clustering algorithms to use) and continuous values (sampling ratio, aggregation weight, fusion coefficient). SAC natively supports both kinds of actions together; DQN only supports discrete actions.

**Q: What dataset did they use?**
A: STIN (a benchmark purpose-built for satellite-terrestrial network security, including satellite-specific DTN-protocol DDoS attacks), plus TON_IoT and Bot-IoT as auxiliary general IoT/network intrusion benchmarks — none are real operational satellite traffic.

**Q: What was their best result?**
A: Under Non-IID conditions, FedSTAR reached 95.43% accuracy on STIN, 89.16% on TON_IoT, and 99.88% on Bot-IoT — the highest of all 8 methods compared, though the improvement over the closest baseline was only statistically significant on Bot-IoT, not on STIN or TON_IoT.

**Q: What is the biggest limitation?**
A: No real satellite traffic was tested, no robustness against malicious/Byzantine clients was addressed, orbital realism (delay, intermittent links) was not modeled, and the method is heavier/slower than lightweight alternatives.

**Q: How is this related to our project?**
A: It is the closest existing paper to your capstone's own title and general idea — federated learning + deep reinforcement learning for satellite anomaly detection — so it is essential "closest prior work" that your project must explicitly differentiate itself from, rather than a paper you can treat as unrelated background.

**Q: What's the single biggest difference between this paper and our project?**
A: This paper uses SAC to choose a clustering algorithm and continuous hyperparameters for grouping+aggregating clients in a cyberattack-detection setting on benchmark network-traffic data; our project uses a much smaller DQN to choose among 9 fixed aggregation weight-profiles (no clustering) in an intrusion-detection setting built from Dirichlet-partitioned NSL-KDD data (Track A), with a separate, honestly-disclosed real-spacecraft-telemetry validation track (OPSSAT-AD, Track B) that is not yet connected to the federated/DRL machinery at all.
