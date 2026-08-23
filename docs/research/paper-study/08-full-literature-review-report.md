# Full Literature Review Report — Deep Reinforcement Learning-Driven Adaptive Federated Framework for Satellite Network Anomaly Detection

**Purpose of this document:** a single, ordered, IEEE-style literature analysis assembled per an explicit 15-section request. It draws on (a) the 4 papers actually provided as PDFs in this project, read in full, and (b) additional papers located via live web search on 2026-08-23, verified against real search results (titles/venues/URLs), never invented. Where a detail could not be verified, it is explicitly marked **"Not verified from the available source."**

**Relationship to the rest of this folder:** this document is the master synthesis. Deeper per-paper breakdowns already exist in `01-paper-notes/`, concept explainers in `02-concept-notes/`, and prior gap/positioning analysis in `04-research-gap/`. This document does not repeat all of that verbatim — it references it — but it does independently re-derive every judgment asked for here (comparison table, novelty statement, phase mapping, etc.) rather than just pointing elsewhere.

**Honesty note on scope:** only 4 PDFs were ever provided to this project (Hundman 2018, Yang 2021, Maleki Sadr 2022, Chen et al. 2026/FedSTAR). This report treats those 4 as "the papers we have already read in full," and everything else in Sections 4/10/12 as "found independently via search, verified only to the depth a search snippet allows" — a materially weaker level of confidence, disclosed throughout.

**Honesty note on project phase numbering:** this task's prompt describes the project as being at "Phase 5 – Local client training / Non-IID federated experiments," with adaptive FL and DRL described as future/later phases. Based on this project's own committed documentation (`docs/project-progress/00-project-plan.md` and Phase 9A/9B write-ups), **the project has already progressed further than that**: rule-based adaptive FL (that plan's "Phase 8") and a full DQN-based adaptive FL experiment (that plan's "Phase 9B," fully executed, with the DQN's real, honestly-reported result already measured) are both complete, alongside a separate real-spacecraft-telemetry validation track (OPSSAT-AD, that plan's "Phase 9A"). Section 11 below maps literature to the **actual, current** phase structure, and explicitly reconciles it against the simpler 1–5 framing this prompt used, so nothing is silently glossed over.

---

## 1. Executive Summary

This project's literature splits into two nearly disconnected sub-fields: (a) single-spacecraft, LSTM-based telemetry forecasting anomaly detection with no distributed-learning structure (Hundman 2018 → Yang 2021 → Maleki Sadr 2022), and (b) an emerging, much newer stream combining federated learning and deep reinforcement learning for satellite-network cyberattack detection (Chen et al. 2026, "FedSTAR"). Independent search surfaced a third, highly relevant cluster: FedSTAR's own direct competitor, **FLOGA-AD** (Wang, Cao & Di, 2026, *Expert Systems with Applications*) — a genetic-algorithm-driven federated anomaly detection framework for satellite networks, using the *same* STIN/TON_IoT/Bot-IoT datasets and the *same* CvT-LSTM-style spatio-temporal backbone family — plus several DRL-based client-selection papers, satellite-specific FL papers (RAFL, ALANINE, DFedSat), and FL+IDS surveys.

**The core finding:** no paper found, including this project's own current implementation, combines (1) real or realistically-simulated satellite network anomaly data, (2) federated learning, (3) adaptive/resource-aware client weighting, (4) deep reinforcement learning, (5) genuine temporal or spatio-temporal modeling, and (6) measured Non-IID handling, **validated against real satellite data of any kind.** FedSTAR and FLOGA-AD come closest (1–5, all six on benchmark, not real, satellite data). This project's own two-track structure (Track A: FL+DRL on simulated satellite clients; Track B: real spacecraft telemetry, no FL) is a defensible, honestly-scoped position **between** the two literatures — not a claim of having closed the gap.

**The defensible novelty** is narrow and specific, not a broad "first to combine X+Y+Z" claim (FedSTAR/FLOGA-AD already made similar combination claims in the same year): a **smaller, fully-discrete, fully-explainable DQN-driven adaptive aggregation mechanism**, validated through a **controlled three-stage comparison** (fixed FedAvg → hand-written adaptive rule → learned DQN policy) on the identical pipeline, with an **honestly-reported negative result** for the DQN stage and a specific, falsifiable explanation for it — a methodological rigor point genuinely less common in this literature than a raw performance claim would be.

---

## 2. Core Paper Analysis — FedSTAR (Chen, Zhang, Chen, Wu, Cai, Chen, 2026)

**Full citation:** Hui Chen, Kun Zhang, Yangtao Chen, Xiaohe Wu, Chenxi Cai, Dianying Chen, "Anomaly detection method for satellite networks based on adaptive federated learning driven by deep reinforcement learning," *Information Sciences*, Vol. 755, Article 123808, 2026. DOI: 10.1016/j.ins.2026.123808. (Nanjing University of Science and Technology / Nankai University, China.)

### 2.1 Exact problem addressed
Detecting **cyberattacks** (DDoS, botnets, backdoors, port scans, and — specifically — DTN-protocol-exploiting satellite DDoS) in **satellite-terrestrial network traffic**, without centralizing raw traffic at a ground station, and without plain FedAvg's performance collapse under Non-IID client data.

### 2.2 Why ordinary anomaly detection is insufficient
Standalone CNN/LSTM models prioritize either spatial or temporal features in isolation (paper's own words, Introduction); serial CNN-then-LSTM fusion (e.g. CvT-LSTM-style hybrids) limits deep spatial-temporal interaction and misses complex, multi-stage attack patterns. A single centralized model also cannot be trained without shipping raw traffic off every satellite — infeasible given bandwidth and privacy constraints.

### 2.3 Why federated learning is used
Two stated reasons: (1) **bandwidth** — satellite-ground/inter-satellite links are scarce and costly; shipping raw traffic causes congestion/latency; (2) **privacy** — uploading raw distributed data risks interception/misuse. FL lets nodes train locally and share only parameter updates.

### 2.4 Why *adaptive* federated learning is needed (not plain FedAvg)
Real orbital data is Non-IID (class imbalance, feature-distribution shift across regions/orbits). Plain FedAvg averages everyone equally, which — proven mathematically in the paper's own variance-decomposition (Section 3.3.3) — forces *all* client-update variance into a single within-group term, hiding heterogeneity, causing weight divergence and degraded generalization.

### 2.5 Why deep reinforcement learning is used
Because the "right" clustering/aggregation configuration changes as training progresses and depends on the current similarity structure of clients — a fixed rule (like a static clustering algorithm/hyperparameter set) cannot adapt round to round. An RL agent, observing the current state, can learn a policy that adjusts these choices dynamically.

### 2.6 What the RL agent (SAC) observes — state representation

| Component | Formula/source |
|---|---|
| Pairwise similarity/topology | `V_sim = vectriu(S_t)`, upper-triangular vectorization of the cosine-similarity matrix `S_i,j = (w_i·w_j)/(‖w_i‖‖w_j‖)` computed from **classifier-head parameters only** (chosen because final layers reflect local-task/data differences more than early layers) |
| Training feedback | `s_loss = tanh(L_{t-1})`, `s_acc = A_{t-1} ∈ [0,1]` (previous-round global loss/accuracy) |
| Algorithm memory | `v_algo = OneHot(m_prev)` (previous round's chosen clustering algorithm) |
| Cluster-count memory | `s_k = (K_prev − K_min)/(K_max − K_min)` (previous cluster count, normalized) |

Full state: `s_t = concat(V_sim, s_loss, s_acc, v_algo, s_k)`.

### 2.7 Action space (hybrid discrete + continuous)
`a_t = {a_d_t, a_c_t}`:
- **Discrete** `a_d_t = {m_t, k_t}`: `m_t ∈ {FedAvg, K-Means, Spectral, Agglomerative, DBSCAN}` (which clustering algorithm), `k_t ∈ {1,…,6}` (cluster count, `K_min=1, K_max=6`).
- **Continuous** `a_c_t = [ρ_t, λ_t, γ_t]^T ∈ [0,1]^3`: `ρ_t` = client sampling ratio within each cluster, `λ_t` = intra-cluster aggregation weight factor (1 → FedAvg-style, 0 → simple mean), `γ_t` = global-local fusion coefficient (how much of the new cluster-aggregated model blends into the persisting global model).

### 2.8 Reward function
`r_t = R_gain − P_degrade − P_div`, where:
- `R_gain = λ_α · (A_t − A_{t-1}) · 100` (accuracy improvement, rewarded).
- `P_degrade = λ_β · |A_t − A_{t-1}| · 100` if accuracy *fell* this round, else 0 (penalizes regression).
- `P_div = λ_γ · (1 − S̄_t)` where `S̄_t` is the mean intra-cluster similarity across all clusters (penalizes clusters that end up internally dissimilar/incoherent).
- Weights used: `λ_α = 1.0, λ_β = 1.0, λ_γ = 0.0001` (chosen via grid search on training-stage feedback, then fixed).

### 2.9 What the RL agent optimizes
A policy (SAC actor + critic networks, 2 hidden layers × 256 neurons, ReLU) that jointly maximizes accuracy gain while minimizing performance regression and cluster incoherence — i.e., it learns **when to change strategy, not just whether accuracy went up**, balancing stability against improvement.

### 2.10 How clients are selected/weighted
Not client *selection* in the sense of dropping clients from a round — all clients are first clustered (via the chosen algorithm/`k`), then within each cluster a subset is sampled (`N_sample = max(1, ⌊|C_k|·ρ_t⌋)`), and aggregated with weight `α_i = λ_t·(D_i/ΣD_l) + (1−λ_t)·(1/|C'_k|)` (a tunable blend between data-volume-weighted and uniform averaging).

### 2.11 How local training works
Each client trains its local spatio-temporal detector (pseudo-video projection → ConvSC spatial encoder → gSTA/LKA spatio-temporal Translator → classifier head) via standard SGD for several local epochs, then uploads only its classifier-head parameters (not the full model, not raw data) for similarity computation, and its full model for aggregation.

### 2.12 How aggregation works
Two-level hierarchical aggregation: (1) intra-cluster weighted average → `θ^cluster_k = Σ α_i θ_i`; (2) cluster models combined into one aggregated model weighted by cluster size → `θ^cluster_t = Σ|C_k|·θ^cluster_k / Σ|C_k|`; (3) global-local momentum fusion → `θ^global_t = γ_t·θ^cluster_t + (1−γ_t)·θ^global_{t-1}`.

### 2.13 How anomaly detection works
A classification head (global average pooling → LayerNorm → GELU → Dropout(0.1) → 2-layer MLP) maps the spatio-temporal feature tensor to attack-category probabilities — a supervised, multi-class classifier, not a forecasting/reconstruction-residual detector (unlike Papers 1–3).

### 2.14 Complete end-to-end workflow
```text
Each client: 1D traffic → pseudo-video (T=4, C=64) → ConvSC encoder (2 blocks:
Conv2D+GroupNorm+SiLU) → Translator (4 gSTA blocks, each using LKA:
5×5 depth-wise conv → dilated depth-wise conv (d=3) → 1×1 point-wise conv →
sigmoid gate) → classifier head → local loss/update
        ↓ (classifier-head params uploaded)
Server: cosine-similarity matrix → state s_t → SAC picks {clustering algo,
k, ρ, λ, γ} → cluster clients → intra-cluster weighted aggregation →
cluster fusion → global-local momentum update → evaluate accuracy/loss →
reward r_t → SAC learns (actor+critic gradient step) → next round
```

### 2.15 Important equations/algorithms
Eqs. (1)–(6): pseudo-video projection and ConvSC spatial encoding. Eqs. (3)–(4): LKA attention (`Attn = Conv_PW(Conv_DW-D(Conv_DW(F)))`, `H_out = Sigmoid(Attn)⊙F`). Eqs. (7)–(37): state/action/reward formalization and the variance-decomposition proof (`V_T = V_W + V_B`, showing FedAvg forces `V_B = 0`). Algorithm 1 (DRL-CFL): the full per-round pseudocode.

### 2.16 Experimental setup
20 clients; IID (uniform) vs. Non-IID (Dirichlet, η=0.5) partitioning; 50 communication rounds, 10 local epochs/round; AdamW, batch 64, lr 1e-4; SAC hyperparameters: replay buffer 10,000, batch 16, discount 0.99, soft-update 0.005; 5 random seeds (2024–2028) for the main comparison.

### 2.17 Dataset
STIN (satellite-terrestrial-specific benchmark, 32 features, 209,564 instances, 9 attack categories incl. 2 satellite-specific DTN-protocol DDoS types), TON_IoT (44 features, 211,043 instances, 10 categories, auxiliary), Bot-IoT (46 features, >73M instances, 5 categories, auxiliary). **None are real operational satellite traffic** — explicitly disclosed by the paper's own Conclusion.

### 2.18 Results
Non-IID accuracy: STIN 95.43%, TON_IoT 89.16%, Bot-IoT 99.88% — best of 8 compared methods (CNN-Bi-LSTM, DeepFed, PFL-IDS, Fed-IDS, KAN-Conv, DFSat, FLOGA-AD). Statistically significant (p<0.05) vs. strongest IID baseline (DeepFed) on all 3 datasets; vs. strongest Non-IID baseline (FLOGA-AD) significant **only** on Bot-IoT (p=6.00×10⁻⁴; not significant on STIN p=0.080 or TON_IoT p=0.519 — disclosed honestly by the paper itself). Variance decomposition: `R_B = V_B/V_T = 0.12` for FedSTAR vs. `0.00` for FedAvg. Cost: heavier than the lightest baselines — 102.82 min training vs. FLOGA-AD's 87.83 min; 4.78 ms/batch latency vs. FLOGA-AD's 3.08 ms/batch — traded for lower communication (372.02 vs. 445.99 MiB) and FLOPs (96.2M vs. 170.0M).

### 2.19 Advantages
Genuine mathematical justification (not just empirical) for why clustering should help; thorough IID-vs-Non-IID ablation isolating each component's contribution; transparent, round-by-round SAC-behavior reporting (which algorithm/cluster-count chosen, over time); honest reporting of non-significant results.

### 2.20 Limitations (self-disclosed by the paper)
No real operational satellite traffic; orbital realism (delay, intermittent links, staleness) not modeled; heavier/slower than lightweight alternatives; **Byzantine/adversarial-client robustness explicitly left to future work** (stated twice in the paper).

### 2.21 Simple-language explanation for viva
"FedSTAR is a system where satellites train an attack detector locally (never sharing raw traffic), and a reinforcement-learning 'manager' sitting on the server watches how similar the satellites' models currently are, and decides — every round — how to group similar satellites together and how to blend their models back into one shared model, instead of just averaging everyone equally like standard federated learning does. It works better than plain averaging specifically when satellites see very different kinds of traffic (Non-IID), because grouping similar ones together before averaging avoids blurring their differences away."

---

## 3. All Existing Papers Analysis (Papers 1–3, condensed — full depth in `01-paper-notes/`)

### 3.1 Hundman, Constantinou, Laporte, Colwell, Soderstrom (2018) — KDD '18, NASA JPL
- **Problem/motivation:** thousands of telemetry channels, existing out-of-limits alarms miss context-dependent ("contextual") anomalies; need automatic, unsupervised thresholding.
- **Dataset:** real NASA SMAP (55 ch.) / MSL (27 ch.) telemetry, ISA-report-derived labels, 105 anomaly sequences.
- **Preprocessing/features:** per-channel history + one-hot encoded command (module + sent/received).
- **Model:** plain LSTM, 2 layers × 80 units, per channel, sequence length 250.
- **FL/client-server:** none. **Non-IID:** N/A (no clients). **RL:** none.
- **Anomaly method:** forecasting residual → EWMA smoothing → **nonparametric dynamic threshold** (`ε = argmax` of normalized mean/std-drop, penalized by anomaly count) → **pruning** (percent-decrease rule) to cut false positives.
- **Aggregation/communication:** N/A (centralized, single system).
- **Evaluation:** Precision/Recall/F0.5. **Result:** 87.5% precision, 80.0% recall, F0.5=0.71 (best variant, w/ pruning) — beat a Gaussian-tail parametric baseline (F0.5=0.66).
- **Limitations:** single-spacecraft, single-channel models, no FL/RL, no cross-channel correlation modeling (explicit future work).
- **Reuse:** the general "forecast → residual → data-driven threshold" recipe, and the SMAP/MSL dataset itself, are directly reused by Papers 1 and 2. **Don't copy:** the Gaussian-threshold assumption (this paper disproves it statistically for this data).
- **Gap:** no multi-satellite structure of any kind.

### 3.2 Yang, Ma, Zeng, Peng, Liu (2021) — *Microelectronics Reliability* (Elsevier)
- **Problem/motivation:** plain LSTM loses long-range info; fixed-window detection misses slow-developing anomalies.
- **Dataset:** same SMAP/MSL benchmark + real Beidou navigation-satellite hydrogen-clock telemetry (synthetic 1% anomaly injected, Eq. 10).
- **Model:** double-layer LSTM + attention (Fig. 2).
- **Anomaly method:** residual thresholding at **two time scales simultaneously** — local sliding-window + global whole-sequence — combined by union.
- **FL/RL/Non-IID:** none, none, N/A.
- **Evaluation:** Recall / False Alarm Rate. **Result:** 86.54% recall, 12.62% FAR (best of 8 method×scale combinations tested).
- **Limitations:** compute-cost increase acknowledged but not quantified; Beidou anomaly is a synthetic injection, not a naturally occurring incident; no statistical-significance testing.
- **Reuse:** attention-over-LSTM (conceptually mirrored in this project's own `TemporalGRUAttention`); multi-scale detection as a "don't trust one time-scale alone" pattern. **Don't copy:** presenting the Beidou result as detection of a *real* incident.
- **Gap:** same as Hundman — no FL, no RL, no multi-satellite.

### 3.3 Maleki Sadr, Zhu, Hu (2022) — *IEEE TAES*
- **Problem/motivation:** forecasting methods give a point prediction with no confidence measure; true Bayesian NNs are computationally intractable.
- **Dataset:** same SMAP/MSL benchmark (82 signals, some excluded as degenerate/constant, listed explicitly).
- **Model:** Bayesian LSTM via **Monte Carlo Dropout** applied inside both LSTM and dense layers (3 LSTM + 2 dense, dropout rate 0.2) — improves on a prior method (MCD-BiLSTM-VAE, Chen et al.) that applied dropout only before dense layers.
- **Preprocessing:** adaptive weighted moving average with a variable-width window (narrower when consecutive-point distance exceeds `mean+2·std`, to avoid smoothing away real anomalies).
- **Anomaly method:** many stochastic forward passes → mean + variance → confidence interval → post-processing rule (confirm only if ≥0.8·N_max of N_max consecutive points fall outside the interval; N_max tuned via grid search against known labels).
- **FL/RL/Non-IID:** none, none, N/A.
- **Evaluation:** F1, MSE, Accuracy, Precision, Recall, parameter count, training time, throughput. **Result:** F1 0.84 (SMAP), 0.74 (MSL) — beat TadGAN, LSTM-AE, MAD-GAN, plain LSTM, and MCD-BiLSTM-VAE, at far lower parameter count/training time than the closest MC-Dropout competitor.
- **Limitations:** `N_max` tuning references true labels (semi-supervised in practice, though framed as unsupervised); command features found irrelevant for this dataset and dropped (dataset-specific, not general).
- **Reuse:** Monte Carlo Dropout is a low-cost, directly transferable technique for adding uncertainty estimates to any of this project's existing detectors. **Don't copy:** presenting the `N_max` tuning as fully label-free.
- **Gap:** same as the other two — no FL, no RL, no multi-satellite.

---

## 4. Additional Papers Found (independent search, by category)

**Method disclosure:** found via live web search on 2026-08-23; verified only to the depth of search-result titles/venues/URLs shown below — **not read in full**. Treat as leads requiring independent verification before citing as primary sources, distinctly weaker confidence than Sections 2–3 above.

### A. Satellite network anomaly detection
- **Wang, Z., Cao, J., Di, X.** — "Anomaly detection method for satellite networks based on genetic optimization federated learning" (**FLOGA-AD**), *Expert Systems with Applications*, Vol. 295, 2026, Article 128627. DOI: 10.1016/j.eswa.2025.128627 (ScienceDirect listing found; exact DOI numerals **not independently re-verified beyond the ScienceDirect URL and FedSTAR's own reference-list citation** — cross-checked against FedSTAR's reference [24], which cites it identically). **Why relevant:** FedSTAR's own direct, strongest competitor — genetic-algorithm client selection + CvT-LSTM spatio-temporal backbone, same STIN/TON_IoT/Bot-IoT datasets. **This is arguably as important a "closest work" as FedSTAR itself** — recommend obtaining the full PDF.
- **"Adaptive Federated Learning for Anomaly Detection in Satellite Telemetry"** — University of East London institutional repository listing found (`uel-repository.worktribe.com/output/439906`). **Why relevant:** appears to directly bridge the "telemetry" and "federated learning" lineages this report's Section 1 identifies as disconnected — potentially the single most gap-relevant paper found. **Author/year/venue not verified from the available source** (only a repository landing page was found in search results, not a publisher record) — must be independently confirmed before citing.

### B. Federated learning for intrusion/anomaly detection
- **Survey**, "Survey on Federated Learning for Intrusion Detection System: Concept, Architectures, Aggregation Strategies, Challenges, and Future Directions," *ACM Computing Surveys*, DOI: 10.1145/3687124. **Why relevant:** a top-tier (ACM CSUR), citable survey anchor for the entire FL+IDS field — good for a literature-review's opening framing paragraph. Author names **not verified from the available source** (only the DOI/venue was confirmed via search).

### C. Adaptive federated learning
- **AdaptFL / FedDA** (ScienceDirect, *Future Generation Computer Systems* family journals) — resource-adaptive FL frameworks assigning heterogeneous model sizes per device capability. **Why relevant:** a different flavor of "adaptive" (model-size adaptation) than this project's weighting-based adaptation — useful contrast in a literature review. Full citation details **not verified from the available source**.

### D. Deep reinforcement learning for federated learning (general, not satellite-specific)
- **"Adaptive Client Selection in Resource Constrained Federated Learning Systems: A Deep Reinforcement Learning Approach,"** IEEE Xplore document 9478892. **Why relevant:** methodologically close analog to this project's own Phase 9B goal (DRL deciding *how* to treat clients each round), just for client selection rather than weighting, and not satellite-specific. **HIGH relevance candidate.**
- **"Robust Peer-to-Peer Federated Learning With Deep Reinforcement Learning Based Client Selection Against Data Poisoning Attacks,"** IEEE Xplore document 11008679. **Why relevant:** directly addresses the Byzantine-robustness gap FedSTAR itself names as unsolved.
- **"FedAEB: Deep Reinforcement Learning Based Joint Client Selection and Resource Allocation Strategy for Heterogeneous Federated Learning,"** IEEE Xplore document 10416384 — uses **Soft Actor-Critic**, same DRL family as FedSTAR, for a non-satellite FL setting. **Why relevant:** independent confirmation that SAC is a standard, recurring choice for this exact class of problem (client selection/resource allocation under heterogeneity), not something FedSTAR invented.

### E. Reinforcement-learning-based client selection
- **"A Federated Learning Client Selection Method via Multi-Task Deep Reinforcement Learning,"** IEEE Xplore document 10840140 — multi-task DRL, two reward terms (model performance + system latency). **Why relevant:** a directly comparable multi-objective reward-shaping precedent to both FedSTAR's and this project's own multi-term reward.
- **"FedPPO: Reinforcement Learning-Based Client Selection for Federated Learning With Heterogeneous Data,"** IEEE Xplore (PDF at iel8/6687307/11288933/10909702.pdf) — uses **PPO**, a third DRL algorithm family (alongside this project's DQN and FedSTAR's SAC) applied to the same problem class, with a noisy-client-filtering step to shrink the action space before DRL. **Why relevant:** evidence that PPO is a live alternative worth naming when justifying "why DQN, not PPO/SAC" in a report.

### F. Resource-aware federated learning
- **"Ed-Fed: A generic federated learning framework with resource-aware client selection for edge devices,"** arXiv preprint. **Why relevant:** resource-awareness (bandwidth/compute/availability) as an explicit selection signal — directly analogous to this project's own `resource_k` signal family (Phase 8/9B), independently confirming this is a recognized, standard signal type in the wider FL literature, not invented for this project.

### G. Non-IID federated learning
- **FedProx** (Li et al.) and **SCAFFOLD** — foundational, highly-cited Non-IID-mitigation algorithms (proximal term / control-variate variance reduction respectively). **Why relevant:** foundational alternatives to the weighting-based adaptation this project and FedSTAR both use — worth naming in a literature review as "other families of Non-IID mitigation not attempted here." Exact original citations (venue/year/DOI) **not independently re-verified in this search pass** — both are extremely well-known in the FL literature (FedProx: Li et al., MLSys 2020; SCAFFOLD: Karimireddy et al., ICML 2020) but should be confirmed against a primary source before citing in a formal document.
- **"Non-IID data in Federated Learning: A Survey with Taxonomy, Metrics, Methods, Frameworks and Future Directions,"** arXiv:2411.12377. **Why relevant:** a broad, recent survey anchor for the Non-IID sub-field generally.

### H. Satellite/space communication security
- **"A Comprehensive Literature Review of Cybersecurity in Satellite Networks,"** MDPI *Aerospace*, Vol. 13, No. 3, Article 249. **Why relevant:** a peer-reviewed (MDPI) survey specifically on satellite cybersecurity — good general background citation, independent of the FL/DRL angle.
- **"Cyber Attacks on Space Information Networks: Vulnerabilities, Threats, and Countermeasures for Satellite Security,"** MDPI, Vol. 5, No. 3, Article 76. **Why relevant:** similarly useful general-background survey.
- **"Security Performance Analysis of LEO Satellite Constellation Networks under DDoS Attack,"** PMC/NCBI-indexed. **Why relevant:** directly relevant to understanding the specific DDoS-style attacks STIN (and this project's NSL-KDD-based Track A) attempt to model.

### I. Spatio-temporal anomaly detection
- Multiple **spatio-temporal graph neural network (STGNN)** papers surfaced (e.g., graph-convolutional-recurrent approaches for network anomaly detection) — a *different* architectural family from FedSTAR's SimVPv2/pseudo-video approach and from this project's GRU+attention approach. **Why relevant:** worth naming in a literature review as an alternative spatio-temporal modeling family not currently used by this project, useful for a "related architectural approaches" paragraph. Individual citations from this search pass are not specific/verifiable enough to cite directly — flagged as a family/direction, not a single paper.

### J. Attention mechanisms for anomaly detection
- Already substantively covered by Paper 1 (Yang et al.) in this project's existing corpus — no additional high-priority attention-specific paper stood out in this search pass as adding a materially new idea beyond what Paper 1 and FedSTAR's own LKA mechanism already cover.

### K. Federated learning + satellite networks
- **"DFedSat: Communication-Efficient and Robust Decentralized Federated Learning for LEO Satellite Constellations,"** arXiv:2407.05850. **Why relevant:** genuinely satellite-orbital (not just satellite-flavored-benchmark) FL — models intra-plane/inter-plane aggregation and inter-satellite-link unreliability (Doppler shifts, pointing limitations) — directly fills this project's own disclosed "orbital realism" gap.
- **"RAFL: Reputation-Aware Federated Learning with hierarchical aggregation in LEO satellite networks,"** ScienceDirect. **Why relevant:** a reputation/trust-based client-weighting scheme for satellite FL — a close conceptual cousin of this project's own performance/resource/fairness signal-based weighting (Phase 8/9B), independently confirming that "which satellite to trust this round" is a recognized, actively-studied problem.
- **"ALANINE: A Novel Decentralized Personalized Federated Learning For Heterogeneous LEO Satellite Constellation,"** arXiv:2411.07752. **Why relevant:** personalized (per-satellite) FL under heterogeneity, decentralized (no single server) — a different architectural philosophy than this project's or FedSTAR's centralized-server design, worth naming as an alternative.
- **"Secure and Efficient Federated Learning in LEO Constellations using Decentralized Key Generation and On-Orbit Model Aggregation,"** arXiv:2309.01828. **Why relevant:** satellite-specific security/key-management for on-orbit aggregation.

### L. DRL + federated anomaly/intrusion detection
- **"A Federated Learning Framework with Self-Attention and Deep Reinforcement Learning for IoT Intrusion Detection,"** ACM, *Proceedings of ICSIE 2024*, DOI: 10.1145/3708635.3708649. **Why relevant:** combines self-attention + DRL + FL + IDS in one system for IoT (not satellites) — the closest non-satellite analog found to this project's own overall combination, useful as evidence the combination pattern is recognized (though applied elsewhere) rather than unique to satellites.
- **"Collaborative privacy-preserving network intrusion detection: A federated multi-agent reinforcement learning approach,"** ScienceDirect. **Why relevant:** *multi-agent* RL (each RL agent handles a specific attack-type detector, aggregated via class-level FedAvg) — a structurally different DRL-FL combination than either FedSTAR's (single SAC agent controlling aggregation policy) or this project's (single DQN controlling aggregation policy) — worth naming as an alternative design.

### M. Edge/IoT anomaly detection using FL + DRL
- **"FedAtten-DRL"** (referenced within search-result summaries, exact venue not independently confirmed in this pass) — DRL for adaptive modeling + FL for communication/privacy in IoT detection. **Not verified from the available source** beyond the search-summary mention — flag for independent confirmation before citing.

---

## 5. Literature Review Matrix

| # | Paper | Year | Venue | Problem | Dataset | Domain | Model | FL | DRL/RL | Non-IID | Attention | Temporal | Anomaly Method | Main Result | Limitation | Relevance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Hundman et al. | 2018 | KDD (ACM) | Telemetry fault detection | SMAP/MSL (real) | Spacecraft telemetry | LSTM | No | No | N/A | No | Yes (LSTM) | Forecast + nonparametric threshold + pruning | F0.5=0.71 | No FL/RL, single-channel | **HIGH** |
| 2 | Yang et al. | 2021 | Microelectronics Reliability (Elsevier) | Telemetry fault detection | SMAP/MSL + Beidou (real, synth. anomaly) | Spacecraft telemetry | LSTM+Attention (2-layer) | No | No | N/A | Yes | Yes | Forecast + multi-scale threshold | Recall 86.54%, FAR 12.62% | No FL/RL, compute cost unquantified | **MEDIUM** |
| 3 | Maleki Sadr et al. | 2022 | IEEE TAES | Telemetry fault detection + uncertainty | SMAP/MSL (real) | Spacecraft telemetry | Bayesian LSTM (MC Dropout) | No | No | N/A | No | Yes | Forecast + confidence region + consecutive-point rule | F1 0.84/0.74 | No FL/RL; threshold tuning label-dependent | **MEDIUM** |
| 4 | Chen et al. (FedSTAR) | 2026 | Information Sciences (Elsevier) | Network cyberattack detection | STIN/TON_IoT/Bot-IoT (benchmark) | Satellite network traffic | SimVPv2+gSTA/LKA | **Yes** | **Yes (SAC)** | Yes (Dirichlet) | Yes (LKA) | Yes (pseudo-video) | Supervised classification | Non-IID acc. 95.43/89.16/99.88% | No real data; no Byzantine robustness; heavy | **VERY HIGH** |
| 5 | Wang, Cao, Di (FLOGA-AD) | 2026 | Expert Systems with Applications (Elsevier) | Network cyberattack detection | STIN/TON_IoT/Bot-IoT (same benchmark) | Satellite network traffic | CvT-LSTM | Yes | No (genetic algorithm, not DRL) | Yes | Not verified from the available source | Yes | Supervised classification | Reported as FedSTAR's strongest Non-IID baseline | Not verified from the available source (full text not read) | **VERY HIGH** |
| 6 | "Adaptive FL for AD in Satellite Telemetry" (repository listing) | Not verified from the available source | Not verified from the available source | Telemetry + FL | Not verified from the available source | Spacecraft telemetry | Not verified from the available source | Yes (claimed) | Not verified from the available source | Not verified from the available source | Not verified from the available source | Not verified from the available source | Not verified from the available source | Not verified from the available source | Not verified from the available source | **HIGH (if confirmed)** |
| 7 | DFedSat | 2024 | arXiv (peer-review status not verified) | Communication-efficient decentralized FL | Not satellite-anomaly-specific | LEO satellite FL infrastructure | Not verified from the available source | Yes (decentralized) | No | Not verified from the available source | Not verified from the available source | Not verified from the available source | N/A (infrastructure paper) | Reported superior convergence/comm. efficiency vs. baselines | Peer-review venue not verified | **MEDIUM-HIGH** |
| 8 | RAFL | Not verified from the available source | ScienceDirect (Elsevier journal, exact title not confirmed) | Reputation-based hierarchical FL aggregation | Not verified | LEO satellite FL | Not verified | Yes | Not verified | Not verified | Not verified | Not verified | N/A (infrastructure) | Not verified | Details not confirmed | **MEDIUM** |
| 9 | IEEE DRL client selection (doc 9478892) | Not verified from the available source (IEEE Xplore listing only) | IEEE | Resource-constrained adaptive client selection | Not satellite-specific | General FL | Not verified | Yes | Yes (DRL) | Not verified | No | No | N/A | Not verified | Not satellite-specific | **MEDIUM-HIGH** |
| 10 | ACM CSUR FL-IDS Survey | Not verified from the available source | ACM Computing Surveys | Survey of FL+IDS field | N/A (survey) | General IDS | N/A | Yes (surveyed) | Partial | Surveyed | Surveyed | Surveyed | Surveyed | N/A | N/A | **MEDIUM (background anchor)** |

*(Rows 5–10 above rely on search-snippet-level verification only, per Section 4's disclosure; do not cite specific numeric results from them without obtaining and reading the full text.)*

---

## 6. Closest Existing Work

Ranked by closeness to this project's actual title/architecture:

1. **FedSTAR (Chen et al., 2026)** — closest by title, by domain (satellite network anomaly detection), and by mechanism (FL + DRL controlling the aggregation/clustering process). The primary comparator.
2. **FLOGA-AD (Wang, Cao & Di, 2026)** — equally close by domain and dataset (literally the same STIN/TON_IoT/Bot-IoT benchmark, and FedSTAR's own strongest baseline) — differs from FedSTAR by using a genetic algorithm instead of DRL for client selection. **Recommend obtaining and reading this paper in full** — it may be as important a "must-differentiate-from" paper as FedSTAR itself, and is currently under-analyzed relative to its apparent importance.
3. **"Adaptive Federated Learning for Anomaly Detection in Satellite Telemetry"** (repository listing) — potentially the single paper most directly relevant to bridging this project's own Track A/Track B split, if verified — **not yet confirmed**, treat with caution until the full source is located and read.
4. **DFedSat** — closest on the "orbital realism" dimension (a gap FedSTAR itself admits), not on the anomaly-detection dimension (it's an FL-infrastructure paper, not an AD paper).

---

## 7. Comparison With Our Project

**Legend:** ✅ Implemented in current project (verified against real code/docs) · 🔶 Planned/proposed, not yet implemented · ❌ Not present/not planned

| Dimension | FedSTAR (Paper 4) | Our Project | Same? | Different? | Improvement/Extension? |
|---|---|---|---|---|---|
| Dataset | STIN/TON_IoT/Bot-IoT (benchmark, not real satellite traffic) | ✅ Track A: NSL-KDD (real, terrestrial), Dirichlet-simulated as 8 satellite clients. ✅ Track B: OPSSAT-AD (real, single spacecraft telemetry) | Both use non-native-satellite or simulated data | Our project separately validates on **real spacecraft telemetry** (Track B); FedSTAR never touches real satellite data of any kind | Track B is a genuine point of difference, not an improvement on Track A's own realism |
| Satellite scenario | 20 simulated clients from centralized benchmark data | ✅ 8 simulated clients (NSL-KDD, Dirichlet) | Same general simulation approach | Fewer clients (8 vs 20); different source dataset | Neither is more "realistic" — both simulate |
| Client simulation method | Dirichlet, η=0.5 | ✅ Dirichlet, α=0.5 (+ full α-sweep 0.1–10.0 in a dedicated Non-IID study) | Same technique | Our project ran a **controlled multi-α sweep** (5 values) isolating heterogeneity as a single variable; FedSTAR reports one η value per experiment | Our project's sweep is a methodological extension not present in FedSTAR |
| Non-IID strategy | SAC-driven clustering + hybrid weighting | ✅ Phase 8: fixed rule; ✅ Phase 9B: DQN-learned profile selection | Same underlying motivation (plain FedAvg fails under Non-IID) | Different mechanism: clustering+continuous knobs (FedSTAR) vs. discrete weight-profile catalogue, no clustering (ours) | Ours adds an explicit, same-pipeline **fixed-rule intermediate stage** FedSTAR doesn't report |
| Local model | SimVPv2 (pseudo-video + ConvSC + gSTA/LKA) | ✅ BaselineMLP (Track A, 121→128→64→1); ✅ TemporalGRUAttention (Track B, `sequence_length=1`) | Both are deep learning classifiers | FedSTAR's model is far heavier/more specialized (spatio-temporal, multi-block) | 🔶 A spatio-temporal upgrade to Track A's model is a plausible, currently-unimplemented future extension |
| Global model | Aggregated + fused SimVPv2 | ✅ Aggregated BaselineMLP (Phases 6–9B) | Same aggregation concept | Simpler architecture | — |
| Federated aggregation | Two-level: intra-cluster weighted avg → cluster fusion → global-local momentum blend | ✅ FedAvg (Phase 6) → weighted aggregation via `aggregate_with_weights` (Phase 8/9B) | Both weighted-averaging based | No clustering step in ours; single-level aggregation | — |
| Client selection | Sampling ratio ρ_t within clusters (some clients per round may be excluded from a cluster's sample) | ✅ **Weighting only — all 8 clients participate every round**, by deliberate design | Different philosophy | FedSTAR can effectively under-sample some clients; ours never drops a client (explicit fairness-preserving design choice, documented) | Ours is a deliberate, disclosed trade-off (safer for a first DRL pilot), not automatically superior |
| DRL algorithm | SAC (continuous + discrete hybrid) | ✅ DQN (discrete-only, 9 fixed profiles) | Both DRL | SAC handles continuous actions; DQN doesn't | Ours is deliberately simpler/more explainable, not more capable |
| State | Client-similarity matrix (classifier-head cosine similarity) + loss/acc history + previous action/k | ✅ 44-dim: performance/data/resource/fairness signals ×8 clients + val-F1 trend + round progress + previous action one-hot | Both use structured, server-side-only signals + memory of the previous action | Different signal composition (raw-parameter similarity vs. derived per-client signal families) | Ours arguably more interpretable (named signal families vs. an abstract similarity matrix) |
| Action | Hybrid: clustering algorithm + k + 3 continuous knobs | ✅ Discrete: pick 1 of 9 named weight-profiles | Both are structural/control decisions about the aggregation process, not detection itself | FedSTAR: 5 algorithms × up to 6 clusters × continuous space. Ours: a fixed catalogue of 9 | — |
| Reward | Accuracy gain − degradation penalty − intra-cluster divergence penalty | ✅ ΔF1 − fairness-spread penalty − action-instability penalty | Structurally analogous (gain − regression-penalty − a "coherence" penalty) | Different specific terms (divergence vs. fairness-spread; no explicit stability term in FedSTAR's formula as given) | — |
| Adaptive mechanism | SAC picks clustering+hyperparameters every round | ✅ Phase 8 (rule) → ✅ Phase 9B (DQN) | Same goal | Ours ran and reported **both** a fixed-rule stage and a learned stage on the identical pipeline | This staged, controlled comparison is not present in FedSTAR's paper |
| Temporal modeling | Pseudo-video (T=4 synthetic time-steps from 1D features) | ✅ GRU+attention (Track B), but currently run with `sequence_length=1` — **not genuinely exercising temporal recurrence**, disclosed explicitly by this project's own documentation | Both claim temporal modeling | FedSTAR's temporal dimension is synthetic (invented from 1D features); ours is architecturally present but not yet genuinely exercised in its current Track B run | 🔶 Extending Track B to real multi-timestep sequences is planned/proposed, not yet done |
| Spatio-temporal modeling | Yes — ConvSC (spatial) + gSTA/LKA (spatio-temporal) | ❌ Not implemented — our current architecture has no genuine spatial dimension | FedSTAR is more advanced here | — | 🔶 A genuine spatio-temporal upgrade is a clear, currently-unimplemented extension opportunity |
| Attention mechanism | LKA (Large Kernel Attention) | ✅ Present in `TemporalGRUAttention` (Track B), conceptually similar to Paper 1's attention-over-LSTM, not LKA-style | Both use attention | Different attention mechanism family (LKA vs. standard additive/learned-weight attention) | — |
| Anomaly detection | Supervised multi-class classification | ✅ Track A: supervised binary classification (network intrusion). ✅ Track B: supervised classification (telemetry fault, segment-level) | Both supervised classification (not forecasting-residual, unlike Papers 1–3) | — | — |
| Communication efficiency | Explicitly measured (MiB/round), compared vs. baselines | ✅ Measured (~14.6 MB/10 rounds, all 8 clients every round, unchanged across Phases 6–9B) | Both measure this | FedSTAR's clustering can, in principle, reduce effective communication by sampling within clusters; ours doesn't reduce communication (by design, for fairness) | — |
| Evaluation | Accuracy/Precision/Recall/F1, 5 seeds, statistical significance testing | ✅ Accuracy/Precision/Recall/F1/ROC-AUC, **single seed**, no significance testing | Both use standard classification metrics | FedSTAR's 5-seed, significance-tested methodology is more rigorous on this dimension | This is a genuine, disclosed shortfall in our project relative to FedSTAR, not a strength |
| Security | Standard FL privacy framing; Byzantine robustness explicitly deferred to future work | ✅ Privacy boundary **mechanically tested** (AST-based source-code inspection confirming the server/DRL modules never import pandas or read a data file); ❌ Byzantine robustness not addressed | Both address basic privacy; neither addresses Byzantine robustness | Our privacy claim is enforced by an automated test, not just design intent — a genuine methodological strength | Byzantine robustness remains equally open in both |
| Explainability | Named, inspectable actions (clustering algorithm choice logged) but continuous knobs are less directly interpretable | ✅ Fully named, inspectable weight-profiles (9 fixed, human-readable options); action distribution explicitly logged and reported, including the round-10 frozen-policy trace | Both log/report agent behavior | Ours has a smaller, more exhaustively enumerable action space, arguably easier to fully audit | — |

---

## 8. Research Gap

Restating and sharpening (with the new Section 4 findings folded in) the gap analysis from `04-research-gap/identified-gaps.md`:

**The requested exact combination check:** *Satellite networks + Federated learning + Adaptive client/resource selection + Deep reinforcement learning + Temporal modeling + Spatio-temporal attention + Non-IID data* — **no single paper found combines all seven.** FedSTAR comes closest (6 of 7 — satellite-*flavored* networks, FL, adaptive/SAC-driven clustering+resource ratio, DRL, temporal via pseudo-video, spatio-temporal attention via LKA, Non-IID via Dirichlet) but fails on **real satellite data** (uses benchmark traffic, not operational traces) and does not address **Byzantine robustness**. FLOGA-AD appears to match a similar profile with genetic-algorithm selection instead of DRL (full text not yet confirmed). No paper combines this with **genuine telemetry-fault-style anomaly detection** — that entire dimension (Papers 1–3's domain) has zero federated or DRL treatment anywhere found.

**Sharpest, most defensible gap for this project specifically:**
1. **No paper — including this project — validates a federated + DRL satellite-anomaly-detection pipeline against real, multi-satellite operational data of any kind** (telemetry or network traffic). This is field-wide, not specific to any one paper, and this project should state it as an honestly-shared limitation, not a differentiator.
2. **No paper reports a controlled, same-pipeline, three-stage comparison** (fixed → hand-written-adaptive → DRL-learned) with an **honestly-disclosed negative DRL result** and a specific, falsifiable explanation. This project already has exactly this — a genuinely distinctive methodological artifact, even though it's a modest, not a headline-performance, contribution.
3. **Byzantine/adversarial-client robustness** remains open in FedSTAR, in FLOGA-AD (unconfirmed), and in this project alike — legitimate, shared future work, not a claimable gap this project currently fills.
4. **Genuine spatio-temporal modeling of satellite data combined with a *learned* (not genetic-algorithm, not fixed-rule) adaptive mechanism, validated against a real spacecraft-telemetry track** — a combination no paper found attempts, and one this project's own two-track structure is positioned to eventually attempt, but has **not yet** attempted (Track A and Track B remain unmerged).

---

## 9. Proposed Novelty (safe, defensible, IEEE-paper-ready statement)

> "This work presents a deep reinforcement learning-driven adaptive federated learning framework for satellite network anomaly detection, evaluated through a controlled, staged comparison against both a fixed (FedAvg) baseline and a hand-engineered adaptive weighting rule on an identical pipeline and client partition. Unlike prior work combining federated learning and deep reinforcement learning for satellite-network anomaly detection (e.g., Chen et al. 2026; Wang et al. 2026), which report a single proposed system's performance gains, this work additionally reports the DRL controller's result honestly relative to both baselines — including a case where the learned policy did not surpass the best hand-engineered rule — together with a specific, falsifiable hypothesis (limited training-partition diversity) for the observed gap, verified through a mechanically-enforced privacy boundary (automated source-code inspection, not only a design claim). The framework additionally incorporates a separate, honestly-scoped validation against real single-spacecraft telemetry data, a track absent from comparable federated-DRL satellite-anomaly-detection literature, without claiming this validation demonstrates the federated mechanism's performance on real multi-satellite data — a distinction this work maintains explicitly rather than conflating."

**What this novelty statement deliberately does NOT claim:**
- It does **not** claim to be the first system combining FL + DRL + satellite anomaly detection (FedSTAR/FLOGA-AD precede or coincide with it).
- It does **not** claim the DQN-driven approach outperforms prior rule-based or SAC-based methods on accuracy.
- It does **not** claim validation on real, multi-satellite federated data.
- It does **not** claim to solve Byzantine/adversarial robustness.

**What it safely does claim:** methodological rigor (staged, controlled, honestly-reported comparison), a mechanically-verified (not merely asserted) privacy boundary, and an explicit, disclosed two-track scope with a real-spacecraft validation component absent from the closest competing papers.

---

## 10. Paper → Idea → Implementation Mapping (HIGH-relevance papers only)

```text
Paper 4 (FedSTAR, Chen et al. 2026)
  ↓ Idea taken: adaptive federated aggregation controlled by a DRL agent, reacting to
    per-round signals rather than a fixed rule
  ↓ Why useful: directly answers Phase 8's own finding that no single fixed rule wins
    on both F1 and fairness simultaneously
  ↓ Where it appears in our architecture: src/drl/ (dqn_agent.py, state.py, action_space.py,
    reward.py), src/federated/drl_environment.py, drl_round_provider.py
  ↓ Implementation phase: Phase 9B (DQN adaptive FL) — COMPLETE, results measured
    and honestly reported (DQN did not beat Phase 8's best rule in the one run so far)

Paper 4 (FedSTAR) — reward-shaping pattern specifically
  ↓ Idea taken: multi-term reward = performance gain − stability/coherence penalty
  ↓ Why useful: prevents the agent from oscillating or gaming a single metric
  ↓ Where it appears: configs/drl.yaml (lambda_fair, lambda_stab), src/drl/reward.py
  ↓ Implementation phase: Phase 9B

Paper 4 (FedSTAR) — Dirichlet Non-IID partitioning
  ↓ Idea taken: use a tunable Dirichlet concentration parameter to simulate
    heterogeneous client data from one centralized dataset
  ↓ Why useful: standard, controllable, reproducible way to create a Non-IID testbed
    without real multi-satellite data (which doesn't exist publicly)
  ↓ Where it appears: src/simulation/partitioner.py
  ↓ Implementation phase: Phase 4 (satellite client simulation), extended in Phase 7
    (full alpha sweep)

Paper 1 (Yang et al. 2021) — attention-over-LSTM
  ↓ Idea taken: let the model learn which past time-steps matter most for the
    current prediction
  ↓ Why useful: improves temporal modeling beyond a plain recurrent unit's own
    (imperfect) memory gating
  ↓ Where it appears: src/models/ TemporalGRUAttention (Track B)
  ↓ Implementation phase: Phase 3 (temporal anomaly detection)

Paper 3 (Hundman et al. 2018) — forecast-residual-threshold recipe
  ↓ Idea taken: the general "predict normal behavior, flag large deviations"
    detection philosophy
  ↓ Why useful: foundational template for any telemetry-style detector
  ↓ Where it appears: conceptually informs Track B's detection framing (though the
    current OPSSAT-AD implementation is classification-based, not residual-forecasting
    — disclosed explicitly by this project's own Phase 9A documentation)
  ↓ Implementation phase: Phase 9A (OPSSAT-AD real spacecraft validation)

FLOGA-AD (Wang, Cao & Di, 2026) — genetic-algorithm client selection (alternative to DRL)
  ↓ Idea taken (as a CONTRAST point, not reused): a non-DRL adaptive mechanism
    (genetic algorithm) can also address Non-IID client weighting
  ↓ Why useful: strengthens the justification for why DRL specifically was chosen
    over other adaptive-optimization families — worth citing as "an alternative
    adaptive mechanism we did not pursue, and why"
  ↓ Where it appears: not implemented — a literature-review discussion point only
  ↓ Implementation phase: N/A (discussion/related-work only)
```

---

## 11. Phase-wise Literature Support

### 11.1 Reconciling the prompt's phase framing with the project's actual, current phase structure

This task described the project's phases as: 1 (dataset/preprocessing) → 2 (baseline AD) → 3 (temporal AD) → 4 (simulated satellite clients) → 5 (local training/Non-IID FL experiments) → *later* (adaptive FL, DRL). **This project's own committed documentation shows more has already been built** than that framing implies: rule-based adaptive FL and a fully-executed DQN-based adaptive FL experiment (with real, measured, honestly-reported results, including a negative one) both already exist. The table below uses the **actual, current** phase numbers (matching `docs/project-progress/00-project-plan.md`) and notes the prompt's simpler label in parentheses where it maps.

| Actual phase | Prompt's label (approx.) | Papers/sources supporting the design | Methodology borrowed/inspired by | Why the design choice is reasonable | What's novel in our implementation |
|---|---|---|---|---|---|
| Phase 1 — Dataset & preprocessing (NSL-KDD) | "Phase 1" | General FL-IDS literature convention (FL-IDS survey, ACM CSUR) of using established intrusion datasets to validate new mechanisms | Standard train/val/test discipline | NSL-KDD is a well-understood, thoroughly-studied benchmark, appropriate for validating a novel FL/DRL pipeline before attempting a less-understood dataset | N/A — a standard, defensible choice, not itself novel |
| Phase 2 — Baseline detection (MLP) | "Phase 2" | Standard supervised-classification framing used by FedSTAR/FLOGA-AD for network-traffic anomaly detection | Simple, explainable classifier as a floor reference | Establishes a non-federated, non-adaptive floor before adding complexity | N/A |
| Phase 3 — Temporal model (GRU+Attention) | "Phase 3" | Yang et al. 2021 (attention-over-LSTM); conceptually related to FedSTAR's LKA (a different attention mechanism, same motivation: better long-range/structural feature use) | Attention mechanism generally | Attention is a well-validated way to improve temporal feature use, independently confirmed by two different papers in two different domains (telemetry vs. network traffic) | The specific GRU+attention combination (vs. Yang et al.'s stacked-LSTM+attention or FedSTAR's LKA) — an implementation choice, not a novel mechanism in itself |
| Phase 4 — Simulated satellite client environment | "Phase 4" | FedSTAR's (and FLOGA-AD's) Dirichlet-based client simulation from a centralized dataset | Dirichlet partitioning, measured (not assumed) via Jensen-Shannon distance | No real multi-satellite public dataset exists (a field-wide gap, Section 8) — simulation is the only currently available option, and this project measures its severity rather than assuming it | Simulated *resource* conditions (bandwidth/latency/compute/availability/connectivity) alongside label-distribution skew — FedSTAR's Non-IID simulation is label-skew only; adding a resource dimension is this project's own extension |
| Phase 5 — Local training (independent, no FL) | "Phase 5 (partial)" | Standard FL-literature practice of reporting an independent-local-training floor before FedAvg | — | Needed to know whether FedAvg is actually adding value over training each client alone | N/A |
| Phase 6 — Federated learning (FedAvg) | "later" per prompt; already done | McMahan et al. 2017 (FedAvg, cited directly in FedSTAR's own reference list) | Standard synchronous FedAvg | The universally-used FL starting point every adaptive method (including FedSTAR's) is compared against | N/A — a deliberately unmodified, standard baseline |
| Phase 7 — Non-IID FL experiments (alpha sweep) | "Phase 5 (Non-IID part)" | FedSTAR's Dirichlet Non-IID framing, extended | Multi-value Dirichlet sweep (0.1/0.5/1.0/5.0/10.0), measuring JS distance and client fairness gap at each value | Isolates heterogeneity as a single controlled variable — stronger evidence than a single fixed η, as FedSTAR uses | The controlled, multi-point sweep (vs. FedSTAR's single η=0.5 report) and the discovery that KDDTest+ F1 is **non-monotonic** in α — a genuine, own measured finding, not from any paper |
| Phase 8 — Rule-based adaptive FL | "later" per prompt; already done | FedSTAR's/RAFL's general "some clients deserve more trust than others" framing, implemented here as a fixed, transparent formula (not learned) | Weighted linear combination of performance/data/resource/fairness signals | Establishes whether adaptive weighting helps *at all* before adding the complexity/cost of a learned (DRL) controller | The specific finding that no single fixed rule wins on both F1 and fairness — the direct empirical motivation for attempting Phase 9B |
| Phase 9A — Real spacecraft telemetry validation (OPSSAT-AD) | not in prompt's list | Hundman/Yang/Maleki Sadr's shared "validate on real spacecraft data" standard | Reuses the Phase 3 GRU+attention architecture on real, independently-verified ESA telemetry | Directly answers "has anything in this project been tested on real spacecraft data" — a distinguishing point relative to FedSTAR/FLOGA-AD, which use no real satellite data at all | Honest disclosure that `sequence_length=1` means genuine temporal modeling is not yet exercised here — a rare degree of self-critical precision |
| Phase 9B — DQN-driven adaptive FL | "later" per prompt; already done | FedSTAR's DRL-controls-aggregation-policy pattern, implemented with DQN instead of SAC and a fixed discrete catalogue instead of clustering+continuous knobs | DQN (44-dim state → 64→64 MLP → 9 Q-values), replay buffer, target network, epsilon-greedy | A smaller, fully-discrete, fully-explainable action space matches this project's capstone-appropriate scope, and is directly comparable (same pipeline, same partition) to Phases 6 and 8 | The controlled three-way comparison itself, and the honestly-reported negative result with a specific, falsifiable explanation (training-partition diversity) |

---

## 12. Top 10–15 Papers, Ranked

1. **Chen et al. 2026 (FedSTAR)** — Information Sciences. Most important: nearly identical title/architecture to this project; the essential primary comparator for every claim this project makes.
2. **Wang, Cao & Di 2026 (FLOGA-AD)** — Expert Systems with Applications. Equally close in domain/dataset to FedSTAR, and is FedSTAR's own strongest baseline — arguably under-weighted so far; recommend obtaining full text urgently.
3. **Hundman et al. 2018** — KDD/ACM. Foundational telemetry-AD method and dataset origin; establishes the "other half" of the literature this project must be able to explain and differentiate from.
4. **"Adaptive Federated Learning for Anomaly Detection in Satellite Telemetry"** (repository listing, details unverified) — potentially the single paper closest to bridging this project's own two tracks; must be located and confirmed.
5. **McMahan et al. 2017 (FedAvg)** — the foundational algorithm every phase of this project (6 onward) and both FedSTAR/FLOGA-AD build directly on top of; essential citation for any methodology section.
6. **Yang et al. 2021** — extends Hundman's method with attention + multi-scale detection; directly informs the reasoning behind this project's own Phase 3 architecture choice.
7. **Maleki Sadr et al. 2022** — adds uncertainty estimation (MC Dropout); the strongest concrete "future work" citation available for extending this project's own detectors.
8. **Haarnoja et al. 2018 (Soft Actor-Critic)** — the DRL algorithm FedSTAR uses; essential for explaining, by contrast, why this project chose DQN instead.
9. **DFedSat (arXiv 2407.05850)** — the closest-found paper addressing genuine orbital-realism (intermittent inter-satellite links), a gap FedSTAR itself admits and this project shares.
10. **IEEE "Adaptive Client Selection in Resource Constrained FL Systems: A DRL Approach"** (doc 9478892) — closest non-satellite methodological analog to this project's own Phase 9B goal; useful for showing the DRL-for-client-treatment idea is a recognized pattern beyond just FedSTAR.
11. **ACM Computing Surveys, "Survey on Federated Learning for Intrusion Detection System"** — best available broad, citable survey anchor for framing the FL+IDS field in an introduction/related-work section.
12. **IEEE "Robust Peer-to-Peer FL With DRL-Based Client Selection Against Data Poisoning"** (doc 11008679) — most directly relevant candidate for a "future work: Byzantine robustness" citation.
13. **RAFL (reputation-aware FL, LEO satellites)** — closest-found conceptual cousin to this project's own performance/resource/fairness signal-weighting approach, specifically in a satellite context.
14. **FedProx / SCAFFOLD** (foundational Non-IID mitigation) — essential "other families of Non-IID mitigation not attempted here" citations for a thorough related-work section.
15. **Mnih et al. 2015, "Human-level control through deep reinforcement learning" (Nature, DQN)** — the foundational DQN paper underlying this project's own Phase 9B algorithm; essential methodology-section citation.

---

## 13. Recommended IEEE Literature Review Structure

```text
I.   Introduction
     A. Motivation: satellite network growth, expanding attack surface, telemetry-fault risk
     B. Two problem framings: physical/telemetry anomalies vs. network/cyberattack anomalies
     C. Why centralized training is infeasible (bandwidth, privacy)
     D. Contribution statement (using Section 9's safe novelty language)

II.  Related Work
     A. Telemetry-based spacecraft anomaly detection (Hundman 2018 → Yang 2021 → Maleki Sadr 2022)
     B. Federated learning fundamentals and Non-IID challenges (McMahan 2017; FedProx; SCAFFOLD)
     C. Federated learning for intrusion/anomaly detection (ACM CSUR survey; general FL-IDS work)
     D. Adaptive and DRL-driven federated learning (FedSTAR; FLOGA-AD; IEEE DRL client-selection
        papers; SAC foundational paper)
     E. Federated learning for satellite networks specifically (DFedSat; RAFL; ALANINE)
     F. Positioning statement: the exact gap (Section 8) and how this work is scoped relative to it

III. Methodology
     A. Dataset and simulated/real satellite client construction (Track A / Track B)
     B. Local detection model architecture
     C. Federated learning baseline (FedAvg)
     D. Rule-based adaptive weighting
     E. DQN-driven adaptive weighting (state/action/reward, architecture, training procedure)
     F. Privacy-boundary enforcement (mechanically tested, not just claimed)

IV.  Experimental Setup
     A. Three-stage comparison protocol (fixed vs. rule vs. learned)
     B. Metrics
     C. Reproducibility (seeds, held-out test discipline)

V.   Results and Discussion
     A. Three-stage comparison results, reported honestly (including the DQN's negative result)
     B. Analysis of why (training-partition diversity hypothesis)
     C. Real-spacecraft-telemetry validation (Track B) results, with explicit scope caveats

VI.  Limitations and Threats to Validity
     (single-seed reporting; simulated Track A; unmerged tracks; no Byzantine robustness;
     no orbital-realism modeling)

VII. Conclusion and Future Work
     (diverse-partition DQN retraining; Byzantine-robust extension; spatio-temporal upgrade;
     merging Track A/B; real satellite-data validation as it becomes available)
```

---

## 14. Risks / Weak Claims We Should Avoid

- ❌ **"We are the first to combine federated learning, DRL, and satellite anomaly detection."** FedSTAR (2026) and FLOGA-AD (2026) already did, the same year.
- ❌ **"Our federated DRL system works on real satellite data."** Only Track B (no FL) uses real spacecraft data; Track A (FL+DRL) uses simulated clients from terrestrial NSL-KDD.
- ❌ **"Our DQN outperforms the state of the art."** It did not beat Phase 8's best rule-based baseline in the one completed run — this must be reported as-is, not reframed.
- ❌ **"Our results are statistically significant."** No significance testing has been performed (single-seed throughout); FedSTAR's own 5-seed methodology is more rigorous on this specific dimension — say so plainly if asked.
- ❌ **"Our system is more secure than FedSTAR's."** Both address only basic data-locality privacy; neither addresses Byzantine/adversarial robustness. Do not imply otherwise.
- ❌ **"NSL-KDD represents real satellite network traffic."** It is real, but terrestrial, data used to simulate satellite clients — always state this precisely, not loosely.
- ❌ **"Our telemetry model (Track B) demonstrates genuine temporal modeling."** The current run uses `sequence_length=1`; this project's own documentation already discloses this honestly — do not overclaim it in a paper draft.
- ❌ **"FLOGA-AD/RAFL/DFedSat details reported in Section 4/5/12 above are fully verified."** They are search-snippet-level leads only — obtain and read the actual papers before citing specific numbers from them in a submitted document.

---

## 15. Final Research Direction

**Immediate next steps, in priority order:**
1. Obtain and fully read **FLOGA-AD** (Wang, Cao & Di, 2026) — currently the largest gap in this project's own related-work depth, given how close it is to FedSTAR.
2. Attempt to locate and verify **"Adaptive Federated Learning for Anomaly Detection in Satellite Telemetry"** — if it genuinely bridges the telemetry/FL divide as its title suggests, it may be the single most important paper to add to this literature base.
3. Re-run Phase 9B's DQN pretraining with genuinely diverse partitions across episodes (the original design intent, not yet executed) to test whether the currently-observed underperformance is a training-diversity artifact, as hypothesized.
4. If pursuing an IEEE submission, adopt Section 13's structure and Section 9's exact novelty wording — do not overreach beyond it.
5. Treat Byzantine-robustness and orbital-realism (delay/intermittent links) as clearly-labeled future work, shared openly with FedSTAR, not as claims to make now.

---

## Answer to: "Your project is very similar to this paper. What exactly did you add or improve?"

**Concise, technically honest, viva-ready answer:**

> "FedSTAR uses a Soft Actor-Critic agent to choose a clustering algorithm and continuous aggregation hyperparameters for a heavy spatio-temporal network-traffic classifier, evaluated once against several baselines on benchmark (not real) satellite-relevant data. My project instead runs a **controlled, staged experiment on the identical pipeline**: plain FedAvg, then a hand-engineered adaptive weighting rule, then a DQN that learns to choose among 9 fixed, fully-named weighting profiles — no clustering, a deliberately smaller and more explainable action space appropriate to this project's scope. I report all three stages honestly, including that the DQN did **not** beat the hand-engineered rule in the one completed run, along with a specific, testable reason why (the training regime reused a single fixed data partition rather than the diverse partitions the design intended). I also separately, and honestly, validate a telemetry-style detector against one real spacecraft's actual data — something FedSTAR never attempts — while being explicit that this validation and the federated/DRL machinery are not yet combined into one system. So the contribution isn't a new performance record; it's a smaller, fully auditable DRL mechanism tested with a rigor — a same-pipeline three-way comparison, a mechanically-enforced privacy boundary, and an honestly-reported negative result — that FedSTAR's own paper doesn't report for its own SAC controller."
