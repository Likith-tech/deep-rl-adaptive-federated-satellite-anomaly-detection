# Recommended Additional Papers (Full List)

## Honesty note (read first)

**Only 4 PDF documents were ever attached to this project's research conversation** — Hundman 2018, Yang 2021, Maleki Sadr 2022, Chen et al. 2026 (FedSTAR). Every paper below was found via **live web search across two research sessions**, verified only to the depth of the search results shown (titles, venues, DOIs where surfaced, short abstract-level summaries) — **none of these have been read in full**. Where an author list, exact year, or DOI could not be confirmed from the search results, it is marked **"Not verified from the available source."** Do not cite specific numeric results from any paper below without obtaining and reading its actual full text first.

**Search method:** targeted queries run against the categories explicitly requested (satellite AD, satellite cybersecurity, satellite telemetry AD, FL for AD, federated intrusion detection, adaptive FL, DRL for FL, DRL client selection, DRL resource allocation, Non-IID FL, communication-efficient FL, LEO satellite FL, edge intelligence for satellite networks, spatio-temporal AD, attention-based AD, transformer/GRU/LSTM AD, FL+DRL+security), run 2026-08-23.

---

### A. Satellite network anomaly detection

**1. Wang, Z., Cao, J., Di, X.** — "Anomaly detection method for satellite networks based on genetic optimization federated learning" (FLOGA-AD)
- **Year:** 2026 (per ScienceDirect listing and FedSTAR's own reference [24], which cites it identically)
- **Venue:** *Expert Systems with Applications*, Vol. 295, Article 128627 (Elsevier)
- **DOI:** 10.1016/j.eswa.2025.128627 — cross-referenced from the ScienceDirect URL and FedSTAR's own bibliography; **exact DOI digits not independently re-verified against the publisher record beyond this**
- **Why relevant:** FedSTAR's own strongest Non-IID baseline; uses the *same* STIN/TON_IoT/Bot-IoT datasets and a CvT-LSTM spatio-temporal backbone, with a **genetic algorithm** (not DRL) driving client selection
- **Which gap it fills:** the "is DRL necessary, or would another adaptive-optimization family also work" question
- **Priority: HIGH — obtain the full text before writing any comparison chapter**

**2. "Adaptive Federated Learning for Anomaly Detection in Satellite Telemetry"**
- **Authors/Year/Venue:** **Not verified from the available source** — found only as a University of East London institutional repository listing (`uel-repository.worktribe.com/output/439906`)
- **Why relevant:** title suggests a direct bridge between the "telemetry" (Papers 1–3) and "federated learning" (FedSTAR) lineages this project identifies as otherwise disconnected — potentially the single most gap-relevant paper found
- **Which gap it fills:** the "no paper combines federated learning with telemetry-fault-style anomaly detection" gap
- **Priority: HIGH — must be independently located and confirmed; do not cite until verified**

---

### B. Federated learning for anomaly/intrusion detection (general survey)

**3.** "Survey on Federated Learning for Intrusion Detection System: Concept, Architectures, Aggregation Strategies, Challenges, and Future Directions"
- **Authors:** Not verified from the available source
- **Venue:** *ACM Computing Surveys*, DOI: 10.1145/3687124
- **Why relevant:** top-tier (ACM CSUR), broad survey anchor for the whole FL+IDS field
- **Which gap it fills:** general related-work framing/background depth
- **Priority: MEDIUM**

---

### C. Adaptive federated learning (general, not satellite-specific)

**4.** "AdaptFL: Adaptive Federated Learning Framework for Heterogeneous Devices" / "FedDA: Resource-adaptive federated learning with dual-alignment aggregation optimization"
- **Authors/exact year:** Not verified from the available source
- **Venue:** ScienceDirect (*Future Generation Computer Systems* family journals)
- **Why relevant:** a different flavor of "adaptive" (per-device *model-size* adaptation) than either FedSTAR's or our project's weighting-based adaptation
- **Which gap it fills:** background contrast for the "adaptive federated learning" concept note
- **Priority: LOW**

---

### D. DRL for federated learning

**5.** "Adaptive Client Selection in Resource Constrained Federated Learning Systems: A Deep Reinforcement Learning Approach"
- **Authors:** Not verified from the available source
- **Venue:** IEEE Xplore, document 9478892
- **Why relevant:** closest non-satellite methodological analog to our own Phase 9B goal (a DRL agent deciding how to treat clients each round)
- **Which gap it fills:** evidence the DRL-controls-FL-process pattern is field-recognized, not unique to FedSTAR/our project
- **Priority: HIGH**

**6.** "Robust Peer-to-Peer Federated Learning With Deep Reinforcement Learning Based Client Selection Against Data Poisoning Attacks"
- **Authors:** Not verified from the available source
- **Venue:** IEEE Xplore, document 11008679
- **Why relevant:** directly addresses the Byzantine/adversarial-client robustness gap FedSTAR itself names as unsolved (twice, in its own text)
- **Which gap it fills:** the "no paper addresses malicious/Byzantine clients" gap
- **Priority: HIGH**

**7.** "FedAEB: Deep Reinforcement Learning Based Joint Client Selection and Resource Allocation Strategy for Heterogeneous Federated Learning"
- **Authors:** Not verified from the available source
- **Venue:** IEEE Xplore, document 10416384
- **Why relevant:** uses **Soft Actor-Critic** (same DRL family as FedSTAR) for a non-satellite FL client-selection/resource-allocation problem
- **Which gap it fills:** independent confirmation that SAC is a standard, recurring choice for this problem class
- **Priority: MEDIUM**

---

### E. DRL client selection

**8.** "A Federated Learning Client Selection Method via Multi-Task Deep Reinforcement Learning"
- **Authors:** Not verified from the available source
- **Venue:** IEEE Xplore, document 10840140
- **Why relevant:** multi-task DRL with two reward terms (model performance + system latency) — a directly comparable multi-objective reward-shaping precedent to both FedSTAR's and our project's multi-term reward
- **Which gap it fills:** reward-design methodology background
- **Priority: MEDIUM**

**9.** "FedPPO: Reinforcement Learning-Based Client Selection for Federated Learning With Heterogeneous Data"
- **Authors:** Not verified from the available source
- **Venue:** IEEE Xplore (PDF path: iel8/6687307/11288933/10909702.pdf)
- **Why relevant:** uses **PPO** — a third DRL algorithm family alongside our DQN and FedSTAR's SAC — with a noisy-client-filtering pre-step (PCA + hierarchical clustering) to shrink the action space before DRL
- **Which gap it fills:** evidence for "why DQN, not PPO" discussions
- **Priority: MEDIUM**

---

### F. Resource-aware federated learning

**10.** "Ed-Fed: A generic federated learning framework with resource-aware client selection for edge devices"
- **Authors:** Not verified from the available source
- **Venue:** arXiv preprint (peer-review status not verified)
- **Why relevant:** resource-awareness (bandwidth/compute/availability) as an explicit selection signal — directly analogous to our own project's `resource_k` signal family (Phase 8/9B)
- **Which gap it fills:** independent confirmation our `resource_k` signal is a recognized, standard FL signal type
- **Priority: MEDIUM**

---

### G. Non-IID federated learning

**11. FedProx** — Li, T., Sahu, A.K., Zaheer, M., Sanjabi, M., Talwalkar, A., Smith, V.
- **Year/Venue:** widely cited as MLSys 2020 (**not independently re-verified via this search pass** — extremely well-established in the FL literature)
- **Why relevant:** foundational proximal-term-based Non-IID mitigation, an alternative family to the clustering/weighting approach both FedSTAR and our project use
- **Priority: MEDIUM (foundational background)**

**12. SCAFFOLD** — Karimireddy, S.P., Kale, S., Mohri, M., Reddi, S., Stich, S., Suresh, A.T.
- **Year/Venue:** widely cited as ICML 2020 (**not independently re-verified via this search pass**)
- **Why relevant:** control-variate variance-reduction approach to client drift under Non-IID data — another alternative family neither FedSTAR nor our project attempts
- **Priority: MEDIUM (foundational background)**

**13.** "Non-IID data in Federated Learning: A Survey with Taxonomy, Metrics, Methods, Frameworks and Future Directions"
- **Authors:** Not verified from the available source
- **Venue:** arXiv:2411.12377
- **Why relevant:** broad, recent survey anchor for the Non-IID sub-field
- **Priority: LOW**

---

### H. Satellite/space communication security

**14.** "A Comprehensive Literature Review of Cybersecurity in Satellite Networks"
- **Authors:** Not verified from the available source
- **Venue:** *Aerospace* (MDPI), Vol. 13, No. 3, Article 249
- **Why relevant:** peer-reviewed survey specifically on satellite cybersecurity — good general background, independent of the FL/DRL angle
- **Priority: MEDIUM**

**15.** "Cyber Attacks on Space Information Networks: Vulnerabilities, Threats, and Countermeasures for Satellite Security"
- **Authors:** Not verified from the available source
- **Venue:** MDPI, Vol. 5, No. 3, Article 76
- **Why relevant:** similarly useful general-background survey on satellite-specific threats
- **Priority: LOW**

---

### I. Spatio-temporal anomaly detection

**16.** Family of spatio-temporal graph neural network (STGNN) papers for network anomaly detection (e.g., graph-convolutional-recurrent architectures)
- **Specific citation:** **Not verified from the available source at a single-paper level** — this search pass surfaced a *family* of related work, not one clearly primary paper
- **Why relevant:** an alternative spatio-temporal modeling architecture family (graph-based) distinct from FedSTAR's pseudo-video/convolutional approach and our own GRU+attention approach
- **Priority: LOW (background/contrast only — do not cite a specific paper from this entry without independently locating and reading one)**

**17.** "Transformer-based multivariate time series anomaly detection using inter-variable attention mechanism"
- **Authors:** Not verified from the available source
- **Venue:** *Knowledge-Based Systems* (Elsevier), DOI: 10.1016/j.knosys.2024.111507
- **Why relevant:** attention *across variables* (a spatial-like dimension) for time series anomaly detection — a genuinely relevant alternative to FedSTAR's LKA and our own attention mechanism
- **Priority: MEDIUM**

---

### J. Attention-based anomaly detection

No additional high-priority paper found beyond what Paper 1 (Yang et al.) and FedSTAR's own LKA mechanism already cover in this project's core 4 papers, plus entry 17 above.

---

### K. Federated learning + satellite networks (genuinely orbital, not just benchmark-flavored)

**18.** "DFedSat: Communication-Efficient and Robust Decentralized Federated Learning for LEO Satellite Constellations"
- **Authors:** Not verified from the available source
- **Venue:** arXiv:2407.05850 (peer-review status not verified)
- **Why relevant:** genuinely models orbital constraints — Doppler shifts, inter-satellite-link unreliability, intra-plane/inter-plane aggregation — directly fills FedSTAR's own admitted "orbital realism" gap
- **Priority: HIGH**

**19.** "RAFL: Reputation-Aware Federated Learning with hierarchical aggregation in LEO satellite networks"
- **Authors:** Not verified from the available source
- **Venue:** ScienceDirect (exact journal title not confirmed from search snippet)
- **Why relevant:** reputation/trust-based client weighting for satellite FL — close conceptual cousin of our own performance/resource/fairness signal-weighting approach
- **Priority: MEDIUM**

**20.** "ALANINE: A Novel Decentralized Personalized Federated Learning For Heterogeneous LEO Satellite Constellation"
- **Authors:** Not verified from the available source
- **Venue:** arXiv:2411.07752
- **Why relevant:** personalized, fully decentralized (no single server) FL for satellites — a different architectural philosophy than FedSTAR's or our project's centralized-server design
- **Priority: LOW**

**21.** "Secure and Efficient Federated Learning in LEO Constellations using Decentralized Key Generation and On-Orbit Model Aggregation"
- **Authors:** Not verified from the available source
- **Venue:** arXiv:2309.01828
- **Why relevant:** satellite-specific security/key-management for on-orbit aggregation
- **Priority: LOW**

---

### L. DRL + FL + security (non-satellite analogs)

**22.** "A Federated Learning Framework with Self-Attention and Deep Reinforcement Learning for IoT Intrusion Detection"
- **Authors:** Not verified from the available source
- **Venue:** ACM, *Proceedings of the 2024 13th International Conference on Software and Information Engineering (ICSIE)*, DOI: 10.1145/3708635.3708649
- **Why relevant:** combines self-attention + DRL + FL + IDS in one system for IoT (not satellites) — the closest non-satellite full analog to our project's own overall combination
- **Priority: MEDIUM**

**23.** "Collaborative privacy-preserving network intrusion detection: A federated multi-agent reinforcement learning approach"
- **Authors:** Not verified from the available source
- **Venue:** ScienceDirect
- **Why relevant:** **multi-agent** RL (one agent per attack-type detector) — a structurally different DRL-FL combination than either FedSTAR's or our project's single-agent design
- **Priority: LOW**

---

### M. Edge/IoT anomaly detection using FL + DRL

**24.** "FedAtten-DRL" (name only, mentioned within a search-result summary)
- **Authors/venue:** **Not verified from the available source** beyond the search-summary mention — flagged for independent confirmation before citing at all
- **Priority: LOW (unconfirmed — do not cite without locating the actual paper)**

---

### Communication efficiency (cross-cutting, not one of the 13 lettered categories but explicitly relevant)

**25.** "Towards Efficient Communications in Federated Learning: A Contemporary Survey"
- **Authors:** Not verified from the available source
- **Venue:** arXiv:2208.01200
- **Why relevant:** broad survey anchor for gradient compression/quantization/sparsification techniques, none of which FedSTAR or our project currently uses
- **Priority: LOW**

---

## Foundational papers (well-established, cited for methodology sections — not found via this session's search, but independently verified/cross-checked)

**26. McMahan, B., Moore, E., Ramage, D., Hampson, S., Arcas, B.A.Y.** — "Communication-efficient learning of deep networks from decentralized data" (**FedAvg**)
- **Year/Venue:** AISTATS 2017 — cited verbatim in FedSTAR's own reference list (ref [11]), which is the direct source of this citation, not an independent search
- **Priority: ESSENTIAL** — the foundational algorithm every FL phase of our project builds on

**27. Haarnoja, T., Zhou, A., Abbeel, P., Levine, S.** — "Soft actor-critic: off-policy maximum entropy deep reinforcement learning with a stochastic actor" (**SAC**)
- **Year/Venue:** ICML 2018 — cited verbatim in FedSTAR's own reference list (ref [31])
- **Priority: ESSENTIAL** — the DRL algorithm FedSTAR uses; essential contrast point for justifying our own DQN choice

**28. Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A.A., Veness, J., Bellemare, M.G., et al.** — "Human-level control through deep reinforcement learning" (**DQN**)
- **Year/Venue:** *Nature*, 518(7540):529-533, 2015 — independently confirmed via live web search this session (title, journal, volume/issue/pages, DeepMind authorship all corroborated)
- **Priority: ESSENTIAL** — the foundational algorithm our own Phase 9B implements

---

## Recommendation

**Do we genuinely need more papers?** A small number, yes — not dozens. Highest priority: obtain the **full text of FLOGA-AD** (#1) and confirm/locate the **"Adaptive FL for Satellite Telemetry"** paper (#2) — these two would materially strengthen the literature base. Everything else above is legitimate background/contrast material, correctly graded MEDIUM or LOW, and should not be over-weighted relative to the 4 core papers actually read in full.
