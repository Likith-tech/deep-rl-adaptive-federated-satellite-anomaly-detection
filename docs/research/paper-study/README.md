# Satellite Anomaly Detection — Research Knowledge Base

This folder exists to help you (the author of *Deep Reinforcement Learning-Driven Adaptive Federated Framework for Satellite Network Anomaly Detection*) genuinely **understand** the research literature behind your capstone — not just have a literature review written for you. Every claim here traces back either to one of the 4 attached papers, to an independently-verified web search (confidence level always stated), or to this project's own existing, verified documentation (`docs/project-progress/`, `docs/research/`, and the real source code in `src/drl/` and `src/federated/`).

## ⚠️ Honest note on paper count — read this first

**You have referred to "10 papers already provided" across several requests. Only 4 PDF documents have ever actually been attached to this conversation:**

1. Hundman et al. (2018) — *Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding* (KDD, ACM)
2. Yang et al. (2021) — *Improved Deep Learning Based Telemetry Data Anomaly Detection to Enhance Spacecraft Operation Reliability* (Microelectronics Reliability, Elsevier)
3. Maleki Sadr et al. (2022) — *An Anomaly Detection Method for Satellites Using Monte Carlo Dropout* (IEEE TAES)
4. Chen et al. (2026) — *Anomaly Detection Method for Satellite Networks Based on Adaptive Federated Learning Driven by Deep Reinforcement Learning* (FedSTAR — *Information Sciences*, Elsevier)

Rather than inventing 6 more papers' worth of content to hit "10" — which would violate the explicit instruction never to fabricate citations, results, or algorithm details — this knowledge base analyzes these 4 with genuine depth, and `07-missing-papers/recommended-additional-papers.md` lists ~28 real, independently web-search-verified candidate papers (confidence-graded, none fabricated) that would strengthen the set further. **If you have the other 6 PDFs, provide them and they will be analyzed with the same depth as these 4.**

## How the papers relate to your project

Your 4 papers split into **two nearly unrelated sub-fields** that both happen to be called "satellite anomaly detection":

- **Papers 1–3 (telemetry lineage):** single-spacecraft, physical/operational fault detection, using LSTM-based forecasting. No federated learning, no reinforcement learning, in any of the three.
- **Paper 4 (FL+DRL lineage):** multi-satellite (simulated), network-traffic cyberattack detection, using federated learning + a Soft Actor-Critic reinforcement-learning agent. Published in 2026 — the same year as your capstone, with an almost identical title.

Independent search surfaced a critical fifth data point, **FLOGA-AD** (Wang, Cao & Di, 2026) — FedSTAR's own strongest baseline, same datasets, a genetic algorithm instead of DRL. See `01-paper-notes/core-paper-fedstar.md` and `07-missing-papers/recommended-additional-papers.md` (#1).

Full detail on the field split: `03-comparison/research-evolution.md`.

## Which papers are most important?

| Priority | Paper | Why |
|---|---|---|
| ⭐⭐⭐ Highest | Paper 4 (Chen et al., FedSTAR) | Nearly the same title and idea as your capstone — see the dedicated deep dive: `01-paper-notes/core-paper-fedstar.md` |
| ⭐⭐⭐ Highest (unread, urgent) | FLOGA-AD (Wang, Cao & Di, 2026) | FedSTAR's own strongest baseline — get the full text (`07-missing-papers/recommended-additional-papers.md` #1) |
| ⭐⭐ High | Paper 1 (Hundman et al.) | Foundational method + origin of the SMAP/MSL dataset used by Papers 2 and 3 |
| ⭐ Medium | Paper 2 (Yang et al.) | Extends Paper 1; conceptually related to your own Phase 3 GRU+attention architecture |
| ⭐ Medium | Paper 3 (Maleki Sadr et al.) | Extends Paper 1 differently (uncertainty); good future-work idea for your own detectors |

*(Note: papers are renumbered 1–4 in chronological/logical order in this folder's actual filenames — not the order the documents happened to be attached in.)*

## Full ranked list (top ~15, all confidence levels)

See `03-comparison/top-papers.md` for the complete ranking with confidence markers (🟢 read in full / 🟡 search-verified only / 🔵 foundational).

## Which papers provide what

- **Methodology:** all 4 (forecasting+thresholding in Papers 1–3; federated clustering+DRL in Paper 4).
- **Datasets:** Papers 1–3 (SMAP/MSL, real NASA data); Paper 4 (STIN/TON_IoT/Bot-IoT, benchmark network-traffic data).
- **Useful algorithms to know cold:** nonparametric thresholding (Paper 1/Hundman), attention (Paper 2/Yang), Monte Carlo Dropout (Paper 3/Maleki Sadr), SAC + Dirichlet Non-IID partitioning + variance decomposition (Paper 4/FedSTAR).
- **Research gap:** all 4 together — see `04-research-gap/identified-gaps.md`.

## Which are directly vs. indirectly related to your project

- **Directly related:** Paper 4/FedSTAR (same problem shape: FL + DRL + satellites) vs. our project's Track A; Papers 1–3 (telemetry forecasting AD) vs. our project's Track B (OPSSAT-AD).
- **Indirectly related:** Papers 1–3 relative to our project's Track A (FL+DRL side) — useful background on the *other* satellite-AD sub-field, not a direct methodological source.

## How to use these notes

Don't read this folder cover-to-cover in one sitting. Follow `00-research-roadmap.md`'s 10-stage reading order — built so each stage's concept notes prepare you for the paper/idea that needs them next.

---

## If I only have 10 minutes

- Your 4 papers split into two sub-fields: **telemetry** (physical spacecraft faults, Papers 1–3) and **network security** (cyberattacks, Paper 4) — treat them as separate literatures, not one smooth progression.
- Papers 1–3 all do the same basic thing — predict the next telemetry value, flag big prediction errors — and improve it step by step: better thresholding (Hundman) → attention + multi-scale (Yang) → uncertainty estimation (Maleki Sadr).
- None of Papers 1–3 use federated learning or reinforcement learning at all.
- Paper 4 (FedSTAR, 2026) is the one paper that combines FL + DRL for satellite anomaly detection — its title is almost identical to your capstone's, and it has its own strongest competitor, FLOGA-AD (2026, genetic algorithm instead of DRL).
- FedSTAR uses SAC (a DRL algorithm that handles continuous actions) to pick a clustering algorithm and aggregation hyperparameters; your project uses a smaller DQN (discrete-only) to pick among 9 fixed weighting profiles — no clustering.
- **No paper — including yours — has tested federated learning + DRL on real, multi-satellite data of any kind.** This is a genuine, field-wide gap.
- Your project already ran a real, three-stage comparison (fixed FedAvg → hand-written rule → DQN-learned policy) and honestly reported that the DQN did **not** beat the best hand-written rule in its one completed run — a legitimate, informative negative result, not a failure to hide.
- Your project also separately validated a telemetry-style detector against one real spacecraft's actual data (OPSSAT-AD) — something FedSTAR has no equivalent of, though your project's two tracks aren't combined yet.
- The single most important thing to be able to say clearly: **how your project differs from FedSTAR** — see `04-research-gap/paper-vs-our-project.md` and the closing answer in `08-full-literature-review-report.md`.
- You do not need to claim your project is "the first" to combine these ideas — FedSTAR (and FLOGA-AD) already did, the same year. Your defensible contribution is narrower and more precise.

## If I have 30 minutes

Read, in order: this README → `03-comparison/research-evolution.md` → `01-paper-notes/core-paper-fedstar.md` (skim: sections 1–3, 11–14, 21) → `04-research-gap/paper-vs-our-project.md`.

## If I have 2 hours

Follow Stages 1, 4, 9, and 10 of `00-research-roadmap.md` (skip the deep dive into Papers 1–3's individual mechanics for now — read only their one-minute summaries).

## If I have 1 day

Follow all 10 stages of `00-research-roadmap.md` in order, taking breaks between stages (total ~10–14 hours — spread across the day/multiple sessions, not one continuous sitting).

## The Big Picture

Researchers currently split "satellite anomaly detection" into two mostly-separate worlds: one mature and well-studied (predicting spacecraft telemetry with LSTMs and flagging big prediction errors, no distributed learning involved), and one brand new and just emerging (federated learning + reinforcement learning for satellite *network security*, still untested on real satellite data, still not robust to malicious clients, still not modeling real orbital communication constraints — a gap FedSTAR's own authors state explicitly, twice). Why FL is useful: satellites can't/shouldn't ship raw data to a ground server (bandwidth + privacy). Why DRL is useful: a fixed aggregation rule can't adapt round-to-round to which clients currently deserve more trust — a learned policy, in principle, can. Why adaptation matters at all: your own project's Phase 8 proved no single fixed rule wins on both accuracy and fairness simultaneously. Why satellite networks specifically add difficulty: intermittent contact windows, scarce bandwidth, and (so far, in every paper found) no real, public, multi-satellite federated dataset to validate against. **Where your project fits:** in between the two worlds — it builds and honestly tests a working FL+DRL pipeline (closer to the new world) while separately, honestly validating a telemetry-style detector against one real spacecraft's data (closer to the mature world), without yet merging the two. The project's most valuable, defensible contribution isn't "first to combine FL and DRL for satellites" (FedSTAR/FLOGA-AD got there first, the same year) — it's a **carefully controlled, honestly-reported experiment** showing a learned (DQN) aggregation policy does not automatically beat a well-designed fixed rule, with a specific, testable explanation for why, under a mechanically-tested privacy boundary. Knowing precisely how to state that, and precisely how it differs from FedSTAR, is the single most valuable thing this knowledge base gives you before your viva.

## Folder structure

```text
docs/research/paper-study/
├── README.md                              — this file
├── 00-research-roadmap.md                 — 10-stage study order
├── 01-paper-notes/
│   ├── paper-01.md                        — Hundman et al. 2018
│   ├── paper-02.md                        — Yang et al. 2021
│   ├── paper-03.md                        — Maleki Sadr et al. 2022
│   ├── paper-04.md                        — Chen et al. 2026 (FedSTAR, standard-depth note)
│   └── core-paper-fedstar.md              — FedSTAR EXTRA-deep dive (the deepest file here)
├── 02-concept-notes/                      — 18 concept explainers (FL, FedAvg, Non-IID, client
│                                             selection, adaptive FL, RL, state/action/reward, DRL,
│                                             DQN, SAC, LSTM, GRU, attention, spatio-temporal
│                                             attention, temporal modeling, communication efficiency,
│                                             satellite networks, anomaly detection, privacy/security)
├── 03-comparison/
│   ├── literature-matrix.md               — full comparison table
│   ├── research-evolution.md              — how the field actually evolved (two lineages)
│   └── top-papers.md                      — ~15 papers ranked, with confidence markers
├── 04-research-gap/
│   ├── identified-gaps.md                 — evidence-based gaps (explicit vs. observed)
│   ├── our-project-positioning.md         — component-by-component positioning
│   └── paper-vs-our-project.md            — rigorous IMPLEMENTED/PLANNED/PROPOSED comparison table
├── 05-project-connection/
│   ├── why-our-approach.md
│   ├── component-to-paper-mapping.md
│   ├── paper-to-project-mapping.md        — phase-by-phase (Phase 1→9B) literature support
│   └── important-techniques.md
├── 06-viva-preparation/
│   ├── simple-explanations.md
│   ├── likely-questions.md
│   └── paper-based-questions.md
├── 07-missing-papers/
│   └── recommended-additional-papers.md   — ~28 candidate papers, full citation format, confidence-graded
└── 08-full-literature-review-report.md    — single master IEEE-style 15-section report
```

**If you only read one file for a fast, complete refresh:** `08-full-literature-review-report.md`. **If you only read one file to prepare for the hardest viva question:** `04-research-gap/paper-vs-our-project.md`. **If you want the single deepest technical breakdown:** `01-paper-notes/core-paper-fedstar.md`.
