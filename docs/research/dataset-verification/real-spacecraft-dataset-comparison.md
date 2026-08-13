# Real Spacecraft Dataset Comparison — Phase 9 Verification

**Status:** Research artifact only. No dataset listed here has been downloaded or
integrated. See `phase-9-dataset-verification.md` for the full narrative report and
`phase-9-real-spacecraft-dataset-search-response.md` for the raw findings this table is
distilled from.

## Evidence-type key

Every factual cell below is implicitly one of:

- **FACT** — directly confirmed by fetching the primary source in this research session
  (a Zenodo record page, an official GitHub repo, a Mendeley Data page, etc.)
- **SOURCE CLAIM** — stated by the dataset's own paper/README/page, not independently
  re-derived (e.g., "105 anomaly sequences" is what the authors report, not something
  recomputed here)
- **ENGINEERING ASSESSMENT** — this project's own judgment about fit, not a claim from
  the dataset's authors (e.g., "temporal fit: strong")
- **UNCERTAINTY** — could not be confirmed from any source reached in this session;
  explicitly do not treat as settled

Where a specific cell is uncertain, it is marked `UNVERIFIED` rather than filled with a
guess.

## Main comparison table

| Dataset | Category | Real spacecraft? | Cyber labels? | Operational anomaly labels? | Timestamps | Samples | Features/channels | Missions/clients | Size | License | Official source | Temporal fit | FL fit | Cybersecurity relevance | Integration difficulty |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **NSL-KDD** *(current, reference row)* | E | No (terrestrial) | Yes | N/A | Weak | 148,517 | 121 (post-encoding) | Simulated (Phase 4, 8 clients) | ~22.5 MB | Standard academic (UNB/CIC) | github.com/jmnwong/NSL-KDD-Dataset | Weak | Simulated only | High (real attacks) | Already integrated |
| **NASA SMAP/MSL** | B | **Yes** (SOURCE CLAIM: SMAP satellite + MSL/Curiosity rover) | No | Yes (SOURCE CLAIM: 105 anomaly sequences) | Yes | 496,444 telemetry values (SOURCE CLAIM) | 82 channels (SOURCE CLAIM) | 2 real sources | Small (not stated in MB; small enough for direct GitHub/Kaggle download) | Apache 2.0 (code); data openly released (FACT: no registration wall found) | github.com/khundman/telemanom; Kaggle mirror | Strong (ENGINEERING ASSESSMENT) | Weak (2 sources) | None (operational faults only) | Easy (ENGINEERING ASSESSMENT) |
| **ESA-ADB** | B | **Yes** (SOURCE CLAIM: 3 real ESA missions) | No | Yes (SOURCE CLAIM: 844 annotated events, 148 classified anomalies, across 2 of the 3 missions per the benchmark paper) | Yes | Not fully enumerated (UNVERIFIED exact record count) | Multiple channels per mission (UNVERIFIED exact count) | 3 real missions | 11.6 GB (FACT, confirmed directly on Zenodo: 3.8+4.1+3.7 GB) | **CC BY 3.0 IGO** (FACT, confirmed directly on Zenodo) | Official: github.com/esa/anomaly-dataset (ESA org); data: zenodo.org/records/12528696; paper: arXiv:2406.17826 | Strong (ENGINEERING ASSESSMENT) | Weak-moderate (3 sources) | None (operational faults only) | Moderate — large download (ENGINEERING ASSESSMENT) |
| **STIN / SAT20+TER20** | D | No (SOURCE CLAIM/search evidence: "authors simulate a real scenario") | Yes (SOURCE CLAIM: 8 classes incl. Signal Disruption, Jamming, UDP flood, DoS, DDoS, Bruteforce, Infiltration) | No | Weak-moderate (UNVERIFIED — flow-based, exact timestamp granularity not confirmed) | UNVERIFIED exact count | UNVERIFIED exact feature count (search evidence suggested ~32, not independently confirmed) | Simulated satellite+terrestrial topology, not real multi-satellite | UNVERIFIED | **UNCLEAR / UNVERIFIED** — repo has only a bare copyright notice (FACT: "Copyright (c) 2020 National Engineering Lab for Next Generation Internet Interconnection Devices, BJTU," confirmed directly), no explicit open-source license grant found | github.com/kun9717/STIN-data-set | Weak-moderate | Moderate (ships FL train/test splits — SOURCE CLAIM) | High (labeled attacks, but simulated) | Moderate, blocked on license clarification |
| **CuCD-ID** | D | No (SOURCE CLAIM: NOS3 software-in-the-loop digital twin) | Yes (SOURCE CLAIM: 5 scenarios, SPARTA-aligned) | No | Moderate (20-second windowed statistics — SOURCE CLAIM) | 25,000 (raw) / 22,465 (augmented) (SOURCE CLAIM) | 31 (raw) / 23 (augmented) (SOURCE CLAIM) | 1 simulated CubeSat | Not stated (CSV, moderate size) | **CC BY 4.0** (FACT, confirmed directly on Mendeley Data) | data.mendeley.com/datasets/7n2d42pm3n/3 | Moderate | None (single spacecraft) | High (labeled attacks, but simulated) | Easy (ENGINEERING ASSESSMENT) |
| **LENS** | C (real traffic, no attack labels — see note below) | Yes (real Starlink hardware) | No | No | Yes | Multiple monthly snapshots (UNVERIFIED total record count) | Latency traces (not multi-channel telemetry) | 13 real dishes, 7 PoPs, 3 continents (SOURCE CLAIM) | Multi-GB, split by month (UNVERIFIED total) | **CC BY-SA 4.0** (FACT, confirmed directly on Zenodo) | zenodo.org/records/15331285; github.com/clarkzjw/LENS | Moderate (latency series, not telemetry) | Strong (real multi-client) | None (no labels at all) | Moderate (ENGINEERING ASSESSMENT) |
| **LASP Satellite-Telemetry-Anomaly-Detection** *(new)* | B, **UNCERTAIN** | Likely yes (SOURCE CLAIM: "LASP spacecraft telemetry," LASP = Laboratory for Atmospheric and Space Physics, CU Boulder — a real institution) but specific spacecraft/mission **UNVERIFIED** | No | Unclear whether genuine expert ground-truth labels exist — **UNVERIFIED** | UNVERIFIED | "Five representative time series datasets" (SOURCE CLAIM) | UNVERIFIED exact count (example columns seen: "Temperature (C)", "WheelTemperature") | UNVERIFIED | UNVERIFIED | **UNVERIFIED** — no license confirmed | github.com/sapols/Satellite-Telemetry-Anomaly-Detection | Possibly strong, **UNVERIFIED** | Weak | None | UNVERIFIED — needs firsthand repo inspection before any use |
| **CATS (Controlled Anomalies Time Series)** *(new)* | Not satellite-specific; simulated | No (SOURCE CLAIM: "simulated complex systems") | No | Yes (SOURCE CLAIM: 200 injected anomalies) | Yes | UNVERIFIED | UNVERIFIED | N/A | UNVERIFIED | UNVERIFIED | Referenced via patrickfleith/awesome-spacecraft-engineering-datasets | Moderate | N/A | None | Low priority — not spacecraft-specific |
| **AegisSat testbed** *(new)* | D | No (SOURCE CLAIM: repeated simulation campaigns, Ben-Gurion University) | Yes (SOURCE CLAIM: labeled telemetry + cyberattack logs) | Unclear | UNVERIFIED | UNVERIFIED | UNVERIFIED | Testbed, not real multi-satellite | UNVERIFIED | UNVERIFIED | dl.acm.org/doi/10.1145/3736731.3746144 | UNVERIFIED | UNVERIFIED | Moderate-high (ENGINEERING ASSESSMENT, pending verification) | UNVERIFIED — very new (2025), not independently inspected beyond the search abstract |
| **HoneySat** *(new — see special note below)* | Novel hybrid, does not fit A-E cleanly | **No** — the "satellite" is an emulated honeypot, not a real operational spacecraft (SOURCE CLAIM) | **Yes, and genuinely real** — 22 real-world adversarial interactions from actual external attackers (SOURCE CLAIM, NDSS 2026) | No | Likely yes, UNVERIFIED in detail | Only 22 real interactions across 5 deployments (SOURCE CLAIM) — extremely small | UNVERIFIED | Single honeypot instance | UNVERIFIED | UNVERIFIED — artifact on Zenodo (10.5281/zenodo.17548980) not license-checked | arxiv.org/abs/2505.24008; ndss-symposium.org paper; Zenodo artifact | Unclear — this is attacker-interaction data, not multivariate telemetry | None (1 honeypot) | **Very high in kind, but N=22 is too small for training** | Not viable as a training dataset; interesting as a case study only |

**Note on LENS's category:** LENS is real satellite network traffic but has **no labels of
any kind** (neither cyberattack nor operational-fault labels). It does not cleanly satisfy
Category C's implicit expectation of *some* labeling; it is included here as the closest
real, unlabeled network-measurement candidate, not as a fully-qualified Category C dataset.

**Note on HoneySat:** this is the single most scientifically interesting new find of this
search, but it must not be mis-classified. The satellite it protects is an **emulated
honeypot**, not a real operational mission — so this is NOT Category A. What is
genuinely real is the **attacker behavior** (actual adversaries in the wild, not
scripted/injected attacks). At only 22 captured interactions it is far too small to serve
as a training dataset, but it is a legitimate, citable existence proof that "real
adversaries target satellite-like systems," which may be useful as a motivating citation
rather than as a data source.

## 1-5 Scoring (5 = best/strongest/easiest; 1 = worst/weakest/hardest)

| Dataset | Spacecraft realism | Cybersecurity relevance | Temporal suitability | Satellite relevance | FL suitability | Accessibility/licensing | Integration ease (5=easy) | Overall project value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NSL-KDD (reference) | 1 | 5 | 2 | 1 (simulated layer only) | 4 (already integrated) | 5 | 5 | 5 (already proven across Phases 1-8) |
| NASA SMAP/MSL | 5 | 1 | 5 | 4 | 2 | 5 | 5 | 4 |
| ESA-ADB | 5 | 1 | 5 | 4 | 2 | 4 (large download) | 3 | 4 |
| STIN/SAT20 | 1 | 5 | 3 | 3 | 4 | **2 (license unclear)** | 3 | 3 |
| CuCD-ID | 1 | 4 | 3 | 3 | 1 | 5 | 4 | 2 |
| LENS | 5 | 1 | 4 | 4 | 4 | 5 | 3 | 3 |
| LASP telemetry | 4 (UNCERTAIN) | 1 | 3 (UNCERTAIN) | 3 | 1 | 2 (UNVERIFIED) | 2 | 2 (needs verification first) |
| CATS | 1 | 1 | 3 | 1 | 1 | 2 (UNVERIFIED) | 2 | 1 |
| AegisSat | 1 | 4 (UNVERIFIED) | 2 (UNVERIFIED) | 3 | 2 (UNVERIFIED) | 2 (UNVERIFIED) | 2 | 2 (too new/unverified) |
| HoneySat | 2 (emulated target, real attackers) | 5 (in kind) / 1 (in volume — N=22) | 1 (not a time-series dataset) | 3 | 1 | 2 (UNVERIFIED) | 1 (too small to train on) | 1 (as training data); high as a citation |

These scores are **ENGINEERING ASSESSMENT**, produced for this project's specific
architecture — they are not a general-purpose ranking of dataset quality.
