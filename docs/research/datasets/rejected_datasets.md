# Rejected Datasets

Purpose: record every candidate this project investigated and ruled out, and why —
so a future teammate or reviewer knows a dataset was considered and consciously not
chosen, rather than assuming it was simply overlooked.

Datasets that were selected/shortlisted (OPSSAT-AD, ESA-ADB, NASA SMAP/MSL) are detailed
in `spacecraft_dataset_search.md` and `dataset_decision.md`, not repeated here.
Datasets that remain open/unresolved-but-not-outright-rejected (STIN/SAT20, CuCD-ID,
LASP telemetry, LENS) are also detailed in `spacecraft_dataset_search.md` — they are
listed briefly below too, for completeness of this specific "why not chosen as primary"
record, but "rejected" below means either fully excluded or excluded from further
consideration absent new information.

---

## STIN / SAT20 + TER20

- **Why it looked promising:** the single best architectural fit found — explicitly
  satellite-framed, genuine cyberattack labels, and ships pre-made FL train/test splits.
- **Why rejected (as primary, for now):** license status is unresolved. Direct inspection
  of the repository's `LICENSE.txt` found only a bare copyright notice ("Copyright (c)
  2020 National Engineering Lab for Next Generation Internet Interconnection Devices,
  BJTU"), no explicit permission to use/redistribute. Using it for an academic capstone
  without resolving this is a real legal risk, not a formality.
- **Real/simulated classification:** Category D — simulated/testbed. Search evidence
  states the authors "simulate a real scenario" for both satellite and terrestrial
  traffic.
- **Main limitation:** license, plus unconfirmed exact record counts/feature schema.
- **Source:** [github.com/kun9717/STIN-data-set](https://github.com/kun9717/STIN-data-set)
- **Disposition:** not rejected outright — could be revisited if the authors grant
  explicit permission or an explicit license is located.

## CuCD-ID

- **Why it looked promising:** clean CC BY 4.0 license, genuine SPARTA-aligned
  cyberattack scenarios, reproducible (scripts included).
- **Why rejected (as primary):** single simulated CubeSat via a NOS3 digital twin — no
  multi-satellite structure, and simulated rather than real, so it does not advance this
  project's "real spacecraft data" goal any further than the project's existing Phase 4
  simulation already does.
- **Real/simulated classification:** Category D — simulation (software-in-the-loop
  digital twin).
- **Main limitation:** single spacecraft, simulated.
- **Source:** [data.mendeley.com/datasets/7n2d42pm3n/3](https://data.mendeley.com/datasets/7n2d42pm3n/3)
- **Disposition:** kept as a documented fallback cyber-labeled candidate, not primary.

## LENS (LEO Satellite Network Measurement Dataset)

- **Why it looked promising:** genuinely real satellite network data (13 actual Starlink
  dishes), the strongest real multi-client structure of any candidate found.
- **Why rejected (as primary):** zero labels of any kind — no anomaly or attack ground
  truth. Using it would require this project to invent its own labels, which conflicts
  with the evidence-based approach used throughout Phases 1-8.
- **Real/simulated classification:** real hardware, real measurements; Category C
  (unlabeled).
- **Main limitation:** no labels.
- **Source:** [zenodo.org/records/15331285](https://zenodo.org/records/15331285)
- **Disposition:** retained as a possible unlabeled realism reference for non-IID client
  diversity, not as a training/evaluation dataset.

## LASP Satellite-Telemetry-Anomaly-Detection

- **Why it looked promising:** real, credible institution (Laboratory for Atmospheric
  and Space Physics, CU Boulder).
- **Why rejected:** insufficient verifiable detail. Repository fetch could not surface a
  specific spacecraft/mission name, record counts, label methodology, or license.
- **Real/simulated classification:** likely real (SOURCE CLAIM), but **UNVERIFIED** in
  every specific detail that would matter for a decision.
- **Main limitation:** under-documented; would need firsthand `/Data` and `/Metadata`
  directory inspection before any further consideration.
- **Source:** [github.com/sapols/Satellite-Telemetry-Anomaly-Detection](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection)
- **Disposition:** not actionable at this time; not fully rejected, just not usable
  without further work.

## CATS (Controlled Anomalies Time Series)

- **Why it looked promising:** appeared in a curated spacecraft-engineering dataset list
  alongside ESA-ADB and SMAP/MSL.
- **Why rejected:** explicitly described as "simulated complex systems" in its own
  documentation, and not confirmed to be satellite-specific at all.
- **Real/simulated classification:** simulated; not clearly satellite-domain.
- **Main limitation:** not real, not clearly satellite-relevant.
- **Source:** referenced via [github.com/patrickfleith/awesome-spacecraft-engineering-datasets](https://github.com/patrickfleith/awesome-spacecraft-engineering-datasets)
- **Disposition:** rejected, low priority for this project.

## AegisSat testbed

- **Why it looked promising:** recent (2025), explicitly satellite-cybersecurity-focused,
  from a credible institution (Ben-Gurion University of the Negev), producing labeled
  telemetry + cyberattack logs.
- **Why rejected (for now):** too new to verify beyond the paper's abstract; no dataset
  structure, size, or license could be confirmed in this research pass.
- **Real/simulated classification:** simulated — "repeated simulation campaigns"
  (SOURCE CLAIM).
- **Main limitation:** insufficient verification; not independently inspected.
- **Source:** [dl.acm.org/doi/10.1145/3736731.3746144](https://dl.acm.org/doi/10.1145/3736731.3746144)
- **Disposition:** worth revisiting as the field matures; not actionable now.

## HoneySat

- **Why it looked promising:** the closest thing to a "real cyberattack against a
  satellite" found anywhere in this research — 22 real-world adversarial interactions
  from actual external attackers, published at NDSS 2026.
- **Why rejected:** the target itself is an **emulated honeypot** simulating a CubeSat,
  not a real operational spacecraft — so it does not qualify as real spacecraft data no
  matter how real the attackers were. Separately, N=22 interactions is far too small to
  serve as a training or evaluation dataset.
- **Real/simulated classification:** simulated target (honeypot) + genuinely real
  attacker behavior — a hybrid that does not fit this project's A-E categories cleanly,
  and does not become Category A merely because part of it is real.
- **Main limitation:** target not real; sample size (22) unusable for training.
- **Source:** [arxiv.org/abs/2505.24008](https://arxiv.org/abs/2505.24008)
- **Disposition:** rejected as a dataset; retained as a citable existence proof that real
  adversaries target satellite-like systems, useful in a literature-review context only.

## Edge-IIoTSet

- **Why it looked promising:** some papers apply it to "satellite-based IoT" security
  scenarios.
- **Why rejected:** direct investigation shows it is built from a physical seven-layer
  IIoT **smart factory testbed** (pH sensors, flame sensors, soil-moisture sensors, etc.)
  — not satellite data by any reasonable reading. Papers using it for satellite framing
  are applying it by analogy only.
- **Real/simulated classification:** real IIoT factory data — **not satellite-relevant
  at all**.
- **Main limitation:** not satellite data.
- **Source:** [ieeexplore.ieee.org/document/9751703](https://ieeexplore.ieee.org/document/9751703/)
- **Disposition:** rejected outright.

## CANSat-IDS

- **Why it looked promising:** a real, credible *Computers & Security* (2024) paper
  describing an ANN+K-SVM CAN-bus intrusion detection method specifically for
  satellites.
- **Why rejected:** the methodology is real, but **no confirmed public dataset release**
  could be located across multiple targeted searches.
- **Real/simulated classification:** cannot be classified — no dataset found to classify.
- **Main limitation:** dataset availability unconfirmed.
- **Source:** [sciencedirect.com/science/article/abs/pii/S0167404824003389](https://www.sciencedirect.com/science/article/abs/pii/S0167404824003389)
- **Disposition:** rejected for now; would require directly contacting the authors.

## "CORTEX" (named candidate, this investigation round)

- **Why it was investigated:** explicitly named as a candidate to check in this round's
  research brief.
- **Why rejected:** **no dataset by this name was found** across two dedicated searches
  this session. Not invented, not substituted with a guess.
- **Real/simulated classification:** N/A — dataset not located.
- **Main limitation:** could not be found at all.
- **Source:** none found.
- **Disposition:** if a specific URL/paper for "CORTEX" exists, it needs to be supplied
  directly for follow-up verification — this research pass could not locate it through
  general search.
