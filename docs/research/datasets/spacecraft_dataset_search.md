# Real Spacecraft/Satellite Dataset Search

**Purpose:** find and verify publicly available REAL spacecraft/satellite datasets that
could support this project's eventual pipeline (real telemetry → temporal anomaly
detection → simulated multi-satellite clients → non-IID FL → adaptive FL → DRL), without
disturbing Phases 1-8 or NSL-KDD.

**Status: research record. No dataset listed here has been downloaded or integrated.**

## Search themes covered (this pass and two prior passes on this same question)

Satellite cybersecurity; spacecraft anomaly detection; satellite telemetry anomaly
detection; spacecraft telemetry; satellite network traffic; space-based communication/
network intrusion detection; satellite communication cybersecurity; spacecraft fault/
anomaly datasets; NASA spacecraft telemetry; ESA spacecraft telemetry; OPS-SAT datasets;
"CORTEX" satellite cybersecurity datasets; JAXA open data; NASA PDS/Earthdata; LEO
constellation measurement; CubeSat cybersecurity; inter-satellite-link anomalies;
ground-station↔satellite anomalies.

**"CORTEX" specifically:** searched directly this session (`CORTEX satellite
cybersecurity dataset intrusion detection` and `"CORTEX" spacecraft network dataset
anomaly NASA ESA`). **No dataset by this name was found.** Results returned CuCD-ID,
ESA-ADB, and NASA SMAP/MSL instead — all already covered below. This is recorded
explicitly rather than silently omitted, per the instruction not to assume a lead exists
just because it was named.

## Evidence-type key

- **FACT** — confirmed by directly fetching the primary source (Zenodo record, official
  GitHub org, Mendeley page, arXiv preprint) during this or a prior research session
- **SOURCE CLAIM** — stated by the dataset's own paper/page, not independently re-derived
- **ENGINEERING ASSESSMENT** — this project's own judgment, not a claim from the authors
- **UNVERIFIED** — could not be confirmed from any source reached; not to be treated as fact

---

## Category A. REAL spacecraft/satellite data, REAL anomaly/attack labels

### None found

No dataset combining REAL spacecraft telemetry/network data with REAL cyberattack labels
was located across three research passes. This is the single most important negative
finding of this entire investigation and is stated here explicitly rather than stretched.

---

## Category B. REAL spacecraft data, operational/physical anomaly labels (not cyber)

### OPSSAT-AD (OPS-SAT Anomaly Dataset) — newly verified, current top candidate

- **Dataset name:** OPSSAT-AD
- **Organization/source:** European Space Agency (ESOC) + KP Labs
- **Real spacecraft?** **Yes** — OPS-SAT, a CubeSat mission operated by ESA
- **Mission/spacecraft name:** OPS-SAT (flew 2019, mission ended night of 22-23 May 2024
  — FACT, confirmed directly from the arXiv preprint text)
- **Real or simulated:** **Real flight telemetry**
- **Collection period:** spans the operational life of OPS-SAT (2019-2024); exact
  sub-range of the labeled segments not independently confirmed — UNVERIFIED
- **Records:** 2,123 labeled telemetry segments (SOURCE CLAIM)
- **Features/channels:** 9 channels — 3 magnetometer + 6 photodiode (SOURCE CLAIM)
- **Labels:** yes, expert-annotated; ~20% of segments anomalous (SOURCE CLAIM)
- **Anomaly/attack types:** operational telemetry anomalies — **NOT cyberattacks**
- **Timestamps:** yes (telemetry time series)
- **Temporal ordering:** yes
- **Multiple spacecraft:** **No — single CubeSat**
- **Multiple communication links/clients:** no
- **Anomaly-detection support:** yes, strong — this is the dataset's stated purpose
- **Temporal-modeling support:** yes, strong (ENGINEERING ASSESSMENT — matches Phase 3's
  GRU+attention shape well)
- **FL-partitioning support:** only via artificial/simulated partitioning of a single
  spacecraft's channels or segments — must be labeled simulated if done
- **Licensing:** **CC-BY-4.0 — FACT, confirmed directly on the Zenodo record**
  ([zenodo.org/records/12588358](https://zenodo.org/records/12588358))
- **Download/access:** direct, no registration — FACT, confirmed on Zenodo
- **Official source URL:** Zenodo record above; official code
  [github.com/kplabs-pl/OPS-SAT-AD](https://github.com/kplabs-pl/OPS-SAT-AD)
- **Research paper:** *"The OPS-SAT benchmark for detecting anomalies in satellite
  telemetry,"* Nature Scientific Data, 2025; preprint
  [arXiv:2407.04730](https://arxiv.org/abs/2407.04730) (used for verification since the
  Nature page itself returned an auth redirect and could not be fetched directly)
- **Citation:** the arXiv/Scientific Data paper above; cite the Zenodo DOI for the data
  itself
- **Known limitations:** single spacecraft; no cyberattack labels; exact channel-level
  statistics not independently re-derived from the raw CSV in this research pass
- **Dataset size:** **19.5 MB — FACT, confirmed directly on Zenodo** (note: the Zenodo
  page's "22.3 GB" figure is cumulative across all historical versions of the record,
  not the current downloadable size — recorded here to avoid a misleading repetition of
  that number)
- **File formats:** CSV (`segments.csv`, `dataset.csv`) + 2 Jupyter notebooks +
  `requirements.txt` + `LICENSE`
- **Preprocessing required:** yes — entirely different schema from NSL-KDD's
  connection-record format; would need a new loader/windowing module
- **Could it replace or complement NSL-KDD?** Complement only — no cyberattack labels,
  cannot replace NSL-KDD's role in Phases 1-8

### ESA-ADB (ESA Anomaly Dataset)

- **Dataset name:** ESA-ADB
- **Organization/source:** European Space Agency (ESOC), Airbus Defence and Space, KP Labs
  — an 18-month industry-agency consortium project funded under ESA's "Artificial
  Intelligence for Automation (A²I) Roadmap," started 2021 (SOURCE CLAIM, OpenReview
  listing)
- **Real spacecraft?** **Yes** — 3 real, large ESA spacecraft
- **Mission/spacecraft name:** 3 missions, individual names not confirmed in what was
  retrieved — UNVERIFIED
- **Real or simulated:** real flight telemetry, "several years of real-life raw data"
  (SOURCE CLAIM)
- **Scale (more precise figure found this session):** **224 channels, 821 control
  signals, 1,430 annotated events** (SOURCE CLAIM, OpenReview paper listing) — this
  supersedes an earlier, smaller "844 events / 148 anomalies" figure found in a prior
  session, which appears to describe only a 2-mission benchmark subset rather than the
  full dataset; both figures are preserved here rather than silently discarding the
  earlier one
- **Labels:** yes, curated/annotated
- **Anomaly type:** operational/physical (solar-array regulator shutoffs, ADCS
  disturbances, video-processing-unit resets) — **NOT cyberattacks**
- **Timestamps/temporal ordering:** yes
- **Multiple spacecraft:** yes, 3 real missions — the strongest real multi-source count
  of any candidate found
- **Licensing:** **CC BY 3.0 IGO — FACT, confirmed directly on Zenodo**
- **Size:** **11.6 GB — FACT, confirmed directly** (3 ZIP files: 3.8 + 4.1 + 3.7 GB)
- **Official source:** [github.com/esa/anomaly-dataset](https://github.com/esa/anomaly-dataset)
  (official ESA GitHub org); data:
  [zenodo.org/records/12528696](https://zenodo.org/records/12528696); code:
  [github.com/kplabs-pl/ESA-ADB](https://github.com/kplabs-pl/ESA-ADB)
- **Paper:** [arXiv:2406.17826](https://arxiv.org/abs/2406.17826)
- **Limitations:** large download; per-mission documentation not fully enumerable from
  what was retrieved; no cyberattack labels
- **Could it replace or complement NSL-KDD?** Complement only

### NASA SMAP/MSL Spacecraft Telemetry Anomaly Dataset

- **Dataset name:** SMAP/MSL ("Telemanom" dataset)
- **Organization/source:** NASA/JPL
- **Real spacecraft?** **Yes** — SMAP satellite + MSL/Curiosity Mars rover
- **Real or simulated:** real telemetry
- **Records:** 496,444 telemetry values, 82 channels, 105 expert-labeled anomaly
  sequences (SOURCE CLAIM, Hundman et al. 2018)
- **Labels:** yes, operational — **not cyberattacks**
- **Timestamps:** sequence-ordered, time-anonymized
- **Multiple spacecraft:** 2 real sources
- **Licensing:** code Apache 2.0; data openly released, no registration wall found
- **Official source:** [github.com/khundman/telemanom](https://github.com/khundman/telemanom);
  Kaggle mirror (clearly a mirror, not the original):
  [kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl/data)
- **Paper:** Hundman et al., *"Detecting Spacecraft Anomalies Using LSTMs and
  Nonparametric Dynamic Thresholding,"* KDD 2018
- **Limitations:** MSL is a Mars rover, not an orbital satellite; older benchmark (2018);
  no cyberattack labels
- **Could it replace or complement NSL-KDD?** Complement only

### LASP Satellite-Telemetry-Anomaly-Detection — credible but under-documented

- **Organization:** Laboratory for Atmospheric and Space Physics, University of Colorado
  Boulder — a real, credible research institution
- **Real spacecraft?** Likely yes (SOURCE CLAIM: "LASP spacecraft telemetry") but
  specific mission name **UNVERIFIED**
- **Records/labels/license:** **UNVERIFIED** — repository fetch surfaced only "five
  representative time series datasets" and two example column names ("Temperature (C)",
  "WheelTemperature"); no exact counts, no confirmed ground-truth label methodology, no
  confirmed license
- **Official source:** [github.com/sapols/Satellite-Telemetry-Anomaly-Detection](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection)
- **Verdict:** not actionable without firsthand inspection of the `/Data` and `/Metadata`
  directories; kept in this record rather than dropped because the institutional
  provenance is credible

---

## Category C. REAL satellite/network measurements, no attack labels

### LENS (LEO Satellite Network Measurement Dataset)

- **Real spacecraft/hardware?** **Yes** — 13 real, operational Starlink dishes across 7
  Points-of-Presence, 3 continents (SOURCE CLAIM, ACM paper)
- **Data type:** network latency/performance measurement, not telemetry
- **Labels:** **none** — no anomaly or attack ground truth of any kind
- **Licensing:** **CC BY-SA 4.0 — FACT, confirmed directly on Zenodo**
- **Official source:** [zenodo.org/records/15331285](https://zenodo.org/records/15331285);
  [github.com/clarkzjw/LENS](https://github.com/clarkzjw/LENS)
- **Multiple clients:** yes, real — the strongest genuine multi-client structure found
  in any candidate
- **Limitations:** would require this project to define its own anomaly labels, which
  must be clearly documented as project-authored, not pre-existing ground truth
- **Could it replace or complement NSL-KDD?** Neither directly — no labels; potentially
  useful only as an unlabeled realism reference for non-IID client diversity

---

## Category D. Simulated/testbed spacecraft data, REAL cyberattack labels

### STIN / SAT20 + TER20

- **Real spacecraft?** **No** — search evidence: "the authors simulate a real scenario
  for both the terrestrial network and satellite network." Testbed-generated, satellite-
  shaped topology, not captured from an orbiting satellite.
- **Organization:** National Engineering Lab for Next Generation Internet Interconnection
  Devices, Beijing Jiaotong University (BJTU) — confirmed directly from the repo's
  copyright notice
- **Labels:** yes, genuine cyberattack labels — Signal Disruption, UDP flood, Jamming
  (satellite-domain); DoS, DDoS, Bruteforce, Infiltration (terrestrial-domain)
- **Licensing:** **UNRESOLVED** — `LICENSE.txt` fetched directly this session contains
  only a bare copyright notice, no explicit open-source grant
- **Official source:** [github.com/kun9717/STIN-data-set](https://github.com/kun9717/STIN-data-set)
- **Multi-client suitability:** repo reportedly ships pre-made FL train/test splits
  (SOURCE CLAIM) — the only candidate built with FL explicitly in mind
- **Exact record counts/features:** **UNVERIFIED**
- **Could it replace or complement NSL-KDD?** Could potentially complement as a second
  simulated benchmark, but blocked on license resolution

### CuCD-ID (CubeSat Cybersecurity Dataset for Intrusion Detection)

- **Real spacecraft?** **No** — generated via NASA's NOS3 software-in-the-loop digital
  twin, not real flight hardware
- **Labels:** yes — 5 scenarios (1 nominal + 4 SPARTA-aligned attacks: command flooding,
  GPS/false-data injection, defense impairment, storage exhaustion)
- **Size:** 25,000 records / 31 features (raw); 22,465 / 23 features (augmented)
  (SOURCE CLAIM)
- **Licensing:** **CC BY 4.0 — FACT, confirmed directly on Mendeley Data**
- **Official source:** [data.mendeley.com/datasets/7n2d42pm3n/3](https://data.mendeley.com/datasets/7n2d42pm3n/3)
- **Limitations:** single simulated CubeSat, no multi-satellite structure
- **Could it replace or complement NSL-KDD?** Complement only, as a second simulated
  attack taxonomy

### AegisSat testbed

- **Real spacecraft?** No — repeated simulation campaigns (SOURCE CLAIM)
- **Organization:** Ben-Gurion University of the Negev
- **Official source:** [dl.acm.org/doi/10.1145/3736731.3746144](https://dl.acm.org/doi/10.1145/3736731.3746144)
- **Verdict:** too new (2025) to verify beyond the abstract; not independently inspected

---

## Category E. Not satellite data / synthetic benchmark / rejected outright

See `rejected_datasets.md` for CATS (Controlled Anomalies Time Series — simulated,
not satellite-specific), Edge-IIoTSet (real IIoT factory testbed, not satellite),
CANSat-IDS (real methodology paper, no confirmed public dataset release), and HoneySat
(real attacker interactions against an emulated honeypot CubeSat — a genuine near-miss
for Category A, but the target itself is not a real operational spacecraft and N=22 is
too small to train on).

---

## Summary table (see `dataset_comparison.md` for full scoring)

| Dataset | Category | Real spacecraft | Real cyberattacks |
|---|---|---|---|
| OPSSAT-AD | B | Yes | No |
| ESA-ADB | B | Yes | No |
| NASA SMAP/MSL | B | Yes | No |
| LASP telemetry | B (uncertain) | Likely | No |
| LENS | C | Yes (network, not telemetry) | No labels at all |
| STIN/SAT20 | D | No | Yes |
| CuCD-ID | D | No | Yes |
| AegisSat | D | No | Yes (unverified detail) |
| CATS, Edge-IIoTSet, CANSat-IDS, HoneySat | E / rejected | See `rejected_datasets.md` | See `rejected_datasets.md` |

**No Category A candidate exists in this table.** See `dataset_decision.md` for the
recommendation that follows from this evidence.
