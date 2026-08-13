# Phase 9 Real Spacecraft Dataset Verification

**Status: RESEARCH RECORD ONLY. No dataset integrated. No Phase 9 implementation
started. See section 23.**

## 1. Purpose

Before beginning what was originally planned as "Phase 9 — DRL-based adaptive federated
learning," the project owner asked for a pause to investigate whether a REAL
spacecraft/satellite dataset exists that could add genuine scientific value alongside
NSL-KDD, without disturbing the reproducibility of Phases 1-8. This document is the
durable record of that investigation: what was searched for, what was found, what could
and could not be verified, and what is (and is not) recommended — pending explicit human
approval before any implementation.

## 2. Current Project Dataset State

- **Dataset in use:** NSL-KDD (`data/raw/KDDTrain+.txt`, `KDDTest+.txt`), processed into
  `data/processed/{train,validation,test}.parquet` — 148,517 total records (107,077
  train / 18,896 validation / 22,544 test), 121 post-encoding features
  (`data/processed/metadata.json`).
- **Documented source:** `docs/datasets/dataset_selection.md` cites the exact
  acquisition URL: `raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/`. This is
  a GitHub mirror of the University of New Brunswick CIC's NSL-KDD release.
- **Satellite layer:** entirely simulated — `data/partitions/SAT-01`…`SAT-08`,
  `manifest.json`, `satellite_metadata.json` (Phase 4). Client IDs, Dirichlet
  non-IID partitioning, and simulated bandwidth/latency/compute/availability/
  connectivity are all synthetic constructs layered on top of the real NSL-KDD
  records.
- **No second dataset currently exists anywhere in this repository.**
- **NSL-KDD is never described as satellite data anywhere in this repo** — checked
  `README.md`, `docs/datasets/dataset_selection.md`, and every file in
  `docs/project-progress/`; all consistently and explicitly state NSL-KDD is
  terrestrial and the satellite framing is simulated.

## 3. Research Questions

1. What is the current git/repository state (confirmed separately not to block this
   research task, though it does block any future branch/commit action)?
2. Which datasets are present, and which are real vs. simulated?
3. Does a genuine, publicly-verifiable Category A dataset (real spacecraft + real
   cyberattack labels) exist?
4. For each serious candidate: real or simulated spacecraft? Operational fault or
   cyberattack? License and redistribution terms? Suitable for Phase 3-style temporal
   modeling? Suitable for Phase 4-8-style multi-client/federated modeling?
5. Which dataset is best for (a) real-spacecraft realism, (b) cybersecurity relevance,
   (c) multi-client structure — evaluated separately, not forced into one winner?
6. What integration strategy preserves Phases 1-8 exactly as they are?

## 4. Candidate Discovery Method

Web search was used across ten themed query rounds (satellite network traffic,
satellite intrusion detection, space cybersecurity, satellite telemetry anomalies,
satellite command-and-control anomalies, satellite IoT security, LEO network traffic,
inter-satellite-link anomalies, ground-station↔satellite anomalies, and a dedicated
Category-A existence search), across two separate research sessions, plus this session's
targeted re-verification and gap-filling (JAXA, NASA PDS/Earthdata, a fresh "additional
candidates" pass, and direct primary-source fetches for every dataset ultimately listed
in section 12). Search-engine summaries were treated as **SOURCE CLAIM** at best;
wherever feasible, the primary page (Zenodo record, official GitHub org, Mendeley Data
page) was fetched directly in this session to upgrade a claim to **FACT**. Where that
wasn't possible, the claim is marked `UNVERIFIED` rather than presented as settled.

## 5. NASA SMAP/MSL Verification

- **Name:** SMAP/MSL Spacecraft Telemetry Anomaly Dataset ("Telemanom" dataset)
- **Real or simulated:** **REAL** (FACT-adjacent — confirmed via the author's own
  repository description, not independently re-derived from raw telemetry)
- **Missions:** SMAP satellite (Soil Moisture Active Passive) + MSL/Curiosity Mars rover
- **Channels/samples:** 82 channels, 496,444 telemetry values, 105 expert-labeled
  anomaly sequences (SOURCE CLAIM, Hundman et al. 2018)
- **Timestamps:** sequence-ordered, time-anonymized
- **Anomaly type:** **operational/physical**, NOT cyberattacks — no attack labels of any
  kind
- **Format:** `.npy` per channel + `labeled_anomalies.csv`
- **License:** code Apache 2.0; data openly released, no registration wall found
- **Official source:** [github.com/khundman/telemanom](https://github.com/khundman/telemanom);
  Kaggle mirror: [patrickfleith/nasa-anomaly-detection-dataset-smap-msl](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl/data)
- **Paper:** Hundman et al., *"Detecting Spacecraft Anomalies Using LSTMs and
  Nonparametric Dynamic Thresholding,"* KDD 2018
- **Redistribution/academic use:** permitted; this is the field's standard benchmark
- **Temporal-model suitability:** strong (ENGINEERING ASSESSMENT) — this is exactly the
  multivariate-time-series-with-labeled-intervals shape Phase 3's GRU+attention
  architecture was built for
- **FL suitability:** weak — only 2 real sources
- **Multi-client suitability:** weak, same reason

## 6. ESA-ADB Verification

- **Name:** ESA Anomaly Dataset (ESA-ADB)
- **Real or simulated:** **REAL** — 3 actual operational ESA missions
- **Verified directly in this session:** official repo is
  [github.com/esa/anomaly-dataset](https://github.com/esa/anomaly-dataset) (the ESA
  GitHub organization itself, a stronger provenance signal than the previously-checked
  third-party `kplabs-pl/ESA-ADB` code mirror, which hosts the *implementation code*,
  not the dataset). Data: [Zenodo record 12528696](https://zenodo.org/records/12528696)
  — **FACT, license = CC BY 3.0 IGO, size = 11.6 GB across 3 ZIPs (3.8 + 4.1 + 3.7 GB)**,
  direct download, no registration found. Paper: [arXiv:2406.17826](https://arxiv.org/abs/2406.17826).
- **Anomaly counts:** a related benchmark paper reports 844 annotated events across 2 of
  the 3 missions, 148 classified as anomalies (SOURCE CLAIM — not independently
  recomputed).
- **Anomaly type:** **operational/physical** (solar-array power-regulator shutoffs,
  video-processing-unit resets, attitude/ADCS disturbances) — NOT cyberattacks
- **Redistribution/academic use:** CC BY is explicitly permissive with attribution —
  this is the clearest license confirmation of any candidate in this report
- **Temporal-model suitability:** strong (ENGINEERING ASSESSMENT)
- **FL/multi-client suitability:** weak-moderate — 3 real missions is a genuine but
  small basis for "multiple clients"

## 7. STIN/SAT20 Verification

- **Name:** Satellite-Terrestrial Integrated Network dataset — SAT20 (satellite) +
  TER20 (terrestrial)
- **Real or simulated:** **SIMULATED/TESTBED.** Search evidence explicitly states "the
  authors simulate a real scenario for both the terrestrial network and satellite
  network" — this is testbed-generated traffic with a satellite-shaped topology, **not**
  captured from an actual orbiting satellite. Do not call this real satellite data.
- **Institution:** confirmed directly this session from the repository's copyright
  notice: "Copyright (c) 2020 National Engineering Lab for Next Generation Internet
  Interconnection Devices, BJTU" (Beijing Jiaotong University) — a real, identifiable
  institution.
- **Labels:** 8 classes — benign + satellite-domain attacks (Signal Disruption, UDP
  flood, Jamming) + terrestrial-domain attacks (DoS, DDoS, Bruteforce, Infiltration).
  These are genuine cyberattack labels, but on simulated traffic.
- **License:** **UNRESOLVED.** Fetched the repo's `LICENSE.txt` directly this session —
  it contains only the copyright notice above, no explicit license grant (no MIT/CC/
  Apache-style permission). Default copyright applies absent an explicit grant. **Do
  not use this dataset for a capstone deliverable without contacting the authors for
  written permission, or finding an explicit license elsewhere in the repo that this
  session did not locate.**
- **Record counts / exact feature schema:** could not be independently confirmed —
  `UNVERIFIED`. Verify by cloning the repository before relying on any specific number.
- **Official source:** [github.com/kun9717/STIN-data-set](https://github.com/kun9717/STIN-data-set);
  related: [github.com/hjp007/STI](https://github.com/hjp007/STI)
- **FL suitability:** the repository reportedly includes a folder with pre-made
  federated-learning train/test splits (SOURCE CLAIM) — the only candidate in this
  report built with FL explicitly in mind.

## 8. CuCD-ID Verification

- **Name:** CubeSat Cybersecurity Dataset for Intrusion Detection (CuCD-ID)
- **Real or simulated:** **SIMULATION** — explicitly generated via NASA's Operational
  Simulator for Small Satellites (NOS3), a software-in-the-loop digital twin, not real
  flight hardware.
- **Labels:** 5 scenarios (1 nominal + 4 attacks aligned to the SPARTA space-attack
  framework: command flooding, GPS/false-data injection, defense impairment via
  limit-checker/checksum manipulation, storage exhaustion)
- **Size:** 25,000 records / 31 features (raw, class-balanced); 22,465 records / 23
  features (augmented, noised) — SOURCE CLAIM
- **Format:** CSV, CCSDS space-packet standard (a genuine aerospace protocol, even
  though the underlying data is simulated)
- **License:** **CC BY 4.0, confirmed directly** on the Mendeley Data page this session
  — clearly permissive
- **Official source:** [data.mendeley.com/datasets/7n2d42pm3n/3](https://data.mendeley.com/datasets/7n2d42pm3n/3)
- **Multi-client suitability:** weak — single simulated CubeSat, no constellation
  structure

## 9. LENS Verification

- **Name:** LENS — A LEO Satellite Network Measurement Dataset
- **Real or simulated:** **REAL** — genuine measurements from 13 operational Starlink
  dishes across 7 Points-of-Presence, 3 continents (SOURCE CLAIM, ACM paper)
- **Labels:** **none** — this is a pure latency/performance measurement dataset with no
  attack or anomaly ground truth of any kind
- **License:** **CC BY-SA 4.0, confirmed directly** on Zenodo this session
- **Official source:** [zenodo.org/records/15331285](https://zenodo.org/records/15331285);
  [github.com/clarkzjw/LENS](https://github.com/clarkzjw/LENS)
- **Multi-client suitability:** the strongest of all candidates for genuine
  (not-fabricated) multi-client structure — 13 real, geographically distinct nodes
- **Caveat:** using this for anomaly detection would require the project team to define
  its own anomaly labels, which must be documented as a project-authored labeling
  decision, never presented as pre-existing ground truth.

## 10. Additional Candidates Found (this session, beyond the original five)

- **LASP Satellite-Telemetry-Anomaly-Detection**
  ([github.com/sapols/Satellite-Telemetry-Anomaly-Detection](https://github.com/sapols/Satellite-Telemetry-Anomaly-Detection)) —
  real institution (Laboratory for Atmospheric and Space Physics, University of
  Colorado Boulder), "five representative time series datasets" of LASP spacecraft
  telemetry (SOURCE CLAIM). **UNVERIFIED**: which specific spacecraft, exact record
  counts, whether genuine expert-labeled ground truth exists, and license. Flagged as a
  credible-provenance but under-documented candidate that would need firsthand repo
  inspection before any decision.
- **CATS (Controlled Anomalies Time Series)** — explicitly **simulated** ("simulated
  complex systems," SOURCE CLAIM), and not confirmed to be satellite-specific. Low
  priority for this project.
- **AegisSat testbed** (Ben-Gurion University,
  [dl.acm.org/doi/10.1145/3736731.3746144](https://dl.acm.org/doi/10.1145/3736731.3746144)) —
  a 2025 reproducible satellite-cybersecurity testbed producing labeled telemetry +
  cyberattack logs from **repeated simulation campaigns** (SOURCE CLAIM) — Category D,
  same limitations as STIN/CuCD-ID, and too new to have been independently verified
  beyond the abstract in this session.
- **HoneySat** ([arXiv:2505.24008](https://arxiv.org/abs/2505.24008), NDSS 2026) — the
  most scientifically interesting new find. A high-interaction honeypot that
  convincingly emulates a CubeSat and captured **22 real-world adversarial
  interactions from actual external attackers** (SOURCE CLAIM). The satellite is
  emulated (not a real operational mission), but the attacker behavior is genuinely
  real, not scripted. At N=22 it is far too small to be a training dataset. See the
  dedicated note in `real-spacecraft-dataset-comparison.md`.

## 11. Category A Search Result

**No verified public Category-A dataset was found.** Category A requires REAL
spacecraft/satellite telemetry combined with REAL cyberattack labels. Every real-data
candidate found (NASA SMAP/MSL, ESA-ADB, LENS, LASP) has either no labels or only
operational-fault labels. Every candidate with genuine cyberattack labels (STIN/SAT20,
CuCD-ID, AegisSat) is simulated/testbed data. HoneySat comes closest to bridging the gap
(real attackers) but the target system itself is an emulated honeypot, not a real
operational spacecraft, and its sample size (22) is not usable for training. This is
consistent with the plausible institutional reason such data would not be publicly
released: real spacecraft cyberattack incidents are commercially and often
nationally/security sensitive.

## 12. Dataset Comparison

See `real-spacecraft-dataset-comparison.md` for the full table and 1-5 scoring. Summary:
ESA-ADB and NASA SMAP/MSL are the two strongest real-spacecraft (Category B) candidates;
STIN/SAT20 is the strongest cybersecurity-labeled (Category D) candidate but has an
unresolved license; LENS is the strongest genuinely-real multi-client candidate but has
no labels at all.

## 13. Best Real Spacecraft Dataset

**ESA-ADB**, with **NASA SMAP/MSL** as a lower-friction alternate/companion. Both are
Category B (real telemetry, operational anomalies, no cyberattack labels). ESA-ADB is
larger (11.6 GB, 3 missions, more recent) and has the clearest license of any real
candidate (CC BY 3.0 IGO, verified). SMAP/MSL is smaller, simpler, faster to integrate,
and is the field's most-cited benchmark.

## 14. Best Satellite Cybersecurity Dataset

**STIN/SAT20**, conditional on resolving its license. It is the only candidate that is
both satellite-*framed* and explicitly attack-labeled AND ships ready-made FL splits —
directly relevant to this project's architecture. **CuCD-ID** is the safer fallback:
smaller and single-spacecraft, but has a clean, confirmed CC BY 4.0 license with no
ambiguity.

## 15. Best Multi-Client Satellite Dataset

**LENS** — 13 real, geographically distinct Starlink dishes is the only genuinely real
multi-client structure found in any candidate. The tradeoff is total absence of labels;
using it for anomaly detection would require this project to define and clearly
document its own labeling scheme.

## 16. What Each Dataset Can Prove

- **NASA SMAP/MSL, ESA-ADB:** that this project's temporal architecture (Phase 3)
  generalizes to real spacecraft telemetry for operational-anomaly detection.
- **STIN/SAT20, CuCD-ID:** that this project's FL/adaptive-FL pipeline can be re-run
  against a differently-simulated, explicitly satellite-framed intrusion dataset —
  useful as a second simulated benchmark, not as evidence of real-world performance.
- **LENS:** that this project's non-IID/heterogeneity assumptions (Phase 4/7) are
  plausible against real client diversity, if labels are added and clearly attributed
  to this project.
- **HoneySat (as a citation, not a dataset):** that real adversaries do target
  satellite-like systems — useful as motivating context in a literature review, not as
  training/evaluation data.

## 17. What Each Dataset Cannot Prove

- **No dataset here** can support a claim of "detects real satellite cyberattacks" —
  that would require a Category A dataset, which does not appear to exist publicly.
- NASA SMAP/MSL and ESA-ADB **cannot** be used to claim anything about intrusion
  detection or cybersecurity — their anomalies are hardware/operational, not malicious.
- STIN/SAT20 and CuCD-ID **cannot** be used to claim "real satellite" validation — they
  are simulated/testbed, same category as this project's existing NSL-KDD-based
  approach.
- None of the real-data candidates (2-3 real sources at most) can support a claim of
  genuine "8-satellite" or large-constellation federated learning — that remains a
  simulated construct in every scenario reviewed.

## 18. Licensing and Access Considerations

| Dataset | License status |
|---|---|
| ESA-ADB | **CC BY 3.0 IGO — confirmed, clear, permissive** |
| CuCD-ID | **CC BY 4.0 — confirmed, clear, permissive** |
| LENS | **CC BY-SA 4.0 — confirmed, clear, permissive** (share-alike clause applies to derivatives) |
| NASA SMAP/MSL | Code Apache 2.0; data openly released, no registration found, but no single explicit data-license statement was located — reasonably safe given it is the field's standard benchmark, but not as crisply confirmed as the three above |
| STIN/SAT20 | **Unresolved** — copyright notice only, no license grant found |
| LASP telemetry, CATS, AegisSat, HoneySat | **Unverified** — insufficient information retrieved this session to state a license |

## 19. Integration Difficulty

- **Easiest:** NASA SMAP/MSL (small, `.npy`/CSV, one Kaggle command) and CuCD-ID (CSV,
  clean license).
- **Moderate:** ESA-ADB (large download, 11.6 GB, but well-documented) and LENS
  (monthly split-tar archives).
- **Hardest / blocked:** STIN/SAT20 (license must be resolved with the authors first),
  LASP/CATS/AegisSat/HoneySat (insufficient verified information to scope integration
  effort at all).
- **In every case:** integration would require a new preprocessing module (none of
  these share NSL-KDD's connection-record schema), which has NOT been designed or
  implemented as part of this research task.

## 20. Recommended Dataset Strategy

Keep NSL-KDD as the unchanged, primary benchmark for Phases 1-8 and any future
FL/DRL work (Phase 9/10). Add real spacecraft telemetry (ESA-ADB and/or NASA SMAP/MSL)
as an **isolated, additive validation track** for the Phase 3 temporal architecture
only — not routed through the FL/adaptive-FL/DRL pipeline, since none of the real
candidates have a genuine multi-satellite structure. Report metrics from each dataset
**separately, never merged into one number**, since NSL-KDD measures cyberattack
detection and the real-spacecraft datasets measure operational-fault detection — two
different phenomena.

## 21. Risks and Limitations

- No Category A dataset exists publicly — the project cannot claim "real satellite
  cyberattack detection" no matter which candidate is chosen.
- STIN/SAT20's license is unresolved; using it without author permission is a legal
  risk for an academic deliverable.
- ESA-ADB is large (11.6 GB) — a real practical cost for repeated experimentation in a
  capstone timeline.
- LASP telemetry, CATS, AegisSat, and HoneySat are under-verified — none should be
  relied upon without further firsthand inspection.
- Adding any second dataset creates two non-comparable metric tracks — a communication
  risk when presenting results, mitigated only by being explicit (section 17) about what
  each number does and does not mean.

## 22. Final Recommendation

### Recommended primary dataset
**ESA-ADB** (real spacecraft telemetry, operational anomalies) — **RECOMMENDATION —
AWAITING USER APPROVAL**

### Recommended cybersecurity dataset
**STIN/SAT20**, conditional on license resolution; **CuCD-ID** as the immediately-usable
fallback — **RECOMMENDATION — AWAITING USER APPROVAL**

### Recommended multi-client dataset
**LENS** (real multi-client structure, requires self-defined labels, must be clearly
attributed as such) — **RECOMMENDATION — AWAITING USER APPROVAL**

## 23. Decision Pending Human Approval

**No dataset described in this document has been downloaded, integrated, or connected
to any part of this project's code, configuration, or experiments. No Phase 9
implementation has been started. No existing Phase 1-8 result has been changed. This
document exists solely to support a future, explicitly-approved decision — it is not
itself that decision, and no further action will be taken on any dataset named here
until the project owner explicitly approves a specific dataset and a specific
integration plan.**
