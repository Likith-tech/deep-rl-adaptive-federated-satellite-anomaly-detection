# Dataset Comparison — Real Spacecraft/Satellite Candidates

Companion to `spacecraft_dataset_search.md` (full per-dataset verification detail) and
`dataset_decision.md` (the recommendation this comparison feeds into).

**Status: research record. No dataset has been downloaded or integrated.**

## Main comparison table

| Dataset | Real spacecraft? | Telemetry/network data | Real timestamps | Labels | Multiple satellites | Cyber anomalies | Temporal suitability | FL suitability | Size | Source quality | License | Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **OPSSAT-AD** | Yes | Telemetry | Yes | Yes (operational) | No (1) | No | Strong | Weak (simulated partitioning only) | 19.5 MB | High (Nature Sci Data 2025, official ESA GitHub) | CC-BY-4.0 | **Primary candidate** |
| **ESA-ADB** | Yes | Telemetry | Yes | Yes (operational) | Yes (3) | No | Strong | Weak-moderate | 11.6 GB | High (official ESA GitHub, arXiv paper) | CC BY 3.0 IGO | Secondary candidate |
| **NASA SMAP/MSL** | Yes | Telemetry | Yes | Yes (operational) | No (2, incl. 1 rover) | No | Strong | Weak | Small | High (KDD 2018, field standard) | Apache 2.0 (code) / open data | Tertiary candidate |
| **LASP telemetry** | Likely | Telemetry | Unverified | Unverified | Unverified | No | Uncertain | Weak | Unverified | Low (unverified) | Unverified | Not actionable yet |
| **LENS** | Yes | Network measurement | Yes | **No labels at all** | Yes (13 real) | No | Moderate | Strong (real clients) | Multi-GB | High (ACM paper) | CC BY-SA 4.0 | Secondary reference only (needs self-defined labels) |
| **STIN/SAT20** | No (testbed) | Network | Weak-moderate | Yes (cyber) | No (simulated topology) | **Yes** | Weak-moderate | Strong (ships FL splits) | Unverified | Moderate (license blocks confidence) | **Unresolved** | Blocked pending license |
| **CuCD-ID** | No (digital twin) | Telemetry/command | Moderate | Yes (cyber) | No (1) | **Yes** | Moderate | Weak | 25,000 records | Moderate (ScienceDirect) | CC BY 4.0 | Fallback cyber-labeled candidate |
| **NSL-KDD** *(reference, already in use)* | No (terrestrial) | Network | Weak | Yes (cyber) | Simulated (Phase 4) | **Yes** | Weak | Strong (already integrated) | 148,517 records | High (established academic use) | Standard academic | Keep as-is |

## Scored criteria (1 = weakest, 5 = strongest)

Every score below is an **ENGINEERING ASSESSMENT** made for this project's specific
architecture and goals — not a general-purpose dataset-quality ranking. Justification is
given for every score, as requested.

### OPSSAT-AD

| Criterion | Score | Justification |
|---|---:|---|
| Realism | 5 | Confirmed real, flown, ESA-operated CubeSat; verified directly via arXiv preprint text |
| Cybersecurity relevance | 1 | Zero attack labels; purely operational anomalies |
| Anomaly-detection suitability | 5 | Purpose-built benchmark, expert-labeled, ~20% anomalous — clean and usable |
| Temporal suitability | 5 | Native multivariate time series across 9 channels |
| Federated-learning suitability | 2 | Single spacecraft; any FL use requires artificial, clearly-labeled partitioning |
| Multi-satellite suitability | 1 | Explicitly one CubeSat |
| Dataset size | 5 (for ease) | 19.5 MB — trivially easy to work with in a capstone timeline |
| Reproducibility | 5 | Peer-reviewed (Nature Scientific Data 2025), official code + data repos, no registration |
| Accessibility | 5 | Direct Zenodo download, CC-BY-4.0, no barriers |
| Research credibility | 5 | ESA-funded, peer-reviewed, same group (KP Labs) behind ESA-ADB |

### ESA-ADB

| Criterion | Score | Justification |
|---|---:|---|
| Realism | 5 | Confirmed real, 3 actual operational ESA missions |
| Cybersecurity relevance | 1 | Operational anomalies only |
| Anomaly-detection suitability | 5 | Large, curated, expert-annotated (1,430 events across 224 channels) |
| Temporal suitability | 5 | Multi-year real telemetry time series |
| Federated-learning suitability | 3 | 3 real missions is a genuine, if small, natural multi-client basis — better than single-spacecraft candidates |
| Multi-satellite suitability | 3 | 3 real spacecraft, more than any other real candidate, but still far short of "8 satellites" |
| Dataset size | 2 (for ease) | 11.6 GB is a real practical cost for repeated capstone experimentation |
| Reproducibility | 4 | Official ESA GitHub org + Zenodo + arXiv paper, but per-mission documentation less completely retrievable than OPSSAT-AD's |
| Accessibility | 4 | Direct download, no registration, but large |
| Research credibility | 5 | ESA/Airbus/KP Labs consortium, A²I Roadmap-funded |

### NASA SMAP/MSL

| Criterion | Score | Justification |
|---|---:|---|
| Realism | 5 | Confirmed real NASA telemetry |
| Cybersecurity relevance | 1 | Operational anomalies only |
| Anomaly-detection suitability | 4 | Well-labeled, but older (2018) benchmark |
| Temporal suitability | 5 | Native time series, 82 channels |
| Federated-learning suitability | 2 | Only 2 real sources, one of which (MSL) is a rover, not orbital |
| Multi-satellite suitability | 2 | 2 sources, weak orbital-satellite framing |
| Dataset size | 5 (for ease) | Small, single-command download |
| Reproducibility | 5 | Field-standard benchmark, widely cited since 2018 |
| Accessibility | 5 | GitHub + Kaggle, no registration |
| Research credibility | 5 | NASA/JPL, KDD 2018 |

### STIN/SAT20

| Criterion | Score | Justification |
|---|---:|---|
| Realism | 1 | Confirmed simulated/testbed, not real satellite traffic |
| Cybersecurity relevance | 5 | Direct, labeled cyberattacks in a satellite-framed topology |
| Anomaly-detection suitability | 4 | Labeled, multi-class attack taxonomy |
| Temporal suitability | 2 | Flow-based records, not confirmed as rich sequential data |
| Federated-learning suitability | 5 | Only candidate shipping pre-made FL train/test splits |
| Multi-satellite suitability | 3 | Simulated satellite+terrestrial split, not real multi-satellite |
| Dataset size | 3 | Unverified exact size; presumed CSV-scale, moderate |
| Reproducibility | 2 | Sparse documentation, exact record counts unconfirmed |
| Accessibility | 2 | **License unresolved — real legal risk for a capstone deliverable** |
| Research credibility | 3 | Real institution (BJTU) confirmed, but thin documentation |

### CuCD-ID

| Criterion | Score | Justification |
|---|---:|---|
| Realism | 1 | Confirmed simulation (NOS3 digital twin) |
| Cybersecurity relevance | 4 | Genuine SPARTA-aligned attack scenarios |
| Anomaly-detection suitability | 3 | Small, single-spacecraft, but clean and labeled |
| Temporal suitability | 3 | 20-second windowed statistics, not raw continuous sequences |
| Federated-learning suitability | 1 | Single simulated CubeSat, no natural multi-client structure |
| Multi-satellite suitability | 1 | Same reason |
| Dataset size | 4 | Small (25,000 records), easy to work with |
| Reproducibility | 5 | Reproduction scripts included, clean documentation |
| Accessibility | 5 | Direct Mendeley download, confirmed CC BY 4.0 |
| Research credibility | 4 | ScienceDirect-published, NASA NOS3 tooling lends credibility despite being simulated |

### LENS

| Criterion | Score | Justification |
|---|---:|---|
| Realism | 5 | Real, operational Starlink hardware |
| Cybersecurity relevance | 1 | No labels of any kind |
| Anomaly-detection suitability | 1 | No ground truth; would require this project to invent labels |
| Temporal suitability | 4 | Real latency time series |
| Federated-learning suitability | 5 | Only genuinely real multi-client structure found (13 dishes) |
| Multi-satellite suitability | 4 | Real, geographically distinct nodes (though "dishes," not satellites themselves) |
| Dataset size | 3 | Multi-GB, monthly split archives |
| Reproducibility | 4 | ACM paper, GitHub, Zenodo |
| Accessibility | 4 | Direct download, CC BY-SA 4.0 |
| Research credibility | 4 | University-affiliated measurement study |

## Reading this table

No dataset scores well across both "Cybersecurity relevance" and "Realism" — this is the
same finding as `spacecraft_dataset_search.md`'s Category A result, expressed
numerically. The recommendation in `dataset_decision.md` follows from treating this as
two separate questions rather than forcing one score to answer both.
