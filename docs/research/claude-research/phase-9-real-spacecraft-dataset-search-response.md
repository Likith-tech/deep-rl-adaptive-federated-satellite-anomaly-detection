# Phase 9 Real Spacecraft Dataset — Raw Research Findings

**Companion to:** `phase-9-real-spacecraft-dataset-search-prompt.md` (the instructions
this research answers) and `../dataset-verification/phase-9-dataset-verification.md`
(the polished narrative report). This file preserves the research trail itself —
including findings that turned out unfavorable, unconfirmed, or contradictory —
rather than only the conclusions.

## Search rounds performed (across this and prior sessions on this topic)

1. `real satellite network intrusion detection dataset publicly available research 2024`
2. `NASA spacecraft telemetry anomaly detection dataset SMAP MSL github`
3. `satellite communication network traffic dataset Kaggle Zenodo download`
4. `"satellite" "IoT" cybersecurity dataset intrusion detection space`
5. `STIN dataset satellite terrestrial integrated network intrusion github kun9717`
6. `spacecraft command and control anomaly detection dataset ground station telemetry public`
7. `inter-satellite link traffic anomaly dataset LEO constellation Starlink measurement`
8. `"space network" cybersecurity dataset federated learning satellite paper 2023 2024 2025`
9. `CANSat-IDS satellite CAN traffic intrusion detection dataset ScienceDirect real simulated`
10. `Edge-IIoTSet dataset satellite communication scenario description`
11. `JAXA satellite telemetry anomaly dataset public open data`
12. `NASA PDS Earthdata spacecraft telemetry anomaly labeled dataset machine learning`
13. `real satellite cyberattack dataset public release 2025 2026 institution verified`
14. `HoneySat satellite honeypot NDSS 2026 dataset real attackers download`
15. `"SAT20" "TER20" dataset satellite testbed simulation generation methodology paper`
16. `CANSat-IDS dataset download availability GitHub Mendeley public release`
17. `Edge-IIoTSet dataset satellite communication scenario description` (repeat, different angle)

Direct primary-source fetches performed (this session, to upgrade SOURCE CLAIM to FACT
wherever possible):

- `github.com/kun9717/STIN-data-set` — repo page
- `github.com/kun9717/STIN-data-set` raw `README.md`
- `github.com/esa/anomaly-dataset` — official ESA org repo page
- `zenodo.org/records/12528696` — ESA-ADB data record (license, size confirmed directly)
- `github.com/khundman/telemanom` — NASA SMAP/MSL repo
- `data.mendeley.com/datasets/7n2d42pm3n/3` — CuCD-ID record (license confirmed directly)
- `github.com/sapols/Satellite-Telemetry-Anomaly-Detection` — LASP repo
- `github.com/patrickfleith/awesome-spacecraft-engineering-datasets` — curated list
- `mdpi.com/2673-4001/7/1/3` (SpIDER paper) — **FAILED: HTTP 403 Forbidden**, could not
  verify this paper's dataset claims directly; anything attributed to SpIDER elsewhere in
  this research is search-snippet-level (SOURCE CLAIM), not independently confirmed.

## Findings preserved as originally encountered (including unfavorable ones)

### NASA SMAP/MSL — favorable, but with a real limitation
Confirmed real, well-documented, easy to access. **The limitation that must not be
dropped:** it contains zero cyberattack labels. Every anomaly is a physical/hardware
fault. Any temptation to describe this as "satellite cybersecurity data" would be false.

### ESA-ADB — favorable, but large and only partially enumerable
Confirmed real, clear CC BY 3.0 IGO license, largest dataset found (11.6 GB). **Limitation
preserved:** I could not retrieve a complete per-channel feature count or exact total
sample count from what was fetched this session — the 844-events/148-anomalies figure is
from a *benchmark paper about* ESA-ADB, not the dataset's own primary documentation page,
and covers only 2 of the 3 missions. Treat exact record/feature counts as unconfirmed
until the archive is actually downloaded and inspected.

### STIN/SAT20 — the most tempting, and the most flawed, candidate
This dataset is attractive because it is explicitly satellite-framed, explicitly
attack-labeled, and explicitly ships FL splits — it matches this project's architecture
better than any other candidate on paper. **This is exactly the finding that must not be
allowed to override the license problem:** fetching the repo's `README.md` and
`LICENSE.txt` directly this session produced only a bare copyright notice attributing the
work to "National Engineering Lab for Next Generation Internet Interconnection Devices,
BJTU" — no MIT/Apache/CC grant, no explicit "you may use this for research" statement.
I could not find exact record counts or a complete feature list either, despite two
separate fetch attempts (repo root page, raw README). **This is preserved as a blocking
issue, not glossed over because the rest of the dataset looks like a great fit.**

### CANSat-IDS — inconclusive, not fabricated
The *paper* (Computers & Security, 2024) is real and describes a genuine ANN+K-SVM
CAN-bus intrusion detection approach for satellites. Repeated searches for an actual
downloadable dataset release associated with it returned nothing conclusive. **Rather
than assume a dataset exists because a credible paper exists, this is explicitly marked:
dataset availability UNCONFIRMED.** Do not treat CANSat-IDS as a usable dataset without
directly contacting the authors or finding a release this search did not surface.

### Edge-IIoTSet — a rejection, kept for transparency
Some papers apply Edge-IIoTSet to "satellite-based IoT" scenarios. Direct investigation
of the dataset's own origin shows it is built from a **physical seven-layer IIoT smart
factory testbed** — sensors like pH meters, flame sensors, soil-moisture sensors. It is
not satellite data by any reasonable reading; papers using it for satellite-adjacent
framing are applying it by analogy, not because the dataset itself contains satellite
data. **Classified NOT SUITABLE and excluded from the comparison table's serious
candidates**, but recorded here so the reasoning for excluding it is auditable later.

### HoneySat — real, exciting, but must not be overstated
This is a genuinely real 2026 NDSS paper with a real Zenodo artifact
(10.5281/zenodo.17548980). The instinct to call this "real satellite cyberattack data"
must be resisted: the *satellite* is an emulated honeypot (the paper's own description:
"capable of convincingly simulating a real-world CubeSat"). What's real is the 22
attacker interactions. **This is preserved precisely because it is the closest thing to
a Category A dataset found, and precisely because it still does not qualify as one** —
exactly the kind of near-miss the research prompt asked not to blur.

### LASP telemetry — a real institution, but genuinely under-documented
LASP (Laboratory for Atmospheric and Space Physics, CU Boulder) is a real, credible
research institution. The GitHub repo fetch, however, could not surface a specific
spacecraft/mission name, exact record counts, or a license. This candidate is preserved
in the report specifically as an **UNVERIFIED, not-yet-actionable lead** rather than
dropped, because provenance-wise it is promising, but nothing beyond "five representative
time series datasets" and two example column names ("Temperature (C)",
"WheelTemperature") could be confirmed.

## Direct quotes captured during source verification (for audit trail)

- STIN repo copyright notice (fetched directly): *"Copyright (c) 2020 National
  Engineering Lab for Next Generation Internet Interconnection Devices, BJTU"*
- Search-engine synthesis on STIN's generation method: *"The authors simulate a real
  scenario for both the terrestrial network and satellite network"* — this is the basis
  for classifying STIN/SAT20 as simulated, not real, satellite data.
- ESA-ADB Zenodo page (fetched directly): license *"CC BY 3.0 IGO"*; three files
  *"ESA-Mission1.zip (3.8 GB), ESA-Mission2.zip (4.1 GB), ESA-Mission3.zip (3.7 GB)"*.
- CuCD-ID Mendeley page (fetched directly): *"operates under CC BY 4.0 licensing,
  permitting broad reuse with attribution."*
- HoneySat search synthesis: *"HoneySat successfully deceived adversaries in the wild
  and collected 22 real-world adversarial interactions... the first public dataset of
  real cyber interactions against a satellite-specific honeypot."*

## Explicit UNVERIFIED list (do not treat as fact without further work)

- STIN/SAT20 exact record counts and complete feature list
- STIN/SAT20's actual license terms (beyond the bare copyright notice found)
- ESA-ADB exact total sample/channel count (only the benchmark-paper subset figure,
  844/148, was found, covering 2 of 3 missions)
- LASP telemetry: specific spacecraft, record counts, label ground truth, license
- CATS: satellite-specificity, size, license
- AegisSat: exact dataset structure, size, license (paper abstract only)
- HoneySat: license of the Zenodo artifact; exact data schema of the 22 interactions
- SpIDER paper's own dataset claims (page fetch returned HTTP 403; not independently
  confirmed this session)

This file should be read alongside the polished report, not instead of it — the polished
report's confidence language (FACT/SOURCE CLAIM/ENGINEERING ASSESSMENT/UNCERTAINTY) is
derived from exactly the findings preserved above.
