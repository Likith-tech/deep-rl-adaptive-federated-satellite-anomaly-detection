# Dataset Decision — Real Spacecraft/Satellite Data

**Status: PROPOSED — AWAITING APPROVAL. Not integrated. NSL-KDD and Phases 0-8
unchanged. No implementation has begun as a result of this document.**

## 1. Best candidate

**OPSSAT-AD** (ESA OPS-SAT CubeSat telemetry anomaly benchmark), with **ESA-ADB** as a
recommended complementary secondary dataset — see section 7 (hybrid strategy) for why
this is proposed as a pair rather than a single winner.

## 2. Why it was selected

- Confirmed real, flown, ESA-operated spacecraft (not simulated) — verified directly via
  the arXiv preprint text, not taken on faith from a search snippet.
- Cleanly and directly confirmed license (CC-BY-4.0, checked on the Zenodo record
  itself).
- Peer-reviewed in a major data journal (*Nature Scientific Data*, 2025) — the strongest
  documentation/credibility signal of any real-telemetry candidate found.
- Small (19.5 MB) — negligible integration cost for a capstone timeline, unlike ESA-ADB's
  11.6 GB.
- Thematically close to this project's own framing: a single small satellite (CubeSat),
  not a large flagship mission or a Mars rover — a more honest analog to what this
  project's "SAT-01..SAT-08" simulated clients are meant to represent individually.

## 3. Why alternatives were rejected (or ranked below it)

- **ESA-ADB:** equally credible, larger and multi-mission (a genuine advantage — see
  section 7), but its 11.6 GB size and less-completely-retrievable per-mission
  documentation make it a secondary rather than primary pick.
- **NASA SMAP/MSL:** excellent and standard, but older (2018) and MSL is a Mars rover,
  not an orbital satellite — weaker thematic fit.
- **LASP telemetry:** credible institution, but too under-documented (no confirmed
  record counts, labels, or license) to act on without further firsthand inspection.
- **LENS:** real and the best multi-client structure found, but zero labels of any kind
  — would require inventing ground truth, which this project's own integrity rules treat
  cautiously.
- **STIN/SAT20:** the best-fitting candidate for cybersecurity + FL relevance on paper,
  but simulated (not real spacecraft) AND has an unresolved license — a real legal risk,
  not a minor caveat, for an academic deliverable.
- **CuCD-ID:** clean license, but simulated and single-CubeSat — adds little beyond what
  this project's own Phase 4 simulation already does.
- **CORTEX:** searched for directly this session under that exact name; **no dataset by
  this name was found**. Not included in scoring because it could not be located at all.

## 4. Whether it is genuinely real spacecraft data

**Yes, for the telemetry itself.** OPS-SAT was a real, launched, ESA-operated CubeSat
that flew from 2019 until its mission end in May 2024. OPSSAT-AD's 9 telemetry channels
and their anomaly labels come from that real flight history, not a simulator. **What is
NOT real:** any framing of this as "multiple satellites" or "a satellite network" — it is
one spacecraft's data.

## 5. What part of our architecture it supports

| Pipeline stage | Supported? |
|---|---|
| Real telemetry input | **Yes** |
| A. Temporal anomaly detection | **Yes — directly, this is its designed purpose** |
| B. Multiple satellite/client simulation | No — would require artificial partitioning of one spacecraft's data, which must be labeled simulated |
| C. Non-IID client partitions | Only via the same artificial partitioning as B |
| D. Federated learning | Only via the same artificial partitioning as B |
| E. Adaptive federated aggregation | Inherits whatever is simulated in B-D; not natively supported |
| F. DRL/DQN client weighting/selection | Same — not natively supported |

## 6. What it does NOT support

Cyberattack/intrusion detection of any kind (no attack labels exist in this dataset),
genuine multi-satellite federated learning (single spacecraft), and any claim that this
project's simulated 8-client structure is "real" merely because the underlying telemetry
for one hypothetical client would be real.

## 7. Whether additional data is needed — YES, and a hybrid strategy is recommended

**Do not force a single dataset to cover both real-spacecraft realism and
cybersecurity relevance — no candidate found does both.** The evidence supports a
two-dataset strategy, kept clearly separated rather than merged into one benchmark:

```
Dataset A — OPSSAT-AD (+ ESA-ADB as an optional larger secondary)
    → REAL spacecraft telemetry
    → validates temporal anomaly detection (Phase 3-style) on real data
    → metrics reported and labeled as "real spacecraft, operational anomalies"

Dataset B — NSL-KDD (already in use, unchanged)
    → REAL terrestrial cybersecurity traffic
    → continues to drive Phases 1-8's cyberattack-detection and FL/adaptive-FL work
    → metrics reported and labeled as "terrestrial, cyberattack detection"
```

These two tracks answer genuinely different scientific questions and their metrics
**must never be merged into a single number** — doing so would misrepresent an
operational-fault-detection result as a cybersecurity result, or vice versa.

### On NSL-KDD's role specifically (per the four options posed)

**Option D is recommended: NSL-KDD is used alongside real spacecraft data, each in its
own clearly-labeled track.** Not Option A alone (implies real data is peripheral, when
it deserves its own real validation claim), not Option B (cybersecurity pretraining
isn't what NSL-KDD is currently doing or needs to do), and explicitly not Option C
(replacing NSL-KDD would invalidate every Phase 2-8 comparison already made).

## 8. Proposed integration strategy (not yet implemented)

A new, narrowly-scoped phase — **"Real Spacecraft Telemetry Validation"** — positioned
after Phase 8 and before DRL (Phase 9), consuming OPSSAT-AD independently of the
NSL-KDD/FedAvg/adaptive-FL code path. It would not modify `src/federated/`, `src/training/`,
or any existing config.

## 9. Expected preprocessing

A new loader for the `segments.csv`/`dataset.csv` schema (9 channels, magnetometer +
photodiode), distinct from NSL-KDD's connection-record schema. Likely reusable pieces:
the sequence-windowing logic already built for Phase 3 (`src/preprocessing/sequences.py`)
may partially generalize, but the feature-scaling and encoding steps would need to be
new, since OPSSAT-AD's channels are physically different quantities than NSL-KDD's
network-flow features.

## 10. Expected temporal structure

Native multivariate time series across 9 channels, pre-segmented into 2,123 labeled
fragments (~20% anomalous). This maps more directly onto Phase 3's GRU+attention input
shape than NSL-KDD ever did — NSL-KDD's temporal framing in Phase 3 required
constructing artificial sequences from otherwise-independent connection records; OPSSAT-AD
is natively sequential.

## 11. Expected satellite/client partitioning strategy

**Simulated, explicitly labeled as such** — for example, partitioning by channel group
(magnetometer vs. photodiode) or by time-segment into pseudo-clients, mirroring how
Phase 4 partitions NSL-KDD today. This must never be described as "real multi-satellite
data" in any report or documentation — the underlying telemetry is real, the client
boundary is not.

## 12. Expected FL usage

Only as a secondary, clearly-labeled experiment showing whether this project's existing
FedAvg/non-IID/adaptive-FL machinery (Phases 6-8, unmodified) behaves reasonably when
fed a simulated partition of real (rather than terrestrial-NSL-KDD) telemetry — not as a
primary result, and not as evidence of genuine multi-satellite federated learning.

## 13. Expected limitations

- Single spacecraft — no genuine multi-satellite claim is supportable.
- No cyberattack labels — cannot support any intrusion-detection claim.
- Small dataset (2,123 segments) relative to NSL-KDD's 148,517 records — statistical
  power for any new benchmark will be more limited.
- New preprocessing pipeline required — added engineering effort, not a drop-in swap.

## 14. Exact official sources

- Data: [zenodo.org/records/12588358](https://zenodo.org/records/12588358) (CC-BY-4.0,
  confirmed directly)
- Code: [github.com/kplabs-pl/OPS-SAT-AD](https://github.com/kplabs-pl/OPS-SAT-AD)
- Secondary/complementary dataset — Data: [zenodo.org/records/12528696](https://zenodo.org/records/12528696)
  (ESA-ADB, CC BY 3.0 IGO, confirmed directly); official repo:
  [github.com/esa/anomaly-dataset](https://github.com/esa/anomaly-dataset)

## 15. Relevant research papers

- *"The OPS-SAT benchmark for detecting anomalies in satellite telemetry,"* Nature
  Scientific Data, 2025 — preprint verified directly: [arXiv:2407.04730](https://arxiv.org/abs/2407.04730)
- ESA-ADB paper: [arXiv:2406.17826](https://arxiv.org/abs/2406.17826)
- NASA SMAP/MSL (for context/comparison): Hundman et al., KDD 2018

---

**This document is a proposal. No dataset has been downloaded, no code has been
written against it, and no Phase 9/real-data phase has been started. Explicit human
approval is required before any of the above is implemented.**
