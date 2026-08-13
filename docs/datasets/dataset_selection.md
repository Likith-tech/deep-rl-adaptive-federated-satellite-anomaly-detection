# Dataset Selection — Phase 1

## Dataset selected

**NSL-KDD** (`KDDTrain+.txt`, `KDDTest+.txt`)

NSL-KDD is a refined version of the original KDD Cup 1999 dataset, published by
the University of New Brunswick's Canadian Institute for Cybersecurity (CIC).
It removes the redundant records and skewed class distribution of the
original KDD'99 dataset while keeping the same 41-feature connection-record
schema.

- Training set: 125,973 labeled connection records
- Test set: 22,544 labeled connection records
- 41 features (numerical + categorical) + attack label + difficulty score
- Labels: `normal` or one of 22 specific attack types across 4 attack
  categories (DoS, Probe, R2L, U2R)

## Why selected

This is an engineering decision made after checking the practical
availability of every candidate dataset (see "Alternatives considered"
below). The deciding factors, in order:

1. **Actual availability.** Of the three network-traffic candidates
   discussed for this project, NSL-KDD was the only one whose data could
   be reliably and reproducibly acquired by an automated script from this
   environment at the time of writing (2026-08-08). See "Reproducibility"
   for exactly how, and why the other two were not viable right now.
2. **Small, fast, reproducible.** ~22.5 MB total, no authentication, no
   massive multi-gigabyte downloads. This matters for a capstone project
   where the same pipeline needs to run repeatedly during development,
   in CI, and during grading/demo without depending on a fragile
   multi-GB download succeeding every time.
3. **Feature quality.** 41 features mixing categorical
   (`protocol_type`, `service`, `flag`) and numerical (byte counts,
   connection-rate statistics, host-based traffic statistics) columns —
   representative of the kind of feature engineering later phases will
   need to handle for real satellite telemetry/traffic.
4. **Labels support both binary and multi-class objectives.** The
   dataset preserves the specific attack name per record, which lets us
   keep `label_binary` (normal/anomaly) as the Phase 1-3 objective while
   retaining `label_original` for later multi-class experiments.
5. **Right-sized for non-IID partitioning and federated simulation.**
   125,973 training records is enough to partition across a realistic
   number of simulated satellite clients (Phase 6+) without any single
   partition being statistically empty, but small enough that federated
   rounds/simulation stay fast during development.
6. **Explicitly named as the baseline/prototyping candidate** in the
   project's own dataset strategy — appropriate given point 1.

This is a **practical/engineering choice, not a claim of superiority**.
CICIDS2017 and UNSW-NB15 remain reasonable candidates for a secondary
validation experiment later (Phase 17-18) if their official distribution
channels become reliably scriptable, or if manual download is done
outside this environment and the resulting CSVs are dropped into
`data/raw/` (the loader is not hard-coded to NSL-KDD's schema needing
adaption — see limitations below).

## Alternatives considered

| Dataset | Relevance | Size | Official access (checked 2026-08-08) | Verdict |
|---|---|---|---|---|
| **CICIDS2017** | Strong — realistic modern traffic (2017), 78 CICFlowMeter features, 8 attack types, real `Timestamp` column enabling a genuine time-based split | ~225 MB zipped CSVs | Official page (`unb.ca/cic/datasets/ids-2017.html`) links to `cicresearch.ca`, which returns HTTP 302 to a generic index page for any non-interactive request (curl, with and without browser-like headers) — not reliably scriptable right now | Not selected for Phase 1; strong secondary-validation candidate if downloaded manually later |
| **UNSW-NB15** | Strong — 49 features, `attack_cat` multi-class label, includes `Stime`/`Ltime` flow timestamps | ~600 MB across 4 CSVs (+ ground truth) | Official page (`research.unsw.edu.au/projects/unsw-nb15-dataset`) hosts data behind a SharePoint link requiring interactive/browser access, not a direct scriptable URL | Not selected for Phase 1; candidate for secondary validation |
| **NSL-KDD** (selected) | Moderate — older (KDD'99-derived) feature set, no continuous timestamp, but clean, small, and well documented | ~22.5 MB total | Official UNB page (`unb.ca/cic/datasets/nsl.html`) explicitly states *"We apologize, this dataset is no longer available"* — the canonical source is confirmed offline. Acquired instead from a long-standing, widely-cited GitHub mirror (`jmnwong/NSL-KDD-Dataset`) whose file sizes and record counts (125,973 / 22,544) match the dataset's well-documented canonical statistics exactly, confirming content integrity | **Selected** — best available compromise between relevance and actually-obtainable data |
| **STIN / satellite-specific traffic dataset** | Would be ideal for domain fit | Unknown | No practically accessible, licensable, downloadable satellite-network intrusion dataset was found during this phase's research pass. This is a known gap in the field — see below. | Not used; satellite behavior is simulated instead (see next section) |

## Satellite-domain limitation

**NSL-KDD is not a satellite dataset.** It is derived from simulated
military-network traffic (the original 1998 DARPA Intrusion Detection
Evaluation dataset) captured in a terrestrial LAN testbed. It contains
no satellite-specific characteristics: no orbital/link-layer metadata, no
intermittent-connectivity patterns, no bandwidth/latency profile
representative of satellite links.

This limitation is explicit and intentional for Phase 1. No claim is
made, anywhere in this project, that NSL-KDD is satellite traffic. The
"satellite" aspect of this project is introduced entirely through
**simulation** applied on top of this terrestrial dataset, starting in
Phase 6.

## How it will be used as a satellite simulation

```text
NSL-KDD network traffic (terrestrial IDS records)
        ↓
non-IID partition across N simulated satellite clients
   (Phase 6 — e.g. Dirichlet-based label-skew partitioning)
        ↓
each partition = one simulated satellite's local traffic
        ↓
simulated satellite constraints applied per client:
   - intermittent availability / connectivity windows
   - latency
   - bandwidth limits
   - CPU/resource budget
   - update staleness
        ↓
federated local training per satellite client (Phase 7+)
```

This satellite simulation layer will be implemented under `src/satellite/`
in a later phase — it does not exist yet. Phase 1 only produces the
cleaned, split, and labeled base dataset that the simulation will
partition.

## Reproducibility

Another developer can reproduce this exact dataset acquisition as
follows:

```bash
mkdir -p data/raw
curl -o data/raw/KDDTrain+.txt \
  "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt"
curl -o data/raw/KDDTest+.txt \
  "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt"
```

Expected result: `KDDTrain+.txt` is 125,973 lines / ~19.1 MB;
`KDDTest+.txt` is 22,544 lines / ~3.4 MB. `scripts/inspect_dataset.py`
will fail clearly and explain what's missing if these files are absent
or the wrong size.

Neither file is committed to this repository (see `.gitignore` —
`data/raw/*` is excluded). Anyone cloning this repo must run the two
commands above (or an equivalent manual download) before running the
preprocessing pipeline.

**Note on provenance:** the GitHub mirror above is not UNB's official
distribution channel (which is currently offline — see table above). It
was chosen because (a) the official source was confirmed dead, (b) it is
a long-standing, widely referenced mirror used across the ML/security
research community, and (c) its record counts and per-column value
ranges were spot-checked against NSL-KDD's published canonical
statistics and match exactly. If UNB's official hosting is restored,
this project should switch back to it.

No copyrighted/private/paid dataset is used or stored in this
repository.

## Status update (post-Phase 8): real spacecraft dataset research

As of this update, **NSL-KDD remains the sole dataset used anywhere in this
project**, and every result from Phase 1 through Phase 8 was produced with it.
Nothing above this section has been changed to produce this update.

Ahead of a planned DRL phase, a separate research pass evaluated whether a
**real** spacecraft/satellite dataset exists that could add scientific value
alongside NSL-KDD. That research — search methodology, every candidate
considered, verification evidence, and explicit limitations — is archived in
full under `docs/research/`:

- `docs/research/claude-research/phase-9-real-spacecraft-dataset-search-prompt.md`
  — the exact research brief used
- `docs/research/claude-research/phase-9-real-spacecraft-dataset-search-response.md`
  — raw findings, including unfavorable/unverified ones
- `docs/research/dataset-verification/phase-9-dataset-verification.md` — the
  full 23-section verification report
- `docs/research/dataset-verification/real-spacecraft-dataset-comparison.md` —
  scored comparison table

**Summary of that research (full detail in the files above):** no publicly
available dataset was found that combines real spacecraft/satellite telemetry
with real cyberattack labels. Real spacecraft telemetry datasets that do exist
publicly (e.g. NASA SMAP/MSL, ESA-ADB) are labeled for **operational/physical
anomalies**, not cyberattacks. Datasets with genuine cyberattack labels framed
around satellites (e.g. STIN/SAT20, CuCD-ID) are **simulated/testbed** data,
not real spacecraft traffic — the same category of construction as this
project's own Phase 4 satellite simulation, not a stronger claim than it.

**As of this update: no real spacecraft dataset has been integrated into this
project. The satellite client layer (Phase 4 onward) remains entirely
simulated, exactly as described earlier in this document.** Any future
integration of a real-spacecraft dataset is planned as an **additive, separate
validation track** alongside NSL-KDD — not a replacement of it, and not a
retroactive change to any Phase 1-8 result — and requires explicit approval
before implementation begins.
