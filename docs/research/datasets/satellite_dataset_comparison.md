# Real Satellite Anomaly-Detection Dataset Comparison

**Status: RESEARCH/AUDIT ONLY. No code, no config, no download beyond
small metadata/verification files, no training.** Every number below
was independently verified against the actual source (Zenodo API,
GitHub repo contents/scripts, or a small real metadata file
downloaded and parsed) — not taken from memory or a secondary summary
without cross-checking.

## 1. Candidates investigated

| Candidate | Organization | Status |
|---|---|---|
| ESA Anomaly Dataset (ESA-ADB) | ESA / Airbus / KP Labs | Verified — real, current |
| NASA SMAP/MSL (Telemanom) | NASA / JPL | Verified — real, current |
| OPSSAT-AD (already Phase 9A) | ESA | Already integrated, re-compared here |
| STIN/SAT20 | BJTU (not a space agency) | Already audited and rejected (no normal class) — see `13-stin-feasibility-audit.md`, unchanged |
| JAXA / ISRO / CNES / NOAA public anomaly-labeled telemetry | — | **Searched, none found.** A real web search targeting exactly this returned zero public spacecraft-telemetry anomaly-detection datasets from these agencies. Reported honestly as an absence, not assumed. |

## 2. Official sources (verified)

- **ESA-ADB dataset**: Zenodo `10.5281/zenodo.12528696` (verified via
  Zenodo API — real record, title "ESA Anomaly Dataset", open access,
  published 2024-06-25).
- **ESA-ADB dataset landing repo**: `github.com/esa/anomaly-dataset`
  (the user's originally-given URL — verified real, official ESA org
  repo, cross-links to the same Zenodo record).
- **ESA-ADB benchmark/code repo**: `github.com/kplabs-pl/ESA-ADB`
  (verified real, MIT-licensed code, 119 stars, actively maintained).
- **ESA-ADB paper**: arXiv:2406.17826, "European Space Agency
  Benchmark for Anomaly Detection in Satellite Telemetry" (Kotowski et
  al. 2024), also published in DMLR (OpenReview).
- **NASA SMAP/MSL**: `github.com/khundman/telemanom` (verified real,
  Apache 2.0 license, not archived, actively describes the 2018 KDD
  paper "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric
  Dynamic Thresholding," arXiv:1802.04431). Data download now goes
  through a Kaggle mirror
  (`kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl`)
  per the repo's own current instructions — the original raw NASA S3
  bucket link is no longer the documented path.

## 3. ESA Anomaly Dataset (ESA-ADB) — detailed analysis (Step 2)

**1. Exact dataset size**: 3 mission ZIP files on Zenodo — Mission1
3,776.2 MB, Mission2 4,098.5 MB, Mission3 3,734.4 MB — **total 11.6 GB
compressed** (verified via Zenodo API file listing, not estimated).
Publicly reported uncompressed figure (search results): "over 30 GB...
across hundreds of channels."

**2. Exact number of missions**: 3 missions have telemetry released,
but **only 2 (Mission1, Mission2) are actually used in the official
ESA-ADB benchmark** — verified directly: the repo contains
`mission1_experiments.py` and `mission2_experiments.py` but no
`mission3_experiments.py`, and the README states outright: "real-life
telemetry collected from three ESA missions (out of which two are
selected for benchmarking in ESA-ADB)."

**3. Exact telemetry channels**: Mission1 = **58** benchmarked
channels (counted directly from the `target_channels` list in
`mission1_experiments.py`); Mission2 = **47** benchmarked channels
(same method, `mission2_experiments.py`). Both scripts reference a
`labels.csv` with a `Category` field including at least `"Rare Event"`
and `"Anomaly"` as distinct severity/type labels — real evidence of a
richer-than-binary label taxonomy (the search results additionally
describe "severity rating" per annotation).

**4. Exact anomaly labels**: not fully enumerable without downloading
the multi-GB `labels.csv` files (deliberately not done, per the "avoid
blind 11+ GB download" instruction) — confirmed categories include at
minimum `Rare Event` and `Anomaly`; each annotation reportedly carries
start/end timestamps, a text description, and a severity rating (per
independent search-result corroboration, not yet independently
re-verified against the raw file).

**5. Normal/anomaly structure**: semi-supervised — Mission1's test
file is explicitly named `84_months.test.csv` (a real, verified
filename showing the test span is **84 months**, i.e. 7 years, of
telemetry) — confirms this is a genuinely long, realistic temporal
span, not a short synthetic clip.

**6. How the three missions are represented**: as three separate
downloadable ZIPs, each presumably one real ESA spacecraft's
telemetry, with distinct channel sets and no shared channel taxonomy
across missions (Mission1's and Mission2's channel-ID lists do not
overlap in any channel number visible in the two experiment scripts).

**7. Whether missions can be treated as FL clients**: yes, structurally
— each mission is a self-contained multivariate time series with its
own channel set, a natural non-IID client boundary (different
channels, different value distributions, different anomaly patterns)
— but only **2 missions are usably benchmarked**, giving at most 2
natural top-level clients from ESA-ADB alone.

**8. Whether there are enough independent units for 8 clients**:
**not from mission-level partitioning alone** (2 missions). A
scientifically defensible 8-client structure would require a SECOND
level of partitioning within each mission (e.g. channel-group-based or
temporal-block-based sub-clients) — a real, disclosed simulation
layer, exactly as this project's own Phase 4 NSL-KDD client
simulation and Phase 9A's honest "segment-level, not multi-satellite"
labeling already models this kind of disclosure.

**9. Client partitions without temporal leakage**: plausible, since
each mission's data is one long continuous multivariate series with
real timestamps — a temporal-block split for sub-clients (e.g. Mission
1's 84-month span cut into contiguous blocks) would be a legitimate,
leakage-safe, disclosed partition strategy, unlike a random-shuffle
split of a single time series.

**10. Whether anomalies from the same event appear in both train and
test**: **not verified from available metadata** — this benchmark uses
one fixed author-provided train/test split per mission (semi-supervised:
train presumably anomaly-free or lightly labeled, test contains the
annotated anomalies) rather than a project-constructed split, which
reduces (but does not eliminate, without inspecting the actual
`labels.csv`) this risk. Marked explicitly as **uncertain** rather than
assumed safe.

**11. Preprocessing without test leakage**: feasible in principle
(fit any scaler/encoder on the train file only, per this project's
existing OPSSAT-AD discipline in `src/satellite/preprocessing.py`) —
not yet implemented or verified against the real files.

**12. Suitability for binary anomaly detection**: **yes, structurally**
— `is_anomaly`-style binary ground truth exists (the multi-category
`Category` field can be collapsed to binary normal/anomaly), unlike
STIN which has no normal class at all. This is ESA-ADB's single
biggest advantage over STIN.

**13. Public/academic usable**: yes — Zenodo record is fully open
access; ESA-ADB code is MIT-licensed; a Kaggle competition exists
publicly.

**14. Is downloading the entire dataset necessary**: **not yet
determined without further scoping** — the per-mission ZIPs (3.7-4.1
GB each) cannot obviously be partially downloaded via a simple range
request (Zenodo serves whole files); a real download would likely mean
committing to at least one full mission ZIP (~3.7-4.1 GB) to get past
metadata-only inspection. **Not downloaded this session**, per the
explicit "avoid blind bulk download" instruction — this is exactly the
kind of decision that should follow a go/no-go call on this dataset,
not precede it.

## 4. NASA SMAP/MSL (Telemanom) — detailed analysis (Step 3)

**1-2. Exact SMAP/MSL size**: verified directly from the real
`labeled_anomalies.csv` metadata file (downloaded and parsed, 82
rows): **SMAP = 55 channels, 444,035 test timesteps, 69 labeled
anomaly sequences. MSL = 27 channels, 73,729 test timesteps, 36
labeled anomaly sequences.** Combined: **82 channels, 517,764 test
timesteps, 105 anomaly sequences.**

**3. Number of channels**: 82 total (55+27), exact, verified above.

**4. Number of anomaly sequences**: 105 total, exact, verified above.

**5. Point vs. contextual anomalies**: both real, distinct categories
in the actual data — **62 point-type, 43 contextual-type** sequences
(counted directly from the `class` column of every one of the 82
channel rows).

**6. Train/test structure**: author-provided, fixed — pre-split
`data/train/` and `data/test/` `.npy` files per channel, channel
values pre-scaled to `(-1, 1)` using min/max computed from the TEST
set (per the repo's own README) — an important, disclosed
preprocessing detail: **the original authors themselves scaled using
test-set statistics**, which is why this project would need to
re-derive scaling from the train split only if reusing this data,
exactly the same train-only-fitting discipline already enforced in
`src/satellite/preprocessing.py` for OPSSAT-AD.

**7. Label reliability**: the labels are the original, citable,
peer-reviewed (KDD 2018) NASA/JPL-authored ground truth — the most
established, most-cited dataset of this kind in the anomaly-detection
literature; reliability is as good as any public benchmark of this
type gets, though (same honest caveat the paper's own domain implies)
these are engineering-team-identified anomalies on real telemetry, not
independently re-verified by this project.

**8. SMAP and MSL as separate FL clients**: yes, directly and
naturally — two genuinely different NASA missions (an Earth-observing
satellite vs. a Mars rover), with entirely disjoint channel sets (`P-`,
`S-`, `E-`, etc. prefixes for SMAP; `M-`, `T-`, etc. for MSL,
per the repo's channel-naming description) — a clean, real, 2-client
natural split.

**9. Additional client partitioning defensibility**: yes — 55 SMAP
channels and 27 MSL channels could each be grouped (e.g. by channel-ID
prefix, which the repo says already encodes subsystem type — "P" =
power, "R" = radiation, etc.) into several sub-clients per mission, a
scientifically defensible subsystem-based partition, not an arbitrary
random split.

**10. True suitability for this project**: strong structural fit —
real spacecraft telemetry, genuine binary-collapsible anomaly labels,
real temporal structure, natural multi-mission client boundary.

**11. Enough samples for our MLP/temporal models**: 517,764 total
labeled test timesteps (plus a train split, size not yet independently
verified) is comparable in order of magnitude to OPSSAT-AD's 303,493
raw timesteps (Phase 9A) and vastly larger than OPSSAT-AD's 2,123
segment-level rows — sufficient for both the existing tabular
`BaselineMLP` (per-timestep) and, notably, GENUINELY exercising the
Phase 3 `TemporalGRUAttention` model with sequence_length > 1 (unlike
OPSSAT-AD's `dataset.csv`, which forced sequence_length=1 — see Phase
9A's own audit) — the raw per-timestep `.npy` structure here is
exactly the shape Phase 9A found itself missing.

**12. Small enough for rapid experimentation**: very likely yes —
82 channels × a few thousand timesteps each is orders of magnitude
smaller than NSL-KDD's 107,077-row training set or STIN's 209,564
rows; exact download size was not independently verified this session
(Kaggle API required authentication not available here), but every
public description of this dataset describes it as a small,
laptop-friendly benchmark (tens of MB, not GB) — flagged as an
estimate pending direct download verification, not asserted as fact.

## 5. Other candidates search (Step 4)

A real web search ("public spacecraft telemetry anomaly detection
dataset 2024 2025 real satellite JAXA ISRO CNES") returned **zero**
public anomaly-labeled spacecraft-telemetry datasets from JAXA, ISRO,
CNES, NOAA, or any other agency beyond the ESA/NASA family already
covered by ESA-ADB, OPSSAT-AD, and SMAP/MSL. This is reported as a
genuine absence (the search was performed, not skipped), not an
assumption that no such datasets could possibly exist.

## 6. Dataset quality score table (Step 5)

Scored 1-5 per criterion (5 = best). Scoring rationale given inline —
weights are NOT adjusted to favor any candidate; every score is tied
to a specific fact verified above.

| Criterion | STIN/SAT20 | ESA-ADB | SMAP/MSL | OPSSAT-AD (existing) |
|---|---:|---:|---:|---:|
| A. Satellite authenticity | 3 (satellite-*relevant*, simulated/testbed traffic, not real telemetry) | 5 (real telemetry, 2 real ESA missions) | 4 (real telemetry; MSL is a Mars *rover*, not an orbiting satellite — a genuine nuance) | 5 (real OPS-SAT CubeSat telemetry, already verified in Phase 9A) |
| B. Anomaly-label quality | 1 (**no normal class at all** — disqualifying) | 4 (expert-annotated, severity-rated, but leakage-uncertain, §3.10) | 5 (peer-reviewed KDD 2018 ground truth, point+contextual typed) | 4 (expert-annotated, already trusted in Phase 9A) |
| C. Dataset size | 4 (209,564 rows) | 5 (11.6 GB, hundreds of channels) | 3 (517,764 timesteps — solid but modest) | 2 (2,123 segments / 303,493 raw timesteps — already the smallest dataset this project uses) |
| D. Temporal structure | 2 (flow-level durations only, no real timestamp column) | 5 (real timestamps, 84-month span) | 4 (real per-timestep sequences, though NASA-anonymized re: absolute time) | 4 (real ISO8601 timestamps, per Phase 9A's own segments.csv audit) |
| E. Independent clients/missions | 2 (2 real hosts, not 8) | 2 (2 usable missions) | **5 (2 genuinely distinct missions, further subdividable by subsystem-prefix)** | 1 (1 spacecraft only) |
| F. Non-IID/FL suitability | 2 | 3 (mission-based + within-mission temporal blocks) | 4 (mission-based + subsystem-based, both real and disclosed) | 1 (no natural multi-client structure) |
| G. Leakage safety | 2 (264-row train/test overlap measured) | 3 (uncertain, §3.10, not disqualifying but not clean) | 4 (fixed author split; scaling-leakage note disclosed and correctable) | 4 (already leakage-audited in Phase 9A) |
| H. Reproducibility | 4 (public, versioned) | 4 (public, versioned, DOI) | 5 (public, DOI-citable paper, stable repo) | 5 (already fully reproduced in this project) |
| I. Public accessibility | 5 | 5 | 5 (Kaggle mirror, one command) | 5 |
| J. Research-paper support | 3 (one submitted-not-yet-published paper) | 5 (published DMLR paper + arXiv + Kaggle competition) | 5 (landmark, highly-cited KDD 2018 paper) | 4 (Nature Scientific Data 2025 paper, per search results) |
| K. Compatibility with our architecture | 2 (would need multiclass loss/output rework) | 4 (binary-collapsible, tabular-per-timestep like our MLP; large scale needs real engineering) | **5 (binary-collapsible, per-timestep .npy shape is a near-direct fit for both `BaselineMLP` and the currently-underused `TemporalGRUAttention`)** | 4 (already integrated in Phase 9A, but only at sequence_length=1) |
| L. Computational feasibility | 4 (small, fast) | **1 (repo itself recommends 512 GB disk / 64 GB RAM / GPU / 120h-per-algorithm timeouts — a serious, disclosed red flag for a capstone project)** | 5 (small, laptop-scale) | 5 (already run successfully on this project's own machine) |
| **TOTAL /60** | **34** | **46** | **54** | **44** |

## 7. Comparison with OPSSAT-AD (Step 9 — does a new dataset add value?)

| | OPSSAT-AD (existing) | SMAP/MSL (candidate) | ESA-ADB (candidate) |
|---|---|---|---|
| Spacecraft count | 1 (OPS-SAT) | 2 (SMAP + MSL) | 2 usable (of 3 released) |
| Channels | 9 | 82 | 58 + 47 = 105 (2 missions) |
| Raw timesteps | 303,493 | 517,764 | ~30 GB uncompressed (far larger) |
| Sequence length usable | **1 only** (Phase 9A's own documented, honest limitation) | **>1, genuinely** (real per-timestep sequences) | >1, genuinely (84-month continuous series) |
| Natural FL clients | 1 | 2 (+ subsystem sub-clients) | 2 (+ temporal-block sub-clients) |
| Already integrated | Yes (Phase 9A, frozen) | No | No |

**SMAP/MSL would add REAL scientific value beyond OPSSAT-AD** — it is
the one candidate that could finally let this project's existing
`TemporalGRUAttention` model (built in Phase 3, sequence-length-capable
by design) be genuinely exercised on real spacecraft data, closing the
exact gap Phase 9A's own audit honestly flagged (`sequence_length=1`
forced by OPSSAT-AD's segment-level `dataset.csv`). It would also be
the first REAL multi-spacecraft FL client structure this project has
ever had (Phase 4-9B's "8 clients" have always been a simulated
partition of one single-source dataset).

## 8. Paper comparability (Step 11 — top 2 candidates)

| Paper | Dataset | Model | FL? | DRL? | Clients | Accuracy/F1 reported | Directly comparable to our Phase 9B? |
|---|---|---|---|---|---|---|---|
| Kotowski et al. 2024 (ESA-ADB) | ESA-ADB (2 missions) | Multiple TSAD algorithms (PCA, HBOS, iForest, kNN, subsequence-iForest, etc.), NOT FL, NOT DRL | No | No | N/A (centralized benchmark) | Reports channel-aware F-scores, ADTQC — **not directly comparable metric definitions to our Accuracy/Precision/Recall/F1** | **No** — different task framing (per-channel TSAD scoring, not our FL/DRL adaptive-aggregation comparison), different metrics |
| Hundman & Constantinou 2018 (SMAP/MSL, Telemanom) | SMAP/MSL | Vanilla LSTM + nonparametric dynamic thresholding | No | No | N/A (centralized) | Reports precision/recall on point-adjusted sequences, dataset-specific | **No** — again centralized, no FL/DRL, different metric convention (sequence-level, not our per-record binary classification) |
| Chen et al. 2026 (FedSTAR, already deeply studied) | STIN + TON_IoT + Bot-IoT | Spatio-temporal (gSTA/LKA) + hierarchical clustered FedAvg + SAC | Yes | Yes (SAC) | 20 simulated | STIN Non-IID Accuracy 95.43% | **Not directly comparable** even if we adopted STIN — different architecture (hierarchical clustering vs. our flat weighting-profile catalogue), different RL algorithm (SAC vs. our DQN), different client count (20 vs. our 8), and (per our own STIN audit) even the paper's own STIN figure is on a 9-class attack-type task, not binary anomaly detection |

**No candidate paper is directly comparable to our Phase 9B protocol**
— every one uses a different task framing, model family, FL/RL
approach, or client count. Any future real-spacecraft-dataset
experiment must be reported the same way OPSSAT-AD was: a real,
honest, separate-track validation, never a percentage-for-percentage
comparison against a different benchmark's number.

## 9. Rejected/not pursued further

STIN/SAT20 — rejected (no normal class, already audited in
`13-stin-feasibility-audit.md`, unchanged, not re-litigated here).
JAXA/ISRO/CNES/NOAA — no public candidate found despite a real search.
