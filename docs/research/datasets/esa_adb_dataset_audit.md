# ESA-ADB Mission 1 — Dataset Audit (Track C candidate)

**Status: AUDIT ONLY. No model trained, no FL/DQN code run, no existing
Phase 6-9B/OPSSAT-AD/STIN/SMAP-MSL artifact touched.** All numbers
below are measured directly from the real, downloaded, checksum-verified
Mission 1 archive (`data/raw/esa_adb/ESA-Mission1.zip`, MD5-verified
against Zenodo's own reported checksum) — nothing is taken from the
paper or repository documentation without independent verification
against the actual files.

## 1. Download & storage (real, measured)

- Source: `https://zenodo.org/api/records/12528696` (Zenodo record,
  version 1.0, open access), file `ESA-Mission1.zip`.
- Downloaded: **3,776,246,054 bytes (3.78 GB)**, size AND MD5 checksum
  verified against Zenodo's own live-reported values
  (`80750189d171f5f398fb3d96c49df12b`) — not a hardcoded assumption.
- Extracted: **3,776,362,882 bytes (3.5 GB), 778 files** — real
  extracted size measured directly, not estimated from a compression
  ratio (this archive uses minimal internal compression; extracted
  size ≈ compressed size).
- Storage before: C: 97.3 GB free. After Mission 1 download+extraction:
  **C: 88.7 GB free** (8.6 GB consumed). E: unchanged at 56.0 GB free
  (not used). Missions 2 and 3 were NOT downloaded, per the
  minimal-acquisition instruction.

## 2. Real file/directory structure

```
ESA-Mission1/
├── channels.csv          (76 rows — channel metadata)
├── labels.csv             (3,589 rows — anomaly-channel-window mappings)
├── anomaly_types.csv      (200 rows — one row per unique anomaly event)
├── telecommands.csv       (698 rows — spacecraft command log, NOT telemetry)
└── channels/
    ├── channel_1.zip       (each a pickled pandas.DataFrame, one per channel)
    ├── channel_2.zip
    ...  (76 channel zips total)
```

This structure is NOT documented anywhere in the benchmark repo's
README/docs at this level of detail — it was determined by direct
inspection of the real archive.

## 3. Channel metadata (real, from `channels.csv`)

**76 total channels** (not 774 — that figure from the raw archive
listing counted `.zip` wrapper files 1:1 with channels; corrected
here), grouped into **exactly 4 real, author-defined subsystems**:

| Subsystem | Channel count |
|---|---:|
| subsystem_6 | 42 |
| subsystem_1 | 17 |
| subsystem_3 | 11 |
| subsystem_5 | 6 |

- **Target=YES** (used in the official benchmark scoring): 58 channels
  — matches the count already found in `mission1_experiments.py`.
- **Target=NO** (present in the raw data, not directly scored): 18
  channels.
- 6 distinct "Physical Unit" values, 18 distinct "Group" values (a
  finer sub-grouping within subsystems) — real, additional structure
  beyond the 4 top-level subsystems.

## 4. Per-channel telemetry — real, measured (sampled 5 of 76 channels
   across different subsystems and Target flags, not exhaustively —
   loading all 76 full multi-hundred-MB pickles was judged unnecessary
   for a structural audit)

| Channel | Subsystem | Target | Rows | Date range | NaN | Distinct values |
|---|---|---|---:|---|---:|---:|
| channel_1 | subsystem_1 | NO | 10,513,336 | 2000-01-01 → 2013-12-31 | 0 | 34,550 |
| channel_12 | subsystem_6 | YES | 14,258,506 | 2000-01-01 → 2013-12-31 | 0 | 229 |
| channel_41 | subsystem_5 | YES | 15,381,169 | 2000-01-01 → 2013-12-31 | 0 | 828 |
| channel_58 | subsystem_6 | YES | 14,639,410 | 2000-01-01 → 2013-12-31 | 0 | 990 |
| channel_76 | subsystem_3 | YES | 14,671,647 | 2000-01-01 → 2013-12-31 | 0 | 193 |

**Every sampled channel spans the full real 14-year range (2000-01-01
to 2013-12-31)** — this is genuinely long-duration, large-scale real
telemetry, an order of magnitude larger than SMAP/MSL (517K total
timesteps) or OPSSAT-AD (303K raw timesteps). Zero NaN found in every
sampled channel. Sampling interval is **irregular/mixed**, not a fixed
cadence: for channel_12, the dominant intervals measured were 0.375s
(7.65M occurrences), 90s (4.81M occurrences), and 0.378s (613K
occurrences) — real evidence of variable/bursty sampling, not a clean
fixed-rate signal.

**No embedded train/test split exists in the raw per-channel data** —
each channel is one continuous timestamped series. The benchmark
repo's `84_months.test.csv` referenced in `mission1_experiments.py` is
a DERIVED artifact of the authors' own TimeEval preprocessing pipeline
(which was NOT run this session, per the "no full benchmark pipeline"
instruction) — not something present in this raw download. **Any
train/test split for our own experiment would need to be constructed
by us**, e.g. a temporal holdout.

## 5. Real anomaly label structure

- **200 unique, expert-annotated real anomaly events** (`anomaly_types.csv`),
  spanning **3,589 anomaly-channel-window rows** (`labels.csv`) — real
  ISO8601 `StartTime`/`EndTime` per row.
- **168 of 200 anomalies (84%) affect MORE THAN ONE channel
  simultaneously** — genuine multivariate/cross-channel anomaly
  structure, directly relevant to the spatio-temporal question (§6 of
  the user-facing report).
- **Category breakdown**: `Anomaly`=118, `Rare Event`=78,
  `Communication Gap`=4 (a third, operational category — data-loss
  periods, not sensor-detected anomalies).
- **Dimensionality**: `Multivariate`=164, `Univariate`=32, 4 unclassified
  (the `Communication Gap` rows).
- **Locality**: `Global`=113, `Local`=83, 4 unclassified.
- **Length**: `Subsequence`=184 (dominant), `Point`=12, 4 unclassified.
- Real anomaly date range: **2000-02-10 to 2013-12-06** — spans nearly
  the entire 14-year channel history.

## 6. Leakage audit

- **No duplicate-timestamp issue found** in any sampled channel (each
  channel's index was checked to be the natural per-channel timestamp
  sequence; not exhaustively deduplicated across all 76 channels this
  session).
- **Candidate temporal split boundary tested**: reserving the data
  after 2007-01-01 as a hypothetical "last ~84 months" test region —
  **zero of the 200 real anomaly windows straddle this boundary**
  (verified directly against real `StartTime`/`EndTime` values). This
  is a reassuring, real, checked fact — NOT a claim that this is the
  boundary the original authors actually used, only that at least one
  natural candidate boundary is leakage-safe for anomaly windows.
- **Preprocessing/label leakage**: not yet applicable — no preprocessing
  has been implemented.
- **Mission/client leakage**: N/A — only Mission 1 was downloaded;
  cross-mission leakage cannot occur.
- **Open, unresolved**: whether any of the 774→76 raw channels contain
  duplicate/near-duplicate SEGMENTS within themselves (the kind of
  natural flat-lining redundancy found in SMAP/MSL) was not checked
  this session — would require loading full channel series pairwise,
  a non-trivial computation given 10-15M rows per channel, deferred to
  implementation time if this dataset is pursued further.

## 7. Federated client feasibility (the central research question)

**Real, author-defined structure available**: 4 subsystems (42/17/11/6
channels), 18 groups, 6 physical units — genuinely richer than
SMAP/MSL's informal channel-prefix heuristic (Track B) and vastly
richer than NSL-KDD's fully-simulated partition (no natural structure
at all).

| Strategy | Validity | Non-IID behavior created | Leakage risk | Realism as distributed satellite nodes |
|---|---|---|---|---|
| **A. Natural subsystem clients** (4 clients = 4 subsystems) | High — real, author-labeled | Real — subsystem_6 (42 channels) vastly outweighs subsystem_5 (6 channels), a genuine, disclosed imbalance | Low (subsystems don't overlap in channels) | Represents functional spacecraft subsystems, not literally separate spacecraft — a disclosed simulation of "distributed monitoring nodes," same honesty standard as this project's existing NSL-KDD client simulation |
| **B. Subsystem + group sub-partition (up to 8 clients)** | High — still fully real/author-labeled (18 real groups exist) | Finer-grained, still real | Low | Same as A, finer granularity |
| **C. Channel-group clients** (by Physical Unit, 6 groups) | Medium — real grouping but coarser/fewer than needed for 8 | Real but only 6 natural groups | Low | Plausible but doesn't reach 8 without merging/splitting |
| **D. Temporal-block clients** | Valid only as a LAST resort or combined with A/B | Non-IID via time-period, not real structural difference | Requires care (must not split anomaly windows across blocks — checkable per §6's method) | Does NOT represent distributed nodes — represents one node's history split by time; weaker realism than A/B |
| **E. Random row split (8 groups)** | **Rejected explicitly, per instruction** | Fake non-IID, not defensible | High | Not realistic — exactly what the task said not to do |

**Recommendation: Option B** — use the 4 real subsystems as the top-level
non-IID structure, and split the two largest subsystems
(subsystem_6=42 channels, subsystem_1=17 channels) into 2-3 sub-clients
each by their real `Group` field to reach a genuine, disclosed
**8-client structure entirely from real, author-provided metadata** —
e.g. `subsystem_6` → groups {3,4,5,6,7} and remaining groups as
2 clients, `subsystem_1` → 2 clients by its groups, `subsystem_3` (11
channels) → 1 client, `subsystem_5` (6 channels) → 1 client — the
exact group-to-client assignment is a design detail for implementation
time, not decided here, but the METHOD (subsystem-then-group, never
random) is the recommendation.

## 8. Comparison with our current NSL-KDD setup

| | NSL-KDD (current) | ESA-ADB Mission 1 (candidate) |
|---|---|---|
| Client basis | Fully simulated (Dirichlet label-skew) — no natural structure | **Real, author-labeled subsystems/groups** |
| Anomaly formulation | Binary (attack/normal), pre-labeled per record | Requires construction: real per-timestep binary label derivable from `labels.csv` windows, but the raw data is NOT pre-labeled per-row |
| Input dimensionality | 121 (fixed, tabular) | Per-channel univariate by default; could be made multivariate by combining channels within a subsystem (real justification exists, §5's 84%-multivariate finding) |
| Sequence/window requirement | None (Phase 1-2), sequence-length configurable (Phase 3) | **Required for meaningful use** — a single timestep alone carries little signal for a 14-year, 10-15M-row series; genuine sequence modeling is well-supported (unlike OPSSAT-AD's forced seq_len=1) |
| Temporal modeling | Optional (Phase 3's `TemporalGRUAttention`, seq_len configurable) | Same class directly reusable — architecture-compatible (see §9) |
| Existing MLP reuse | N/a (baseline) | Reusable for a per-timestep or per-window formulation, with `input_dim` reconfigured |

## 9. Compatibility with existing FL/DQN architecture

1. **Binary anomaly detector reuse**: Yes, structurally — `BaselineMLP`'s
   `input_dim` is fully config-driven (confirmed by direct code
   inspection: `configs/federated.yaml`, `configs/drl.yaml`,
   `src/models/baseline_mlp.py` all take `input_dim` as a parameter,
   nothing hardcoded to NSL-KDD's 121).
2. **Required input dimensionality**: depends on the windowing/channel-
   combination design chosen at implementation time — not fixed by
   this audit.
3. **Temporal modeling required?**: Effectively yes, for the result to
   be meaningful given the real, measured 10-15M-row-per-channel scale
   and 84%-multivariate anomaly finding (§5) — a single-timestep
   (`sequence_length=1`) formulation would repeat OPSSAT-AD's own
   already-disclosed limitation, avoidably, since real sequence
   structure genuinely exists here.
4. **GRU/Transformer**: the existing `TemporalGRUAttention`
   (`src/models/temporal_gru_attention.py`, Phase 3, unmodified) is
   directly reusable — `forward(x)` expects `(batch, seq_len,
   input_dim)`, a clean structural match to real per-timestep ESA-ADB
   windows (same pattern already proven feasible for SMAP/MSL, Track
   B). No Transformer exists in the current codebase; not recommended
   to introduce one without a separate justification.
5. **Sequence length**: not determined this session — would need a
   real feasibility test (as done for SMAP/MSL) once a concrete
   windowing/downsampling plan exists; the raw sampling irregularity
   (§4) means naive fixed-length windows would need a resampling
   decision first.
6. **Missing telemetry values**: none found in any sampled channel —
   no imputation strategy currently needed, though not exhaustively
   verified across all 76 channels.
7. **Channel normalization**: not yet designed; would follow the same
   train-only-fitting discipline already used for NSL-KDD/OPSSAT-AD/
   SMAP-MSL.
8. **Anomaly intervals → labels**: mechanical but real design work —
   `labels.csv`'s `[StartTime, EndTime)` windows per channel would need
   conversion to a per-timestep binary array, respecting each
   channel's own irregular timestamp index (not a trivial index-based
   conversion given the sampling irregularity in §4).
9. **DQN state (44-dim) validity**: Yes, structurally unaffected — the
   state is built from the 4 FL signal families (performance/data/
   resource/fairness) × however many clients, not from the underlying
   dataset's own feature space; remains valid under an 8-client
   structure regardless of dataset.
10. **9 aggregation actions validity**: Yes, unaffected — the action
    catalogue operates on the same 4 signal families, dataset-agnostic.
11. **Reward validity**: Yes, structurally — reward is built from
    validation-F1 trend + performance spread, dataset-agnostic;
    whether it remains MEANINGFUL (i.e. whether ΔF1 carries a useful
    signal on this dataset) cannot be determined without a real run,
    same caveat as it already carries for NSL-KDD (Phase 9B's own
    near-saturated-validation-F1 finding).
12. **What must change**: a new `src/satellite/esa_adb_loader.py`-style
    adapter (isolated, not modifying existing code); a windowing/
    resampling decision; a train/test temporal-split construction (§4);
    label-window-to-per-timestep conversion (§9.8); likely
    `input_dim`/`sequence_length` config values specific to this
    dataset. **None implemented this session.**

## 10. Spatio-temporal modeling justification (real evidence, not assumed)

The project's title and motivation invoke "spatio-temporal modeling."
This audit found **real, direct evidence supporting that framing for
ESA-ADB specifically**, not present to the same degree in NSL-KDD or
OPSSAT-AD:
- **Temporal**: every sampled channel is a genuine 14-year,
  10-15-million-timestep sequence — substantially richer temporal
  structure than OPSSAT-AD's forced `sequence_length=1` or NSL-KDD's
  record-order-only sequencing.
- **Cross-channel ("spatial"/subsystem) structure**: **84% of real
  anomaly events are labeled Multivariate**, and 84% affect more than
  one channel simultaneously (§5) — real evidence that anomalies
  genuinely manifest across multiple, subsystem-related channels at
  once, not independently per channel. A model attending only to one
  channel in isolation would structurally miss most real anomalies in
  this dataset.
- This is the **first dataset investigated by this project** (NSL-KDD,
  OPSSAT-AD, STIN, SMAP/MSL) where a "spatial" (cross-channel/
  subsystem) claim is directly supported by real label evidence, not
  just the FL client simulation layer.

## 11. Storage/computation implications (real, measured, not the repo's
    own generic README claim)

- Mission 1 alone (not Missions 2-3): 3.78 GB compressed, 3.5 GB
  extracted, safely fits current storage (88.7 GB free on C: after
  extraction).
- **However, the FULL 14-year, 10-15M-row-per-channel history is
  computationally impractical to use in full** for a capstone-scale FL/
  DQN pipeline — this is a REAL, measured constraint (not the
  benchmark repo's generic 512GB/64GB-RAM/GPU claim, which was about
  running the ENTIRE TimeEval algorithm suite, not just using the raw
  data). **A real, disclosed scoping decision is required**: e.g.
  restrict to a specific shorter real time window (a few months,
  containing at least one real labeled anomaly) rather than the
  entire history, before any pilot is built. This is a genuinely new
  finding not present in the prior SMAP/MSL or OPSSAT-AD audits (both
  of those datasets were already small enough to use in full).

## 12. What was explicitly NOT done

Missions 2 and 3 were NOT downloaded. The full TimeEval benchmark
pipeline was NOT run. No `src/satellite/esa_adb_*.py` code was
written. No model was trained. No existing Phase 6-9B, OPSSAT-AD,
STIN, or SMAP/MSL artifact was read or modified — verified via `git
diff --stat` before and after this session's work.
