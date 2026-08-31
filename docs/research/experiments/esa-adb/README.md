# ESA-ADB Track C Candidate — Research Log

New, isolated research log, kept separate from
`docs/research/experiments/{phase-9b-drl,stin-dataset,dataset-selection,smap-msl}/`
(all frozen/unaffected by this work).

---

## Entry 1 — Branch creation, Mission 1 acquisition, and full structural/leakage/FL-feasibility audit

**Date:** 2026-08-15/16 (spanned two sessions due to a slow real
download). **Branch:** `project-development-6-esa-adb`, created from
`b4010c8` (after committing the prior SMAP/MSL audit to
`project-development-6-smap-msl` first — same established pattern as
every prior dataset-track transition this project has used, disclosed
each time).

**Task:** Begin ESA-ADB integration as a new experimental track
(Option A-derived: NSL-KDD remains the primary frozen benchmark).
Acquire, audit, assess FL feasibility, and design a pilot — no
training, no full benchmark pipeline, no Mission 2/3 download unless
proven necessary.

**Session 1 (branch setup + download start):** Verified git safety
(status, branch, HEAD, protected-artifact diff all clean), created the
branch, re-verified ESA-ADB's official Zenodo metadata fresh (record
12528696, 3 mission ZIPs, 11.61 GB total, MD5 checksums now captured),
re-verified the GitHub benchmark repo fresh (one correction to a prior
session's claim: Mission 3 does have an exploratory
`Mission3_timelines.ipynb` notebook, though still no
`mission3_experiments.py` benchmark script). Real disk space checked
(C: 97.3 GB free, E: 56.0 GB free, matching the user's stated
figures). Started `scripts/download_esa_adb.py --mission 1` (Mission 1
only, per the minimal-acquisition instruction) — the download proved
much slower than SMAP/MSL's, requiring it to run across the session
boundary in the background.

**Session 2 (continuation, per explicit "do not restart/repeat"
instruction):** Verified the existing background download was still
alive and progressing (not stalled, not duplicated) before taking any
action. Download completed: **3,776,246,054 bytes, size AND MD5
checksum verified** against Zenodo's own live-reported value. Ran
`scripts/inspect_esa_adb.py` to extract (3.5 GB extracted, 778 files)
and reveal the real archive structure — **which is NOT documented
anywhere in the benchmark repo's own README/docs at this level of
detail**: `channels.csv`, `labels.csv`, `anomaly_types.csv`,
`telecommands.csv`, plus one pickled-DataFrame `.zip` per channel.

**Key real findings (full detail in
`docs/research/datasets/esa_adb_dataset_audit.md`):**
- **76 total channels** (not the 774 "archive entries" — corrected
  after realizing that count included one `.zip` wrapper per channel),
  grouped into **4 real, author-labeled subsystems** (42/17/11/6
  channels) — a genuinely richer, more scientifically defensible FL
  client basis than anything found for SMAP/MSL or STIN.
- Sampled 5 channels directly (not all 76, to avoid excessive
  unpickling time/memory): **every one spans the full real 14-year
  range (2000-01-01 to 2013-12-31), 10-15 million rows each**, zero
  NaN, irregular/mixed real sampling intervals.
- **200 real, expert-annotated anomaly events**, 3,589 anomaly-channel-
  window rows; **168/200 (84%) are genuinely multivariate/cross-channel**
  — real, direct evidence supporting this project's "spatio-temporal"
  framing for the FIRST time across all four datasets audited so far
  (NSL-KDD, OPSSAT-AD, STIN, SMAP/MSL never had this kind of
  cross-channel anomaly evidence).
- A third anomaly category, `Communication Gap` (4 events) — a real
  operational-data-loss category distinct from `Anomaly`/`Rare Event`.
- No embedded train/test split in the raw data — the benchmark's own
  `84_months.test.csv` is a derived artifact of a preprocessing
  pipeline not run this session; any split is ours to construct.
- Leakage check: a candidate 2007-01-01 temporal-split boundary was
  tested directly against all 200 real anomaly windows — zero
  straddle it (reassuring, though not a claim this is the "correct"
  boundary, only that a safe one exists).
- **Real, measured computational-feasibility finding, distinct from
  the benchmark repo's own generic 512GB/64GB/GPU claim**: the FULL
  14-year, 10-15M-row-per-channel history is impractical to use in
  full for a capstone-scale pipeline — a real scoping decision (e.g. a
  short real time window) is required before any pilot, unlike
  SMAP/MSL or OPSSAT-AD which were already small enough to use whole.

**Decisions:** Recommended FL-client strategy = subsystem-then-group
partitioning (Option B in the audit doc) — 4 real subsystems, with the
two largest further split by their real `Group` field to reach a
genuine, disclosed 8-client structure, entirely from real metadata,
never a random row split.

**Problems encountered:** The Mission 1 download itself was the
primary practical obstacle — took long enough to span two sessions;
handled correctly by checking the existing background process's real
progress (not assuming, not duplicating) before continuing. No code
bugs this entry — both scripts (`download_esa_adb.py`,
`inspect_esa_adb.py`) ran successfully on their first real invocation.

**Files created:** `scripts/download_esa_adb.py`,
`scripts/inspect_esa_adb.py`,
`results/reports/esa_adb_mission1_inspection.md`,
`docs/research/datasets/esa_adb_dataset_audit.md`, this log file.
**No `src/` implementation code was written.**

**Data acquired:** `data/raw/esa_adb/` (git-ignored) — the compressed
`ESA-Mission1.zip` plus its extracted `Mission1/` contents. Missions 2
and 3 NOT downloaded.

**Storage:** C: 97.3 GB → 88.7 GB free after Mission 1
download+extraction (8.6 GB consumed). E: unchanged (56.0 GB, unused).

**Tests:** none new (no `src/` code exists for this track yet).

**Integrity verification:** `git diff --stat` against every protected
path (Phase 6-9B, Improvement 01, OPSSAT-AD, STIN audit, SMAP/MSL
audit) confirmed empty throughout both sessions.

**Next step:** await the user's review of the full audit (delivered as
the user-facing report) before any pilot implementation begins — the
audit's own §11 finding (full-history use is impractical) means the
pilot design itself must specify a concrete, real, short time-window
scoping decision, not just a client-count/architecture decision.

---

## Entry 2 — Real time-window selection analysis

**Date:** 2026-08-16. Continues directly from Entry 1's audit; no
branch change, no repeated steps (per explicit instruction).

**Task:** Select the scientifically strongest real time window for the
ESA-ADB pilot, using real timeline analysis of `labels.csv`/
`anomaly_types.csv`, not guesswork.

**Implementation:** `scripts/analyze_esa_adb_time_window.py` — builds
200 real anomaly EVENTS (grouping `labels.csv`'s 3,589 row-level
anomaly-channel mappings by their real `ID`, computing each event's
true min/max start/end across all its affected channels), a real
monthly timeline, and evaluates candidate 1/3/6/12-month windows by
sliding across all 168 real months and scoring by actual overlapping-
event count. For the winning window at each length, real per-channel
row density was measured by loading one real representative
Target=YES channel per subsystem (4 channels total, not all 58) and
counting actual rows inside the window boundaries — a real
measurement, not a rate-based estimate, with the resulting per-channel
sum explicitly labeled as a "rough extrapolation" when scaled to all
58 channels.

**Tests:** `tests/scripts/test_analyze_esa_adb_time_window.py` — 6
focused tests on the script's pure logic (`month_range`, `build_events`,
`window_stats`) using synthetic fixtures, no real data required.
**6/6 passed.**

**Problems encountered and fixed:** The first real run completed all
its expensive computation (200 events, 4 candidate windows, 16
real per-channel-window row-density measurements) but crashed at the
final report-write step with a `UnicodeEncodeError` (a `≈` character
not encodable under the default `cp1252` codec on this Windows
environment). Fixed by writing with explicit `encoding="utf-8"` and
replacing the special character with plain ASCII — re-run succeeded.

**Key real findings:**
- All four candidate windows (1/3/6/12-month) are nested around the
  SAME real period: the busiest 12-month window in the entire 14-year
  history is **2000-01-01 to 2000-12-31** (44 real anomaly events, 53
  of 58 Target=YES channels affected, 3 of 4 subsystems), containing
  the busiest 6-month (Jul-Dec 2000, 29 events), 3-month (Oct-Dec
  2000, 20 events), and 1-month (Dec 2000, 9 events) sub-windows —
  a clean, nested, internally consistent candidate set.
- **Zero anomaly events straddle any of these window boundaries**
  (verified directly, not assumed) — each candidate is a clean,
  self-contained slice of the real timeline.
- **`subsystem_1` has zero anomaly presence anywhere in the year
  2000** — verified directly. Checked further: across the ENTIRE
  14-year history, subsystem_1 appears in only 5 real anomaly events,
  and 4 of those 5 are massive, mission-wide "Global" multi-subsystem
  events that merely happen to include a couple of subsystem_1
  channels among dozens of others — only ONE event (`id_43`,
  2008-11-27 to 2008-11-30) is a genuinely subsystem_1-specific
  anomaly. This is a real, disclosed characteristic of the mission,
  not a bug in the analysis or the window choice.
- **A candidate in-window temporal train/test split at 2000-10-01**
  (9 months train, 3 months test — the test portion exactly matching
  the busiest 3-month candidate) has **zero anomaly events straddling
  it** (verified directly against all 44 real events in the year-2000
  window).
- Real measured row density (4 representative channels, one per
  subsystem) for the 12-month primary candidate: 2,553,456 rows summed
  across the 4 channels; rough extrapolation to all 58 Target=YES
  channels: ~37 million rows total — large, but tractable, and a small
  fraction of the full 14-year corpus's several-hundred-million-row
  scale.

**8-client mapping finalized** (real `Subsystem`/`Group` metadata,
computed from exact per-group channel counts already gathered in
Entry 1's follow-up analysis): `subsystem_5`(6ch, 1 client),
`subsystem_3`(7ch Target=YES via group 18, 1 client), `subsystem_1`
(3ch Target=YES via group 14, 1 client), `subsystem_6` split by its 10
real groups into 5 balanced clients (8/9/9/8/8 channels) — 8 clients
total, 58 channels, zero random splitting.

**Decision:** Primary window recommendation = **2000-01-01 to
2000-12-31**; backup = **2000-10-01 to 2000-12-31** (the 3-month
sub-window, for a faster/lighter pilot if the 12-month scale proves
too heavy in practice). Full reasoning in the user-facing report this
turn.

**Files created:** `scripts/analyze_esa_adb_time_window.py`,
`tests/scripts/{__init__.py,test_analyze_esa_adb_time_window.py}`,
`results/reports/esa_adb_time_window_analysis.md`. This log entry.
**No `src/` implementation code, no training.**

**Integrity verification:** `git diff --stat` against every protected
path confirmed empty before and after this entry's work.

**Next step:** await approval before building the actual ESA-ADB
pilot pipeline (loader/preprocessing/partition code) around the
recommended window.

---

## Entry 3 — Data preparation + FL data pipeline (loading, labeling,
preprocessing, sequencing, 8-client partition) — NO TRAINING

**Date:** 2026-08-23. Continues directly from Entry 2's approved scope;
no branch change. Implements the user's "APPROVED DATASET SCOPE" and
Steps 1-10 of the data-pipeline instruction. **This entry covers data
preparation only — no model was trained, no FL round was run, no DQN
code was invoked.**

**Step 1 (git safety):** Verified branch `project-development-6-esa-adb`,
HEAD `b4010c8`, working tree unchanged apart from the already-known
Entry 1/2 untracked files. `git diff --stat` against every protected
path (Phase 6-9B, Improvement 01, OPSSAT-AD, STIN, SMAP/MSL, the
existing `src/satellite/{loader,preprocessing}.py`, `src/data/`)
confirmed empty before and after this entry's work.

**Real per-channel sampling-rate investigation (new this entry):**
before any resampling/sequencing decision could be made, the real
native sampling interval was measured directly (not assumed) for one
representative channel per subsystem, *inside* the approved 2000
window: `channel_12` (subsystem_6) ~90s dominant, `channel_41`
(subsystem_5) ~30s dominant, `channel_61` (subsystem_1) ~900s/15min
dominant (by far the coarsest), `channel_70` (subsystem_3) ~30s
dominant. This >30x spread across real subsystems is why a single
resampling cadence has to be chosen deliberately rather than assumed.

**Design decisions made and documented (before writing any pipeline
code, never tuned against a result):**
- **Resampling: 300s (5min)**, last-value-per-bin + forward-fill,
  leading unfilled bins dropped. Coarser than every fast channel's
  native rate (genuine downsampling), finer than subsystem_1's ~15min
  native rate (forward-filled between real readings, never
  interpolated/fabricated). Full reasoning in
  `src/satellite/esa_adb/preprocessing.py`'s module docstring.
- **Label alignment**: resampled timestamps are labeled directly
  against the real `[StartTime, EndTime]` intervals (continuous time),
  not by propagating pre-resample per-reading labels. Disclosed
  edge-precision note: because `resample(...).last()` labels each bin
  by its LEFT edge, and the labeling check tests only that instant,
  the single bin in which an anomaly interval starts or ends can be
  missed if the interval doesn't cover the bin's left-edge instant.
  Given anomalies are dominated by multi-hour/multi-day `Subsequence`
  events (Entry 1), this affects a negligible fraction of total
  anomalous time and is documented, not hidden.
- **Model input shape: input_dim=1 (univariate per channel)**, the SAME
  architecture (`TemporalGRUAttention`) shared across all 8 clients — a
  client's local training pool is the UNION of its channels' univariate
  sequences. Rejected the alternative (combining a client's channels
  into one multivariate vector) because clients have different channel
  counts (3 to 9), which would give each client a different
  `input_dim` and break FedAvg's shared-architecture requirement — the
  same dimension-mismatch problem already flagged and rejected during
  the SMAP/MSL audit.
- **Sequence length 50, stride 25** (~4.2 hours of context at 5min
  resampling, 50% overlap).
- **Label strategy: "last"** (sequence label = label of its final
  timestep) — adopted directly from this project's own prior,
  evidence-based NSL-KDD decision (`src/preprocessing/sequences.py`),
  which found "any"-anomaly labeling degenerate on a dataset where
  anomalous records are common inside a window. The same risk applies
  here (many anomalies are `Subsequence`-length), so "any" was rejected
  for the same, already-documented reason without re-litigating it.

**Files created:**
- `src/satellite/esa_adb/{__init__,loader,labeling,clients,preprocessing,pipeline}.py`
  — new, isolated package. Does NOT modify `src/satellite/{loader,preprocessing}.py`
  (OPSSAT-AD) or `src/data/` (NSL-KDD).
- `scripts/run_esa_adb_pipeline.py` — end-to-end runner.
- `tests/satellite/esa_adb/{conftest,test_labeling,test_preprocessing,test_clients,test_pipeline_checks,test_real_data_pipeline}.py`
  — 39 new tests covering all 12 requested categories (loader
  determinism, timestamp ordering, channel selection, label
  correctness, normalization leakage, temporal split correctness,
  anomaly-window/split-boundary leakage, sequence shape, sequence-label
  correctness, 8-client partition correctness, no duplicate samples
  across clients, reproducibility).
- `data/processed/esa_adb/{manifest.json,client_metadata.json,preprocessing.json}`
  (git-ignored, matching the existing `data/processed/` convention).
- `results/models/esa_adb_preprocessing/{channel_scalers.joblib,channel_scalers_summary.json}`
  (git-ignored, matching the existing `results/models/` convention).
- `results/reports/esa_adb_pipeline_inspection.md` — lightweight
  inspection artifact (counts/distributions only, git-ignored).

**Real pipeline results (measured, not fabricated):**
- 58/58 Target=YES channels processed. Raw rows loaded across all 58
  channels in the approved window: **37,956,815** (matches Entry 2's
  "~37M rows" rough extrapolation closely). Resampled (labeled) records
  across all 58 channels: **6,113,661**.
- Split sizes (sequences, all channels pooled): train 162,922 (4,000
  anomalous, 2.46%), validation 19,952 (0 anomalous), test 61,364
  (1,413 anomalous, 2.30%).
- **Validation split has zero anomalous SEQUENCES despite 8 real
  anomalous resampled RECORDS existing in that split** — investigated,
  not silently accepted: this is a genuine interaction between the
  real, sparse anomaly presence in Sep 2000 (only 4 real events fully
  inside that month) and the "last"-timestep label rule's sensitivity
  to exactly which stride-aligned window boundary a short anomaly
  happens to land on. Confirmed via direct record-level inspection
  (8/501,120 validation records are anomalous), not a pipeline bug.
  Disclosed here rather than adjusted, per the "worse result is still a
  valid result" rule — no design parameter was changed to fix this.
- **8-client partition**: all 58 channels assigned, zero duplication,
  zero missing, verified programmatically (`clients.validate_partition`)
  in addition to the manual metadata audit in Entry 2. `client_3`
  (subsystem_1, channels 61-63) has **zero anomalies in every split**
  (train/validation/test) — a real, disclosed consequence of Entry 2's
  finding that subsystem_1 has almost no anomaly presence anywhere in
  the mission's history; not rebalanced or fabricated.
- **All 5 automated leakage/integrity checks passed** on the real data:
  no anomaly interval straddles the train/validation or
  validation/test boundary, no channel duplication across clients,
  split boundaries strictly ordered/disjoint, every channel's fitted
  scaler is reproducible from its own train split only (re-fit and
  compared), no duplicate sequences possible across clients (channels
  are disjoint by construction).

**Problems encountered and fixed (both caught by the test suite before
being reported, not left unresolved):**
1. `labeling.label_timestamps` initially called `.to_numpy()` on a
   pandas boolean array that was already a plain `numpy.ndarray`
   (`AttributeError`) — fixed by using `np.asarray(mask)`.
2. A test (`test_resample_channel_labels_directly_against_intervals`)
   caught a genuine subtlety: `label_timestamps` checks a resampled
   bin's LEFT-edge instant against each interval, not the bin's full
   5-minute span — documented explicitly in the module docstring (see
   "edge-precision note" above) rather than silently left implicit.
3. A test (`test_fit_channel_scaler_flags_constant_channel`) caught
   that `ChannelScaler.is_constant` was being derived from
   `sklearn.StandardScaler.scale_`, which sklearn internally
   substitutes to `1.0` for zero-variance input specifically to avoid a
   division by zero — meaning constant channels could never actually be
   detected that way. Fixed by computing `is_constant` from the raw
   train-value standard deviation directly, before fitting the scaler.
   (No channel in the real 2000-window data is actually constant, so
   this had no effect on the manifest's real numbers — but the
   detection logic itself was silently broken until the test caught it.)
4. Two client-partition tests initially asserted the wrong failure
   mode (mutated the *approved* list instead of the *client
   assignment*, so "missing channel" and "extra channel" scenarios were
   swapped) — fixed by mutating `CLIENT_CHANNELS` itself (restored via
   `try/finally`) while keeping the approved list fixed at its correct,
   real 58-channel value.

**Tests:** 39/39 new ESA-ADB tests passed. Full repository suite:
**341/341 passed** (`python -m pytest tests/ -q`). No historical test
was modified to force a pass.

**Integrity verification:** `git diff --stat` against every protected
path confirmed empty both before and after this entry's work (see
Step 1 above and the final report for the exact command run).

**Storage:** `data/processed/esa_adb/` + `results/models/esa_adb_preprocessing/`
+ `results/reports/esa_adb_pipeline_inspection.md` together total
**~52KB** — no raw data copied, no large plots. `data/raw/esa_adb/`
unchanged at ~7.1GB (no new download this entry). Free disk space:
**~88.5GB** (C:), essentially unchanged from Entry 1.

**What was explicitly NOT done:** no model trained, no FedAvg round,
no rule-based or DQN-adaptive FL run, no accuracy/F1/ROC-AUC computed
on this data, no hyperparameter search, no comparison to NSL-KDD or
OPSSAT-AD results. Sequence-length/stride/label-strategy/resampling
choices were fixed BEFORE any such metric could have been computed.

**Next step:** await review of this pipeline phase. Per explicit
instruction, do NOT begin the first ESA-ADB FL experiment (FedAvg
baseline or otherwise) until approved.

---

## Entry 4 — Validation-split redesign: Sep-only validation had zero
anomaly sequences; approved and implemented "Option B" (Jul-Sep)

**Date:** 2026-08-23. Continues directly from Entry 3; no branch
change. **Two sessions of work folded into one entry**: (1) discovery
and candidate analysis of the validation-split defect, (2) the user's
approval of Option B and its implementation. No model was trained in
either part.

**Problem discovered:** Entry 3's split (train Jan-Aug, validation Sep,
test Oct-Dec) produced **zero anomaly sequences in validation**, even
though 8 real anomalous resampled records existed in that month.
Investigated, not silently accepted: September 2000 contains 5 real
anomaly events (`anomaly_types.csv`), every one `Length=Point` (zero
duration) and single-channel (subsystem_3 only). Because
`resample(...).last()` labels each 5-min bin by its LEFT-edge instant,
and the "last"-timestep sequence-labeling rule only fires when a
stride-aligned window's *final* timestep lands exactly on an anomalous
instant, none of September's isolated point events happened to land on
such a boundary — a genuine, disclosed interaction between real event
sparsity/shape and the sequencing rule, not a pipeline bug.

**Candidate analysis** (`scripts/analyze_esa_adb_validation_candidates.py`,
`results/reports/esa_adb_validation_split_candidates.md`): computed
REAL sequence-level counts (not event-metadata estimates) for 4
candidates, all sharing the same fixed test period (2000-10-01 to
2000-12-31), varying only the train/validation boundary:

| Candidate | Validation period | Val sequences | Val anomalies | Clients w/ val anomaly |
|---|---|---:|---:|---:|
| A | Aug-Sep | 40,658 | 9 | 1/8 |
| B | Jul-Sep | 61,364 | 104 | 4/8 |
| C (Entry 3's original) | Sep only | 19,952 | 0 | 0/8 |
| D | Jun-Sep | 81,432 | 136 | 5/8 |

All 4 candidates passed the anomaly-interval/split-boundary leakage
check (zero straddling events, verified directly against all 44
real in-window events per candidate).

**Recommendation delivered:** Option B — real multi-channel events
(id_22, id_23 in July; id_36 in August) give 4/8 clients genuine
validation signal from a validation set the same size as test (61,364),
at a smaller training-data cost than Option D (which only adds 1 more
client and 32 more anomalies while sacrificing two additional real
training months). Recommended keeping validation F1 (computed on the
aggregated global model against the pooled validation set, matching
Phase 9B's existing `drl_round_provider.py` architecture exactly — no
reward-code change needed), while flagging that ~0.17% positive
prevalence will make F1 noisier round-to-round than Phase 9B's
near-balanced NSL-KDD signal; recall alone was rejected (ignores false
positives); a PR-AUC-based signal was noted as a fallback if F1 proves
too noisy once a pilot actually runs — not adopted now.

**User approved Option B explicitly.** Implemented:

- `src/satellite/esa_adb/loader.py`: `TRAIN_END` changed
  2000-08-31→**2000-06-30**, `VALIDATION_START` changed
  2000-09-01→**2000-07-01**. `VALIDATION_END` (2000-09-30) and both
  `TEST_START`/`TEST_END` (2000-10-01/2000-12-31) **unchanged** — test
  was never touched, per instruction. Module-level docstring updated
  to record both the original Sep-only choice and the reason it was
  replaced (historical reasoning preserved, not deleted).
- `tests/satellite/esa_adb/test_pipeline_checks.py`: one test's
  synthetic interval was tuned to straddle the OLD 2000-09-01 boundary
  and would have silently passed for the wrong reason under the new
  boundary — fixed to straddle the new 2000-07-01 boundary instead, and
  a new test added covering the (unchanged) validation/test boundary
  straddle case specifically, so both moved and fixed boundaries have
  explicit coverage.
- `clients.py`, `pipeline.py`, `preprocessing.py`: **unmodified** — all
  split-boundary consumers read `loader.TRAIN_START` etc. dynamically,
  so no other code changes were required.

**Real results after re-running the pipeline** (`scripts/run_esa_adb_pipeline.py`):
train 121,510 seq (3,888 anomaly, 3.20%), validation 61,364 seq (104
anomaly, 0.17%), test 61,364 seq (1,413 anomaly, 2.30%) — **exact match**
to the candidate analysis, confirming the standalone analysis script and
the production pipeline agree (both call the same tested functions).
Per-client validation anomalies: client_1=30, client_6=9, client_7=30,
client_8=35, all others (including client_3)=0 — matches the candidate
analysis exactly. **Client_3 (subsystem_1) remains at zero anomalies in
every split**, preserved as a real characteristic, not rebalanced.

**Leakage/integrity checks:** all 5 automated checks pass under the new
split (no anomaly interval straddles either boundary, no channel
duplication, boundaries ordered/disjoint, scalers re-derivable from
train only, no cross-client duplicate sequences). 8-client channel
mapping confirmed byte-for-byte unchanged (`clients.py` untouched).

**Tests:** 47/47 ESA-ADB tests pass (46 prior + 1 new boundary test).
Full repository suite: **349/349 passed**. No historical test weakened.

**Files created/modified this entry:**
`scripts/analyze_esa_adb_validation_candidates.py`,
`results/reports/esa_adb_validation_split_candidates.md` (analysis,
git-ignored), `src/satellite/esa_adb/loader.py` (split constants),
`tests/satellite/esa_adb/test_pipeline_checks.py` (one test fixed, one
added), this log entry. `data/processed/esa_adb/{manifest,
client_metadata,preprocessing}.json` and
`results/reports/esa_adb_pipeline_inspection.md` /
`results/reports/esa_adb/*.json` regenerated (git-ignored, not
committed).

**Integrity verification:** `git diff --stat` against every protected
path, plus `src/satellite/esa_adb/clients.py` specifically, confirmed
empty before and after this entry's work.

**Next step:** await review of the updated split before any ESA-ADB
model training. Still no MLP/GRU/Transformer/FedAvg/adaptive-FL/DQN run
this entry.

---

## Entry 5 — Experiment 1 (centralized single-channel baseline) and
Experiment 1A (centralized multivariate baseline)

**Date:** 2026-08-23. Continues directly from Entry 4 (Option B split,
approved). Two real, measured centralized (non-federated) baselines —
no FedAvg, no DQN.

**Experiment 1 (`experiments/esa_adb/baseline_01/`):**
`TemporalGRUAttention(input_dim=1)`, all 58 channels pooled as
independent univariate sequences, `BCEWithLogitsLoss(pos_weight=30.25,
train-only)`, seed=42/Adam lr=0.001/batch=128/epochs=25/patience=4
(Phase 3 defaults, unmodified). **Result: a failed baseline** — test
ROC-AUC 0.4734 (at/below random), PR-AUC 0.0234, F1 0.0482, FPR 0.8484
(the model flags ~85% of normal sequences as anomalous). All sanity/
leakage checks passed — this is a real, disclosed model failure, not a
data-pipeline defect. New isolated code:
`src/training/esa_adb_trainer.py`, `scripts/train_esa_adb_baseline.py`
(neither modifies `src/training/{baseline_trainer,temporal_trainer,
opssat_trainer}.py`).

**Experiment 1A (`experiments/esa_adb/baseline_02_multivariate/`):**
Investigated whether the single-channel input formulation was the
cause. Before building the multivariate tensor, verified real
per-channel timestamp alignment (not assumed) — found 55/58 channels
resample to 105,408 rows from 2000-01-01 00:00:00, but the 3
subsystem_1 channels (61/62/63, client_3's channels) resample to
105,407 rows, missing exactly the leading 00:00:00 bin. Fixed via an
inner join on the real timestamp intersection (drops 1 shared row,
zero fabrication) — new isolated module `src/satellite/esa_adb/
multivariate.py` (does not modify loader/labeling/preprocessing/
clients/pipeline.py). Joint per-timestep label = OR across all 58
channels' own official labels (a direct, disclosed extension of the
already-approved rule, not a new invention); sequence label reuses the
same "last" strategy. Same `TemporalGRUAttention` class, only
`input_dim` changed 1→58; same pos_weight philosophy, same
hyperparameters, same split. **Result: dramatic improvement** — test
ROC-AUC 0.4734→**0.9085**, PR-AUC 0.0234→**0.7514**, F1
0.0482→**0.4603**, FPR 0.8484→**0.0000**, precision 1.0000 (zero false
positives), recall 0.2989. Confirms the single-channel representation,
not the class weighting or optimization setup, was the primary cause
of Experiment 1's failure — consistent with Entry 1's audit finding
that 84% of real ESA-ADB anomalies are multivariate/cross-channel
events. Since test ROC-AUC is well above 0.5, Experiment 1B (only
triggered if 1A failed) was correctly not run.

**A caught-and-fixed bug in the verification code itself:** the first
run of `scripts/train_esa_adb_baseline_multivariate.py` correctly
STOPPED before training — its own sanity check expected ~121,510 train
sequences (Experiment 1's *pooled-across-58-channels* total) but the
real multivariate pipeline produces one sequence per aligned
cross-channel window, i.e. 121,510/58 = 2,095.0 exactly. The pipeline
was correct; the check's expectation was wrong. Fixed and re-verified
before re-running — preserved here as an example of a sanity check
doing its job (stopping on a real discrepancy) even though the root
cause was in the check, not the pipeline.

**Tests:** 10 new (`tests/satellite/esa_adb/test_multivariate.py`).
Full ESA-ADB suite: 67/67 (57 prior + 10 new). Full repository suite:
359/359.

**Integrity verification:** `git diff --stat` against every protected
path (including the generic `src/models/`, `src/evaluation/`,
`src/training/{baseline_trainer,temporal_trainer,opssat_trainer}.py`)
confirmed empty. `baseline_01/` untouched by the `baseline_02_
multivariate/` run.

**Next step (at the time):** await review before any FL work — the
real client structure (3-9 channels per client, all disjoint) means
Experiment 1A's exact architecture cannot be handed to any single
client unmodified, which became the subject of Entry 6.

---

## Entry 6 — Federated architecture design analysis (no code, no training)

**Date:** 2026-08-23. Continues directly from Entry 5's finding that
Experiment 1A's centralized 58-channel architecture cannot be deployed
per-client as-is (no client owns all 58 channels).

**Task:** design/analysis only — compare four candidate architectures
for reconciling heterogeneous per-client channel ownership with
FedAvg's shared-parameter-shape requirement, evaluated against 10
criteria (FedAvg compatibility, ownership preservation, variable
channel-count handling, temporal/cross-channel modeling, communication/
computational cost, DQN compatibility, implementation complexity,
scientific defensibility) — never optimizing for accuracy alone, per
explicit instruction.

**Full analysis:** `docs/research/datasets/esa_adb_federated_architecture_design.md`.

**Key grounding fact (verified by reading the actual code, not
assumed):** `src/federated/fedavg.py`'s `federated_average`/
`weighted_average` require identical state_dict keys/shapes across all
client updates — directly rules out any architecture whose parameter
shapes vary by client (e.g. a client-local input-projection layer)
from using the existing aggregation code unmodified.

**Four options compared:** (A) global 58-D input with an explicit
availability mask (mask must be a separate channel, never inferred
from a bare zero — a zero is a valid post-normalization value for a
channel's OWNING client); (B) shared per-channel encoder + mean/sum
pooling; (C) client-local input projection + shared temporal core
(requires personalized/partial-parameter FedAvg — a real, undone
modification to `fedavg.py`/`server.py` — and leaves an open question
for how DQN's single-global-model validation evaluation would work);
(D) shared per-channel encoder + attention pooling (a refinement of B,
not a distinct strategic option).

**Recommendation: Option A.** Reuses `TemporalGRUAttention` and
Experiment 1A's proven joint-multivariate mechanism directly (only the
input construction changes), requires zero modification to the
existing FedAvg or DQN code, and preserves real per-client ownership
via explicit masking rather than fabrication or borrowing. Option C is
named as the strongest architecture if the team later invests in
personalized-FL-style aggregation, but is not ready today. Experiment
1A's exact architecture is confirmed to remain a **centralized-only**
reference — not directly deployable per-client.

**Files created:** `docs/research/datasets/esa_adb_federated_architecture_design.md`,
this log entry. **No `src/` code, no tests, no training.**

**Integrity verification:** no protected path touched; `src/federated/
fedavg.py` and `server.py` were read-only inspected to ground the
FedAvg-compatibility analysis, never modified.

**Next step:** await review/approval of the Option A recommendation
before any implementation, FedAvg run, or DQN work begins.

---

## Entry 7 — Option A implementation + validation (no training)

**Date:** 2026-08-23. Continues directly from Entry 6's approved
recommendation. Implementation + validation ONLY — no FedAvg, no DQN,
no centralized training, no test evaluation.

**Real client ownership re-verified (not assumed):** direct read of
`clients.py` — client_1=6, client_2=7, client_3=3, client_4=8,
client_5=9, client_6=9, client_7=8, client_8=8 (sums to 58). This
differs from the per-client-label counts given in this turn's
instruction (which used the same `{3,6,7,8,8,8,9,9}` multiset attached
to different client labels) — the real `clients.py` mapping was used
throughout, per the explicit "verify, do not assume" instruction, and
the discrepancy is recorded rather than silently reconciled.

**New isolated module:** `src/satellite/esa_adb/federated_representation.py`
(does not modify loader/labeling/preprocessing/clients/pipeline/
multivariate.py). `GLOBAL_CHANNEL_ORDER` = `loader.target_yes_channels()`
(58 channels, ascending numeric index) — the single deterministic order
used identically by every client for both the X and M halves of the
116-dim input. Mask semantics implemented and tested exactly as
specified: owned channel → real normalized value + M=1; unowned →
fixed 0.0 placeholder + M=0, concatenated (never gated/multiplied) so a
real normalized-zero reading from an owning client is never confused
with an absent channel.

**Real, non-bug finding worth recording:** every client's per-split
sequence count and anomaly count came out IDENTICAL (train=2,095/
val=1,058/test=1,058, 347/15/184 anomalies) regardless of how many
channels that client owns (3 to 9). This is expected under Option A's
design — one sequence is built per aligned TIME window (driven by the
shared 105,407-row timeline and the same joint per-timestep label,
independent of channel count), not per channel as in Experiment 1's
univariate design. Verified this is not an alignment bug by direct
inspection before writing it up.

**Global validation representation — decision documented, not left
implicit:** investigated the "M=all-1s" (all 58 channels available)
hypothetical explicitly asked about in the instruction and **rejected**
it — that representation is never a scenario any real client or real
deployment of the federated model would encounter; it is exactly what
Experiment 1A's centralized baseline already measures, so using it as
the DQN's validation signal would silently re-test the centralized
ceiling rather than genuinely evaluating the federated model. **Decision:**
the pooled global validation set is the union of each client's own
masked (real-ownership) validation sequences (8,464 sequences, 120
anomaly, 1.42%) — both variants were computed and saved for comparison
(`experiments/esa_adb/option_a_federated_design/manifest.json`).

**Model shape / FedAvg compatibility — verified, not assumed:**
instantiated 8 independent `TemporalGRUAttention(input_dim=116)`
models (config: projection_dim=64, hidden_dim=64, num_gru_layers=1,
dropout=0.3, output_dim=1; 32,578 parameters each), pairwise-compared
all 28 pairs' state_dict keys/shapes/parameter counts — **all
identical, 0 violations**. Called the REAL, unmodified
`src.federated.fedavg.federated_average` on the 8 models' (untrained,
randomly-initialized) state_dicts with real per-client sample counts —
**succeeded, zero code changes to `src/federated/`** — confirming
Option A's core promise (no FedAvg modification required).

**Leakage checks:** no cross-client telemetry leakage (every unowned
channel slot is exactly 0.0/M=0 for every client, verified across all
8 clients' full train-split frames), scalers re-verified fit on train
split only (0 violations, same discipline as Experiment 1A).

**Files created:** `src/satellite/esa_adb/federated_representation.py`,
`scripts/build_esa_adb_federated_option_a.py`,
`tests/satellite/esa_adb/test_federated_representation.py` (18 tests,
all 14 requested categories covered),
`experiments/esa_adb/option_a_federated_design/{manifest.json,
per_client_sequence_summary.json}` (new, isolated directory — does not
touch `baseline_01/` or `baseline_02_multivariate/`, confirmed via
file mtimes unchanged).

**A caught-and-fixed bug in the script itself (not the design):** the
first run of `scripts/build_esa_adb_federated_option_a.py` hit a
`SyntaxError` from the same "mixed static+generated keys in one dict
literal" pattern seen once before in Entry 4 — fixed by building the
summary dict with an explicit loop instead of a comprehension. Caught
before any output was produced, not a data or design defect.

**Tests:** 18/18 new. Full ESA-ADB suite: 75/75 (57 prior + 18 new).
Full repository suite: **377/377 passed**.

**Integrity verification:** `git diff --stat` against every protected
path (Phase 6-9B, Improvement 01, NSL-KDD, OPSSAT-AD, generic model/
evaluation/training modules, and all existing ESA-ADB data-pipeline
modules) confirmed empty. `experiments/esa_adb/baseline_01/` and
`baseline_02_multivariate/` file modification times confirmed
unchanged from their original runs.

**Next step:** await explicit approval before any FedAvg training run
— this entry only implemented and validated the data representation
and confirmed shape/aggregation compatibility; no model was ever
trained under Option A.

---

## Entry 8 — Experiment 2: standard FedAvg baseline under Option A (REAL TRAINING)

**Date:** 2026-08-23. Continues directly from Entry 7's approved
implementation. Training approved and run this entry — 10 real FedAvg
rounds, no adaptive weighting, no DQN.

**Infrastructure reuse (verified before writing new code):** inspected
`src/federated/{client,server,trainer,protocol,fedavg}.py` and
`configs/federated.yaml` directly. `FederatedServer` and `ClientUpdate`/
`RoundRecord` are dataset-agnostic (operate on state_dicts/metadata
only) and reused UNCHANGED. `FederatedClient`/`run_federated_training`
hardcode `BaselineMLP` + parquet-partition loading and were NOT
reusable as-is — new isolated `src/training/esa_adb_federated.py`
mirrors their structure/conventions instead of copying them. Phase 6's
config defaults (local_epochs=1, batch_size=128, Adam lr=0.001,
seed=42, all 8 clients every round, best round selected by validation
loss) reused unmodified; the one deliberate addition is per-client
`pos_weight` (absent from Phase 6, required here per Experiments 1/1A's
already-established finding). The Phase 8/9B fairness-gap formula
(`max(client_f1) - min(client_f1)`, found in
`scripts/evaluate_adaptive_fl_results.py`/`run_phase9b_*.py`) was
reused exactly, not reinvented.

**Real configuration:** `TemporalGRUAttention(input_dim=116,
projection_dim=64, hidden_dim=64, num_gru_layers=1, dropout=0.3,
output_dim=1)`, 32,578 parameters. 10 rounds, all 8 real clients every
round, local_epochs=1, batch_size=128, Adam lr=0.001, seed=42,
`BCEWithLogitsLoss(pos_weight)` computed per client from that client's
own train labels only — all 8 clients got the SAME pos_weight (5.0375),
a direct, expected consequence of Entry 7's finding that every client's
train sequence/anomaly COUNT is identical under Option A's
one-sequence-per-aligned-window design (347 anomaly / 1,748 normal for
every client, regardless of channel count).

**Real result: FedAvg struggled badly, a genuine (not fabricated)
finding.** Global validation loss was already best at **round 1** and
got monotonically WORSE every round after (1.38 → 1.44 → 1.72 → 2.26 →
... → 2.73 by round 10); validation ROC-AUC stayed at/near random
(0.46-0.50) throughout, never improving. Frozen threshold (selected on
pooled validation only, F1-grid-search, threshold=0.52): **pooled test
ROC-AUC 0.4855, PR-AUC 0.2117, F1 0.1985, recall 0.3329, FPR 0.4253** —
essentially no better than random ranking, far below Experiment 1A's
centralized ceiling (ROC-AUC 0.9085) and not decisively better than
Experiment 1's failed single-channel baseline (ROC-AUC 0.4734) either.

**Severe per-client fairness disparity, real and disclosed:** test F1
ranged from **0.0000** (client_1/2/3/4 — the model predicts nothing
useful for these clients' own local test data) to **0.2963**
(client_6/8, both with recall=1.0000 — flagging everything as
anomalous). Test fairness gap (max-min F1) = **0.2963**, roughly 10x
larger than the round-level validation fairness gap seen during
training (~0.03) — a real, important discrepancy investigated below,
not glossed over.

**Investigated, not hidden: validation/test anomaly-prevalence
mismatch.** Pooled validation anomaly rate is 1.42% (120/8,464); pooled
test anomaly rate is **17.39%** (1,472/8,464) — a >12x difference. This
is not a bug: it is the same real joint-label/multivariate-windowing
property already visible in Experiment 1A (whose test set had the
identical 17.39% rate) — the OR-across-58-channels joint label makes
far more windows "anomalous" than per-channel labeling, and the real
Oct-Dec test period genuinely contains broader, more anomaly-dense
real events (id_59, id_200, id_35 — up to 35 affected channels each)
than the Jul-Sep validation period. A threshold selected on a
1.42%-prevalence validation set is poorly calibrated for a
17.39%-prevalence test set — a real, disclosed contributor to the weak
test performance and the fairness gap, not a new problem introduced by
FedAvg or Option A.

**Comparison with centralized baselines (real numbers, no claim that
FedAvg should beat them):**

| | ROC-AUC | PR-AUC | F1 | Recall | FPR |
|---|---:|---:|---:|---:|---:|
| Experiment 1 (single-channel, centralized) | 0.4734 | 0.0234 | 0.0482 | 0.9144 | 0.8484 |
| Experiment 1A (58-channel, centralized) | **0.9085** | **0.7514** | 0.4603 | 0.2989 | **0.0000** |
| Experiment 2 (FedAvg, Option A, this entry) | 0.4855 | 0.2117 | 0.1985 | 0.3329 | 0.4253 |

FedAvg did **not** preserve Experiment 1A's centralized multivariate
performance — a large, real gap (ROC-AUC 0.9085 → 0.4855). This is
reported as the honest answer to this experiment's own stated question
("can FedAvg preserve useful multivariate anomaly-detection performance
while respecting distributed channel ownership?") — the answer this
run gives is **not yet**, under plain, unweighted FedAvg.

**No implementation defect found.** All pre-training sanity/leakage
checks passed (validation representation matches the approved 8,464/120
design, scalers fit train-only, each client receives only its own
channels, identical global initialization, structural no-test-leakage).
The weak result is consistent with, and plausibly explained by,
genuine optimization/aggregation difficulty under severe non-IID
heterogeneous masking plus the prevalence mismatch above — not
retuned or "fixed" in this entry, per the explicit instruction to
report a clean baseline rather than tune aggressively.

**Files created:** `src/training/esa_adb_federated.py`,
`scripts/train_esa_adb_fedavg_baseline.py`,
`tests/federated/test_esa_adb_federated.py` (6 tests),
`experiments/esa_adb/fedavg_baseline_01/` (new, isolated — round
checkpoints, best model, round history, validation/test/per-client
metrics, threshold selection, fairness summary, comparison table,
config/manifest with dataset-manifest SHA-256).

**Tests:** 6/6 new. Full repository suite: **383/383 passed**.

**Integrity verification:** `git diff --stat` against every protected
path confirmed empty. `baseline_01/`, `baseline_02_multivariate/`, and
`option_a_federated_design/` file modification times confirmed
unchanged.

**Next step:** await review of this FedAvg baseline before any
adaptive-FL or DQN work. Given the weak, real result, the natural
follow-up questions (not decided here) are whether the val/test
prevalence mismatch, the pos_weight magnitude, or the round-1-then-
diverge instability pattern (echoing Experiment 1's own early-divergence
finding) is the dominant cause — worth investigating one variable at a
time before any adaptive-aggregation work is layered on top.