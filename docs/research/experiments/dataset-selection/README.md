# Real Satellite Dataset Selection — Research Log

New, isolated research log for the search for a real satellite/
spacecraft anomaly-detection dataset to potentially complement (not
replace) the existing NSL-KDD FL/DRL benchmark and the OPSSAT-AD
Track B validation. Kept separate from
`docs/research/experiments/phase-9b-drl/` (NSL-KDD DQN track, frozen)
and `docs/research/experiments/stin-dataset/` (STIN audit, already
rejected, unchanged).

---

## Entry 1 — Broad candidate search: ESA-ADB, NASA SMAP/MSL, and others

**Date:** 2026-08-15. **Branch:** `project-development-5-stin`
(continued — this task is research/documentation only, no code, so no
new branch was created; nothing here touches STIN's own audit).

**Task:** Determine whether a better real satellite anomaly-detection
dataset exists than what the project currently has (NSL-KDD for
FL/DRL, OPSSAT-AD for real-spacecraft validation, STIN already
rejected). Audit ESA-ADB and NASA SMAP/MSL specifically, search
broadly for other candidates, score them, and recommend a research
strategy — no code, no training.

**What was verified (all real, independently checked, not taken from
memory):**
- ESA-ADB: confirmed real via the Zenodo API (record 12528696, open
  access, 3 files, 11.6 GB total) and both GitHub repos the user asked
  about (`esa/anomaly-dataset` — confirmed real, not a wrong URL as
  initially suspected; `kplabs-pl/ESA-ADB` — confirmed real, MIT,
  119 stars). Channel counts (58 Mission1, 47 Mission2) counted
  directly from the actual `mission1_experiments.py`/
  `mission2_experiments.py` scripts' `target_channels` lists, not
  estimated. Discovered only 2 of 3 released missions are actually
  used in the benchmark (no `mission3_experiments.py` exists).
  Discovered a serious, disclosed computational-feasibility concern:
  the repo's own README recommends 512 GB disk, 64 GB RAM, a GPU, and
  allows up to 120 hours per algorithm run.
- NASA SMAP/MSL: confirmed real via the `khundman/telemanom` repo
  (Apache 2.0, not archived, links to the real 2018 KDD paper). Exact
  statistics computed by downloading and parsing the real
  `labeled_anomalies.csv` (82 rows): SMAP 55 channels/69 anomaly
  sequences/444,035 test timesteps; MSL 27 channels/36 anomaly
  sequences/73,729 test timesteps; 62 point-type and 43
  contextual-type anomalies across the combined 105 sequences.
- Broader search (JAXA/ISRO/CNES/NOAA): a real web search was
  performed; it returned zero public spacecraft-telemetry
  anomaly-detection datasets from these agencies — reported as a
  genuine absence, not skipped or assumed.

**Decisions:** Scored all candidates (STIN, ESA-ADB, SMAP/MSL,
existing OPSSAT-AD) across 12 criteria with evidence-tied rationale
for every score (`docs/research/datasets/satellite_dataset_comparison.md`
§6) — SMAP/MSL scored highest (54/60), primarily because ESA-ADB's
real computational-feasibility red flag (§ above) scored it a 1/5 on
that criterion despite otherwise strong data quality, and STIN remains
disqualified by its already-documented missing-normal-class problem.

**Recommendation reached (Option A, see the user-facing report this
turn for full reasoning):** keep NSL-KDD as the primary, frozen FL/DRL
benchmark (Phase 6-9B is complete, mature, and already produced a
valid, honestly-reported result); do not attempt to reproduce or beat
any candidate paper's percentage (Step 11's comparability analysis
found no candidate paper uses a comparable FL/DRL/client/metric
protocol to ours); if a new real-spacecraft track is pursued, SMAP/MSL
— not ESA-ADB — is the recommended choice specifically because it
would let the project's own existing `TemporalGRUAttention` model
(built in Phase 3) be genuinely exercised with sequence_length>1 on
real data for the first time, directly closing the exact limitation
Phase 9A's own audit already disclosed for OPSSAT-AD
(`sequence_length=1` forced by `dataset.csv`'s segment-level
structure).

**Problems encountered:** Kaggle's dataset-metadata API required
authentication not available in this session, so SMAP/MSL's exact
download size (as opposed to its exact timestep/channel/anomaly
counts, which WERE verified) is reported as an order-of-magnitude
estimate, explicitly labeled as such rather than asserted as a
verified fact. The OPSSAT-AD benchmark's own new Nature Scientific
Data paper is paywalled/auth-gated and could not be directly read —
Phase 9A's own already-verified numbers (from direct data inspection
in that phase) were reused instead of re-deriving from the paper.

**Files created:** `docs/research/datasets/satellite_dataset_comparison.md`,
this log file. No `src/` code, no config changes, no downloads beyond
small metadata/CSV files used for verification (never copied into the
repository).

**Tests:** none — no source code was touched this entry.

**Integrity check:** `git diff --stat` against every protected
Phase 6/7/8/9A/9B/Improvement-01/STIN-audit path confirmed empty
before and after this entry's work.

**Next step:** await the user's decision on Step 10's research
strategy (Option A/B/C/D/E) before any `src/satellite/smap_msl_*.py`
loader, preprocessing, or FL-client code is written for SMAP/MSL — or
before any full ESA-ADB download, if that path is chosen instead.