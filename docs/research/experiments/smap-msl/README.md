# SMAP/MSL Track B Candidate — Research Log

New, isolated research log, kept separate from
`docs/research/experiments/{phase-9b-drl,stin-dataset,dataset-selection}/`
(all frozen/unaffected by this work).

---

## Entry 1 — Branch creation, real dataset acquisition, integrity/leakage audit, temporal feasibility test, partition design

**Date:** 2026-08-15. **Branch:** `project-development-6-smap-msl`,
created from `e33c4f8` (after committing the prior STIN audit and
dataset-comparison research to `project-development-5-stin` first, so
this new branch starts from a genuine, isolated commit — same
discipline as the Phase 9B → STIN branch transition).

**Prompt/task:** Proceed with Option A + SMAP/MSL as a new isolated
Track B validation. Perform ONLY dataset acquisition, integrity
verification, train/test leakage audit, preprocessing design,
temporal-sequence feasibility test, and FL-client partition design —
no training, no modification of any existing Phase 6-9B or OPSSAT-AD
artifact.

**What was done (all real, verified — full detail in
`docs/project-progress/14-smap-msl-track-b-audit.md`):**
1. `scripts/download_smap_msl.py` written and run for real: 86,004,011
   bytes downloaded from the public, unauthenticated Kaggle-mirror
   endpoint (verified this session — no API key required), extracted
   to `data/raw/smap_msl/` (82 train + 82 test `.npy` files +
   `labeled_anomalies.csv`).
2. `scripts/inspect_smap_msl.py` written and run for real — discovered
   a genuine defect in the OFFICIAL metadata file itself (not
   introduced by this project): `labeled_anomalies.csv` has 82 rows
   but only 81 unique channel IDs (`P-2` duplicated with two
   different, non-identical anomaly-window annotations), leaving one
   real on-disk channel (`T-10`) with zero unambiguous metadata.
   Clean statistics over the remaining 80 channels: SMAP 53
   channels/25-dim/427,617 test rows/67 anomaly sequences; MSL 27
   channels/55-dim/73,729 test rows/36 anomaly sequences. Zero NaN
   values found anywhere.
3. Leakage audit (same script): 59/80 channels show some exact
   train/test row overlap; explained largely by real, natural
   telemetry flat-lining (channel A-1: 83.8% of its own test rows are
   consecutive exact duplicates) rather than a flawed split; the
   single worst channel (`P-11`, 29.2% overlap) flagged for exclusion/
   review.
4. `scripts/test_smap_msl_temporal_feasibility.py` written and run for
   real: genuine sequence-length>1 windows (100 and 250 timesteps)
   from real SMAP and MSL test data pushed through the EXISTING,
   UNMODIFIED `TemporalGRUAttention` class via `model.eval()` +
   `torch.no_grad()` — all shape checks passed. This is a forward-pass
   feasibility proof only; no weight was ever updated.
5. Preprocessing design and FL-client partition design documented
   (not implemented) — critically, discovered that SMAP (25-dim) and
   MSL (55-dim) do NOT share a feature space (zero exceptions across
   all channels), meaning they cannot be combined into one shared
   FedAvg architecture without an explicit alignment/padding decision;
   recommended running them as two separate experiments instead, with
   SMAP's 10 real channel-prefixes (subsystems) groupable into a
   genuine 8-client, subsystem-based, non-IID partition.

**Problems encountered:**
- The `labeled_anomalies.csv` duplicate-`chan_id`/orphaned-channel
  defect (§2 above) — a real upstream data-quality issue, not a bug in
  this project's own scripts. Handled by explicit detection and
  exclusion (not a silent overwrite of one `P-2` row by the other),
  with the decision of how to ultimately resolve it left open for the
  user.
- `scripts/inspect_smap_msl.py` initially crashed on an
  over-strict `assert channels == sorted(meta.keys())` once the above
  defect was hit — fixed by replacing the assertion with explicit
  detection/reporting logic before any further failures occurred.
- An f-string nested-quote syntax error in the same script, caught and
  fixed immediately (Python version compatibility issue, not a logic
  bug).

**Fixes applied:** both described above, applied before the script was
considered complete; the final run succeeded and produced the report
now in `results/reports/smap_msl_inspection.md`.

**Files created:** `scripts/download_smap_msl.py`,
`scripts/inspect_smap_msl.py`,
`scripts/test_smap_msl_temporal_feasibility.py`,
`results/reports/smap_msl_inspection.md`,
`docs/project-progress/14-smap-msl-track-b-audit.md`, this log file.
**No `src/` code was written** — no loader, no preprocessing pipeline,
no FL/DQN wiring, per the explicit "do not train the model yet /
implementation" boundary.

**Data acquired:** `data/raw/smap_msl/` — git-ignored (matches the
existing `data/raw/*` blanket rule), never committed, real downloaded
NASA/JPL telemetry via the public Kaggle mirror.

**Tests:** none new (no `src/` code exists yet for this track); the
project's existing full test suite was not expected to be affected and
was re-verified unaffected (see the user-facing report this turn).

**Integrity verification:** `git diff --stat` against every protected
path (Phase 6-9B, Improvement 01, OPSSAT-AD, STIN audit) confirmed
empty both before and after this entry's work.

**Open decisions (handed back to the user, per
`14-smap-msl-track-b-audit.md` §7):** how to resolve the `P-2`/`T-10`
metadata ambiguity; whether to proceed with SMAP alone first (this
entry's recommendation) or attempt a combined SMAP+MSL design; how to
treat the high-overlap channels (`P-11`, `C-1`).

**Next step:** await explicit approval before writing any
`src/satellite/smap_msl_*.py` implementation code or beginning any
training.