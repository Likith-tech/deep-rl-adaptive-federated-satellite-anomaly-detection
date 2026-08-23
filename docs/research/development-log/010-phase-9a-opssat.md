# 010 — Phase 9A: OPSSAT-AD Real Spacecraft Telemetry Validation

**Reconstruction note:** Unlike Phases 0-8, this entry covers work done
across recent sessions in the SAME conversation history as the one that
created this development-log system, so more detail (including actual
audit findings) is directly available rather than reconstructed
secondhand. The dataset-search research step that preceded
implementation has a genuinely preserved prompt archive: see
`docs/research/claude-research/phase-9-real-spacecraft-dataset-search-prompt.md`
(dated 2026-08-13) and `...-response.md` — these are real artifacts,
not reconstructed. The implementation step's own original chat
prompt/response text is not separately preserved as a standalone file;
this entry describes what was actually asked and built based on the
committed code, docs, and this session's own audit work.

1. **Date/time:** Dataset-search research step dated 2026-08-13 (per
   the preserved research archive). Implementation commit
   `c58a8a2` dated **2026-08-14 22:12:28 +0530**.
2. **Objective:** Validate the project's temporal model architecture
   (Phase 3's `TemporalGRUAttention`, unmodified) against REAL
   spacecraft telemetry — the project's first data point beyond
   NSL-KDD — as a separate, never-merged Track B, without touching
   NSL-KDD or any Phase 1-8 result.
3. **Original task/prompt:** The dataset-search step's prompt is
   preserved verbatim in
   `docs/research/claude-research/phase-9-real-spacecraft-dataset-search-prompt.md`.
   The subsequent implementation and audit prompts (this session) are
   summarized accurately below but not preserved as separate archived
   files the way the dataset-search step was.
4. **What Claude/AI was asked to do:** (a) investigate real spacecraft
   datasets without taking implementation action until approved (dataset
   search step, prompt preserved); (b) implement OPSSAT-AD integration
   end-to-end (download, inspect, preprocess, train, evaluate,
   document, test) once a dataset was approved; (c) in a later,
   separate turn, perform a rigorous scientific/code audit of that
   implementation BEFORE committing, specifically investigating whether
   `sequence_length=1` could honestly be called "temporal anomaly
   detection"; (d) commit only after the audit's findings were
   incorporated and explicitly approved.
5. **Important Claude response/report information:** The dataset search
   concluded OPSSAT-AD (ESA OPS-SAT anomaly benchmark, Zenodo
   record 12588358, CC-BY-4.0 per Zenodo metadata) was the strongest
   real-spacecraft candidate — see `docs/research/datasets/dataset_decision.md`
   and `docs/research/datasets/dataset_comparison.md`. The later audit's
   full findings (dataset structure, leakage analysis, Option A vs.
   Option B comparison, final recommendation) are preserved verbatim in
   this session's own conversation and summarized in point 17 below;
   they are not separately archived as a standalone research file the
   way the dataset-search step is.
6. **What was actually implemented:** Zenodo-API-based downloader with
   live license verification (`scripts/download_opssat_ad.py`); dataset
   inspection (`scripts/inspect_opssat_ad.py` →
   `results/reports/opssat_ad_inspection.md`); `src/satellite/loader.py`
   (author train/test split respected via the `train` column);
   `src/satellite/preprocessing.py` (StandardScaler + OneHotEncoder fit
   on training data only); `configs/opssat_ad.yaml`
   (`sequence_length: 1`, extensively commented); `src/training/opssat_trainer.py`;
   `scripts/train_opssat_temporal.py` / `scripts/evaluate_opssat_temporal.py`;
   16 tests (`tests/satellite/`, `tests/training/test_opssat_trainer.py`);
   `docs/project-progress/10-phase-9-opssat-validation.md`.
7. **Files created:** 20 files (`git show --stat c58a8a2`) — see the
   commit for the exhaustive list; matches exactly the 20-file dry-run
   staged and committed after this session's audit.
8. **Files modified:** `.gitignore` (additive exceptions for the new
   `results/plots/opssat_ad/`, `results/reports/opssat_ad_*.md` paths
   only). No Phase 1-8 file touched — verified explicitly during the
   audit (`git status --porcelain` showed zero Phase 1-8 modifications).
9. **Dataset used:** OPSSAT-AD (ESA OPS-SAT anomaly detection
   benchmark, Zenodo record 12588358) — `dataset.csv` (2,123 rows × 23
   columns, one pre-extracted feature vector per segment) for model
   training/evaluation; `segments.csv` (303,493 rows × 8 columns, real
   per-timestep telemetry) investigated during the audit but not used
   for training.
10. **Dataset status:** REAL spacecraft telemetry from a real,
    ESA-operated CubeSat (2019-2024) — Track B, entirely separate from
    NSL-KDD's Track A. Operational/physical anomaly labels, NOT
    cyberattack labels. `dataset.csv` gives no timestep dimension; the
    raw per-timestep structure exists only in `segments.csv`.
11. **Model architecture:** The exact same `TemporalGRUAttention` class
    from Phase 3, completely unchanged, no fork.
12. **Hyperparameters:** Adam, lr 0.001, BCEWithLogitsLoss, batch size
    64, seed 42, up to 50 epochs with early stopping (patience 8 on
    validation loss); validation carved from the author's 1,594
    training segments only (stratified, 15%, seed 42) — the author's
    529 test segments never touched until final evaluation.
13. **Experiments:** Single training run; one final evaluation on the
    529 held-out author-test segments.
14. **Results (real, measured):** Training converged at epoch 43
    (early-stopped, best epoch 35, best val_loss 0.1048, best val_f1
    0.9213, 5.5s). Final KDDTest+-style evaluation (analogous one-time
    held-out eval, on OPSSAT-AD's own test segments): Accuracy 0.9716,
    Precision 0.9623, Recall 0.9027, F1 0.9315, ROC-AUC 0.9886, FPR
    0.0096, MCC 0.9144, AUCPR 0.9759. Confusion matrix
    `[[412,4],[11,102]]`.
15. **Problems/errors:** (a) A transcription error in the
    pre-integration audit reported 1,494 training segments where the
    real downloaded file has 1,594 (test count 529 was correct) —
    caught and corrected, documented as a correction rather than a
    silent fix. (b) Channel names in the paper's prose
    (`I_B_FB_MM_0`, etc.) don't match the actual file's opaque codes
    (`CADC0872`, etc.) — documented as an unreconciled discrepancy, not
    guessed at. (c) License discrepancy: Zenodo metadata says
    CC-BY-4.0, the bundled `LICENSE` file's text says MIT — both
    recorded honestly (both permissive, no blocking issue). (d) **The
    central audit finding**: `dataset.csv` has no timestep dimension,
    so every sample is fed to the GRU as a sequence of length 1 — the
    GRU's recurrence and attention are not meaningfully exercised.
16. **How problems were solved:** (a)-(c) documented as corrections/
    discrepancies rather than silently resolved. (d) was resolved by a
    dedicated audit (this session): investigated `segments.csv` in
    depth (verified 1:1 row-count correspondence with `dataset.csv`'s
    `len` column, confirmed `anomaly`/`train`/`channel` are
    segment-level constants, found the `label` string column is a
    degenerate artifact, found segments are chronologically contiguous
    with train/test membership genuinely INTERLEAVED at the segment
    level — meaning any cross-segment sequence window would risk
    leakage). Conclusion: within-segment genuine sequences ("Option B")
    are technically feasible and leakage-safe but require a
    substantial new pipeline (different feature representation,
    padding/masking for 8-1,040-length segments) — not a minimal
    change. **Decision: keep the existing implementation ("Option A"),
    correct the terminology.** `docs/project-progress/10-phase-9-opssat-validation.md`'s
    opening "Scope" paragraph was rewritten to state plainly: this
    phase performs "segment-level anomaly classification, using the
    Phase 3 GRU+attention architecture with sequence_length=1" — NOT
    genuine temporal/sequential modeling. This is the one and only code/doc
    change the audit made; no experiment was rerun, no result changed.
17. **Important decisions:** OPSSAT-AD selected over other real
    spacecraft candidates (see `docs/research/datasets/dataset_decision.md`,
    `dataset_comparison.md`, `rejected_datasets.md`); the author's own
    train/test split (the `train` column) used as-is, never re-split;
    Option A retained over Option B specifically because Option B,
    while feasible, is a distinct new pipeline rather than a minimal
    fix — deferred as a possible explicitly-scoped future phase rather
    than implemented reactively; terminology corrected rather than
    left standing, per an explicit "do not hide or minimize this issue"
    instruction.
18. **Rejected alternatives:** ESA-ADB and other candidates considered
    and rejected during the dataset-search step — see
    `docs/research/datasets/rejected_datasets.md`. Option B (genuine
    within-segment temporal sequences from `segments.csv`) was not
    rejected outright but deferred as a separately-scoped future phase
    rather than implemented now.
19. **Limitations:** Single spacecraft — no multi-satellite claim
    supportable; no cyberattack labels (operational/physical faults
    only); sequence_length=1 — not genuine temporal modeling (see
    above); small dataset (1,594 training segments) vs. NSL-KDD's
    107,077; no federated/multi-client structure.
20. **Tests:** `tests/satellite/__init__.py`, `conftest.py`,
    `test_loader.py`, `test_preprocessing.py`;
    `tests/training/test_opssat_trainer.py` — all synthetic-fixture
    based, never touch the real downloaded dataset.
21. **Test count:** 16 (verified twice this session by actually running
    `pytest tests/satellite/ tests/training/test_opssat_trainer.py` —
    16 passed both times). Cumulative repository test count at commit
    `c58a8a2` (`git grep "def test_" c58a8a2 -- '*.py'`): 184 (net +16
    from Phase 7-8's 168 — an exact match with the independently-run
    16-test count, cross-validating this log's methodology). Full
    `tests/` suite (excluding an unrelated pre-existing `backend/tests`
    FastAPI-import issue) verified this session: **191 passed**.
22. **Git branch:** `project-development-3`.
23. **Commit hash:** `c58a8a2215c4ecfda7620ed12742aa58f0950e5a`
24. **Push status:** Local `project-development-3` is 1 commit ahead of
    `origin/project-development-3` (which sits at `45c8d7d`, Phase
    7-8) — **this Phase 9A commit is NOT pushed to the remote**, as of
    this log's writing.
25. **What was intentionally NOT done:** No DRL/DQN; no NSL-KDD, Phase
    6 FedAvg, Phase 7 non-IID, or Phase 8 adaptive-FL modification; no
    genuine within-segment temporal sequence implementation (Option B,
    deferred); no ESA-ADB download; no merging of OPSSAT-AD metrics
    with NSL-KDD metrics; no new dataset branch.
26. **Lessons learned:** The audit demonstrated the value of
    investigating a suspicious architectural choice (`sequence_length=1`)
    even after an implementation was otherwise complete and its
    results were real — the fix required was documentation-only, not a
    rerun, because the underlying results were valid all along; only
    their description had overclaimed. This is the same "investigate,
    don't hide" discipline first established in Phase 3's rejected
    ANY-anomaly experiment.
27. **Next step:** Phase 9B — a DQN-based controller that learns
    federated aggregation weighting (replacing Phase 8's fixed rule),
    developed on an isolated branch (`project-development-4-drl`,
    created from this phase's commit `c58a8a2`), staying strictly
    within Track A (NSL-KDD) and not touching Track B (OPSSAT-AD).
