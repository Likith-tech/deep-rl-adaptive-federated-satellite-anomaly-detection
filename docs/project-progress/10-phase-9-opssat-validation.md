# Phase 9A — OPSSAT-AD Real Spacecraft Telemetry Validation

**Scope of this phase, stated plainly, and corrected after a follow-up audit
(see below):** this phase performs **segment-level anomaly classification,
using the Phase 3 temporal model architecture (GRU + attention) with
sequence_length=1**, against REAL spacecraft telemetry from ESA's OPS-SAT
CubeSat. Earlier drafts of this document described this as validating "our
temporal anomaly-detection methodology" — that phrasing overclaimed relative
to what a sequence length of 1 actually exercises (see the "Model" section
below) and is corrected here rather than left standing. The model's
classification head and general architecture are genuinely validated on
real spacecraft-derived features; genuine sequential/temporal modeling is
**not** — that would require the raw `segments.csv` telemetry and is left
for a possible future phase (see the audit note at the end of this
document). This phase does **not** implement DRL/DQN (that remains a later
phase), does **not** touch NSL-KDD or any Phase 1-8 result, and does **not**
claim cyberattack detection or real multi-satellite federated learning —
neither is supported by this dataset.

## Why real spacecraft data was added

Every phase up to now used NSL-KDD — real, but terrestrial, network
intrusion data — with a simulated satellite layer built on top of it
(Phase 4 onward). That's a reasonable and clearly-documented engineering
choice, but it means no part of this project had ever been tested against
data that actually came from a spacecraft. A dedicated research phase
(`docs/research/datasets/`) searched for and verified real spacecraft
datasets, and a pre-integration audit confirmed OPSSAT-AD as the strongest
available candidate. This phase is that integration.

## Dataset provenance

- **OPSSAT-AD** — ESA OPS-SAT anomaly detection benchmark.
- OPS-SAT was a real CubeSat, launched and operated by the European Space
  Agency, flying from 2019 until its mission ended in May 2024.
- Official source: https://zenodo.org/records/12588358. License: CC-BY-4.0
  per Zenodo's own record metadata (the bundled `LICENSE` file's text says
  MIT — both are permissive open licenses; both are recorded honestly in
  `results/reports/opssat_ad_inspection.md` rather than picking one
  silently).
- Downloaded via `scripts/download_opssat_ad.py`, a one-time, explicitly
  human-triggered script — never called automatically during training.

## Dataset structure (from direct inspection of the real downloaded file)

- 2,123 total telemetry segments: 1,689 nominal, 434 anomalous.
- Author-provided split: 1,594 training segments, 529 test segments.
- **Correction, recorded honestly:** the pre-integration audit (based on
  extracting numbers from the paper's text) reported 1,494 training
  segments. Direct inspection of the actual downloaded file shows the real
  number is **1,594** — the audit had a transcription error. The test
  count (529) was correct. This project trusts the real, downloaded data
  over the earlier paper-text extraction, and says so rather than quietly
  fixing the number without comment.
- 9 telemetry channels, identified in the real file by internal codes
  (e.g. `CADC0872`) rather than the human-readable names
  (`I_B_FB_MM_0`, `I_PD1_THETA`, etc.) the paper's text described — this
  project does not know the exact mapping between the two and does not
  guess at it.
- No missing values, no duplicate rows, no duplicate segment IDs.

## Preprocessing

Entirely separate from the NSL-KDD pipeline — no shared code, scaler, or
encoder. A new `StandardScaler` (19 numeric columns: statistical and
derivative features already extracted by the dataset's authors) and a new
`OneHotEncoder` (the 9-category `channel` column) were fit **only** on this
project's own training portion, never on validation or test data.

## Validation strategy

The dataset's authors defined the 1,594/529 train/test split; this project
never touches the 529 test segments until the one, final evaluation. A
validation set was carved out of the 1,594 training segments only
(stratified by label, seed 42, 15% held out — 1,354 train / 240
validation), exactly mirroring how Phase 1/3 carve validation out of
NSL-KDD's training split.

## Leakage prevention

- Scaler/encoder fit on training segments only.
- Validation carved from training only, stratified, seeded.
- Test segments (train=0) never used for training, validation, or any
  model/config selection decision — touched exactly once, in the final
  evaluation script.
- Every segment ID is unique (confirmed by direct inspection), so no
  segment appears in more than one split.

## Model

The exact same `TemporalGRUAttention` class Phase 3 uses for NSL-KDD —
completely unchanged, no fork, no modification. **Important honesty
note:** OPSSAT-AD's `dataset.csv` (the file this phase uses) gives ONE
fixed-width feature vector per segment, with no per-timestep sequence
inside it — the underlying raw telemetry is variable-length and
variable-rate, but by the time it reaches `dataset.csv` it has already
been reduced to 18 summary statistics per segment. So every sample here is
fed to the GRU as a "sequence" of length 1. This means the GRU's
recurrence and the temporal attention mechanism are not meaningfully
exercised in this run. What this phase actually validates is the model's
classification head and general architecture on real spacecraft-derived
features — not genuine sequential/temporal modeling of spacecraft data.
A stronger temporal test would need the raw `segments.csv` file, which
was deliberately out of scope for this phase (see the Phase 9A
pre-integration audit).

## Training

Adam optimizer, learning rate 0.001, BCEWithLogitsLoss, batch size 64,
seed 42, up to 50 epochs with early stopping (patience 8, on validation
loss). Training converged in 43 epochs (stopped early), best epoch 35,
best validation loss 0.1048, in 5.5 seconds — a tiny dataset compared to
NSL-KDD's, so training is correspondingly fast.

## Results

Evaluated once, on the untouched 529-segment test set:

| Metric | Value |
|---|---:|
| Accuracy | 0.9716 |
| Precision | 0.9623 |
| Recall | 0.9027 |
| F1 | 0.9315 |
| ROC-AUC | 0.9886 |
| False Positive Rate | 0.0096 |
| MCC | 0.9144 |
| AUCPR | 0.9759 |

Confusion matrix: 412 true nominal, 4 false positives, 11 false negatives,
102 true anomalous. Full detail, plots, and the Phase 3 side-by-side
(never-merged) comparison are in `results/reports/opssat_ad_results.md`.

## Limitations

- Single spacecraft — no multi-satellite claim is supportable.
- No cyberattack labels — every anomaly here is an operational/physical
  telemetry fault, not malicious activity.
- Sequence length of 1 — this run does not genuinely test temporal
  modeling, as explained above.
- Small dataset (1,594 training segments) relative to NSL-KDD's 107,077.
- No federated/multi-client structure — any FL use of this data would
  need artificial, explicitly-simulated partitioning, not performed here.

## What this proves

That the temporal model architecture built in Phase 3 can be trained and
evaluated on real, independently-verified ESA spacecraft telemetry, and
achieves a strong, plausible (not suspiciously perfect) result on genuine
operational-anomaly detection — a real second data point beyond NSL-KDD
for this project's core detection methodology.

## What this does NOT prove

- That this project can detect cyberattacks on spacecraft — no such
  labels exist in this dataset.
- That this project's Federated Learning / adaptive-FL machinery (Phases
  6-8) works on real multi-satellite data — only one real spacecraft's
  data was used, with no federated structure attempted here.
- That the temporal architecture's GRU/attention mechanism specifically
  helps on spacecraft data — the sequence-length-1 limitation means this
  run does not exercise genuine temporal recurrence.

## What comes next

DRL-based adaptive Federated Learning remains a separate, later phase —
not started here, and not implied by this phase's results. If a genuine
temporal test on raw OPS-SAT telemetry (`segments.csv`) or a second real
dataset (ESA-ADB, already identified as the documented secondary
candidate) becomes useful, that would be its own explicitly-scoped future
phase, not an automatic extension of this one.
