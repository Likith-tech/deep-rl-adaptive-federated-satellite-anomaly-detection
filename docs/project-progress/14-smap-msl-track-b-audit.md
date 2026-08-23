# SMAP/MSL — Track B Candidate: Acquisition, Integrity, Leakage, Preprocessing & Partition Design

**Status: AUDIT + DESIGN ONLY. No `src/satellite/*` loader/preprocessing
code was written. No model was trained (the one forward pass performed
for the temporal-feasibility test used `torch.no_grad()` and an
untrained, randomly-initialized model — no weight was ever updated).
No existing Phase 6-9B, OPSSAT-AD, STIN, or Improvement 01 artifact was
read or modified.** This document follows directly from Option A of
`docs/research/datasets/satellite_dataset_comparison.md`: NSL-KDD
remains the primary, frozen FL/DRL benchmark; SMAP/MSL is being
evaluated as a new, second, independent real-spacecraft Track B
validation dataset (parallel to, not a replacement for, OPSSAT-AD).

## 1. Dataset acquisition (real, reproducible)

`scripts/download_smap_msl.py` — downloads directly from the public,
unauthenticated endpoint verified this session
(`kaggle.com/api/v1/datasets/download/patrickfleith/nasa-anomaly-detection-dataset-smap-msl`,
302-redirects to a signed, public `storage.googleapis.com` URL, no
Kaggle API key required). Run for real: **86,004,011 bytes downloaded**,
extracted to `data/raw/smap_msl/` (git-ignored, same convention as
`data/raw/opssat_ad/`) — **82 train `.npy` files, 82 test `.npy`
files, `labeled_anomalies.csv`**. The bundled `2018-05-19_15.00.10/`
directory (the original authors' own pre-trained Keras models/results
from their 2018 paper run — 247 files) was deliberately NOT extracted;
it is not part of the dataset itself.

## 2. Integrity verification (`scripts/inspect_smap_msl.py`, real output in `results/reports/smap_msl_inspection.md`)

**A real defect was found in the official metadata file itself** (not
introduced by this project): `labeled_anomalies.csv` has 82 rows but
only **81 unique `chan_id` values** — `P-2` appears twice with
different, overlapping-but-non-identical `anomaly_sequences`
(`[[5350,6575]]` vs `[[5300,6420]]`). Consequence: one real channel
present on disk (`T-10`) has zero unambiguous metadata rows. Both
`P-2` and `T-10` were excluded from every statistic below rather than
guessed at, pending an explicit resolution decision (documented as
still open, §7).

**Clean statistics (80 unambiguous channels):**

| Spacecraft | Channels | Feature dim | Train rows | Test rows | Anomaly sequences |
|---|---:|---:|---:|---:|---:|
| SMAP | 53 | 25 | 135,183 | 427,617 | 67 |
| MSL | 27 | 55 | 58,317 | 73,729 | 36 |

Zero NaN values found in any channel (all 80). Every channel's actual
test-array row count matches `labeled_anomalies.csv`'s `num_values`
field exactly (0 mismatches, among the 80 unambiguous channels).

**Critical architectural finding**: SMAP channels are 25-dimensional,
MSL channels are 55-dimensional — **confirmed with zero exceptions**.
SMAP and MSL do NOT share a feature space. This has a direct
consequence for FL design (§6).

## 3. Leakage audit (real, measured, per-channel)

**59 of 80 channels** show at least one exact-duplicate feature row
shared between that channel's train and test arrays. This is NOT
necessarily a flawed author-provided split: many channels independently
show very high WITHIN-test-set consecutive-duplicate rates (e.g.
channel A-1: 83.8% of its own test rows are exact duplicates of the
immediately preceding timestep) — a real, natural telemetry
characteristic (values held constant between actual sensor updates),
not an artifact of how train/test were divided. The cross-split
overlap is consistent with this same low-entropy characteristic.

**Highest-overlap channel: `P-11` — 29.2% of its test rows exactly
match a train row** (1,032 of 3,535). `C-1` (23.3%) and `D-15` (15.8%)
are the next highest. **Recommendation**: exclude or specially review
`P-11` and `C-1` before either is used in any labeled evaluation;
treat the author-provided train/test split as authoritative throughout
(never re-derive a new split from a pooled train+test set, which would
make this natural duplication into genuine leakage).

## 4. Preprocessing design (design only — not implemented)

Following the same train-only-fitting discipline already enforced for
NSL-KDD (`src/preprocessing/pipeline.py`) and OPSSAT-AD
(`src/satellite/preprocessing.py`):

- **Scaling**: fit any scaler on the TRAIN split only, per channel or
  per mission (not globally across SMAP+MSL, given their disjoint
  feature spaces) — note the original Telemanom authors' own README
  states they scaled using TEST-set min/max, which this project would
  explicitly NOT reproduce, to avoid a real, disclosed leakage source
  the original benchmark itself does not avoid.
- **Missing values**: none found (§2) — no imputation strategy needed,
  but the pipeline should still assert this rather than silently
  assume it holds for a future re-download.
- **Label binarization**: `labeled_anomalies.csv`'s `anomaly_sequences`
  (start/end index pairs) would need conversion into a per-timestep
  binary `is_anomaly` array per channel — a straightforward, mechanical
  step, not yet implemented.
- **The P-2/T-10 ambiguity (§2) must be resolved BEFORE this step** —
  an explicit decision (exclude both, or take the union of P-2's two
  windows and manually source T-10's labels if possible) is required,
  not a default silently chosen by the loader.
- **Dimension mismatch (§2)**: SMAP (25-dim) and MSL (55-dim) cannot
  share one `BaselineMLP`/`TemporalGRUAttention` instance's `input_dim`
  without either (a) running them as two entirely separate experiments
  (recommended, §6), or (b) a disclosed, explicitly-flagged padding/
  alignment step — this project should not default to (b) without a
  separate, explicit approval, since it introduces meaningless
  zero-features for whichever mission has fewer dimensions.

## 5. Temporal-sequence feasibility test (real, executed, forward-pass only)

`scripts/test_smap_msl_temporal_feasibility.py` — built real sliding
windows from real downloaded test data and pushed them through the
**existing, unmodified** `TemporalGRUAttention` class (Phase 3),
`model.eval()` + `torch.no_grad()` throughout (no training, no weight
update):

```
A-1 (dim=25, seq_len=100): input (8,100,25) -> logits (8,1), attn (8,100) | 8,541 possible windows in the full file | OK
T-1 (dim=25, seq_len=250): input (8,250,25) -> logits (8,1), attn (8,250) | 8,363 possible windows in the full file | OK
M-1 (dim=55, seq_len=100): input (8,100,55) -> logits (8,1), attn (8,100) | 2,178 possible windows in the full file | OK
ALL FEASIBILITY CHECKS PASSED
```

**This directly closes the exact limitation Phase 9A's own audit
disclosed for OPSSAT-AD**: `dataset.csv` gave one fixed-width vector
per segment, forcing `sequence_length=1` and preventing the GRU/
attention mechanism from being genuinely exercised. SMAP/MSL's real
per-timestep structure supports genuine `sequence_length > 1` — proven
here with a real forward pass, not asserted from the file shape alone.

## 6. FL-client partition design

**Real subsystem/channel-prefix breakdown** (from `labeled_anomalies.csv`,
80 unambiguous channels):
- SMAP: `{A:9, B:1, D:12, E:13, F:3, G:6, P:4, R:1, S:1, T:3}` (10 distinct prefixes, 53 channels)
- MSL: `{C:2, D:3, F:4, M:7, P:4, S:1, T:6}` (7 distinct prefixes, 27 channels)

**Recommended design, given the §2/§4 dimension-mismatch finding**:
run SMAP and MSL as **two separate FL experiments**, never combined
into one shared-architecture FedAvg round (a real architectural
constraint, not a preference). For the primary experiment (SMAP, the
larger and cleaner of the two — 53 channels, 67 anomaly sequences, no
P-2/T-10 ambiguity issue since P-2 is a SMAP channel but simply
excluded), an **8-client, subsystem-based, non-IID partition** is
directly achievable by grouping SMAP's 10 real channel-prefixes into 8
clients (merging the three smallest — `B`, `R`, `S`, 1 channel each —
into one combined "misc" client to avoid a degenerate 1-channel
client): `{A, D, E, F, G, P, T, B+R+S}` = 8 clients. This is a real,
disclosed, evidence-based partition — not chosen to maximize any
metric, and not a random/Dirichlet split of a pool that has no natural
structure (unlike this project's own NSL-KDD partition, which
genuinely has no natural client identifier and must simulate one).

MSL (27 channels, 7 prefixes) would be a smaller, secondary
experiment — e.g. a 4-7 client subsystem partition — explicitly framed
as a second, independent validation run, not forced into the same
8-client structure as SMAP.

## 7. Open decisions (not resolved by this audit — require your explicit direction)

1. How to resolve the `P-2`/`T-10` metadata ambiguity (§2/§4) before
   any preprocessing code is written.
2. Whether to pursue SMAP alone first (recommended, §6) or attempt a
   combined SMAP+MSL design via padding/alignment (not recommended
   without separate approval).
3. Whether/how to treat the `P-11`/`C-1` high-overlap channels (§3) —
   exclude, or include with a documented caveat.

## 8. What was explicitly NOT done

No `src/satellite/smap_msl_loader.py`, no `preprocessing.py`, no
`configs/smap_msl.yaml`, no `FederatedClient`/`FederatedServer` wiring,
no training of any kind. No existing Phase 6-9B, OPSSAT-AD, STIN, or
Improvement 01 file was read or modified — verified via `git diff
--stat` before and after this session's work.
