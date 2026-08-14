# Phase 9A — OPSSAT-AD Real Spacecraft Telemetry Validation Results

## What this is, and what it is NOT

**REAL:** the telemetry (from ESA's OPS-SAT CubeSat, a real, flown, now-decommissioned mission) and the anomaly labels (expert-annotated operational anomalies) below are genuine.

**NOT PROVIDED by this dataset, and NOT claimed here:** cyberattack labels of any kind, and real multi-satellite federated data. The anomalies detected below are operational/physical telemetry anomalies — sensor/subsystem behavior outside its normal envelope — NOT malicious network traffic or intrusions. This is a completely separate track from the NSL-KDD-based cybersecurity/FL work in Phases 1-8, and the two are never merged into a single metric.

## 1. Dataset

OPSSAT-AD (ESA OPS-SAT CubeSat telemetry anomaly benchmark), CC-BY-4.0 (per Zenodo's structured license metadata; the bundled `LICENSE` file's own text says MIT — both permissive, both recorded in `results/reports/opssat_ad_inspection.md` rather than silently resolved). Source: https://zenodo.org/records/12588358. 1594 author-training segments, 529 author-test segments (author-defined split, used exactly as provided).

## 2. Model architecture

The SAME `TemporalGRUAttention` class Phase 3 uses on NSL-KDD (`src/models/temporal_gru_attention.py`), completely UNCHANGED — no fork, no modification. **Important honesty note:** OPSSAT-AD's `dataset.csv` gives one fixed-width feature vector per segment with no timestep dimension (confirmed by direct inspection — see `results/reports/opssat_ad_inspection.md`), so every sample is fed as a sequence of length 1. This means the GRU's recurrence and the temporal attention mechanism are NOT meaningfully exercised in this run — this validates the model's classification head and general architecture on real spacecraft-derived features, but is NOT a genuine test of sequential/temporal modeling on OPS-SAT data. A true temporal test would require the raw `segments.csv` file, explicitly out of scope for this phase (see the Phase 9A pre-integration audit).

- Input dimension: 28 (19 numeric + 9 one-hot channel features)
- Projection dim: 32, hidden dim: 32
- GRU layers: 1, dropout: 0.3

## 3. Preprocessing

Entirely independent of the NSL-KDD pipeline — no file, scaler, or encoder from `src/preprocessing/` or `results/models/preprocessing/` is read or modified. A new `StandardScaler` (19 numeric columns) and `OneHotEncoder` (the `channel` column, 9 categories) were fit **only on the training portion** (see section 4) and reused unchanged for validation and test.

## 4. Train/validation/test split

- Author-provided split (via `dataset.csv`'s own `train` column): 1594 training segments, 529 test segments — used exactly as given.
- This project's own validation carve-out (from the author's training segments ONLY, stratified by label, seed=42, validation_size=0.15): 1354 train / 240 validation.
- The 529 author-test segments were never touched until this evaluation.

## 5. Training configuration

- Optimizer: adam (Adam)
- Learning rate: 0.001
- Loss: bce_with_logits (BCEWithLogitsLoss)
- Batch size: 64
- Max epochs: 50
- Early stopping patience: 8 epochs (on validation loss)
- Seed: 42

## 6. Validation results (best epoch, checkpoint selection)

- Epochs run: 43 (stopped early)
- Best epoch (by validation loss): 35
- Best validation loss: 0.1048
- Training time: 5.5s

| Metric | Value |
|---|---:|
| val_accuracy | 0.9708 |
| val_precision | 1.0000 |
| val_recall | 0.8542 |
| val_f1 | 0.9213 |

See `results/plots/opssat_ad/training_curves.png`.

## 7. Final KDDTest+-style evaluation — OPSSAT-AD held-out test segments (evaluated ONCE)

| Metric | Value |
|---|---:|
| Accuracy | 0.9716 |
| Precision | 0.9623 |
| Recall | 0.9027 |
| F1 | 0.9315 |
| ROC-AUC | 0.9886 |
| False Positive Rate | 0.0096 |
| MCC (Matthews Correlation Coefficient) | 0.9144 |
| AUCPR (Area Under Precision-Recall Curve) | 0.9759 |

**These are real spacecraft OPERATIONAL anomaly detection results, NOT cyberattack detection results.**

## 8. Confusion matrix (OPSSAT-AD test segments)

```
                 Predicted Nominal   Predicted Anomalous
Actual Nominal                412                     4
Actual Anomalous               11                   102
```

See `results/plots/opssat_ad/confusion_matrix.png`.

## 9. Comparison with Phase 3 (dataset-labeled, never merged)

**These two rows are NOT a like-for-like benchmark.** They are shown side by side only to give a rough sense of scale, each explicitly labeled with its own dataset and task — never combined into one score.

| Track | Dataset | Task | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---:|---:|---:|---:|---:|
| Phase 3 | NSL-KDD (KDDTest+, terrestrial cyberattack detection) | Cyberattack detection | 0.7580 | 0.9220 | 0.6400 | 0.7558 | 0.8923 |
| Phase 9A | OPSSAT-AD (real ESA CubeSat telemetry) | Operational anomaly detection | 0.9716 | 0.9623 | 0.9027 | 0.9315 | 0.9886 |

The OPSSAT-AD numbers are higher, but this reflects a smaller, cleaner, less ambiguous benchmark (2,123 total segments vs. NSL-KDD's 148,517 records, and OPSSAT-AD's test set does not contain the deliberately novel/harder attack types NSL-KDD's KDDTest+ does) — not evidence that either task is intrinsically "easier" to solve in general.

## 10. Limitations

- **Single spacecraft.** No genuine multi-satellite claim is supportable from this dataset.
- **No cyberattack labels.** Every anomaly here is an operational/physical telemetry anomaly, not malicious activity.
- **Sequence length of 1.** As discussed in section 2, this run does not genuinely exercise temporal/sequential modeling — it validates the model's architecture and classification head on real spacecraft-derived tabular features.
- **Small dataset.** 1,594 author-training segments (1,354 after this project's own validation carve-out) is far smaller than NSL-KDD's 107,077 training records — results should be read with that sample size in mind.
- **No federated/multi-client structure.** Any FL use of this data would require artificial, explicitly-simulated partitioning — none was performed in this phase.
- Channel identity (`CADC08xx` codes) is opaque without ESA's internal documentation — this project does not know, and does not claim to know, the exact physical subsystem each code maps to beyond the paper's general description (magnetometer/photodiode).