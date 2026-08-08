# Temporal Results — GRU + Attention Anomaly Detector

## Dataset

NSL-KDD (terrestrial network intrusion dataset — see `docs/datasets/dataset_selection.md`). NSL-KDD has no genuine timestamp field; sequences here are built from the dataset's own ORDERED rows (see `docs/project-progress/04-phase-3-spatio-temporal.md` and `src/preprocessing/sequences.py` for the documented limitation).

## Model architecture

Input(121) -> Linear projection(64) -> ReLU -> GRU(hidden=64, layers=1) -> Temporal attention -> Dropout(0.3) -> Dense(1) -> logit

## Sequence construction

- Sequence length (selected): **16**
- Stride: 16 (non-overlapping windows)
- Sequence label strategy: ANY-anomaly (1 if any record in the window is anomalous, else 0)
- Test sequences (KDDTest+): 1409
- Train/validation sequences use an order-preserving, contiguous split (NOT the shuffled split used for the MLP baseline) — see `src/preprocessing/sequences.py` for why sequences require this.

## Sequence length selection (validation-only)

Candidates tried: [8, 16, 32]

| Sequence length | Train sequences | Validation sequences | Best val loss | Best epoch |
|---|---|---|---|---|
| 8 | 13384 | 2362 | 0.0087 | 6 |
| 16 **(selected)** | 6692 | 1181 | 0.0001 | 25 |
| 32 | 3346 | 590 | 0.0002 | 25 |

Selection criterion: **lowest validation loss**. KDDTest+ was not used for this selection.

## Training configuration

- Batch size: 128
- Learning rate: 0.001
- Optimizer: adam (Adam)
- Loss: bce_with_logits (BCEWithLogitsLoss)
- Max epochs: 25
- Early stopping patience: 4 epochs (on validation loss)
- Seed: 42

## Training (selected model)

- Epochs run: 25 (completed max epochs)
- Best epoch (by validation loss): 25
- Best validation loss: 0.0001
- Training time (selected candidate only): 29.9s

## Best validation performance (selected model)

| Metric | Value |
|---|---|
| accuracy | 1.0000 |
| precision | 1.0000 |
| recall | 1.0000 |
| f1 | 1.0000 |

## Final test performance (KDDTest+, evaluated once)

| Metric | Value |
|---|---|
| Accuracy | 1.0000 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1 | 1.0000 |
| ROC-AUC | not available |
| False Positive Rate | 0.0000 |

## Confusion matrix (KDDTest+)

```
                Predicted Normal   Predicted Anomaly
Actual Normal                  0                    0
Actual Anomaly                 0                 1409
```

See `results/plots/temporal/confusion_matrix.png` for the visual version.

## Comparison against MLP baseline (Phase 2)

Baseline numbers parsed directly from `results/reports/baseline_results.md` (commit `661ae74`) — not re-typed, so they cannot drift from that approved report.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---:|---:|---:|---:|---:|---:|
| MLP Baseline (Phase 2) | 0.7832 | 0.9278 | 0.6713 | 0.7790 | 0.8972 | 0.0690 |
| Temporal GRU+Attention (Phase 3) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | N/A (single class in test labels — see Limitations) | 0.0000 |

Raw F1 delta (temporal − baseline): **+0.2210**. **Do not read this as "temporal modeling won" without reading the Investigation section immediately below — this comparison is numerically real but the temporal task is degenerate, as explained there.**

## Investigation: why are the temporal metrics near-perfect?

This section exists because the raw numbers above (F1 = 1.0000) look suspiciously perfect, and the project rule is to investigate rather than accept flattering numbers at face value.

- Record-level anomaly rate in KDDTest+: **0.5692** (measured directly from the processed test data — matches `results/reports/dataset_quality.md`).
- Sequence-level anomaly rate at sequence_length=16 (ANY-anomaly rule): **1.0000**.

**Root cause:** the Phase 3 sequence-labeling rule ("sequence = anomaly if ANY record in the window is anomalous") interacts badly with NSL-KDD's near-uniform ~46-57% per-record anomaly rate. If per-record labels were independent draws at that rate, the chance that a window of 16 consecutive records contains *zero* anomalies is roughly (1 - 0.57)^16 ≈ 0.0000 — i.e. almost every window ends up labeled "anomaly" regardless of its actual content. This was confirmed directly (not just estimated): 100.0% of the real test sequences at this length carry the anomaly label. The same effect was measured on the training/validation sequences during the sequence-length experiment above (validation F1 was already ≈1.0 within a few epochs for sequence_length 16 and 32).

**What this means:** the near-perfect scores are NOT evidence that the GRU+attention model learned meaningful temporal patterns. With a test set that is ~100% one class, a model that always predicts "anomaly" would score identically on accuracy/precision/recall/F1 (and the confusion matrix's "Actual Normal" row is empty — 0 true negatives, 0 false positives — confirming there were essentially no negative examples left to get wrong). ROC-AUC is correctly reported as unavailable (not fabricated as 1.0 or any other value) precisely because scikit-learn — and this project's own `compute_metrics` — refuse to compute it when only one class is present in the true labels.

**This is a genuine, evidence-based limitation of the mandated first-implementation labeling strategy (Phase 3 instructions, "SEQUENCE LABEL = 1 if ANY record in the sequence is anomalous"), not a limitation of the GRU/attention architecture itself, and not a data leakage bug** (train/validation/test sequence boundaries were verified never to cross — see `tests/data/test_sequences.py`). The comparison table above should be read as "the ANY-anomaly sequence task was not a meaningful benchmark at this sequence length on this dataset" rather than "temporal modeling beats the baseline." A fair temporal-vs-baseline comparison needs a less degenerate sequence-labeling strategy (e.g. majority-vote across the window, or labeling by only the last record in the window) — left for future work rather than silently substituted here, since the instructions for this phase specified the ANY-anomaly rule explicitly for the first implementation.

**A further consequence worth flagging:** the sequence-length selection table above shows sequence_length=8 had a noticeably *higher* validation loss (0.0087) than 16 or 32 (0.0001-0.0002). Shorter sequences are mechanically less likely to trivially contain an anomaly, so they are the *least* degenerate of the three candidates — yet "lowest validation loss" as a selection criterion systematically favors the *more* degenerate, easier configuration. This is a second, independent piece of evidence pointing at the same root cause, not a coincidence.

## Attention visualization

See `results/plots/temporal/attention_weights.png` — real attention weights from the trained model on 5 real KDDTest+ sequences. Each bar chart shows how much weight the model assigned to each timestep within that sequence when forming its final decision. This indicates which positions contributed more strongly to the learned temporal representation — it is a diagnostic signal, not a complete or guaranteed explanation of the model's reasoning.

5 of the 5 plotted example sequences carry the anomaly label — a direct consequence of the near-100% sequence anomaly rate discussed above, not a cherry-picked selection (the first 5 test sequences were used, unfiltered). The weights themselves are close to uniform (~1/sequence_length each) after an initial low-weight warm-up over the first 1-2 timesteps, rather than sharply concentrated on a few positions — consistent with a model that did not need to learn a discriminative temporal pattern to solve this particular (degenerate) task.

## Notes

- KDDTest+ was used for this final evaluation only — never for training, sequence-length selection, or checkpoint selection.
- Sequences are built from the dataset's row order, not genuine timestamps — see Limitations in `docs/project-progress/04-phase-3-spatio-temporal.md`.
- No spatial/multi-satellite component exists yet — this is the temporal half only.