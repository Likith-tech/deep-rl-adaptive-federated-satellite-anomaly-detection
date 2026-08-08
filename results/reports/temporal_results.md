# Temporal Results — GRU + Attention Anomaly Detector

## Dataset

NSL-KDD (terrestrial network intrusion dataset — see `docs/datasets/dataset_selection.md`). NSL-KDD has no genuine timestamp field; sequences here are built from the dataset's own ORDERED rows (see `docs/project-progress/04-phase-3-spatio-temporal.md` and `src/preprocessing/sequences.py` for the documented limitation).

## Model architecture

Input(121) -> Linear projection(64) -> ReLU -> GRU(hidden=64, layers=1) -> Temporal attention -> Dropout(0.3) -> Dense(1) -> logit

## Sequence construction

- Sequence length (selected): **8**
- Stride: 8 (non-overlapping windows)
- Sequence label strategy: **last** — see `configs/temporal.yaml` and `results/reports/sequence_labeling_analysis.md` for the full strategy comparison and reasoning.
- Test sequences (KDDTest+): 2818 (1646 anomaly / 1172 normal, 58.41% anomaly)
- Train/validation sequences use an order-preserving, contiguous split (NOT the shuffled split used for the MLP baseline) — see `src/preprocessing/sequences.py` for why sequences require this.

## Initial sequence-labeling experiment (rejected)

The first Phase 3 attempt used the "any" strategy (sequence = anomaly if ANY record in the window is anomalous). That experiment was **rejected** after investigation — preserved here for research integrity rather than hidden.

- Strategy: `any`
- Candidates tried and their real measured validation loss (lower is not better here — see why below):

| Sequence length | Train sequences | Validation sequences | Best val loss |
|---|---|---|---|
| 8 | 13384 | 2362 | 0.0087 |
| 16 | 6692 | 1181 | 0.0001 |
| 32 | 3346 | 590 | 0.0002 |

- Selected (by lowest validation loss, as intended): sequence_length=16
- Measured KDDTest+ sequence class balance under "any" at sequence_length=16: **1409 anomaly / 0 normal** out of 1409 sequences (100.00% anomaly) — recomputed live just now from the real data, not a cached/hand-typed figure.

**Why it was rejected:** NSL-KDD's per-record anomaly rate is ~46-57%. "Anomaly if ANY of N records is anomalous" makes the chance of an all-normal window vanishingly small once N reaches ~16, so almost every sequence — train, validation, *and* test — ends up labeled anomaly. The resulting ~100% test score (see `experiments/temporal/initial_any_anomaly_temporal_results_ARCHIVE.md` for the full original report) was not evidence of temporal learning; a model that always predicted "anomaly" would have scored identically. Full investigation, including why the "lowest validation loss" selection rule made this worse (it favored the *more* degenerate configuration), is preserved in that archive.

**Corrected approach:** `src/preprocessing/sequences.py` was generalized to support four strategies (any / majority / last / ratio-with-threshold); `scripts/analyze_sequence_labeling.py` measured the real class distribution of each, across all three candidate sequence lengths and all three splits (`results/reports/sequence_labeling_analysis.md`); and **"last"** was selected on that evidence — see the Sequence construction section above and `configs/temporal.yaml` for the full reasoning (closely tracks the per-record base rate so class balance stays consistent and non-degenerate on every split, needs no arbitrary threshold, and makes the temporal task a strict superset of the baseline's task for a fair comparison).

## Sequence length selection (validation-only)

Candidates tried: [8, 16, 32]

| Sequence length | Train sequences | Validation sequences | Best val loss | Best epoch |
|---|---|---|---|---|
| 8 **(selected)** | 13384 | 2362 | 0.0290 | 11 |
| 16 | 6692 | 1181 | 0.0462 | 17 |
| 32 | 3346 | 590 | 0.0763 | 24 |

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

- Epochs run: 15 (stopped early)
- Best epoch (by validation loss): 11
- Best validation loss: 0.0290
- Training time (selected candidate only): 84.7s

## Best validation performance (selected model)

| Metric | Value |
|---|---|
| accuracy | 0.9907 |
| precision | 0.9880 |
| recall | 0.9917 |
| f1 | 0.9899 |

## Final test performance (KDDTest+, evaluated once)

| Metric | Value |
|---|---|
| Accuracy | 0.7583 |
| Precision | 0.9221 |
| Recall | 0.6403 |
| F1 | 0.7558 |
| ROC-AUC | 0.8923 |
| False Positive Rate | 0.0759 |

## Confusion matrix (KDDTest+)

```
                Predicted Normal   Predicted Anomaly
Actual Normal               1083                   89
Actual Anomaly               592                 1054
```

See `results/plots/temporal/confusion_matrix.png` for the visual version.

## Comparison against MLP baseline (Phase 2)

Baseline numbers parsed directly from `results/reports/baseline_results.md` (commit `661ae74`) — not re-typed, so they cannot drift from that approved report.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---:|---:|---:|---:|---:|---:|
| MLP Baseline (Phase 2) | 0.7832 | 0.9278 | 0.6713 | 0.7790 | 0.8972 | 0.0690 |
| Temporal GRU+Attention (Phase 3) | 0.7583 | 0.9221 | 0.6403 | 0.7558 | 0.8923 | 0.0759 |

F1 delta (temporal − baseline): **-0.0232** (did not improve over the baseline on this run). See the class balance check immediately below confirming this is a non-degenerate comparison this time, unlike the rejected initial experiment above.

## Class balance check (this experiment)

Run automatically on every evaluation, after the initial experiment's degenerate task went undetected until manual investigation — this section makes that check visible every time.

- Record-level anomaly rate in KDDTest+: **0.5692** (measured directly from the processed test data — matches `results/reports/dataset_quality.md`).
- Sequence-level anomaly rate at sequence_length=8 (`last` rule): **0.5841** (1646 anomaly / 1172 normal).
- Minority class share of test sequences: **41.59%**.

This is **not** degenerate: both classes are meaningfully represented in the test sequences, unlike the rejected initial experiment (which had a 0% minority class). The confusion matrix below has real entries in all four cells (or a defensible reason if not), and ROC-AUC is computable because both classes are present.

## Attention visualization

See `results/plots/temporal/attention_weights.png` — real attention weights from the trained model on 5 real KDDTest+ sequences. Each bar chart shows how much weight the model assigned to each timestep within that sequence when forming its final decision. This indicates which positions contributed more strongly to the learned temporal representation — it is a diagnostic signal, not a complete or guaranteed explanation of the model's reasoning.

1 of the 5 plotted example sequences carry the anomaly label (the first 5 test sequences, unfiltered — not cherry-picked).

**Observed pattern:** across these 5 examples, the model puts an average of **85.8%** of its attention weight on the LAST timestep alone (out of 8 timesteps, so uniform would be 12.5%). This is sensible given the `last` labeling strategy: the sequence label IS the last record's own label, so a model that leans heavily on the last record's features — with a little weight on the immediately preceding record(s) as context — is behaving exactly as the task defines "correct." It also suggests a plausible reading of the F1 gap vs. the baseline above: the temporal model appears to mostly rediscover what the baseline already gets directly from the current record's own 121 features, while the projection+GRU bottleneck the temporal model routes that information through may cost a small amount of accuracy rather than add to it. This is an interpretation consistent with the attention pattern, not a proven causal claim.

## Notes

- KDDTest+ was used for this final evaluation only — never for training, sequence-length selection, or checkpoint selection.
- Sequences are built from the dataset's row order, not genuine timestamps — see Limitations in `docs/project-progress/04-phase-3-spatio-temporal.md`.
- No spatial/multi-satellite component exists yet — this is the temporal half only.