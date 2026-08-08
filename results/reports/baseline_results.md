# Baseline Results — MLP Anomaly Detector

## Dataset

NSL-KDD (terrestrial network intrusion dataset — see `docs/datasets/dataset_selection.md`). Processed via the Phase 1 pipeline: cleaned, labeled, split (train/validation/KDDTest+), one-hot encoded, standard-scaled.

## Model architecture

Feed-forward MLP: input(121) -> Dense(128) -> ReLU -> Dropout(0.3) -> Dense(64) -> ReLU -> Dropout(0.3) -> Dense(1) -> logit (sigmoid at inference)

## Hyperparameters

- Batch size: 256
- Learning rate: 0.001
- Optimizer: adam (Adam)
- Loss: bce_with_logits (BCEWithLogitsLoss)
- Max epochs: 30
- Early stopping patience: 5 epochs (on validation loss)
- Seed: 42

## Training

- Epochs run: 18 (stopped early)
- Best epoch (by validation loss): 13
- Best validation loss: 0.0112
- Training time: 48.0s

## Best validation performance (checkpoint selection)

| Metric | Value |
|---|---|
| accuracy | 0.9961 |
| precision | 0.9949 |
| recall | 0.9968 |
| f1 | 0.9959 |

## Final test performance (KDDTest+, evaluated once)

| Metric | Value |
|---|---|
| Accuracy | 0.7832 |
| Precision | 0.9278 |
| Recall | 0.6713 |
| F1 | 0.7790 |
| ROC-AUC | 0.8972 |
| False Positive Rate | 0.0690 |

## Confusion matrix (KDDTest+)

```
                Predicted Normal   Predicted Anomaly
Actual Normal               9041                  670
Actual Anomaly              4218                 8615
```

See `results/plots/baseline/confusion_matrix.png` for the visual version.

## Training curves

See `results/plots/baseline/training_curves.png` (training/validation loss per epoch, validation F1 per epoch).

## Notes

- KDDTest+ was used for this final evaluation only — never for training or checkpoint selection (validation loss on the held-out validation split was used for that).
- KDDTest+ contains attack types absent from training (by NSL-KDD's own design), which affects generalization and should be kept in mind when interpreting recall/precision here.