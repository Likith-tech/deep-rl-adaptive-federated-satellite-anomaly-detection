# Phase 2 — Baseline Anomaly Detection Model

## What was the goal?

Build our **first real AI model** — a simple one, on purpose — that can
look at a network connection record and decide: "Normal" or "Anomaly."
This gives us a trustworthy reference point ("baseline") to compare
every future, more advanced model against.

## Why do we need a baseline?

Later in this project we'll build much fancier models: one that
understands patterns over time and across multiple satellites
(spatio-temporal), one trained across many separate "satellite" clients
without sharing raw data (federated learning), and one where an AI
agent makes smart decisions about that federated process (deep
reinforcement learning / DRL). Without a simple baseline, we'd have no
way to know if those fancier techniques are actually helping — or just
adding complexity for no benefit. Every future result gets compared
back to this one.

## What model did we use?

A **Multi-Layer Perceptron (MLP)** — the simplest standard kind of
neural network. In plain terms: it takes in 121 numbers describing a
network connection, passes them through two hidden layers that learn to
combine and weigh that information, and outputs a single number between
0 and 1 — how confident it is that the connection is an attack.

```
121 input numbers (the connection's features)
        ↓
Dense layer (128 neurons) → ReLU → Dropout
        ↓
Dense layer (64 neurons) → ReLU → Dropout
        ↓
1 output number → Normal or Anomaly
```

- **Dense layer:** every input is connected to every neuron — the
  layer "mixes" all the information together.
- **ReLU:** a simple on/off-style rule that lets the network learn
  non-linear (curved, not just straight-line) patterns.
- **Dropout:** during training, randomly "turns off" some neurons so
  the model doesn't over-rely on any single one — helps prevent the
  model from just memorizing the training data.

## What input does it receive?

The 121 processed features from Phase 1 (scaled numeric traffic
statistics + one-hot encoded categories like protocol type). It does
**not** receive the raw attack name or the "difficulty" column — those
would be cheating (see Phase 1's leakage documentation).

## What output does it produce?

A single probability from 0 to 1. If it's 0.5 or higher, we call it
"Anomaly"; otherwise "Normal."

## How was it trained?

- **Training data:** 107,077 records (from KDDTrain+)
- **Validation data:** 18,896 records (held out from KDDTrain+, used to
  pick the best version of the model — never used to teach it directly)
- **Loss function:** Binary Cross-Entropy (a standard way to measure
  "how wrong" a probability prediction was)
- **Optimizer:** Adam (a standard, reliable training algorithm)
- **Batch size:** 256, **Learning rate:** 0.001, **Seed:** 42 (fixed,
  for reproducibility)
- **Max epochs:** 30, with **early stopping**: if validation loss
  doesn't improve for 5 epochs in a row, stop — no point training
  longer if it's not getting better.

**What actually happened:** training stopped early at epoch 18 (out of
a max of 30) because validation loss stopped improving. The best
version of the model was from **epoch 13**. Total training time: about
48 seconds on a normal CPU (no special GPU hardware needed for a model
this small).

## How was validation used?

After every training epoch, we checked the model against the
validation set (data it hadn't trained on) and saved a checkpoint
*only* when validation loss improved. This means the final saved model
is the best-performing version seen during training — not just
whatever the last epoch happened to produce.

## How was final testing done?

**Only once**, after training was completely finished and the best
checkpoint was already selected, we ran the model against KDDTest+
(22,544 records) — a completely separate set of data the model has
never seen in any way during training or tuning. This is the honest,
final measure of performance.

## Actual results (real, measured — not estimated)

**On validation data** (used to pick the best checkpoint):

| Metric | Value |
|---|---|
| Accuracy | 99.61% |
| Precision | 99.49% |
| Recall | 99.68% |
| F1 | 99.59% |

**On KDDTest+ — the true, final, honest result:**

| Metric | Value |
|---|---|
| Accuracy | 78.32% |
| Precision | 92.78% |
| Recall | 67.13% |
| F1 | 77.90% |
| ROC-AUC | 89.72% |
| False Positive Rate | 6.90% |

## Why is the test score so much lower than the validation score?

This is expected, not a mistake. As documented back in Phase 1,
KDDTest+ deliberately contains **attack types the model never saw
during training** — this is a well-known, intentional part of the
original NSL-KDD benchmark, designed to test whether a model can
generalize to genuinely new threats, not just memorize known ones. Our
baseline, being simple, catches the attacks it has seen patterns for
reasonably well (high precision — when it says "anomaly," it's usually
right) but misses more of the truly novel unseen attacks (lower
recall). This exact validation-vs-test gap is consistently reported in
published NSL-KDD research using similar models — it's a known,
documented property of this specific benchmark, not a red flag in our
implementation.

## What does the confusion matrix tell us?

```
                Predicted Normal   Predicted Anomaly
Actual Normal        9,041                 670
Actual Anomaly        4,218               8,615
```

- **9,041** normal connections correctly identified as normal
- **670** normal connections wrongly flagged as attacks (false alarms)
- **8,615** attacks correctly caught
- **4,218** attacks missed (called "normal" when they were actually attacks)

In plain terms: our baseline is fairly cautious about false alarms
(only 6.9% of normal traffic gets wrongly flagged) but misses a
meaningful chunk of attacks — mostly the unfamiliar ones. That's a
completely reasonable place for a *simple, first* model to land — and
exactly why later, smarter models exist: to catch more of those missed
attacks without raising the false-alarm rate.

See `results/plots/baseline/confusion_matrix.png` for the visual chart
and `results/plots/baseline/training_curves.png` for how training
progressed epoch by epoch.

## Very simple viva explanation

"We first built a basic anomaly detection model — a small neural
network — so that we have a reference point. It correctly classifies
about 78% of the final, unseen test data, and does a good job avoiding
false alarms. Later, when we introduce spatio-temporal modeling,
federated learning, and DRL, we can compare those approaches against
this baseline to see whether they actually improve on it — and by how
much."

## Limitations

- This baseline looks at each connection **completely independently**
  — it has no concept of time or sequences of events. A later phase
  (spatio-temporal modeling) will address this.
- It misses about a third of attacks in the test set, mostly novel
  ones it never saw examples of during training.
- It was trained on ordinary computer network traffic, not satellite
  traffic — satellite simulation comes later.
- It's a single, centralized model — no federated learning, no
  multiple "satellite" clients, no privacy-preserving training yet.
- We did not aggressively tune hyperparameters (learning rate, layer
  sizes, etc.) — this was intentional, since the goal was a simple,
  trustworthy reference point, not the best possible score.

## What comes next?

Phase 3 will build a spatio-temporal model — one that can look at
*sequences* of connections over time (and eventually across multiple
satellites) instead of judging each record in total isolation. We'll
compare its results directly against this baseline's real numbers
above.
