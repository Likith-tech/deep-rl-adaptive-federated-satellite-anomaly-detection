# Phase 5 — Local Satellite Training

## What was the goal?

Before we do Federated Learning, we need to know what happens if every
satellite just trains alone, with no help from the others. This phase
answers that question honestly, with real measured numbers, so we have
something solid to compare Federated Learning against later.

## What did we do?

Each of the 8 simulated satellites created in Phase 4 (`SAT-01` through
`SAT-08`) trained its own copy of the same simple neural network (the
Phase 2 baseline MLP), using ONLY the real NSL-KDD records that
satellite holds locally. All 8 models started from the exact same
random initial weights, so any differences between them come from their
different local data — not from a lucky or unlucky starting point.

## Did satellites communicate?

**No.** There is no message passing, no averaging of model weights, no
central server, and no shared training in this phase. Each satellite's
training loop never looks at any other satellite's data or model. That
communication step is exactly what Federated Learning (the next phase)
will add.

## Why is this important?

These local-only results become the baseline that Federated Learning
needs to beat. If a federated approach can't outperform what a single
well-trained local model already achieves, then federating wasn't
worth it. We can't know that without first honestly measuring what
"alone" looks like.

## What model did we use?

The same small feed-forward neural network from Phase 2:

```
121 input features
    -> Dense(128) -> ReLU -> Dropout
    -> Dense(64)  -> ReLU -> Dropout
    -> Dense(1)   -> logit -> probability of "anomaly"
```

We deliberately used the simple, already-understood Phase 2 model
instead of the more complex Phase 3 temporal model, so this phase's
results are easy to interpret on their own before adding more
complexity.

## What did we measure?

Two different things, and we kept them clearly separate:

1. **Local training performance** — how well a satellite's model fits
   the data it was trained on. This mostly reflects memorization of
   local patterns.
2. **Global validation performance** — how well each satellite's model
   does on one shared, held-out set of records that no satellite
   trained on. This is the fair way to compare all 8 satellites against
   each other, because they're all being tested on the exact same data.

KDDTest+ (the final test set) was not touched anywhere in this phase —
that's reserved for later, once Federated Learning is in place.

## What happened?

All 8 satellites trained successfully. On the shared global validation
set:

| Satellite | Local samples | Anomaly % | Validation F1 |
|---|---:|---:|---:|
| SAT-01 | 8,675 | 99.07% | 0.9242 |
| SAT-02 | 39,030 | 63.89% | 0.9918 |
| SAT-03 | 6,813 | 16.20% | 0.9768 |
| SAT-04 | 6,246 | 81.89% | 0.9714 |
| SAT-05 | 8,382 | 39.19% | 0.9738 |
| SAT-06 | 4,819 | 43.76% | 0.9913 |
| SAT-07 | 19,144 | 17.47% | 0.9693 |
| SAT-08 | 13,968 | 9.65% | 0.9781 |

Full numbers (accuracy, precision, recall, ROC-AUC, false positive rate,
confusion matrices, training curves) are in
`results/reports/local_training_results.md`.

## Which satellites performed best/worst?

- **Best: SAT-02** (validation F1 = 0.9918). It has the largest local
  dataset (39,030 samples) and a fairly balanced mix of normal and
  attack traffic (63.9% anomaly), and it saw every attack category.
- **Worst: SAT-01** (validation F1 = 0.9242). Its local data is almost
  entirely one kind of attack (99.07% anomaly, dominated by "dos"). It
  still catches almost all real attacks (recall 0.9958) but produces
  more false alarms than the others (precision 0.8623) — a natural
  consequence of training on data that's almost never "normal", so the
  model has too little practice recognizing normal traffic.

## Why might performance differ?

The satellites' local data is deliberately **non-IID** (Phase 4) — each
one sees a very different mix of traffic. We observed that satellites
with very extreme local anomaly rates (nearly all-attack, like SAT-01,
or nearly all-normal) or missing an attack category (SAT-07 has zero
"r2l" records locally) tend to land at the weaker end of the validation
ranking. We're careful to say this is an **observed pattern in this
one run with 8 clients**, not proof that non-IID data *causes* worse
performance — that would need a larger, more controlled study. Full
discussion in `results/reports/local_training_results.md` section 12.

## How do I explain this to my mam?

"In Phase 5, we trained a separate anomaly-detection model on each
simulated satellite using only that satellite's own local data. The
satellites did not communicate with each other at all. We then tested
all 8 of their models on the exact same validation dataset, so we could
see how much their performance changes just because of their different
local data. The satellite that saw the most balanced mix of traffic did
best; the satellite that saw almost nothing but one kind of attack did
worst. These results give us the honest baseline that we'll compare
against once we introduce Federated Learning, which lets satellites
learn from each other's experience without sharing their raw data."

## Important terms

- **Local training:** training a model using only one client's own
  data, with no outside help.
- **Local model:** the model that results from local training — one
  per satellite, all independent of each other.
- **Global validation:** one shared, held-out set of records used to
  test every satellite's model on equal footing.
- **Non-IID data:** "not independent and identically distributed" —
  each satellite's local data comes from a meaningfully different mix
  than the others (established in Phase 4).
- **Client:** one simulated satellite, with its own local data and now
  its own independently-trained local model.
- **Checkpoint:** a saved snapshot of a model's weights at its best
  training epoch (chosen by lowest validation loss), so we can reload
  and evaluate it later without retraining.

## Limitations

- NSL-KDD remains a terrestrial network intrusion dataset — nothing
  here changes that or claims otherwise.
- The satellite/client framing is simulated (from Phase 4); only the
  underlying network records are real.
- Local models never talk to each other — there is no aggregation, no
  FedAvg, no global model in this phase.
- The numbers above are global-**validation** results, used because
  KDDTest+ must stay untouched until federated learning is in place.
  They are notably higher than Phase 2's final KDDTest+ numbers — that
  gap is expected and not a red flag: Phase 2's own validation F1 was
  also much higher (0.9959) than its final test F1 (0.7790), because
  KDDTest+ deliberately contains attack types the training data doesn't
  have. See the explicit caveat in
  `results/reports/local_training_results.md` section 13.
- With only 8 clients, statistics like mean/standard deviation describe
  this specific run — they're not a statistically powered study.

## What comes next?

Phase 6 (Federated Learning Baseline): let the 8 satellites train
together using a federated algorithm (e.g. FedAvg) that combines what
each one learns into a shared global model, without any satellite
sending its raw data anywhere. We'll then compare that federated
model's performance against the local-only numbers measured here, to
see whether federating actually helps.
