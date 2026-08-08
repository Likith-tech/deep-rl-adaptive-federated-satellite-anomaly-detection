# Phase 3 — Temporal Anomaly Detection (Foundation for the Future Spatio-Temporal Model)

**Scope of this phase, stated plainly:** this phase builds only the
**TEMPORAL** component/foundation of the eventual spatio-temporal
model. It does **not** implement genuine spatial or multi-satellite
modeling — there is still only one data source, not multiple
satellites. The "spatio-" half only becomes real once the satellite
simulation environment exists (the next milestone in this project's
plan — see "What comes next?" below).

## What was the goal?

Build a model that looks at a **sequence** of network connections
together, instead of judging each one completely on its own — the
temporal "half" of the eventual spatio-temporal model. The spatial
"half" (multiple satellites) doesn't exist yet — that needs the
satellite simulation from the next phase.

## What problem did the baseline have?

Our Phase 2 MLP looks at one connection record at a time and forgets
everything else. Record 1, record 2, record 3... each judged in total
isolation, with no memory of what came before. Real attacks can show up
as a *pattern* across several connections in a row, not just one weird
connection — the baseline structurally cannot see that.

## What did we build?

- **Sequence construction** (`src/preprocessing/sequences.py`): groups
  consecutive records into fixed-length windows, so the model receives
  a *sequence* instead of a single record.
- **A GRU** (Gated Recurrent Unit): a type of neural network built to
  read a sequence step by step and carry forward a "memory" of what it
  has seen so far.
- **Temporal attention**: after the GRU has processed the whole
  sequence, this lets the model learn which timesteps mattered most for
  the final decision, instead of treating them all equally.

```
Record 1 → Record 2 → ... → Record N
                    ↓
      Linear projection → GRU (reads the sequence)
                    ↓
        Temporal attention (weighs each timestep)
                    ↓
                  Dense → Normal / Anomaly
```

## Attempt 1: a labeling mistake we caught ourselves

Our first version labeled a sequence "Anomaly" if **any single record**
inside it was an attack. We trained it, and it scored **100%** on every
metric on the final test set.

That should have been a red flag, and we treated it as one instead of
reporting it proudly. We investigated *why* instead of accepting the
flattering number, and found the real problem: NSL-KDD has an attack in
roughly every second record (46-57% of the time). If a sequence counts
as "Anomaly" the moment even *one* of its 16 records is bad, then almost
every possible sequence in the entire dataset ends up labeled "Anomaly"
— we measured this directly: **100% of the real test sequences** at
that length carried the anomaly label. A model that always shouted
"Anomaly!" without reading the data at all would have scored exactly as
well as ours did. The confusion matrix proved it — there wasn't a
single genuine "Normal" sequence left in the test set to get right or
wrong.

**We rejected that result.** It is preserved (not deleted) in
`experiments/temporal/initial_any_anomaly_temporal_results_ARCHIVE.md`
and summarized inside `results/reports/temporal_results.md` under
"Initial sequence-labeling experiment (rejected)," because hiding a
failed experiment would be worse than reporting it honestly.

## Attempt 2: fixing it with evidence, not guesswork

Instead of guessing a replacement rule, we measured four candidate
labeling strategies directly against the real data
(`scripts/analyze_sequence_labeling.py` →
`results/reports/sequence_labeling_analysis.md`):

| Strategy | What it means | Result |
|---|---|---|
| Any | anomaly if ANY record in the window is bad | Degenerate (the mistake above) |
| Majority | anomaly if MORE THAN HALF the records are bad | Non-degenerate, but train (~29%) and test (~52-72%) balance shift a lot |
| **Last** | sequence label = the label of the LAST record; earlier records are just context | Non-degenerate, and its balance (~46% train/val, ~58% test) closely tracks the real per-record attack rate |
| Ratio (threshold) | anomaly if the fraction of bad records passes a chosen cutoff | Behaves like Any or Majority depending on the cutoff — arbitrary to justify |

We picked **"Last"**: label a sequence by its *final* record, using the
earlier records purely as context. Reasons:

1. It doesn't collapse to one class on any split (confirmed by direct
   measurement, not assumed).
2. It doesn't need us to invent an arbitrary cutoff number.
3. It has the cleanest real-world meaning: *"given what just happened,
   is the most recent connection an attack?"* — a natural framing for
   live anomaly detection.
4. It makes the comparison against the Phase 2 baseline as fair as
   possible: the temporal model is judged on the exact same target
   record the baseline was judged on, just with extra historical
   context added — so any difference in score is actually about
   whether that context helped, not about a different task entirely.

## How does a sequence work now?

We take 8 connection records in a row (see "why 8" below) and treat
them as one unit. The **label** of that unit is simply whatever the
8th (last) record's own true label is — Normal or Anomaly. The GRU
reads all 8 in order and has to decide, using the 8th record's own
information *plus* whatever it can usefully extract from the 7 that
came before it.

## What is temporal attention?

After the GRU reads all timesteps of a sequence, attention lets the
model give each timestep a "vote" (weights between 0 and 1, adding up
to 1) on how much it should count toward the final decision — instead
of only trusting the very last GRU output blindly. We can look at these
weights afterward to see which positions the model leaned on.

## What did we actually measure?

We tried sequence lengths 8, 16, and 32 again (under the corrected
"Last" strategy this time) and picked the best on the **validation**
set only. Length **8** was selected.

**On KDDTest+ (final, one-time evaluation):**

| Metric | Value |
|---|---|
| Accuracy | 75.83% |
| Precision | 92.21% |
| Recall | 64.03% |
| F1 | 75.58% |
| ROC-AUC | 89.23% |
| False Positive Rate | 7.59% |

Real confusion matrix — all four cells populated this time:

```
                Predicted Normal   Predicted Anomaly
Actual Normal               1083                   89
Actual Anomaly                592                 1054
```

## Did it improve over the baseline?

**Honestly, no — not on this run**, and this time that's a real,
trustworthy answer rather than a flawed one.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---|---|---|---|---|---|
| MLP Baseline (Phase 2) | 78.32% | 92.78% | 67.13% | 77.90% | 89.72% | 6.90% |
| Temporal GRU+Attention (Phase 3, corrected) | 75.83% | 92.21% | 64.03% | 75.58% | 89.23% | 7.59% |

The temporal model is slightly *behind* the baseline on every metric
(F1 −2.32 points). We looked at the attention weights to understand
why: the model puts the overwhelming majority of its attention on the
*last* timestep — which makes sense, since that's the record being
labeled — with only a small amount on the record just before it. That
suggests the temporal model is mostly rediscovering what the baseline
already sees directly (the current record's own features), while the
extra steps it takes to get there (compressing through a smaller
projection layer, then a GRU) may cost a little accuracy rather than
add any. That's a reasonable, evidence-backed explanation — not a proven
fact — and it's a legitimate, useful result for a first temporal model:
it tells us extra sequential context, at least in this simple form,
isn't yet paying for itself on this dataset.

## What is still missing?

- **No genuine timestamps.** NSL-KDD doesn't come with real
  chronological time — we used the dataset's own row order as a stand-in
  for "sequence," documented as a limitation, not claimed as true
  satellite time-series data.
- **No spatial/satellite dimension.** Still one data source, not
  multiple satellites. The "spatio-" part of "spatio-temporal" doesn't
  exist until the satellite simulation phase.
- **Temporal context didn't help yet.** This is a real, useful finding,
  not a failure of the exercise — it tells us the next iteration needs
  either richer context (e.g. actual satellite-to-satellite
  relationships once they exist) or a different way of using the
  sequence (not just predicting the last record) to justify the added
  complexity over the simple baseline.

## How do I explain this to my mam?

"We first labeled a sequence as anomalous if any record in it was an
attack. That made almost every sequence anomalous, so the model could
score 100% by predicting anomaly every time — we caught that ourselves,
rejected the result, and kept the mistake documented rather than
hiding it. We then measured several other labeling options directly
against the data instead of guessing, and picked the one with the
clearest meaning and the most balanced, honest task: label a sequence
by its most recent record, using the earlier records as context. With
that fixed, our temporal model scored close to — but slightly below —
our baseline. That's a trustworthy result: it tells us that just adding
sequence context, in this simple form, doesn't yet beat looking at a
single record carefully. That's useful information for what we build
next."

## Important technical terms

- **Sequence:** a fixed-size group of consecutive records treated as
  one unit (here, 8 records).
- **Timestep:** one position within a sequence.
- **GRU (Gated Recurrent Unit):** a neural network layer designed to
  read data step-by-step and keep a running memory.
- **Temporal attention:** a mechanism that learns how much weight to
  give each timestep instead of treating them all equally.
- **Sequence label:** the single Normal/Anomaly label assigned to an
  entire sequence. We use "the label of the last record in the window."
- **Degenerate task:** a classification problem where nearly all
  examples belong to one class, making high scores meaningless (what
  happened in Attempt 1).
- **Validation:** data used *during* development to pick the best
  model/settings — never the final judge.
- **Test:** data used only once, at the very end, for the honest final
  score.
- **Overfitting:** when a model performs great on data it has seen but
  poorly on new data — not what happened here in Attempt 2 (train and
  validation loss tracked each other closely; see
  `results/plots/temporal/training_curves.png`).

## What comes next?

The next phase introduces a **simulated satellite network** — splitting
this traffic data across multiple pretend "satellite" clients, each
with its own limited, unevenly-distributed slice of data (non-IID), and
its own simulated constraints like limited connectivity, bandwidth, and
compute. That's what finally gives meaning to the "spatial" side of
"spatio-temporal," and sets up the federated learning phases after it.
