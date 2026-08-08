# Phase 1 — Dataset & Preprocessing

## What was the goal?

Get a real network-traffic dataset, understand it, clean it, and turn
it into a well-organized, ready-to-use format that a machine learning
model can actually be trained on.

## What did we do?

- Chose a dataset called **NSL-KDD** — a well-known, publicly available
  network intrusion detection dataset.
- Downloaded it (~22.5 MB, two files: `KDDTrain+.txt`, `KDDTest+.txt`).
- Inspected it: how many rows, how many columns, what type of data,
  any missing/broken values.
- Cleaned it (checked for missing values, duplicate rows, broken
  "infinity" values — there turned out to be none, but the cleaning
  code exists and is tested in case a different dataset needs it later).
- Labeled every row as **Normal** or **Anomaly** (attack), while also
  keeping the original, more detailed attack name.
- Converted text columns (like "tcp", "http") into numbers a model can
  understand (one-hot encoding).
- Scaled the numeric columns so they're all on a similar range
  (standard scaling).
- Split the data into three parts: **training**, **validation**, and
  **test** — explained below.
- Saved everything in a clean, reusable format (`data/processed/`).
- Generated real charts and a written data-quality report.
- Wrote 16 automated tests to make sure this process works correctly
  and keeps working correctly in the future.

## Why did we do it?

A machine learning model is only as good as the data it learns from.
If we feed a model messy, mislabeled, or leaking data, any results
later would be meaningless — or worse, falsely impressive. This phase
makes sure we have a dataset we can trust and that every later model
(baseline, spatio-temporal, federated, DRL) uses the exact same,
correctly-prepared data, so comparisons between them are fair.

**Important honesty note:** NSL-KDD is **not** a satellite dataset. It's
normal (terrestrial) computer network traffic. We picked it because it's
small, clean, and free to get — a good "practice dataset" — and we are
upfront about this everywhere in the documentation. Later (Phase 4+),
we will simulate satellite behavior (multiple satellites, unreliable
connections, delays) on top of this traffic data ourselves.

## What did we create?

- `docs/datasets/dataset_selection.md` — explains exactly why NSL-KDD
  was picked, and what alternatives were considered and why they
  weren't used
- `docs/datasets/feature_decisions.md` — explains what every single
  column in the data means and whether we kept, changed, or removed it
- `src/data/` — code that loads and inspects the dataset
- `src/preprocessing/` — code that cleans, labels, splits, encodes, and
  scales the data
- `scripts/inspect_dataset.py` — a command that prints a full summary
  of the raw data
- `scripts/generate_dataset_report.py` — a command that generates the
  charts and report below
- `notebooks/01_dataset_analysis.ipynb` — a notebook exploring the data
- `results/reports/dataset_quality.md` — the actual measured numbers
- `results/plots/dataset/` — 5 real charts (class balance, attack
  types, feature distributions, feature correlation)
- `tests/data/` — 16 automated tests

## What did we actually achieve?

Real, measured numbers (not estimates):

- **Training data:** 125,973 rows, **Test data:** 22,544 rows
- **0** missing values, **0** duplicate rows, **0** broken/infinite
  values in either file
- Training set: 53.46% Normal, 46.54% Anomaly (fairly balanced)
- Test set: 43.08% Normal, 56.92% Anomaly
- 41 original columns → 121 usable numeric columns after encoding
- Data split: 107,077 training rows, 18,896 validation rows, 22,544
  test rows (the official NSL-KDD test set, kept completely separate)
- All 16 automated tests passed

## How do I explain this to my mam?

"I took a real, publicly available network traffic dataset, checked it
carefully for problems, cleaned it, labeled every record as normal or
attack, and split it properly into training/validation/test sets so
nothing gets tested unfairly. I generated real charts and a report from
the actual numbers — nothing here is made up. This is the exact dataset
every model I build afterward will be trained and compared on."

## Important technical terms

- **NSL-KDD:** the specific dataset we're using — think of it as a big
  spreadsheet of network connection records, each labeled normal or
  attack.
- **Preprocessing:** cleaning and reformatting raw data so a model can
  use it.
- **One-hot encoding:** turning a text category (like "tcp" or "udp")
  into a set of 0/1 numbers a model can read.
- **Scaling:** adjusting numbers so they're all in a similar range (so
  one huge number, like bytes sent, doesn't dominate the model just
  because it's a bigger number than the others).
- **Train/Validation/Test split:** training data teaches the model,
  validation data helps us pick the best version of the model, and test
  data is used only once at the very end to report honest results.
- **Data leakage:** accidentally letting information "cheat" into the
  model (e.g. using test data to help decide how to clean/scale the
  training data). We specifically checked and avoided this.

## Problems/limitations

- NSL-KDD is not satellite data — it's a stand-in until we build the
  satellite simulation layer later.
- NSL-KDD's test set intentionally contains attack types the training
  set has never seen (this is by the dataset's own original design, to
  test generalization) — so a model won't get 100% on those, and that's
  expected, not a bug.
- The dataset itself is from 1998-99 in origin (refined/relabeled since)
  — it doesn't reflect modern network traffic patterns perfectly.

## What comes next?

Phase 2 — actually train our first real AI model (a baseline anomaly
detector) on this prepared data, and see how well it performs using
real, measured results.
