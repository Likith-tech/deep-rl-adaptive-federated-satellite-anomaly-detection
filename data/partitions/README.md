# data/partitions/

This directory contains the **generated** simulated satellite client
datasets. Nothing here is a source file — everything is reproducibly
derived from `data/processed/train.parquet` (Phase 1 output).

## What's in here

```
data/partitions/
├── SAT-01/train.parquet   \
├── SAT-02/train.parquet    |  per-client real training records
├── ...                     |  (generated, git-ignored)
├── SAT-08/train.parquet   /
├── manifest.json           <- partition metadata (committed, small)
└── satellite_metadata.json <- simulated resource conditions (committed, small)
```

The per-client `train.parquet` files are **not committed to Git** (see
`.gitignore`) — they are large, fully reproducible, and derived data.
`manifest.json` and `satellite_metadata.json` ARE committed: they're
small, human/machine-readable summaries that make the partition
inspectable without regenerating it.

## How these are generated

```bash
# 1. Requires data/processed/train.parquet (Phase 1 pipeline output)
python -m src.preprocessing.pipeline   # if not already run

# 2. Generate the client partitions + manifest + satellite metadata
python scripts/create_satellite_partitions.py

# 3. Analyze the result (non-IID metrics, resource stats, plots, report)
python scripts/analyze_satellite_partitions.py
```

Configuration lives in `configs/satellite_simulation.yaml` (number of
clients, Dirichlet alpha, partitioning constraints, simulated resource
ranges, seed).

## Which configuration created the current partition

See `manifest.json` for the exact `seed`, `alpha`, `category_column`,
and per-client sample counts used for the partition currently on disk.
Given the same config and seed, `create_satellite_partitions.py` always
regenerates byte-for-byte the same partition and resource values
(deterministic — see `tests/simulation/`).

## What is real vs. simulated

- **Real:** every record in every `SAT-XX/train.parquet` file — these
  are actual NSL-KDD training records (with their real labels),
  distributed (not duplicated, not invented) across clients.
- **Simulated:** the client/satellite framing itself — client IDs,
  and every value in `satellite_metadata.json` (bandwidth, latency,
  compute score, availability, connectivity). NSL-KDD is a terrestrial
  network intrusion dataset; see `docs/datasets/dataset_selection.md`
  and `docs/project-progress/05-phase-4-satellite-simulation.md`.

## Data source and split boundaries

Only `data/processed/train.parquet` (KDDTrain+, after Phase 1
preprocessing) is partitioned here. `data/processed/validation.parquet`
remains a single global held-out set (not partitioned per client in
this phase), and `data/processed/test.parquet` (KDDTest+) is never
touched by this simulation in any way.
