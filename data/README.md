# Data Directory

This directory is intentionally empty at the end of Phase 0. No dataset has
been downloaded yet.

## Structure

- `raw/` — Original, unmodified downloaded datasets (Phase 1).
- `interim/` — Intermediate cleaned/transformed data (Phase 2).
- `processed/` — Final feature-engineered datasets ready for modeling (Phase 2-4).
- `partitions/` — Per-satellite non-IID partitions used for federated learning (Phase 6+).

## Notes

- Large data files are excluded from Git via `.gitignore`. Only directory
  structure (`.gitkeep`) and this README are tracked.
- Dataset selection and acquisition (e.g. CICIDS2017, UNSW-NB15, NSL-KDD,
  or a satellite-specific dataset) happens in Phase 1, with an explicit
  disk-space check and explanation before any download.
