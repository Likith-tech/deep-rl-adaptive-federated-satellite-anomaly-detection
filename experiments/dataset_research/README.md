# Dataset Research Experiments

This directory is reserved for **future** reproducible experiment artifacts (run
configs, small metadata, results) produced once a real spacecraft/satellite dataset is
actually approved and integrated — mirroring how `experiments/non_iid/` and
`experiments/adaptive_fl/` hold this project's other phase-specific experiment records.

## Current status

**Empty of experiment results.** No dataset has been downloaded, no preprocessing has
been run, and no model has been trained against any candidate identified in
`docs/research/datasets/`. This directory exists now only to reserve the location and
naming convention for that future work, consistent with this project's practice of
scaffolding a phase's structure before populating it (see `docs/research/datasets/README.md`
for the research trail that will inform what eventually goes here).

## What will go here once a dataset is approved

Following this project's existing convention (see `experiments/non_iid/alpha_*/` and
`experiments/adaptive_fl/<rule>/` for the pattern): a subdirectory per real-spacecraft-
dataset experiment, each containing a small `run_config.json` and any round/result
history — never raw dataset files or model checkpoints, which stay git-ignored per this
project's existing `.gitignore` conventions.

## Related documentation

- `docs/research/datasets/dataset_decision.md` — the proposed dataset and integration
  plan this directory will eventually implement, pending approval
- `docs/research/decisions/research_decision_log.md` — the chronological record of how
  that decision was reached
