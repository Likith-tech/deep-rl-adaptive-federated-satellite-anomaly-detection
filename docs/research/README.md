# Research Records

This directory contains the project's **permanent research evidence and design
decisions** — the trail of investigation behind choices that aren't visible from
reading the code alone. It exists so that this project's dataset and architecture
decisions can be defended to a project guide, audited by a future teammate, or revisited
months later without having to reconstruct the reasoning from memory.

Nothing in this directory is itself an implementation. It documents *why* a decision was
made (or is proposed), not the decision's code.

## Structure

```
docs/research/
├── README.md                      — this file
├── datasets/                      — real spacecraft/satellite dataset investigation
│   ├── README.md                  — how this sub-folder is organized
│   ├── spacecraft_dataset_search.md   — full search + per-dataset verification
│   ├── dataset_comparison.md      — scored comparison table across candidates
│   ├── dataset_decision.md        — the current recommendation (pending approval)
│   └── rejected_datasets.md       — candidates investigated and rejected, with reasons
├── decisions/
│   └── research_decision_log.md   — chronological log of research-driven decisions
├── claude-research/                — earlier research session artifacts (prompt + raw findings)
└── dataset-verification/           — earlier research session artifacts (verification report + comparison)
```

## Relationship to the earlier `claude-research/` and `dataset-verification/` folders

Two prior research passes on this same question (real spacecraft dataset discovery)
already produced `docs/research/claude-research/` and `docs/research/dataset-verification/`.
Those files are **not duplicated or deleted** — they are preserved as-is and referenced
from `docs/research/datasets/spacecraft_dataset_search.md`. The new `datasets/` and
`decisions/` folders created alongside them consolidate and extend that earlier work into
the specific structure requested for this investigation, add newly-verified candidates
(notably OPSSAT-AD), and add the formal decision-log format. Read the `datasets/`
folder first; treat `claude-research/` and `dataset-verification/` as supporting
archival detail behind it.

## Status of every decision recorded here

Unless a file explicitly states otherwise, everything in this directory is a
**research finding or a proposed recommendation — not an approved, implemented
decision.** No dataset described here has been downloaded or integrated into this
project's code, and no Phase 9 work has begun as a result of this research.
