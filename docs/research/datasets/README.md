# Dataset Research

## Why this folder exists

This project's dataset choices need to be defensible — to a project guide, to a future
teammate, and to this project's own stated rule of never fabricating or overstating what
a dataset can prove (see `docs/datasets/dataset_selection.md`'s satellite-domain
limitation section). This folder is where that defense lives: what was searched for,
what was found, what was verified against a primary source versus merely claimed, and
why the final recommendation is what it is.

## How dataset decisions are documented

Each investigation follows the same three-document pattern:

1. **Search + verification** (`spacecraft_dataset_search.md`) — every candidate found,
   with per-dataset structured facts (provenance, real-vs-simulated, labels, license,
   size, source) and an explicit evidence-type label (FACT / SOURCE CLAIM / ENGINEERING
   ASSESSMENT / UNVERIFIED) on every claim that isn't independently confirmed.
2. **Comparison** (`dataset_comparison.md`) — a single table plus scored criteria across
   all serious candidates, so datasets can be compared on equal footing rather than
   argued about individually.
3. **Decision** (`dataset_decision.md`) — the actual recommendation, explicitly marked
   as proposed/pending approval until a human reviewer signs off, with what it does and
   does not let this project claim.

## Where official sources are recorded

Every dataset's official source URL (Zenodo record, official GitHub org, Mendeley Data
page, journal DOI) is recorded directly in `spacecraft_dataset_search.md` next to the
claims it supports — not centralized separately, so the source stays next to what it's
evidence for.

## Where comparisons are stored

`dataset_comparison.md` — the single comparison table and the per-dataset scored
criteria with justifications.

## Where rejected datasets are recorded

`rejected_datasets.md` — every candidate that was investigated and NOT selected,
including ones that are still "open" (e.g., blocked on an unresolved license) rather than
definitively dead. The point is that "we looked at X and didn't pick it" is recorded even
when X never makes it into the comparison table's shortlist.

## Where future researchers should add new dataset evaluations

1. Add the new candidate's structured verification to `spacecraft_dataset_search.md`,
   using the same field template already used there, and mark every unconfirmed claim
   `UNVERIFIED` rather than presenting it as settled.
2. Add it to the comparison table and scored criteria in `dataset_comparison.md`, with a
   justification for every score — don't add a score without a reason.
3. If it changes the recommendation, update `dataset_decision.md` and add a new entry to
   `../decisions/research_decision_log.md` describing what changed and why — do not
   silently edit the existing decision without a logged reason.
4. If it is investigated and not selected, add it to `rejected_datasets.md` with the
   same rigor as an accepted candidate — a rejection without a documented reason is not
   useful to a future reader.
5. Never delete a prior finding to make room for a new one. If a prior finding turns out
   to be wrong, correct it in place and say so, rather than removing the record that it
   was ever believed.

## Relationship to earlier research artifacts

Two earlier research passes produced `../claude-research/` (the exact prompts used and
raw findings) and `../dataset-verification/` (an earlier verification report and
comparison, using a slightly different document structure). Those are preserved as
historical record and cross-referenced from `spacecraft_dataset_search.md`, not merged
or deleted — this folder's structure supersedes theirs going forward for new work, but
their content remains valid supporting evidence.
