# STIN Dataset Track — Research Log

New, isolated research log for the STIN (Satellite-Terrestrial
Integrated Network) dataset exploration, kept separate from
`docs/research/experiments/phase-9b-drl/` (the NSL-KDD DQN track,
frozen and unaffected by this work).

---

## Entry 1 — Branch isolation + full feasibility audit (no training)

**Date:** 2026-08-15 (later session). **Branch:** `project-development-5-stin`,
created from `a6cd9af` (the newly-committed Phase 9B state — see §"branch
isolation" below for why a commit was required first).

**Prompt/task:** Move to a new, isolated dataset experiment (STIN) to
test the existing FL/DQN framework on a satellite-specific
network-security dataset, WITHOUT touching any NSL-KDD result. First
perform a complete feasibility audit before any download-and-train.

**Branch isolation — a real discrepancy found and resolved before
branching:** git history had no commit beyond `c58a8a2` (Phase 9A);
all of Phase 9B (design, pilot, full run, Improvement 01 — ~35 files)
existed only as uncommitted working-tree files. Since uncommitted/
untracked files are not tied to any branch in git, creating
`project-development-5-stin` directly would NOT have isolated it from
Phase 9B's uncommitted state — both would have shared the same working
tree. Flagged to the user via `AskUserQuestion`; approved response:
commit Phase 9B first, then branch. Two commits made on
`project-development-4-drl`: `4e64f29` (development-log +
paper-study documentation) and `a6cd9af` (the Phase 9B DQN work
itself — design, pilot, full run, Improvement 01, source, tests).
`project-development-5-stin` was then created from `a6cd9af`, a real,
genuinely isolated commit. Working tree confirmed clean before
branching; original Phase 9B artifacts confirmed present after.

**What was inspected:** the actual STIN GitHub repository
(`github.com/kun9717/STIN-data-set`), both its `main` and `master`
branches (not assumed identical — the `master` README explicitly
warns it differs), the FL-specific `training and testing set of FL/`
directory, and the project's own pre-existing
`docs/research/paper-study/01-paper-notes/core-paper-fedstar.md` (the
FedSTAR paper this comparison is meant to reference). All dataset
files were downloaded and inspected directly with real tools (not
summarized by a secondary source) — see
`docs/project-progress/13-stin-feasibility-audit.md` for the complete,
detailed audit.

**The decisive finding:** SAT20.csv and TER20.csv (both branches)
contain **zero normal/benign-labeled rows** — verified by reading
every label value (SAT20: UDP_DDoS=43,244, Syn_DDoS=39,076; TER20: 9
attack classes, no "Normal"). This exactly matches the `master`
README's explicit disclosure ("we only open source the characteristics
of malicious traffic"). `82,320 + 127,244 = 209,564` — an exact match
to the FedSTAR paper's own stated STIN instance count, independently
confirming the paper's "STIN" is the SAT20+TER20 union, and that even
the paper's own dataset (not just what's on GitHub) is fundamentally a
9-class attack-type classification set at the data level.

**Other real findings:** the paper's own FL split (`SAT20_host1.csv` =
100% Syn_DDoS, `SAT20_host2.csv` = 100% UDP_DDoS — single-class-pure
clients) only provides 2 real satellite-side clients, not 8; the host
files cover only 65,856 of SAT20's 82,320 rows (16,464-row shortfall,
unexplained by the repo); a 264-row exact-feature-duplicate overlap
was measured between `SAT20_host1.csv` and the shared `testing set.csv`
— a real de-duplication requirement for any split we build ourselves,
not necessarily evidence of the authors' own leakage (this dataset has
many naturally-identical minimal-flow feature vectors). 12 of 30
feature columns are constant (zero variance) across the entire SAT20
file, checked over all 82,320 rows, not sampled.

**Decision point surfaced, not resolved:** per this project's
scientific-integrity policy (never conflate attack-type classification
with normal/anomaly detection — the same discipline already applied to
OPSSAT-AD in Phase 9A), the missing normal class is a genuine fork
point handed back to the user (three options recorded in
`13-stin-feasibility-audit.md` §7) before any loader, preprocessing,
partition, or FL-smoke-test code is written — building that code
before the fork point is resolved would risk exactly the "blindly
download and train" mistake this task explicitly warned against.

**Problems encountered:** the branch-isolation gap (resolved via user
decision, above); no coding problems — this entry is audit-only, no
source code was written.

**Files created this entry:** `docs/project-progress/13-stin-feasibility-audit.md`,
this log file. No `src/` files. No preprocessing, partition, or
loader code.

**Tests:** none run — no code exists yet for this track. The
project's full existing test suite was NOT rerun this entry since no
source file was touched (only new documentation).

**Artifacts:** dataset files downloaded to a local scratch/temp
location for inspection only (never committed, never copied into the
repository) — `data/raw/stin/` does not exist and was not created,
since Step 6 explicitly deferred download/wiring pending the fork
point.

**Next step:** await the user's decision on the §7 fork point
(multi-class reframing vs. not proceeding vs. — rejected —
fabricating normal traffic) before writing any `src/satellite/stin_*.py`
code, partition strategy, or smoke test.