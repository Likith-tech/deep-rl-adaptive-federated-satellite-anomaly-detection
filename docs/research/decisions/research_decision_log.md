# Research Decision Log

Chronological record of research-driven decisions for this project. Each entry is
append-only — corrections to a prior entry's findings are recorded as a new entry that
references the old one, not as a silent edit.

---

## 2026-08-13 — First dataset discovery pass

**Decision:** No decision made; initial broad search only, in response to a request to
investigate whether a real satellite/space-network cybersecurity dataset could
strengthen the project before Phase 9.

**Evidence:** Ten themed search categories (satellite network traffic, satellite
intrusion detection, space cybersecurity, spacecraft telemetry anomalies, satellite
command-and-control, satellite IoT security, LEO network traffic, inter-satellite-link
anomalies, ground-station anomalies, spacecraft telemetry generally). Identified NASA
SMAP/MSL, ESA-ADB (via third-party `kplabs-pl` mirror), STIN/SAT20, CuCD-ID, LENS as
initial candidates. Classified each by real-vs-simulated and cyber-vs-operational.

**Alternatives considered:** N/A — first pass.

**Reason:** Establish a baseline of what exists before committing to any direction.

**Impact on project:** None — research only, no files changed at that point in this
project's own tracked history (this was prior to `docs/research/` existing).

**Next action:** Deeper verification of the top candidates at primary sources.

---

## 2026-08-13 — Git/branch audit + Category A-E formalization

**Decision:** Formalized the A/B/C/D/E dataset classification scheme (real+cyber /
real+operational / real+unlabeled / simulated+cyber / terrestrial) and explicitly
searched for whether a Category A dataset exists.

**Evidence:** Re-verified NASA SMAP/MSL and STIN/SAT20 directly; confirmed via direct
repo fetch that STIN's authors "simulate a real scenario" (not real satellite capture),
and confirmed via direct license file fetch that STIN's repo carries only a bare
copyright notice, no license grant. Also audited the git branch structure at this
point: `project-development-2` had Phases 0-6 committed locally but only Phase 0-5
pushed to origin; Phases 7-8 existed only as uncommitted working-tree changes.

**Alternatives considered:** Whether to recommend replacing NSL-KDD outright (rejected
— would invalidate Phase 2-8 comparisons); whether one dataset could satisfy both
real-spacecraft and cybersecurity relevance (rejected — no evidence found for this).

**Reason:** Establish precise category boundaries so no dataset gets misclassified by
optimistic reading of its own marketing/abstract.

**Impact on project:** None — research only.

**Next action:** Formal per-dataset primary-source verification (license, size,
provenance) for the strongest candidates.

---

## 2026-08-13 — Primary-source verification pass; first `docs/research/` archive created

**Decision:** Created `docs/research/claude-research/` (prompt + raw findings archive)
and `docs/research/dataset-verification/` (23-section verification report + scored
comparison table) as the first permanent, version-controlled research record for this
investigation, additively alongside a small, additive-only update to
`docs/datasets/dataset_selection.md` noting that real-spacecraft datasets were under
evaluation and none had been integrated.

**Evidence:** Directly fetched and confirmed: ESA-ADB license (CC BY 3.0 IGO) and size
(11.6 GB) on Zenodo; CuCD-ID license (CC BY 4.0) on Mendeley Data; LENS license (CC
BY-SA 4.0) on Zenodo; STIN/SAT20's copyright-only license status and BJTU institutional
attribution directly from the repo.

**Alternatives considered:** Whether to recommend STIN/SAT20 as primary despite its
license gap (rejected pending author contact); whether to recommend LENS despite having
no labels (rejected as a primary training source, retained as a realism reference only).

**Reason:** A recommendation without independently-verified license/provenance evidence
is not defensible to a project guide.

**Impact on project:** None on Phases 0-8 or NSL-KDD. `docs/datasets/dataset_selection.md`
received one additive section; nothing above it was changed.

**Next action:** Continue searching for additional real-spacecraft candidates
(specifically OPS-SAT, JAXA, and any named-but-unverified leads) before finalizing a
recommendation.

---

## 2026-08-14 — OPSSAT-AD discovered and verified; current recommendation established

**Decision:** Identified and directly verified **OPSSAT-AD** (ESA OPS-SAT CubeSat
telemetry benchmark) as the new top real-spacecraft candidate, superseding ESA-ADB as
the *primary* recommendation (ESA-ADB remains a recommended secondary/complementary
dataset). Consolidated all prior findings into the structured `docs/research/datasets/`
and `docs/research/decisions/` folders requested for permanent record-keeping.

**Evidence:** Directly fetched and confirmed on Zenodo: OPSSAT-AD license (CC-BY-4.0),
size (19.5 MB), open no-registration access. Directly fetched and confirmed via the
arXiv preprint (Nature page itself was paywall/auth-blocked): OPS-SAT is a real,
ESA-operated CubeSat that flew 2019 - May 2024. Directly searched for a dataset named
"CORTEX" per this round's brief — **none found**, recorded explicitly rather than
guessed at. Obtained a more precise ESA-ADB scale figure (224 channels, 821 control
signals, 1,430 annotated events) from an OpenReview listing, preserved alongside the
earlier, smaller figure rather than silently overwriting it.

**Alternatives considered:** ESA-ADB as primary (kept as strong secondary — larger,
multi-mission, but 11.6 GB and less per-mission documentation than OPSSAT-AD); NASA
SMAP/MSL as primary (kept as tertiary — excellent but older, and MSL is a rover not an
orbital satellite); a single-dataset strategy covering both real-spacecraft and
cybersecurity relevance (rejected again — no evidence any such dataset exists; a
two-dataset hybrid strategy, real spacecraft telemetry + NSL-KDD kept separate and never
merged into one metric, is recommended instead).

**Reason:** OPSSAT-AD is simultaneously real, cleanly licensed, peer-reviewed, and
small enough to integrate within a capstone timeline — the strongest combination of
verifiability and practicality found across three research passes.

**Impact on project:** None on Phases 0-8, NSL-KDD, or any model/training code. This
entry and the accompanying `docs/research/datasets/` files are the only changes.

**Next action:** Await explicit human review and approval of `dataset_decision.md`
before any Phase 9 / real-data-integration implementation work begins.
