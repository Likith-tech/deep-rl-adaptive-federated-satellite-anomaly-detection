# Phase 9B — Design Checkpoint Review

**Addendum, 2026-08-15 (same day, follow-up session):** all four issues
identified below were corrected in
`docs/project-progress/11-phase-9b-drl-adaptive-federated-learning.md`
(§8, §12, new §26, new §27). This original review log is preserved
unchanged below — the correction record itself, and the SECOND design
check verifying the corrections, are in
`docs/research/experiments/phase-9b-drl/design-checkpoint-review-v2.md`.

## 1. Date

2026-08-15 (session continuing directly from the Phase 9B design work
recorded in `docs/research/experiments/phase-9b-drl/README.md`, Entries
1-3).

## 2. Repository / branch

Branch `project-development-4-drl`, HEAD `c58a8a2` ("Phase 9A:
OPSSAT-AD real spacecraft validation"), confirmed via `git show -s
c58a8a2` (commit intact, message and hash match prior records). `git
status` at the start of this review: only pre-existing untracked
documentation files from the prior two sessions
(`docs/project-progress/11-phase-9b-drl-adaptive-federated-learning.md`,
`docs/project-progress/11-phase-9b-drl-adaptive-fl-design.md`,
`docs/research/development-log/`, `docs/research/experiments/`) —
nothing staged, nothing committed, no Phase 1-9A file modified.

## 3. Documents inspected

- `docs/project-progress/11-phase-9b-drl-adaptive-federated-learning.md`
  (the canonical Phase 9B design document — read in full, all 25
  numbered sections plus the reusable-code/new-files/test-plan/
  reproducibility/git-history/next-steps sections).
- `docs/research/experiments/phase-9b-drl/README.md` (prior
  prompt/response research-log entries 1-3, for continuity).
- `docs/research/development-log/009-phase-8.md` and
  `MASTER_PROJECT_JOURNAL.md` (re-confirmed Phase 8's committed numbers
  match what the design document cites).
- `results/reports/adaptive_fl_results.md` and
  `results/reports/federated_results.md` (source of the baseline
  numbers the design document cites — re-verified, not re-derived).

## 4. Code inspected (this session; also previously read in depth in
   earlier sessions of this same conversation and re-confirmed here)

- `scripts/evaluate_adaptive_fl_results.py` (lines 82-212) — re-read
  specifically to verify HOW Phase 8's "client fairness gap" metric is
  actually computed, since the Phase 9B reward formula reuses it.
  Confirmed: this script reads `PARTITIONS_DIR / cid / "train.parquet"`
  directly (`evaluate_on_parquet`, called once per client, line
  204-205) to evaluate the FINAL selected model against each client's
  own local data — a standalone, offline, POST-HOC analysis script run
  once after training completes, entirely outside the live
  server/client privacy-boundary protocol that `src/federated/server.py`
  is tested against.
- `src/federated/adaptive_trainer.py`, `src/federated/adaptive.py`,
  `src/federated/server.py` (re-confirmed from earlier sessions): the
  ONLY per-round, in-training signal Phase 8 computes is `performance_k`
  (each client's update evaluated against the server-owned GLOBAL
  validation set — no client-local data touched). The `fairness_k`
  signal fed into the live scoring rule is a STATIC, precomputed
  Shannon-entropy value passed in as a parameter
  (`client_fairness_entropy`), not derived from any per-round
  evaluation. Confirms Phase 8 never evaluates the in-progress global
  model against client-local data during training — only Phase 8's
  offline analysis script does that, once, at the end.

## 5. Design verification

**A. Research question / DQN necessity** — Sound. Phase 8's own
ablation table (`resource_only` best F1, `performance_only` — not the
fairness-weighted `combined` rule — best fairness) is real, cited
evidence that a fixed linear rule cannot resolve which signal matters
when, which is exactly the adaptability gap a learned per-round policy
targets. This is a defensible motivation, not an assumed one.

**B. RL environment (state/action/reward/episode/transition/terminal)**
— Structurally sound and consistent with Phase 6/8's actual round loop
(§5, §9, §10 map cleanly onto `adaptive_trainer.py`'s existing
train→evaluate→aggregate→evaluate sequence). One arithmetic
inconsistency found in the state definition (see Issue 2 below); one
substantive gap found in the reward's fairness term's data source (see
Issue 1 below, the primary finding of this review).

**C. Action space** — Confirmed still the discrete 9-profile weighting
catalogue (§7); no continuous action was introduced, and the document
explicitly re-evaluated and rejected a delta-based discrete alternative
with a stated reason (harder to learn, harder to interpret). No change
recommended here — the existing choice remains the more defensible one.

**D. Reward gameability / rare-client neglect** — The reward cannot be
trivially gamed on the performance term (`ΔF1_t` requires genuine
validation improvement) and the fairness term is a penalty on spread,
not a reward for favoring any one client, so it does not create an
incentive to ignore rare clients — the intent is sound. **However**,
the mechanism by which `fairness_gap_t` would be computed EVERY ROUND
is not actually available anywhere in the current codebase (see Issue
1) — this is a design gap, not a gameability flaw.

**E. Fairness assumption** — Verified: §24 explicitly states Phase
7/8's finding that more fairness weight does not guarantee better
fairness, and explicitly declines to assume the DQN will behave
differently. No correction needed here.

**F. Privacy** — The stated boundary (§11, §15) is correct in
principle (clients never send raw data; server receives
`ClientUpdate`/statistics only) and matches the current, tested Phase
6-8 boundary. **However**, Issue 1 directly threatens this boundary in
practice: as written, the reward computation would need to reach
client-local data every round to produce `fairness_gap_t`, which either
breaks the boundary (if done server-side, mirroring what
`evaluate_adaptive_fl_results.py` does — acceptable for Phase 8's
one-time offline analysis, NOT acceptable for a per-round, in-training
signal touched up to ~1,600 times per the design's own pretraining
budget) or requires an undocumented protocol extension.

**G. Experimental comparability** — Sound. §14 fixes 8 clients, the
Phase 4/6 exact partition, the Phase 2 MLP, local epochs, batch size,
learning rate, optimizer, validation/test splits, and shared init seed
42 identically across all three methods for the final comparison run —
matches the instruction's checklist exactly. No historical Phase 6/7/8
file or result is read as anything other than a fixed reference number.

**H. Real spacecraft dataset (OPSSAT-AD) boundary** — **Gap found**:
the canonical design document (`11-phase-9b-drl-adaptive-federated-learning.md`)
does not contain any explicit statement that OPSSAT-AD/Track B is
excluded from Phase 9B. The EARLIER draft
(`11-phase-9b-drl-adaptive-fl-design.md`) does contain this statement
("Real spacecraft data boundary" section), but the canonical document
that supersedes it dropped it during consolidation. This is a
documentation-completeness gap, not a code risk (no code exists yet
that could actually inject OPSSAT-AD into the FL experiment), but it
should be explicitly present in the canonical document before
implementation begins, per the instruction's explicit requirement to
confirm this boundary. See Issue 3.

**I. Reproducibility** — Mostly sound: seeds (§14, "Reproducibility
information"), configuration file location (`configs/drl.yaml`), and
the disclosed seed asymmetry (episode partition seeds vary, DQN/final-run
seed fixed at 42) are all specified. **Minor gap**: the document does
not specify WHERE Phase 9B's checkpoints/experiment directories/plots
will be written, unlike Phase 7/8 which established clear conventions
(`experiments/non_iid/alpha_<X>/`, `experiments/adaptive_fl/<rule>/`,
`results/models/federated/round_history.json`). See Issue 4.

**J. Documentation** — The decision trail is well recorded across the
design document and the three prior research-log entries; this review
itself continues that record rather than replacing it.

## 6. Issues found

**Issue 1 (primary, substantive) — the reward's fairness term is not
actually computable per-round without either breaking the privacy
boundary or extending the client protocol.**
`fairness_gap_t = max_k(local_F1_k) − min_k(local_F1_k)` (§8) reuses
Phase 8's "own-local-data F1 gap" metric. That metric is real and
correctly computed by Phase 8 — but only ONCE, offline, in
`scripts/evaluate_adaptive_fl_results.py`, by directly reading every
client's `train.parquet` (confirmed in code, §4 above). Nothing in the
current Phase 6/7/8 codebase computes a per-client local-data
evaluation DURING training — the only per-round signal is
`performance_k`, evaluated against the server-owned GLOBAL validation
set, which never touches client-local data. Using the local-F1 gap as
a per-round, in-training reward signal (as the design currently
specifies) would require either (a) the reward-computation code
reading client parquet files directly, up to ~1,600 times across the
design's own stated pretraining budget — a direct violation of the
privacy boundary `server.py`'s existing AST test enforces — or (b) an
undocumented extension to `ClientUpdate`/the client protocol so each
client locally evaluates the newly-aggregated global model and reports
back a scalar F1 each round. **Why it matters**: this is exactly the
kind of gap that would only surface during implementation, likely as a
rushed, undocumented workaround under time pressure — catching it now,
before any code exists, is the entire purpose of a design checkpoint.

**Issue 2 (minor, arithmetic) — stated state dimension is inconsistent
with its own components.**
§12 states the state is "≈43-dim." Counting §6's own listed components
(8 performance + 8 data + 8 resource + 8 fairness = 32, + 2 validation
trend, + 1 round-progress, + 9 previous-action one-hot) sums to 44, not
43. Off-by-one, cosmetic, but worth fixing before it propagates into
actual code (`input_dim` in `configs/drl.yaml`).

**Issue 3 (documentation completeness) — canonical document is missing
an explicit OPSSAT-AD/Track B exclusion statement.**
See item H above. The content already exists (in the superseded
draft) and just needs to be carried into the canonical document.

**Issue 4 (documentation completeness) — no stated output-path
convention for Phase 9B experiment artifacts.**
See item I above. Phase 6/7/8 each established a clear, consistent
`experiments/<phase>/...` and `results/models/<phase>/...` convention;
Phase 9B's document doesn't yet say where its own checkpoints/round
histories/plots will live.

## 7. Corrections recommended

For Issue 1 (the only issue with scientific/architectural weight): the
**smallest** scientifically justified correction is to NOT extend the
client protocol and NOT touch the privacy boundary. Instead, replace
the PER-ROUND reward's fairness term with a proxy that is already
computed every round, server-side, with no new client-local access:
the **spread of `performance_k`** (e.g. `max_k(performance_k) −
min_k(performance_k)`, using the already-existing global-validation
performance signal every round already computes) as the in-training
fairness proxy. The TRUE own-local-data fairness gap (Phase 8's exact
metric) is retained exactly as originally specified for the **final,
post-hoc comparison table** (§25's required metrics), computed once,
the same way Phase 8 computes it — never as a live per-round training
signal. This is the smallest change because it touches only one term
of one formula, introduces no new protocol, no new privacy exposure,
and no new code path — it simply moves the boundary between "training
signal" and "final report metric" to match where the codebase's actual
capabilities already sit, mirroring how Phase 8 itself only ever used
the local-F1 gap as a post-hoc report, never a live signal.

For Issue 2: correct "≈43-dim" to "44-dim" (or state the formula
symbolically, `8×4 + 2 + 1 + |A|`, so it stays correct if `|A|` changes
during implementation).

For Issue 3: add a short "Real spacecraft data boundary" note to the
canonical document, carrying forward the equivalent section already
written in the superseded draft (`11-phase-9b-drl-adaptive-fl-design.md`)
essentially verbatim — no new reasoning required, just restoring what
was dropped during consolidation.

For Issue 4: add a short "Output/experiment directory convention"
note specifying paths consistent with the existing pattern, e.g.
`experiments/drl/<episode_id>/`, `results/models/drl/round_history.json`,
`configs/drl.yaml` — to be finalized at implementation time but at
least named in the design now.

None of these four corrections require touching Phase 6/7/8/9A code,
re-running any experiment, or changing the action space, the DQN
architecture, or the overall research question — the core design
(discrete weighting-profile action, validation-only performance
reward term, weighting-not-selection, many-episode pretraining) is
sound and is NOT being redesigned.

## 8. Final decision

**NEEDS DESIGN REVISION** — narrowly. This is not a rejection of the
approach; it is one substantive, concrete fix (Issue 1) plus three
small documentation completions (Issues 2-4), all scoped to edits
within the existing design document. No architectural rework, no
change to the action space, DQN, or experiment plan is required.

## 9. Exact next step

Update `docs/project-progress/11-phase-9b-drl-adaptive-federated-learning.md`
§8 (Reward) to use the performance-spread proxy for the per-round
fairness term (retaining the true own-local-data gap for the final
comparison table only), fix the state-dimension arithmetic in §12, add
the OPSSAT-AD boundary note, and add the output-path convention note —
then return for a second, brief confirmation pass before any DQN
implementation begins. No code, no commit, no push until that revision
is reviewed and approved.
