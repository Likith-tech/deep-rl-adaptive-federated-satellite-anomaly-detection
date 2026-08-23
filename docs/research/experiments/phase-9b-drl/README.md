# Phase 9B — DRL Adaptive FL: Experiment & Prompt/Response Research Log

This directory holds the concise, ongoing research log for Phase 9B —
NOT raw conversation transcripts (per project policy, see
`docs/research/development-log/CLAUDE_PROMPT_RESPONSE_INDEX.md` for the
general recording policy this log follows). Each entry below records
what was asked, what was investigated, what was found, what was
decided, and what happened — concisely, in the project's own words,
not pasted chat logs.

Full design detail lives in
`docs/project-progress/11-phase-9b-drl-adaptive-federated-learning.md`
(canonical) and `11-phase-9b-drl-adaptive-fl-design.md` (earlier draft,
preserved, superseded). This file indexes the reasoning history behind
that document rather than duplicating it.

---

## Entry 1 — Initial Phase 9B design request (design-only)

**Date:** 2026-08-14 (session following the Phase 9A commit, on branch
`project-development-4-drl`, created that session from `c58a8a2`).

**Task/prompt summary:** Design (not implement) a DQN controller to
replace Phase 8's fixed rule weights, covering RL environment, state,
action, reward, DQN architecture, training strategy, a 3-way experiment
plan against Phase 6/Phase 8, the Track A/B (NSL-KDD vs. OPSSAT-AD)
boundary, leakage controls, reusable code, new files, and a test plan.

**What Claude investigated:** Read the actual implementation (not just
docs) of `src/federated/{client,server,protocol,fedavg,adaptive,adaptive_trainer}.py`,
`src/simulation/{non_iid_metrics,network_conditions}.py`,
`configs/{federated,adaptive_fl,drl}.yaml`, and confirmed `configs/drl.yaml`
was still a null placeholder and `src/drl/` still empty (no prior DRL
work existed).

**Important findings:** Phase 8's four signals (performance/data/
resource/fairness) are all already server-side, already normalized,
already reusable without new code. `server.py`'s privacy boundary is
enforced by a real AST-based test, not just documented. Phase 8's own
result table shows fairness-weighted rules do not automatically win on
fairness — an explicit finding this design must not assume away.

**Decisions:** Discrete weight-profile action space (Phase 8's 4 named
rules + uniform + interpolated blends, `|A|=9`) chosen over continuous
weighting, hard selection, or a hybrid — weighting-only for the first
version, justified by the same rare-category-client risk Phase 8
already reasoned through. Reward = validation-F1 delta minus a fairness
penalty minus a stability penalty, never touching KDDTest+. Training
via many short pretraining episodes (fresh Dirichlet partitions), not a
single long run, since one 10-round run gives too few transitions.

**Implementation outcome:** Design document only
(`11-phase-9b-drl-adaptive-fl-design.md`). No code written.

**Verification outcome:** Verified via direct code reads (not assumed
from docs); git status confirmed clean before and after, no Phase 1-9A
file touched.

---

## Entry 2 — Documentation system + second Phase 9B audit pass

**Date:** Same session block, subsequent turn.

**Task/prompt summary:** (a) Build a permanent development-log system
reconstructing Phases 0-9A from repository artifacts, explicitly
marking anything not recoverable rather than inventing it; (b) produce
a second, more detailed Phase 9B design pass answering a specific
20-question checklist plus a "recommended initial direction" review and
an explicit fairness-assumption check.

**What Claude investigated:** Full Git history (`git log`, `git show
--stat` per phase commit), every `docs/project-progress/*.md` file,
every `results/reports/*.md` file, `docs/research/claude-research/`
(found genuinely preserved prompt/response artifacts for the Phase 9
dataset-search step — the one real exception to "not preserved"), and
computed per-phase test counts via `git grep "def test_" <commit>` as a
real, reproducible measurement rather than a recollection.

**Important findings:** Phase 9A's independently-measured test count
(16, via actually running pytest) matched exactly the `git grep`-based
historical count (+16 over Phase 7-8's 168 → 184) — cross-validating
the counting methodology used for the whole historical log.

**Decisions:** Every development-log entry that could not recover an
original prompt/response says so explicitly
("Original chat prompt/response not preserved in repository;
reconstructed from available project artifacts") rather than
inventing content. Historical phase docs were read but never edited;
only new files were added.

**Implementation outcome:** `docs/research/development-log/` (README +
11 phase entries + master journal + prompt/response index policy);
`docs/project-progress/11-phase-9b-drl-adaptive-fl-design.md` (first
design pass, from Entry 1) left as-is.

**Verification outcome:** `git diff --stat` against every Phase 1-9A
path confirmed zero modifications; only new untracked files existed
at the end of the turn.

---

## Entry 3 — Repository audit + expanded Phase 9B design (this entry)

**Date:** Same session block, current turn.

**Task/prompt summary:** Re-audit git state (branch, status, recent
commits, Phase 9A commit integrity), then produce an expanded Phase 9B
design answering 20 specific numbered questions (RL environment
through ablation experiments), evaluate the user's own suggested
state/action/reward direction against the alternative already chosen
in Entry 1, explicitly restate the Phase 7/8 fairness-assumption
warning, and establish the full documentation structure requested
(`docs/project-progress/11-phase-9b-drl-adaptive-federated-learning.md`,
this experiments-log directory) — design only, no DQN code.

**What Claude investigated:** Fresh `git status`/`git log`/`git show
-s c58a8a2` to reconfirm nothing had changed since Entry 2; the
existing draft design doc and development-log to avoid re-deriving
already-established conclusions from scratch.

**Important findings:** The user's message stated the current branch
as `project-development-3`, but the actual current branch (created and
approved in the Entry 1 turn) is `project-development-4-drl`, with
identical history to `project-development-3` — flagged explicitly to
the user rather than silently proceeding on an assumed branch or
silently switching branches.

**Decisions:** Consolidated the two design passes into one canonical
document (`11-phase-9b-drl-adaptive-federated-learning.md`) rather than
maintaining two competing "current" designs — the earlier draft is
kept, not deleted, with a cross-reference note added at its top
(append-a-note convention, not a rewrite). Re-confirmed weighting-only,
discrete-profile-action, validation-only-reward design from Entry 1
after evaluating the user's alternative delta-based action suggestion
and finding the profile-catalogue approach more interpretable and
easier to learn.

**Implementation outcome:** This directory
(`docs/research/experiments/phase-9b-drl/README.md`);
`docs/project-progress/11-phase-9b-drl-adaptive-federated-learning.md`
(canonical, 22-section + 20-question design document); a superseded-note
added to the Entry 1 draft. No DQN code written.

**Verification outcome:** `git status`/`git diff --stat` confirmed no
Phase 1-9A file modified; Phase 9A commit `c58a8a2` reconfirmed intact
via `git show -s`.

---

## Entry 4 — Design correction pass (post first design checkpoint)

**Date:** 2026-08-15, same day as Entry 3, follow-up session.

**Task/prompt summary:** Apply exactly the four narrowly-scoped
corrections identified by `design-checkpoint-review.md` (reward
fairness signal, state dimension arithmetic, OPSSAT-AD separation
statement, artifact path convention) — no other design change — then
perform a second design check and record its outcome.

**What Claude investigated:** Re-read `scripts/evaluate_adaptive_fl_results.py`
lines 82-212 to confirm precisely how Phase 8's local-F1 fairness gap
is computed (offline, reads client parquet directly), and the repo's
actually-used (vs. placeholder) `experiments/`/`results/` naming
conventions, before choosing the corrected reward proxy and artifact
paths.

**Important findings:** Confirmed the corrected `perf_spread_t` proxy
requires no new computation — it reuses the exact per-round
`performance_k` signal Phase 8's `adaptive_trainer.py` already computes
against the server-owned global validation set.

**Decisions:** All four corrections applied directly to the canonical
design document (§8, §12, new §26, new §27); the true local-F1 fairness
gap retained as a post-hoc-only metric, never read by the reward.

**Implementation outcome:** Canonical design doc corrected;
`design-checkpoint-review-v2.md` created with the full correction
record and a second design check.

**Verification outcome:** Second design check (items A-J) all passed.
**Final decision: APPROVED FOR IMPLEMENTATION.** No Phase 1-9A file
modified; nothing committed/pushed.

---

## Entry 5 — Pilot implementation

**Date:** 2026-08-15, same day as Entry 4, follow-up session.

**Task/prompt summary:** Implement ONLY the Phase 9B pilot scope
(config, state construction, 9-action catalogue, action→weight
conversion, reward calculation, DQN forward pass/action selection,
privacy-boundary tests) — explicitly no replay buffer, no learning
loop, no multi-episode pretraining, no real-data experiment.

**What Claude investigated:** Re-confirmed `src/federated/adaptive.py`'s
exact `ClientSignals`/`compute_client_scores`/`scores_to_aggregation_weights`
API before writing `src/drl/action_space.py` (to reuse, not
reimplement, Phase 8's aggregation pipeline); re-read
`tests/federated/test_server.py`'s AST-based privacy test to mirror its
exact pattern for the new `tests/drl/test_privacy_boundary.py`.

**Important findings:** An early draft of a "no literal partition path"
privacy test flagged `reward.py`'s own explanatory docstring (which
mentioned `train.parquet` in prose) as a false positive — not a real
privacy issue (no executable code path was involved), fixed by
rewording the docstring rather than weakening the test.

**Decisions:** No replay buffer or DQN learning-update implemented this
turn (not required by the pilot's own test list); `lambda_fair=0.1`,
`lambda_stab=0.05` recorded in `configs/drl.yaml` as documented
starting defaults, explicitly not tuned against any result.

**Implementation outcome:** `configs/drl.yaml` filled in;
`src/drl/{action_space,state,reward,dqn_agent,config}.py`;
`tests/drl/` (46 tests); `scripts/run_drl_pilot.py` plus its generated
artifacts (`experiments/drl_adaptive_fl/pilot/pilot_verification.json`,
`results/plots/drl_adaptive_fl_pilot/action_profile_weights.png`,
`results/reports/drl_adaptive_fl_pilot.md`);
`docs/project-progress/12-phase-9b-drl-pilot.md`.

**Verification outcome:** `tests/drl/` 46/46 passed; full suite
`pytest tests` 237/237 passed (191 pre-existing + 46 new, zero
regressions); `git diff --stat` against every Phase 1-9A path empty.
Nothing committed or pushed.

---

## Entry 6 — Pilot extension: replay buffer, target network, environment, controller

**Date:** 2026-08-15, same day as Entry 5, follow-up session.

**Task/prompt summary:** Extend the pilot (still pilot scope, not the
full pretraining budget) with a replay buffer, a DQN target network and
real `learn()` step, an environment `reset()/step()` abstraction
(`src/federated/drl_environment.py`), and a `DQNController`
(`src/federated/dqn_controller.py`) combining agent + buffer + epsilon
schedule — plus a genuinely tiny (3 episodes × 3 rounds) deterministic
rollout, entirely on synthetic data, with determinism explicitly
verified by running it twice and diffing the output.

**What Claude investigated:** Re-confirmed the current git branch/HEAD/
status matched exactly where Entry 5 left off (no drift); re-read
`tests/federated/test_server.py`'s AST privacy-test pattern once more
to extend it correctly to the two new `src/federated/` modules.

**Important findings:** An initial version of the target-network-sync
test made a false assumption about WHEN the main network would have
diverged from the target relative to the configured sync frequency,
given the tiny pilot's `replay_warmup`/`target_update_frequency`
values — the test's own premise, not the implementation, was wrong;
fixed by explicitly forcing divergence before testing the sync trigger
in isolation.

**Decisions:** The environment (`DRLFederatedEnvironment`) takes an
injected `round_provider` callable rather than calling
`FederatedClient`/`FederatedServer` itself — this keeps it testable on
pure synthetic fixtures now, and reusable against real federated
training later, without ever needing to change once a real
`round_provider` is written (not yet implemented; recorded as the
explicit next step).

**Implementation outcome:** `src/drl/replay_buffer.py`;
`src/drl/dqn_agent.py` extended (target network + `learn()`);
`src/federated/{drl_environment,dqn_controller}.py`;
`scripts/run_drl_pilot_rollout.py` plus its generated artifacts
(`experiments/drl_adaptive_fl/pilot_run/rollout_summary.json`,
`results/plots/drl_adaptive_fl_pilot/pilot_rollout_rewards.png`);
31 new tests; `docs/project-progress/12-phase-9b-drl-pilot.md` extended
with an addendum (original first-pass content preserved unchanged);
`docs/project-progress/11-phase-9b-drl-adaptive-federated-learning.md`
given an implementation-status addendum (no design section rewritten).

**Verification outcome:** New/extended tests 77/77 passed; full suite
268/268 passed (237 prior + 31 new, zero regressions); tiny rollout
(9 environment steps, 5 learn() steps) executed successfully and
reproduced byte-for-byte identically on a second run; `git diff --stat`
against every Phase 1-9A path empty. No design/scientific problem was
found — nothing was silently redesigned. Nothing committed or pushed.

---

## Entry 7 — Aggregation-correctness integration test (re-confirmation pass)

**Date:** 2026-08-15, same day as Entry 6, follow-up session.

**Task/prompt summary:** Re-confirm the Phase 9B pilot against the same
approved design and an explicit 14-item pilot checklist (state, action,
reward, aggregation, privacy, determinism, staging safety, Phase 1-9A
integrity). Design, state, action, reward, and DQN were unchanged from
Entries 5-6 and re-verified against the canonical document — no
redesign occurred.

**What Claude investigated:** Re-read the canonical design doc and both
review checkpoints to re-confirm no drift; checked whether every one of
the 14 pilot-checklist items had a DEDICATED test — found that item 10
("FedAvg/adaptive aggregation remains mathematically valid") was only
indirectly covered (`action_to_aggregation_weights` was tested in
isolation for validity, but nothing proved those weights, once handed
to the REAL `FederatedServer.aggregate_with_weights`, produce a
mathematically correct weighted average of actual model parameters).

**Important findings:** This was a real test-coverage gap, not a code
defect — `action_to_aggregation_weights` and
`FederatedServer.aggregate_with_weights` were each already correct and
already tested independently (Entries 1-6; Phase 6/8's own tests), but
nothing had yet exercised them TOGETHER end-to-end.

**Decisions:** Closed the gap with a new, dedicated integration test
file rather than modifying either existing module — both remain
exactly as approved/reused.

**Implementation outcome:**
`tests/federated/test_drl_aggregation_integration.py` (3 tests): DQN
action weights, for all 9 actions, fed into the real, unmodified
`FederatedServer.aggregate_with_weights` produce a result matching a
hand-computed weighted average (`atol=1e-6`); every action's weight
dict covers all 8 client IDs (no client silently droppable); the
server's round counter increments correctly after a DRL-driven
aggregation call.

**Verification outcome:** New tests 3/3 passed; combined Phase 9B pilot
tests 80/80 passed; full suite **271/271 passed** (268 prior + 3 new,
zero regressions). `git diff --stat` against every Phase 1-9A path
still empty. Nothing committed or pushed.

---

## Entry 8 — Full pilot reconfirmation pass (no new design/code needed)

**Date:** 2026-08-15, same day as Entry 7, follow-up session.

**Task/prompt summary:** A fresh, from-scratch-styled request to
implement "the Phase 9B pilot" against the same approved design and the
same 15-item pilot-capability / 15-item test checklist already
satisfied by Entries 1-7. Followed the requested STEP 1-15 order as a
verification pass rather than re-implementing anything already done.

**What Claude investigated:** Re-ran (not just re-read) the full
focused Phase 9B pilot suite and the full repository test suite fresh,
re-ran both pilot scripts (`run_drl_pilot.py`,
`run_drl_pilot_rollout.py`) fresh, and re-checked `git diff --stat`
against every historical Phase 1-9A path.

**Important findings:** Re-running `scripts/run_drl_pilot.py`
regenerates `results/reports/drl_adaptive_fl_pilot.md` from scratch
(`Path.write_text`, not append), which silently erased the
hand-written "tiny deterministic pilot ROLLOUT" addendum section added
in Entry 6 — a real, reproducible finding about this project's own
tooling (not a bug that needed fixing, but worth recording so it isn't
mistaken for lost work next time).

**Decisions:** Restored the erased addendum by hand, with an explicit
note inside the file itself explaining why it can disappear and that it
must be re-added if `run_drl_pilot.py` is rerun — rather than
"fixing" `write_report()` to preserve it (out of scope for a pilot
reconfirmation pass; a real fix, if wanted later, would involve either
having `run_drl_pilot.py` also regenerate the rollout section from a
saved rollout summary, or splitting the two scripts' reports into
separate files — noted as a possible future cleanup, not done now).

**Implementation outcome:** No new source files. One documentation
restoration (`results/reports/drl_adaptive_fl_pilot.md`'s rollout
addendum, re-added). This log entry.

**Verification outcome:** Focused Phase 9B suite re-run fresh — 80/80
passed. Full suite re-run fresh — 271/271 passed, zero regressions.
Both pilot scripts re-run fresh and succeeded; the rollout script's
output (episode rewards -0.1686/-0.1536/-0.1650) matched every prior
run exactly. `git diff --stat` against every Phase 1-9A path confirmed
empty. Nothing committed or pushed. No scientific/design problem was
found; the approved design was not altered.

---

## Entry 9 — Re-verification against an explicit protected-file list

**Date:** 2026-08-15, same day as Entry 8, follow-up session.

**Task/prompt summary:** Another from-scratch-styled Phase 9B pilot
implementation request, this time additionally naming an explicit list
of protected Phase 6/7/8/9A artifacts
(`results/reports/federated_results.md`, `results/models/federated/`,
`results/plots/federated/`, `results/reports/non_iid_results.md`,
`results/reports/adaptive_fl_results.md`,
`results/reports/opssat_ad_results.md`) to check were unchanged, with
an instruction to STOP and report if any had changed.

**What Claude investigated:** Confirmed branch/HEAD unchanged
(`project-development-4-drl` @ `c58a8a2`); ran
`git diff --stat`/`git status --short` scoped to exactly the named
protected-file list (not the broader historical-path check used in
prior entries) — both returned empty; re-ran the focused Phase 9B
pilot suite together with the directly relevant Phase 6/8 tests
(`tests/federated/test_{adaptive,fedavg,server}.py`) and the full
repository suite.

**Important findings:** Every capability this turn's message requested
(state/action/reward/environment/replay-buffer/privacy-boundary/
determinism/config structure, all 15 named test categories) was already
implemented and tested in Entries 1-8 — no gap was found this pass, so
no new source file was created.

**Decisions:** Did not re-implement or duplicate anything already
present, per the project's own "do not create a second incompatible
implementation" rule extended to the DRL pilot's own code — re-running
and re-verifying was judged the scientifically honest response to a
repeated request whose scope was already fully satisfied, rather than
generating redundant files to appear more responsive.

**Implementation outcome:** No new source files. This log entry only.

**Verification outcome:** Named protected-file list — confirmed
untouched (`git diff --stat` and `git status --short`, both empty).
Focused pilot + relevant Phase 6/8 tests — **110/110 passed**. Full
suite — **271/271 passed**, zero regressions. Nothing staged,
committed, or pushed.

---

## Entry 10 — Full DQN pretraining request: STOPPED at pre-flight audit

**Date:** 2026-08-15, same day as Entry 9, follow-up session.

**Task/prompt summary:** A request to run the full Phase 9B DQN
pretraining experiment against real NSL-KDD data (multiple episodes,
10-round FL protocol, 8 real clients), then baseline comparison,
fairness analysis, ablations, plots, and a final results report — with
an explicit instruction to STOP if the approved full-training budget
is not actually defined in the canonical design/config, or if
reproducibility/real-data requirements cannot be met.

**What Claude investigated:** Re-verified branch (`project-development-4-drl`)
and HEAD (`c58a8a2`, unpushed on this branch — no remote tracking
branch exists for it); re-confirmed via `grep` that `configs/drl.yaml`'s
`pretraining:` block contains only `episodes_min/max` and
`rounds_per_episode_min/max` — a range, never a single approved number;
re-confirmed via `grep` that `FederatedClient`/`FederatedServer` are
referenced only in docstrings of `src/federated/drl_environment.py`/
`dqn_controller.py`, never imported or called — every existing
`round_provider` in the codebase (tests, pilot scripts) is synthetic;
ran the full protected-artifact check (Phase 6/7/8/9A reports,
plots, models, and all of `docs/project-progress/`) via
`git diff --stat` — empty; ran the focused Phase 9B + Phase 6/7/8
federated test suite (121/121 passed) and the full repository suite
(271/271 passed).

**Important findings:** Two of the prompt's own explicit CRITICAL STOP
CONDITIONS are triggered: (1) the requested full-training budget is
not actually defined as a single value anywhere in the canonical
design or `configs/drl.yaml` — only a range, explicitly left
uncalibrated in every prior pilot document; (2) no code exists yet
that trains the DQN environment against real `data/partitions/` via
the unmodified Phase 6/8 client/server — only synthetic
`round_provider`s have ever been built or tested, and building that
integration was consistently recorded across Entries 5-9 as "the next
step," requiring its own review, not something to build and
immediately run inside the same turn as a multi-episode real-data
experiment.

**Decisions:** Stopped before Part 4 (full DQN pretraining) and all
subsequent parts (baseline comparison, fairness analysis, ablations,
plots, final report) that depend on it — none of those were attempted,
none of their outputs were fabricated. Did not invent an episode/round
count, and did not build the real-data integration layer
unreviewed-and-immediately-run-at-scale, per the prompt's own explicit
prohibitions ("Do NOT invent a new training budget," "Do NOT 'fix' a
scientific problem by changing the experiment silently").

**Implementation outcome:** No new source files. This log entry only.
Parts 1-3 (pre-flight audit, protected-baseline verification, pilot/
regression test verification) were completed in full and are reported
in this session's response to the user.

**Verification outcome:** State dimension 44 (`STATE_DIM == 44`
assertion, source-verified), action count 9 (`NUM_ACTIONS`,
source-verified), reward formula matches the approved equation exactly
(source-verified line-by-line against `src/drl/reward.py`), weighting-
only confirmed (no selection/dropout code path exists), OPSSAT-AD
confirmed absent from all Phase 9B code (only one documentation
comment mentioning it, explicitly as an exclusion note). Protected
Phase 1-9A artifacts and all `docs/project-progress/` files confirmed
unmodified. 271/271 tests passed. Nothing staged, committed, or
pushed. **Full DQN pretraining was NOT run.**

---

## Entry 11 — Real NSL-KDD round provider implemented and validated

**Date:** 2026-08-15, follow-up session (continuing directly from the
end of Entry 10, where the pre-flight audit identified "no real
`round_provider`" as the exact remaining blocker).

**Starting branch/HEAD:** `project-development-4-drl` @ `c58a8a2`
(re-verified at the start of this session — unchanged; note: a prior
response in this same task was interrupted by an API connection error
after creating `src/federated/drl_round_provider.py` and updating
`configs/drl.yaml`; this session began by inspecting the filesystem
directly rather than assuming that work was complete or correct, per
explicit instruction).

**Purpose:** Close the one blocker Entry 10 identified — implement an
adapter wrapping the UNMODIFIED Phase 6/8 `FederatedClient`/
`FederatedServer` so `DRLFederatedEnvironment` can run against real
`data/partitions/` instead of only synthetic `round_provider`s — and
validate it with real data BEFORE any full DQN pretraining budget is
spent.

**Files inspected:** `src/federated/{client,server,protocol,adaptive,adaptive_trainer}.py`,
`configs/{federated,adaptive_fl}.yaml`, `scripts/run_adaptive_fl_experiments.py`
(the source of the `build_fairness_entropy` pattern, relocated not
reinvented), `src/federated/drl_environment.py`'s `RoundProvider`
contract, and the on-disk contents of `data/partitions/`,
`data/processed/`, `results/models/preprocessing/`.

**Implementation decisions:**
- `RealNSLKDDRoundProvider` is a stateful adapter class, not a new FL
  implementation — every local-training/aggregation call delegates to
  the exact, unmodified `FederatedClient.train_round` /
  `FederatedServer.aggregate_with_weights` / `get_global_state`.
- `build_fairness_entropy` was relocated (identical logic, not
  reimplemented differently) from `scripts/run_adaptive_fl_experiments.py`
  into `src/federated/drl_round_provider.py` so it's importable as a
  library function by both the script and the new provider, without
  importing a `scripts/` module.
- `configs/drl.yaml`'s `federation:` block was missing a `seed` key
  (needed by `FederatedClient.train_round`'s signature, which Phase
  6/8's own `training:` config blocks always included) — added
  `seed: 42`, matching Phase 6/8's convention exactly; this was the
  one gap found in the interrupted prior session's work.
- The provider caches each round's trained `ClientUpdate`s internally
  (`_pending_updates`) so `round_provider()` can aggregate the SAME
  updates the DQN's chosen weights apply to, without re-training or
  re-reading any client's data a second time.

**Privacy checks (all confirmed, source-verified, not assumed):** no
`pandas`/`pyarrow` import; no `read_parquet`/`read_csv`/`open()` call
in the module; no `pandas.DataFrame` construction; no literal
`test.parquet`/`data/processed/test` reference anywhere; the existing
`FederatedServer` privacy-boundary AST test still governs the
unmodified server code the provider calls.

**Tests added:** `tests/federated/test_drl_round_provider.py` — 17
tests (provider construction/determinism/error-handling, privacy
AST checks, all-8-clients participation, one real round's signal
validity/weight validity/finite-F1-output, cross-instance
determinism). One test bug found and fixed during this session: an
initial `test_module_never_references_kddtest` flagged the module's
own docstring sentence *disclosing* "KDDTest+ is never referenced" as
if it were a violation — a self-defeating false positive (the same
class of bug as an earlier `train.parquet`-in-docstring false positive
from Phase 9B's pilot sessions); fixed by checking the concrete file
path instead of the prose word.

**Real-data pilot configuration:** seed 42; model = Phase 6/8's exact
`BaselineMLP` config (121→[128,64]→1, dropout 0.3); training = local_epochs
1, batch_size 128, lr 0.001, Adam, BCEWithLogitsLoss; all 8 real
`SAT-01..08` partitions; real `data/processed/validation.parquet`;
2 rounds only (the minimum to exercise `round_provider`'s
"continue"/"terminate" branches once each); a FIXED action (index 3,
"combined" — Phase 8's own primary rule, not a DQN decision) used only
to exercise the interface, explicitly not a learned choice.

**Real-data pilot result (measured, one execution,
`scripts/verify_real_round_provider.py`):** initial (untrained) global
model validation F1 = **0.6190**. After round 1 aggregation (8 real
clients trained, "combined"-weighted): F1 = **0.9652**. After round 2:
F1 = **0.9725**. All 8 clients produced valid signals every round;
aggregation weights summed to 1.0 (within 1e-9) and were all
non-negative every round. These numbers are in the same plausible
range as Phase 6/8's own known trajectories (e.g. Phase 6 round 1 val
F1 = 0.9683) — a reassuring sanity signal, but explicitly NOT a
formal comparison and NOT a Phase 9B result (only 2 rounds, one fixed
non-learned action, no DQN involved).

**Timing (measured, not estimated):** provider initialization 0.067s;
round 1 (8 clients training + per-client validation eval) 6.229s;
round 1 aggregation + round 2 client training 4.798s; round 2
aggregation alone 0.182s; **total 11.276s** for init + 2 real rounds.
Separately, the full 17-test pytest file (which builds 3 total
provider instances and runs `initial_signals_provider`/`round_provider`
several times across its fixtures) measured 32-77s across two runs
(run-to-run variance observed, not investigated further — both figures
reported rather than picking the more favorable one).

**Problems encountered:** (1) the interrupted prior session's edits
needed re-verification rather than being assumed correct — done via
direct file inspection, confirmed complete and correct except the
missing config `seed` key (see below); (2) `configs/drl.yaml`'s
`federation:` block was missing `seed`, which would have caused a
`KeyError` the first time `FederatedClient.train_round` was actually
called with real training config; (3) the test-authoring false
positive described above.

**Fixes made:** added `federation.seed: 42` to `configs/drl.yaml`;
corrected the `test_module_never_references_kddtest` test.

**Files changed this session:** created
`src/federated/drl_round_provider.py` (in the interrupted prior
session, re-verified this session),
`tests/federated/test_drl_round_provider.py`,
`scripts/verify_real_round_provider.py`,
`experiments/drl_adaptive_fl/real_round_provider_verification/verification_summary.json`.
Modified `configs/drl.yaml` (added `federation.seed: 42`, in the
interrupted prior session). This log entry and (separately)
`docs/project-progress/12-phase-9b-drl-pilot.md`.

**Tests before/after:** before this session's changes: 271/271 (per
Entry 10). After: focused provider tests 17/17; Phase 9B pilot suite
80/80; relevant Phase 6/7/8 federated tests 41/41; full repository
suite **288/288 passed** (271 + 17 new), zero regressions.

**Integrity verification:** `git diff --stat` against every protected
Phase 6/7/8/9A artifact (`federated_results.md`,
`results/models/federated/`, `results/plots/federated/`,
`non_iid_results.md`, `results/plots/non_iid/`,
`adaptive_fl_results.md`, `results/plots/adaptive_fl/`,
`opssat_ad_results.md`, `results/plots/opssat_ad/`) and every
`docs/project-progress/*.md` file returned empty both before and after
implementation.

**Git status:** `configs/drl.yaml`, `src/drl/__init__.py` modified
(both from prior sessions, `configs/drl.yaml`'s new `seed` line from
the interrupted session re-confirmed present); all Phase 9B files
(prior sessions' + this session's new files) untracked. Nothing
staged, committed, or pushed.

**Exact next step:** the real round provider is now implemented and
validated against real data — the blocker Entry 10 identified is
closed. The exact full-training budget (episode count, rounds per
episode) is still NOT chosen, per explicit instruction — that
remains a separate decision for a future session, to be made
deliberately rather than inferred from this validation run.

---

## Entry 12 — THE FULL PHASE 9B DQN EXPERIMENT (real result)

**Date:** 2026-08-15, follow-up session (continuing directly from
Entry 11's real round provider).

**Prompt/goal:** Move out of the design/validation loop and run the
actual full Phase 9B DQN experiment: multi-episode pretraining against
real NSL-KDD data, a frozen-policy final run, one KDDTest+ evaluation,
post-hoc fairness, baseline comparison against stored Phase 6/8
results, and full documentation — explicitly instructed not to stop
for further minor design decisions, and to pick the lower/middle
already-approved config value where a range existed rather than pause.

**Starting branch/commit:** `project-development-4-drl` @ `c58a8a2`
(unchanged throughout — this entry records a real experiment run, not
a commit).

**Configuration (all pre-approved values; documented choices where the
config held a range):** episodes=50, rounds_per_episode=15 (lower
bound of the approved 50-80 / 15-20 ranges); replay_warmup=32
(=dqn.batch_size); epsilon_decay_steps=750 (=episodes×rounds, the
approved 1.0→0.05 schedule mechanically scaled to the real budget);
`λ_fair=0.1`, `λ_stab=0.05`, γ=0.95, lr=0.001, target_update_frequency=10
— all read directly from `configs/drl.yaml`, none retuned. FL protocol
(8 clients, real `data/partitions/`, real
`data/processed/validation.parquet`, Phase 2/6/8's exact MLP,
local_epochs=1, batch_size=128, lr=0.001, Adam, BCEWithLogitsLoss,
seed=42) identical to Phase 6/8.

**Decisions made this session:**
- Picked episodes=50/rounds=15 (the lower bound) rather than pause to
  ask, per explicit instruction.
- Reused the SAME fixed real partition for all 50 pretraining episodes
  (per this turn's explicit "do not create another partition"
  instruction) rather than implementing the original design's
  "many fresh partitions per episode" strategy — an intentional,
  disclosed simplification, not an oversight; recorded as a limitation
  in the results report (§9/§15/§16 of
  `results/reports/drl_adaptive_fl_results.md`).
- Extended `RealNSLKDDRoundProvider` with a non-breaking
  `round_eval_history`/`_evaluate_and_record_round` addition so the
  orchestration script could log full validation metrics
  (accuracy/precision/recall/ROC-AUC/loss) per round, not just the F1
  the DQN state/reward are allowed to use — the `RoundProvider`
  contract signature itself was not changed.

**Commands/scripts executed:**
1. A cheap smoke test (`smoke_test.py`, scratchpad only, tiny
   monkey-patched scope, never touching the real script's approved
   50/15 values) BEFORE committing to the real ~100-minute run.
2. `scripts/run_phase9b_full_experiment.py` (the real run, launched
   in the background).
3. `scripts/analyze_drl_full_run.py` (plots + summary stats, after
   the run completed).

**Problems encountered and fixes (both real, caught before/after the
long run, not hidden):**
1. **A segfault** during the smoke test: the orchestration script
   imported `matplotlib`/`torch` BEFORE the pandas-dependent
   `src/federated/*` modules — the exact ordering issue already
   documented in `scripts/evaluate_adaptive_fl_results.py`'s own
   comments about this Windows/Python 3.14 environment, which this
   session initially overlooked. Fixed by reordering imports (pandas-
   touching modules first) and re-running the smoke test clean BEFORE
   launching the real ~100-minute run — caught cheaply, not after
   burning the full compute budget.
2. **The first background-launch attempt failed instantly** (exit 1):
   the shell redirect target directory
   (`experiments/drl_adaptive_fl/full_run/`) didn't exist yet — bash
   redirection doesn't create parent directories. Fixed by creating
   the directory first, then relaunching.

**Training duration / runtime bottlenecks (measured):** pretraining
6088.3s (101.5 min) for 750 rounds (≈8.1s/round average, matching the
per-round cost measured in Entry 11's smaller validation run); final
10-round run 97.4s; total wall-clock 6189.7s (103.2 min). The dominant
cost is real local training for 8 clients + up to 9 validation
evaluations per round — no unexplained slowdown; timing matches
straightforward extrapolation from Entry 11's per-round measurement.

**Episode results (real, measured — full detail in
`results/reports/drl_adaptive_fl_results.md` §10):** episode reward
rose from −0.3572 (episode 1) to a mean of +0.1329 over the last 10
episodes (mean across all 50: −0.1414). DQN loss fell from 0.048 to
~0.0001. Action distribution shifted toward `combined` (62.7% of the
last 150 rounds) during near-greedy late pretraining, but the FROZEN
greedy final run instead selected `data_only` for 9 of its 10 rounds —
an inconsistency reported honestly, not resolved.

**Final validation result:** best round = 10 (val_loss 0.04105, val F1
0.9861), selected by validation loss alone, before any KDDTest+
access.

**Final KDDTest+ result (measured once):** Accuracy 0.7548, Precision
0.9231, Recall 0.6210, **F1 0.7425**, ROC-AUC 0.8361, FPR 0.0684,
confusion matrix `[[9047,664],[4864,7969]]`.

**Fairness result (post-hoc only, computed after model selection,
never used in training):** client F1 range 0.8148 (SAT-08) to 0.9982
(SAT-01), gap **0.1833**.

**Baseline comparison (stored Phase 6/8 numbers, not rerun):** DQN F1
0.7425 vs. Phase 6 FedAvg 0.7404 (DQN +0.0021), Phase 8 `resource_only`
0.7547 (DQN −0.0122, DQN did NOT beat Phase 8's best), Phase 8
`combined` 0.7439 (DQN −0.0014), Phase 8 `data_only` 0.7397 (DQN
+0.0028), Phase 8 `performance_only` 0.7447 (DQN −0.0022). DQN's ROC-AUC
(0.8361) is below Phase 6's (0.8479). DQN's fairness gap (0.1833) is
worse than Phase 8's `combined` (0.1496) and `performance_only`
(0.1245).

**Ablation:** the one minimum-required comparison — DQN vs. Phase 8's
best rule (`resource_only`) — DQN underperforms by 0.0122 F1. No
broader sweep was run, per explicit instruction to avoid an
open-ended search.

**Scientific conclusion (honest, not adjusted):** in this one real,
measured run, the DQN did **not** beat Phase 8's best rule-based
result on either F1 or fairness, and its trained (near-greedy,
mid-pretraining) behavior was not fully consistent with its frozen
final policy's behavior — most plausibly explained by the
single-fixed-partition training regime (§9 of the results report)
limiting the diversity of experience available to the controller. This
is reported as a valid, negative-relative-to-best-baseline scientific
result, not modified or rerun to seek a better number.

**Tests:** Phase 9B + federated tests 138/138 passed (before and after
the experiment); full suite **288/288 passed**, zero regressions,
both before launching and after the experiment completed.

**Artifacts generated:** `experiments/drl_adaptive_fl/full_run/`
(`pretraining_log.jsonl`, `final_run.json`, `kddtest_evaluation.json`,
`fairness_evaluation.json`, `timing.json`, `analysis_summary.json`,
`stdout.log`), 10 plots under `results/plots/drl_adaptive_fl/`,
`results/reports/drl_adaptive_fl_results.md` (full 20-section report).

**Files created this session:** `scripts/run_phase9b_full_experiment.py`,
`scripts/analyze_drl_full_run.py`, all files under
`experiments/drl_adaptive_fl/full_run/`, all 10 plots, the results
report, this log entry.

**Files modified this session:** `src/federated/drl_round_provider.py`
(non-breaking `round_eval_history` addition, described above).

**Integrity verification:** `git diff --stat` against every protected
Phase 6/7/8/9A artifact and `docs/project-progress/*.md` returned empty
both before launching and after the experiment completed. Real
`data/partitions/`/`data/processed/` files were only read, never
modified (verified).

**Next step:** await review of this real, measured, honestly-reported
result. Per the project's standing rule, no further tuning/rerunning
is planned unless explicitly requested — a worse-than-baseline result
is a valid stopping point for this experiment's primary report.

---

## Entry 13 — Post-hoc scientific analysis of the completed Phase 9B run

**Date:** 2026-08-15, follow-up session. **No training run, no code
change, no config change this entry — read-only analysis of the
artifacts Entry 12 already produced**, plus direct re-verification
against the stored Phase 6/8 report files.

**What was inspected:** `experiments/drl_adaptive_fl/full_run/{pretraining_log.jsonl,
final_run.json,kddtest_evaluation.json,fairness_evaluation.json,timing.json}`
in full (all 50 episode records, all 750 rounds); `results/reports/{federated_results,
adaptive_fl_results,non_iid_results}.md` (direct number-by-number
re-verification, not trusted from memory); `src/federated/dqn_controller.py`,
`src/drl/dqn_agent.py`, `src/drl/state.py`, `scripts/run_phase9b_full_experiment.py`
(line-by-line trace of the pretraining vs. final-run action-selection
code paths).

**Task 1 — verification result:** every headline number in the
completed run's artifacts matches what was reported in
`results/reports/drl_adaptive_fl_results.md` (Acc 0.7548, P 0.9231,
R 0.6210, F1 0.7425, ROC-AUC 0.8361, FPR 0.0684, fairness gap 0.1833).
Every Phase 6/8 comparison number matches the stored report files
exactly (Phase 6 F1 0.7404/ROC-AUC 0.8479; Phase 8 performance_only
0.7447, resource_only 0.7547, data_only 0.7397, combined 0.7439;
fairness gaps 0.1245/0.2259/0.2113/0.1496 respectively). **No
discrepancy found.**

**Task 2 — DQN training analysis (from `pretraining_log.jsonl`, all
750 rounds, 50 episodes, real data, no estimation):**

| Phase | Episodes | Mean total reward | Notes |
|---|---|---:|---|
| Early | 1-16 | −0.3191 | |
| Mid | 17-33 | −0.1808 | |
| Late | 34-50 | **+0.0652** | monotonic improvement across all three phases |

Epsilon: 0.9987 (ep1) → 0.8277 (ep10) → 0.5427 (ep25) → 0.2577 (ep40)
→ 0.0677 (ep50) — matches the configured linear 1.0→0.05 schedule
over 750 steps exactly. Replay buffer grew linearly (15→150→375→750),
never saturating its 5,000 capacity — every transition from the entire
run remained sampleable throughout, with no prioritization/decay.
DQN loss: 0.0480 (step 1) → 0.0001 (final), mean of last 50 steps
0.00011, stdev of last 50 steps 5.8e-5 — very low, very stable.

**Action distribution table (as requested):**

| Action | Early (1-16, n=240) | Mid (17-33, n=255) | Late (34-50, n=255) | Overall (n=750) |
|---|---:|---:|---:|---:|
| combined | 13.8% | 29.4% | **58.8%** | 34.4% |
| uniform | 11.7% | 11.4% | 16.1% | 13.1% |
| data_only | 12.5% | 12.2% | 8.2% | 10.9% |
| resource_leaning | 12.5% | 11.4% | 2.0% | 8.5% |
| fairness_leaning | 11.7% | 10.6% | 3.5% | 8.5% |
| balanced_performance_fairness | 10.4% | 8.2% | 2.7% | 7.1% |
| resource_only | 9.6% | 6.7% | 3.5% | 6.5% |
| performance_only | 10.0% | 4.7% | 3.5% | 6.0% |
| performance_leaning | 7.9% | 5.5% | 1.6% | 4.9% |

**Convergence**: yes, toward `combined` — a real, monotonic,
substantial trend (13.8%→29.4%→58.8%) across the three training
phases. This IS evidence of learned preference, though the "late"
window still includes real residual exploration (ε ranged 0.05-0.26
across episodes 34-50 — not zero), so this statistic is NOT a pure
measure of the greedy policy alone (see Task 3).

**Task 3 — the combined→data_only investigation.** Traced, not
assumed. Ruled OUT with certainty by direct code inspection:
- **Checkpoint/model loading**: `scripts/run_phase9b_full_experiment.py`'s
  `main()` passes the SAME in-memory `controller` object from
  `run_pretraining()` directly into `run_final_frozen_evaluation()` —
  no serialization/deserialization occurs anywhere. Ruled out.
- **Target network synchronization**: `DQNAgent.act()`/`q_values()`
  (`src/drl/dqn_agent.py` lines ~75-91) read `self.network` only —
  `self.target_network` is used exclusively inside `learn()` for TD
  targets, never for action selection. Ruled out.
- **Random seed / RNG**: the final run calls
  `controller.agent.act(state, epsilon=0.0, rng=zero_rng)`
  (script line 187) directly — `DQNAgent.act()` line 97:
  `if epsilon > 0.0 and rng.random() < epsilon` — since `epsilon=0.0`
  is not `> 0.0`, the random branch is never taken and `rng` is never
  even consumed. Ruled out.
- **Controller reset**: no reset occurs; same object, same trained
  weights, same buffer, same `total_steps` counter, throughout. Ruled
  out.
- **Action mapping**: `action_to_aggregation_weights` is the same
  pure, unit-tested function in both contexts. Ruled out as a source
  of a pretraining-vs-final-run INCONSISTENCY specifically (it would
  affect both contexts identically for a given action index).

**NOT ruled out — real, code-verified, disclosed candidates:**
1. **Residual exploration noise inflating the "late" statistic**: the
   "late pretraining" 58.8%-combined figure is computed over ALL
   actions taken at ε=0.05-0.26, not the pure greedy policy — a
   meaningful fraction of those 255 rounds were genuinely random
   draws, not the network's argmax choice. This is NOT an
   apples-to-apples comparison to the final run's 100%-greedy (ε=0)
   behavior.
2. **State normalization mismatch (`round_progress`)**:
   `scripts/run_phase9b_full_experiment.py` builds pretraining
   episodes with `num_rounds=ROUNDS_PER_EPISODE=15` (lines 113/117)
   but the final run with `num_rounds=FINAL_RUN_ROUNDS=10` (lines
   171/175); `src/drl/state.py` line 72:
   `components.append(float(round_number) / float(num_rounds))` — so
   for the SAME round position, the state's `round_progress` scalar
   differs numerically between the two contexts (e.g. round 10:
   10/15=0.667 during training vs. 10/10=1.0 in the final run). This
   is an intentional, disclosed design choice (training episodes must
   be long enough for learning; the final comparison run must match
   Phase 6/8's exact 10-round protocol) — but its effect on Q-argmax
   for states with closely-competing action values cannot be ruled
   out.
3. **Non-stationarity of online DQN training**: the network's weights
   changed continuously via `learn()` throughout pretraining (719
   total gradient steps); "late pretraining" (episodes 34-50) reflects
   a range of evolving weight states, not one fixed policy — the
   FINAL network (end of episode 50) is a further-trained version of
   what was active during most of the "late" window.
4. **Circumstantial supporting evidence for #1-3**: isolating just the
   LAST round of each episode (round 15) across all 50 episodes:
   `combined` was chosen 16/50 times (32%), `data_only` 9/50 times
   (18%) — `data_only` was a real, non-trivial "runner-up" during
   training, not an arbitrary or never-seen choice; and the final
   run's own round-1 choice (`performance_leaning`) was picked at
   round-1 only 1/50 times during the ENTIRE pretraining — consistent
   with a late-training/final-weight-state preference that differs
   from what dominated mid-training, not with an implementation error.

**Conclusion (per the requested A/B/C categories): definitively NOT
B** (no implementation bug found — five candidate mechanisms
individually traced and ruled out with code citations above). Best
characterized as **A** (expected behavior, given verified properties
of online DQN training and the intentional 15-vs-10-round protocol
difference) **for four of the five factors**, combined with **C**
(needs a controlled diagnostic to fully quantify) for the ONE
remaining open question: exactly how much of the shift is attributable
specifically to the `round_progress` normalization difference vs.
residual exploration/non-stationarity. This diagnostic was NOT run
this session (see Task 9 below — it requires a small prerequisite code
change, disclosed rather than made silently).

**Task 4 — reward analysis (evidence-based, from the same log):**
1. *Does maximizing this reward encourage higher validation F1?* Yes
   in principle (`ΔF1_t` is the only positive-reward-generating term),
   but with poor signal-to-noise: validation F1 was already 0.9654
   (mean round-1) to 0.9889 (mean round-15) — i.e. already
   near-saturated — leaving very little room for a meaningfully
   positive `ΔF1_t` most rounds, while the two penalty terms are
   subtracted whenever incurred. **Evidence**: 655 of 750 rounds
   (87.3%) had a NEGATIVE reward; mean reward across all rounds was
   −0.0094 (only the "late" phase averaged positive per-episode
   totals). This is a real, structural property of this reward on
   this near-saturated validation set, not a bug.
2. *Does it meaningfully encourage fairness?* **No clear evidence that
   it does.** Mean `perf_spread_t` showed no improving trend across
   training (ep1 mean 0.0512, ep25 0.0483, ep50 0.0517 — flat, not
   decreasing), and the final model's post-hoc fairness gap (0.1833)
   is worse than Phase 8's `combined` (0.1496) and `performance_only`
   (0.1245).
3. *Is the fairness signal sufficiently informative?* Uncertain from
   this one run — `perf_spread_t`'s flat trend could mean either the
   signal is too weak (`λ_fair=0.1` too small relative to `ΔF1_t`'s
   own small magnitude) or that performance spread across these 8
   clients is not very controllable by weighting-profile choice alone
   given how quickly all clients converge to high local F1 in this
   dataset.
4. *Can the reward be gamed?* Structurally, no obvious exploit was
   observed — an agent cannot gain reward by favoring/ignoring any
   specific client (the penalty is a symmetric spread measure, per the
   design's own argument), and no runaway or degenerate reward-seeking
   behavior appeared in the logs (bounded range: −0.0664 to +0.3159,
   no NaN/Inf).
5. *Does the reward explain why DQN did not outperform Phase 8
   resource_only?* Partially: the reward optimizes VALIDATION F1
   trend, not KDDTest+ generalization, and validation F1 barely
   differs across the 9 profiles on this dataset (see Task 5) — so
   even a "successfully optimized" policy under this reward has little
   basis to discover a KDDTest+-superior profile-selection strategy,
   since the reward signal itself carries little information
   distinguishing profiles' true test-time value.
6. *Are λ_fair/λ_stab reasonable given observed magnitudes?*
   **Observationally, `λ_stab=0.05` did not strongly suppress
   switching**: the action changed round-to-round in 438/700 (62.6%)
   of transitions despite the penalty. `λ_fair=0.1` applied to a
   perf_spread that stayed around 0.05 contributes a roughly
   constant ~0.005 drag, comparable in magnitude to typical `ΔF1_t`
   values late in an episode — plausible reason both fairness and
   stability failed to show strong learned effects. **No λ value was
   changed — this is an observation for a future, explicitly-approved
   diagnostic, not an action taken.**

**Task 5 — generalization analysis.** Validation F1 (0.9654-0.9889
within episodes) vs. KDDTest+ F1 (0.7425) is a large, real gap —
**but it is not unique to Phase 9B**: Phase 2's baseline already
established and documented this exact pattern (99.6% validation vs.
77.9% KDDTest+ F1), attributed to KDDTest+ containing attack types
absent from training (the benchmark's own documented, intentional
design). Phase 6 (98.79% val vs. 74.04% test) and every Phase 8 rule
(98.5-98.9% val vs. 73.97-75.47% test) show the SAME magnitude of gap.
**Phase 9B's val-test gap (98.61% vs. 74.25%) is fully consistent with
this established, dataset-level pattern — there is no evidence in this
run that Phase 9B's gap is larger, smaller, or qualitatively different
from Phase 6/8's own gaps.** No claim of "overfitting" is supported
beyond what every prior phase already documented as a known NSL-KDD
characteristic, not a Phase-9B-specific problem. Client
partitioning/non-IID distribution and preprocessing are UNCHANGED from
Phase 6/8 (same partition, same `data/processed/` artifacts), so
neither is a plausible NEW source of this gap. Model selection was by
validation loss only (never KDDTest+), matching Phase 6/8's protocol
exactly — no evidence of a model-selection-induced generalization
problem beyond what Phase 6/8 already carry.

**Task 6 — fairness analysis.**
1. DQN's fairness gap (0.1833) is **worse** than Phase 8's `combined`
   (0.1496) and `performance_only` (0.1245), better than
   `data_only`/`resource_only` (0.2113/0.2259). DQN did not improve
   fairness relative to Phase 8's best.
2. The reward's fairness penalty does not show clear evidence of
   affecting behavior (Task 4.2's flat perf_spread trend).
3. **One final run — not statistically interpretable as a robust
   estimate.** Phase 8's own fairness table is also single-seed per
   rule (same caveat already disclosed there).
4. **What can honestly be claimed**: "the DQN's client fairness gap in
   this one run (0.1833) fell within the range spanned by Phase 8's
   four rules but did not match its best (performance_only, 0.1245)" —
   nothing stronger.

**Task 7 — comparison table (all numbers from stored files, this
run's own artifacts, or explicitly marked unavailable):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | Fairness gap | Adaptive? |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Phase 6 FedAvg | 0.7531 | 0.9219 | 0.6186 | 0.7404 | 0.8479 | 0.0693 | not measured | No |
| Non-IID (alpha=0.5, = Phase 6 partition) | 0.7531 | 0.9219 | 0.6186 | 0.7404 | 0.8479 | 0.0693 | not measured | No |
| Phase 8 performance_only | 0.7563 | 0.9224 | 0.6244 | 0.7447 | 0.8497 | 0.0694 | 0.1245 | Rule-based |
| Phase 8 data_only | 0.7526 | 0.9223 | 0.6175 | 0.7397 | 0.8313 | 0.0688 | 0.2113 | Rule-based |
| Phase 8 resource_only | 0.7631 | 0.9190 | 0.6402 | **0.7547** | 0.8438 | 0.0746 | 0.2259 | Rule-based |
| Phase 8 combined | 0.7556 | 0.9217 | 0.6236 | 0.7439 | **0.8536** | 0.0700 | 0.1496 | Rule-based |
| **Phase 9B DQN** | 0.7548 | 0.9231 | 0.6210 | 0.7425 | 0.8361 | 0.0684 | 0.1833 | **Learned** |

**"Did DRL improve anomaly-detection performance?" — evidence-based
answer: No, not in this run.** DQN's F1 (0.7425) sits inside the tight
cluster of all six other methods (0.7397-0.7547, a 0.015 F1 range) —
essentially statistically indistinguishable from the middle of the
pack on a single-seed comparison, below Phase 8's best
(`resource_only`), and its ROC-AUC (0.8361) is the LOWEST of all seven
methods compared.

**Task 8 — the real, defensible contribution.** NOT "highest accuracy"
(it isn't). What this project's Phase 6-9B sequence actually
demonstrates, with real evidence: (a) a working, privacy-preserving,
non-IID federated learning pipeline (Phase 6/7) with a measured,
honest heterogeneity-fairness tradeoff (Phase 7's 41x fairness-gap
swing); (b) a rule-based adaptive-weighting framework (Phase 8) that
measurably beats plain FedAvg on SOME metrics via simple, transparent,
reasoned rules — with the honest, published finding that the
fairness-weighted rule does NOT win on fairness (a real, disclosed,
non-obvious result); (c) a from-scratch DQN controller (Phase 9B) that
integrates cleanly with that SAME infrastructure (weighting-only,
identical FL protocol, identical 4 signal families, unmodified
FedAvg/server/client code) and was rigorously, honestly evaluated
against every one of Phase 8's ablations under matched conditions —
producing a genuine, evidence-based NEGATIVE result relative to the
best fixed rule, with a well-reasoned, code-traced candidate
explanation (single-partition training limiting experience diversity,
disclosed in advance as a simplification, not discovered after the
fact as an excuse). **The legitimate claim: this project built and
rigorously, honestly compared THREE FUNDAMENTALLY DIFFERENT adaptive
federated-aggregation strategies (fixed sample-count weighting,
reasoned rule-based weighting, and learned RL-based weighting) under
matched, controlled conditions on real network-intrusion data across a
non-IID 8-client satellite simulation — and found that, in this
capstone-scale experiment, simple reasoned rules were at least as
good as (and on the best metric, better than) a learned policy, a
scientifically meaningful and honestly-reported finding in its own
right.**

**Task 9 — is another experiment justified?** **Primary
recommendation: A — no further experiment is required for the current
result to stand as complete and reportable.** The core comparison
(DQN vs. Phase 6/8, matched protocol, one honest run) is scientifically
valid and sufficient for the capstone's central claim, negative result
included.

**Optional, NOT required — one lightweight diagnostic (category B),
disclosed here for approval, not started:**
- **Hypothesis**: the `combined`→`data_only` inconsistency (Task 3) is
  driven more by the `round_progress` state-normalization difference
  (15-round vs. 10-round protocols) than by residual exploration
  noise, OR vice versa.
- **One variable changed**: add DQN network checkpoint-saving to
  `scripts/run_phase9b_full_experiment.py` (a small, disclosed code
  change — NOT made this session), then replay the ALREADY-LOGGED
  late-episode round-15 states through the frozen network twice: once
  with `round_progress` as originally encountered (round/15) and once
  recomputed as round/10.
- **Everything else held constant**: same trained network weights, same
  already-logged states, zero new FL training, zero new client
  training, zero KDDTest+ access.
- **Expected result**: if a meaningful fraction (e.g. >20%) of replayed
  states' argmax action flips between the two `round_progress`
  conventions, that supports the state-mismatch explanation; if few
  flip (<5%), the explanation shifts toward exploration
  noise/non-stationarity.
- **Success/failure criterion**: as stated above — a clear percentage
  threshold decided in advance, not adjusted after seeing the result.
- **Estimated runtime**: seconds to low minutes (pure offline forward
  passes on a tiny 44-input MLP) — but this REQUIRES the one small
  code change (checkpoint saving) and, since no checkpoint exists from
  the completed run, would need one additional short rerun to produce
  one to diagnose. **Not started. Requires your explicit approval per
  rule #14.**
- **Why scientifically necessary (if pursued)**: purely explanatory —
  would strengthen the paper's discussion section but does NOT change
  the primary, already-valid result. **Optional, not blocking.**

**No hyperparameter search, no repeated training for a better score,
and no category-D redesign is recommended.**

**Files inspected, not modified:** all files listed under "What was
inspected" above; zero source files, configs, or historical results
touched this session.

**Verification:** `git diff --stat` against every protected Phase
6/7/8/9A artifact and the just-completed Phase 9B artifacts confirms
this session made no code, config, or result changes — analysis only.

**Next step:** await explicit approval before pursuing the optional
diagnostic (Task 9), or move to documentation/writeup of the current,
already-valid result if no further investigation is wanted.

---

## Entry 14 — Improvement 01 design + pilot (per-episode local-training seed diversity)

**Date:** 2026-08-15, follow-up session. **Original Phase 9B result
(F1=0.7425) explicitly preserved — not touched this session.**

**Prompt/task:** Investigate whether the DRL component can be
scientifically improved: identify ONE well-justified limitation and
test ONE controlled improvement, via a pilot first, stopping before
any full run for explicit approval.

**Inspection performed (read-only):** re-confirmed (via `grep`) that
`configs/drl.yaml`'s `federation.seed: 42` (line 33) is threaded as
`training_config["seed"]` into every one of the original run's 750+10
`FederatedClient.train_round()` calls — a single constant across the
entire experiment, meaning every pretraining episode's local-training
stochasticity (dropout masks, minibatch order) was identical given the
same round/action, so the DQN's only source of inter-episode variation
was its own action sequence. This directly matches Entry 12/13's own
identified "most likely structural explanation" for the generalization/
policy-inconsistency findings.

**Candidates considered (A-H):** reward redesign, state-representation
change, DQN hyperparameter change, exploration-schedule change,
protocol-length matching (15-vs-10-round mismatch, Entry 13's other
open question), generalization-aware training (seed diversity),
multi-seed training, other. Evaluated each for justification/impact/
complexity/leakage-risk/cost — table recorded in the user-facing
response this turn (not duplicated here).

**Decision: selected "generalization-aware training strategy" —
per-episode local-training seed diversity** (`training_config["seed"]
= 42 + episode_number`, varied only across episodes, constant within
one episode's 15 rounds/8 clients). Rejected reward redesign (reopens
an approved, twice-reviewed design), state change (no supporting
evidence), hyperparameter search (explicitly disfavored), protocol
matching (a *different* single variable — deferred, not bundled),
multi-seed training (too expensive for "one controlled variable").

**Hypothesis (H1, pre-registered):** varying the local-training seed
per episode will increase training-experience diversity and may
improve the frozen policy's KDDTest+ generalization relative to the
original run.
- Independent variable: `training_config["seed"]` (episode-varied).
- Controlled: partition, `shared_init_seed=42`, architecture, all
  other hyperparameters, reward formula/lambdas, state/action design,
  episodes=50/rounds=15, final-run protocol (10 rounds, seed=42
  unchanged, ε=0, val-loss selection) — all IDENTICAL to the original
  run.
- Success (pre-registered): KDDTest+ F1 ≥ 0.7500 AND no core metric
  degrades >0.01 without offsetting benefit AND fairness gap ≤ 0.20.
- Failure (pre-registered): F1 ≤ 0.7450 OR any core metric degrades
  >0.01 without benefit OR fairness gap worsens >0.02.

**Implementation:** `scripts/run_phase9b_improvement_01.py` — a new
orchestration script; **zero changes to any `src/` module** (reuses
`DQNController`, `DRLFederatedEnvironment`, `build_real_round_provider`
by direct import identity, verified by a dedicated test). Writes only
to new, separate paths (`experiments/drl_adaptive_fl/improvement_01/`,
`results/plots/drl_adaptive_fl_improvement_01/`,
`results/reports/drl_adaptive_fl_improvement_01.md`) — the original
run's artifacts are never opened for writing.

**Tests added:** `tests/federated/test_drl_improvement_01.py` (8
tests) — seed formula determinism/distinctness, "only seed changes"
(every other config key equality-checked against
`configs/drl.yaml`'s stored values), shared-init-seed/budget match the
original run, final-run seed is NOT episode-varied (the manipulation
is pretraining-only, verified by source inspection), artifact-path
separation, no KDDTest+ path reference, no direct pandas import,
reused-infrastructure import-identity check. **8/8 passed.**

**Pilot (real data, tiny scope — 2 episodes x 2 rounds, monkey-patched,
the real script's approved 50/15 constants never touched):** confirmed
distinct per-episode seeds actually used (43, 44); confirmed a full
pretraining→frozen-final-run→KDDTest+→fairness pipeline executes
without error; confirmed original Phase 9B artifacts
(`pretraining_log.jsonl`, `kddtest_evaluation.json`,
`fairness_evaluation.json`, `results/reports/drl_adaptive_fl_results.md`)
are **byte-identical (SHA-256 hash match) before and after the pilot
ran** — the strongest available proof of non-interference.

**Problems encountered:** none this session — the pilot passed on the
first attempt (the import-order segfault fix and directory-creation
fix from Entry 12 were already carried over correctly into this new
script).

**Full test suite:** run after the pilot to confirm zero regressions
across the whole repository (result recorded in the user-facing
response this turn once the run completed).

**Interpretation:** the pilot demonstrates the ONE controlled variable
is correctly isolated (verified by both unit tests and real-data
execution) and the original result is provably preserved. This alone
does not test H1 itself — H1 can only be evaluated after the full
50x15 run, which was NOT started this session per explicit instruction.

**Decision (per the requested classification):** implementation and
pilot are complete and clean; **recommending the full run** (closest
to outcome A, "ready," though final classification of H1 itself is
necessarily deferred until real results exist) — but NOT started
without your explicit approval.

**Next step:** await approval to run
`python scripts/run_phase9b_improvement_01.py` (50 episodes x 15
rounds, estimated ~100-105 minutes based on the original run's
measured per-round timing).

---

## Entry 15 — Improvement 01 FULL RUN: H1 REJECTED (real, negative result)

**Date:** 2026-08-15, follow-up session. Approved and launched
immediately following Entry 14's pilot. **Original Phase 9B result
(F1=0.7425, fairness gap=0.1833) reconfirmed byte/content-identical
before and after this run — unchanged.**

**Prompt:** "Approved. Run the full Improvement 01 experiment exactly
as designed. Do not modify the experimental scope, do not access
KDDTest+ until the final frozen evaluation, and document every
result."

**Execution:** `scripts/run_phase9b_improvement_01.py` run exactly as
designed — no scope change, no config change, no KDDTest+ access until
the single final evaluation call (verified: the script's structure is
identical to the pilot-verified version; `_evaluate_on_parquet` against
`data/processed/test.parquet` is called exactly once, after
`run_final_frozen_evaluation` already selected the best round by
validation loss).

**Result vs. the pre-registered (Entry 14) criteria:**

| Criterion | Threshold | Measured | Outcome |
|---|---|---:|---|
| F1 success | ≥0.7500 | 0.7397 | not met |
| F1 failure | ≤0.7450 | 0.7397 | **triggered** |
| Fairness success | ≤0.20 | 0.2113 | not met |
| Fairness failure | worsens >0.02 vs 0.1833 | 0.2113 (+0.0279) | **triggered** |

**Both pre-registered failure conditions triggered — a clean, real,
non-ambiguous negative result, not adjusted or reinterpreted after the
fact.**

**Full metric comparison (Improvement 01 vs. original, both frozen,
real numbers):** Accuracy 0.7526 vs 0.7548 (−0.0022); Precision 0.9223
vs 0.9231 (−0.0008); Recall 0.6175 vs 0.6210 (−0.0035); F1 0.7397 vs
0.7425 (−0.0028); ROC-AUC 0.8313 vs 0.8361 (−0.0048); FPR 0.0688 vs
0.0684 (+0.0004); fairness gap 0.2113 vs 0.1833 (+0.0279, worse).
Every metric moved unfavorably.

**Evidence the manipulation had a real, non-trivial effect (i.e. this
was not a wasted/no-op run):** all 50 episodes used verified-distinct
local-training seeds (43-92); the late-training dominant action
differed markedly from the original run — `fairness_leaning` (52.9%
of late-phase rounds) here vs. `combined` (58.8%) in the original —
real evidence seed diversity changed what the policy converged toward
during exploration, even though it did not improve final generalization.

**A notable cross-run consistency finding:** despite that different
late-training convergence, BOTH experiments' frozen final runs
converged to `data_only` (10/10 rounds here, 9/10 in the original) —
weak but now twice-observed support for Entry 13's still-untested
hypothesis that the final run's 10-round `round_progress` state
context, not what dominated pretraining exploration, drives the
frozen greedy choice.

**Timing:** pretraining 7982.1s (133.0 min), final run 80.4s, total
8065.0s (134.4 min) — longer than the original run's 103.2 min;
plausibly session-to-session machine load variance, not investigated
further (not required to explain a negative result).

**Interpretation:** the single-fixed-trajectory limitation targeted by
H1, while real, does not appear to be the DOMINANT factor limiting
Phase 9B's KDDTest+ generalization — a more likely dominant factor
(untested by this experiment): the reward's weak signal-to-noise ratio
once validation F1 is near-saturated (86.0% negative rewards here,
87.3% in the original — essentially unchanged by seed diversity,
consistent with this explanation) and/or the round_progress protocol
mismatch (still not directly tested).

**Tests:** focused Phase 9B + federated tests 146/146 passed after the
run; full repository suite run in background (result to be confirmed
in the user-facing response this turn).

**Verification:** original run's `kddtest_evaluation.json` re-read
during this session's own comparison script and confirmed to still
report F1=0.7425 exactly (content-level proof of non-interference,
since `results/reports/drl_adaptive_fl_results.md` and the
`experiments/.../full_run/` JSON artifacts are `.gitignore`d and so
`git diff` cannot itself prove no change — content re-verification was
used instead, and matched exactly).

**Files created:** `results/reports/drl_adaptive_fl_improvement_01.md`
(full report — hypothesis, pre-registered criteria, full comparison,
interpretation, classification). `experiments/drl_adaptive_fl/improvement_01/`
artifacts were already created during Entry 14's pilot/launch; this
entry is the completed-run documentation.

**Final classification: B — improvement is not justified; the
original Phase 9B result stands as the project's Phase 9B result.**
This negative result is preserved, not deleted or hidden, per the
project's standing rule that a worse result is a valid scientific
outcome.

**Next step:** none required — H1 has been definitively tested and
rejected against pre-registered criteria. Any further Phase 9B
investigation (e.g. the round_progress-mismatch diagnostic named in
Entry 13, or a reward-redesign candidate, each as its OWN
single-variable, separately-approved experiment) would need a fresh,
explicit request — no experiment is currently planned or implied.
