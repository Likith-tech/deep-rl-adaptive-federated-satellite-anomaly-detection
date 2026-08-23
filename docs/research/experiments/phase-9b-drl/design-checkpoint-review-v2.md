# Phase 9B — Design Checkpoint Review v2 (post-correction)

## 1. Date

2026-08-15 (same day as the first review, follow-up session — no other
work occurred between the two reviews).

## 2. Repository / branch

Branch `project-development-4-drl`, HEAD `c58a8a2` ("Phase 9A:
OPSSAT-AD real spacecraft validation") — unchanged since the first
review. `git status` before and after this session's edits: only
documentation files untracked, nothing staged/committed/pushed, zero
modification to any Phase 1-9A tracked file.

## 3. What was corrected, and why

All four items from
`docs/research/experiments/phase-9b-drl/design-checkpoint-review.md`'s
§6-§7 were corrected directly in the canonical document
(`docs/project-progress/11-phase-9b-drl-adaptive-federated-learning.md`).
Nothing else in that document was touched — the action space, DQN
architecture, training strategy, experiment matrix, and research
question are all unchanged from the first review, which already found
them sound.

### Correction 1 — reward fairness signal (the substantive issue)

**What was wrong:** §8 used `fairness_gap_t = max_k(local_F1_k) −
min_k(local_F1_k)` — Phase 8's own-local-data fairness metric — as a
PER-ROUND reward term. That metric is only ever computed by
`scripts/evaluate_adaptive_fl_results.py`, an offline script that
reads every client's `train.parquet` directly (confirmed in code,
lines 204-206: `client_parquet = PARTITIONS_DIR / cid / "train.parquet"`,
then `evaluate_on_parquet(...)`), entirely outside the live
server/client privacy boundary. No code anywhere in Phase 6/7/8
evaluates the in-progress global model against client-local data
DURING training.

**Why it matters (privacy implication):** if implemented literally, the
reward function — invoked up to ~1,600 times across the design's own
stated pretraining budget — would need to read client parquet files
every round, either server-side (a direct violation of the boundary
`src/federated/server.py`'s existing AST-based test enforces) or via an
undocumented client-protocol extension. Either path would have
introduced a real privacy regression that did not exist anywhere in
Phase 6/7/8.

**Exact final reward formulation** (now in §8 of the canonical
document):

```
r_t = ΔF1_t − λ_fair · perf_spread_t − λ_stab · 1[action_t ≠ action_{t-1}]

ΔF1_t         = val_F1_t − val_F1_{t-1}
perf_spread_t = max_k(performance_k,t) − min_k(performance_k,t)
1[...]        = 1 if the chosen profile changed this round, else 0
```

`performance_k,t` is the EXISTING per-round signal (§6) — each client's
just-trained update evaluated against the server-owned global
validation set — already computed by the unmodified
`adaptive_trainer.py` evaluation step, with no new computation and no
new data access.

**Distinction now explicit in the document:**
- **(a) reward-time fairness proxy** = `perf_spread_t`, computed every
  round, server-side, from global-validation-only signals.
- **(b) final fairness-gap evaluation** = Phase 8's exact
  `max_k(local_F1_k) − min_k(local_F1_k)`, computed ONCE, post-hoc,
  after training, exactly as Phase 8 computes it — reported in the
  final comparison table, never read by the reward or the DQN.

No new data source and no hidden protocol were invented — the fix uses
only signals that already exist and are already inside the tested
boundary.

### Correction 2 — state dimension arithmetic

**What was wrong:** §12 stated "≈43-dim" while §6's own listed
components summed to 44.

**Final 44-dimensional state breakdown** (verified by direct
arithmetic, now in §12):

| Component | Dimensions |
|---|---:|
| `performance_k` (8 clients) | 8 |
| `data_k` (8 clients) | 8 |
| `resource_k` (8 clients) | 8 |
| `fairness_k` (8 clients) | 8 |
| `val_F1_t`, `val_F1_{t-1}` | 2 |
| `round / num_rounds` | 1 |
| `previous_action` (one-hot, `\|A\|=9`) | 9 |
| **Total** | **44** |

`8×4 + 2 + 1 + 9 = 32 + 2 + 1 + 9 = 44`. The intended state
representation itself was not changed — only the previously-incorrect
total was corrected, and the formula is now stated symbolically
(`8×4 + 2 + 1 + |A|`) so it remains correct if `|A|` changes later.

### Correction 3 — OPSSAT-AD separation

Added as new §26 in the canonical document, restoring (not
re-reasoning) the equivalent statement that existed in the superseded
first-pass draft (`11-phase-9b-drl-adaptive-fl-design.md`) but was
dropped during consolidation: OPSSAT-AD is Track B / real-spacecraft
validation, entirely separate from Phase 9B's FL/DQN pipeline; NSL-KDD
remains the only dataset used for training/evaluation, identically to
Phase 6/7/8; OPSSAT-AD's metrics are never combined with Track A's.

### Correction 4 — artifact path convention

Added as new §27. Inspected the repository's actually-used
conventions (not just the empty Phase-0-skeleton placeholders
`experiments/drl/`, `experiments/fedavg/`, `experiments/adaptive/`,
which were never populated) and found Phase 6 uses
`experiments/federated/` + `results/models/federated/`, Phase 8 uses
`experiments/adaptive_fl/<rule>/` + `results/plots/adaptive_fl/` +
`results/reports/adaptive_fl_results.md`. Phase 9B adopts the same
actually-used `adaptive_fl`-style pattern:

```
experiments/drl_adaptive_fl/
results/models/drl_adaptive_fl/round_history.json   (committed)
results/models/drl_adaptive_fl/*.pt                  (gitignored, like all *.pt)
results/plots/drl_adaptive_fl/*.png                  (committed)
results/reports/drl_adaptive_fl_results.md           (committed)
configs/drl.yaml                                     (filled in — currently null placeholder)
```

## 4. Second design check

**A. Every reward component computable without server access to
client raw data?** Yes. `ΔF1_t` uses only `val_F1_t`/`val_F1_{t-1}`
(global validation, server-owned). `perf_spread_t` uses only
`performance_k,t`, which is itself computed by loading a client's
returned `state_dict` (not raw data — `ClientUpdate`, Phase 6's
existing typed shape) into an eval model and scoring it against the
server-owned global validation set — the exact same computation
`adaptive_trainer.py` already performs every round for Phase 8's
`performance` signal. The stability term reads only the previous
action index. No term reads a parquet file, a client's local dataset,
or KDDTest+.

**B. Privacy boundary intact?** Yes. No new read of client-local data
was introduced; `server.py` is unmodified; the AST-based
no-pandas/no-parquet test that currently protects `server.py` continues
to apply unchanged, and the design document now commits to extending
that same style of test to the new `environment.py`/`drl_trainer.py`
modules (Test Plan, unchanged from the first review).

**C. State dimension exactly 44?** Yes — verified by direct arithmetic
in §3 above and now stated correctly (both numerically and
symbolically) in the canonical document's §12.

**D. Action space still the discrete 9-profile weighting catalogue?**
Yes — §7 of the canonical document was not touched by any of the four
corrections. No continuous action was introduced.

**E. Weighting-only, no selection?** Yes — §7's weighting-vs-selection
reasoning was not touched. All 8 clients still participate every
round; the DQN only ever changes the aggregation weighting, never
excludes a client.

**F. Phase 6 and Phase 8 remain valid baselines?** Yes — §4, §14, §20,
§21 (baseline numbers, fixed experimental protocol, comparison targets)
were not touched by any correction, and no Phase 6/7/8 file, config, or
result was read as anything other than a fixed, already-committed
reference number. `git diff --stat` against every Phase 1-9A path
confirms zero modification (see §6 below).

**G. OPSSAT-AD remains separate?** Yes — now explicitly stated in new
§26, closing the gap the first review found. NSL-KDD remains the only
dataset in the Phase 9B pipeline.

**H. Experiment remains reproducible?** Yes — seed 42 usage (§14,
"Reproducibility information") is unchanged; the new artifact-path
convention (§27) makes reproducibility easier to verify in practice
(clear, predictable output locations) without changing any seed or
hyperparameter.

**I. Reward cannot trivially be gamed by ignoring rare clients?** Yes.
`perf_spread_t` is a *penalty* on the spread of per-client validation
performance, not a reward for favoring any specific client's weight —
the agent can only gain reward by reducing that spread, which requires
genuinely balancing clients' contributions, not ignoring the
worst-performing (often rare-category) one. This argument is
structurally identical to the one already made for the original
local-F1-based term in the first design pass, and still holds for the
corrected proxy since both are spread/gap penalties over a per-client
performance vector — only the underlying performance measurement
source changed (global-validation-evaluated vs. local-data-evaluated),
not the penalty's shape or incentive structure.

**J. Final local-F1 fairness gap clearly labeled as post-hoc, not an RL
training signal?** Yes — §8(b) and the updated §25 (Experimental
design) both now explicitly state the true own-local-data fairness gap
is computed once, post-hoc, and is never read by the reward or the DQN
— with an explicit parenthetical in §25 warning against reporting the
reward-time proxy as "the" fairness result.

## 5. Final decision

**APPROVED FOR IMPLEMENTATION**

All four issues from the first checkpoint review are resolved:
substantively (Issue 1 — the reward no longer requires any privacy
boundary violation or undocumented protocol extension) and
editorially (Issues 2-4 — arithmetic, OPSSAT-AD statement, artifact
paths). All ten items (A-J) of this second design check pass. No
further design revision is identified. The core architecture (state,
discrete 9-profile action space, corrected reward, small DQN, weighting-
only mechanism, multi-episode pretraining strategy, fixed 3-way
comparison protocol) is internally consistent with Phase 6/7/8/9A and
ready for implementation to begin.

## 6. Git status at time of this review

```
On branch project-development-4-drl
Untracked files:
  docs/project-progress/11-phase-9b-drl-adaptive-federated-learning.md
  docs/project-progress/11-phase-9b-drl-adaptive-fl-design.md
  docs/research/development-log/
  docs/research/experiments/
nothing added to commit but untracked files present
```

`git diff --stat` against every Phase 1-9A path (src/, configs/,
results/, all historical docs/project-progress/*.md,
docs/research/datasets/, docs/research/decisions/) returns empty — no
historical file touched. No DQN code, replay buffer, RL environment
code, training script, experiment, or model checkpoint exists anywhere
in the working tree.

## 7. Exact next step

Begin implementation, following the order already specified in the
canonical document's "Next steps" / recommended implementation order
(configs/drl.yaml → replay buffer → DQN agent → environment/action
space → integration trainer → pretraining script → final-run script →
evaluation script → tests → results write-up), starting with the
PILOT scope defined in §25 only (correctness checks: state shape,
action validity, reward formula, DQN trains without error, aggregation
correctness, privacy-boundary test, reproducibility) — not the full
50-80-episode pretraining run — and returning for review before
proceeding past the pilot. No commit, no push, no DQN code was written
in this session; both remain pending explicit approval.
