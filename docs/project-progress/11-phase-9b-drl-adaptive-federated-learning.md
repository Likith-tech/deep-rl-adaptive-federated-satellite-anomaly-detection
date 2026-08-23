# Phase 9B — DRL (DQN) Adaptive Federated Learning

**Original status (design phase): DESIGN ONLY. No DQN code had been
implemented, no experiment had been run, no Phase 6/7/8/9A file or
result had been modified.** This is the canonical Phase 9B research
log (see `docs/project-progress/11-phase-9b-drl-adaptive-fl-design.md`
for the earlier first-pass draft this consolidates and extends —
preserved unchanged, not rewritten).

**Implementation status addendum (added 2026-08-15, after design
approval — see
`docs/research/experiments/phase-9b-drl/design-checkpoint-review-v2.md`,
"APPROVED FOR IMPLEMENTATION"):** the PILOT scope of this design (state
construction, the 9-action catalogue, the corrected reward, a DQN with
replay buffer + target network + a real `learn()` gradient step, and an
environment `reset()/step()` abstraction) has been implemented and
verified — see `docs/project-progress/12-phase-9b-drl-pilot.md` for the
full pilot write-up. **The full multi-episode pretraining budget and
the final 3-way comparison experiment (Phase 6 vs. Phase 8 vs. Phase
9B) have NOT been run.** This addendum records status only; no design
decision in the sections below was changed as a result of
implementation — where implementation required a concrete choice the
design left open (e.g. `lambda_fair`/`lambda_stab` numeric values), that
choice is recorded in `configs/drl.yaml` and in
`docs/project-progress/12-phase-9b-drl-pilot.md`, not silently folded
into this document's original design language.

**FULL EXPERIMENT COMPLETE (added 2026-08-15):** the real, full-budget
DQN experiment described by this design has now been run once
(50 episodes × 15 rounds pretraining — the lower bound of this
document's own approved range — against the real NSL-KDD partition via
`src/federated/drl_round_provider.py`, followed by a frozen-policy
10-round final run and one KDDTest+ evaluation). Full results,
including the honest comparison against Phase 6/8 (DQN did NOT beat
Phase 8's best rule on F1 or fairness in this run), are in
`results/reports/drl_adaptive_fl_results.md` and
`docs/research/experiments/phase-9b-drl/README.md` (Entry 12). As
before, this status note does not alter any design section below —
the design was implemented as approved; the result is reported
separately, honestly, exactly as measured.

## 1. Motivation

Phase 8 replaced Phase 6's sample-count-only FedAvg weighting with a
transparent, deterministic 4-signal rule (`performance`, `data`,
`resource`, `fairness`), but the weights combining those signals
(`0.40/0.25/0.15/0.20`) were fixed by hand-reasoning, not learned. Phase
8's own ablation results showed this matters: `resource_only` won on
KDDTest+ F1, while `performance_only` — not the fairness-weighted
`combined` rule — won on fairness. A fixed rule cannot adapt to that
kind of counterintuitive, round-varying reality. Phase 9B asks whether
a DQN controller can learn better adaptation from a reward signal
instead of guessing weights up front.

## 2. Research question

Can a DQN controller, observing the same four signal families Phase 8
already computes, learn an aggregation-weighting policy that matches or
improves on Phase 8's best rule-based result (on validation and,
separately, on the untouched KDDTest+ held-out set) — measured
honestly, including the possibility that it does not?

## 3. Hypothesis

A learned policy *may* adapt its weighting per round in ways a static
rule cannot (e.g. leaning on `performance` early and `fairness` later,
or reacting to a bad round differently than a good one) and so *may*
improve on Phase 8's `combined` result. This is a hypothesis to test,
not a target to engineer toward — per the project's standing rule, a
worse DQN result is still a valid, reportable result.

## 4. Baselines

- **Phase 6 FedAvg** (`results/reports/federated_results.md`): sample-count
  weighting, KDDTest+ F1 **0.7404**.
- **Phase 8 rule-based adaptive FL** (`results/reports/adaptive_fl_results.md`):
  best KDDTest+ F1 **0.7547** (`resource_only`); best fairness gap
  **0.1245** (`performance_only`); `combined` (the intended primary
  rule) scored F1 0.7439, fairness gap 0.1496.

Both are real, already-run, committed results — neither is rerun for
this design phase.

## 5. RL environment (Q1)

The environment IS one federated-learning round, from the DQN's point
of view: the controller observes the current round's state, chooses an
aggregation policy, that policy is applied through the **existing,
unmodified** `FederatedServer.aggregate_with_weights` (Phase 8), the
round's outcome is evaluated on the global validation set, and a reward
is computed. FedAvg itself is not replaced or reimplemented — the DQN
only replaces the *weight-computation step* Phase 8's
`compute_client_scores` currently performs on its own.

## 6. State (Q2, Q13, Q14, Q15)

A compact, server-side, normalized vector per round, built entirely
from signals Phase 8 already computes (no new information channel):

| Component | Size | Source |
|---|---|---|
| `performance_k` (this round's local-update F1 on global validation) | 8 | `adaptive_trainer.py`'s existing per-client eval step |
| `data_k` (min-max normalized local sample count) | 8 | `ClientUpdate.num_samples` via `adaptive.py:normalize` |
| `resource_k` (min-max normalized simulated resource desirability) | 8 | `adaptive.py:build_resource_signal`, Phase 4's simulated bandwidth/latency/compute/availability/connectivity |
| `fairness_k` (Shannon entropy of local category distribution) | 8 | `src/simulation/non_iid_metrics.py:shannon_entropy` — this is how non-IID-ness/diversity is represented (Q13, Q15): the exact same measured (not assumed) entropy signal Phase 7 used to characterize heterogeneity and Phase 8 used as its fairness signal |
| `val_F1_t`, `val_F1_{t-1}` | 2 | current validation performance + recent trend |
| `previous_action` (one-hot over the action catalogue) | `\|A\|` | last round's chosen policy, so the agent can learn stability |
| `round / num_rounds` | 1 | round progress, without leaking future rounds |

Resource availability (Q14) is represented exactly as Phase 8 already
represents it — no new simulation is introduced. Everything in this
vector is either this round's or a past round's information; nothing
from a future round or from KDDTest+ ever enters it.

## 7. Action (Q3, Q10, Q11, Q12)

**Recommended-direction evaluation:** the prompt's suggested
"increase/decrease performance/fairness/resource/data influence"
four-way discrete-delta action was considered and compared against a
**fixed catalogue of discrete weight profiles** (Phase 8's own four
named rules — `performance_only`, `resource_only`, `data_only`,
`combined` — plus a uniform profile and a few interpolated blends,
`|A|=9` total). The delta-based formulation requires the agent to
implicitly integrate deltas over many rounds to reach a useful
weighting, which is harder to learn and harder to interpret after the
fact ("why did it end up here" requires replaying the whole trajectory).
The fixed-catalogue formulation lets the agent pick a complete,
named, already-validated weighting directly each round, is trivially
interpretable (log the profile name chosen per round), and reuses
`compute_client_scores`/`scores_to_aggregation_weights` unchanged. **The
fixed weight-profile catalogue is the recommended action space.**

**How client weighting is produced (Q10):** the chosen profile's
`(w_perf, w_data, w_resource, w_fair)` is passed through the same
`compute_client_scores()` → `scores_to_aggregation_weights()` pipeline
Phase 8 already uses — the DQN decides *which* rule to apply this
round, not a new weighting mechanism.

**Weighting vs. selection (Q11, Q12):** the first version uses
**weighting only**, not selection, and not a hybrid. This is the same
reasoning Phase 8 already documented and this project is not
re-litigating without cause: hard selection risks silently dropping a
client holding a rare attack category for an entire round, which
directly conflicts with the fairness goal Phase 7 showed matters (a
41x fairness-gap swing from heterogeneity alone). Weighting keeps all 8
clients participating every round — scientifically safer for a first
DRL experiment because it isolates "did the agent learn a good
weighting policy" from "did the agent also learn not to accidentally
starve a client," two separable questions that a first, deliberately
small pilot should not conflate.

## 8. Reward (Q4, Q16)

**Corrected 2026-08-15, after the first design checkpoint review**
(`docs/research/experiments/phase-9b-drl/design-checkpoint-review.md`,
Issue 1). The original version of this section used
`fairness_gap_t = max_k(local_F1_k) − min_k(local_F1_k)` — Phase 8's
own-local-data fairness metric — as a PER-ROUND reward term. That
metric is real, but it is only ever computed by
`scripts/evaluate_adaptive_fl_results.py`, an offline, POST-HOC script
that reads every client's `train.parquet` directly, entirely outside
the live server/client privacy boundary. Nothing in Phase 6/7/8 (or
anywhere else in this repository) evaluates the in-progress global
model against client-local data DURING training — the only per-round,
in-training signal that exists is `performance_k`, evaluated against
the server-owned GLOBAL validation set. Using the local-F1 gap as a
live reward signal, computed up to ~1,600 times across the pretraining
budget, would have required the server (or the DQN) to read client
parquet files every round — a direct violation of the privacy boundary
`server.py`'s existing AST-based test enforces. This is corrected
below by distinguishing two clearly separate things that were
previously conflated:

**(a) Reward-time fairness proxy** — used INSIDE the reward, every
round, computed only from information already available server-side
without touching client-local data:

```
r_t = ΔF1_t − λ_fair · perf_spread_t − λ_stab · 1[action_t ≠ action_{t-1}]

ΔF1_t       = val_F1_t − val_F1_{t-1}                         (global validation set only)
perf_spread_t = max_k(performance_k,t) − min_k(performance_k,t)   (spread of the EXISTING
                per-round performance_k signal — §6 — each client's just-trained update
                evaluated against the server-owned global validation set; no client-local
                data read, no new computation, no protocol change)
1[...]       = 1 if the chosen profile changed this round, else 0
```

`λ_fair`, `λ_stab` are fixed constants in `configs/drl.yaml`, chosen
before any KDDTest+ evaluation.

**(b) Final fairness-gap evaluation** — Phase 8's exact own-local-data
`max_k(local_F1_k) − min_k(local_F1_k)` metric is KEPT, unchanged, but
ONLY as a POST-HOC metric computed once, after training, by the Phase
9B equivalent of `evaluate_adaptive_fl_results.py` (see "Artifact path
convention" below) — the same way Phase 8 itself only ever used it. It
is never read by the reward function, the DQN, or any code path that
executes during an RL round.

**Privacy implication of this correction:** with (a) and (b) clearly
separated, the reward function now has a strict, verifiable data
dependency: it reads only `val_F1_t`/`val_F1_{t-1}` (global validation)
and `performance_k,t` (already computed by the existing, unmodified
`adaptive_trainer.py` evaluation step against global validation) — both
already inside the tested privacy boundary. No new read of any client
partition file occurs at reward-computation time, and no client
protocol extension is introduced. The server/DQN still never accesses
raw client datasets — the boundary is not weakened, and no new or
hidden data source was invented.

**Reward-leakage prevention (Q16), reconfirmed:** both terms are built
exclusively from the global validation set and the per-round
`performance_k` signal — KDDTest+ is never loaded anywhere in the
training/environment code path, to be verified by a dedicated boundary
test (see Test Plan). This mirrors how `evaluate_opssat_temporal.py`
was kept structurally separate from `train_opssat_temporal.py` in
Phase 9A.

**Why the fairness proxy doesn't create undesirable behavior:**
`perf_spread_t` is a *penalty* on the spread between clients'
validation-evaluated performance, not a reward for high entropy or for
favoring any specific client — so the agent cannot maximize reward by
arbitrarily boosting a low-performing client's weight; it can only
gain reward by reducing the performance spread across clients, which
requires genuinely balancing them, not gaming a single component (see
§ "Second design check," item I, for the explicit rare-client-neglect
argument). It also does not dominate `ΔF1_t` by construction (`λ_fair`
kept small, calibrated during the pilot, not asserted a priori).

## 9. One RL timestep (Q5) and one episode (Q6)

**One timestep** = one federated-learning round: state → DQN picks an
action → clients train (unchanged `FederatedClient`) → server
aggregates with the resulting weights (unchanged
`aggregate_with_weights`) → global validation evaluation → reward →
next state.

**One episode** = one complete simulated FL run over a fixed round
budget (e.g. 15-20 rounds) on one client partition. Termination is
simply reaching that round budget — no early-stopping condition, since
FL rounds are cheap relative to the value of a full, comparably-sized
episode for replay.

## 10. How the DQN interacts with FedAvg (Q7)

The DQN never touches `FederatedServer`, `FederatedClient`, or
`fedavg.py` directly — it only decides which named weight profile is
handed to the **unmodified** `aggregate_with_weights`. FedAvg itself
(`federated_average`) and its Phase 8 generalization
(`weighted_average`) are not replaced, forked, or modified in any way.

## 11. Server/client information boundary (Q8, Q9)

**Available to the server (and therefore to the DQN, which is a
server-side controller):** per-client `ClientUpdate` (state_dict,
`num_samples`, training metadata — unchanged Phase 6 `protocol.py`
shape), the global validation set, simulated Phase 4 resource metadata,
each client's local category-distribution entropy (a scalar, computed
once per experiment, not raw data). **Must remain private to each
client:** the raw partition file itself
(`data/partitions/<id>/train.parquet`) — never read by the server,
never read by the DQN, never included in the state vector in any raw
form. This boundary is currently enforced for `server.py` by a real
AST-based test (`test_server_module_never_imports_pandas_or_touches_parquet`);
the same style of test will be extended to the new DRL modules (see
Test Plan) rather than merely asserted in prose.

## 12. DQN architecture

**Corrected 2026-08-15** (first design checkpoint review, Issue 2): the
state dimension was previously stated as "≈43-dim," which does not
match its own listed components. Verified by direct arithmetic against
§6's table:

```
performance_k   8
data_k          8
resource_k      8
fairness_k      8
                --   (= 8 × 4 = 32)
val_F1_t, val_F1_{t-1}          2
round / num_rounds              1
previous_action (one-hot, |A|)  9
                                --
Total = 32 + 2 + 1 + 9 = 44
```

Small MLP Q-network: input **exactly 44-dim** (symbolically,
`8×4 + 2 + 1 + |A|` where `|A|=9` — stated symbolically so the figure
stays correct if `|A|` changes during implementation), hidden 64→64
ReLU, output 9 (one Q-value per weight profile). Adam, lr 1e-3, γ=0.95,
replay buffer ~5,000 transitions, batch size 32, target network with
hard update every 10 learning steps, ε 1.0→0.05 decayed over the
pretraining budget, seed 42. No PPO, SAC, multi-agent RL, or
transformer components — a single, small, standard DQN, per the
explicit instruction to avoid unnecessary complexity. The intended
state representation itself is unchanged — only the stated total was
corrected to match it.

## 13. Training procedure

One 10-round FL run gives too few transitions (~10) for meaningful DQN
learning and conflates model convergence with agent convergence in a
single non-stationary run. Instead: many independent pretraining
**episodes** (fresh Dirichlet partitions, varied seed/alpha, reusing
`src/simulation/partitioner.py` exactly as Phase 7's alpha sweep did),
each ~15-20 rounds, for roughly 50-80 episodes total (~750-1,600
transitions with replay) — calibrated against this machine's actual
per-round timing before being locked in, not guessed. Every episode is
a real federated run using the unmodified client/server code, not a
proxy.

## 14. Experimental configuration

Held fixed for the **final comparison run** across all three methods
(Phase 6, Phase 8, Phase 9B): 8 clients, `data/partitions/` (Phase 4/6
exact partition), the Phase 2 MLP architecture, validation/test split,
preprocessing, 1 local epoch/round, batch size 128, learning rate
0.001, Adam, BCEWithLogitsLoss, shared init seed 42, 10 communication
rounds. The only intentionally varied factor is the
aggregation-weighting mechanism. Seed 42 used throughout, per the
stated constraint, including for the DQN's own initialization and the
pretraining episode generator (episode-specific partition seeds vary
deliberately, to generate diverse experience — the controller's own
network init and the final comparison run both use 42).

## 15. Data/privacy boundary

Restated concretely from §11: no client ever receives another client's
data; the server/DQN never reads a parquet file (to be verified by an
extended AST test); the reward and state never reference KDDTest+; no
future-round information appears in the state; hyperparameters
(`λ_fair`, `λ_stab`, DQN architecture/γ/lr/ε-schedule) are fixed in
`configs/drl.yaml` before any KDDTest+ evaluation, exactly as Phase 8's
rule weights were.

## 16. Pilot results

Not yet available — no code has been implemented. To be filled in once
the pilot (environment/state/action/reward/DQN-learns-without-error/
aggregation-correctness/privacy-boundary/reproducibility checks, per
the Experimental Design section below) actually runs.

## 17. Final results

Not yet available.

## 18. Ablations (Q20)

Planned, not yet run: (a) weight-profile catalogue granularity (9 vs. a
coarser/finer grid); (b) `λ_fair` sensitivity; (c) with vs. without the
stability penalty `λ_stab`; (d) pretraining episode budget (fewer vs.
more episodes) vs. final policy quality; (e) weighting-only (this
version) vs. a future selection or hybrid variant, once weighting alone
has been honestly evaluated.

## 19. Failure cases

Not yet available (no run yet). What "failure" will mean here, defined
in advance per the project's honesty rule: DQN final KDDTest+ F1 below
Phase 8's `combined` (0.7439) or below Phase 6 (0.7404); DQN fairness
gap worse than Phase 8's best (0.1245); the learned policy collapsing
to always choosing one profile (effectively re-deriving one of Phase
8's rules rather than learning something new) — any of these would be
reported and analyzed exactly as measured, not hidden or re-run to
look better (Q19: "what happens if DRL performs worse" — it gets
documented honestly, with an analysis of why, exactly as Phase 3's
rejected experiment and Phase 7's non-monotonic F1 finding were).

## 20. Comparison with Phase 6

Same held-fixed protocol as §14; Phase 6's committed numbers
(val F1 0.9879 round 10, KDDTest+ F1 0.7404) are the floor reference —
not rerun.

## 21. Comparison with Phase 8

Phase 8's four rule configs (`results/reports/adaptive_fl_results.md`)
are the direct comparison target, particularly `combined` (F1 0.7439,
fairness gap 0.1496) and the two ablation extremes (`resource_only`
best F1 0.7547; `performance_only` best fairness 0.1245) — not rerun.

## 22. Limitations (present, more added once results exist)

Pretraining compute cost is a real multiple of a single Phase 6/8 run;
single-seed capstone scale (like Phase 7/8); discretized weight-profile
action space approximates true continuous weighting; reward-shaping
constants are a disclosed design choice, not tuned against test data;
at 8 clients the learned policy may simply converge near an existing
Phase 8 profile rather than discovering new behavior — a legitimate,
reportable outcome, not a failure to hide; the fairness proxy inherits
Phase 8's own documented own-local-data limitation.

## 23. Success metric (Q18) and what happens if DRL performs worse (Q19)

**Success metric, defined before any run**: KDDTest+ F1 of the frozen
DQN policy's final comparison run, evaluated exactly once, compared
directly against Phase 6 (0.7404) and Phase 8 `combined` (0.7439) —
alongside the client fairness gap (max−min local F1) compared against
Phase 8's range (0.1245-0.2259). Neither metric alone determines
"success" in isolation; both are reported together, matching how Phase
8 itself reported a case where the best-F1 rule was not the best-fairness
rule. If DQN underperforms either baseline on either metric, that is
reported as a real, valid result with an accompanying analysis (e.g.
insufficient pretraining experience, action space too coarse, reward
weighting miscalibrated) — never as grounds to adjust the setup until
a better number appears (Q19, restated from the project's "most
important rule").

## 24. The critical fairness question (explicit acknowledgment)

Phase 7 and Phase 8 already demonstrated that "more fairness weight ⇒
better fairness" is FALSE in this system — Phase 8's fairness-weighted
`combined` rule was beaten on fairness by `performance_only`. This
design explicitly does **not** assume the DQN will behave differently;
the reward's fairness term is a penalty shaping pressure, not a
guarantee, and the pilot/experiment plan (§25) requires empirically
measuring the DQN's actual fairness-gap outcome against Phase 7/8's
established range rather than assuming it will improve on it.

## 25. Experimental design — pilot first (Q17 baseline restated)

**Pilot (small, correctness-focused) — must pass before any larger
run:** environment executes without error; state vector has the
correct shape/values on a tiny synthetic fixture; every action maps to
a valid weight vector; reward matches a hand-computed example; the DQN
trains without error on a handful of synthetic episodes and its loss
is finite/non-NaN; aggregation output matches a hand-computed weighted
average; the AST-based privacy-boundary test passes for the new
modules; two identical-seed pilot runs produce identical results.

**Only after the pilot passes**, the 3-way comparison (Phase 6 vs.
Phase 8 vs. Phase 9B) is run under §14's fixed protocol, reporting (all
required metrics): validation accuracy/precision/recall/F1/ROC-AUC,
KDDTest+ accuracy/precision/recall/F1/ROC-AUC, client fairness gap
(**the true own-local-data metric, §8(b) — computed once, post-hoc,
exactly as Phase 8 computes it; NOT the reward-time `perf_spread_t`
proxy from §8(a), which is never reported as "the" fairness result**),
communication estimate, training time, controller behavior (which
profiles were chosen, how often), and the action distribution over the
comparison run's 10 rounds.

## 26. Real spacecraft data (OPSSAT-AD) separation

**Added 2026-08-15** (first design checkpoint review, Issue 3 — this
statement existed in the earlier, superseded draft
`11-phase-9b-drl-adaptive-fl-design.md` and was inadvertently dropped
during consolidation into this canonical document; restored here, not
newly reasoned).

OPSSAT-AD is **Track B — real-spacecraft validation**
(`docs/project-progress/10-phase-9-opssat-validation.md`), entirely
separate from the Phase 9B DQN/FL experiment. **NSL-KDD remains the
only dataset used for Phase 9B's FL/DQN training and evaluation**,
identically to Phase 6, Phase 7, and Phase 8 — this is what keeps Phase
9B directly comparable with Phase 6 and Phase 8 under §14's fixed
protocol. OPSSAT-AD is not loaded, partitioned, or referenced anywhere
in the DQN pretraining episodes, the final comparison run, the reward
function, or the state vector. OPSSAT-AD has no federated/multi-client
structure (a single spacecraft; Phase 9A's own "What this does NOT
prove" section already states its FL machinery was never applied to
it) and remains, as always, a track whose metrics (F1/MCC/AUCPR) are
never combined with NSL-KDD/Track A's metrics into one number, in
either direction.

## 27. Artifact path convention

**Added 2026-08-15** (first design checkpoint review, Issue 4).
Inspected the repository's actual established conventions before
choosing: Phase 6 uses `experiments/federated/` and
`results/models/federated/round_history.json`; Phase 8 uses
`experiments/adaptive_fl/<rule>/` and `results/plots/adaptive_fl/` +
`results/reports/adaptive_fl_results.md`. (Note: `experiments/drl/`,
`experiments/fedavg/`, and `experiments/adaptive/` also exist in the
repository, but only as empty `.gitkeep` placeholders from the Phase 0
skeleton — never actually used; Phase 6 and Phase 8 both ended up using
differently-named directories that match their own results-report
filenames instead. Phase 9B follows that same actually-used pattern,
not the unused skeleton placeholders.)

Phase 9B will use, consistent with the `adaptive_fl` pattern:

```
experiments/drl_adaptive_fl/                          run configs, episode logs, round histories
results/models/drl_adaptive_fl/round_history.json      round-by-round metrics (small, committed)
results/models/drl_adaptive_fl/*.pt                    model/DQN checkpoints (gitignored, like all *.pt)
results/plots/drl_adaptive_fl/*.png                    training curves, action-distribution plots (small, committed)
results/reports/drl_adaptive_fl_results.md             final results report, incl. the post-hoc fairness-gap table (small, committed)
configs/drl.yaml                                       filled in (currently a null placeholder)
```

This matches the existing `.gitignore` pattern already established for
every prior phase (`*.pt` ignored; `results/reports/*`,
`results/plots/*`, `results/models/*` ignored by default with explicit
per-phase exceptions carved out for the small, reviewable files) — no
`.gitignore` change is needed until these paths actually contain files
to commit.

## Reusable existing code

`src/federated/{client,server,protocol,fedavg}.py` (unchanged);
`src/federated/adaptive.py` (`ClientSignals`, `build_resource_signal`,
`normalize`, `compute_client_scores`, `scores_to_aggregation_weights`);
`src/federated/adaptive_trainer.py` as the round-loop template;
`src/simulation/{partitioner,non_iid_metrics,network_conditions}.py`;
`src/training/local_trainer.py:build_initial_state_dict`;
`src/training/baseline_trainer.py:{load_features_and_labels,make_dataloader,run_validation}`;
`src/evaluation/metrics.py`; the already-scaffolded, empty `src/drl/`
package and the null-placeholder `configs/drl.yaml`.

## New files required (not created yet)

```
src/drl/{replay_buffer,dqn_agent,action_space,environment}.py
src/federated/drl_trainer.py
configs/drl.yaml (filled in)
scripts/{train_drl_controller,run_drl_adaptive_fl,evaluate_drl_fl_results}.py
tests/drl/{test_replay_buffer,test_dqn_agent,test_environment,test_drl_trainer}.py
```

## Test plan

State construction shape/values on synthetic fixtures; action validity
(every index → non-negative weights summing to 1); reward calculation
against hand-computed examples with an explicit assertion that no
KDDTest+ path is reachable from the reward function; replay buffer
capacity/sampling determinism; epsilon-greedy behavior at ε=0 and ε=1;
target-network update cadence; aggregation correctness against a
hand-computed weighted average; an AST-based privacy-boundary test for
`environment.py`/`drl_trainer.py` (extending the existing `server.py`
pattern); a dedicated "no test-set path" scan across the new
training/environment code; seeded-reproducibility (two identical-seed
runs produce identical results); an end-to-end pilot smoke test on
tiny synthetic partitions.

## Reproducibility information

Seed 42 used for: DQN network initialization, the final comparison
run's client/server initialization (shared init seed, matching Phase
6/8), and the reference partition. Pretraining episodes deliberately
vary their own partition seed (to generate diverse experience) while
keeping the DQN's own learning-process seed fixed at 42 — this
asymmetry will be stated explicitly in the eventual results report, not
left implicit.

## Git history (as of this document)

Branch `project-development-4-drl` (see the git-audit note at the top
of this session's response — created from `project-development-3` at
commit `c58a8a2`, identical history, no divergence). No commit made for
Phase 9B work yet.

## Next steps

1. Calibrate the exact pretraining episode/round budget against real
   per-round timing on this machine.
2. Implement the pilot-scope code only (§25), get it reviewed.
3. Run the pilot; if it passes, run the 3-way comparison under §14.
4. Fill in §16-19 (pilot/final results/ablations/failure cases) with
   real, measured numbers — worse-than-baseline included, if that's
   what happens.
5. Update `docs/research/experiments/phase-9b-drl/README.md` and
   `docs/research/development-log/` with the outcome, following the
   preserve-don't-rewrite convention for any correction to this design
   document's assumptions.
