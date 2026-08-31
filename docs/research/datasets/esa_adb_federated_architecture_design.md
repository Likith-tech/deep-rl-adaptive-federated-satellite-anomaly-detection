# ESA-ADB Federated Architecture Design — Reconciling Heterogeneous Client Channel Sets with FedAvg

**DESIGN/ANALYSIS ONLY. No training run, no code written for any
candidate architecture, no FedAvg/DQN executed. Grounded in the real,
measured Experiment 1/1A results and a direct reading of the existing
`src/federated/fedavg.py`/`server.py` implementation — not assumption.**

## 1. The problem, precisely stated

Experiment 1A (`experiments/esa_adb/baseline_02_multivariate/`) proved
that a **centralized, jointly-multivariate** model
(`TemporalGRUAttention(input_dim=58)`, all 58 channels seen together at
every timestep) resolves the near-random performance of the
single-channel baseline (test ROC-AUC 0.473 → 0.909, PR-AUC 0.023 →
0.751, FPR 0.848 → 0.000). This is strong evidence that genuine
cross-channel interaction is what the model needs.

But **no single one of the 8 real FL clients owns all 58 channels** —
real ownership (verified, `src/satellite/esa_adb/clients.py`) is
3/6/7/8/8/9/9/8 channels per client. Experiment 1A's exact architecture
cannot be handed to each client unchanged: doing so would require every
client to either fabricate values for channels it doesn't own (forbidden)
or borrow another client's real telemetry (forbidden — violates data
ownership and the non-IID structure this project deliberately
preserves).

Separately, **`src/federated/fedavg.py`'s existing aggregation is a hard
constraint, not a preference** — read directly from the code:

> `federated_average`/`weighted_average`: "All state_dicts must share
> the same keys and tensor shapes (true by construction here, since
> every client starts each round from the same global model and trains
> the same architecture)."

Any candidate architecture that gives different clients different
*parameter shapes* cannot use this code unmodified.

## 2. The four candidate architectures

### Option A — Global 58-D input with availability masking

Every client instantiates the **identical** architecture: a fixed
58-dim (or 116-dim if the mask is concatenated rather than multiplied)
input, `TemporalGRUAttention(input_dim=58)` unchanged from Experiment
1A. For channels a client doesn't own, the value slot is the
**normalized zero** (never a fabricated telemetry value) and a parallel
58-dim binary availability mask `M` marks which slots are real. Two
concrete input-construction sub-variants exist:
- **Concatenate**: model input = `[X (58); M (58)]`, `input_dim=116`.
- **Gate**: `X' = X * M` fed alone, `input_dim=58`, with the mask only
  implicitly encoded (masked values are always exactly 0 for that
  client, so the model can learn to associate "value == 0" with "not
  present" only if 0 never occurs as a genuine normalized reading —
  worth flagging as a real risk: after per-channel StandardScaler
  normalization, 0 IS a valid, meaningful value for the *owning*
  client, since it means "at the train mean." **The gate-only variant
  is therefore NOT safe** — a client's real "at-baseline" reading
  would be indistinguishable from "channel absent." The concatenate
  variant avoids this collision by keeping the mask as an explicit,
  separate signal.** This is a genuine investigation finding, not an
  assumption: masking is only correct if the mask is passed explicitly,
  never inferred from a bare zero.

**Every client's local model has identical, fixed parameter shapes** —
this is the property `federated_average` requires, satisfied without
any change to the existing aggregation code.

A second real property, worth stating precisely rather than glossed
over: for a client that owns only 3 of 58 channels (client_3), 55 of
its 58 input slots are *structurally* zero+unavailable for every one of
its samples. This does **not** mean those columns of the shared input
projection matrix go untrained — federated averaging combines
contributions *across* clients, and each client naturally drives
meaningful gradient signal into the columns of the projection matrix
corresponding to channels *it actually owns*; other clients drive the
other columns. Over a full federation, every column gets real training
signal from whichever client(s) genuinely have that channel — this is
a coherent, defensible collaborative-learning property of the design,
not a flaw, though it does mean any *one* client's local update is
sparse (touches few columns), which slows how quickly the shared
projection matrix converges compared to a hypothetical client that saw
all 58 channels.

### Option B — Shared per-channel encoder + pooling

A single small shared encoder (e.g. a compact GRU or embedding module)
processes **each available channel's own univariate sequence
independently** into a fixed-size embedding; a client's per-channel
embeddings (however many it has) are then aggregated (mean or
attention pooling) into one fixed-size vector, fed to a shared anomaly
head.

The encoder and head have parameter shapes that do **not** depend on
channel count at all (the encoder is invoked once per channel, however
many channels exist — a runtime loop, not an architectural
difference), so this is natively, cleanly FedAvg-compatible with zero
aggregation-code changes.

Cross-channel interaction is captured **only at the pooling step**,
after each channel has already been encoded independently through
time. This is architecturally weaker than Experiment 1A's proven
mechanism, which let the GRU/attention see all channels jointly *at
every timestep* — Option B cannot represent "channel A's spike at t,
followed by channel B's spike at t+2" as directly, since by the time
pooling happens, per-channel temporal information has already been
compressed into single vectors with no cross-channel timing left.

### Option C — Client-local input projection + shared temporal core

Each client learns its own local projection `Linear(N_client → D)`
mapping its own channels into a common latent width `D`; the
**shared** `TemporalGRUAttention`-style core (GRU + attention + head)
then operates on this D-dim latent sequence, jointly, at every
timestep — architecturally the closest match to Experiment 1A's
proven mechanism, just fed a compressed multivariate signal instead of
the raw 58-dim one.

**This is not standard FedAvg.** The projection layer's shape depends
on `N_client` (3 for client_3, 9 for client_5/6) — these cannot be
averaged by `weighted_average` as written (mismatched tensor shapes).
The only correct way to use this architecture is a **personalized-FL
pattern** (e.g. FedPer/LG-FedAvg-style): split each client's
`state_dict` into a *shared* subset (GRU + attention + head — same
shape everywhere, safe to pass to the existing `weighted_average`) and
a *local* subset (the client's own projection layer — never
aggregated, never leaves the client, persists across rounds). This
requires a real, disclosed modification to the aggregation code (a
key-filtering step before calling `weighted_average`, plus per-client
local-state persistence in `FederatedClient`) — not just a
configuration change.

A second, equally real wrinkle: the DQN reward/state design
(`src/federated/drl_round_provider.py`, confirmed by direct reading)
evaluates **one aggregated global model** against **one pooled global
validation set**. Under Option C, "the aggregated global model" only
fully exists for the *shared* subset — evaluating it needs *some*
projection layer to turn validation channels into the shared latent
space, and no single client's local projection layer is the "right"
one to use globally. This needs an explicit resolution before Option C
could support the existing DQN evaluation pattern unmodified — e.g., a
held-out, separately-trained "reference" projection layer used only
for validation, or an ensemble/average over client projection layers
(itself a design decision requiring justification, not attempted
here). This is flagged as an open problem, not solved in this
analysis.

### Option D — Subset-pooling / set-based encoder (attention pooling)

Structurally Option B with attention-based pooling instead of mean
pooling (Set-Transformer/Deep-Sets-style) — same FedAvg compatibility
(fully shared, standard aggregation, no code change), same clean
handling of variable channel counts, same fundamental limitation
(cross-channel interaction only at the pooling stage, after
independent per-channel temporal encoding) — attention pooling can
*weight* channels adaptively per client, which is a real improvement
over plain mean pooling, but does not add cross-channel *timing*
interaction that Option C preserves. In the design space, D is best
understood as "B done more carefully," not a distinct strategic
choice.

## 3. Comparison table

| Approach | Same FedAvg architecture? | Handles different channels? | Cross-channel modeling? | Temporal modeling? | Data ownership preserved? | Communication cost | Computational cost | DQN compatibility | Complexity | Scientific suitability |
|---|---|---|---|---|---|---|---|---|---|---|
| **A — Global 58-D + mask** | **Yes**, unmodified `federated_average` | Yes (fixed-size, masked) | **Strong** — joint, per-timestep, same mechanism proven in 1A | **Strong** — full GRU+attention over real seq_len=50 | Yes — zero placeholder + explicit mask, never fabricated/borrowed | Moderate (58 or 116-dim input layer; whole model still tiny) | Low (same as 1A's model, negligible for a GRU this size) | **Yes, unmodified** — existing single-global-model DQN evaluation applies directly | **Low** — reuses `TemporalGRUAttention` unchanged | Strong, standard pattern in federated heterogeneous-feature literature |
| **B — Shared per-channel encoder + pooling** | **Yes**, unmodified | Yes, natively (pooling over a variable-size set) | **Weak-moderate** — only at the late pooling step, no cross-channel timing | Moderate — per-channel only, timing lost after pooling | Yes — no fabrication needed at all | **Low** — small fixed encoder+head | Low-moderate (per-channel forward pass, still small) | Yes, unmodified | **Moderate-high** — new architecture, not a reuse | Strong, novel (Deep Sets-style), but weaker fit to the proven 1A mechanism |
| **C — Client-local projection + shared core** | **No** — requires personalized/partial-parameter aggregation (real code change to `fedavg.py`/`server.py`) | Yes, natively | **Strongest** — joint, per-timestep, closest match to 1A | **Strong** — same GRU+attention mechanism as 1A | Yes | **Lowest** — local layers never transmitted | Low | **Open problem** — no single "global" projection layer exists for validation; needs an explicit, unresolved design choice | **Moderate-high** — needs partial-aggregation support + validation-evaluation resolution | Strongest scientific fit to the stated research architecture, but least "ready to implement" today |
| **D — Set-based encoder + attention pooling** | **Yes**, unmodified | Yes, natively | Weak-moderate (better than B via learned weighting, still no cross-channel timing) | Moderate (same limitation as B) | Yes | Low-moderate (small attention-pooling parameters added to B) | Low-moderate | Yes, unmodified | **High** — most implementation work of the four | Strong, publishable, but marginal improvement over B for real added complexity |

## 4. Recommendation: **Option A**

**Why it fits our ESA-ADB data:** it is the most direct, minimally-altered
extension of the architecture Experiment 1A already *proved* works on
this specific dataset (test ROC-AUC 0.909) — same GRU+attention core,
same joint per-timestep multivariate reasoning, only the input
construction changes (masked 58-dim vector instead of a fully-populated
one). Nothing about the mechanism that produced 1A's strong result is
lost.

**Why it works with Federated Learning as it exists in this
repository:** `src/federated/fedavg.py`'s `federated_average` and
`weighted_average` require identical state_dict keys/shapes across
clients — Option A satisfies this by construction, with **zero
modification** to any protected FL/DQN code (`src/federated/`,
`src/drl/`, Phase 6-9B). Options B/D would also satisfy this but at
the cost of the cross-channel-timing capability that 1A showed
matters; Option C would require rewriting the aggregation protocol
itself.

**Why it preserves non-IID client ownership:** every value a client
contributes is either its own real telemetry or an explicit,
never-fabricated "absent" marker — no channel is invented, borrowed,
or duplicated across clients, and the real per-client channel-count
imbalance (3 to 9 channels) remains fully visible to the model via the
mask rather than being hidden or normalized away.

**Why it supports the eventual DRL adaptive-aggregation framework:**
the DQN controller's existing state/reward design already assumes one
aggregated global model evaluated on one pooled validation set
(confirmed directly in `src/federated/drl_round_provider.py`) — Option
A is the only candidate that satisfies this assumption with no
additional design work. Options B/D satisfy it too but with weaker
modeling; Option C actively breaks it until a further, currently
unresolved design decision is made.

**Known limitation, disclosed rather than hidden:** small-channel
clients (client_3, 3/58 real dims populated) get sparse local gradient
signal into the shared projection matrix; the shared matrix's full
58-column quality depends on the federation as a whole, not any one
client's local training. This is expected to be a real, measurable
effect once training is attempted — worth watching for in the next
experiment, not a reason to reject Option A now.

## 5. Whether Experiment 1A's exact architecture can become the shared federated model

**No, not unmodified.** `TemporalGRUAttention(input_dim=58)` trained on
the fully-pooled, ungated 58-channel signal is a **centralized-only**
reference — it assumes every sample has all 58 real channels present,
which is never true for any single real client. It must remain
`experiments/esa_adb/baseline_02_multivariate/`'s role: the
centralized upper-bound reference point that any FL design is measured
against, not itself deployed per-client.

What **does** carry forward unchanged is the underlying
`TemporalGRUAttention` model class and its GRU+attention mechanism —
Option A reuses it exactly, only changing what is fed into
`input_dim` (58 real values + an explicit 58-dim mask, `input_dim=116`
for the concatenated variant) and how that input is constructed per
client. This is a direct, disclosed reuse of proven infrastructure,
not a new invented architecture.

## 6. What is explicitly NOT decided or done in this document

No code was written for any of the four options. No training was run.
No FedAvg, DQN, or hyperparameter search was executed.
`experiments/esa_adb/baseline_01/` and `baseline_02_multivariate/` were
not touched. No protected artifact (Phase 6-9B, Improvement 01,
NSL-KDD, OPSSAT-AD) was read or modified in the course of this
analysis beyond reading `src/federated/fedavg.py`/`server.py` to
verify the shape-matching constraint cited above (read-only).

## 7. Open questions for the next, separately-approved step

If Option A is approved: exact mask-construction mechanics (per-client,
deterministic, derived directly from `clients.CLIENT_CHANNELS`), how
the global validation set is masked for evaluation (a client-neutral
choice — e.g. no masking at validation time, since validation is
inherently a pooled/global signal, not a per-client one — needs
explicit design, not assumed here), and whether `input_dim=58`
(gated) or `input_dim=116` (concatenated mask) is used — this document
argues concatenation is the only safe choice (Section 2), but the
actual implementation is future work, not this document.
