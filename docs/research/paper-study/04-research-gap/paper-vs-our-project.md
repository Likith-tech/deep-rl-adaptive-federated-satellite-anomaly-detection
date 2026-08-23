# Paper vs. Our Project — Rigorous Component-by-Component Comparison

**Scope:** compares the two closest papers (FedSTAR primarily; Hundman et al. secondarily, as the closest telemetry-lineage analog) against our actual project. Every row for "Our Project" is explicitly labeled **IMPLEMENTED**, **PLANNED**, **PROPOSED**, or **NOT YET VALIDATED** — never blurred. "IMPLEMENTED" means verified against real, already-run code/results in this repository (`src/drl/`, `src/federated/`, `results/reports/`), not aspiration.

## FedSTAR vs. Our Project

| Component | FedSTAR (existing paper) | Our Project | Same/Different | Explanation |
|---|---|---|---|---|
| Dataset | STIN/TON_IoT/Bot-IoT (benchmark, not real satellite traffic) | **IMPLEMENTED** — Track A: NSL-KDD (real, terrestrial). **IMPLEMENTED** — Track B: OPSSAT-AD (real, single spacecraft) | Different | Neither uses real *multi-satellite* data; ours additionally has a real *single-spacecraft* telemetry track FedSTAR lacks entirely |
| Satellite environment | 20 simulated clients from centralized benchmark data | **IMPLEMENTED** — 8 simulated clients (NSL-KDD, Dirichlet) | Same approach (simulation), different scale/source | Fewer clients; different underlying dataset domain (network-security benchmark vs. terrestrial IDS data) |
| Client simulation | Dirichlet, η=0.5 (single value reported in main results) | **IMPLEMENTED** — Dirichlet α=0.5, plus a **full controlled sweep** (α=0.1/0.5/1.0/5.0/10.0, Phase 7) | Same technique | Ours adds a dedicated multi-point heterogeneity study FedSTAR's main results don't report |
| Non-IID strategy | SAC-driven clustering (5 algorithms) + continuous weighting | **IMPLEMENTED** — Phase 8 fixed rule; **IMPLEMENTED** — Phase 9B DQN-learned discrete profile selection | Different mechanism | No clustering step in ours; discrete-only vs. FedSTAR's hybrid discrete+continuous |
| Local model | SimVPv2 (pseudo-video + ConvSC + gSTA/LKA), heavy spatio-temporal | **IMPLEMENTED** — BaselineMLP (Track A); TemporalGRUAttention (Track B) | Different | Ours is deliberately smaller/simpler; no spatial dimension implemented |
| Temporal model | Synthetic pseudo-video (T=4 fabricated timesteps from 1D features) | **IMPLEMENTED (but limited)** — TemporalGRUAttention exists, but the current Track B run uses `sequence_length=1`, so genuine temporal recurrence is **NOT YET VALIDATED** — disclosed explicitly by this project's own Phase 9A documentation | Different | FedSTAR's temporal dimension, while synthetic, is at least exercised (T=4); ours is architecturally present but not currently exercised |
| Attention mechanism | LKA (Large Kernel Attention, decomposed large-kernel convolution) | **IMPLEMENTED** — standard learned-weight attention over GRU hidden states (Track B), conceptually closer to Paper 1 (Yang et al.) than to LKA | Different mechanism, same general goal | Neither borrowed from the other; independently converging on "attention helps" |
| Federated algorithm | Hierarchical: cluster → intra-cluster weighted avg → cluster fusion → global-local momentum | **IMPLEMENTED** — flat: FedAvg (Phase 6) → `aggregate_with_weights` (Phase 8/9B), single-level, no clustering | Different | Ours is architecturally simpler by design |
| Aggregation | Weighted average, tunable λ between data-volume and uniform, plus momentum blend γ | **IMPLEMENTED** — weighted average via named profile's fixed (w_perf, w_data, w_resource, w_fair) vector, no momentum-blend step | Different formula, same weighted-averaging family | — |
| Client selection | Sampling ratio ρ within clusters — can effectively exclude some clients from a round's aggregation | **IMPLEMENTED** — weighting only; **all 8 clients participate every round**, by deliberate design (documented fairness rationale) | Different philosophy | Explicit, disclosed trade-off — not a gap, a choice |
| DRL | SAC (actor+critic, hybrid discrete+continuous action) | **IMPLEMENTED** — DQN (single Q-network, discrete-only, 9 named actions), fully trained (750 rounds) and evaluated | Different algorithm family | DQN chosen because the action space was deliberately kept small/discrete; not a capability gap, a scoping choice |
| State | Cosine-similarity matrix of classifier-head params + loss/acc history + previous algorithm/k | **IMPLEMENTED** — 44-dim: per-client performance/data/resource/fairness (4×8) + val-F1 trend (2) + round progress (1) + previous-action one-hot (9) | Different composition | Ours uses derived, named per-client signals; FedSTAR uses raw-parameter similarity — different information sources for a similar purpose |
| Action | Hybrid: {clustering algorithm, k, ρ, λ, γ} | **IMPLEMENTED** — pick 1 of 9 fixed, named weight-profiles | Different | FedSTAR's action space is strictly larger/more expressive |
| Reward | Accuracy gain − regression penalty − intra-cluster divergence penalty | **IMPLEMENTED** — ΔF1 (validation) − fairness-spread penalty − action-instability penalty | Structurally analogous (gain − regression-like term − a coherence/stability term) | Different specific terms, same 3-part shape |
| Resource allocation | **Not present** — FedSTAR's state has no resource-availability dimension | **IMPLEMENTED** — explicit `resource_k` signal (simulated bandwidth/latency/compute/availability/connectivity), used in both Phase 8's rule and Phase 9B's state | Different | A genuine capability our project has that FedSTAR does not |
| Adaptive mechanism | One reported system (SAC directly) | **IMPLEMENTED** — a controlled **three-stage** comparison: fixed FedAvg → hand-rule → DQN, all three run on the identical pipeline and partition | Different methodology | This staged, same-pipeline comparison is not present in FedSTAR's own reported experiments |
| Communication efficiency | Measured (MiB/round); lower than its own closest competitor (FLOGA-AD) via clustering | **IMPLEMENTED** — measured (~14.6 MB/10 rounds), **unchanged** across Phases 6–9B since all 8 clients always participate | Both measure it | FedSTAR's clustering can reduce effective per-round communication (via ρ<1 sampling); ours doesn't reduce it, by the same fairness-preserving design choice above |
| Security | Standard FL privacy framing; Byzantine robustness **explicitly deferred** to future work (stated twice in the paper) | **IMPLEMENTED** — privacy boundary **mechanically enforced** via an AST-based automated test (not just claimed); Byzantine robustness **NOT ADDRESSED** (equally open) | Partially different | Our privacy claim is testable, not just asserted — a genuine methodological strength; Byzantine robustness is an equally-shared open gap, not a differentiator |
| Evaluation rigor | 5 random seeds, paired t-test significance testing | **IMPLEMENTED, but weaker** — single-seed, single-run reporting throughout (Phases 6–9B) | Different, and **this is a real shortfall on our side**, not a strength | Must be disclosed honestly if compared directly |
| Explainability | Named clustering-algorithm choices logged; continuous knobs less directly interpretable | **IMPLEMENTED** — fully named, small (9-option), exhaustively enumerable action catalogue; full action-distribution trace reported (exploration phase vs. frozen final-run behavior) | Ours arguably more auditable | A genuine, if modest, strength |

## Hundman et al. (2018) vs. Our Project's Track B (OPSSAT-AD)

| Component | Hundman et al. | Our Project (Track B) | Same/Different | Explanation |
|---|---|---|---|---|
| Detection philosophy | Forecasting: predict next value, threshold the residual | **IMPLEMENTED, but different** — current OPSSAT-AD run is **supervised segment-level classification**, not residual-forecasting | Different | Disclosed explicitly by this project's own Phase 9A documentation — do not conflate the two |
| Model | Plain LSTM, per channel | **IMPLEMENTED** — TemporalGRUAttention (shared across the dataset's segments, not per-channel) | Different | — |
| Dataset | Real NASA SMAP/MSL, ISA-report labels | **IMPLEMENTED** — real ESA OPS-SAT (OPSSAT-AD), author-provided labels | Both real spacecraft data | Different missions/agencies/datasets |
| Multi-satellite / FL | None | **NOT IMPLEMENTED** — single spacecraft, no federated structure attempted | Same (neither has FL here) | Explicitly disclosed by our own Phase 9A doc as something this phase does NOT prove |
| Threshold mechanism | Nonparametric dynamic threshold + pruning | **N/A** — current implementation is a direct classifier, not a threshold-on-residual system | Different | — |

## Summary of labeling discipline used above

- **IMPLEMENTED** = verified against real, already-executed code and reported results in this repository.
- **PLANNED** = named in project design documents as a next step, not yet built.
- **PROPOSED** = suggested here as a reasonable future direction, not committed to in any project document.
- **NOT YET VALIDATED** = code/architecture exists, but the specific claim (e.g., "genuine temporal modeling") has not been demonstrated by an actual run.

No row above claims Track A and Track B are combined — they are not, and no future-work item in this project's own documentation currently commits to combining them on a specific timeline; any such combination should be labeled **PROPOSED**, not planned, unless a project document says otherwise.
