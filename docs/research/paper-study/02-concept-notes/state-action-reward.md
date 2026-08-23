# Concept — State, Action, Reward (the RL Triple)

**What is it?** The three pieces every reinforcement learning problem must define precisely before anything can be learned: **state** (what the agent observes), **action** (what it can choose to do), **reward** (the feedback signal telling it how good that choice was).

**Simple example:** think of a thermostat that "learns." State = current room temperature + time of day. Action = {heat, cool, do nothing}. Reward = +1 if the room stays comfortable, −1 if someone complains it's too hot/cold. Over many days, it learns which action to take in which state.

## Side-by-side: how FedSTAR and our project define each piece

| | FedSTAR (SAC) | Our Project (DQN, Phase 9B) |
|---|---|---|
| **State** | Pairwise cosine-similarity matrix of clients' classifier-head parameters + previous loss/accuracy + previous algorithm/cluster-count | 44-dim vector: per-client performance/data/resource/fairness (4×8=32) + validation-F1 trend (2) + round progress (1) + previous action one-hot (9) |
| **Action** | Hybrid: {clustering algorithm ∈ 5 options, cluster count ∈ 1-6} (discrete) + {sampling ratio, aggregation weight, fusion coefficient} ∈ [0,1]³ (continuous) | Discrete only: pick 1 of 9 fixed, named weight-profiles |
| **Reward** | `(accuracy gain) − (regression penalty) − (intra-cluster divergence penalty)` | `ΔF1 (validation) − (fairness-spread penalty) − (action-instability penalty)` |

**Technical explanation:** formally, an RL problem is a Markov Decision Process `(S, A, R, P, γ)` — state space `S`, action space `A`, reward function `R`, transition dynamics `P` (how the state changes given an action), and a discount factor `γ` weighting future rewards. Both FedSTAR and our project treat one federated-learning round as one "timestep" in this framework.

**Why papers define these so carefully:** get any one wrong and the agent either can't learn anything useful (state too sparse/uninformative), can't express the decision you actually want (action space too narrow, or — for DQN — wrongly discrete), or learns to game the wrong thing (reward badly shaped, e.g. rewarding raw accuracy alone without a stability term, which can cause round-to-round oscillation).

**Advantages of getting this right:** a well-designed state/action/reward triple lets a DRL agent learn genuinely useful, context-sensitive behavior no hand-written rule could easily replicate.

**Limitations:** designing these well is genuinely hard and mostly manual/human-judgment-driven — both FedSTAR and our project's reward-weighting constants (`λ_α, λ_β, λ_γ` for FedSTAR; `λ_fair, λ_stab` for us) were chosen by hand/grid-search before seeing final results, not learned themselves.

**Which papers use it:** FedSTAR (fully specified); our project (fully specified, Phase 9B, `configs/drl.yaml` + `src/drl/state.py`/`action_space.py`/`reward.py`).

**Why it matters to our project:** these three definitions are the exact, already-implemented core of Phase 9B — not a plan, real code with real, already-measured behavior (see `results/reports/drl_adaptive_fl_results.md`).
