# Concept — Deep Reinforcement Learning (DRL): DQN vs. SAC

## 1. What is it?

> **Technical term → simple meaning → example**
> **Deep Reinforcement Learning:** reinforcement learning (see `reinforcement-learning.md`) where the agent's decision-making function (its policy, or its value estimates) is represented by a deep neural network instead of a small lookup table — needed whenever the state space is too large/continuous to enumerate by hand.
> *Analogy:* a simple RL problem (like a 3x3 tic-tac-toe board) can literally store "best move for every possible board state" in a table. A satellite federated-learning round's state (a 44-dimensional vector of continuous numbers, in your project's case) has effectively infinite possible values — you can't build a table for that; you need a neural network that can generalize across similar-but-not-identical states it has never seen exactly before.

## 2. Why does it exist?

Plain ("tabular") reinforcement learning only works when the number of possible states/actions is small enough to store explicitly. Real-world problems — including both Paper 4's and your project's — have continuous, high-dimensional states (similarity matrices, per-client performance vectors), which classic tabular RL simply cannot represent. Deep neural networks solve this by *learning to generalize* across similar states.

## 3. DQN (Deep Q-Network) — used by your project

> **DQN:** a neural network that learns to estimate "how good is it to take each possible action from this state" (a Q-value for each action), and picks the action with the highest estimated Q-value.

**Key components (both used in your project's Phase 9B, confirmed by the actual code):**
- **Replay buffer:** stores past experience (state, action, reward, next state) so the network can learn from a random, decorrelated *mix* of past experiences instead of only the most recent one — improves training stability. Your project: capacity 5,000, uniform sampling.
- **Target network:** a slowly-updated copy of the Q-network, used to compute stable learning targets — prevents the network from "chasing its own tail." Your project: hard-synced every 10 learning steps.
- **Epsilon-greedy exploration:** as described in `reinforcement-learning.md`.

**Limitation of DQN, important for comparing to Paper 4:** DQN only outputs a **discrete** choice among a fixed list of actions — it cannot directly output a continuous number (like "use exactly 37% of clients"). This is *why* your project's action space is a fixed catalogue of 9 named weight-profiles, not a continuously-tunable weighting.

## 4. SAC (Soft Actor-Critic) — used by Paper 4 (FedSTAR)

> **SAC:** a more advanced DRL algorithm that can output continuous actions (and, with the right formulation, discrete ones too), and explicitly balances "maximize reward" against "stay diverse/keep exploring" through an entropy term built into its objective.

**Why Paper 4 needed SAC and your project didn't:** FedSTAR's action includes continuous values (client sampling ratio, aggregation weight factor, global-local fusion coefficient) *alongside* a discrete choice (which clustering algorithm) — a hybrid action space DQN cannot represent. Your project's action space is *entirely* discrete (pick 1 of 9 fixed profiles) by deliberate design, so DQN is sufficient and appropriately simpler.

## 5. Simple example

**DQN-style (your project):** "Given this round's signals, which of these 9 named weighting strategies should I use?" — a multiple-choice question.
**SAC-style (Paper 4):** "Given this round's signals, which clustering algorithm should I use, AND exactly what sampling ratio (any number 0–1), AND exactly what aggregation weight (any number 0–1) should I use?" — a mixed multiple-choice-plus-fill-in-the-number question.

## 6. Advantages

- DQN: simpler, more explainable (you can always name exactly which of the 9 profiles was chosen, and inspect that profile's fixed weight recipe), cheaper to train, well-suited to a bounded, interpretable action catalogue — appropriate for a capstone-scale, honestly-scoped experiment.
- SAC: more expressive (true continuous control), can in principle find better solutions than any fixed catalogue of discrete options could ever contain.

## 7. Limitations

- DQN: can only choose among pre-defined discrete options — if none of the 9 profiles is truly optimal for a given round, DQN cannot invent a better in-between weighting.
- SAC: more complex to implement and tune (multiple networks: actor + twin critics + optionally an automatically-tuned entropy coefficient), more computationally expensive — Paper 4's own Table 6 shows its full system costs more training time (102.82 min) and inference latency (4.78 ms/batch) than a lighter baseline.
- Both: DRL in general needs a lot of experience (many episodes) to learn reliably — see the "Limitations" point in `reinforcement-learning.md` about your project's Phase 9B needing 750 rounds and still not beating the rule-based baseline in its one completed run.

## 8. Which papers use it

Only Paper 4 (SAC). Papers 1–3 use no RL of any kind, deep or otherwise.

## 9. How it connects to our project

DQN is literally implemented in your project's `src/drl/dqn_agent.py`, with a QNetwork (44 → 64 → 64 → 9), target network, and a real `learn()` gradient step (standard DQN target: `r + γ·(1−done)·max_a' Q_target(s',a')`, MSE loss, Adam) — this is not a plan, it's real, tested, already-run code (`results/reports/drl_adaptive_fl_results.md`). When explaining "why DQN and not SAC/PPO" in viva, the honest, defensible answer is: **the action space was deliberately kept small and discrete (9 named, inspectable weight-profiles) for explainability and capstone-appropriate scope, and DQN is the natural, simplest algorithm for a purely discrete action space** — not "SAC is better and we couldn't do it," but "our problem was scoped to not need it."
