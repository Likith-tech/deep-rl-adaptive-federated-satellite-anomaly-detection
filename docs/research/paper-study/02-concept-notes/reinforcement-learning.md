# Concept — Reinforcement Learning (RL): State, Action, Reward, Policy

## 1. What is it?

> **Technical term → simple meaning → example**
> **Reinforcement Learning:** an agent learns what to do by trying actions, observing the resulting outcome (reward), and gradually improving its decisions over many attempts — instead of being shown labeled "correct answer" examples like supervised learning.
> *Analogy:* learning to ride a bike. Nobody hands you a labeled dataset of "correct" handlebar angles. You try things, sometimes you fall (bad reward), sometimes you stay balanced (good reward), and over many attempts you get better — without ever being told the "right answer" directly.

**The four core pieces every RL problem needs, precisely defined:**

| Term | Simple meaning | In Paper 4 (FedSTAR) | In your project (Phase 9B) |
|---|---|---|---|
| **State** | What the agent observes before deciding | Client-similarity matrix + recent loss/accuracy + previous action | 44-dim vector: per-client performance/data/resource/fairness signals + validation-F1 trend + round progress + previous action |
| **Action** | What the agent can choose to do | Clustering algorithm + cluster count + 3 continuous knobs (hybrid) | Pick 1 of 9 fixed aggregation weight-profiles (discrete only) |
| **Reward** | Feedback signal telling the agent how good its last action was | Accuracy gain − performance-drop penalty − cluster-divergence penalty | ΔF1 (validation) − fairness-spread penalty − action-instability penalty |
| **Policy** | The agent's learned rule for picking actions given a state | A neural network (SAC actor) | A neural network (DQN Q-network) |

## 2. Why does it exist?

Some decisions can't be labeled in advance because the "right answer" depends on a whole sequence of future consequences, not just the current input — e.g., "which aggregation weighting is best *this* round" depends on what happens over many future rounds, not something you can hand-label ahead of time the way you can label "cat vs. dog" images.

## 3. Simple example

A federated-learning server tries different aggregation weightings round after round; whichever weighting leads to the biggest validation-accuracy improvement (its reward) gets reinforced — over many rounds, the agent learns which weighting tends to work well in which situation, without a human ever hand-specifying the rule.

## 4. How it works (episode/timestep structure)

- **One timestep** = one decision point (in both Paper 4 and your project: one federated-learning round).
- **One episode** = a full sequence of timesteps from start to a defined end (in your project: one complete simulated FL run of N rounds over one client partition).
- **Exploration vs. exploitation:** the agent must sometimes try new/uncertain actions (exploration) rather than always picking what currently looks best (exploitation), or it can get stuck in a mediocre habit. Both Paper 4's SAC and your project's DQN use this idea — your project specifically uses **epsilon-greedy** exploration (a random action with probability ε, the best-known action otherwise; ε decays from 1.0 to 0.05 over training).

## 5. Why the papers/project use it

Both Paper 4 and your project use RL for the same underlying reason: **the "right" way to combine federated clients changes over time and depends on context (which client is currently trustworthy, how heterogeneous the data currently is)** — a fixed, hand-written rule (like your project's own Phase 8) can't adapt to that changing context; RL, in principle, can learn to.

## 6. Advantages

- Can in principle discover better decision rules than a human would hand-design, especially when the "good rule" depends on complex interacting signals.
- Naturally supports sequential decision-making, where today's choice affects tomorrow's options.

## 7. Limitations

- Needs **many** trial-and-error attempts to learn well — your own project's Phase 9B needed 750 real federated-learning rounds (50 episodes × 15 rounds) just for pretraining, a real cost your own project explicitly measured (101.5 minutes) and disclosed.
- Not guaranteed to beat a well-designed fixed rule — your own project's Phase 9B result is the clearest possible illustration of this: the DQN did **not** beat Phase 8's best rule-based baseline on F1 or fairness in its one completed run. This is a genuinely useful, honest finding — RL is not automatically superior to a good heuristic, especially with limited training budget/diverse experience.
- Reward design is hard and consequential — a poorly designed reward can be "gamed" by the agent in unintended ways (your project's own reward-design documentation explicitly reasons through why its fairness penalty can't be gamed by favoring one client arbitrarily — see the Phase 9B design doc's "Why the fairness proxy doesn't create undesirable behavior" section).

## 8. Which papers use it

Only Paper 4 (SAC). Papers 1–3 use no RL of any kind.

## 9. How it connects to our project

RL is the "Reinforcement" half of "Deep Reinforcement Learning" in your capstone's title — and your project's Phase 9B is a complete, already-executed RL experiment (not just a design), whose honest, disclosed negative result (DQN underperforming the rule-based baseline) is itself a legitimate and defensible research finding, not a failure to hide. See `deep-reinforcement-learning.md` for the "deep" part (why a neural network is used to represent the policy/value function at all).
