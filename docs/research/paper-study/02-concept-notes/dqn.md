# Concept — DQN (Deep Q-Network)

**What is it?** A deep reinforcement learning algorithm that learns to estimate a Q-value — "how good is it to take this action from this state" — for every possible discrete action, using a neural network, then picks the action with the highest estimated value.

**Simple example:** each federated round, our project's DQN looks at the current 44-dimensional state and estimates 9 numbers (one per weight-profile): "if I pick `performance_only` here, I expect a Q-value of 0.8; if I pick `resource_only`, 0.6; ..." and picks the highest.

**Technical explanation:**
- **Network:** input (44-dim state) → hidden (64 → 64, ReLU) → output (9 Q-values, one per action).
- **Replay buffer:** stores past `(state, action, reward, next_state)` transitions; the network learns from random mini-batches of past experience, not just the most recent step — this decorrelates training data and stabilizes learning. Our project: capacity 5,000, uniform sampling.
- **Target network:** a slowly-updated copy of the Q-network used to compute stable learning targets (`r + γ·max_a' Q_target(s',a')`), preventing the network from chasing its own constantly-shifting predictions. Our project: hard-synced every 10 learning steps.
- **Epsilon-greedy:** with probability ε, pick a random action (explore); otherwise pick the best-known action (exploit). ε decays from 1.0 to 0.05 over training.
- **Foundational paper:** Mnih et al., "Human-level control through deep reinforcement learning," *Nature* 518(7540):529-33, 2015 (independently verified via search: DeepMind, Atari-game-playing agent).

**Why our project uses DQN specifically:** the action space is deliberately kept small and **entirely discrete** — 9 fixed, named weight-profiles, no clustering, no continuous knobs. DQN is the simplest, standard algorithm for a purely discrete action space.

**Advantages:** simple, well-understood, cheap to train, every decision is fully named/inspectable (you can always say exactly which of the 9 profiles was chosen).
**Limitations:** cannot output a continuous number directly — if the truly optimal weighting lies "between" two of the 9 profiles, DQN cannot invent it. Needs many training rounds (our project used 750) and is not guaranteed to beat a good fixed rule (our own Phase 9B result: it didn't, in the one completed run).

**Which papers use it:** none of our 4 core papers use DQN specifically — FedSTAR uses SAC (see `sac.md`), because its action space needs continuous outputs DQN can't provide. Several search-found papers (FedPPO uses PPO; others use SAC) confirm DQN, PPO, and SAC are all live, recognized choices in this space, each suited to a different action-space shape.

**Why it matters to our project:** `src/drl/dqn_agent.py` implements exactly this (Q-network, replay buffer, target network, epsilon-greedy, real gradient-descent `learn()` step) — this is the actual, already-run algorithm behind Phase 9B, not a plan.
