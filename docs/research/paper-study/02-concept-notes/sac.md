# Concept — SAC (Soft Actor-Critic)

**What is it?** A deep reinforcement learning algorithm that learns both a policy (actor: what action to take) and a value estimate (critic: how good that action was), while also explicitly rewarding the policy for staying diverse/uncertain (entropy) rather than collapsing too quickly onto one behavior — and, unlike DQN, can output **continuous** actions (real numbers), not just a choice from a fixed list.

**Simple example:** FedSTAR's SAC agent doesn't just pick "which clustering algorithm" (a multiple-choice question, like DQN could handle) — it *also* outputs "exactly what sampling ratio (any number between 0 and 1)" and "exactly what aggregation weight (any number between 0 and 1)" in the same decision — a mixed multiple-choice-plus-fill-in-the-number question DQN cannot answer.

**Technical explanation:** SAC maximizes a modified reward objective that includes an entropy bonus (`reward + α·entropy(policy)`), encouraging exploration; it typically uses twin critic networks (to reduce value-overestimation bias) and a separate actor network. Foundational paper: Haarnoja, Zhou, Abbeel, Levine, "Soft actor-critic: off-policy maximum entropy deep reinforcement learning with a stochastic actor," ICML 2018 — cited directly, verbatim, in FedSTAR's own reference list (ref [31]).

**FedSTAR's specific implementation:** actor + critic networks, each an MLP with 2 hidden layers × 256 neurons, ReLU; replay buffer 10,000; batch 16; discount γ=0.99; soft-update coefficient 0.005.

**Why FedSTAR needed SAC (and our project didn't):** its action space genuinely mixes a discrete choice (5 clustering algorithms) with 3 continuous values (sampling ratio, aggregation weight, fusion coefficient) — a hybrid action space DQN cannot represent at all.

**Advantages:** far more expressive (true continuous control); can in principle find better solutions than any fixed catalogue of discrete options.
**Limitations:** more complex to implement/tune (multiple networks, an entropy coefficient); more computationally expensive — FedSTAR's own Table 6 shows its full system costs more training time (102.82 min) and higher inference latency (4.78 ms/batch) than a lighter baseline.

**Which papers use it:** FedSTAR (the only one of our 4 core papers using any RL at all). Search-found: FedAEB (IEEE, doc 10416384) also uses SAC for a non-satellite FL client-selection/resource-allocation problem — confirming SAC is a recurring, recognized choice for this exact class of problem, not something FedSTAR invented.

**Why it matters to our project:** the direct point of comparison for justifying "why DQN, not SAC" — our action space (9 fixed discrete profiles, no clustering, no continuous knobs) simply doesn't need SAC's extra expressive power; choosing DQN was a scoping decision, not a capability limitation.
