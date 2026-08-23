# Concept — FedAvg (Federated Averaging)

**What is it?** The original, simplest algorithm for combining model updates from many federated clients: average their weights, weighted by how much local data each client trained on.

**Simple example:** 8 satellite clients train locally; client A has 10,000 samples, client B has 2,000. The global model becomes (10000×A's weights + 2000×B's weights + ...) / total samples — clients with more data pull the average more toward themselves.

**Technical explanation:** `w_global = Σ_k (n_k/N) · w_k`, where `n_k` is client `k`'s local sample count and `N = Σ n_k`. Introduced by McMahan et al., "Communication-efficient learning of deep networks from decentralized data" (AISTATS 2017) — cited directly, verbatim, in FedSTAR's own reference list (ref [11]).

**Why papers use it:** it's the universal starting point/baseline every adaptive method (FedSTAR's SAC-driven aggregation, our own Phase 8 rule and Phase 9B DQN) is compared against — you can't claim an improvement without first measuring the thing you're improving on.

**Advantages:** simple, cheap, no extra hyperparameters, well understood, provably converges under IID data.

**Limitations:** performs poorly under Non-IID client data (see `non-iid-data.md`) — averaging clients with very different data distributions produces a "blurry compromise" model, not a genuinely good one for any client. This single limitation is the entire motivation for FedSTAR's clustering, and for our own project's Phase 8 (rule) and Phase 9B (DQN).

**Which papers use it:** FedSTAR (as its floor baseline); our project's Phase 6 (implemented exactly, unmodified).

**Why it matters to our project:** `src/federated/fedavg.py`'s `federated_average` function implements this formula exactly, unchanged, since Phase 6 — every later phase (7, 8, 9B) builds on top of it without modifying it.
