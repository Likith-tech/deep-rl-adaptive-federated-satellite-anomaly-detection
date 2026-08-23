# CORE PAPER — Deep Dive: FedSTAR

**"Anomaly detection method for satellite networks based on adaptive federated learning driven by deep reinforcement learning"**
Hui Chen, Kun Zhang, Yangtao Chen, Xiaohe Wu, Chenxi Cai, Dianying Chen — *Information Sciences* (Elsevier), Vol. 755, Article 123808, 2026. DOI: 10.1016/j.ins.2026.123808.

> This is the single most important paper in your literature set — nearly identical title to your capstone, published the same year. This file is the deepest, most exhaustive analysis in the whole knowledge base, on purpose. A shorter version lives at `01-paper-notes/paper-04.md`; this file goes further, strictly using the paper's own actual architecture — nothing invented.

---

## 1. Satellite scenario

The paper models a **space-based LEO satellite network**: satellites communicate directly via Inter-Satellite Links (ISLs) with On-Board Processing (OBP), rather than relaying everything through ground stations. This matters because it means detection must, in principle, be able to run close to the satellite (on-board or at a nearby aggregation point), not only at a ground data center — motivating the paper's federated design. 20 simulated satellite-network clients are used in experiments; data is split IID or Non-IID (Dirichlet, η=0.5) across them.

## 2. Anomaly detection problem

Not a physical/telemetry fault — a **cyberattack / malicious network traffic** problem: satellite-terrestrial channels face DDoS, botnets, backdoors, web attacks, port scans, and (specific to this domain) attacks exploiting the **Delay-Tolerant Networking (DTN)** forwarding protocol used for inter-satellite store-and-forward communication, via SYN/UDP flood patterns. The paper frames this as a multi-class classification problem: given a traffic flow's features, output which category (normal, or one of several attack types) it belongs to.

## 3. Dataset

Three datasets (details already fully captured in `paper-04.md` §4 — repeated here for completeness of this dedicated file):
- **STIN**: 32 features, 209,564 instances, 9 attack categories including 2 satellite-specific DTN-protocol DDoS types. Purpose-built for satellite-terrestrial integrated networks.
- **TON_IoT**: 44 features, 211,043 instances, 10 categories — auxiliary, general IoT/edge benchmark.
- **Bot-IoT**: 46 features, >73 million instances, 5 categories — auxiliary, large-scale botnet/IoT benchmark.

**None are real, operational satellite traffic captures** — the paper's own Conclusion says so directly.

## 4. Preprocessing

**Pseudo-video projection** (the paper's own term): a 1D traffic feature vector `x ∈ ℝ^D` is mapped through a linear embedding layer + Layer Normalization + a learnable positional encoding `E_pos`, then `T=4` independent per-timestep MLP "View Generators" project it into `T` synthetic time-step feature maps `V_t ∈ ℝ^(C×H×W)` with `C=64` channels — stacking these `T` maps produces a pseudo-video tensor `X ∈ ℝ^(T×C×H×W)`. **Important nuance:** the paper is explicit that "this pseudo-video projection does not assume natural image-like neighborhoods among tabular traffic features; rather, it provides a learnable latent arrangement for subsequent spatio-temporal modeling" — i.e., the "video" structure is synthetic/learned, not a literal image of the traffic.

## 5. Federated architecture

Hierarchical, two-level: (a) **client level** — 20 satellite nodes, each running the full local detection model; (b) **server level** — a **meta-controller** running the SAC reinforcement-learning agent, which dynamically decides how to cluster clients and combine their updates each round. This is explicitly *not* a flat, single-level FedAvg system.

## 6. Clients

Each client trains its own copy of the local detection model (below) for several local epochs via AdamW (batch 64, lr 1e-4, weight decay 1e-4), then uploads **only its classifier-head parameters** (not the full model, not raw data) to the server for similarity computation — and its full model weights for the actual aggregation step.

## 7. Local training (the client-side model, in detail)

```text
1D traffic feature vector (D-dim)
        ↓
Pseudo-video projection → X ∈ ℝ^(T=4 × C=64 × H × W)
        ↓
Encoder: 2× ConvSC blocks, each = Conv2D (downsamples H,W) → GroupNorm → SiLU
        ↓
Translator: 4× gSTA blocks. Each gSTA block's core is Large Kernel Attention (LKA):
    F → DW-Conv (5×5 depth-wise, local context)
      → DW-D-Conv (depth-wise DILATED conv, dilation rate 3, long-range context)
      → PW-Conv (1×1 point-wise, channel fusion)
      → Sigmoid → element-wise multiply with F (gated attention)
        ↓
Classifier head: Global average pooling → Flatten → LayerNorm → GELU
      → Dropout(0.1) → 2-layer MLP → attack-category probabilities
```

## 8. Aggregation

Two-level, weighted, hierarchical:
1. **Intra-cluster:** for cluster `C_k`, sample `N_sample = max(1, ⌊|C_k|·ρ_t⌋)` clients, aggregate with weight `α_i = λ_t·(D_i / Σ D_l) + (1−λ_t)·(1/|C'_k|)` — a tunable blend between FedAvg-style (data-volume-weighted, `λ_t→1`) and simple-mean (`λ_t→0`) aggregation.
2. **Cluster fusion:** `θ^cluster_t = Σ|C_k|·θ^cluster_k / Σ|C_k|` (size-weighted average across clusters).
3. **Global-local momentum blend:** `θ^global_t = γ_t·θ^cluster_t + (1−γ_t)·θ^global_{t-1}` — controls how much the new round's clustered result overwrites vs. preserves the persisting global model.

## 9. Adaptive mechanism

The entire clustering configuration (which algorithm, how many clusters) **and** the three continuous aggregation knobs (`ρ, λ, γ`) above are **not fixed** — they are chosen fresh, every single communication round, by the SAC agent's action, based on the current state. This is what "adaptive" means precisely in this paper: not "the model adapts to new data" but "the *federated process itself* — how clients are grouped and combined — adapts round to round."

## 10. DRL algorithm: Soft Actor-Critic (SAC)

**What:** an off-policy, maximum-entropy deep reinforcement learning algorithm that learns both an actor (policy) and (typically twin) critic networks, explicitly balancing reward maximization against maintaining exploration entropy — and, in this implementation, formulated to output a **hybrid** discrete + continuous action.

**Why SAC specifically (not DQN/PPO):** the action space genuinely mixes a discrete choice (clustering algorithm) with continuous values (sampling ratio, aggregation weight, fusion coefficient) — DQN cannot output continuous values at all; SAC natively can.

**Network sizes:** both actor and critic are MLPs with 2 hidden layers × 256 neurons, ReLU activation.

**Hyperparameters:** replay buffer 10,000, batch size 16, discount factor γ=0.99, soft-update coefficient 0.005 — fixed across all experiments.

## 11. State (the RL environment's observation)

`s_t = concat(V_sim, s_loss, s_acc, v_algo, s_k)`:
- `V_sim = vectriu(S_t)`: upper-triangular vectorization of the `N×N` pairwise **cosine similarity matrix** of clients' classifier-head parameter vectors, `S_i,j = (w_i·w_j)/(‖w_i‖‖w_j‖)`. Reduces dimensionality from O(N²) to O(N²/2) while keeping every unique pairwise relationship.
- `s_loss = tanh(L_{t-1})`, `s_acc = A_{t-1} ∈ [0,1]`: previous round's global loss/accuracy, as training feedback.
- `v_algo = OneHot(m_prev)`: the clustering algorithm chosen last round.
- `s_k = (K_prev − K_min)/(K_max − K_min)`: last round's cluster count, normalized.

**Why classifier-head parameters specifically (not the whole model, not raw activations):** the paper cites empirical evidence (from the client-clustering literature it references) that final/classifier layers reflect local-task/data differences more strongly than early layers — so this is both cheaper to compute *and* arguably more informative for detecting heterogeneity than full-model comparison.

## 12. Action

`a_t = {a_d_t, a_c_t}`:
- **Discrete** `a_d_t = {m_t, k_t}`: `m_t ∈ {FedAvg, K-Means, Spectral, Agglomerative, DBSCAN}`, `k_t ∈ {1, 2, ..., 6}`.
- **Continuous** `a_c_t = [ρ_t, λ_t, γ_t]^T ∈ [0,1]^3`.

**How the discrete part is actually produced:** the SAC policy outputs logits over the algorithm pool and candidate cluster numbers; `m_t` is sampled from the categorical distribution during training, chosen by argmax during evaluation. The continuous part: three policy outputs passed through a sigmoid, giving values in `[0,1]`.

## 13. Reward

`r_t = R_gain − P_degrade − P_div`:
- `R_gain = λ_α · (A_t − A_{t−1}) · 100` — reward for accuracy improvement (`λ_α=1.0`).
- `P_degrade = λ_β · |A_t − A_{t−1}| · 100` **if** `A_t < A_{t−1}`, else 0 — a penalty specifically for regression, not symmetric with the gain term (`λ_β=1.0`).
- `P_div = λ_γ · (1 − S̄_t)`, where `S̄_t` is the mean intra-cluster similarity across all `K` clusters — penalizes clusters whose members end up internally dissimilar (`λ_γ=0.0001`, deliberately tiny relative to the other two terms).

**Why `λ_γ` is so much smaller:** the divergence penalty is meant as a light, secondary shaping signal (avoid grossly incoherent clusters), not to dominate the accuracy-driven terms — this is a deliberate, disclosed design choice, not an oversight.

## 14. RL environment

One **timestep** = one full communication round (similarity computation → SAC action → clustering/sampling/aggregation → evaluation → reward). The environment transition is: `s_t --(SAC picks a_t)--> cluster & aggregate --> evaluate new global model --> r_t --> s_{t+1}` (new similarity matrix + this round's loss/accuracy + this round's chosen algorithm/k as memory). Not explicitly stated as having a fixed "episode" boundary in the way your own project's Phase 9B does (one episode = one full FL run of N rounds) — **the paper's own text treats the entire 50-round training run as one continuous learning process for the SAC agent**, not a repeated multi-episode structure. **Not verified from the available source** whether the paper explicitly uses the word "episode" — its own Algorithm 1 shows a single loop over rounds `t=1..T`, without an outer episode loop.

## 15. Client selection

Not hard "drop this client" selection — clients are always first assigned to *some* cluster; within each cluster, a **sample** of size `⌊|C_k|·ρ_t⌋` participates in that round's aggregation for that cluster (so, in principle, some clients within a cluster could be excluded from a given round's aggregation, unlike your own project's "all 8 clients, every round" design).

## 16. Resource allocation

**Not present as a distinct mechanism.** Unlike your own project's Phase 8/9B `resource_k` signal (simulated bandwidth/latency/compute/availability/connectivity), FedSTAR's state/action has no explicit resource-availability dimension — its adaptivity is entirely about *data similarity/heterogeneity*, not client resource constraints. This is a genuine, useful point of difference to name explicitly.

## 17. Communication process

Classifier-head parameters uploaded every round (for similarity computation); full model parameters uploaded/downloaded for the actual aggregation and redistribution. Communication cost is explicitly measured: 372.02 MiB (FedSTAR) vs. 445.99 MiB (FLOGA-AD) per the paper's own Table 6 comparison — lower than its closest competitor, attributed to the clustering step's effect on aggregation efficiency.

## 18. Anomaly detection (inference time)

A trained global model's classifier head outputs a probability distribution over categories (normal + attack types) for a given traffic sample — standard supervised multi-class classification at inference time, no separate residual/threshold step (unlike Papers 1–3's forecasting-based approach).

## 19. Evaluation

Accuracy, Precision, Recall, F1 (weighted); paired t-tests for statistical significance (5 seeds: 2024–2028); Jensen-Shannon-divergence-style attack-phase-behavior analysis; ablation study (8 variants: alternative backbones, fixed clustering, no clustering pool, no divergence penalty, full-model similarity instead of classifier-head, no gSTA, no LKA); parameter-sensitivity analysis (Dirichlet η, client count, communication rounds, reward-weight groups); system-level efficiency table (params, training time, model size, FLOPs, latency, communication cost).

## 20. Results

Non-IID accuracy: **STIN 95.43%, TON_IoT 89.16%, Bot-IoT 99.88%** — best of 8 compared methods. Statistically significant vs. DeepFed (IID, all 3 datasets, p<0.05). Vs. FLOGA-AD (Non-IID, its strongest competitor): significant **only on Bot-IoT** (p=6.00×10⁻⁴); **not significant** on STIN (p=0.080) or TON_IoT (p=0.519) — disclosed honestly. Variance decomposition: `R_B=0.12` (FedSTAR) vs. `0.00` (FedAvg) — proving clustering reveals real between-cluster heterogeneity FedAvg mathematically hides. Cost: heavier than the lightest baselines (102.82 min training vs. 87.83 min for FLOGA-AD; 4.78 ms/batch vs. 3.08 ms/batch), but lower communication (372.02 vs. 445.99 MiB) and FLOPs (96.2M vs. 170.0M) than FLOGA-AD specifically.

## 21. Limitations (self-disclosed by the paper's own Conclusion)

- No real operational satellite traffic used anywhere.
- Orbital networking realism (delayed updates, intermittent links, stale client states, time-varying link availability) not modeled.
- Heavier/slower than lightweight federated baselines — may not suit tightly power/latency-constrained satellites.
- **"The adversarial robustness of the federated optimization process itself is left for future investigation"** — stated in both the Introduction and the Conclusion, i.e., a deliberate, acknowledged scope boundary, not an oversight.

---

## The complete system, end to end (actual FedSTAR architecture — nothing invented)

```text
Input: 1D satellite-network traffic features (per client)
        ↓
Satellite Clients (20, simulated IID/Non-IID via Dirichlet η=0.5)
        ↓
Local Models: pseudo-video projection → ConvSC spatial encoder → gSTA/LKA
              spatio-temporal translator → classifier head (per client, trained
              locally via AdamW for several epochs)
        ↓
Federated Process: clients upload classifier-head params (similarity) +
              full model (aggregation) to the server
        ↓
DRL Decision: server computes cosine-similarity matrix → builds state s_t
              → SAC agent (actor+critic, 256×2 MLPs) samples hybrid action
              a_t = {clustering algorithm, k, ρ, λ, γ}
        ↓
Adaptive Aggregation/Selection: cluster clients per a_t → sample ρ_t of each
              cluster → weighted intra-cluster aggregation (λ_t) → cluster
              fusion (size-weighted) → global-local momentum blend (γ_t)
        ↓
Global Model: updated θ^global_t, evaluated on validation data → accuracy/loss
              → reward r_t computed → SAC learns (actor+critic gradient step)
              → sent back to all clients for next round
        ↓
Anomaly Detection (inference): classifier head → attack-category probabilities
              for new incoming traffic samples
```

This is the paper's *actual* Fig. 2 architecture, transcribed faithfully — not an invented generic FL+RL diagram.
