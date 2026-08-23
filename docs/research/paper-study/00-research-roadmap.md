# Research Roadmap — Recommended Study Order

**Honesty note:** you have 4 fully-read papers (not 10 — see `README.md`), plus ~28 independently-found candidate papers of varying confidence (`07-missing-papers/recommended-additional-papers.md`). This roadmap is built around what those 4 papers + the concept notes can actually teach you, expanded into the 10-stage shape you asked for — not artificially stretched or compressed to hit a number.

```text
Stage 1  — Understand satellite anomaly detection (the field split)
Stage 2  — Understand deep learning anomaly detection (the telemetry lineage)
Stage 3  — Understand temporal models (LSTM/GRU/attention)
Stage 4  — Understand federated learning
Stage 5  — Understand Non-IID FL
Stage 6  — Understand adaptive FL
Stage 7  — Understand reinforcement learning
Stage 8  — Understand DRL-based FL
Stage 9  — Study FedSTAR deeply
Stage 10 — Understand our research gap and project positioning
```

---

### Stage 1 — Understand satellite anomaly detection (30–40 min)

**Read:** `02-concept-notes/satellite-networks.md`, `02-concept-notes/anomaly-detection.md`, `03-comparison/research-evolution.md`

**Why:** before any paper makes sense, you need the single biggest idea in this whole knowledge base: "satellite anomaly detection" actually means two nearly unrelated things — physical/telemetry faults on one spacecraft (Papers 1–3) vs. malicious network traffic across many satellites (FedSTAR) — and your 4 papers split 3-vs-1 across that line.

**After this stage, you should understand:** why 3 of your 4 papers have zero FL/RL while the 4th has both — it's a different sub-problem, not a difference in sophistication.

---

### Stage 2 — Understand deep learning anomaly detection: the telemetry lineage (2–2.5 hrs)

**Read, in this exact order (each paper improves on the last):**
1. `01-paper-notes/paper-01.md` (Hundman et al., 2018) — LSTM forecasting + nonparametric dynamic thresholding; origin of the SMAP/MSL dataset.
2. `01-paper-notes/paper-02.md` (Yang et al., 2021) — adds attention + multi-scale (local+global) detection.
3. `01-paper-notes/paper-03.md` (Maleki Sadr et al., 2022) — adds uncertainty estimation via Monte Carlo Dropout.

**After this stage, you should understand:** the forecast → residual → data-driven-threshold recipe cold, why a Gaussian threshold assumption is statistically wrong for real telemetry, and what each successive paper specifically added.

---

### Stage 3 — Understand temporal models (1–1.5 hrs)

**Read:** `02-concept-notes/lstm.md`, `02-concept-notes/gru.md`, `02-concept-notes/attention.md`, `02-concept-notes/temporal-modeling.md`

**Why:** these are the actual mechanical building blocks behind Stage 2's papers and your own project's Phase 3 architecture — worth knowing precisely, not just "LSTM = good for sequences."

**After this stage, you should understand:** exactly how an LSTM's gates work, why GRU is a lighter alternative (and why your own project chose GRU, unlike any of the 3 telemetry papers), and what attention mathematically adds on top.

---

### Stage 4 — Understand federated learning (1–1.5 hrs)

**Read:** `02-concept-notes/federated-learning.md`, `02-concept-notes/fedavg.md`, `02-concept-notes/client-selection.md`

**After this stage, you should understand:** why raw data never leaves a client, exactly what FedAvg's formula does, and the difference between client *selection* and client *weighting* (a distinction your own project deliberately resolves one specific way).

---

### Stage 5 — Understand Non-IID FL (45 min – 1 hr)

**Read:** `02-concept-notes/non-iid-data.md`

**After this stage, you should understand:** why plain FedAvg breaks down when clients' data genuinely differs, how Dirichlet partitioning simulates this in a controllable way, and your own project's own finding that KDDTest+ F1 is *non-monotonic* in heterogeneity — a real, own-measured nuance no paper in your set reports.

---

### Stage 6 — Understand adaptive FL (45 min – 1 hr)

**Read:** `02-concept-notes/adaptive-federated-learning.md`, `02-concept-notes/communication-efficiency.md`

**After this stage, you should understand:** the difference between a *rule-based* adaptive mechanism (fixed formula, Phase 8) and a *learned* one (Phase 9B) — and why your project's Phase 8 ablation result (no single fixed rule wins on both F1 and fairness) is the direct empirical reason Phase 9B was attempted at all.

---

### Stage 7 — Understand reinforcement learning (1–1.5 hrs)

**Read:** `02-concept-notes/reinforcement-learning.md`, `02-concept-notes/state-action-reward.md`

**After this stage, you should understand:** state/action/reward/policy precisely, with FedSTAR's and your own project's exact definitions of each, side by side.

---

### Stage 8 — Understand DRL-based FL (1.5–2 hrs)

**Read:** `02-concept-notes/deep-reinforcement-learning.md`, `02-concept-notes/dqn.md`, `02-concept-notes/sac.md`, `02-concept-notes/spatio-temporal-attention.md`

**After this stage, you should understand:** why a neural network (not a lookup table) is needed to represent the policy at this scale, exactly how DQN (your project) and SAC (FedSTAR) differ mechanically, and what FedSTAR's LKA spatio-temporal attention mechanism does that your project's architecture currently doesn't attempt.

---

### Stage 9 — Study FedSTAR deeply (1.5–2 hrs)

**Read, in this order:**
1. `01-paper-notes/core-paper-fedstar.md` ⚠️ **the deepest single file in this entire knowledge base — read it slowly**
2. `01-paper-notes/paper-04.md` (shorter companion version, useful for a second pass/quick review)

**After this stage, you should understand:** FedSTAR's complete architecture end to end — pseudo-video projection, ConvSC encoder, gSTA/LKA translator, SAC state/action/reward, hierarchical aggregation — well enough to redraw its full pipeline from memory.

---

### Stage 10 — Understand our research gap and project positioning (1.5–2 hrs)

**Read, in this order:**
1. `03-comparison/literature-matrix.md`
2. `04-research-gap/identified-gaps.md`
3. `04-research-gap/paper-vs-our-project.md` ⚠️ **the most rigorous, most viva-relevant comparison in this folder**
4. `04-research-gap/our-project-positioning.md`
5. `05-project-connection/paper-to-project-mapping.md`
6. `08-full-literature-review-report.md` §9 (Proposed Novelty) and the final "answer to the professor's question" section

**After this stage, you should understand:** exactly what already exists, exactly what your project adds, exactly what remains only planned/proposed vs. actually implemented, and be able to state — without hedging — the one-sentence answer to "your project is very similar to FedSTAR, what did you add?"

---

## Then, before your viva: `06-viva-preparation/`

`simple-explanations.md` → `paper-based-questions.md` → `likely-questions.md`, ideally the day before.

## Total estimated time: 10–14 hours across the 10 stages — spread across multiple days, not one sitting.
