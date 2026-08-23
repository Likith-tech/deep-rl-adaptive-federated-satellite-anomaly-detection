# Concept — Non-IID Data (and the Dirichlet Distribution)

## 1. What is it?

> **Technical term → simple meaning → example**
> **IID (Independent and Identically Distributed):** every client's local data looks statistically like a random sample from the same overall population.
> **Non-IID:** clients' local data distributions genuinely differ from each other — different clients see different mixes of classes/patterns.
> *Analogy:* IID is like 20 grocery stores each getting a random, representative slice of a country's shopping habits. Non-IID is like 20 grocery stores in 20 different neighborhoods, each seeing very different buying patterns (one mostly sees vegetarian shoppers, another mostly sees families with young kids) — averaging their sales data blindly gives a "national average" that doesn't describe any real store well.

**Dirichlet distribution / Dirichlet partitioning:** a standard, controllable way to *simulate* Non-IID data from one centralized dataset, by randomly splitting it across clients so that each client ends up with a different, skewed mix of classes. A single number (concentration parameter, often called α or η) controls how skewed it is — low α = very skewed (highly Non-IID), high α = close to uniform (close to IID).

## 2. Why does it exist as a research problem?

In the real world, no two satellites (or hospitals, or phones) see identical data. FedAvg's simple averaging assumption quietly relies on clients being similar enough that averaging makes sense — when that assumption breaks, the combined model can end up worse than some individual clients' own local models.

## 3. Simple example

8 simulated satellite clients, each seeing a different mix of "attack categories" in their local network traffic — one client sees mostly DDoS, another sees almost no rare attack types at all. Plain averaging blurs all 8 into one model that may badly underperform on whichever attack type a specific client actually needs to catch.

## 4. How it's measured (both Paper 4 and your project use the same idea)

**Jensen-Shannon (JS) divergence / distance** between clients' class distributions — a number between 0 (identical) and 1 (maximally different) — used to *measure* Non-IID-ness rather than just assume it. Your project's own Phase 4 measured this directly: average pairwise JS distance of 0.527 across its 8 simulated satellite clients (α=0.5).

## 5. Why the papers/project use it

Paper 4 uses Dirichlet partitioning (η=0.5) to build its Non-IID experimental condition on STIN/TON_IoT/Bot-IoT. Your project uses the exact same technique (`src/simulation/partitioner.py`, Dirichlet, α=0.5 for the main partition, plus a full α-sweep in Phase 7: 0.1/0.5/1.0/5.0/10.0) to build its own 8 simulated satellite clients from NSL-KDD.

## 6. Advantages (of Dirichlet partitioning as a research tool)

- Fully controllable, reproducible severity of heterogeneity (one parameter).
- Lets you isolate heterogeneity as a single experimental variable, exactly what your project's Phase 7 did (same FedAvg pipeline, only α varied).

## 7. Limitations

- It's a **simulation**, not real heterogeneity — real satellites' data differences come from orbit, mission, region, sensor drift, etc., not a random statistical knob. Both your project and Paper 4 disclose this honestly.
- Your project's own Phase 7 found a genuinely important, counterintuitive result worth remembering: **more heterogeneity doesn't always monotonically hurt accuracy** — KDDTest+ F1 peaked at α=5.0, not at the least-heterogeneous α=10.0, while the *client fairness gap* did shrink monotonically as α increased. This shows "Non-IID hurts performance" is not automatically true for every metric — a nuance worth having ready for viva.

## 8. Which papers use it

Paper 4 (Chen/FedSTAR) explicitly. Papers 1–3 don't use it at all — there's no multi-client concept in those papers.

## 9. How it connects to our project

Non-IID handling is the **entire motivation** for your project's Phase 8 (rule-based adaptive weighting) and Phase 9B (DQN-driven adaptive weighting) — both exist specifically because plain FedAvg (Phase 6) doesn't adapt to which clients currently deserve more or less trust in the aggregation. This is the same underlying motivation Paper 4's SAC-based clustering addresses, just with a different mechanism (clustering + hybrid continuous/discrete SAC control vs. your project's fixed discrete DQN weight-profile selection).
