# Concept — Client Selection (vs. Client Weighting)

**What is it?** Deciding, each federated round, *which* clients actually participate (selection) versus *how much influence* each participating client gets (weighting). These are two different mechanisms often confused with each other.

**Simple example:** 8 satellite clients exist. **Selection** = "only clients 1, 3, 5, 7 train this round, the others sit out." **Weighting** = "all 8 clients train and are aggregated, but client 3's update counts twice as much as client 6's."

**Technical explanation:** selection typically reduces the aggregation to a subset `S_t ⊂ {1..N}` chosen each round (e.g., FedSTAR's `ρ_t` sampling ratio within each cluster, which can effectively drop some clients from a round); weighting keeps the full client set but changes the coefficients `α_i` in `w_global = Σ α_i w_i`.

**Why papers use selection:** reduces communication/compute cost per round (fewer clients training/uploading); lets you deliberately favor higher-quality or more available clients.

**Why our project uses weighting instead:** documented, explicit reasoning (Phase 8/9B design docs) — hard selection risks silently dropping a client that happens to hold a rare attack category for an entire round, directly conflicting with the fairness goal Phase 7 demonstrated matters (a 41x fairness-gap swing was measured from heterogeneity alone).

**Advantages of selection:** lower communication/compute cost; can exclude clearly bad/stale clients entirely.
**Limitations of selection:** fairness risk (an excluded client's rare data never gets used that round); more complex bookkeeping.
**Advantages of weighting:** every client always contributes something; simpler fairness reasoning.
**Limitations of weighting:** doesn't reduce communication cost; a very bad client still costs bandwidth even if given near-zero weight.

**Which papers use it:** FedSTAR uses **selection within clusters** (`ρ_t` sampling ratio) — some clients in a cluster may not participate in a given round's aggregation. IEEE's "Adaptive Client Selection in Resource Constrained FL Systems: A DRL Approach" and several other search-found papers (Section 4/E of the full report) focus specifically on selection as the mechanism DRL controls.

**Why it matters to our project:** our project deliberately chose **weighting only** (Phase 8, Phase 9B) — all 8 clients participate every round — a disclosed, principled trade-off, not an oversight or a missing capability.
