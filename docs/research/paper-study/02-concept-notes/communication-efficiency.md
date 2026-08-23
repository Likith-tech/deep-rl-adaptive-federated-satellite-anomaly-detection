# Concept — Communication Efficiency in Federated Learning

**What is it?** How much data must actually be transmitted between clients and the server to complete federated training — a real, measurable cost distinct from accuracy, especially important for satellites (limited, expensive, sometimes intermittent bandwidth).

**Simple example:** if each of 8 satellite clients sends a full copy of its model (say, ~100KB) to the server every round, over 10 rounds that's `8 × 100KB × 10 = 8MB` of uplink traffic alone — a real, budgetable cost, before any accuracy consideration.

**Technical explanation:** typically measured in MiB/round or total MiB for a full training run; techniques to reduce it include gradient/weight compression (quantization, sparsification — e.g., "top-K" sparsification keeping only the largest gradient values), sending only certain layers (like FedSTAR sending classifier-head parameters for similarity computation, separate from full-model aggregation), or reducing how many clients participate per round (client selection/sampling).

**Why papers use it as a metric:** it's a direct, practical deployment cost, independent of accuracy — a highly accurate method that costs 10x the communication of a slightly-less-accurate one may not be deployable on a real bandwidth-constrained satellite link.

**Advantages of measuring/optimizing it:** makes accuracy-vs-cost trade-offs explicit and comparable across methods; directly relevant to the "why federated learning at all" motivation (avoiding raw-data transmission).

**Limitations:** reducing communication often trades against something else — accuracy, latency, or fairness (e.g., sampling fewer clients per round to save bandwidth is exactly the selection-vs-weighting trade-off discussed in `client-selection.md`).

**Which papers use it:** FedSTAR explicitly measures and compares communication cost (372.02 MiB vs. FLOGA-AD's 445.99 MiB, Table 6) as one of its reported advantages, attributing the improvement to its clustering step. None of Papers 1–3 have any client/server structure, so this metric doesn't apply to them. Search-found: "Towards Efficient Communications in Federated Learning: A Contemporary Survey" (arXiv:2208.01200) is a good general background anchor for this sub-field, covering quantization/sparsification techniques neither FedSTAR nor our project currently uses.

**Why it matters to our project:** our project measures this too (~14.6 MB/10 rounds), but — unlike FedSTAR — this figure is **unchanged** across Phases 6 through 9B, because our project's weighting-only (never selection) design means all 8 clients always transmit a full update every round; reducing communication further would require either compression techniques or moving to a selection-based design, both currently **PROPOSED**, not implemented.
