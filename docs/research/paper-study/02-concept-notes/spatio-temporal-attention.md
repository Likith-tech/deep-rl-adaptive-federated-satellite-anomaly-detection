# Concept — Spatio-Temporal Attention (and LKA specifically)

**What is it?** Attention (see `attention.md`) applied jointly across **both** the spatial dimension (relationships *between* different features/variables at the same time) and the temporal dimension (relationships *across* time) — rather than handling each dimension separately (e.g., a CNN for space, then an LSTM for time, one after another).

**Simple example:** for satellite network traffic, a purely temporal model asks "how does *this one* feature change over time?" A spatio-temporal model additionally asks "how do *different* features relate to each other, and does *that relationship itself* change over time?" — e.g., "does a spike in feature A tend to precede a spike in feature B, and does this pattern happen only during certain kinds of attacks?"

**Technical explanation — FedSTAR's specific mechanism (Large Kernel Attention, LKA):** approximates the effect of a very large convolutional filter (one that can "see" far across a feature map, like a wide field of view) through a cheaper decomposition: a small depth-wise convolution (5×5, local detail) → a depth-wise **dilated** convolution (dilation rate 3, long-range context, still cheap) → a 1×1 point-wise convolution (channel-mixing) → a sigmoid gate multiplied element-wise back onto the original features. This gives a large effective receptive field (long-range spatial pattern recognition) without the enormous compute cost a literally-large convolution kernel would need.

**Why papers use it:** standard "process space, then time, serially" fusion (e.g., a CNN followed by an LSTM) can lose subtle *interactions* between the two dimensions — FedSTAR's own Introduction argues this limits detection of complex, multi-stage attack patterns.

**Advantages:** captures richer, coupled spatial-temporal patterns than serial fusion; LKA specifically is computationally cheaper than a literal large-kernel convolution while approximating its effect.

**Limitations:** more architecturally complex than either a plain temporal model (our own `TemporalGRUAttention`) or a plain spatial model alone; requires a genuinely multi-dimensional/multi-feature input to be worthwhile — less obviously beneficial for single-channel, purely temporal problems (like Papers 1–3's per-channel telemetry forecasting).

**Which papers use it:** only FedSTAR, among our 4 core papers (via its gSTA blocks and LKA mechanism). Search-found related work: several spatio-temporal graph neural network (STGNN) papers surfaced independently (Section 4/I of the full report) as an alternative architectural family for the same general goal, using graph structures instead of FedSTAR's pseudo-video/convolutional approach.

**Why it matters to our project:** our project currently has **no genuine spatio-temporal modeling** — this is an honest, disclosed architectural gap relative to FedSTAR, and a clearly-labeled **PROPOSED** (not planned) future extension, not something to claim is already present.
