# Concept — Temporal Modeling

**What is it?** Building a model that explicitly accounts for the *order and timing* of data points, not just their values in isolation — necessary whenever "what happened before" changes the meaning of "what's happening now."

**Simple example:** a telemetry reading of 21°C means something different depending on whether the previous 10 readings were rising sharply (concerning) or stable (normal) — a model that only looks at the single 21°C value in isolation can't tell the difference; a temporal model, looking at the sequence, can.

**Technical explanation:** temporal models (RNN, LSTM, GRU, Transformer-based, or a "pseudo-video" trick like FedSTAR's) process a sequence `x_1, ..., x_T` and produce an output that depends on the whole sequence's structure, not just the most recent value — typically via recurrence (LSTM/GRU) or attention over the sequence (Transformers, or GRU+attention hybrids).

**Why papers use it:** both sub-fields in our literature need it, for different reasons — telemetry papers (1–3) need it because normal spacecraft behavior has trends/periodicity that a single-timestep model would ignore; FedSTAR needs it (via its pseudo-video trick) because network traffic patterns unfold over time (a multi-stage attack looks different moment to moment).

**Advantages:** dramatically better at catching anomalies that only become visible *over time* (e.g., a slow drift, or a multi-step attack sequence) — exactly the "contextual anomaly" category Hundman et al. define and specifically target.

**Limitations:** requires more data/compute than a single-timestep model; a genuinely temporal model needs genuinely sequential input to be worthwhile — this is precisely why our own project's Phase 9A/OPSSAT-AD run, using `sequence_length=1`, is honestly disclosed as **not yet exercising genuine temporal modeling**, despite using an architecture (`TemporalGRUAttention`) built for it.

**Which papers use it:** Hundman et al. (LSTM, sequence length 250); Yang et al. (double-layer LSTM); Maleki Sadr et al. (Bayesian LSTM); FedSTAR (pseudo-video, T=4 synthetic timesteps).

**Why it matters to our project:** genuinely exercising temporal modeling on Track B (running OPSSAT-AD with `sequence_length > 1`, using the raw multi-timestep telemetry rather than the pre-summarized per-segment feature vector) is a concrete, currently-**PROPOSED** (not committed/planned) extension worth naming explicitly in future work.
