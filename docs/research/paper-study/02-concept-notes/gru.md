# Concept — GRU (Gated Recurrent Unit)

**What is it?** A simplified, lighter-weight cousin of the LSTM — it merges the LSTM's separate forget/input gates into a single "update gate," and drops the separate cell state, using only the hidden state. Fewer parameters, faster to train, often comparably accurate.

**Simple example:** if an LSTM is a detailed notebook with three separate rules for what to write/erase/read, a GRU is a simpler notebook with two rules that do almost the same job with less bookkeeping overhead.

**Technical explanation:** update gate `z_t = σ(W_z·[h_{t-1},x_t])` decides how much of the past to keep vs. replace; reset gate `r_t = σ(W_r·[h_{t-1},x_t])` decides how much past information to ignore when computing a candidate state; candidate `h̃_t = tanh(W·[r_t*h_{t-1}, x_t])`; final hidden state `h_t = (1-z_t)*h_{t-1} + z_t*h̃_t`. Introduced by Cho et al., 2014.

**Why our project uses GRU (not LSTM, unlike Papers 1–3):** fewer parameters and faster training for a comparable ability to model sequences — a reasonable, standard efficiency trade-off, not a capability sacrifice for this project's scale.

**Advantages:** fewer parameters/faster training than LSTM; simpler to reason about (2 gates vs. 3 + a separate cell state); often performs comparably to LSTM on many tasks.
**Limitations:** on some tasks requiring very precise long-range memory control, LSTM's extra gating can outperform GRU — a known, task-dependent trade-off, not a universal ranking.

**Which papers use it:** none of the 4 core papers use GRU — all three telemetry papers use LSTM specifically, and FedSTAR uses neither (it uses ConvSC+gSTA/LKA instead). **Our project's own choice of GRU over LSTM is therefore an independent design decision, not something borrowed from any of the 4 papers.**

**Why it matters to our project:** `TemporalGRUAttention` (Phase 3, reused unchanged in Phase 9A/OPSSAT-AD) is built on GRU, not LSTM — worth being able to explain in viva as a deliberate efficiency choice, and to note explicitly that this differs from every telemetry paper in our literature set, which all use LSTM.
