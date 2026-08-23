# Concept — LSTM (Long Short-Term Memory)

**What is it?** A type of recurrent neural network with internal "gates" (input, forget, output) that let it decide, at each time step, what to remember, what to forget, and what to output — solving plain RNNs' tendency to "forget" information from many steps ago.

**Simple example:** reading a long mystery novel and needing a clue from Chapter 1 to understand the ending in Chapter 20. A plain RNN tends to lose that clue along the way; an LSTM has a dedicated "notebook" (its cell state) it can choose to keep writing in or erase, so long-range clues survive.

**Technical explanation:** at each timestep `t`, given input `x_t` and previous hidden state `h_{t-1}`: forget gate `f_t = σ(W_f·[h_{t-1},x_t]+b_f)` decides what to discard from the cell state; input gate `i_t = σ(W_i·[h_{t-1},x_t]+b_i)` decides what new information to add; candidate cell state `C̃_t = tanh(W_C·[h_{t-1},x_t]+b_C)`; updated cell state `C_t = f_t*C_{t-1} + i_t*C̃_t`; output gate `o_t = σ(W_o·[h_{t-1},x_t]+b_o)`; final output `h_t = o_t*tanh(C_t)`. Originally introduced by Hochreiter & Schmidhuber, 1997.

**Why papers use it:** telemetry and network traffic are both time series with long-range dependencies (a command sent now may affect a reading many steps later); LSTMs handle this far better than plain RNNs.

**Advantages:** strong track record on sequential data; well-understood; widely supported in every deep learning framework.
**Limitations:** even with gating, very long sequences can still lose older information; computationally heavier per step than simpler alternatives (like GRU); doesn't natively model *spatial* relationships (why FedSTAR adds a separate spatial encoder before its temporal component).

**Which papers use it:** Hundman et al. (plain LSTM, 2 layers × 80 units, per channel); Yang et al. (double-layer LSTM + attention); Maleki Sadr et al. (Bayesian LSTM via MC Dropout, 3 layers). FedSTAR does **not** use LSTM at all — it uses a convolutional/attention-based spatio-temporal architecture (SimVPv2) instead.

**Why it matters to our project:** conceptually, LSTM is the direct ancestor of our own Phase 3/9B `TemporalGRUAttention` — we chose GRU (see `gru.md`) instead of LSTM, a lighter-weight cousin with fewer gates.
