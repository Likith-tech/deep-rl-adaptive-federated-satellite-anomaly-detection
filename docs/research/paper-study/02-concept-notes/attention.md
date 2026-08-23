# Concept — Attention Mechanism

**What is it?** A mechanism that lets a model assign different learned "importance weights" to different parts of its input (e.g., different past time steps) when producing its current output, instead of treating all input equally or relying purely on a recurrent unit's built-in memory.

**Simple example:** when answering a question about a long meeting transcript, you don't recall every sentence with equal weight — you focus more on the parts most relevant to the question being asked. Attention lets a network do the same thing automatically, with learned (not hand-programmed) weights.

**Technical explanation (standard additive/learned-weight attention, as in Yang et al.):** given a sequence of hidden states `h_1, ..., h_T` (from an LSTM/GRU), attention computes a weight `α_t` for each hidden state (often via a small learned scoring function + softmax), then produces a context vector as the weighted sum `c = Σ α_t · h_t`. This context vector, not just the last hidden state, feeds into the final prediction.

**Why papers use it:** a plain recurrent unit's own gating can still lose older/less-recent information over long sequences; attention lets the network explicitly learn which specific historical moments matter most for the current decision.

**Advantages:** improves use of long-range information; provides some interpretability (you can inspect which time steps got high attention weight); consistently improves performance in Yang et al.'s controlled comparison (Table 1: attention improves recall from 71.15%/72.12% to 80.77%/82.69% when combined with a second LSTM layer).

**Limitations:** adds parameters/computation over a plain recurrent model; doesn't by itself solve spatial modeling (see `spatio-temporal-attention.md` for how FedSTAR's LKA differs) or provide uncertainty estimates.

**Which papers use it:** Yang et al. 2021 (attention-over-LSTM, the clearest, most standard example in our set); FedSTAR (LKA — a very different, convolution-based attention mechanism, see `spatio-temporal-attention.md`). Hundman et al. and Maleki Sadr et al. use plain LSTMs with **no attention**.

**Why it matters to our project:** our own `TemporalGRUAttention` (Phase 3/9A) uses a standard attention mechanism conceptually closer to Yang et al.'s design than to FedSTAR's LKA — worth being precise about which "kind" of attention is being discussed in viva, since the term covers genuinely different mechanisms across these papers.
