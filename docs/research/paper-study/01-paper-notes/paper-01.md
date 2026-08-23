# Paper 3 — Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding

> **Read this one first.** Chronologically and conceptually, this is the root of the "telemetry anomaly detection" branch of your literature. Papers 1 and 2 in this folder both build on the dataset and the general idea this paper introduced.

**Authors:** Kyle Hundman, Valentino Constantinou, Christopher Laporte, Ian Colwell, Tom Soderstrom (NASA Jet Propulsion Laboratory / Caltech)
**Year:** 2018
**Venue:** KDD '18 (24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining) — a top-tier data mining conference. Also on arXiv (1802.04431).
**Source file:** `1.Lstms_and_NDT.pdf`

---

## 1. One-Minute Explanation

NASA JPL engineers had a real, practical problem: thousands of spacecraft telemetry channels (temperature, power, radiation, etc.) had to be watched by human operators for anything abnormal, and the existing method — "alarm if the value goes outside a fixed safe range" — misses anything that looks fine in isolation but is wrong *in context* (e.g., a temperature that's technically in-range but wrong for what the spacecraft is currently doing). This paper trains one small LSTM neural network **per telemetry channel** to predict "what should this channel's next value look like, given its recent history and the commands sent to the spacecraft?" When the real value differs a lot from the prediction, that's a candidate anomaly. The paper's real novelty isn't the LSTM itself (LSTMs already existed) — it's a **fully automatic, unsupervised way to decide how big a prediction error has to be before it counts as "anomalous,"** without a human ever setting a fixed threshold, plus a way to prune out noise-driven false alarms. They tested this on real, expert-labeled anomaly reports from two real NASA missions (SMAP satellite, MSL/Curiosity rover) and then actually deployed a pilot version on the real SMAP mission.

---

## 2. The Problem

- Spacecraft have **thousands of telemetry channels**. A human cannot watch all of them well.
- The standard approach ("out-of-limits" / OOL: alarm if value > X or < Y) is simple but **blind to context** — Section 2 of the paper calls these missed cases *contextual anomalies*: a value that is not extreme on its own, but is wrong given what else is happening.
- Other fancier approaches (clustering, nearest-neighbor, expert systems) exist but each has its own weakness: expensive to compute at scale, need constant re-tuning, or need expert knowledge that's costly to maintain.
- Crucially: **however good your prediction model is, you still need to decide "how wrong is too wrong?"** Most other systems either use a manually-tuned fixed threshold, or assume prediction errors are Gaussian-distributed (bell-curve shaped) so they can use "N standard deviations away from the mean" — but the paper shows (Section 4.3) that this Gaussian assumption is **statistically false** for real telemetry error data.

**Simple example:** Imagine your body temperature. 37°C is normal at rest. If a fixed-threshold alarm only fires above 39°C, it will miss a 37.8°C reading that is actually abnormal *because you're supposed to be resting, not exercising* — that's the "contextual" case this paper targets.

---

## 3. What Is the Main Idea?

**Input → Processing → Model → Output**

```text
Per-channel telemetry history + one-hot encoded spacecraft commands
                    ↓
     LSTM (2 hidden layers, 80 units) — trained to PREDICT
     the next telemetry value from the recent window
                    ↓
   prediction error e(t) = |actual - predicted|, smoothed with EWMA
                    ↓
   Nonparametric Dynamic Thresholding
   (finds the error cutoff that, if you removed everything above it,
   would most shrink the mean & std of the remaining errors —
   no Gaussian assumption, no fixed number)
                    ↓
        Pruning (remove anomalies that are too close, in
        percentage terms, to the next-largest "normal" error —
        i.e. kill noise-driven false alarms)
                    ↓
         Final list of anomalous time segments, each with
         a severity score
```

One separate, independent LSTM model is trained **per channel** — not one giant model for the whole spacecraft. This was a deliberate design choice (see Section 6).

---

## 4. Dataset

- **Source:** NASA's own SMAP satellite (Soil Moisture Active Passive) and the MSL Mars rover (Curiosity) — real operational telemetry from real missions.
- This paper is the **origin** of the SMAP/MSL benchmark dataset that Papers 1 and 2 in this folder both use (Paper 2 cites it directly as "the dataset in [7]").
- **Labels** were built by mining real **Incident Surprise, Anomaly (ISA) reports** — official documents mission engineers write when something actually went wrong — and hand-labeling the exact time range in the telemetry where each anomaly is visible.
- **Scale (Table 1 of the paper):** 82 unique telemetry channels total (55 SMAP + 27 MSL), 105 labeled anomaly sequences (69 SMAP + 36 MSL), ~496,444 individual telemetry values evaluated.
- **Anomaly type split:** 62 *point* anomalies (obviously wrong even without context — 59%) and 43 *contextual* anomalies (wrong only in context — 41%).
- **Train/test:** for each incident, a 5-day window around it is used; the model is trained on the 2 days before that window using only that channel's own history + command data — genuinely unsupervised at the modeling stage (labels are only used afterward, to *score* the detector, never to train it).

**What one "sample" represents:** one univariate time series (a single telemetry channel's readings over time), paired with one-hot-encoded metadata about which spacecraft commands were sent/received during the same period. The model is not shown "this is an anomaly" during training — training data intentionally excludes anomalous periods.

---

## 5. Features / Input Data

| Feature | What it represents | Why it matters |
|---|---|---|
| Prior telemetry values of the channel (sequence length 250) | The channel's own recent trajectory | LSTM needs history to predict what "normal" looks like next |
| One-hot encoded command info (module + sent/received) | Which spacecraft subsystem was commanded, and whether the command was sent or acknowledged received | Telemetry often changes *because* of a command — without this, the model would flag normal command-driven changes as anomalies |

Only **one target dimension** is predicted per model (the channel's own next value) — deliberately kept simple; the paper explains that LSTMs get much harder to train well as the number of predicted dimensions grows, so they scale via "many small models," not "one big model."

---

## 6. Methodology

```text
Step 1 → Collect real telemetry + command history for one channel
Step 2 → Train one small LSTM per channel to predict the NEXT
          telemetry value from a window of the recent past
          (training data intentionally excludes known-anomalous periods)
Step 3 → On new (test) data, compute prediction error e(t) = |real - predicted|
Step 4 → Smooth e(t) with an Exponentially Weighted Moving Average
          (kills noise spikes that LSTMs create around sudden but normal changes)
Step 5 → Nonparametric Dynamic Thresholding: search over candidate
          thresholds (mean + z*std, z from 2 to 10) and pick the one that
          maximizes the % drop in mean/std of the remaining "normal" errors,
          penalized for flagging too many points/sequences
Step 6 → Pruning: walk down the list of anomaly-candidate sequences from
          most to least severe; if a sequence's severity is not at least
          p% higher than the next one down, reclassify it as normal
          (this specifically targets false positives caused by ordinary
          spiky/noisy channel behavior)
Step 7 → Score each surviving anomaly sequence by how far its worst point
          is from the threshold
Step 8 → Evaluate against real ISA-derived labels: Precision, Recall, F0.5
```

---

## 7. Model / Algorithm

### LSTM (Long Short-Term Memory network)

**Technical term → simple meaning → example**
> **LSTM:** A type of recurrent neural network built with internal "gates" that decide what to remember, what to forget, and what to output at each time step — so it can learn patterns that span many time steps without the gradient vanishing.
> *Analogy:* Imagine reading a long book and needing to remember one clue from Chapter 1 to understand the ending in Chapter 20. A plain RNN tends to "forget" that clue along the way; an LSTM has a dedicated notebook (the cell state) it can choose to keep writing in or erase, so long-range clues survive.

**Why the authors used it:** telemetry is a time series with trends and delayed effects (a command sent now may affect a temperature reading many steps later). LSTMs are specifically good at this kind of long-range dependency, more so than plain RNNs.

**What goes in / what comes out:** a window of past telemetry + command one-hot vectors goes in; a single predicted scalar value (the next telemetry reading) comes out.

**Architecture used:** 2 hidden layers, 80 units each, dropout 0.3, sequence length 250, trained with Adam, up to 35 iterations with early stopping — deliberately small and shallow (the paper explicitly says bigger models gave "little to no prediction benefit" while costing more to train).

### Nonparametric Dynamic Thresholding (the paper's real contribution)

**Simple explanation:** Instead of a human choosing a fixed cutoff, or assuming errors follow a bell curve, the algorithm tries a range of candidate cutoffs and picks the one that, if you drew a line there, would make the "normal" side of the data noticeably tighter (lower mean and lower spread) — while being penalized for declaring too many points anomalous, so it doesn't just cheat by drawing the line at zero.

**Technical explanation:** ε = argmax over z ∈ {2.5, 3.0, ..., 10.0} of `((Δμ/μ) + (Δσ/σ)) / (|anomalous points| + |anomalous sequences|)`, where Δμ and Δσ are the drop in mean/std caused by removing everything above the candidate threshold `μ(e_s) + z·σ(e_s)`.

**Why it matters for your project:** this is the historical ancestor of "learn a decision boundary from the data itself instead of hand-tuning it" — conceptually a cousin of what your project's DQN does (learn how to weight federated clients instead of hand-tuning fixed weights, as Phase 8 did). Same philosophy, completely different mechanism and completely different problem (per-channel thresholding vs. per-round aggregation weighting).

---

## 8. Architecture

```text
Channel telemetry (t-250 ... t)  ─┐
Command one-hot vectors (t-250...t) ─┤──> LSTM (2 layers, 80 units) ──> predicted value ŷ(t+1)
                                    ┘
                                                     │
                                     actual y(t+1) ──┤── e(t) = |y - ŷ|
                                                     │
                                             EWMA smoothing ──> e_s(t)
                                                     │
                                   Nonparametric Dynamic Threshold ε
                                                     │
                                        Pruning (false-positive control)
                                                     │
                                     Anomalous sequence + severity score
```

One such pipeline is run **independently per channel** (82 separate small models, not one shared model).

---

## 9. Results

Table 2 of the paper (their proposed non-parametric method with pruning, vs. a Gaussian-tail parametric baseline used in the Numenta Anomaly Benchmark's top algorithm):

| Approach | Precision | Recall | F0.5 |
|---|---:|---:|---:|
| **Non-parametric + pruning (proposed)** | **87.5%** | **80.0%** | **0.71** |
| Non-parametric, no pruning | 48.9% | 84.8% | 0.47 |
| Gaussian-tail parametric, no pruning | 87.5% | 66.7% | 0.66 |
| Gaussian-tail parametric + pruning | 91.7% | 62.9% | 0.66 |

- Average LSTM prediction error: **5.9%** (5.5% SMAP, 6.8% MSL — SMAP's operations are more routine/predictable, so its errors are lower).
- Pruning cost only 4.8 percentage points of recall but gained 38.6 percentage points of precision — the paper's headline point: false-positive control is what makes the difference between "researchy demo" and "something engineers will trust."
- Contextual anomalies (harder, need real temporal modeling) were still detected at 69% recall overall — the paper argues this is close to the *ceiling* achievable, since these are (by definition) invisible to threshold/distance-based methods.
- **Deployed pilot:** as of the paper's writing, this system was running on real SMAP telemetry (700+ channels, near real-time), and had already found confirmed real anomalies.

---

## 10. What Did This Paper Do Well?

- Used **real, expert-labeled anomaly data**, not synthetic or injected anomalies — a genuinely harder and more credible test than most papers attempt.
- The nonparametric thresholding idea is simple, cheap, and — crucially — the paper *proves* the Gaussian assumption used by competing methods is statistically wrong for this data (D'Agostino-Pearson normality test, rejected at α=0.005) rather than just asserting it.
- Actually deployed the system as a pilot on a real mission — very few academic anomaly-detection papers get this far.
- Very transparent about tradeoffs (Precision vs Recall vs pruning aggressiveness), with an explicit knob (`p`) the user can tune.
- Released open-source code and the dataset (`github.com/khundman/telemanom`) — this is *why* Papers 1 and 2 in your set could build directly on it.

---

## 11. What Are Its Limitations?

- **No federated learning.** Each channel's model is trained centrally at JPL from data already on the ground — there is no "multiple satellites, each with private local data" setting at all. *(explicit — the paper never claims this)*
- **No reinforcement learning of any kind.**
- **One model per channel, not per spacecraft.** This doesn't scale naturally to "detect anomalies across a fleet of satellites jointly" — it scales to "watch this many separate univariate signals." *(inference: a reasonable reading of the single-channel design choice, not stated as a limitation by the authors)*
- Command information is only "one-hot encoded at the module level" — no detail about *what* the command actually did. The authors flag this themselves as future work.
- Precision in the real deployed system is *lower* than the experimental numbers above, because in production every channel must be watched all the time (including quiet, non-anomalous periods) — the authors say this explicitly in Section 5 ("Deployment").
- Detects **operational/physical anomalies**, not cybersecurity attacks — there is no notion of a malicious actor in this paper at all. *(important — this distinguishes it sharply from Paper 4 in your set.)*
- Single-seed, single-run reporting style similar in spirit to some of your own project's phases, though the paper does not use that exact language.

---

## 12. Future Work

The authors explicitly list:
1. Richer command-feature engineering (more than just one-hot module-level encoding).
2. Better training-data selection for missions without routine, repeatable operations (e.g. MSL, which behaves less predictably than SMAP).
3. Continued refinement of false-positive mitigation.
4. Modeling **cross-channel correlations** (currently every channel is treated in complete isolation from every other channel on the same spacecraft) — the authors call this out as unaddressed.

**Why this matters:** point 4 (cross-channel correlation) is conceptually a step toward exactly the kind of "one satellite = many correlated signals" reasoning your Phase 3 GRU+attention architecture explores, and point 2 (better training-data selection under irregular operations) foreshadows the kind of "which client/data is trustworthy right now" reasoning your Phase 8 adaptive-weighting and Phase 9B DQN address — just at the single-spacecraft, single-channel level rather than the multi-satellite federated level.

---

## 13. Research Gap

### Explicit gap (stated by the authors)
- No cross-channel modeling.
- False positives are still the main obstacle to full operator trust, even after pruning.

### Observed gap (not stated, reasonably inferred from the methodology)
- No multi-satellite setting exists at all in this paper — "detecting anomalies" here always means "on this one spacecraft's own data." There is no notion of comparing, combining, or learning across *different* spacecraft.
- No adaptive/learned control over *how much to trust* a given channel's model over time — the thresholding is adaptive round-to-round in a statistical sense, but there is no learning agent making that decision from experience (no RL).

---

## 14. Connection to Our Capstone

| Our component | Does this paper address it? | How? |
|---|---|---|
| Satellite network | Partially | Single-spacecraft telemetry only, not a network of satellites |
| Anomaly detection | Yes | Core contribution — forecasting-based, per-channel, physical/operational anomalies |
| Deep learning | Yes | LSTM forecasting model |
| Federated learning | No | No multi-client/multi-satellite training exists in this paper |
| Reinforcement learning | No | Not used anywhere |
| Adaptive behavior | Partially | The threshold *itself* is adaptive/data-driven per window — but nothing is "learned from a reward" |
| Privacy | No | Not discussed — telemetry is assumed centrally available |
| Non-IID clients | No | No client concept exists |
| Communication efficiency | No | Not discussed — no distributed training |
| Real-time detection | Yes | The paper explicitly targets and achieves near-real-time monitoring, and was actually deployed |

---

## 15. What We Can Learn From This Paper

- The **general recipe** — "train a model on normal behavior, measure how wrong its predictions get, decide statistically (not by hand) when 'wrong enough' means anomalous" — is the conceptual grandparent of your project's own OPSSAT-AD (Phase 9A) work, which also trains on real spacecraft telemetry and evaluates against real labels.
- The emphasis on **honest, disclosed tradeoffs** (Precision/Recall/pruning, deployed-vs-experimental precision gap) is a good model for how your own project already documents Phase 9B's DQN result honestly, including when it didn't beat the baseline.
- The "one model per channel" design choice is a useful contrast case: your project instead uses **one shared model per satellite client** in the federated setting (Phase 6–9B) — a different scaling decision, worth being able to explain *why* in viva (traceability/interpretability per-channel vs. shared representation-learning across a client).

---

## 16. What We Should NOT Copy

- Do **not** claim this paper (or your own telemetry-side work) demonstrates anything about federated learning, multi-satellite coordination, or cybersecurity — it explicitly does none of these.
- Do **not** borrow the "Gaussian threshold is fine" assumption anywhere in your own work — this paper specifically *disproves* it for real telemetry error data; if your project ever reasons about thresholds statistically, prefer this paper's nonparametric logic over a Gaussian one.
- Do **not** present "one model per channel" as automatically the *better* choice than a shared model — it's a scale/traceability tradeoff, not a universal truth.

---

## 17. Simple Example

**Illustrative example — not from the paper.**

Imagine a satellite's battery-temperature channel normally reads between 18–22°C, and its "expected" value at each moment depends on whether the satellite is currently facing the sun (a command signal). The LSTM learns: "when facing the sun, expect ~21°C; when not, expect ~19°C." One day, while *not* facing the sun, the sensor reads 21°C. In isolation that's a perfectly normal-looking number (18–22°C range) — a fixed-threshold alarm would never fire. But the LSTM predicted ~19°C, so the error is large, and it gets flagged as a *contextual* anomaly — exactly the kind of case this paper targets.

---

## 18. Explain Like I'm Preparing for Viva

**Q: What problem does this paper solve?**
A: Automatically detecting spacecraft telemetry anomalies — including ones that look normal in isolation but are wrong given the current context — without needing a human to hand-tune alarm thresholds for thousands of channels.

**Q: Why did they use an LSTM specifically?**
A: Telemetry is a time series with long-range dependencies (a command now can affect a reading much later); LSTMs handle that better than plain RNNs because of their gated memory cells.

**Q: What dataset did they use?**
A: Real NASA SMAP satellite and MSL/Curiosity rover telemetry, with anomaly labels mined from real incident reports (ISAs) — 82 channels, 105 labeled anomaly sequences.

**Q: What was their best result?**
A: Their non-parametric thresholding + pruning approach reached 87.5% precision, 80.0% recall, F0.5 = 0.71, beating a Gaussian-based parametric baseline.

**Q: What is the biggest limitation?**
A: Everything is single-spacecraft and single-channel — there is no federated learning, no reinforcement learning, and no notion of a network of multiple satellites at all.

**Q: How is this related to our project?**
A: It is the intellectual ancestor of the "telemetry forecasting-based anomaly detection" idea our project's Track B (OPSSAT-AD) validation touches, but it has none of our project's federated or DRL machinery — it establishes *why* single-spacecraft forecasting-based detection works, which our project then extends toward a federated, multi-satellite setting (a step this paper never takes).

**Q: Why does pruning matter so much in this paper?**
A: Because without it, precision collapses to under 50% — meaning more than half of all "anomalies" flagged would be false alarms, which destroys operator trust in a real deployed system. Pruning trades a small amount of recall for a huge gain in precision.
