# Paper 2 — An Anomaly Detection Method for Satellites Using Monte Carlo Dropout

**Authors:** Mohammad Amin Maleki Sadr, Yeying Zhu, Peng Hu (University of Waterloo / National Research Council Canada)
**Year:** 2022 (final version published 22 Sep 2022)
**Venue:** *IEEE Transactions on Aerospace and Electronic Systems* (TAES) — a well-established IEEE journal
**Source file:** `TAES_LSTM_Uncertainty_Final_Files__Copy___Arxiv_.pdf`

---

## 1. One-Minute Explanation

Papers 1 and 3 both train an LSTM to predict telemetry and flag big prediction errors — but neither tells you *how confident* the model is in its own prediction. This paper's core idea is: what if the model could say "I predict 21.3°C, and I'm fairly sure it's between 20.9°C and 21.7°C" instead of just "I predict 21.3°C"? That confidence interval (called an **uncertainty region**) is genuinely useful for anomaly detection — a true anomaly should fall *outside* the model's own confidence region, not just far from a single point prediction. Computing this properly with a true Bayesian neural network is normally too computationally expensive, so the authors use a clever, cheap trick called **Monte Carlo (MC) Dropout**: keep the normal "dropout" regularization technique switched *on* even during testing (normally it's switched off), run the same input through the network many times, and treat the *spread* of those different outputs as the uncertainty estimate. They apply this inside both the LSTM layers and the dense layers (more thoroughly than a prior method that only applied it to the dense layers), add a smarter pre-processing filter, and add a post-processing rule that only flags something as anomalous if enough *consecutive* points fall outside the confidence region (to fight false positives).

---

## 2. The Problem

- Forecasting-based and reconstruction-based anomaly detection methods (like Papers 1 and 3) give you a single predicted number, with no sense of how trustworthy that number is.
- A true Bayesian Neural Network (BNN) — which would give you a genuine probability distribution over predictions — requires computing an intractable integral over all possible network weights; it's not practically usable at scale.
- A prior competing method (Chen et al., "MCD-BiLSTM-VAE" [1]) already tried MC Dropout for this, but only applied dropout **before the dense layers**, not inside the LSTM cells themselves — so it only captures part of the model's uncertainty, and it used a static (fixed) threshold rather than a dynamic one.

**Simple example:** if a weather forecaster says "70% chance of rain" instead of just "it will rain," you can decide how much to trust that call. This paper is the anomaly-detection equivalent — giving each telemetry prediction a confidence band, not just a single guess.

---

## 3. What Is the Main Idea?

```text
Raw telemetry
     ↓
Adaptive pre-processing (variable-width smoothing window — narrower when values
are changing fast, so a real anomaly point never gets smoothed away)
     ↓
Bayesian LSTM (LSTM layers + dense layers, MC Dropout applied INSIDE BOTH)
     ↓
Run the SAME input through the network many times (each time with a different
random dropout pattern) → many slightly different predictions
     ↓
Mean of those predictions = the forecast
Spread (variance) of those predictions = the uncertainty region (confidence interval)
     ↓
If the REAL value falls outside the confidence interval → tentative anomaly
     ↓
Post-processing filter: only confirm as anomaly if enough CONSECUTIVE points
are outside the interval (reduces false positives)
```

---

## 4. Dataset

- The **same NASA SMAP/MSL benchmark** originally built by Hundman et al. (Paper 3) — 82 signals total, 54 of them continuous (the rest discrete and excluded from this univariate analysis), some further excluded because they are constant/degenerate (e.g., channel "M6" is always −1).
- SMAP: 55 signals, 67 anomaly sequences, 562,800 data points, 54,696 anomaly points.
- MSL: 27 signals, 36 anomaly sequences, 132,046 data points, 7,766 anomaly points.
- **What one "sample" represents:** exactly as in Papers 1 and 3 — one telemetry channel's time series, with command features present in the raw data but explicitly found by this paper's own analysis to have "negligible correlation" with the telemetry signal, so they are **not used** as input here (a difference from Papers 1 and 3, which do use command information).

---

## 5. Features / Input Data

| Feature | What it represents | Why it matters |
|---|---|---|
| Smoothed telemetry value (own adaptive pre-processing) | The channel's cleaned recent history | Standard weighted-moving-average smoothing can accidentally blur out a real anomaly; this paper's adaptive window width avoids that |
| (Command features considered but excluded) | Spacecraft command metadata | Found to correlate negligibly with the telemetry signal in this dataset, so dropped — a deliberate, evidence-based choice, unlike Papers 1/3 which keep command features |

---

## 6. Methodology

```text
Step 1 → Adaptive pre-processing: for each point, measure how different it
          is from its neighbor; if the difference is small, widen the
          smoothing window; if large (possible anomaly), narrow it to 1
          point (don't smooth it away)
Step 2 → Train a Bayesian LSTM (3 LSTM layers + 2 dense layers) with MC
          Dropout (rate 0.2) applied at BOTH the LSTM and dense layers,
          during BOTH training and testing
Step 3 → At test time, run each input through the network many times
          (different random dropout masks each time) → a distribution
          of predictions, not just one number
Step 4 → Compute mean (forecast) and variance (uncertainty) of that
          distribution → build a confidence interval (e.g. 99%)
Step 5 → Flag points outside the confidence interval as tentative anomalies
Step 6 → Post-processing: only confirm a sequence as truly anomalous if
          at least 0.8 * N_max of N_max CONSECUTIVE points are outside
          the interval — this N_max is tuned per-signal via grid search
Step 7 → Evaluate: MSE, F1, Accuracy, Precision, Recall, and compare
          computational cost (parameters, training time, throughput)
```

---

## 7. Model / Algorithm

### Monte Carlo (MC) Dropout

**Technical term → simple meaning → example**
> **Dropout:** During training, a neural network randomly "turns off" some of its neurons on each pass, forcing the rest to not over-rely on any single one — this normally prevents overfitting and is turned *off* at test time for a normal prediction.
> **MC Dropout:** Keep dropout *on* even at test time, and run the same input through the network many times. Because a different random set of neurons is switched off each time, you get a slightly different answer each time — and the *spread* of those answers approximates how uncertain a full Bayesian network would have been, without the enormous cost of actually being Bayesian.
> *Analogy:* Imagine asking 50 slightly different versions of a doctor (each missing a slightly different bit of training/knowledge) to estimate your recovery time. If all 50 give almost the same number, you can be confident. If their answers are all over the place, that spread itself tells you something — the estimate is uncertain.

**Why the authors used it:** it is a cheap, practical way to approximate the uncertainty a full Bayesian Neural Network would give, without the intractable computation.

**What goes in / what comes out:** a windowed sequence of (smoothed) telemetry goes in; **many** predicted values come out (one per stochastic forward pass), which are then summarized into a mean prediction and a variance (uncertainty).

**Where the authors applied it that a prior method (MCD-BiLSTM-VAE) didn't:** dropout is applied **inside the LSTM recurrent connections themselves**, not just before the final dense layers — the paper's stated key technical difference and improvement over prior work.

### Bayesian LSTM

The architecture: 3 LSTM layers → 2 dense layers, dropout rate 0.2 throughout, implemented in PyTorch.

---

## 8. Architecture

```text
Telemetry Storage → Pre-processing (adaptive window)
        ↓
    Input Layer
        ↓
LSTM 1 (MC Dropout) → LSTM 2 (MC Dropout) → LSTM 3 (MC Dropout)
        ↓
    Output Dense Layer (MC Dropout)
        ↓
Post-processing (confidence region + consecutive-point rule)
        ↓
   Anomaly flag + uncertainty bounds
```

---

## 9. Results

**F1 score comparison (Table III):**

| Baseline | SMAP | MSL |
|---|---:|---:|
| **Bayesian LSTM (proposed)** | **0.84** | **0.74** |
| TadGAN | 0.66 | 0.55 |
| LSTM AE (Hundman et al.'s method) | 0.69 | 0.55 |
| MadGAN | 0.12 | 0.11 |
| LSTM (plain) | 0.62 | 0.48 |
| MCD-BiLSTM-VAE (Chen et al., prior MC-dropout method) | 0.67 | 0.64 |

**Complexity comparison (Table IV):** the proposed model has 229,409 parameters, trains in 1.48 sec/epoch, and achieves 0.57×10³ predictions/sec — dramatically cheaper than the closest-performing prior MC-dropout competitor MCD-BiLSTM-VAE (580,993 parameters, 6.96 sec/epoch, only 0.032×10³ predictions/sec) and much cheaper than TadGAN (447,189 parameters, 78 sec/epoch).

**Accuracy/Precision/Recall (Table V):** SMAP — Accuracy 0.75, Precision 0.82, Recall 0.87; MSL — Accuracy 0.70, Precision 0.74, Recall 0.87 — best or near-best across all compared methods.

**In simple terms:** this method beats every other method it compares against on F1 score, while being one of the *cheapest* to train and run among the deep-learning competitors (only the much simpler ARIMA statistical method is cheaper) — and, as a bonus, it also gives you an uncertainty estimate the others don't provide at all.

---

## 10. What Did This Paper Do Well?

- A genuinely useful, well-motivated technical contribution: applying MC Dropout to *both* the LSTM and dense layers, not just the dense layers as prior work did, with a clear mathematical justification (Section IV).
- Directly, fairly compares against 6 other real published methods (including the closest prior competitor) on the same dataset, reporting F1 *and* computational cost — not just accuracy.
- Adds a genuinely new capability (uncertainty/confidence region) that the forecasting-only methods in Papers 1 and 3 simply do not have.
- Practical post-processing rule (consecutive-point confirmation) directly targeting false positives, similar in spirit to Paper 3's pruning step.
- Released source code (`https://bit.ly/3zf1Ipb`).

---

## 11. What Are Its Limitations?

- **No federated learning, no reinforcement learning, no multi-satellite setting** — same category of limitation as Papers 1 and 3.
- Command features were found irrelevant *for this specific dataset* and dropped — a reasonable, evidence-based choice here, but it means the model uses less contextual information than Papers 1 and 3, which could matter on a different spacecraft/dataset. *(inference)*
- `N_max` (the consecutive-point post-processing parameter) is tuned per-signal via grid search that is explicitly guided by minimizing "dissimilarity between predicted and known labels" — meaning it is effectively **tuned using the test labels**, a mild form of information leakage from labels into a "supposedly unsupervised" pipeline. *(inference: the paper states the tuning procedure explicitly but does not discuss this as a leakage concern)*
- Some signals (about 19 named channels) were excluded entirely because they are degenerate/constant — a legitimate practical choice, but it slightly narrows what "the dataset" means in this paper's results versus the full original benchmark. *(explicit, listed by name in the paper)*
- Not clearly specified in the paper: any statistical significance testing across multiple runs/seeds — results appear to be single-run numbers, similar to Papers 1 and 3.

---

## 12. Future Work

**Not clearly specified in the paper** — the Conclusion section describes what the method achieves and its intended generic applicability to FDIR (fault detection, identification, recovery) tasks and other applications needing on-board hardware/software anomaly detection, but does not list explicit numbered future-work items the way Papers 1 and 3 do.

**Why this (still) matters for your project:** the paper's framing — "generic time series analytics tool for autonomous FDIR tasks of satellite missions" and "efficient utilization... for on-board hardware and software AD" — is again the same "needs to run on the spacecraft itself, can't ship all raw data" motivation that connects, at a conceptual level, to why federated learning is relevant for satellites, even though this paper doesn't implement anything federated.

---

## 13. Research Gap

### Explicit gap
Not clearly specified as a numbered list in the paper (no formal "Future Work" section) — but the closing paragraph frames the work as a step toward on-board, real-time FDIR, implying deployment/efficiency work remains.

### Observed gap
- No multi-satellite/federated structure — exactly the same gap as Papers 1 and 3.
- No reinforcement learning — the uncertainty region and post-processing thresholds are fixed/tuned procedures, not something an agent learns from a reward signal.
- The `N_max` tuning procedure's reliance on label-based grid search (see Section 11 above) is a soft form of the same broader problem your own project is careful to avoid: letting test-set information leak into a decision that's supposed to be made without it. Your project's own documentation (e.g. the DRL design doc) treats exactly this kind of leakage risk as something to test for explicitly (AST-based privacy/leakage tests) — a good contrast point.

---

## 14. Connection to Our Capstone

| Our component | Does this paper address it? | How? |
|---|---|---|
| Satellite network | Partially | Single-spacecraft telemetry (SMAP/MSL), no network/fleet |
| Anomaly detection | Yes | Core contribution — Bayesian forecasting-based detection with uncertainty |
| Deep learning | Yes | Bayesian LSTM via MC Dropout |
| Federated learning | No | Not present |
| Reinforcement learning | No | Not present |
| Adaptive behavior | Partially | Post-processing threshold parameter tuned per-signal, not learned online |
| Privacy | No | Not discussed |
| Non-IID clients | No | No client concept |
| Communication efficiency | Partially | The paper explicitly emphasizes low computational cost as a deployment advantage, relevant to bandwidth/compute-constrained satellites, but this is not framed as "communication efficiency" in a distributed-training sense |
| Real-time detection | Yes | Explicitly designed as a low-latency, low-complexity method suitable for real-time/on-board use |

---

## 15. What We Can Learn From This Paper

- **Uncertainty estimation is a genuinely different, complementary idea to what your project currently does** — your project's detectors currently produce point predictions/classifications (Phase 2/3/6-9B), without a confidence interval. If your project's discussion section wants a concrete, well-evidenced "future work" direction beyond DRL, MC Dropout is a cheap, proven way to add "how confident is this detection?" to any of your existing models with minimal architecture change.
- The "compare F1 **and** computational cost together" evaluation style (Table III + Table IV together) is a good model for how to fairly report your own project's DQN vs. rule-based comparison, which already reports timing (Phase 9B: 101.5 min pretraining, 97.4s final run) alongside F1/fairness.
- Good example of narrowing a dataset carefully and disclosing exactly which signals were excluded and why — matches your project's own norm of disclosing exactly what was/wasn't included (e.g., your project's honest documentation of which NSL-KDD splits are used for what).

---

## 16. What We Should NOT Copy

- Don't present the `N_max` post-processing tuning as fully label-free/unsupervised without qualification — it is tuned by minimizing dissimilarity to known labels, which is closer to a semi-supervised calibration step than a purely unsupervised one.
- Don't assume dropping command features is always correct — it was correct *for this dataset*, based on measured negligible correlation, not a general rule about telemetry data.

---

## 17. Simple Example

**Illustrative example — not from the paper.**

Imagine a temperature sensor. A plain LSTM might predict "21.3°C" for the next reading. This paper's Bayesian LSTM instead runs the same input through the network 50 times (each time randomly dropping different internal connections) and gets 50 slightly different predictions clustered around 21.3°C, say ranging from 20.9°C to 21.7°C. That range becomes the 99% confidence interval. If the real next reading comes in at 21.4°C, it's inside the interval — normal. If it comes in at 24°C, it's way outside — flagged as a tentative anomaly, and if several readings in a row stay outside, it's confirmed.

---

## 18. Explain Like I'm Preparing for Viva

**Q: What problem does this paper solve?**
A: It adds a genuine, cheaply-computed confidence interval to telemetry forecasting-based anomaly detection, so the model can say not just "I predict X" but "I predict X, and here's how sure I am" — and use that uncertainty region itself to detect anomalies.

**Q: Why did they use Monte Carlo Dropout instead of a real Bayesian Neural Network?**
A: A true Bayesian Neural Network requires an intractable computation over all possible network weights. MC Dropout is a cheap approximation: keep dropout switched on at test time, run the same input through the network many times, and use the spread of the resulting predictions as the uncertainty estimate.

**Q: What dataset did they use?**
A: The same NASA SMAP/MSL benchmark that Papers 1 and 3 use.

**Q: What was their best result?**
A: F1 = 0.84 (SMAP) and 0.74 (MSL), beating TadGAN, LSTM-AE, MAD-GAN, plain LSTM, and a prior MC-Dropout method (MCD-BiLSTM-VAE), while using far fewer parameters and much less training time than the closest competitors.

**Q: What is the biggest limitation?**
A: No federated learning, no reinforcement learning, no multi-satellite setting; the post-processing threshold parameter is tuned using a procedure that references the true labels, which is closer to semi-supervised calibration than fully unsupervised detection.

**Q: How is this related to our project?**
A: It belongs to the same single-spacecraft telemetry forecasting lineage as Papers 1 and 3, contributing nothing to the federated or DRL side of your project — but its uncertainty-estimation idea (MC Dropout) is a genuinely useful, low-cost technique your project could cite as a concrete future-work direction for adding confidence estimates to its own detectors.
