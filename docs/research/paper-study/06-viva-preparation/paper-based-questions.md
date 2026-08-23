# Viva Prep — Sharp, Paper-Specific Questions an Examiner Might Ask

## About Papers 1–3 (telemetry lineage)

- **"These three papers never use federated learning or RL — why did you include them at all?"** — They establish the technical and methodological foundation for telemetry-based anomaly detection (forecasting + residual thresholding), which is directly relevant to Track B (OPSSAT-AD) of our project, even though Track B currently doesn't combine with FL/DRL.

- **"Paper 3's dataset (SMAP/MSL) is also used by Papers 1 and 2 — does that mean their results are directly comparable?"** — Not precisely comparable: each paper reports different metric combinations, uses slightly different train/test protocols, and none report statistical significance testing — the fair claim is "directionally improving across the lineage," not "paper X beats paper Y by N points, controlled for everything."

- **"Paper 1 injects a synthetic anomaly into the Beidou dataset — is that a fair test?"** — It tests whether the method can detect an anomaly shape of that kind, but not whether it would catch whatever real anomalies actually occurred on that clock — an important caveat, and the paper is explicit that it's an injection, not a naturally occurring incident.

- **"Paper 2's uncertainty region — could you add this to your own project's models?"** — Yes, conceptually: Monte Carlo Dropout is a low-effort technique (keep dropout on at test time, run multiple forward passes) that could be added to BaselineMLP or TemporalGRUAttention without redesigning the architecture — named explicitly as a good candidate future-work direction in `01-paper-notes/paper-03.md`.

## About Paper 4 (FedSTAR) — expect the most scrutiny here

- **"FedSTAR already combines FL and DRL for satellite anomaly detection, published the same year as your project. Isn't your project just redoing this?"** — No: FedSTAR targets network-traffic cyberattack detection using SAC (clustering + continuous hyperparameters) on non-satellite-specific-in-origin benchmark data; our project uses a smaller, fully-discrete DQN (no clustering) with a controlled three-stage comparison against a fixed and a rule-based baseline on the same pipeline, plus a separate real-spacecraft-telemetry validation track FedSTAR has no equivalent of. See `04-research-gap/our-project-positioning.md` for the full, careful answer — have this one memorized, it is the single most likely hard question you'll face.

- **"FedSTAR uses real satellite-relevant attack categories (DTN-protocol DDoS in STIN). Does your NSL-KDD data have anything satellite-specific?"** — No — NSL-KDD is generic terrestrial network-intrusion data; our project's "satellite" framing comes entirely from how the data is *partitioned* (Dirichlet, simulating 8 separate satellite clients), not from anything satellite-specific in the traffic content itself. This is an honest limitation, not something to obscure.

- **"FedSTAR reports statistically significant improvements on some datasets, not others (Table 4) — does your project test for significance?"** — No, our project reports single-seed results throughout (Phases 6–9B), consistent with a disclosed capstone-scale limitation — this is a genuine gap relative to FedSTAR's more rigorous (5-seed) methodology, worth acknowledging directly rather than deflecting.

- **"FedSTAR explicitly says Byzantine robustness is future work. Does your project do any better here?"** — No, this remains equally open in our project — a shared, honestly-acknowledged limitation, not a point of differentiation.

- **"Your DQN's action space (9 fixed profiles) seems much simpler than FedSTAR's SAC action space (clustering algorithm + 3 continuous values). Isn't that a weaker system?"** — It's a deliberately narrower, more explainable design choice appropriate to a capstone-scale, single-DQN-algorithm experiment — not an attempt to match FedSTAR's scale or performance. The trade-off (simplicity/explainability vs. expressiveness) is named explicitly, not hidden.

- **"FedSTAR's variance-decomposition proof (Table 2) mathematically justifies why clustering should help. Does your project have an equivalent mathematical justification for its own weighting approach?"** — Not currently — our project's justification for adaptive weighting is empirical (Phase 8's ablation results showing no single fixed rule wins on every metric), not a formal proof. This is a legitimate, honest gap; FedSTAR's variance-decomposition technique (Section 3.3.3 of Paper 4) is a good template if a future iteration of the project wanted to add a comparable mathematical argument.
