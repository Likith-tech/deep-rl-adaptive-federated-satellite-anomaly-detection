# Viva Prep — Likely General Questions (Beyond the Papers)

These are questions likely to come up regardless of which papers you've read, focused on your project's own choices. Short, direct answers — expand using the relevant concept note or paper note if pressed further.

1. **"Why NSL-KDD and not real satellite data for Track A?"** — NSL-KDD is a well-established, thoroughly-studied real (terrestrial) intrusion dataset, used here to validate the federated + DRL *mechanism* on a trusted base while a separate track (OPSSAT-AD) validates against real spacecraft data. No public, real, multi-satellite federated dataset currently exists (a field-wide gap, not unique to this project — see `04-research-gap/identified-gaps.md`, Gap 3).

2. **"Isn't NSL-KDD an old dataset?"** — Yes, and that's disclosed; it was chosen for being well-understood and easy to validate a novel FL/DRL pipeline against, not for state-of-the-art intrusion-detection accuracy.

3. **"Why 8 clients, not 20 like FedSTAR?"** — Not documented as an explicit trade-off decision in the material reviewed for this analysis; a reasonable, honest answer is that 8 was chosen as a manageable, capstone-appropriate scale for a fully-tested, honestly-reported pipeline, and Paper 4's own experiments (also simulated, also not real satellites) show 20 is not inherently necessary for meaningful Non-IID experimentation.

4. **"Why didn't you just use SAC like the most relevant paper (FedSTAR) does?"** — See `05-project-connection/why-our-approach.md`: SAC's extra expressive power (continuous actions) matches FedSTAR's richer action space (clustering algorithm + continuous knobs); our action space was deliberately kept smaller and fully discrete (9 named profiles, no clustering), for which DQN is the simpler, sufficient, more explainable choice.

5. **"What would you do differently if you had more time/compute?"** — Run the originally-designed diverse-partition DQN pretraining (varying the client partition/seed across episodes, not the single fixed partition the completed run used) — the project's own results report names this as the most likely fix for the DQN's current underperformance.

6. **"How do you know your privacy claims are actually true, not just claimed?"** — An automated test (`tests/federated/test_server.py`, extended to `tests/drl/test_privacy_boundary.py`) parses the actual source code of the server and DRL modules and fails the build if pandas is imported or a parquet/csv file is read — a mechanically enforced boundary, not just a design statement.

7. **"Is your result statistically significant?"** — Not tested for statistical significance; every phase (6 through 9B) reports single-seed, single-run results, consistent with a disclosed, capstone-scale limitation shared with Papers 1–3 (which also report single-run results) — Paper 4 is the only paper in the set that used multiple seeds (5) for its main comparison table.

8. **"What's the difference between your Phase 8 and Phase 9B?"** — Phase 8 uses a hand-written, fixed formula to combine 4 client signals into aggregation weights. Phase 9B uses a DQN that learns, from experience, which of 9 pre-defined weighting profiles (including Phase 8's own 4 profiles) to apply each round — the mechanism for producing weights from a chosen profile is unchanged; what's learned is *which profile to pick, when*.

9. **"Could this scale to real satellite constellations?"** — Not demonstrated; genuine barriers (real orbital communication constraints — delay, intermittent contact — are not modeled by this project or by Paper 4) would need to be addressed first, as both this project's own limitations list and Paper 4's own Conclusion acknowledge.

10. **"What's the most interesting/valuable finding from your own project, independent of the papers?"** — Phase 8's own ablation result: no single fixed weighting rule wins on both accuracy and fairness simultaneously (`resource_only` wins F1, `performance_only` wins fairness) — a genuinely useful, measured finding that directly motivated attempting Phase 9B, and that stands on its own regardless of whether Phase 9B's DQN ultimately beats it.
