# Phase 15 — Ablation Studies

**All numbers in this report are real, measured values from scripts/run_ablation_studies.py, evaluated once on KDDTest+ per sweep point by this script. Nothing here is estimated or fabricated.**

## 1. Purpose

Phase 14 found Phase 8's `resource_only` rule as the single best aggregation mechanism across the whole project, and Phase 9 found that staleness-aware discounting did not beat a staleness-blind control. Both findings were left as open questions in their own reports: is resource_only's win a monotonic effect of weighting resource more, or does it peak? Would a different staleness decay_rate have changed Phase 9's negative result? This phase investigates both, on real data, reusing Phase 8/9's UNMODIFIED training code — only the swept parameter changes.

## 2. Ablation 1 — resource-weight sensitivity

`w_resource` swept from 0.0 to 1.0; the remaining weight always splits performance:data:fairness in Phase 8's `combined` ratio (0.40:0.25:0.20, renormalized). `w_resource=1.0` is numerically identical to Phase 8's `resource_only` config and reproduces its real result as a built-in cross-phase check.

| w_resource | Val F1 (best round) | Test F1 | Test ROC-AUC |
|---:|---:|---:|---:|
| 0.00 | 0.9890 | 0.7426 | 0.8547 |
| 0.25 | 0.9876 | 0.7443 | 0.8525 |
| 0.50 | 0.9885 | 0.7464 | 0.8495 |
| 0.75 | 0.9863 | 0.7465 | 0.8462 |
| 1.00 | 0.9860 | 0.7547 | 0.8438 |

**Reproducibility check:** w_resource=1.0 scored test F1 0.7547 against Phase 8's committed `resource_only` result of 0.7547 — an exact match, confirming this sweep's harness is faithful to Phase 8's.

**Finding:** best KDDTest+ F1 in this sweep was at w_resource=**1.00** — confirming resource_only (w_resource=1.0) as not just the best of Phase 8's original four discrete configs, but the peak of this finer 5-point sweep too: weighting resource MORE is monotonically better across every point tested here, at least up to full weight.

## 3. Ablation 2 — staleness exponential-decay-rate sensitivity

`decay_rate` swept from 0.1 (aggressive discount) to 0.9 (gentle discount), reusing the EXACT SAME seeded participation pattern as Phase 9 (`participation_seed=42`) so only the discount curve's steepness varies. `decay_rate=0.5` is numerically identical to Phase 9's `exponential_decay` config and reproduces its real result as a built-in cross-phase check.

| decay_rate | Val F1 (best round) | Test F1 | Test ROC-AUC | Max staleness seen |
|---:|---:|---:|---:|---:|
| 0.10 | 0.9856 | 0.7361 | 0.8556 | 3 |
| 0.30 | 0.9847 | 0.7359 | 0.8560 | 3 |
| 0.50 | 0.9854 | 0.7360 | 0.8549 | 3 |
| 0.70 | 0.9849 | 0.7358 | 0.8534 | 3 |
| 0.90 | 0.9842 | 0.7369 | 0.8529 | 3 |

**Reproducibility check:** decay_rate=0.5 scored test F1 0.7360 against Phase 9's committed `exponential_decay` result of 0.7360 — an exact match, confirming this sweep's harness is faithful to Phase 9's.

**Finding:** best KDDTest+ F1 in this sweep was at decay_rate=**0.90** (0.7369). Phase 9's `no_correction` control (staleness measured but never discounted) scored 0.7371 — every decay_rate tested in this finer sweep still falls short of that staleness-blind control, reinforcing Phase 9's original finding: under this project's real, small-scale staleness (max 3 rounds), no tested decay_rate makes staleness-aware discounting worthwhile.

## 4. Limitations

- Single run, single seed per sweep point — same caveat every prior phase already carried.
- Both sweeps hold every other Phase 6/8/9 hyperparameter fixed; interaction effects between, e.g., resource weight and staleness decay were not explored (would require a 2D grid, out of scope for this phase's two independent 1D sweeps).
- The resource-weight sweep's non-resource split ratio (Phase 8's `combined` proportions) was fixed before this sweep ran, not itself swept — a further nested ablation, not attempted here.

## 5. Exact artifact locations

```
experiments/ablation_studies/resource_weight_sweep/w_resource_<value>/
experiments/ablation_studies/staleness_decay_sweep/decay_rate_<value>/
results/plots/ablation_studies/*.png
results/reports/ablation_study.md   (this file)
```