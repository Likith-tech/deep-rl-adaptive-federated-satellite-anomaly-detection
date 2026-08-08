# Sequence-Labeling Strategy Analysis

Measured class distributions for candidate sequence-labeling strategies, computed directly from the real (order-preserving) train/validation splits and the real KDDTest+ split — no estimates. This analysis exists because an initial experiment using the "any" strategy at sequence_length=16 produced a 100%-anomaly sequence dataset (see `results/reports/temporal_results.md`, "Initial sequence-labeling experiment"), which was rejected as an invalid benchmark.

## Record-level anomaly rate (for reference)

- Train: 46.57%
- Validation: 46.39%
- Test (KDDTest+): 56.92%

## Sequence length 8

| Strategy | Split | Sequences | Normal | Anomaly | Anomaly % |
|---|---|---:|---:|---:|---:|
| any | train | 13384 | 91 | 13293 | 99.32% |
| any | validation | 2362 | 19 | 2343 | 99.20% |
| any | test | 2818 | 3 | 2815 | 99.89% |
| majority | train | 13384 | 9445 | 3939 | 29.43% |
| majority | validation | 2362 | 1686 | 676 | 28.62% |
| majority | test | 2818 | 1353 | 1465 | 51.99% |
| last | train | 13384 | 7224 | 6160 | 46.03% |
| last | validation | 2362 | 1280 | 1082 | 45.81% |
| last | test | 2818 | 1172 | 1646 | 58.41% |
| ratio@0.25 | train | 13384 | 749 | 12635 | 94.40% |
| ratio@0.25 | validation | 2362 | 124 | 2238 | 94.75% |
| ratio@0.25 | test | 2818 | 40 | 2778 | 98.58% |
| ratio@0.5 | train | 13384 | 5860 | 7524 | 56.22% |
| ratio@0.5 | validation | 2362 | 1065 | 1297 | 54.91% |
| ratio@0.5 | test | 2818 | 612 | 2206 | 78.28% |
| ratio@0.75 | train | 13384 | 11997 | 1387 | 10.36% |
| ratio@0.75 | validation | 2362 | 2105 | 257 | 10.88% |
| ratio@0.75 | test | 2818 | 2108 | 710 | 25.20% |

## Sequence length 16

| Strategy | Split | Sequences | Normal | Anomaly | Anomaly % |
|---|---|---:|---:|---:|---:|
| any | train | 6692 | 0 | 6692 | 100.00% |
| any | validation | 1181 | 0 | 1181 | 100.00% |
| any | test | 1409 | 0 | 1409 | 100.00% |
| majority | train | 6692 | 4690 | 2002 | 29.92% |
| majority | validation | 1181 | 844 | 337 | 28.54% |
| majority | test | 1409 | 521 | 888 | 63.02% |
| last | train | 6692 | 3613 | 3079 | 46.01% |
| last | validation | 1181 | 629 | 552 | 46.74% |
| last | test | 1409 | 591 | 818 | 58.06% |
| ratio@0.25 | train | 6692 | 163 | 6529 | 97.56% |
| ratio@0.25 | validation | 1181 | 28 | 1153 | 97.63% |
| ratio@0.25 | test | 1409 | 2 | 1407 | 99.86% |
| ratio@0.5 | train | 6692 | 3388 | 3304 | 49.37% |
| ratio@0.5 | validation | 1181 | 604 | 577 | 48.86% |
| ratio@0.5 | test | 1409 | 281 | 1128 | 80.06% |
| ratio@0.75 | train | 6692 | 6550 | 142 | 2.12% |
| ratio@0.75 | validation | 1181 | 1152 | 29 | 2.46% |
| ratio@0.75 | test | 1409 | 1259 | 150 | 10.65% |

## Sequence length 32

| Strategy | Split | Sequences | Normal | Anomaly | Anomaly % |
|---|---|---:|---:|---:|---:|
| any | train | 3346 | 0 | 3346 | 100.00% |
| any | validation | 590 | 0 | 590 | 100.00% |
| any | test | 704 | 0 | 704 | 100.00% |
| majority | train | 3346 | 2357 | 989 | 29.56% |
| majority | validation | 590 | 428 | 162 | 27.46% |
| majority | test | 704 | 198 | 506 | 71.88% |
| last | train | 3346 | 1817 | 1529 | 45.70% |
| last | validation | 590 | 311 | 279 | 47.29% |
| last | test | 704 | 300 | 404 | 57.39% |
| ratio@0.25 | train | 3346 | 17 | 3329 | 99.49% |
| ratio@0.25 | validation | 590 | 4 | 586 | 99.32% |
| ratio@0.25 | test | 704 | 0 | 704 | 100.00% |
| ratio@0.5 | train | 3346 | 1932 | 1414 | 42.26% |
| ratio@0.5 | validation | 590 | 340 | 250 | 42.37% |
| ratio@0.5 | test | 704 | 118 | 586 | 83.24% |
| ratio@0.75 | train | 3346 | 3345 | 1 | 0.03% |
| ratio@0.75 | validation | 590 | 590 | 0 | 0.00% |
| ratio@0.75 | test | 704 | 687 | 17 | 2.41% |

## Recommendation

A defensible strategy for this dataset must avoid a near-single-class sequence set (the flaw in the rejected "any" experiment) on ALL three splits, at the sequence length actually used, while staying simple to explain and semantically sensible for anomaly detection.

Strategy/length combinations where BOTH classes have at least 15% representation on every split (train, validation, and test) — i.e. not degenerate:

| Sequence length | Strategy | Train anomaly % | Validation anomaly % | Test anomaly % |
|---|---|---:|---:|---:|
| 8 | majority | 29.43% | 28.62% | 51.99% |
| 8 | last | 46.03% | 45.81% | 58.41% |
| 8 | ratio@0.5 | 56.22% | 54.91% | 78.28% |
| 16 | majority | 29.92% | 28.54% | 63.02% |
| 16 | last | 46.01% | 46.74% | 58.06% |
| 16 | ratio@0.5 | 49.37% | 48.86% | 80.06% |
| 32 | majority | 29.56% | 27.46% | 71.88% |
| 32 | last | 45.70% | 47.29% | 57.39% |
| 32 | ratio@0.5 | 42.26% | 42.37% | 83.24% |
