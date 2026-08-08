# NSL-KDD Data Quality Report

Generated from the actual raw files present in `data/raw/` at run time.

## KDDTrain+

- Total rows: 125973
- Total features: 41 (38 numerical + 3 categorical, excluding label/difficulty)
- Missing values: 0
- Duplicate rows: 0
- Infinite values: 0
- Normal: 67343 (53.46%)
- Anomaly: 58630 (46.54%)
- Number of distinct attack labels: 23

### Class distribution (KDDTrain+)

| Label | Count | % |
|---|---|---|
| normal | 67343 | 53.46% |
| neptune | 41214 | 32.72% |
| satan | 3633 | 2.88% |
| ipsweep | 3599 | 2.86% |
| portsweep | 2931 | 2.33% |
| smurf | 2646 | 2.10% |
| nmap | 1493 | 1.19% |
| back | 956 | 0.76% |
| teardrop | 892 | 0.71% |
| warezclient | 890 | 0.71% |
| pod | 201 | 0.16% |
| guess_passwd | 53 | 0.04% |
| buffer_overflow | 30 | 0.02% |
| warezmaster | 20 | 0.02% |
| land | 18 | 0.01% |
| imap | 11 | 0.01% |
| rootkit | 10 | 0.01% |
| loadmodule | 9 | 0.01% |
| ftp_write | 8 | 0.01% |
| multihop | 7 | 0.01% |
| phf | 4 | 0.00% |
| perl | 3 | 0.00% |
| spy | 2 | 0.00% |

## KDDTest+

- Total rows: 22544
- Missing values: 0
- Duplicate rows: 0
- Infinite values: 0
- Normal: 9711 (43.08%)
- Anomaly: 12833 (56.92%)
- Number of distinct attack labels: 38

### Class distribution (KDDTest+)

| Label | Count | % |
|---|---|---|
| normal | 9711 | 43.08% |
| neptune | 4657 | 20.66% |
| guess_passwd | 1231 | 5.46% |
| mscan | 996 | 4.42% |
| warezmaster | 944 | 4.19% |
| apache2 | 737 | 3.27% |
| satan | 735 | 3.26% |
| processtable | 685 | 3.04% |
| smurf | 665 | 2.95% |
| back | 359 | 1.59% |
| snmpguess | 331 | 1.47% |
| saint | 319 | 1.42% |
| mailbomb | 293 | 1.30% |
| snmpgetattack | 178 | 0.79% |
| portsweep | 157 | 0.70% |
| ipsweep | 141 | 0.63% |
| httptunnel | 133 | 0.59% |
| nmap | 73 | 0.32% |
| pod | 41 | 0.18% |
| buffer_overflow | 20 | 0.09% |
| multihop | 18 | 0.08% |
| named | 17 | 0.08% |
| ps | 15 | 0.07% |
| sendmail | 14 | 0.06% |
| xterm | 13 | 0.06% |
| rootkit | 13 | 0.06% |
| teardrop | 12 | 0.05% |
| xlock | 9 | 0.04% |
| land | 7 | 0.03% |
| xsnoop | 4 | 0.02% |
| ftp_write | 3 | 0.01% |
| loadmodule | 2 | 0.01% |
| worm | 2 | 0.01% |
| perl | 2 | 0.01% |
| sqlattack | 2 | 0.01% |
| udpstorm | 2 | 0.01% |
| phf | 2 | 0.01% |
| imap | 1 | 0.00% |

## Class imbalance

KDDTrain+ normal:anomaly ratio is approximately 1.15:1.

## Note on KDDTest+ label distribution

KDDTest+ deliberately includes attack types absent from KDDTrain+ (this is intentional in the original NSL-KDD design, to test generalization to unseen attacks) — its class distribution should not be expected to match KDDTrain+.

## Generated plots

- `results/plots/dataset/class_distribution_train.png`
- `results/plots/dataset/class_distribution_test.png`
- `results/plots/dataset/attack_category_distribution_train.png`
- `results/plots/dataset/feature_distributions.png`
- `results/plots/dataset/feature_correlation.png`
