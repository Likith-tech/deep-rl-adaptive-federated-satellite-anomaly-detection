# OPSSAT-AD — Dataset Inspection Report

**Source:** `data/raw/opssat_ad/dataset.csv`, downloaded via `scripts/download_opssat_ad.py` from the official Zenodo record (https://zenodo.org/records/12588358, CC-BY-4.0 per Zenodo's own structured metadata).

**License note (recorded, not silently resolved):** the bundled `LICENSE` file's own first line reads: *"MIT License for OPS-SAT Dataset"* — i.e. the file itself says MIT, while Zenodo's record metadata says CC-BY-4.0. Both are recorded here rather than picking one silently; either way, the dataset is licensed for open reuse with attribution (MIT and CC-BY-4.0 are both permissive).

## Shape and structure

- Rows (segments): 2123
- Columns: 23 — ['segment', 'anomaly', 'train', 'channel', 'sampling', 'duration', 'len', 'mean', 'var', 'std', 'kurtosis', 'skew', 'n_peaks', 'smooth10_n_peaks', 'smooth20_n_peaks', 'diff_peaks', 'diff2_peaks', 'diff_var', 'diff2_var', 'gaps_squared', 'len_weighted', 'var_div_duration', 'var_div_len']
- ID/split columns (not features): ['segment', 'train']
- Label column: `anomaly`
- Categorical column: `channel` (9 unique values)
- Numeric feature columns used for modeling (19): ['sampling', 'duration', 'len', 'mean', 'var', 'std', 'kurtosis', 'skew', 'n_peaks', 'smooth10_n_peaks', 'smooth20_n_peaks', 'diff_peaks', 'diff2_peaks', 'diff_var', 'diff2_var', 'gaps_squared', 'len_weighted', 'var_div_duration', 'var_div_len']

## Train/test split (author-provided, via the `train` column)

**Correction versus the Phase 9A pre-integration audit:** the audit (transcribed from the arXiv paper text) reported 1,494 training segments. Direct inspection of the actual downloaded file shows **1594 training segments** — the audit's train figure was a transcription error (1,273 + 321 was already recorded correctly and sums to 1,594, not 1,494). The test count (529) matches the audit exactly. Corrected here from the real data, which takes precedence.

| | Nominal (0) | Anomalous (1) | Total |
|---|---:|---:|---:|
| train=1 (training) | 1273 | 321 | 1594 |
| train=0 (test) | 416 | 113 | 529 |
| **Total** | 1689 | 434 | 2123 |

## Channel distribution

**Correction versus the pre-integration audit:** the audit (extracted from the paper's text) reported human-readable channel names (`I_B_FB_MM_0/1/2`, `I_PD1_THETA`..`I_PD6_THETA`). The actual downloaded `dataset.csv` uses opaque internal telemetry codes instead — recorded here exactly as found, not reconciled by guessing which name maps to which code:

| Channel code | Segment count |
|---|---:|
| CADC0873 | 593 |
| CADC0872 | 546 |
| CADC0888 | 252 |
| CADC0892 | 211 |
| CADC0874 | 194 |
| CADC0884 | 158 |
| CADC0894 | 144 |
| CADC0890 | 14 |
| CADC0886 | 11 |

## Data quality

- Missing values: **0**
- Fully duplicate rows: **0**
- Duplicate segment IDs: **0** (every segment ID is unique — confirmed 1:1 with row count, so no segment appears more than once and no adjacency/relatedness between rows is introduced via repeated IDs)

## Sampling / duration / length (evidence that segments are genuinely variable)

- `sampling` column: min=1, max=5, most common values are 1 and 5 — confirms variable sampling rate across segments, as the paper stated.
- `duration`: min=35, max=1335
- `len` (segment length in samples): min=8, max=1040

These confirm that the underlying telemetry segments are genuinely variable-length and variable-rate. `dataset.csv`, however, has already reduced each segment to ONE fixed-width row of summary statistics — there is no per-timestep sequence dimension left in this file (see the model-architecture discussion in `results/reports/opssat_ad_results.md`).

## Data types

```
segment               int64
anomaly               int64
train                 int64
channel                 str
sampling              int64
duration              int64
len                   int64
mean                float64
var                 float64
std                 float64
kurtosis            float64
skew                float64
n_peaks               int64
smooth10_n_peaks      int64
smooth20_n_peaks      int64
diff_peaks            int64
diff2_peaks           int64
diff_var            float64
diff2_var           float64
gaps_squared          int64
len_weighted          int64
var_div_duration    float64
var_div_len         float64
dtype: object
```