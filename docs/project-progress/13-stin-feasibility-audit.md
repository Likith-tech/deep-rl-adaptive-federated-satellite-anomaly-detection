# STIN Dataset — Feasibility Audit (Track C: new dataset exploration)

**Status: AUDIT ONLY. No preprocessing pipeline, no partition, no
FederatedClient wiring, no training has been implemented or run.**
Every number in this document was measured directly from the actual
downloaded files (`main` and `master` branches, plus the FL-specific
split files) — nothing here is taken from the paper or secondary
sources without independent verification against the real CSVs.

## 1. Why STIN was considered

The project's existing NSL-KDD-based Phase 6-9B pipeline has never
been tested on a dataset that is even nominally satellite-specific in
its traffic content (as opposed to Phase 4-9B's simulated satellite
*clients* on real but terrestrial NSL-KDD data). STIN
(Satellite-Terrestrial Integrated Network) was proposed as a candidate
because it is the dataset used by this project's own closest
comparable paper, already deeply documented in
`docs/research/paper-study/01-paper-notes/core-paper-fedstar.md`
("FedSTAR" — *Information Sciences*, 2026) — the paper whose reported
STIN Non-IID result (Accuracy 95.43%, per that paper's own Table) the
user asked to compare against, correctly and honestly, not to claim
reproduction of without verification.

## 2. Source

`https://github.com/kun9717/STIN-data-set/` — two branches exist,
**both inspected directly** (not assumed identical):
- `main`: README states "we have kept the original experimental data
  in this branch" — 32 columns (30 features + id + Label).
- `master`: README states "the latest changes are kept under this
  branch" **and explicitly discloses**: *"We only open source the
  characteristics of malicious traffic, and normal traffic can be
  collected by users from any legal environment."* — 17 columns (15
  features + id + Label), a reduced feature set.

License (`LICENSE.txt`): citation-required, points to "Distributed
Network Intrusion Detection System in Satellite-Terrestrial Integrated
Networks using Federated Learning" (Kun Li et al., submitted to IEEE
Access) — no redistribution restriction beyond citation; compatible
with academic project use, same posture as NSL-KDD/OPSSAT-AD.

## 3. THE DECISIVE FINDING — verified directly against the actual CSV content

**Neither `SAT20.csv` nor `TER20.csv`, on either branch, contains a
single row of normal/benign traffic.** Verified by reading every
label value in both files:

```
SAT20 (main, 82,320 data rows): UDP_DDoS=43,244  Syn_DDoS=39,076
TER20 (main, 127,244 data rows): Syn_DDoS=15,713  LDAP_DDoS=15,694
  MSSQL_DDoS=15,688  Botnet=14,622  Portmap_DDoS=14,380  UDP_DDoS=13,838
  Web Attack=13,017  Backdoor=12,762  NetBIOS_DDoS=11,530
```

No `Normal`/`BENIGN`/equivalent label exists anywhere. This exactly
confirms the `master` branch README's disclosure, and shows it applies
to **both** files, not just SAT20. `82,320 + 127,244 = 209,564` —
**exactly** matching the FedSTAR paper's own stated "STIN: ... 209,564
instances, 9 attack categories" (`core-paper-fedstar.md` §3) — this
independently confirms the paper's "STIN" is the SAT20+TER20 union,
and that even the paper's own headline dataset description is
9 attack categories, not "9 attack categories + normal." (The paper's
prose in §2 loosely says the task is "given a traffic flow's features,
output which category (normal, or one of several attack types) it
belongs to" — this is a generic problem-framing sentence, not a
STIN-specific dataset claim, and is not supported by the actual
released STIN files, which we have now verified directly.)

**Consequence:** the publicly released STIN data supports only
**multi-class attack-TYPE classification** (which DDoS/attack family a
malicious flow belongs to), not **binary normal-vs-anomaly
detection** — the task framing every prior phase of this project
(NSL-KDD Phase 1-9B, OPSSAT-AD Phase 9A) has used throughout. This is
a scope-defining fact, not a preprocessing detail, and is the single
most important output of this audit.

## 4. Exact dataset structure (measured, not assumed)

| File | Branch | Data rows | Columns | Duplicates (excl. id) | Missing values |
|---|---|---:|---:|---:|---:|
| SAT20.csv | main | 82,320 | 32 (30 features + id + Label) | 2,190 | 0 |
| SAT20.csv | master | 82,320 (+1 malformed trailing blank row) | 17 (15 features + id + Label) | not re-checked (reduced feature set) | 1 fully-blank row |
| TER20.csv | main | 127,244 | 32 (same feature set as SAT20 main) | 5,203 | 0 |

Feature list (main branch, shared by SAT20/TER20): `fl_dur, fw_pk,
l_fw_pkt, l_bw_pkt, pkt_len_min, pkt_len_max, pkt_len_std, fl_byt_s,
fl_iat_min, bw_iat_tot, bw_iat_min, bw_psh_flag, fw_urg_flag,
bw_urg_flag, fw_hdr_len, bw_pkt_s, fin_cnt, syn_cnt, psh_cnt, urg_cnt,
ece_cnt, down_up_ratio, fw_byt_blk_avg, fw_pkt_blk_avg,
fw_blk_rate_avg, bw_byt_blk_avg, bw_pkt_blk_avg, bw_blk_rate_avg,
fw_win_byt, bw_win_byt` — all numeric (no categorical/string feature
columns; CICFlowMeter-style flow statistics). Label column name is
literally `" Label"` (leading space) — a real parsing detail (must
strip column-name whitespace on load).

**Full-file (all 82,320 rows, not sampled) per-column variance check
on SAT20 main — 12 of 30 feature columns are CONSTANT (zero variance)
across the entire file**: `bw_psh_flag, fw_urg_flag, bw_urg_flag,
fin_cnt, psh_cnt, ece_cnt, fw_byt_blk_avg, fw_pkt_blk_avg,
fw_blk_rate_avg, bw_byt_blk_avg, bw_pkt_blk_avg, bw_blk_rate_avg` — all
always `0`. These would need to be dropped (zero information),
consistent with the rigor Phase 1's `feature_decisions.md` applied to
NSL-KDD.

## 5. The FL-specific split files — real structure, not assumed

`training and testing set of FL/` (main branch) contains the paper's
actual federated-learning client files:

| File | Rows | Label(s) present |
|---|---:|---|
| `SAT20_host1.csv` | 31,339 | **100% Syn_DDoS** (single class) |
| `SAT20_host2.csv` | 34,517 | **100% UDP_DDoS** (single class) |
| `TER20_host1.csv` | 56,203 | Backdoor, Botnet, LDAP_DDoS, MSSQL_DDoS, NetBIOS_DDoS |
| `TER20_host2.csv` | 45,592 | Portmap_DDoS, Syn_DDoS, UDP_DDoS, Web Attack |
| `testing set.csv` | 41,913 | **all 9 classes combined** |

**This confirms the paper's actual FL setup is 4 raw client files (2
satellite + 2 terrestrial), each maximally non-IID (single- or
few-class per host), evaluated on one shared multi-class test set** —
not the "20 simulated satellite-network clients" figure quoted in
`core-paper-fedstar.md` §1, which must therefore refer to a further
Dirichlet-based subdivision the paper performs itself beyond what's
in these 4 raw files (a real, disclosed gap between what's in the
public repo and what the paper's own experiments used).

**Leakage note (measured, not assumed):** `SAT20_host1.csv`
(31,339 rows) + `SAT20_host2.csv` (34,517 rows) = 65,856 rows —
**16,464 fewer than SAT20.csv's 82,320 total** — the host files are
NOT a complete partition of SAT20; a meaningful fraction of SAT20 rows
went elsewhere (plausibly into `testing set.csv`, which does contain
Syn_DDoS/UDP_DDoS rows). A direct exact-row overlap check (all feature
columns, excluding `id`) between `SAT20_host1.csv` and
`testing set.csv` found **264 exact-duplicate feature rows** present
in both. This dataset has many naturally-degenerate/identical flows
(e.g. minimal single-SYN-packet DDoS flows collapse to identical
feature vectors) — so this is not necessarily a train/test
construction bug, but it DOES mean **any train/validation/test split
we build ourselves must de-duplicate by full feature-row content, not
rely on the `id` column, or genuine leakage will occur.**

## 6. Answering the 20 audit questions (Step 2), using only verified facts

1. **Files available**: `SAT20.csv`, `TER20.csv` (both branches, differ
   in row/feature detail per §3-4), plus 5 files under `training and
   testing set of FL/`.
2. **SAT20 row count**: 82,320 (main branch; verified).
3. **TER20 row count**: 127,244 (main branch; verified).
4. **Feature count**: 30 numeric features (main branch); 15 on master
   (reduced set).
5. **Feature names**: listed in §4.
6. **Label column**: `" Label"` (leading space), string attack-family name.
7. **Attack classes**: SAT20 = 2 (UDP_DDoS, Syn_DDoS); TER20 = 9 total
   combined with SAT20 (adds LDAP_DDoS, MSSQL_DDoS, Botnet,
   Portmap_DDoS, Web Attack, Backdoor, NetBIOS_DDoS).
8. **Normal/attack distribution**: **0% normal, 100% attack**, in both
   SAT20 and TER20, both branches — §3's decisive finding.
9. **Missing values**: 0 empty fields in SAT20/TER20 main; 1 fully-blank
   malformed trailing row in SAT20 master.
10. **Duplicate records**: 2,190 (SAT20 main), 5,203 (TER20 main),
    excluding the `id` column.
11. **Categorical features**: none — all 30 features are numeric.
12. **Numeric features**: all 30 (12 of which are constant/zero-variance
    across the whole file, per §4).
13. **Train/test splits**: yes — `training and testing set of FL/`
    provides an author-defined split (§5), though it is NOT a clean
    partition of SAT20/TER20.csv (16,464-row shortfall, §5) and its
    exact construction method is undocumented.
14. **Can it be split without leakage**: only with explicit
    full-feature-row de-duplication first (§5's 264-row overlap
    finding) — not safe to split naively by `id` or by naive
    random/stratified split without dedup.
15. **Is SAT20 genuinely satellite traffic**: per the paper's own
    Conclusion (`core-paper-fedstar.md` §3, §21): **"None are real,
    operational satellite traffic captures."** SAT20 is
    satellite-network-*relevant* (includes DTN-protocol-adjacent DDoS
    patterns per the paper) but is simulated/testbed-generated, not
    real captured satellite telemetry — the same "simulated, not real"
    caveat this project already applies rigorously to its own Phase
    4-9B satellite-client simulation, and clearly distinct from Phase
    9A's genuinely real OPSSAT-AD telemetry.
16. **Temporal information**: flow-level duration/inter-arrival-time
    features exist (`fl_dur`, `fl_iat_min`, `bw_iat_tot`, `bw_iat_min`)
    but no wall-clock timestamp column — cannot order flows in real
    time or check for temporal leakage the way OPSSAT-AD's real
    timestamps allowed.
17. **Client/node identifier**: no explicit per-satellite/node ID
    column inside SAT20.csv itself — the only real "client" structure
    that exists is the author-provided `SAT20_host1`/`SAT20_host2`
    file split (§5), which is 2 clients, not 8.
18. **Does it support our 8-client FL setup**: **not natively** — only
    2 real satellite-side host files exist. An 8-client structure
    would have to be a project-imposed simulation (further subdividing
    host1/host2, or incorporating the 2 terrestrial hosts as
    additional "clients" under an explicit, disclosed
    satellite-terrestrial-integrated framing) — must be stated
    explicitly as simulated, exactly as Step 4 requires.
19. **Compatible with existing MLP/local-training pipeline**: the
    *shape* is compatible (fixed-width numeric feature vector, same
    pattern as NSL-KDD/OPSSAT-AD) — but the *loss/output layer* is
    not, without a decision: our `BaselineMLP` + `BCEWithLogitsLoss` +
    `output_dim=1` assumes binary classification; STIN as actually
    released only supports multi-class (`CrossEntropyLoss`,
    `output_dim=2` for SAT20-only or up to 9 for the SAT+TER union) —
    see §7.
20. **Does the paper's STIN result use the same SAT20 data**: **the
    paper's reported STIN result uses SAT20+TER20 COMBINED** (§3's
    209,564-row match), not SAT20 alone — a claim of comparability to
    the paper's 95.43% figure using SAT20 alone would already be
    invalid on dataset-scope grounds, independent of every other
    protocol difference.

## 7. The fork point — requires an explicit decision before any code is built

Per this project's own "do not blindly download and train" instruction
and its scientific-integrity policy (never conflate attack-type
classification with anomaly/normal detection — the exact discipline
already applied to OPSSAT-AD's operational-fault-vs-cyberattack
distinction in Phase 9A), the missing normal class is a genuine fork
point, not a detail to route around silently:

- **Option 1 — reframe the task honestly as multi-class attack-type
  classification** (SAT20-only: 2-class; or SAT20+TER20: up to
  9-class) under the existing FL/DQN framework. Scientifically valid,
  but a materially different task from every prior phase's anomaly
  detection, and requires real architecture changes (loss function,
  output layer, evaluation metric definitions) — not a drop-in dataset
  swap.
- **Option 2 — do not proceed with STIN** for an anomaly-detection
  comparison; document why (this audit) and consider it closed, or
  revisit only if a genuinely normal-traffic-inclusive version becomes
  available.
- **Option 3 — source separate normal traffic** ourselves, as the
  `master` README suggests users do. Rejected as infeasible under this
  project's real-data-only, no-fabrication policy: we have no
  legitimate satellite/terrestrial normal-traffic capture source, and
  synthesizing one would not be "real data."

**This audit does not choose between these — it is the explicit
decision point handed back to the user before Steps 4-6 (partitioning,
compatibility code, smoke test) are built**, since the correct design
for all of those depends entirely on which option is chosen.

## 8. Existing-code compatibility (assessed, not yet wired up)

Reusable unchanged regardless of which option above is chosen:
`FederatedClient`, `FederatedServer`, `fedavg.py`,
`src/federated/adaptive.py`, `src/drl/{action_space,state,reward,dqn_agent,replay_buffer}.py`,
`src/federated/{dqn_controller,drl_environment}.py` — all
dataset-agnostic (operate on parquet paths + a `feature_columns` list
+ a generically-configured `BaselineMLP`). A new
`src/satellite/stin_loader.py` (paralleling the existing
`src/satellite/loader.py` built for OPSSAT-AD) would be the isolated
adapter — **not created yet**, since its exact label-handling logic
depends on the §7 decision. `BaselineMLP`'s architecture itself
(`input_dim`/`hidden_dimensions`/`output_dim` all configurable) is
reusable either way; only its configured `output_dim` and the training
loss function would differ.

## 9. What was NOT done (explicitly, per instruction)

No preprocessing pipeline, no train/validation/test protocol, no
8-client partition, no `src/satellite/stin_*.py` files, no
compatibility test, no one-round FL smoke test, no DQN training. Only
download + inspection (Steps 1-3) were performed, using the real
dataset throughout — no secondary-source numbers were trusted without
independent verification.
