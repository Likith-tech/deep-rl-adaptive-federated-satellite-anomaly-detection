# Feature Decisions — NSL-KDD

Every raw column is listed below with an explicit keep/transform/drop
decision and the reason. NSL-KDD has no identifier, source/destination
IP, or timestamp columns (unlike raw packet-capture datasets), so the
leakage surface here is much smaller than for CICIDS2017/UNSW-NB15 —
but two columns still require a deliberate decision.

| Feature | Decision | Reason |
|---|---|---|
| `duration` | Keep | Legitimate traffic feature (connection length in seconds). |
| `protocol_type` | Keep (encode) | Categorical traffic feature (tcp/udp/icmp). |
| `service` | Keep (encode) | Categorical traffic feature (network service, e.g. http, ftp). |
| `flag` | Keep (encode) | Categorical traffic feature (connection status flag). |
| `src_bytes` | Keep | Legitimate traffic volume feature. |
| `dst_bytes` | Keep | Legitimate traffic volume feature. |
| `land` | Keep | Binary traffic feature (same src/dst host+port). |
| `wrong_fragment` | Keep | Legitimate traffic feature. |
| `urgent` | Keep | Legitimate traffic feature. |
| `hot` | Keep | Legitimate traffic feature ("hot indicator" count). |
| `num_failed_logins` | Keep | Legitimate traffic feature. |
| `logged_in` | Keep | Binary traffic feature. |
| `num_compromised` | Keep | Legitimate traffic feature. |
| `root_shell` | Keep | Legitimate traffic feature. |
| `su_attempted` | Keep | Legitimate traffic feature. |
| `num_root` | Keep | Legitimate traffic feature. |
| `num_file_creations` | Keep | Legitimate traffic feature. |
| `num_shells` | Keep | Legitimate traffic feature. |
| `num_access_files` | Keep | Legitimate traffic feature. |
| `num_outbound_cmds` | Keep | Legitimate traffic feature (constant 0 in NSL-KDD — see limitation below, but not leakage). |
| `is_host_login` | Keep | Binary traffic feature. |
| `is_guest_login` | Keep | Binary traffic feature. |
| `count` | Keep | Legitimate traffic-window feature (connections to same host in last 2s). |
| `srv_count` | Keep | Legitimate traffic-window feature. |
| `serror_rate` | Keep | Legitimate traffic-window feature. |
| `srv_serror_rate` | Keep | Legitimate traffic-window feature. |
| `rerror_rate` | Keep | Legitimate traffic-window feature. |
| `srv_rerror_rate` | Keep | Legitimate traffic-window feature. |
| `same_srv_rate` | Keep | Legitimate traffic-window feature. |
| `diff_srv_rate` | Keep | Legitimate traffic-window feature. |
| `srv_diff_host_rate` | Keep | Legitimate traffic-window feature. |
| `dst_host_count` | Keep | Legitimate host-based traffic feature. |
| `dst_host_srv_count` | Keep | Legitimate host-based traffic feature. |
| `dst_host_same_srv_rate` | Keep | Legitimate host-based traffic feature. |
| `dst_host_diff_srv_rate` | Keep | Legitimate host-based traffic feature. |
| `dst_host_same_src_port_rate` | Keep | Legitimate host-based traffic feature. |
| `dst_host_srv_diff_host_rate` | Keep | Legitimate host-based traffic feature. |
| `dst_host_serror_rate` | Keep | Legitimate host-based traffic feature. |
| `dst_host_srv_serror_rate` | Keep | Legitimate host-based traffic feature. |
| `dst_host_rerror_rate` | Keep | Legitimate host-based traffic feature. |
| `dst_host_srv_rerror_rate` | Keep | Legitimate host-based traffic feature. |
| `attack` | **Split into `label_original` + `label_binary`; not used as an input feature** | This is the ground-truth attack name — it is the prediction target, not a feature. Preserved verbatim as `label_original` for future multi-class work; mapped to `label_binary` (0=normal, 1=anomaly) for the current binary anomaly-detection objective. Never fed into X. |
| `difficulty` | **Drop from features; kept in metadata only** | This is a difficulty score assigned by the *original KDD'99 classifiers' error rate* on each record — i.e. it directly encodes how easy/hard prior models found this exact record to classify. Using it as an input feature would leak information correlated with the label (harder records skew toward specific attack types) and is not a real network-traffic characteristic. It is not deleted from the dataset — it is retained in `data/interim/` for reference/analysis — but it is excluded from the model input feature set `X`. |

## No timestamp columns

NSL-KDD connection records have no timestamp or sequence-order field —
this is a per-connection dataset, not a continuous flow capture. There is
therefore nothing to (incorrectly) treat as temporal order, and no
timestamp leakage risk to consider. The train/validation/test split
strategy is documented in `docs/datasets/dataset_selection.md` and
`results/reports/dataset_quality.md`.

## No identifier columns

Unlike CICIDS2017/UNSW-NB15 (which include flow IDs, source/destination
IPs and ports), NSL-KDD's schema does not expose IP addresses or record
identifiers, so there is no host-identifier leakage risk to remove.
