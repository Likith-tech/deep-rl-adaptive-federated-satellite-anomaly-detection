# Phase 6 — Federated Learning Baseline (FedAvg)

## What was the goal?

Introduce the first actual Federated Learning system in this project.
Up to now, Phase 5 trained 8 separate models that never talked to each
other. Phase 6 lets those same 8 satellites cooperate to build ONE
shared model — without any of them sending their raw data anywhere.

## What is Federated Learning?

A way for multiple devices (here, 8 simulated satellites) to jointly
train one shared model while each device's data stays exactly where it
is. Instead of collecting everyone's raw data in one place, each device
trains locally and only sends back what it learned (its updated model),
which then gets combined into a better shared model.

## What is FedAvg?

"Federated Averaging" — the original, simplest way to combine what
different clients learned. Each client sends back its updated model
parameters, and the server averages them, weighted by how much local
data each client trained on (a client with more data gets more say in
the average). That's the ENTIRE aggregation rule:

```
w_global = sum over clients k of (n_k / N) * w_k

n_k = client k's number of local training samples
N   = total samples across all participating clients
w_k = client k's model parameters after local training
```

## What happens in one round?

1. The server has a current shared ("global") model.
2. It sends an identical copy of that model to all 8 satellites.
3. Each satellite trains it for a bit using ONLY its own local data.
4. Each satellite sends back its updated model (not its data).
5. The server averages all 8 updated models (weighted by sample count)
   into a new global model.
6. That new global model becomes the starting point for the next round.

We ran this 10 times ("10 communication rounds").

## Do satellites send raw data?

**No.** At no point does any satellite send a data record to the
server. The server only ever receives a client's model parameters and
how many samples it trained on — never the samples themselves. This is
enforced in the code, not just promised: the server's aggregation
module (`src/federated/server.py`) never imports anything capable of
reading a data file, and a dedicated test checks that.

## What did we use?

The same 8 simulated satellite clients from Phase 4 (`SAT-01` through
`SAT-08`), each with its own non-IID slice of real NSL-KDD training
data.

## What model?

The exact same simple Phase 2 MLP architecture used in every earlier
phase (`121 → Dense(128) → ReLU → Dropout → Dense(64) → ReLU → Dropout
→ Dense(1) → logit`). We kept the architecture unchanged on purpose —
this way, any difference in results comes from Federated Learning
itself, not from a different or bigger model.

## How many rounds?

**10 communication rounds**, 1 local training epoch per client per
round. All 8 clients participated in every single round — no clients
were skipped or selected differently (that's a later phase).

## What happened?

Global validation performance improved steadily and consistently every
round, from F1 = 0.9683 (round 1) to F1 = 0.9879 (round 10) — the best
round was the last one, round 10, chosen using validation loss alone
(never KDDTest+). We then evaluated that one selected model, once, on
KDDTest+:

| Metric | Phase 6 FedAvg (KDDTest+) |
|---|---:|
| Accuracy | 0.7531 |
| Precision | 0.9219 |
| Recall | 0.6186 |
| F1 | 0.7404 |
| ROC-AUC | 0.8479 |
| FPR | 0.0693 |

Full round-by-round numbers, the confusion matrix, and every comparison
are in `results/reports/federated_results.md`.

## Did FedAvg improve over centralized MLP?

**Honestly, no — not on this run.** On the same final KDDTest+ set,
FedAvg scored F1 = 0.7404 versus the Phase 2 centralized model's F1 =
0.7790 — about 3.9 points lower. This is a real, measured result, not
adjusted to look better. A plausible reason: FedAvg only got 10 rounds
× 1 local epoch of training (10 total local passes over each client's
data), while the centralized model trained for up to 30 full epochs
over the whole dataset — so this comparison isn't yet about whether
federation *can* work, but about whether *this specific*, deliberately
small, easy-to-interpret baseline configuration already matches a much
more thoroughly trained centralized model. It doesn't, and that's a
useful, honest finding to build on.

## Did FedAvg improve over local-only training?

**On the global validation set — yes, on average.** FedAvg's validation
F1 (0.9879) was higher than the mean of the 8 Phase 5 local-only
clients (0.9721), though not quite as high as the single best
individual local client (SAT-02, 0.9918). We're careful to compare like
with like here: Phase 5 only has global-validation numbers (it never
touched KDDTest+), so this specific comparison also uses global
validation for FedAvg, not its KDDTest+ number — see
`results/reports/federated_results.md` section 14 for the full, explicit
breakdown of which comparison uses which dataset split.

## How do I explain this to my mam?

"In Phase 6, we introduced real Federated Learning using an algorithm
called FedAvg. Each of our 8 simulated satellites trained the same
model locally on its own data, which is different for every satellite,
and sent back only the trained model — never the data itself — to a
central coordinator. The coordinator averaged all 8 updated models into
one shared model and sent it back out for another round. We did this 10
times. We picked the best round using a validation dataset, and then
tested that one final model, just once, on our held-out test set. It
did better than training completely alone on any single satellite on
average, but it didn't yet beat our original, more thoroughly-trained
single-computer model — a fair and useful starting point to improve on
in later phases."

## Important terms

- **Federated Learning:** training one shared model across multiple
  devices/clients without any of them sharing their raw data.
- **Client:** one simulated satellite that trains locally and sends
  back only its updated model.
- **Server:** the central coordinator that averages client updates into
  a new global model. It never sees raw client data.
- **Local update:** a client's model after it has trained a bit on its
  own local data — what actually gets sent to the server.
- **Global model:** the current shared model, identical across all
  clients at the start of each round.
- **Communication round:** one full cycle of: server sends global model
  → clients train locally → clients send updates → server averages.
- **FedAvg:** the specific averaging rule used to combine client
  updates, weighted by how much data each client trained on.
- **Weighted averaging:** giving clients with more local data
  proportionally more influence over the resulting global model.
- **Non-IID:** each client's local data comes from a meaningfully
  different distribution than the others (established in Phase 4) —
  this is what makes Federated Learning genuinely hard.

## Limitations

- Satellite conditions (bandwidth, latency, compute, availability,
  connectivity) remain simulated, as in Phase 4 — they were recorded in
  this phase's round logs but did NOT influence training or
  aggregation.
- NSL-KDD is a terrestrial network intrusion dataset — not real
  satellite telemetry.
- Only standard, **synchronous** FedAvg is implemented — every client
  finishes its local training before the round can complete; no
  asynchronous updates.
- **All 8 clients participate every round** — no adaptive client
  selection, no partial participation, no dropout modeling.
- No DRL, DQN, or reinforcement learning of any kind was used to guide
  client selection, weighting, or aggregation.
- No FedProx, personalization, secure aggregation, or differential
  privacy — those are separate techniques, out of scope here.
- The round count (10) and local epochs (1) were chosen for a clean,
  interpretable baseline, not tuned through a large search.

## What comes next?

Phase 7 — further Federated Learning experiments focused specifically
on the non-IID aspect (per this tracker's numbering and the README's
20-phase table, where "Non-IID experiments" is row 9, right after this
phase's "Standard FedAvg", row 8). Adaptive Federated Learning — using
the simulated resource metadata already being recorded (bandwidth,
latency, compute, availability, connectivity) to make smarter decisions
about which clients participate or how their updates are weighted —
follows after that (row 10). Both build on this phase's honest FedAvg
baseline.
