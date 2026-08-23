# Concept — Federated Learning (FL), FedAvg, and Client/Server Architecture

## 1. What is it?

> **Technical term → simple meaning → example**
> **Federated Learning:** training one shared model across many separate devices/clients, where each client trains on its own local data and only sends the *model updates* (not the raw data) to a central server, which combines them.
> *Analogy:* imagine 20 hospitals training the same "diagnose disease X" AI. Instead of sending patient records to one place (privacy risk, huge bandwidth), each hospital trains locally on its own patients and only sends the *lessons learned* (model weight changes) to a coordinator, who blends them into one better shared model.

**FedAvg (Federated Averaging):** the original, simplest way to combine client updates — average them, weighted by how much data each client trained on. Introduced by McMahan et al. (2017), and the starting point every paper/your-project's own work builds on.

## 2. Why does it exist?

Two forces push toward it: **(a) privacy** — clients may not be allowed/willing to share raw data (patient records, network traffic, spacecraft telemetry); **(b) bandwidth** — sending raw data (potentially terabytes for satellites) costs far more than sending compact model updates.

## 3. Simple example

8 simulated satellite clients each train a small neural network on their own local slice of network-traffic data (never shared), then send their trained weights to a server, which averages them (weighted by how much data each client had) into one improved global model — sent back to every client for the next round.

## 4. How it works

```text
for each communication round:
    server sends the current global model to all (or a sampled subset of) clients
    each client trains locally for a few epochs on its own private data
    each client sends back only its updated model weights (not its data)
    server combines all received weights into a new global model
    (repeat)
```

## 5. Why the papers/project use it

Paper 4 (FedSTAR) uses it because raw satellite network traffic shouldn't leave each satellite (privacy + bandwidth). Your own project uses it for the same stated reasons, and treats it as the honest structural stand-in for "many satellites, one shared detector, no raw-data sharing" even while its underlying data (NSL-KDD) is terrestrial, not real orbital traffic.

## 6. Advantages

- No raw data leaves the client → real privacy benefit.
- Much less communication than shipping raw data.
- Lets many small, resource-constrained devices/satellites contribute to one better shared model.

## 7. Limitations

- **Plain FedAvg breaks down under Non-IID data** (see `non-iid-data.md`) — this is the single most important limitation driving *both* Paper 4's SAC-based clustering and your own project's Phase 8 (rule-based) / Phase 9B (DQN-based) adaptive weighting.
- Vulnerable to malicious/compromised clients (Byzantine attacks, poisoning) — explicitly flagged as unsolved by Paper 4's own Conclusion, and equally unaddressed in your own project.
- Still requires *some* communication every round — not free, and this cost is explicitly measured in Paper 4 (MiB/round) and implicitly relevant to your own project's DQN action space (all 8 clients participate every round, by design, in every phase of your project — Phase 6 through 9B — precisely to avoid the fairness risk of dropping a client).

## 8. Which papers use it

Only Paper 4 (Chen/FedSTAR) among your 4 papers. Papers 1–3 are entirely single-spacecraft, centralized-training papers with no FL of any kind.

## 9. How it connects to our project

FL is the "Federated" half of your capstone's title, and your project already has a working, tested FL implementation across Phases 6–9B: `src/federated/{client,server,protocol,fedavg}.py` (plain FedAvg, Phase 6), extended by `adaptive.py`/`adaptive_trainer.py` (Phase 8's rule-based weighting) and `drl_environment.py`/`dqn_controller.py`/`drl_round_provider.py` (Phase 9B's DQN-driven weighting) — all real, already-run code, not a plan.
