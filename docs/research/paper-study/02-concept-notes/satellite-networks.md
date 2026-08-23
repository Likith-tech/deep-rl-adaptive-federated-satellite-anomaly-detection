# Concept — Satellite Networks (and the "telemetry vs. network traffic" distinction)

## 1. What is it?

A **satellite network** is a set of satellites (often Low Earth Orbit / LEO constellations today) that communicate with each other (inter-satellite links), with ground stations, and with end users, to provide services like internet connectivity, navigation, or Earth observation.

There are two almost entirely different things people mean by "anomaly" on a satellite, and your 4 papers split cleanly along this line:

| | **Telemetry anomaly** | **Network anomaly** |
|---|---|---|
| What's abnormal | A physical/operational fault: overheating, power fault, sensor drift, attitude-control failure | Malicious network traffic: DDoS, botnet, port scan, intrusion |
| Data looks like | A time series of sensor readings (temperature, voltage, radiation…) | Network flow records (packet size, protocol, timing, source/destination…) |
| Papers in your set | Paper 1 (Yang), Paper 2 (Maleki Sadr), Paper 3 (Hundman) | Paper 4 (Chen/FedSTAR) |
| Your project's matching track | Track B — OPSSAT-AD (real single-spacecraft telemetry) | Track A — NSL-KDD (terrestrial intrusion data, simulated as satellite clients) |

## 2. Why does it exist (as a research area)?

Satellites are becoming critical infrastructure (6G, global broadband, navigation, disaster response). Two separate engineering concerns follow from that: (a) keep the spacecraft itself healthy (telemetry monitoring — decades-old concern, predates cybersecurity concerns), and (b) keep the network they form secure from attackers (network anomaly/intrusion detection — a newer concern, growing as satellite internet becomes more valuable and more exposed).

## 3. Simple example

- **Telemetry anomaly:** a battery temperature sensor reads 45°C when it should read ~20°C given the spacecraft's current orientation — Papers 1–3's world.
- **Network anomaly:** a burst of malformed DTN-protocol packets floods a satellite's inter-satellite link, trying to deny service to legitimate traffic — Paper 4's world.

## 4. How it works (architecturally, per your papers)

- **Ground-based satellite architecture:** satellites relay all traffic through terrestrial gateways (simple, but ground-dependent).
- **Space-based architecture:** satellites use Inter-Satellite Links (ISLs) and On-Board Processing (OBP) to route directly between each other, reducing terrestrial dependence — this is the architecture Paper 4 (FedSTAR) focuses on, and it's why on-board, low-bandwidth solutions (like federated learning) matter there.

## 5. Why the papers use it

- Papers 1–3 use it because a single spacecraft's health directly determines whether the mission survives — this has been an engineering necessity since long before "AI" was involved (traditional out-of-limits alarms).
- Paper 4 uses it because as satellite internet scales up (LEO mega-constellations), the *attack surface* scales up too — more satellites, more links, more entry points for attackers.

## 6. Advantages / 7. Limitations (of the *satellite* setting itself, not any one paper)

**Advantages of studying satellites specifically (vs. generic IoT/network security):** satellite-specific constraints (limited bandwidth, intermittent contact windows, high mobility, harsh radiation environment) are genuinely different from terrestrial networks, so satellite-specific datasets/methods (like STIN, or OPSSAT-AD) matter.

**Limitations that show up repeatedly across all 4 papers:** genuine, real, multi-satellite operational data (either telemetry *or* network traffic) is essentially unavailable publicly. Every paper in your set works around this: Papers 1–3 use one real spacecraft's telemetry at a time (never a fleet); Paper 4 uses benchmark datasets built to *resemble* satellite-terrestrial traffic, not live captures. Your own project does the same thing, honestly: it simulates a multi-satellite population from terrestrial NSL-KDD data (Track A) and separately validates on one real spacecraft's telemetry (Track B, OPSSAT-AD) — but never combines the two.

## 8. Which papers use it

All 4 — but split into the two sub-types above (telemetry: Papers 1, 2, 3; network traffic: Paper 4).

## 9. How it connects to our project

Your capstone's title says "Satellite Network Anomaly Detection" — given your project's actual implementation (NSL-KDD-based Track A simulating satellite clients as network-intrusion detectors, plus a separate real-telemetry Track B), your project's *network* framing is architecturally closer to Paper 4's problem (network/intrusion anomalies across multiple clients) than to Papers 1–3's problem (single-spacecraft physical telemetry). Being able to explain this distinction clearly is one of the most important things to walk into your viva with.
