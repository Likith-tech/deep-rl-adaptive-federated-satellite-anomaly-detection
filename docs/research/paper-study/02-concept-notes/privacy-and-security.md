# Concept — Privacy, Security, and the Server/Client Information Boundary

## 1. What is it?

> **Technical term → simple meaning → example**
> **Privacy boundary (in federated learning):** an explicit, enforced rule about exactly what information is allowed to cross from a client to the server (and vice versa) — usually: model weights and simple summary statistics, never raw data.
> *Analogy:* an exam where students submit only their final answer sheet, never their scratch paper — the grader (server) never sees *how* a student worked through the problem, only the outcome.

**Two different "security" concerns show up in your papers, and it's important not to conflate them:**

1. **Privacy** (keeping raw data local) — addressed, at least at a basic level, by Paper 4 and by your own project.
2. **Robustness against malicious/Byzantine clients** (a compromised client deliberately sending bad updates to sabotage the global model) — **explicitly left unsolved** by Paper 4 ("The adversarial robustness of the federated optimization process itself is left for future investigation" — stated twice in the paper), and **also unaddressed** by your own project (no phase currently tests against a malicious/adversarial client).

## 2. Why does it exist?

Satellites and their ground operators may not want (or be legally/operationally able) to share raw telemetry or network-traffic data outside their own control — and even if they were willing, the bandwidth cost of shipping raw data down from orbit is a genuine engineering constraint, independent of privacy.

## 3. Simple example

A satellite detects unusual network traffic. Under a privacy-respecting federated system, it never sends the raw traffic logs anywhere — it only sends its locally-updated model weights, from which the raw traffic itself cannot be directly reconstructed by the server.

## 4. How it works, concretely, in your own project (this is a genuine engineering feature, not just a stated goal)

Your project enforces this boundary with **real, automated tests**, not just documentation: `tests/federated/test_server.py` includes an AST-based test (`test_server_module_never_imports_pandas_or_touches_parquet`) that parses the actual Python source of `server.py` and fails the build if it ever imports pandas or reads a parquet file — i.e., the server is *mechanically prevented* from reading client data files, not just asked nicely not to. This same style of test was extended to every new Phase 9B DRL module (`tests/drl/test_privacy_boundary.py`).

## 5. Why the papers/project use it

Paper 4 frames privacy as one of federated learning's two core motivations (alongside bandwidth). Your project extends this from "a design goal" to "a continuously tested invariant" — arguably a stronger, more defensible standard than either paper in your set states explicitly (Papers 1–3 don't discuss privacy at all, since they have no multi-client structure).

## 6. Advantages

- Genuinely reduces what a compromised or curious server could learn.
- A tested (not just claimed) boundary is far more defensible in a viva or a report than an untested one — you can say "here is the actual test that fails the build if this is ever violated," which is a materially stronger claim than "we designed it to be private."

## 7. Limitations

- Model weights themselves *can* leak some information about training data under certain attacks (e.g., membership inference, model inversion) — neither Paper 4 nor your project addresses this deeper privacy-attack surface; both only address the more basic "raw data never transmitted" property.
- **Byzantine robustness is a genuinely open problem** in both Paper 4 and your own project — a malicious client could, in principle, send a deliberately corrupted model update, and nothing in either system currently detects or defends against that.

## 8. Which papers use it

Only Paper 4 discusses privacy explicitly (as FL's core motivation). None of Papers 1–3 have a multi-client setting for privacy to even apply to.

## 9. How it connects to our project

This is a genuinely strong, concrete point in your project's favor for viva: while Paper 4 (and most FL literature generally) *states* a privacy goal, your project *tests* it mechanically via AST-based source-code inspection, for both the FedAvg server (Phase 6) and every new DRL module (Phase 9B). If asked "how do you know the server can't see raw client data," the correct, strong answer is: "there is an automated test that inspects the actual import statements and function calls in the server code and fails if it ever touches a data file — this isn't just a design intention." At the same time, be honest that **Byzantine/adversarial-client robustness remains open** — this is a genuine, shared limitation with Paper 4, and a legitimate direction for future work, not something to claim is already solved.
