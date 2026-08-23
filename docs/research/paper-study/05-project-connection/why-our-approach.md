# Why Our Approach (Explaining Design Choices Against the Literature)

This file exists to answer, in advance, the "why did you do X instead of Y" questions that naturally follow from having read these 4 papers. Each answer is grounded in your project's actual documented reasoning, not invented after the fact.

## Why DQN instead of SAC (like Paper 4)?

Because the action space was **deliberately kept small and fully discrete**: 9 named, inspectable weight-profiles, with no clustering step and no continuous knobs. DQN is the natural, simplest algorithm for a purely discrete action space; SAC's extra machinery (actor + twin critics, continuous-action support, entropy tuning) would add complexity with no corresponding benefit for a problem this narrowly scoped. See `02-concept-notes/deep-reinforcement-learning.md`.

## Why weighting, not client selection or clustering (unlike Paper 4)?

Your project's own Phase 8/9B design documentation gives an explicit, principled reason: hard selection (or clustering that could effectively exclude a client from a round) risks silently dropping a client that happens to hold a rare attack category for that entire round — directly conflicting with the fairness goal Phase 7 demonstrated matters (a 41x fairness-gap swing was measured from heterogeneity alone). Weighting keeps all 8 clients participating every round, which is a **safer choice for a first DRL experiment** because it isolates "did the agent learn a good weighting policy" from "did the agent also learn not to accidentally starve a client" — two separable questions a small, first pilot should not conflate.

## Why NSL-KDD (terrestrial data), not a satellite-specific benchmark like STIN?

Not fully explained in existing project documents beyond "Track A uses NSL-KDD" as an established prior-phase decision (Phase 1) that Phase 4 onward builds on. **Honest answer for viva:** this is a reasonable question to flag as a genuine design choice worth re-examining — STIN (Paper 4's dataset) is a more satellite-relevant benchmark that didn't exist as a well-known option when this project's early phases were scoped, or was not selected for reasons not fully documented in the material available for this analysis. If asked directly, the defensible answer is: "our project's Track A intentionally uses a well-established, thoroughly-studied terrestrial intrusion dataset to isolate and validate the *federated + DRL mechanism* on a trusted, easy-to-benchmark base, while Track B (OPSSAT-AD) separately validates against real spacecraft data — using STIN specifically for Track A is a legitimate future extension, not something ruled out."

## Why a three-stage comparison (fixed → rule-based → learned) instead of going straight to DRL like Paper 4?

Because a fixed rule (Phase 6's plain FedAvg) is the field-standard baseline, and jumping directly to a learned policy without first establishing whether a *simple, hand-written* rule already captures most of the benefit would make it impossible to know whether DRL is adding anything. Phase 8's own ablation results (no single fixed rule wins on both F1 and fairness) is exactly the evidence needed to justify attempting Phase 9B at all — and, honestly, Phase 9B's result (not yet beating Phase 8) shows this staged approach was the right caution: a less careful project might have gone straight to "we built a DQN, therefore it's better," without ever measuring whether that claim is actually true.

## Why report the DQN's negative result instead of tuning until it wins?

Because doing otherwise would make the result scientifically meaningless. Your project's own standing rule (documented across Phases 3, 7, and 9B) is that a worse-than-baseline result is a valid, reportable scientific outcome — not something to be quietly re-run or adjusted until a better number appears. This is the same standard of honesty Paper 4 itself follows when it reports non-significant p-values (Table 4) rather than hiding them.

## Why does Track B (OPSSAT-AD) exist separately from Track A, instead of trying to force them together?

Because OPSSAT-AD is a single real spacecraft with no federated/multi-client structure — attempting to artificially split one spacecraft's data into "multiple clients" would require an invented, unjustified partitioning scheme, and your project's documentation is explicit that this "would require its own separately-scoped design, not a Phase 9B side effect." Keeping the two tracks honestly separate, rather than merging them prematurely, avoids overclaiming ("our federated DRL system works on real spacecraft data") when what's actually true is narrower ("our federated DRL mechanism works on simulated satellite clients; our detection architecture separately works on real spacecraft data").
