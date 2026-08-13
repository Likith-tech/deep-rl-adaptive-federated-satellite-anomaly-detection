# Phase 9 Real Spacecraft Dataset — Research Prompt Archive

**Date:** 2026-08-13
**Branch at time of research:** `project-development-2` (Phases 0-6 committed locally, Phase 6 not yet pushed; Phases 7-8 uncommitted in working tree)
**Requested by:** Project owner, as a pre-Phase-9 pause for dataset due diligence

## Purpose

Before beginning Phase 9 (originally planned as DRL-based adaptive federated learning), the
project owner asked for a structured, evidence-based investigation into whether a REAL
spacecraft/satellite dataset exists that could strengthen the project's dataset story —
without replacing NSL-KDD, without touching Phases 1-8, and without taking any
implementation action until explicitly approved.

This document preserves the exact instructions given for that investigation, so the
research trail is reproducible and explainable to a project guide later, independent of
this conversation's context window.

## Project context (as stated at the time)

- Project: "Deep Reinforcement Learning-Driven Adaptive Federated Framework for Satellite
  Network Anomaly Detection"
- Repository: `deep-rl-adaptive-federated-satellite-anomaly-detection`
- Completed phases: 0 (foundation), 1 (NSL-KDD pipeline), 2 (centralized MLP baseline),
  3 (temporal GRU+attention), 4 (simulated 8-satellite non-IID environment),
  5 (independent local satellite training), 6 (synchronous FedAvg baseline),
  7 (controlled non-IID FedAvg experiments), 8 (rule-based adaptive federated learning)
- Current primary dataset: NSL-KDD — real, terrestrial, network intrusion-detection data
- Satellite layer: entirely simulated (client IDs, Dirichlet partitioning, simulated
  resource conditions) built on top of the real NSL-KDD records

## Research scope requested

1. Inspect the current repository state (git branches/commits, existing dataset docs,
   existing research docs) before creating anything.
2. Create a durable research-archive directory structure under `docs/research/` (this
   file lives in that structure).
3. Archive this exact prompt (this file).
4. Independently search for real spacecraft/satellite datasets, prioritizing primary
   institutional sources (NASA, ESA, JAXA, official mission repositories, Zenodo,
   IEEE/ACM, dataset-author repositories) over blogs/Kaggle-only listings.
5. Independently re-verify five previously-identified candidates: NASA SMAP/MSL,
   ESA-ADB, STIN/SAT20, CuCD-ID, LENS.
6. Search for additional candidates beyond those five, and explicitly determine whether
   any publicly available dataset combines REAL spacecraft/satellite data with REAL
   cyberattack labels ("Category A") — and to state plainly if none is found, rather than
   stretching a weaker candidate to fit.
7. Produce a formal comparison document with a scoring table and explicit
   FACT / SOURCE CLAIM / ENGINEERING ASSESSMENT / UNCERTAINTY labeling.
8. Produce a 23-section detailed verification report.
9. Preserve the raw research findings verbatim, including inconvenient limitations —
   explicitly instructed not to silently drop unfavorable findings, and to mark
   unverifiable claims as `UNVERIFIED`.
10. Make only small, additive documentation updates to `docs/datasets/dataset_selection.md`
    noting that real-spacecraft datasets are under evaluation — no replacement of NSL-KDD,
    no change to any Phase 1-8 result.
11. End with an explicit, clearly-labeled recommendation that is NOT an approval to act —
    every recommendation must be marked "RECOMMENDATION — AWAITING USER APPROVAL."
12. Validate the documentation itself (no duplicate files, valid Markdown, every major
    factual claim sourced, no guessed licenses, real-vs-simulated stated explicitly,
    cyber-vs-operational anomaly distinction stated explicitly).

## Explicit restrictions given

Do NOT, under any circumstances in this task:
- integrate any dataset
- download large datasets
- modify existing model/training code
- modify Phase 1-8 implementations
- replace NSL-KDD
- create Phase 9 implementation code
- create a new branch (`project-development-3` or otherwise)
- commit
- push
- delete or overwrite existing research
- change any existing experimental result

## Exact questions being investigated

1. What is the current repository/git state?
2. What datasets are currently present, and which are real vs. simulated?
3. Does a genuine, publicly-available "Category A" dataset (real spacecraft + real
   cyberattack labels) exist?
4. For each candidate: is the spacecraft/mission real or simulated/testbed? Are the
   anomalies operational/physical faults or actual cyberattacks? What is the license,
   access process, and academic-reuse permissibility? Is it suitable for temporal
   modeling (Phase 3-style) and/or federated/multi-client modeling (Phase 4-8-style)?
5. Which dataset (if any) is the best real-spacecraft candidate, the best
   cybersecurity-labeled candidate, and the best multi-client candidate — evaluated
   separately, since no single dataset was assumed to satisfy all three?
6. What is the recommended integration strategy, and what phase should it occupy,
   without disturbing Phases 1-8?

This file is a verbatim-intent archive of the governing instructions for the research
that follows in the sibling documents under `docs/research/`.
