"""Phase 16 — reads this project's real, already-committed experiment
artifacts and shapes them into the API's response schemas.

This module trains nothing and re-derives no metric: every number
below is either read directly from a committed JSON artifact under
results/ or experiments/, or copied from the same fixed reference table
Phase 14's own report already used
(scripts/generate_full_evaluation_report.py's `METHODS`) — duplicated
here rather than imported, matching this project's own established
convention of keeping each script/service self-contained (see e.g.
scripts/run_ablation_studies.py's `build_fairness_entropy`) since the
underlying source of truth in both cases is the same committed report,
not either script.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from backend.app.core.paths import EXPERIMENTS_DIR, RESULTS_DIR
from backend.app.schemas.drl import DRLControllerSummary, DRLDecision
from backend.app.schemas.experiments import ExperimentDetail, ExperimentSummary, RoundMetric
from backend.app.schemas.federated import AdaptiveRuleConfig, FederatedOverview


@dataclass(frozen=True)
class _RegistryEntry:
    id: str
    name: str
    phase: str
    family: str
    test_f1: float
    test_roc_auc: float | None
    fairness_gap: float | None
    source_report: str
    round_history_path: Path | None = None
    round_history_kind: str = "generic"  # "generic" (Phase 6/8/9 round_history.json) or "drl" (Phase 9B final_run.json)
    rule_weights: dict[str, float] | None = None


# Every (test_f1, test_roc_auc, fairness_gap) triple below is copied
# verbatim from the named report, exactly as
# scripts/generate_full_evaluation_report.py's METHODS list already
# does — see that script for the full source-by-source citation.
_REGISTRY: tuple[_RegistryEntry, ...] = (
    _RegistryEntry(
        id="phase2_centralized_baseline", name="Phase 2 — Centralized MLP baseline",
        phase="Phase 2", family="non-federated", test_f1=0.7790, test_roc_auc=0.8972,
        fairness_gap=None, source_report="results/reports/baseline_results.md",
    ),
    _RegistryEntry(
        id="phase3_temporal", name="Phase 3 — Temporal GRU+Attention (centralized)",
        phase="Phase 3", family="non-federated", test_f1=0.7558, test_roc_auc=0.8923,
        fairness_gap=None, source_report="results/reports/temporal_results.md",
    ),
    _RegistryEntry(
        id="phase6_fedavg", name="Phase 6 — FedAvg (sample-count weighting)",
        phase="Phase 6", family="federated", test_f1=0.7404, test_roc_auc=0.8479,
        fairness_gap=None, source_report="results/reports/federated_results.md",
        round_history_path=RESULTS_DIR / "models" / "federated" / "round_history.json",
    ),
    _RegistryEntry(
        id="phase7_alpha_0.1", name="Phase 7 — Non-IID sweep, alpha=0.1",
        phase="Phase 7", family="federated (non-IID study)", test_f1=0.7409, test_roc_auc=0.8172,
        fairness_gap=None, source_report="results/reports/non_iid_results.md",
    ),
    _RegistryEntry(
        id="phase7_alpha_5.0", name="Phase 7 — Non-IID sweep, alpha=5.0",
        phase="Phase 7", family="federated (non-IID study)", test_f1=0.7640, test_roc_auc=0.8641,
        fairness_gap=None, source_report="results/reports/non_iid_results.md",
    ),
    _RegistryEntry(
        id="phase7_alpha_10.0", name="Phase 7 — Non-IID sweep, alpha=10.0",
        phase="Phase 7", family="federated (non-IID study)", test_f1=0.7541, test_roc_auc=0.8637,
        fairness_gap=None, source_report="results/reports/non_iid_results.md",
    ),
    _RegistryEntry(
        id="phase8_performance_only", name="Phase 8 — Adaptive FL, performance_only",
        phase="Phase 8", family="federated (rule-based adaptive)", test_f1=0.7447, test_roc_auc=0.8497,
        fairness_gap=0.1245, source_report="results/reports/adaptive_fl_results.md",
        round_history_path=EXPERIMENTS_DIR / "adaptive_fl" / "performance_only" / "round_history.json",
        rule_weights={"performance": 1.0, "data": 0.0, "resource": 0.0, "fairness": 0.0},
    ),
    _RegistryEntry(
        id="phase8_resource_only", name="Phase 8 — Adaptive FL, resource_only",
        phase="Phase 8", family="federated (rule-based adaptive)", test_f1=0.7547, test_roc_auc=0.8438,
        fairness_gap=0.2259, source_report="results/reports/adaptive_fl_results.md",
        round_history_path=EXPERIMENTS_DIR / "adaptive_fl" / "resource_only" / "round_history.json",
        rule_weights={"performance": 0.0, "data": 0.0, "resource": 1.0, "fairness": 0.0},
    ),
    _RegistryEntry(
        id="phase8_data_only", name="Phase 8 — Adaptive FL, data_only",
        phase="Phase 8", family="federated (rule-based adaptive)", test_f1=0.7397, test_roc_auc=0.8313,
        fairness_gap=0.2113, source_report="results/reports/adaptive_fl_results.md",
        round_history_path=EXPERIMENTS_DIR / "adaptive_fl" / "data_only" / "round_history.json",
        rule_weights={"performance": 0.0, "data": 1.0, "resource": 0.0, "fairness": 0.0},
    ),
    _RegistryEntry(
        id="phase8_combined", name="Phase 8 — Adaptive FL, combined",
        phase="Phase 8", family="federated (rule-based adaptive)", test_f1=0.7439, test_roc_auc=0.8536,
        fairness_gap=0.1496, source_report="results/reports/adaptive_fl_results.md",
        round_history_path=EXPERIMENTS_DIR / "adaptive_fl" / "combined" / "round_history.json",
        rule_weights={"performance": 0.40, "data": 0.25, "resource": 0.15, "fairness": 0.20},
    ),
    _RegistryEntry(
        id="phase9b_dqn", name="Phase 9B — DQN adaptive FL",
        phase="Phase 9B", family="federated (DRL adaptive)", test_f1=0.7425, test_roc_auc=0.8361,
        fairness_gap=0.1833, source_report="results/reports/drl_adaptive_fl_results.md",
        round_history_path=EXPERIMENTS_DIR / "drl_adaptive_fl" / "full_run" / "final_run.json",
        round_history_kind="drl",
    ),
    _RegistryEntry(
        id="phase9_no_correction", name="Phase 9 — Staleness-aware FL, no_correction",
        phase="Phase 9", family="federated (staleness-aware)", test_f1=0.7371, test_roc_auc=0.8515,
        fairness_gap=0.1964, source_report="results/reports/staleness_fl_results.md",
        round_history_path=EXPERIMENTS_DIR / "staleness_fl" / "no_correction" / "round_history.json",
    ),
    _RegistryEntry(
        id="phase9_linear_decay", name="Phase 9 — Staleness-aware FL, linear_decay",
        phase="Phase 9", family="federated (staleness-aware)", test_f1=0.7354, test_roc_auc=0.8556,
        fairness_gap=0.1964, source_report="results/reports/staleness_fl_results.md",
        round_history_path=EXPERIMENTS_DIR / "staleness_fl" / "linear_decay" / "round_history.json",
    ),
    _RegistryEntry(
        id="phase9_exponential_decay", name="Phase 9 — Staleness-aware FL, exponential_decay",
        phase="Phase 9", family="federated (staleness-aware)", test_f1=0.7360, test_roc_auc=0.8549,
        fairness_gap=0.1994, source_report="results/reports/staleness_fl_results.md",
        round_history_path=EXPERIMENTS_DIR / "staleness_fl" / "exponential_decay" / "round_history.json",
    ),
)

_BY_ID: dict[str, _RegistryEntry] = {e.id: e for e in _REGISTRY}


def _to_summary(entry: _RegistryEntry) -> ExperimentSummary:
    return ExperimentSummary(
        id=entry.id, name=entry.name, phase=entry.phase, family=entry.family,
        test_f1=entry.test_f1, test_roc_auc=entry.test_roc_auc, fairness_gap=entry.fairness_gap,
        has_round_history=entry.round_history_path is not None, source_report=entry.source_report,
    )


def _load_generic_round_history(path: Path) -> list[RoundMetric]:
    """Phase 6 (RoundRecord) / Phase 8 (AdaptiveRoundRecord) / Phase 9
    (StalenessRoundRecord) all serialize to `{"round_history": [...]}`
    with a `val_metrics.f1` field; Phase 8/9 additionally carry
    `aggregation_weights`, and Phase 9 additionally carries
    `client_staleness`. Fields absent for a given phase are simply
    absent from that phase's JSON, and RoundMetric's optional fields
    default to None."""
    data = json.loads(path.read_text())
    rounds = []
    for r in data["round_history"]:
        rounds.append(
            RoundMetric(
                round_number=r["round_number"],
                val_loss=r.get("val_loss"),
                val_f1=r["val_metrics"]["f1"],
                val_roc_auc=r["val_metrics"].get("roc_auc"),
                aggregation_weights=r.get("aggregation_weights"),
                client_staleness=r.get("client_staleness"),
            )
        )
    return rounds


def _load_drl_round_history(path: Path) -> list[RoundMetric]:
    """Phase 9B's final_run.json uses `{"rounds": [...]}` with
    `action_name`/`weights` instead of `aggregation_weights` — a
    genuinely different shape (the DQN's chosen profile per round), not
    an inconsistency to hide."""
    data = json.loads(path.read_text())
    rounds = []
    for r in data["rounds"]:
        rounds.append(
            RoundMetric(
                round_number=r["round"],
                val_loss=r.get("val_loss"),
                val_f1=r["val_f1"],
                val_roc_auc=r.get("val_metrics", {}).get("roc_auc"),
                aggregation_weights=r.get("weights"),
                action_name=r.get("action_name"),
            )
        )
    return rounds


@lru_cache(maxsize=32)
def _load_round_history(entry_id: str) -> tuple[RoundMetric, ...]:
    entry = _BY_ID[entry_id]
    if entry.round_history_path is None or not entry.round_history_path.exists():
        return ()
    loader = _load_drl_round_history if entry.round_history_kind == "drl" else _load_generic_round_history
    return tuple(loader(entry.round_history_path))


def list_experiments() -> list[ExperimentSummary]:
    return [_to_summary(e) for e in _REGISTRY]


def get_experiment(experiment_id: str) -> ExperimentDetail | None:
    entry = _BY_ID.get(experiment_id)
    if entry is None:
        return None
    summary = _to_summary(entry)
    return ExperimentDetail(**summary.model_dump(), rounds=list(_load_round_history(entry.id)))


# The "aggregation mechanism only" family from Phase 14's report — every
# entry here shares the exact same Phase 4/6 partition/model/rounds and
# varies only how client updates are combined, the fair like-for-like
# comparison (as opposed to `list_experiments()`, which also includes
# Phase 7's data-partition sweep and the two non-federated references).
_MECHANISM_FAMILIES = (
    "federated", "federated (rule-based adaptive)",
    "federated (DRL adaptive)", "federated (staleness-aware)",
)


def get_federated_overview() -> FederatedOverview:
    mechanisms = [e for e in _REGISTRY if e.family in _MECHANISM_FAMILIES]
    best = max(mechanisms, key=lambda e: e.test_f1)
    fairest = min((e for e in mechanisms if e.fairness_gap is not None), key=lambda e: e.fairness_gap)
    return FederatedOverview(
        total_methods_compared=len(_REGISTRY),
        best_mechanism=_to_summary(best),
        fairest_mechanism=_to_summary(fairest),
        mechanisms=[_to_summary(e) for e in mechanisms],
    )


def list_adaptive_rule_configs() -> list[AdaptiveRuleConfig]:
    return [
        AdaptiveRuleConfig(experiment_id=e.id, name=e.name, weights=e.rule_weights,
                            test_f1=e.test_f1, fairness_gap=e.fairness_gap)
        for e in _REGISTRY if e.rule_weights is not None
    ]


# Phase 9B DQN controller metadata — the action catalogue is read
# directly from the real, committed action_distribution rather than a
# hardcoded list, so it can never drift out of sync with what the run
# actually observed. State dimensionality (44) is a fixed architectural
# fact of Phase 9B's design, documented in
# docs/project-progress/11-phase-9b-drl-adaptive-federated-learning.md
# section 6 — not derived from a file here.
_DRL_STATE_DIMENSIONS = 44


def get_drl_controller_summary() -> DRLControllerSummary | None:
    analysis_path = EXPERIMENTS_DIR / "drl_adaptive_fl" / "full_run" / "analysis_summary.json"
    final_run_path = EXPERIMENTS_DIR / "drl_adaptive_fl" / "full_run" / "final_run.json"
    if not analysis_path.exists() or not final_run_path.exists():
        return None
    analysis = json.loads(analysis_path.read_text())
    final_run = json.loads(final_run_path.read_text())
    dqn_entry = _BY_ID["phase9b_dqn"]
    best_rule_based = max(
        (e for e in _REGISTRY if e.family == "federated (rule-based adaptive)"), key=lambda e: e.test_f1
    )
    return DRLControllerSummary(
        action_catalogue=sorted(analysis["action_distribution_all"].keys()),
        state_dimensions=_DRL_STATE_DIMENSIONS,
        episodes_trained=analysis["num_episodes"],
        rounds_per_episode=analysis["num_rounds_total"] // analysis["num_episodes"],
        total_env_steps=analysis["num_rounds_total"],
        final_run_best_round=final_run["best_round"],
        final_run_best_val_loss=final_run["best_val_loss"],
        test_f1=dqn_entry.test_f1,
        test_roc_auc=dqn_entry.test_roc_auc,
        beat_best_rule_based_baseline=dqn_entry.test_f1 > best_rule_based.test_f1,
    )


def get_drl_decisions() -> list[DRLDecision]:
    rounds = _load_round_history("phase9b_dqn")
    return [
        DRLDecision(
            round_number=r.round_number, action_name=r.action_name or "unknown",
            weights=r.aggregation_weights or {}, val_f1=r.val_f1, val_loss=r.val_loss or 0.0,
        )
        for r in rounds
    ]
