"""Filesystem paths to the repository's committed result artifacts.

Phase 16 (OrbitShield integration) reads its data straight from the
same committed JSON files every prior phase's own report and this
project's ML pipeline already produce (results/, experiments/) — the
backend never trains anything and never re-derives a metric; it only
reads and shapes what's already on disk.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
