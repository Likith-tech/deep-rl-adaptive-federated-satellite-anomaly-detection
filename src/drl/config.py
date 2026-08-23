"""Phase 9B — configuration loading for `configs/drl.yaml`.

Validates only the invariants the approved canonical design fixes
(state dimension 44, action count 9, 8 clients) — guards against
config/design drift rather than re-implementing full schema validation.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REQUIRED_TOP_LEVEL_KEYS = (
    "reproducibility",
    "federation",
    "model",
    "state",
    "action",
    "reward",
    "dqn",
    "paths",
    "data",
)


def load_drl_config(path: Path | str) -> dict:
    """Load and validate `configs/drl.yaml` against the approved
    design's fixed invariants. Raises `ValueError` with a clear message
    if the file is missing required sections or violates an approved
    design value — never silently accepts a drifted config."""
    text = Path(path).read_text()
    config = yaml.safe_load(text)

    missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in config]
    if missing:
        raise ValueError(f"{path}: missing required top-level section(s): {missing}")

    if config["state"]["dimension"] != 44:
        raise ValueError(
            f"{path}: state.dimension must equal the approved design value 44, "
            f"got {config['state']['dimension']}"
        )
    if config["action"]["count"] != 9:
        raise ValueError(
            f"{path}: action.count must equal the approved design value 9, "
            f"got {config['action']['count']}"
        )
    if config["federation"]["num_clients"] != 8:
        raise ValueError(
            f"{path}: federation.num_clients must equal the approved design value 8, "
            f"got {config['federation']['num_clients']}"
        )

    return config
