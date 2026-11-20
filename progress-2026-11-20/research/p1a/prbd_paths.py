"""Release I/O paths; numerical code never selects assets or checkpoints.

Defaults read the two packaged run directories and the preceding public speed
snapshot. Reproduction outputs go to a separate directory, preserving archives.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT.parent / "progress-2026-11-19"
SPEED_DATA = Path(os.environ.get("PRBD_SPEED_DATA_DIR", str(PREVIOUS / "runs/prbd_tworoom_speed_20261119"))).expanduser().resolve()
STATISTICS = PREVIOUS / "runs/prbd_official_protocols_20261119/tworoom"
EVIDENCE = ROOT / "runs/prbd_evidence_trajectory_20261119"
CONSISTENCY = ROOT / "runs/prbd_consistency_20261120"
LEWM_SOURCE = Path(os.environ.get("PRBD_LEWM_SOURCE", str(ROOT / "external/le-wm"))).expanduser().resolve()
MODEL_DIR = Path(os.environ.get("PRBD_MODEL_DIR", str(ROOT / "external/lewm-tworooms"))).expanduser().resolve()
OUTPUT = Path(os.environ.get("PRBD_OUTPUT_DIR", str(ROOT / "reproduce-output"))).expanduser().resolve()


def output_path(path):
    """Keep packaged run inputs read-only; honor other explicit destinations."""
    path = Path(path).expanduser().resolve()
    try:
        relative = path.relative_to(ROOT / "runs")
    except ValueError:
        result = path
    else:
        result = OUTPUT / relative
    result.parent.mkdir(parents=True, exist_ok=True)
    return result
