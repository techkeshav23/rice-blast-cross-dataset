"""Application configuration.

Exposes a single ``settings`` object holding resolved absolute paths,
class labels, and tunable constants. Paths are resolved relative to the
``backend/`` directory so the app behaves identically regardless of the
current working directory it is launched from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
# This file lives at ``backend/app/config.py`` so ``BASE_DIR`` (the ``backend``
# directory) is two parents up.
BASE_DIR: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = BASE_DIR / "models"
DATA_DIR: Path = BASE_DIR / "data"

MODEL_PATH: Path = MODELS_DIR / "model.pt"
METRICS_PATH: Path = MODELS_DIR / "metrics.json"
LABELS_PATH: Path = MODELS_DIR / "labels.txt"
FEEDBACK_DIR: Path = DATA_DIR / "feedback"

# The canonical class order. This order is part of the API contract and must
# not be reordered.
DEFAULT_CLASSES: List[str] = ["Rice Blast", "Brown Spot", "Healthy"]


def _load_classes() -> List[str]:
    """Load class labels from ``models/labels.txt`` if present.

    The file should contain one class label per line. Blank lines are
    ignored. If the file is missing, unreadable, or empty, the default
    three-class list is returned.

    Returns:
        The list of class labels in their canonical order.
    """
    try:
        if LABELS_PATH.is_file():
            lines = [
                line.strip()
                for line in LABELS_PATH.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if lines:
                return lines
    except OSError:
        # Fall through to defaults on any IO error.
        pass
    return list(DEFAULT_CLASSES)


@dataclass(frozen=True)
class Settings:
    """Immutable application settings container."""

    APP_NAME: str = "Rice Blast Detector"
    VERSION: str = "v0.1-mock"
    CLASSES: List[str] = field(default_factory=_load_classes)
    LOW_CONF_THRESHOLD: float = 0.55
    IMAGE_SIZE: int = 224

    # Resolved absolute paths (mirrors of the module-level constants so they
    # are conveniently accessible via the ``settings`` object).
    BASE_DIR: Path = BASE_DIR
    MODELS_DIR: Path = MODELS_DIR
    DATA_DIR: Path = DATA_DIR
    MODEL_PATH: Path = MODEL_PATH
    METRICS_PATH: Path = METRICS_PATH
    LABELS_PATH: Path = LABELS_PATH
    FEEDBACK_DIR: Path = FEEDBACK_DIR


settings = Settings()

# Ensure the feedback directory exists so storage writes never fail on a
# missing parent directory.
try:
    settings.FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    # Directory creation failures are handled gracefully at write time.
    pass
