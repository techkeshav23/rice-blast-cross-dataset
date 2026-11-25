"""Feedback storage utilities.

Persists user-submitted correction feedback: the uploaded image is written
as a JPEG and a corresponding metadata record is appended as one JSON line
to ``feedback.jsonl``. All IO is best-effort and errors are handled
gracefully so a storage hiccup never crashes a request.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from io import BytesIO
from typing import Optional

from PIL import Image

from app.config import settings

_JSONL_PATH = settings.FEEDBACK_DIR / "feedback.jsonl"


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def save_feedback(
    image_bytes: bytes,
    predicted_label: str,
    correct_label: str,
    notes: Optional[str],
) -> str:
    """Persist a feedback record and its associated image.

    A new ``uuid4`` hex id is generated. The image is saved to
    ``FEEDBACK_DIR/<id>.jpg`` and a JSON line describing the record is
    appended to ``FEEDBACK_DIR/feedback.jsonl``.

    Args:
        image_bytes: Raw bytes of the uploaded image.
        predicted_label: The label the model predicted.
        correct_label: The user-supplied correct label.
        notes: Optional free-text notes.

    Returns:
        The generated record id (uuid4 hex).

    Raises:
        OSError: If the metadata line cannot be written. Image-saving
            failures are swallowed so the record is still logged.
    """
    record_id = uuid.uuid4().hex

    # Ensure the destination directory exists (defensive; also done at import
    # time in config).
    try:
        settings.FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    # Save the image as JPEG. Failures here should not prevent the metadata
    # from being recorded.
    image_path = settings.FEEDBACK_DIR / f"{record_id}.jpg"
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img.save(image_path, format="JPEG", quality=90)
    except (OSError, ValueError):
        # Corrupt/unsupported image or write error: continue with metadata.
        pass

    record = {
        "id": record_id,
        "ts": _utc_now_iso(),
        "predicted": predicted_label,
        "correct": correct_label,
        "notes": notes,
    }

    with _JSONL_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record_id


def feedback_count() -> int:
    """Return the number of saved feedback records.

    Counts non-empty lines in ``feedback.jsonl``. Returns ``0`` if the file
    does not exist or cannot be read.

    Returns:
        The number of stored feedback records.
    """
    try:
        if not _JSONL_PATH.is_file():
            return 0
        with _JSONL_PATH.open("r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return 0
