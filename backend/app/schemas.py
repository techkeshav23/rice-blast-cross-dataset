"""Pydantic v2 response models for the Rice Blast Detector API.

These models document and validate the JSON shapes returned by the API
endpoints. The metrics endpoint returns a free-form dict and therefore has
no dedicated model here.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response body for ``GET /health``."""

    status: str = Field(..., description="Service status, e.g. 'ok'.")
    model_loaded: bool = Field(..., description="Whether a model is loaded.")
    version: str = Field(..., description="Model/API version string.")


class ModelInfo(BaseModel):
    """Response body for ``GET /model-info``."""

    backbone: str = Field(..., description="Model backbone architecture.")
    method: str = Field(..., description="Domain generalization method used.")
    classes: List[str] = Field(..., description="Ordered class labels.")
    trained_on: str = Field(..., description="Training dataset name.")
    version: str = Field(..., description="Model/API version string.")


class PredictResponse(BaseModel):
    """Response body for ``POST /predict``."""

    label: str = Field(..., description="Predicted class label (argmax).")
    confidence: float = Field(..., description="Confidence of the top class.")
    scores: Dict[str, float] = Field(
        ..., description="Per-class probability scores."
    )
    low_confidence: bool = Field(
        ..., description="True if confidence < threshold."
    )
    version: str = Field(..., description="Model/API version string.")


class FeedbackResponse(BaseModel):
    """Response body for ``POST /feedback``."""

    id: str = Field(..., description="Unique identifier of the saved record.")
    status: str = Field(..., description="Save status, e.g. 'saved'.")


# ``MetricsResponse`` is intentionally a plain dict at the endpoint level, but
# an alias is provided for callers that want a documented type hint.
MetricsResponse = Dict[str, object]

__all__ = [
    "HealthResponse",
    "ModelInfo",
    "PredictResponse",
    "FeedbackResponse",
    "MetricsResponse",
    "Optional",
]
