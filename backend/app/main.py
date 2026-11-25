"""FastAPI application for the Rice Blast Detector backend.

Wires together configuration, the (mock) model, Grad-CAM generation, and
feedback storage into the public HTTP API. Launch with::

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import json
from io import BytesIO
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from app.config import settings
from app.gradcam import generate_gradcam
from app.inference import BaseModel, get_model
from app.schemas import (
    FeedbackResponse,
    HealthResponse,
    ModelInfo,
    PredictResponse,
)
from app.storage import feedback_count, save_feedback

# Module-global model, populated at startup.
model: Optional[BaseModel] = None

app = FastAPI(title=settings.APP_NAME, version=settings.VERSION)

# Permissive CORS so the hybrid (mobile/web) frontend can call the API from
# any origin during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _load_model_on_startup() -> None:
    """Load the model singleton into the module global at startup."""
    global model
    model = get_model()


def _get_model() -> BaseModel:
    """Return the loaded model, loading it on demand if necessary."""
    global model
    if model is None:
        model = get_model()
    return model


async def _read_image(image: UploadFile) -> "Image.Image":
    """Read and decode an uploaded file into an RGB PIL image.

    Args:
        image: The uploaded multipart file.

    Returns:
        The decoded RGB :class:`PIL.Image.Image`.

    Raises:
        HTTPException: 400 if the upload is not a decodable image.
    """
    content_type = (image.content_type or "").lower()
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="Uploaded file is not an image."
        )

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")

    try:
        return Image.open(BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400, detail="Could not decode image."
        )


@app.get("/")
def root() -> dict:
    """Return basic service metadata."""
    return {
        "name": settings.APP_NAME,
        "version": settings.VERSION,
        "model": _get_model().version,
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service health and whether a model is loaded."""
    current = _get_model()
    return HealthResponse(
        status="ok",
        model_loaded=bool(getattr(current, "loaded", False)),
        version=current.version,
    )


@app.get("/model-info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    """Describe the model that is actually loaded (ONNX model or mock)."""
    current = _get_model()
    return ModelInfo(
        backbone=current.backbone,
        method=current.method,
        classes=list(settings.CLASSES),
        trained_on=current.trained_on,
        version=current.version,
    )


@app.get("/metrics")
def metrics() -> dict:
    """Return evaluation metrics from ``models/metrics.json``.

    If the file is missing or unreadable, a sensible illustrative default is
    returned instead.
    """
    default = {
        "primary_metric": "macro_f1",
        "baseline": {"in_set": 96, "unseen": 61, "gap": 35},
        "mixstyle": {"in_set": 95, "unseen": 74, "gap": 21},
        "note": "illustrative",
    }
    try:
        if settings.METRICS_PATH.is_file():
            with settings.METRICS_PATH.open("r", encoding="utf-8") as fh:
                return json.load(fh)
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return default


@app.post("/predict", response_model=PredictResponse)
async def predict(image: UploadFile = File(...)) -> PredictResponse:
    """Classify an uploaded rice leaf image."""
    pil_image = await _read_image(image)
    current = _get_model()
    try:
        result = current.predict(pil_image)
    except NotImplementedError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500, detail=f"Inference failed: {exc}"
        )

    return PredictResponse(
        label=result["label"],
        confidence=result["confidence"],
        scores=result["scores"],
        low_confidence=result["low_confidence"],
        version=current.version,
    )


@app.post("/gradcam")
async def gradcam(image: UploadFile = File(...)) -> Response:
    """Return a Grad-CAM style heatmap PNG for an uploaded image."""
    pil_image = await _read_image(image)
    current = _get_model()
    try:
        png_bytes = generate_gradcam(pil_image, current)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Grad-CAM generation failed: {exc}"
        )
    return Response(content=png_bytes, media_type="image/png")


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(
    image: UploadFile = File(...),
    predicted_label: str = Form(...),
    correct_label: str = Form(...),
    notes: Optional[str] = Form(None),
) -> FeedbackResponse:
    """Persist user correction feedback and the associated image."""
    # Validate the upload is an image, then read its raw bytes for storage.
    content_type = (image.content_type or "").lower()
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="Uploaded file is not an image."
        )

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")

    try:
        record_id = save_feedback(raw, predicted_label, correct_label, notes)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Could not save feedback: {exc}"
        )

    return FeedbackResponse(id=record_id, status="saved")


@app.get("/feedback-count")
def feedback_count_endpoint() -> dict:
    """Return the number of stored feedback records (utility endpoint)."""
    return {"count": feedback_count()}
