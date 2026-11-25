"""Shared pytest fixtures for the Rice Blast Detector API tests.

Provides an in-memory JPEG image (as raw bytes) and a FastAPI ``TestClient``
so individual tests stay focused on asserting the API contract.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture
def image_bytes() -> bytes:
    """Return the bytes of a small, valid 224x224 RGB JPEG.

    The image is built entirely in memory with Pillow so the tests do not
    depend on any files on disk. A deterministic solid-colour image keeps the
    seeded mock model's output stable across calls.
    """
    img = Image.new("RGB", (224, 224), color=(34, 139, 34))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def client() -> TestClient:
    """Return a ``TestClient`` wrapping the FastAPI application."""
    return TestClient(app)
