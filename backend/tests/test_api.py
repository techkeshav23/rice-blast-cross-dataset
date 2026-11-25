"""API contract tests for the Rice Blast Detector FastAPI backend.

Each test exercises a single endpoint (or a single behaviour of one) and
asserts the exact response shape promised by the API contract. The app runs a
seeded MOCK model, so predictions are deterministic for a given image.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

CLASSES = ["Rice Blast", "Brown Spot", "Healthy"]


def _image_file(image_bytes: bytes) -> dict:
    """Build the multipart ``files`` payload for the ``image`` field."""
    return {"image": ("leaf.jpg", image_bytes, "image/jpeg")}


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200

    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["model_loaded"], bool)
    assert isinstance(body["version"], str)


def test_model_info(client: TestClient) -> None:
    resp = client.get("/model-info")
    assert resp.status_code == 200

    body = resp.json()
    for key in ("backbone", "method", "classes", "trained_on", "version"):
        assert key in body

    assert isinstance(body["classes"], list)
    for cls in CLASSES:
        assert cls in body["classes"]


def test_metrics(client: TestClient) -> None:
    resp = client.get("/metrics")
    assert resp.status_code == 200

    body = resp.json()
    assert "baseline" in body
    assert "mixstyle" in body
    assert "primary_metric" in body

    for group in ("baseline", "mixstyle"):
        for key in ("in_set", "unseen", "gap"):
            assert key in body[group]


def test_predict(client: TestClient, image_bytes: bytes) -> None:
    resp = client.post("/predict", files=_image_file(image_bytes))
    assert resp.status_code == 200

    body = resp.json()
    assert body["label"] in CLASSES
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["scores"].keys()) == set(CLASSES)
    assert isinstance(body["low_confidence"], bool)
    assert isinstance(body["version"], str)


def test_predict_is_deterministic(client: TestClient, image_bytes: bytes) -> None:
    first = client.post("/predict", files=_image_file(image_bytes))
    second = client.post("/predict", files=_image_file(image_bytes))

    assert first.status_code == 200
    assert second.status_code == 200

    # The mock model is seeded by image content, so identical input must yield
    # an identical label (and, in fact, identical scores).
    assert first.json()["label"] == second.json()["label"]
    assert first.json()["scores"] == second.json()["scores"]


def test_gradcam(client: TestClient, image_bytes: bytes) -> None:
    resp = client.post("/gradcam", files=_image_file(image_bytes))
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/png")
    assert len(resp.content) > 0


def test_feedback(client: TestClient, image_bytes: bytes) -> None:
    resp = client.post(
        "/feedback",
        files=_image_file(image_bytes),
        data={
            "predicted_label": "Rice Blast",
            "correct_label": "Brown Spot",
            "notes": "Edge of leaf looked more like brown spot.",
        },
    )
    assert resp.status_code == 200

    body = resp.json()
    assert isinstance(body["id"], str)
    assert body["status"] == "saved"


def test_predict_rejects_non_image(client: TestClient) -> None:
    resp = client.post(
        "/predict",
        files={"image": ("leaf.jpg", b"notanimage", "image/jpeg")},
    )
    assert resp.status_code == 400
