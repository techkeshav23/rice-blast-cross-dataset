# Rice Blast Detector — API Examples

Base URL (local dev): `http://localhost:8000`

Start the server from the `backend/` directory:

```bash
uvicorn app.main:app --reload
```

All `curl` examples below assume you have a leaf photo at `leaf.jpg` in the
current directory.

---

## GET /health

Liveness/readiness probe.

```bash
curl http://localhost:8000/health
```

Example response:

```json
{ "status": "ok", "model_loaded": true, "version": "v0.1-mock" }
```

---

## GET /model-info

Metadata about the served model.

```bash
curl http://localhost:8000/model-info
```

Example response:

```json
{
  "backbone": "resnet18",
  "method": "MixStyle",
  "classes": ["Rice Blast", "Brown Spot", "Healthy"],
  "trained_on": "PlantVillage (rice subset)",
  "version": "v0.1-mock"
}
```

---

## GET /metrics

Reports in-set vs. unseen performance for the baseline and MixStyle models.

```bash
curl http://localhost:8000/metrics
```

Example response:

```json
{
  "baseline": { "in_set": 0.98, "unseen": 0.71, "gap": 0.27 },
  "mixstyle": { "in_set": 0.97, "unseen": 0.83, "gap": 0.14 },
  "primary_metric": "unseen_accuracy"
}
```

---

## POST /predict

Classify a leaf image. Send the file in the multipart field named `image`.

```bash
curl -X POST http://localhost:8000/predict \
  -F "image=@leaf.jpg;type=image/jpeg"
```

Example response:

```json
{
  "label": "Rice Blast",
  "confidence": 0.87,
  "scores": { "Rice Blast": 0.87, "Brown Spot": 0.09, "Healthy": 0.04 },
  "low_confidence": false,
  "version": "v0.1-mock"
}
```

> The mock model is seeded by the image content, so the same image always
> returns the same prediction.

---

## POST /gradcam

Returns a Grad-CAM overlay as a PNG image. Save the raw response body to a file
with `-o` / `--output`.

```bash
curl -X POST http://localhost:8000/gradcam \
  -F "image=@leaf.jpg;type=image/jpeg" \
  -o gradcam.png
```

The response `Content-Type` is `image/png` and `gradcam.png` will contain the
heatmap overlay.

---

## POST /feedback

Submit a correction. Sends the image file plus form fields. `notes` is
optional.

```bash
curl -X POST http://localhost:8000/feedback \
  -F "image=@leaf.jpg;type=image/jpeg" \
  -F "predicted_label=Rice Blast" \
  -F "correct_label=Brown Spot" \
  -F "notes=Edge of leaf looked more like brown spot."
```

Example response:

```json
{ "id": "a1b2c3d4", "status": "saved" }
```

---

## Run the tests

From the `backend/` directory:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pytest -q
```

The test suite spins up the app in-process with FastAPI's `TestClient` and
builds its test image in memory with Pillow, so no running server or external
files are required.
