---
title: Rice Blast Detector API
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Rice Blast Detector API

A FastAPI backend that classifies rice-leaf images into **Rice Blast**, **Brown Spot**,
or **Healthy**, returns confidence scores, and can produce Grad-CAM explainability
overlays. It currently runs on a **mock model** so the service deploys today; the
trained PyTorch model drops in later without changing the API surface.

## Endpoints

| Method | Path          | Description                                                        |
| ------ | ------------- | ------------------------------------------------------------------ |
| GET    | `/health`     | Liveness check. Returns a simple OK status.                        |
| GET    | `/model-info` | Model version, class labels, and confidence threshold.            |
| GET    | `/metrics`    | Illustrative evaluation metrics (macro-F1, per-class recall, etc.).|
| POST   | `/predict`    | Upload a leaf image; returns predicted class + confidence scores.  |
| POST   | `/gradcam`    | Upload a leaf image; returns a Grad-CAM explainability overlay.    |
| POST   | `/feedback`   | Submit a corrected label for an image to collect training data.    |

Interactive API docs (Swagger UI) are served at `/docs`.

## Local quickstart (Windows)

Using the system Python at
`C:\Users\KESHAV UPADHYAY\AppData\Local\Programs\Python\Python311\python.exe`:

```powershell
# From the backend/ directory.

# 1. Create and activate a virtual environment.
& "C:\Users\KESHAV UPADHYAY\AppData\Local\Programs\Python\Python311\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies (torch-free; deploys as-is).
pip install -r requirements.txt
pip install -r requirements-dev.txt   # optional: for running tests

# 3. Run the dev server with auto-reload.
uvicorn app.main:app --reload --port 8000
```

Then open <http://localhost:8000/docs> to try the endpoints.

> The active dependency set is intentionally **torch-free** so the image stays small
> and deploys immediately. `torch`, `torchvision`, and `grad-cam` are listed
> (commented out) in `requirements.txt` — uncomment them when the trained model is added.

## Deploy to Hugging Face Spaces (Docker SDK)

Hugging Face Spaces runs the container and provides `$PORT=7860`, which the
`Dockerfile` honors automatically.

1. **Create a Space**: go to <https://huggingface.co/new-space>, choose a name, and
   select **Docker** as the SDK (Blank template).
2. **Confirm settings**: the YAML front-matter at the top of this `README.md` sets
   `sdk: docker` and `app_port: 7860`, so the Space is preconfigured.
3. **Push the code**: clone the Space repo and add these backend files (including the
   `Dockerfile`, `requirements.txt`, `app/`, and `models/`):

   ```bash
   git clone https://huggingface.co/spaces/<your-username>/<space-name>
   cd <space-name>
   # Copy the backend files in, then:
   git add .
   git commit -m "Add Rice Blast Detector API"
   git push
   ```

4. **Wait for the build**: the Space builds the Docker image and starts the container.
   Once it shows **Running**, the API is live at
   `https://<your-username>-<space-name>.hf.space`.
5. **Verify**: open `/docs` on the Space URL, or curl `/health`.

### Deploy anywhere else

Any Docker host works. Build and run locally with:

```bash
docker build -t rice-blast-detector .
docker run -p 7860:7860 rice-blast-detector
# Override the port if needed:
docker run -e PORT=8080 -p 8080:8080 rice-blast-detector
```
