# Rice Blast Cross-Dataset Generalization

Code for the B.Tech major project **"Improving Cross-Dataset Generalization for Rice Blast
Detection Under Real-Field Conditions"** (Department of CSE, ABES Engineering College,
Ghaziabad, 2026-27) and the accompanying paper **"Confidence-Gated Test-Time Adaptation for
Cross-Dataset Rice Blast Detection"**.

## Contents

| Folder | What it contains |
|---|---|
| [`ml/`](ml/) | Research pipeline: leakage-controlled splits, training (MobileNetV3-Small, EfficientNet-B0), MixStyle and RandConv, AdaBN, Tent / EATA / SAR, the proposed Confidence-Gated AdaBN, the perceptual-hash leakage audit, Grad-CAM grounding and statistics. [`ml/outputs/`](ml/outputs/) holds the result JSON files and figures; [`ml/data/splits/`](ml/data/splits/) holds the exact split manifests. |
| [`backend/`](backend/) | FastAPI inference service that serves the trained model with ONNX Runtime (CPU). |
| [`rice_blast_detector/`](rice_blast_detector/) | Flutter mobile application (English and Hindi). |

## Main results

Two backbones x four source-to-target dataset pairings = eight cells, 5-8 seeds each.

- Internal Macro-F1 of 97-99% falls to 26-70% on unseen target datasets (gap 27.7-72.4 points).
- MixStyle and RandConv: none of the ten comparisons with the baseline survives Holm correction.
- AdaBN gives significant gains in four cells (up to +21.4 points) but -5.15 on the best-calibrated cell.
- Confidence-Gated AdaBN has the best mean gain (+6.01) and the safest worst cell (-0.23); for
  comparison, Tent +2.40 / -10.16, EATA +3.52 / -9.50, SAR +2.55 / -7.95.

## Run the backend locally

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

Interactive API documentation: <http://localhost:8000/docs>.
Tests: `pip install -r requirements-dev.txt` and then `pytest`.

## Deploy the backend on Render (free)

1. In Render, choose **New > Web Service** and connect this repository.
2. Set **Root Directory** to `backend`. Render detects the `Dockerfile`, so the runtime is **Docker**.
3. Choose the **Free** instance type and set the health check path to `/health`.
4. After the build, `https://<your-service>.onrender.com/health` returns `"status": "ok"`.

Free services sleep after 15 minutes without traffic, so the first request after that takes
about a minute.

## Reproducing the experiments

The datasets are public and are not included in this repository:

- Paddy Doctor (Kaggle competition `paddy-disease-classification`)
- Philippines Rice Diseases (`shrupyag001/philippines-rice-diseases`)
- Rice Disease Dataset (`anshulm257/rice-disease-dataset`)
- Rice Leaf Images (`nizorogbezuode/rice-leaf-images`)

They can be downloaded with `ml/fetch_real_data.py` and `ml/fetch_one.py`, which need a Kaggle API
token in `ml/kaggle.json` (this file is git-ignored and must never be committed). Images go
into `ml/data/raw/<domain>/<class>/`. The source and target domains are selected with the
`RICE_SOURCE_DOMAIN` and `RICE_TARGET_DOMAIN` environment variables.

The 32 Philippines images that duplicate Paddy Doctor images are listed in
`ml/outputs/canopy_leak_exclude.txt`. Point `RICE_EXTERNAL_EXCLUDE` to this file to evaluate on
the de-duplicated canopy target.

## Current status

- The mobile application still uses a mock predictor, and its results are labelled as
  simulated. Connecting it to the backend is in progress.
- The `/gradcam` endpoint returns an illustrative overlay. Real Grad-CAM maps are computed
  offline in `ml/gradcam.py`.

## Team

Harsh Kumar Sah and Keshav Upadhyay. Guide: Mr. Prashant Kumar Singh, Department of Computer
Science and Engineering, ABES Engineering College, Ghaziabad.
