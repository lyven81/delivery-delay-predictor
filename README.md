# Delivery Delay Predictor

Predicts which e-commerce orders will be delivered late **before they ship**, names the top reason for each at-risk order, and turns the numbers into verified plain-language recommendations. Built for a warehouse operations manager. Live, public, on Cloud Run.

**Live app:** https://delivery-delay-predictor-522143897885.asia-southeast1.run.app

Google Cloud Gen AI Academy APAC Edition, Cohort 2 (Top 101) — Prototype Refinement.

---

## What this refinement changed over the first prototype

The refinement makes the app **reliable enough for a real operations manager**, and every claim is verifiable on held-out data.

| Area | Prototype (v1) | Refined |
|---|---|---|
| **Model** | XGBoost trained locally, uncalibrated scores | **Tuned + calibrated on Vertex AI**; beats v1 on precision (38→49%) and recall (42→50%), Brier 0.111→0.077 |
| **Data** | ran off a fixed file | **live BigQuery** data + KPIs |
| **AI recommendations** | Gemini, unchecked | **verified by an evaluation** (LLM-as-judge): 4.28/5, 89% pass, 94% grounded |
| **UI** | single dashboard | **two tabs**: Operations + a Model Performance tab that proves the model |

The manager-facing result: the "high-risk" shortlist shrank from **296 → ~107** (v1 flagged more orders than are ever actually late), and when the model flags an order it is right more than half the time.

## Architecture

```
Order data (CSV) → Cloud Storage → BigQuery (live: store, clean, features, KPIs)
                                        │
         training features ────────────┼──────────── live features + KPIs
                                        ▼                         │
                        Vertex AI: tune + calibrate               │
                        the XGBoost + SHAP model                  ▼
                                        │            Cloud Run (FastAPI):
                                        └─ tuned model ─▶ serves model, scores orders,
                                                          hosts the two-tab dashboard
                                                                  │
                        Cloud Run ⇄ Gemini (explains + recommends) ⇄ evaluation (verifies)
                                                                  ▼
                                                        Operations Manager
```

Fully serverless: BigQuery (data) + Vertex AI (model tuning + Gemini) + Cloud Run (app), scale-to-zero.

## Files

| File | Purpose |
|---|---|
| `main.py` | FastAPI app: `/api/data` (live KPIs + alerts), `/api/model` (validated metrics + eval), `/api/ask`, `/api/score` (CSV upload) |
| `index.html` | Two-tab front end (Operations Dashboard + Model Performance), Chart.js |
| `model.pkl` | The Vertex-tuned, calibrated model (XGBoost inside a CalibratedClassifierCV) |
| `metrics.json` | Held-out validation metrics (v1 baseline vs tuned) |
| `eval_report.json` | Recommendation-quality evaluation results |
| `train_on_vertex.py` | The Vertex AI custom-training script (tune + calibrate + score) |
| `eval_recs.py` | The recommendation evaluation (LLM-as-judge) |
| `app_data.csv` | Bundled fallback data + dropdown options |
| `Dockerfile` / `requirements.txt` | Python 3.12 container (pinned so `model.pkl` loads) |

## Run locally

```bash
pip install -r requirements.txt
# reads BigQuery + Gemini when these are set; falls back to app_data.csv otherwise
export GOOGLE_CLOUD_PROJECT=gen-lang-client-0752018449
export GOOGLE_GENAI_USE_VERTEXAI=True
export GOOGLE_CLOUD_LOCATION=us-central1
uvicorn main:app --port 8080
# open http://localhost:8080
```
(Note: `model.pkl` is pinned to Python 3.12 + xgboost 3.3.0; the CSV-upload feature needs that environment. The dashboard itself reads pre-scored data and does not need the model.)

## Deploy (update the existing service)

```bash
gcloud run deploy delivery-delay-predictor --source . --region asia-southeast1 \
  --allow-unauthenticated --memory 1Gi \
  --set-env-vars GOOGLE_CLOUD_PROJECT=gen-lang-client-0752018449,GOOGLE_GENAI_USE_VERTEXAI=True,GOOGLE_CLOUD_LOCATION=us-central1
```

## Tech stack

BigQuery · Vertex AI (XGBoost tuning + Gemini 2.5 Flash) · SHAP · Cloud Run · FastAPI · Chart.js

© 2026 Lee Yih Ven
