# Delivery Delay Predictor

An AI-powered decision support application for e-commerce fulfillment teams. It predicts which orders are at risk of being delivered late **before dispatch**, explains why, and recommends actions in plain language, so operations managers can intervene before customers are affected.

Built for the Gen AI Academy APAC Cohort 2 Hackathon (Track 2).

**Live demo:** https://delivery-delay-predictor-522143897885.asia-southeast1.run.app

> The dashboard loads with live demo data on open, no upload required. The optional "Score New Orders" feature has a "Download sample CSV" link so you can test scoring against your own file.
---

## What it does

Most delivery dashboards only report what already happened (completed or late orders). This one predicts which open orders are likely to be delayed **before they ship**, so teams can act early instead of reacting to complaints.

Features:

1. **Delivery risk prediction**: a risk score (0-100) per open order, from order signals known before dispatch only (leakage-safe).
2. **Priority alert list**: the highest-risk orders, each with its top risk driver and a recommended action.
3. **Courier performance analysis**: on-time rates compared across couriers.
4. **Delivery zone analysis**: zones with consistently higher delay rates.
5. **Operational bottleneck detection**: splits delay hours across warehouse handling, dispatch, and courier transit.
6. **AI recommendations (Gemini)**: turns the metrics into three specific, actionable recommendations and answers free-text questions.
7. **Score new orders (upload CSV)** (optional): upload an orders file and get risk scores back.

## How it works

```
Orders data (cleaned.csv)
        │
        ▼
  BigQuery / CSV      ── clean, validate, feature-engineer, compute KPIs
        │
        ▼
  XGBoost + SHAP      ── per-order delivery-risk score + top risk driver
        │
        ▼
  Gemini (Vertex AI)  ── plain-language explanations + recommendations
        │
        ▼
  Cloud Run (FastAPI) ── serves the dashboard + APIs behind a public URL
```

The XGBoost model is trained **locally** and bundled inside the Cloud Run container (served via `model.pkl`). There is no always-on prediction endpoint, which keeps the deployment cost near zero. Only Gemini runs on Vertex AI.

## Model performance

Trained on 2,409 orders with a 12% late rate (class imbalance handled with `scale_pos_weight`).

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.776 |
| PR-AUC | 0.487 (~4x the 12% base rate) |
| Precision | 0.373 |
| Recall | 0.431 |
| F1 | 0.40 |

Precision and recall (not accuracy) are the honest metrics here because the target is rare. SHAP gives each prediction an explainable per-order reason, which feeds the "Key Driver" column.

## Tech stack

- **BigQuery**: data layer the deployed app queries for KPIs (CSV fallback for local runs)
- **XGBoost + SHAP**: risk scoring and per-order explanations
- **Gemini 2.5 Flash on Vertex AI**: natural-language recommendations and Q&A
- **Cloud Run**: serverless host for the dashboard and APIs
- **FastAPI**: backend framework
- **Chart.js**: dashboard charts

## Repository structure

```
├── main.py                  FastAPI backend: dashboard, KPIs, scoring, Gemini
├── index.html               Dashboard front-end (Chart.js)
├── prepare_and_train.py     Data prep + XGBoost training + SHAP (produces the artifacts below)
├── Dockerfile               Container build for Cloud Run
├── requirements.txt         Python dependencies
├── model.pkl                Trained XGBoost model + feature metadata
├── cleaned.csv              Source dataset (2,409 orders) for training
├── app_data.csv             Scored data the deployed app serves
├── sample_new_orders.csv    Sample file for the "Score New Orders" upload feature
├── scored_orders.csv        Model output: risk score + key driver per order
├── kpis.json                Precomputed operational KPIs
├── metrics.json             Model evaluation metrics
├── global_importance.json   SHAP global feature importance
└── docs/                    Prototype snapshots
```

## Run locally

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. (optional) retrain the model from source data
python prepare_and_train.py

# 3. start the app
python -m uvicorn main:app --port 8080

# 4. open the dashboard
# http://localhost:8080/
```

The app runs with no cloud credentials: it serves KPIs and risk scores from the bundled data and uses a data-driven fallback for recommendations. When deployed with Vertex AI credentials, the recommendations and Q&A switch to live Gemini automatically.

## Deploy to Cloud Run

```bash
gcloud run deploy delivery-delay-predictor \
  --source . \
  --region asia-southeast1 \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_GENAI_USE_VERTEXAI=true,GOOGLE_CLOUD_PROJECT=<your-project>,GOOGLE_CLOUD_LOCATION=global,GEMINI_MODEL=gemini-2.5-flash
```

## API reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Serves the dashboard |
| `/api/data` | GET | KPIs, priority alerts, and recommendations (filterable by courier, zone, service) |
| `/api/ask` | POST | Free-text question answered from the metrics via Gemini |
| `/api/score` | POST | Upload an orders CSV, get risk scores back |
| `/api/sample` | GET | Download a sample orders CSV |

---

Lee Yih Ven · Gen AI Academy APAC Cohort 2 · 2026
