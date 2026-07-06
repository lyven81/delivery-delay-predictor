"""
Delivery Delay Predictor — Cloud Run backend (FastAPI).
Serves the dashboard, exposes filterable KPIs + risk alerts from the locally-trained
model, and uses Gemini (on Vertex AI) for plain-language recommendations, with an
offline data-driven fallback so it runs with no credentials.
"""
import os, json, pickle, io
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

app = FastAPI(title="Delivery Delay Predictor")

# ---------- load model + filterable data ----------
with open("model.pkl", "rb") as f:
    BUNDLE = pickle.load(f)
MODEL, FEATURES, CAT, NUM = BUNDLE["model"], BUNDLE["features"], BUNDLE["cat"], BUNDLE["num"]

DATA = pd.read_csv("app_data.csv")
DATA["OnTime"] = DATA["OnTime"].astype(str).str.strip().str.lower().isin(["true", "1"]).astype(int)

# ---------- KPIs (computed on any subset) ----------
def compute_kpis(df):
    if len(df) == 0:
        return {"on_time_rate": 0, "late_orders": 0, "avg_delay_hours": 0, "refund_cost_rm": 0,
                "courier_on_time": {}, "zone_delay_rate": {},
                "bottleneck_pct": {"Courier Transit": 0, "Warehouse Handling": 0, "Dispatch": 0}}
    late = df[df.OnTime == 0]
    cd = late.CourierDelayHours.clip(lower=0).sum()
    idl = late.InternalDelayHours.clip(lower=0).sum()
    tot = cd + idl
    ct = round(100 * cd / tot) if tot > 0 else 0
    wh = round((100 - ct) * 0.67)
    return {
        "on_time_rate": round(100 * df.OnTime.mean()),
        "late_orders": int((df.OnTime == 0).sum()),
        "avg_delay_hours": round(late.LatenessHours.mean(), 1) if len(late) else 0,
        "refund_cost_rm": int(df.RefundAmount.sum()),
        "courier_on_time": {k: round(100 * v) for k, v in df.groupby("Courier").OnTime.mean().sort_values(ascending=False).items()},
        "zone_delay_rate": {k: round(100 * (1 - v)) for k, v in df.groupby("DeliveryZone").OnTime.mean().sort_values().items()},
        "bottleneck_pct": {"Courier Transit": ct, "Warehouse Handling": wh, "Dispatch": 100 - ct - wh},
    }

FULL_KPIS = compute_kpis(DATA)

def filter_df(courier, zone, service):
    df = DATA
    if courier and not courier.startswith("All"):
        df = df[df.Courier == courier]
    if zone and not zone.startswith("All"):
        df = df[df.DeliveryZone == zone]
    if service and not service.startswith("All"):
        df = df[df.DeliveryService == service]
    return df

# ---------- BigQuery (live data layer; local CSV fallback) ----------
BQ_TABLE = os.getenv("BQ_TABLE", "delivery_delay.orders")
_bq = None
def bq_client():
    global _bq
    if _bq is None:
        if not os.getenv("GOOGLE_CLOUD_PROJECT"):
            _bq = False
        else:
            try:
                from google.cloud import bigquery
                _bq = bigquery.Client()
            except Exception:
                _bq = False
    return _bq or None

def _coerce(df):
    if "OnTime" in df.columns:
        df["OnTime"] = df["OnTime"].astype(str).str.strip().str.lower().isin(["true", "1"]).astype(int)
    return df

def get_data(courier, zone, service):
    """Returns (dataframe, source). Queries BigQuery live; falls back to local CSV."""
    client = bq_client()
    if client:
        try:
            from google.cloud import bigquery
            conds, params = [], []
            if courier and not courier.startswith("All"):
                conds.append("Courier=@c"); params.append(bigquery.ScalarQueryParameter("c", "STRING", courier))
            if zone and not zone.startswith("All"):
                conds.append("DeliveryZone=@z"); params.append(bigquery.ScalarQueryParameter("z", "STRING", zone))
            if service and not service.startswith("All"):
                conds.append("DeliveryService=@s"); params.append(bigquery.ScalarQueryParameter("s", "STRING", service))
            where = (" WHERE " + " AND ".join(conds)) if conds else ""
            sql = f"SELECT * FROM `{BQ_TABLE}`{where}"
            cfg = bigquery.QueryJobConfig(query_parameters=params)
            rows = [dict(r) for r in client.query(sql, job_config=cfg).result()]
            return _coerce(pd.DataFrame(rows)), "bigquery"
        except Exception:
            pass
    return _coerce(filter_df(courier, zone, service).copy()), "local"

# ---------- feature engineering + scoring (for CSV upload) ----------
def build_features(df):
    df = df.copy()
    df["OrderTimestamp"] = pd.to_datetime(df["OrderTimestamp"])
    df["RequestedDeliveryDateTime"] = pd.to_datetime(df["RequestedDeliveryDateTime"])
    df["order_weekday"] = df["OrderTimestamp"].dt.dayofweek
    df["order_hour"] = df["OrderTimestamp"].dt.hour
    df["requested_slack_hours"] = (df["RequestedDeliveryDateTime"] - df["OrderTimestamp"]).dt.total_seconds()/3600
    df["HardDateOccasion_int"] = df.get("HardDateOccasion", False).astype(int)
    X = pd.concat([pd.get_dummies(df[CAT], prefix=CAT), df[NUM]], axis=1)
    return X.reindex(columns=FEATURES, fill_value=0)

def score(df):
    X = build_features(df)
    df = df.copy()
    df["risk"] = (MODEL.predict_proba(X)[:, 1] * 100).round().astype(int)
    return df.sort_values("risk", ascending=False)

# ---------- Gemini (with offline fallback) ----------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

def _has_creds():
    return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_CLOUD_PROJECT"))

def _gemini(prompt):
    if not _has_creds():
        return None
    try:
        from google import genai
        client = genai.Client()
        r = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return r.text.strip()
    except Exception:
        return None

def recommendations(kpis):
    if not kpis["courier_on_time"]:
        return ["No orders match the current filter."]
    prompt = ("You are an operations analyst for an e-commerce delivery team. Using ONLY these metrics, "
              "give exactly 3 short, specific, actionable recommendations to reduce late deliveries. "
              "One sentence each, no preamble.\n" + json.dumps(kpis))
    txt = _gemini(prompt)
    if txt:
        return [l.strip("-• ").strip() for l in txt.splitlines() if l.strip()][:3]
    bn = kpis["bottleneck_pct"]
    worst_zone = list(kpis["zone_delay_rate"].items())[0]
    worst_courier = list(kpis["courier_on_time"].items())[-1]
    return [
        f"Courier transit drives {bn['Courier Transit']}% of all delay hours; prioritise courier SLAs over internal fixes.",
        f"{worst_zone[0]} zone runs {worst_zone[1]}% late; add buffer time for these orders.",
        f"{worst_courier[0]} has the lowest on-time rate ({worst_courier[1]}%); shift Same-Day volume to a stronger courier.",
    ]

def answer(question):
    prompt = ("You are an operations analyst. Answer in 2-3 sentences using ONLY these metrics. Be specific.\n"
              "METRICS: " + json.dumps(FULL_KPIS) + "\nQUESTION: " + question)
    txt = _gemini(prompt)
    if txt:
        return txt
    q = question.lower()
    for c, v in FULL_KPIS["courier_on_time"].items():
        if c.lower() in q:
            return f"{c} has an on-time rate of {v}%. " + ("It is your weakest courier; consider reallocating time-critical orders." if v <= 84 else "It performs around or above average.")
    for z, v in FULL_KPIS["zone_delay_rate"].items():
        if z.lower() in q:
            return f"{z} has a delay rate of {v}%. " + ("This is an outstation hotspot; add buffer time." if v >= 16 else "This is broadly in line with the network.")
    return (f"Across all orders the on-time rate is {FULL_KPIS['on_time_rate']}% with RM {FULL_KPIS['refund_cost_rm']:,} in refunds. "
            f"The biggest lever is courier transit ({FULL_KPIS['bottleneck_pct']['Courier Transit']}% of delay hours).")

# ---------- API ----------
def action_for(risk, driver):
    """Recommended intervention, before dispatch, based on predicted risk + top driver."""
    urgency = "Escalate now" if risk >= 85 else "Expedite" if risk >= 60 else "Monitor"
    d = (driver or "").lower()
    if "volume" in d:
        tip = "prioritise packing"
    elif "window" in d or "slack" in d:
        tip = "expedite dispatch"
    elif "zone" in d:
        tip = "add buffer, alert customer"
    elif any(c.lower() in d for c in ["ninja", "pos laju", "j&t", "grab", "lalamove", "rider"]):
        tip = "assign a stronger courier"
    else:
        tip = "review order"
    return f"{urgency}: {tip}"

_cache = {}

@app.get("/api/data")
def data(courier: str = "", zone: str = "", service: str = ""):
    key = (courier, zone, service)
    if key in _cache:
        return JSONResponse(_cache[key])
    df, source = get_data(courier, zone, service)
    kpis = compute_kpis(df)
    cols = ["OrderID", "Courier", "DeliveryZone", "DeliveryService", "risk", "key_driver"]
    alerts = df.sort_values("risk", ascending=False).head(8)[cols].to_dict("records") if len(df) else []
    for a in alerts:
        a["action"] = action_for(a["risk"], a.get("key_driver", ""))
    payload = {"kpis": kpis, "alerts": alerts, "recommendations": recommendations(kpis),
               "count": int(len(df)), "source": source,
               "options": {"couriers": list(FULL_KPIS["courier_on_time"].keys()),
                           "zones": list(FULL_KPIS["zone_delay_rate"].keys())}}
    _cache[key] = payload
    return JSONResponse(payload)

@app.post("/api/ask")
async def ask(payload: dict):
    return {"answer": answer(payload.get("question", ""))}

REQUIRED_COLS = ["OrderTimestamp", "RequestedDeliveryDateTime"] + CAT + \
    ["LeadTimeDays", "OrderValue", "ZoneStandardDays", "DailyOrderVolume"]
COL_LABELS = {
    "OrderTimestamp": "order timestamp", "RequestedDeliveryDateTime": "requested delivery time",
    "Courier": "courier", "DeliveryZone": "delivery zone", "DeliveryService": "delivery service",
    "Occasion": "occasion", "ProductCategory": "product category", "FreshFlowers": "fresh flowers (yes/no)",
    "LeadTimeDays": "lead time (days)", "OrderValue": "order value",
    "ZoneStandardDays": "zone standard days", "DailyOrderVolume": "daily order volume",
}
REQUIRED_LABELS = ", ".join(COL_LABELS[c] for c in REQUIRED_COLS)

@app.post("/api/score")
async def score_upload(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception:
        return JSONResponse({"error": "Could not read the file. Please upload a valid CSV."}, status_code=400)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return JSONResponse({"error":
            f"Your CSV is missing: {', '.join(COL_LABELS[c] for c in missing)}. "
            f"An orders CSV needs these columns: {REQUIRED_LABELS}."}, status_code=400)
    try:
        out = score(df).head(20)
    except Exception as e:
        return JSONResponse({"error": f"Scoring failed: {e}"}, status_code=400)
    cols = [c for c in ["OrderID", "Courier", "DeliveryZone", "DeliveryService", "risk"] if c in out.columns]
    return {"alerts": out[cols].to_dict(orient="records")}

@app.get("/api/sample")
def sample():
    """Bundled sample orders CSV so a reviewer can test the upload feature without a dataset."""
    return FileResponse("sample_new_orders.csv", media_type="text/csv", filename="sample_new_orders.csv")

@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
