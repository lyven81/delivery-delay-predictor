"""Recommendation-quality evaluation (LLM-as-judge).
Generates Gemini recommendations from real KPIs across scenarios, then judges each for
accuracy (no invented numbers), usefulness, and relevance. Produces the real quality score."""
import os, json, time
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "gen-lang-client-0752018449")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")
import pandas as pd
from google import genai

CL = "C:/Users/Lenovo/Documents/04_Learning/Gen AI Academy/Cohort 2/Hackhaton/cleaned.csv"
df = pd.read_csv(CL)
df["OnTime"] = df["OnTime"].astype(str).str.strip().str.lower().isin(["true", "1"]).astype(int)
client = genai.Client()
MODEL = "gemini-2.5-flash"

def compute_kpis(d):
    late = d[d.OnTime == 0]
    cd = late.CourierDelayHours.clip(lower=0).sum(); idl = late.InternalDelayHours.clip(lower=0).sum()
    tot = cd + idl; ct = round(100*cd/tot) if tot > 0 else 0; wh = round((100-ct)*0.67)
    return {"on_time_rate": round(100*d.OnTime.mean()), "late_orders": int((d.OnTime == 0).sum()),
            "avg_delay_hours": round(late.LatenessHours.mean(), 1) if len(late) else 0,
            "refund_cost_rm": int(d.RefundAmount.sum()),
            "courier_on_time": {k: round(100*v) for k, v in d.groupby("Courier").OnTime.mean().sort_values(ascending=False).items()},
            "zone_delay_rate": {k: round(100*(1-v)) for k, v in d.groupby("DeliveryZone").OnTime.mean().sort_values().items()},
            "bottleneck_pct": {"Courier Transit": ct, "Warehouse Handling": wh, "Dispatch": 100-ct-wh}}

scenarios = {"All orders": df,
             "Ninja Van": df[df.Courier == "Ninja Van"], "In-house Rider": df[df.Courier == "In-house Rider"],
             "Other zone": df[df.DeliveryZone == "Other"], "Penang": df[df.DeliveryZone == "Penang"],
             "Same Day": df[df.DeliveryService == "Same Day"]}

def gen_recs(kpis):
    p = ("You are an operations analyst for an e-commerce delivery team. Using ONLY these metrics, "
         "give exactly 3 short, specific, actionable recommendations to reduce late deliveries. "
         "One sentence each, no preamble.\n" + json.dumps(kpis))
    t = client.models.generate_content(model=MODEL, contents=p).text
    return [l.strip("-*0123456789. ").strip() for l in t.splitlines() if l.strip()][:3]

def judge(kpis, rec):
    p = ("You are a STRICT evaluator of operations recommendations. Given METRICS (the only ground truth) "
         "and a RECOMMENDATION, score it. ACCURATE only if every number/claim is supported by METRICS "
         "(any invented figure = not grounded). USEFUL if specific and actionable for an operations manager. "
         "RELEVANT if it helps reduce late deliveries from these metrics. "
         'Return ONLY JSON: {"accuracy":1-5,"usefulness":1-5,"relevance":1-5,"grounded":true/false,"overall":1-5,"pass":true/false}. '
         "pass = overall>=4 AND grounded.\nMETRICS: " + json.dumps(kpis) + "\nRECOMMENDATION: " + rec)
    cfg = {"response_mime_type": "application/json"}
    r = client.models.generate_content(model=MODEL, contents=p, config=cfg)
    return json.loads(r.text)

rows = []
for name, d in scenarios.items():
    if len(d) == 0: continue
    k = compute_kpis(d)
    for rec in gen_recs(k):
        j = judge(k, rec)
        j["scenario"] = name; j["rec"] = rec
        rows.append(j)
        print(f"[{name}] pass={j['pass']} overall={j['overall']} grounded={j['grounded']} :: {rec[:70]}")
        time.sleep(0.3)

n = len(rows)
passed = sum(1 for r in rows if r["pass"])
avg = lambda key: round(sum(r[key] for r in rows)/n, 2)
report = {"n": n, "pass_rate_pct": round(100*passed/n),
          "avg_overall": avg("overall"), "avg_out_of_5": avg("overall"),
          "avg_accuracy": avg("accuracy"), "avg_usefulness": avg("usefulness"), "avg_relevance": avg("relevance"),
          "grounded_rate_pct": round(100*sum(1 for r in rows if r["grounded"])/n), "rows": rows}
with open("C:/Users/Lenovo/Documents/04_Learning/Gen AI Academy/Cohort 2/Hackhaton/delivery-delay-predictor/eval_report.json", "w") as f:
    json.dump(report, f, indent=2)
print("\n=== RECOMMENDATION QUALITY ===")
print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
