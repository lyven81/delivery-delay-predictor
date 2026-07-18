"""Vertex AI custom training: tune + calibrate the delivery-delay model, score every order.
Reads cleaned.csv from GCS, writes model.pkl + scored.csv + metrics.json back to GCS."""
import argparse, json, warnings, numpy as np, pandas as pd, pickle
warnings.filterwarnings("ignore")
from google.cloud import storage
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
                             recall_score, f1_score, precision_recall_curve, brier_score_loss)
from xgboost import XGBClassifier
import shap

def _split(uri): b, _, k = uri[5:].partition("/"); return b, k
def dl(uri, local): b, k = _split(uri); storage.Client().bucket(b).blob(k).download_to_filename(local)
def ul(local, uri): b, k = _split(uri); storage.Client().bucket(b).blob(k).upload_from_filename(local)

ap = argparse.ArgumentParser()
ap.add_argument("--data_uri", required=True)
ap.add_argument("--out_uri", required=True)   # gs://bucket/prefix
A = ap.parse_args()

dl(A.data_uri, "cleaned.csv")
df = pd.read_csv("cleaned.csv")
df["OrderTimestamp"] = pd.to_datetime(df["OrderTimestamp"])
df["RequestedDeliveryDateTime"] = pd.to_datetime(df["RequestedDeliveryDateTime"])
df["order_weekday"] = df["OrderTimestamp"].dt.dayofweek
df["order_hour"] = df["OrderTimestamp"].dt.hour
df["requested_slack_hours"] = (df["RequestedDeliveryDateTime"] - df["OrderTimestamp"]).dt.total_seconds()/3600
df["HardDateOccasion_int"] = df["HardDateOccasion"].astype(int)
ot = df["OnTime"].astype(str).str.strip().str.lower().isin(["true", "1"])
df["IsLate"] = (~ot).astype(int)

CAT = ["DeliveryZone", "Courier", "DeliveryService", "Occasion", "ProductCategory", "FreshFlowers"]
NUM = ["LeadTimeDays", "OrderValue", "ZoneStandardDays", "DailyOrderVolume",
       "order_weekday", "order_hour", "requested_slack_hours", "HardDateOccasion_int"]
X = pd.concat([pd.get_dummies(df[CAT], prefix=CAT), df[NUM]], axis=1)
y = df["IsLate"]; feat = list(X.columns)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
spw = float((ytr == 0).sum() / (ytr == 1).sum())

def M(yt, pr, pd_): return {"recall": round(recall_score(yt, pd_), 3),
    "precision": round(precision_score(yt, pd_, zero_division=0), 3),
    "pr_auc": round(average_precision_score(yt, pr), 3), "roc_auc": round(roc_auc_score(yt, pr), 3),
    "f1": round(f1_score(yt, pd_, zero_division=0), 3)}

base = XGBClassifier(n_estimators=350, max_depth=4, learning_rate=0.07, subsample=0.9,
    colsample_bytree=0.9, min_child_weight=2, scale_pos_weight=spw, eval_metric="aucpr",
    random_state=42, n_jobs=4).fit(Xtr, ytr)
pb = base.predict_proba(Xte)[:, 1]
baseline = M(yte, pb, (pb >= 0.5).astype(int))

grid = {"n_estimators": [300, 400, 600], "max_depth": [3, 4, 5], "learning_rate": [0.03, 0.05, 0.07],
        "subsample": [0.8, 0.9, 1.0], "colsample_bytree": [0.7, 0.9, 1.0],
        "min_child_weight": [1, 2, 4], "gamma": [0, 0.5, 1]}
srch = RandomizedSearchCV(XGBClassifier(scale_pos_weight=spw, eval_metric="aucpr", random_state=42, n_jobs=4),
    grid, n_iter=20, scoring="average_precision",
    cv=StratifiedKFold(3, shuffle=True, random_state=42), random_state=42, n_jobs=4).fit(Xtr, ytr)
best = srch.best_estimator_
cal = CalibratedClassifierCV(best, method="isotonic", cv=5).fit(Xtr, ytr)
pt = cal.predict_proba(Xte)[:, 1]
ptr = cal.predict_proba(Xtr)[:, 1]
prec, rec, thr = precision_recall_curve(ytr, ptr)
f1s = 2*prec*rec/(prec+rec+1e-9)
best_thr = float(thr[int(np.nanargmax(f1s[:-1]))])
tuned = M(yte, pt, (pt >= best_thr).astype(int)); tuned["threshold"] = round(best_thr, 3)

risk_all = (cal.predict_proba(X)[:, 1] * 100).round().astype(int)
at_risk = int((risk_all >= 60).sum())

def readable(f):
    m = {"LeadTimeDays": "Lead time", "OrderValue": "Order value", "ZoneStandardDays": "Zone standard days",
         "DailyOrderVolume": "High daily volume", "order_weekday": "Order weekday",
         "order_hour": "Late-day order cut-off", "requested_slack_hours": "Tight delivery window",
         "HardDateOccasion_int": "Hard-date occasion"}
    if f in m: return m[f]
    for c in CAT:
        if f.startswith(c+"_"):
            v = f[len(c)+1:]
            return {"DeliveryZone": v+" zone", "Courier": v, "DeliveryService": v+" service",
                    "Occasion": v+" occasion", "ProductCategory": v,
                    "FreshFlowers": "Fresh flowers" if v == "Yes" else "Non-fresh"}.get(c, v)
    return f
sv = shap.TreeExplainer(best).shap_values(X)
drivers = [readable(feat[i]) for i in np.argmax(sv, axis=1)]
# low-risk orders have no meaningful risk driver
drivers = [d if r >= 25 else "Within normal range" for d, r in zip(drivers, risk_all)]

scored = df[["OrderID"]].copy()
scored["risk"] = risk_all
scored["key_driver"] = drivers
scored.to_csv("scored.csv", index=False)

metrics = {"baseline": baseline, "tuned": tuned,
           "brier_base": round(brier_score_loss(yte, pb), 3), "brier_tuned": round(brier_score_loss(yte, pt), 3),
           "at_risk_count": at_risk, "n": int(len(df)), "late": int(y.sum())}
with open("metrics.json", "w") as f: json.dump(metrics, f, indent=2)
with open("model.pkl", "wb") as f: pickle.dump({"model": cal, "features": feat, "cat": CAT, "num": NUM}, f)

for name in ["model.pkl", "scored.csv", "metrics.json"]:
    ul(name, A.out_uri.rstrip("/") + "/" + name)
print("TRAINING DONE. metrics:", json.dumps(metrics))
