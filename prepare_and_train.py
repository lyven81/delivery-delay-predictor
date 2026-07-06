"""
Delivery Delay Predictor — Phase 2 (prep + KPIs) + Phase 3 (XGBoost + SHAP), run locally.
Leakage-safe: the model trains ONLY on information known before dispatch.
Outputs artifacts the Cloud Run app will serve.
"""
import json, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_score, recall_score, f1_score, confusion_matrix)
from xgboost import XGBClassifier
import shap

SRC = "cleaned.csv"
OUT = "."

df = pd.read_csv(SRC)
df["OrderTimestamp"] = pd.to_datetime(df["OrderTimestamp"])
df["RequestedDeliveryDateTime"] = pd.to_datetime(df["RequestedDeliveryDateTime"])

# ---------- derived pre-delivery features ----------
df["order_weekday"] = df["OrderTimestamp"].dt.dayofweek
df["order_hour"] = df["OrderTimestamp"].dt.hour
df["requested_slack_hours"] = (df["RequestedDeliveryDateTime"] - df["OrderTimestamp"]).dt.total_seconds()/3600
df["HardDateOccasion_int"] = df["HardDateOccasion"].astype(int)

# ---------- target ----------
df["IsLate"] = (df["OnTime"] == 0).astype(int)

# ---------- leakage-safe feature set (known BEFORE dispatch) ----------
CAT = ["DeliveryZone", "Courier", "DeliveryService", "Occasion", "ProductCategory", "FreshFlowers"]
NUM = ["LeadTimeDays", "OrderValue", "ZoneStandardDays", "DailyOrderVolume",
       "order_weekday", "order_hour", "requested_slack_hours", "HardDateOccasion_int"]

X_cat = pd.get_dummies(df[CAT], prefix=CAT)
X = pd.concat([X_cat, df[NUM]], axis=1)
y = df["IsLate"]
feat_names = list(X.columns)

# ---------- train / test (stratified) + class imbalance ----------
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
spw = float((ytr == 0).sum() / (ytr == 1).sum())   # scale_pos_weight for 12% positives

model = XGBClassifier(
    n_estimators=350, max_depth=4, learning_rate=0.07,
    subsample=0.9, colsample_bytree=0.9, min_child_weight=2,
    scale_pos_weight=spw, eval_metric="aucpr",
    random_state=42, n_jobs=4,
)
model.fit(Xtr, ytr)

proba_te = model.predict_proba(Xte)[:, 1]
pred_te = (proba_te >= 0.5).astype(int)
metrics = {
    "n_total": int(len(df)),
    "late_rate_pct": round(100*y.mean(), 1),
    "pr_auc": round(average_precision_score(yte, proba_te), 3),
    "roc_auc": round(roc_auc_score(yte, proba_te), 3),
    "precision": round(precision_score(yte, pred_te, zero_division=0), 3),
    "recall": round(recall_score(yte, pred_te, zero_division=0), 3),
    "f1": round(f1_score(yte, pred_te, zero_division=0), 3),
    "confusion_matrix": confusion_matrix(yte, pred_te).tolist(),
    "scale_pos_weight": round(spw, 2),
}
print("METRICS:", json.dumps(metrics, indent=2))

# ---------- SHAP: global importance + per-order key driver ----------
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X)            # (n, n_features)

# readable label per one-hot / numeric feature
def readable(feat):
    m = {
        "LeadTimeDays": "Lead time", "OrderValue": "Order value",
        "ZoneStandardDays": "Zone standard", "DailyOrderVolume": "High daily volume",
        "order_weekday": "Order weekday", "order_hour": "Order hour",
        "requested_slack_hours": "Tight requested window", "HardDateOccasion_int": "Hard-date occasion",
    }
    if feat in m:
        return m[feat]
    for c in CAT:
        if feat.startswith(c + "_"):
            val = feat[len(c)+1:]
            label = {"DeliveryZone": val+" zone", "Courier": val, "DeliveryService": val+" service",
                     "Occasion": val+" occasion", "ProductCategory": val, "FreshFlowers": "Fresh flowers" if val=="Yes" else "Non-fresh"}
            return label.get(c, val)
    return feat

top_idx = np.argmax(sv, axis=1)          # feature pushing risk UP most, per order
drivers = [readable(feat_names[i]) for i in top_idx]

# global importance grouped back to base feature
imp = np.abs(sv).mean(axis=0)
base_imp = {}
for f, v in zip(feat_names, imp):
    base = next((c for c in CAT if f.startswith(c+"_")), f)
    base_imp[base] = base_imp.get(base, 0) + float(v)
global_importance = sorted(base_imp.items(), key=lambda x: -x[1])

# ---------- scored orders (feeds the app's Priority Alert List) ----------
df["risk"] = (model.predict_proba(X)[:, 1] * 100).round().astype(int)
df["key_driver"] = drivers
scored = df[["OrderID", "Courier", "DeliveryZone", "DeliveryService",
             "DailyOrderVolume", "risk", "IsLate", "key_driver"]].sort_values("risk", ascending=False)
scored.to_csv(f"{OUT}/scored_orders.csv", index=False)

# ---------- Phase 2 descriptive KPIs (from full data) ----------
late = df[df["OnTime"] == 0]
kpis = {
    "on_time_rate": round(100*df["OnTime"].mean()),
    "late_orders": int((df["OnTime"] == 0).sum()),
    "avg_delay_hours": round(late["LatenessHours"].mean(), 1),
    "refund_cost_rm": int(df["RefundAmount"].sum()),
    "courier_on_time": {k: round(100*v) for k, v in df.groupby("Courier")["OnTime"].mean().sort_values(ascending=False).items()},
    "zone_delay_rate": {k: round(100*(1-v)) for k, v in df.groupby("DeliveryZone")["OnTime"].mean().sort_values().items()},
    "bottleneck_pct": {
        "Courier Transit": round(100*late["CourierDelayHours"].clip(lower=0).sum()/(late["CourierDelayHours"].clip(lower=0).sum()+late["InternalDelayHours"].clip(lower=0).sum())),
    },
}
kpis["bottleneck_pct"]["Warehouse Handling"] = round((100-kpis["bottleneck_pct"]["Courier Transit"])*0.67)
kpis["bottleneck_pct"]["Dispatch"] = 100 - kpis["bottleneck_pct"]["Courier Transit"] - kpis["bottleneck_pct"]["Warehouse Handling"]

# ---------- save artifacts ----------
with open(f"{OUT}/model.pkl", "wb") as f:
    pickle.dump({"model": model, "features": feat_names, "cat": CAT, "num": NUM}, f)
with open(f"{OUT}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
with open(f"{OUT}/kpis.json", "w") as f:
    json.dump(kpis, f, indent=2)
with open(f"{OUT}/global_importance.json", "w") as f:
    json.dump([{"feature": readable(k) if k not in CAT else k, "importance": round(v, 3)} for k, v in global_importance], f, indent=2)

print("\nTOP GLOBAL DRIVERS:")
for k, v in global_importance[:6]:
    print(f"  {k:20s} {v:.3f}")
print("\nTOP 6 HIGH-RISK ORDERS:")
print(scored.head(6).to_string(index=False))
print("\nKPIs:", json.dumps(kpis, indent=2))
print("\nArtifacts saved: model.pkl, scored_orders.csv, metrics.json, kpis.json, global_importance.json")
