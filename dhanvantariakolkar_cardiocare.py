# %% [markdown]
# # CardioCare — Multi-Agent Diagnosis & Risk Prediction System
# 
# Demo Kaggle notebook implementing:
# - XGBoost heart-risk model (train + eval)
# - Multi-agent orchestration (A2A-like messages)
# - Tools (preprocess, inference, thresholding)
# - Memory Bank (SQLite) and InMemory sessions
# - Observability (logging + simple tracing)
#
# NOTE: This notebook uses a **mock LLM** (simple template-based responses) for reproducible demo.
# Replace the `mock_llm` with real LLM calls (Gemini / OpenAI / etc.) when available.



# %% [markdown]
# ## 1. Setup — Install / Imports



# %%
# Install xgboost if not available (Kaggle usually has it)
!pip install -q xgboost==1.7.6

import os
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score
import xgboost as xgb

# For nicer display in Kaggle
pd.set_option('display.max_columns', 200)



# %%
# Try common Kaggle paths. If not found, generate a synthetic-ish dataset.
dataset_paths = [
    "/kaggle/input/heart-disease-uci/heart.csv",
    "/kaggle/input/uci-heart-disease/heart.csv",
    "/kaggle/input/heart-disease/heart.csv",
]

df = None
for p in dataset_paths:
    if os.path.exists(p):
        df = pd.read_csv(p)
        print("Loaded dataset from:", p)
        break

if df is None:
    print("No dataset file found in common Kaggle paths. Generating a synthetic example dataset (small).")
    # Generate a simple dataset with realistic-ish features
    rng = np.random.RandomState(42)
    n = 2000
    df = pd.DataFrame({
        "age": rng.randint(29, 77, size=n),
        "sex": rng.randint(0, 2, size=n),
        "cp": rng.randint(0, 4, size=n),
        "trestbps": rng.randint(90, 200, size=n),
        "chol": rng.randint(150, 320, size=n),
        "fbs": rng.randint(0, 2, size=n),
        "restecg": rng.randint(0, 2, size=n),
        "thalach": rng.randint(80, 200, size=n),
        "exang": rng.randint(0, 2, size=n),
        "oldpeak": np.round(rng.uniform(0, 6, size=n), 1),
        "slope": rng.randint(0, 3, size=n),
        "ca": rng.randint(0, 4, size=n),
        "thal": rng.randint(1, 4, size=n),
    })
    # Create synthetic target with some signal
    score = (
        0.03 * (df.age - 50)
        + 0.01 * (df.trestbps - 120)
        + 0.02 * (df.chol - 200)
        + 0.04 * (df.exang)
        + 0.03 * (df.oldpeak)
        + 0.02 * (df.ca)
    )
    probs = 1 / (1 + np.exp(-score))
    df["target"] = (rng.rand(n) < probs).astype(int)

print("Dataset shape:", df.shape)
df.head()



# %%
# Simple preprocessing: drop rows with missing values (dataset variants might have none).
df = df.dropna().reset_index(drop=True)

# Use all columns except 'target' as features
target_col = "target"
feature_cols = [c for c in df.columns if c != target_col]

X = df[feature_cols].astype(float)
y = df[target_col].astype(int)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Basic XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
model.fit(X_train, y_train)

# Save model artifact
model_path = "xgboost_model.json"
model.save_model(model_path)
print("Saved model to:", model_path)

# Evaluate
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_pred_proba)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
print(f"AUC: {auc:.4f}  Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}")



# %%
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0,1],[0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve (AUC={auc:.3f})")
plt.grid(alpha=0.3)
plt.show()



# %%
# Tool: preprocess input (simulate real-world incoming patient data)
def preprocess_patient_input(raw: Dict[str, Any]) -> Tuple[Dict[str, float], List[str]]:
    """
    raw: dict of user-provided fields (strings/numbers).
    Returns (features dict, list of validation messages)
    """
    msgs = []
    features = {}
    # We'll try to coerce known fields; unknown fields are ignored.
    for col in feature_cols:
        if col in raw:
            try:
                features[col] = float(raw[col])
            except Exception:
                msgs.append(f"Could not parse {col}, using default/mean.")
                features[col] = float(X[col].mean())
        else:
            # If missing, use training mean
            features[col] = float(X[col].mean())
            msgs.append(f"{col} missing — using population mean.")
    return features, msgs

# Tool: inference
def run_inference(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Run model inference and return structured response.
    """
    df_feat = pd.DataFrame([features])
    proba = model.predict_proba(df_feat)[0,1]
    risk_score = float(proba)               # 0..1
    # Also compute feature contributions via simple SHAP-like approximation (mock)
    # For demo, compute feature z-scores * model feature importance as a proxy
    importance = model.get_booster().get_score(importance_type="weight")
    # normalize importance onto existing columns
    contribs = {}
    for f in features:
        imp = importance.get(f, 0.0)
        contribs[f] = float((features[f] - X[f].mean()) / (X[f].std() + 1e-6) * imp)
    return {"risk": risk_score, "contribs": contribs}

# Tool: thresholding
def categorize_risk(risk_score: float) -> Tuple[str, Dict[str, Any]]:
    """
    Map risk_score to Low / Moderate / High with thresholds and meta info.
    """
    if risk_score < 0.25:
        return "Low", {"threshold": 0.25}
    elif risk_score < 0.6:
        return "Moderate", {"threshold": 0.6}
    else:
        return "High", {"threshold": 1.0}



# %%
# Simple MemoryBank using SQLite to store patient interactions
mem_db = "memory_bank.sqlite"

if os.path.exists(mem_db):
    os.remove(mem_db)  # fresh for demo

conn = sqlite3.connect(mem_db, check_same_thread=False)
cur = conn.cursor()
cur.execute("""
CREATE TABLE interactions (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    timestamp REAL,
    patient_name TEXT,
    features_json TEXT,
    risk REAL,
    risk_label TEXT,
    explanation TEXT
)
""")
conn.commit()

def save_interaction(session_id: str, patient_name: str, features: Dict[str,Any], risk: float, risk_label: str, explanation: str):
    row_id = str(uuid.uuid4())
    cur.execute("INSERT INTO interactions (id, session_id, timestamp, patient_name, features_json, risk, risk_label, explanation) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
               (row_id, session_id, time.time(), patient_name, json.dumps(features), float(risk), risk_label, explanation))
    conn.commit()
    return row_id

def get_patient_history(patient_name: str, limit:int=10) -> List[Dict[str,Any]]:
    cur.execute("SELECT timestamp, features_json, risk, risk_label, explanation FROM interactions WHERE patient_name = ? ORDER BY timestamp DESC LIMIT ?", (patient_name, limit))
    rows = cur.fetchall()
    out = []
    for r in rows:
        out.append({
            "timestamp": r[0],
            "features": json.loads(r[1]),
            "risk": r[2],
            "risk_label": r[3],
            "explanation": r[4]
        })
    return out

# InMemory session store (simple dict)
SESSION_STORE = {}
def create_session():
    sid = str(uuid.uuid4())
    SESSION_STORE[sid] = {"created": time.time(), "context": {}}
    return sid

def get_session(sid: str):
    return SESSION_STORE.get(sid)



# %%
LOGS = []

def log_event(level: str, agent: str, message: str, meta: Optional[Dict[str,Any]]=None):
    entry = {
        "ts": time.time(),
        "agent": agent,
        "level": level,
        "message": message,
        "meta": meta or {}
    }
    LOGS.append(entry)
    # also print for notebook readability
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry["ts"]))
    print(f"[{t}] [{level}] [{agent}] {message} {json.dumps(entry['meta']) if entry['meta'] else ''}")

def get_logs():
    return LOGS



# %%
def mock_llm_generate_explanation(patient_name: str, risk_score: float, risk_label: str, contribs: Dict[str,float], history: List[Dict[str,Any]]):
    """
    Return a human-facing explanation string given risk and top contributors.
    """
    # Top contributors
    sorted_contribs = sorted(contribs.items(), key=lambda x: -abs(x[1]))
    top = [f"{k} (impact {v:.2f})" for k,v in sorted_contribs[:4]]
    top_str = ", ".join(top) if top else "no dominant factors detected"
    # Simple history summary
    hist_msg = ""
    if history:
        last = history[0]
        hist_msg = f" Your last recorded risk was {last['risk']:.2f} ({last['risk_label']})."
    # Tone + advice (non-prescriptive)
    advice = ""
    if risk_label == "Low":
        advice = "Continue healthy lifestyle; routine checkups recommended."
    elif risk_label == "Moderate":
        advice = "Adopt lifestyle changes (diet, exercise), and consult a clinician for a personalized plan."
    else:
        advice = "High risk detected — please consult a healthcare professional soon for clinical evaluation."
    explanation = (f"Hello {patient_name}, your estimated cardiovascular risk score is {risk_score:.2f} ({risk_label}). "
                   f"Top contributing factors: {top_str}. {hist_msg} {advice} "
                   f"(This is an informational estimate, not a medical diagnosis.)")
    return explanation



# %%
@dataclass
class AAgent:
    name: str
    role: str

    def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Override per-agent in practice"""
        raise NotImplementedError

# Clinician Agent: collects info and asks clarifying questions (mock)
class ClinicianAgent(AAgent):
    def handle_message(self, message):
        # message contains 'raw_input' and 'session_id' and 'patient_name'
        raw_input = message.get("raw_input", {})
        session_id = message.get("session_id")
        patient_name = message.get("patient_name", "Patient")
        log_event("INFO", self.name, "Received raw input", {"raw_input": raw_input})
        # Preprocess
        features, msgs = preprocess_patient_input(raw_input)
        log_event("DEBUG", self.name, "Preprocessed input", {"messages": msgs})
        return {"features": features, "messages": msgs, "patient_name": patient_name, "session_id": session_id}

# Risk Model Agent: run inference tool
class RiskModelAgent(AAgent):
    def handle_message(self, message):
        features = message.get("features")
        log_event("INFO", self.name, "Running inference", {"feature_count": len(features)})
        inf_out = run_inference(features)
        risk = inf_out["risk"]
        contribs = inf_out["contribs"]
        log_event("DEBUG", self.name, "Inference result", {"risk": risk})
        return {"risk": risk, "contribs": contribs}

# Follow-up Agent: schedule/resume simulated long-running ops (mock)
class FollowUpAgent(AAgent):
    def __init__(self, name, role, memory_db_conn):
        super().__init__(name, role)
        self.conn = memory_db_conn

    def schedule_followup(self, session_id, patient_name, days=7):
        # For demo, just log a scheduled followup and write to memory as an 'event'
        log_event("INFO", self.name, f"Scheduled follow-up in {days} days", {"session_id": session_id, "patient": patient_name})
        # In a real system, you'd create a background job; here we store a record
        cur.execute("INSERT INTO interactions (id, session_id, timestamp, patient_name, features_json, risk, risk_label, explanation) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (str(uuid.uuid4()), session_id, time.time()+days*24*3600, patient_name, "{}", 0.0, "scheduled_followup", f"Follow-up scheduled in {days} days"))
        conn.commit()
        return {"status":"scheduled", "in_days": days}

# Orchestrator: coordinates A2A messages
class Orchestrator:
    def __init__(self, clinician: ClinicianAgent, risk_agent: RiskModelAgent, followup_agent: FollowUpAgent):
        self.clinician = clinician
        self.risk_agent = risk_agent
        self.followup_agent = followup_agent

    def run_flow(self, raw_input: Dict[str,Any], patient_name: str = "Patient"):
        session_id = create_session()
        log_event("INFO", "Orchestrator", "Starting flow", {"session_id": session_id, "patient_name": patient_name})
        # 1. Clinician collects + preprocesses
        clin_out = self.clinician.handle_message({"raw_input": raw_input, "session_id": session_id, "patient_name": patient_name})
        features = clin_out["features"]

        # 2. Risk model agent inference
        risk_out = self.risk_agent.handle_message({"features": features})
        risk = risk_out["risk"]
        contribs = risk_out["contribs"]

        # 3. Thresholding
        risk_label, meta = categorize_risk(risk)

        # 4. Fetch history from memory for context
        history = get_patient_history(patient_name)

        # 5. Generate explanation via LLM (mock)
        explanation = mock_llm_generate_explanation(patient_name, risk, risk_label, contribs, history)

        # 6. Save to memory
        row_id = save_interaction(session_id, patient_name, features, risk, risk_label, explanation)
        log_event("INFO", "Orchestrator", "Saved interaction to memory", {"row_id": row_id})

        # 7. If high risk, schedule follow-up
        followup_res = None
        if risk_label == "High":
            followup_res = self.followup_agent.schedule_followup(session_id, patient_name, days=7)

        # 8. Return structured response
        return {
            "session_id": session_id,
            "patient_name": patient_name,
            "features": features,
            "risk": risk,
            "risk_label": risk_label,
            "explanation": explanation,
            "followup": followup_res,
            "logs": get_logs()
        }

# Instantiate agents and orchestrator
clin_agent = ClinicianAgent("ClinicianAgent", "Collect data & interact")
risk_agent = RiskModelAgent("RiskModelAgent", "Run model")
follow_agent = FollowUpAgent("FollowUpAgent", "Manage follow-ups", conn)
orch = Orchestrator(clin_agent, risk_agent, follow_agent)



# %%
# Clear logs for clean demo
LOGS.clear()

# Example 1: typical patient
raw_1 = {
    "age": 62,
    "sex": 1,
    "cp": 3,
    "trestbps": 140,
    "chol": 250,
    "fbs": 0,
    "restecg": 0,
    "thalach": 130,
    "exang": 1,
    "oldpeak": 2.3,
    "slope": 2,
    "ca": 2,
    "thal": 3
}
res1 = orch.run_flow(raw_1, patient_name="Asha")
print("\n--- Explanation to user ---\n")
print(res1["explanation"])
print("\n--- Follow-up info ---\n")
print(res1["followup"])



# %%
# Example 2: younger low-risk
raw_2 = {
    "age": 36,
    "sex": 0,
    "cp": 1,
    "trestbps": 118,
    "chol": 180,
    "fbs": 0,
    "restecg": 0,
    "thalach": 170,
    "exang": 0,
    "oldpeak": 0.0,
    "slope": 1,
    "ca": 0,
    "thal": 2
}
res2 = orch.run_flow(raw_2, patient_name="Rohit")
print("\n--- Explanation to user ---\n")
print(res2["explanation"])



# %%
# Example 3: missing some fields (tests preprocessing defaults)
raw_3 = {
    "age": 55,
    "trestbps": 150,
    "chol": 280,
    # missing many fields
}
res3 = orch.run_flow(raw_3, patient_name="Priya")
print("\n--- Explanation to user ---\n")
print(res3["explanation"])



# %%
hist = get_patient_history("Asha")
print("Asha history entries:", len(hist))
for h in hist:
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(h["timestamp"]))
    print(f"- {t} | risk={h['risk']:.2f} ({h['risk_label']}) | explanation starts: {h['explanation'][:80]}...")



# %%
# Print the last 20 logs
for e in LOGS[-20:]:
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e["ts"]))
    print(f"[{t}] [{e['level']}] [{e['agent']}] {e['message']} {e['meta']}")



# %%
print(f"AUC on test set: {auc:.4f}")
print(f"Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}")

# Display feature importances (from XGBoost)
boost = model.get_booster()
importance = boost.get_score(importance_type="weight")
imp_df = pd.DataFrame(sorted(importance.items(), key=lambda x: x[1], reverse=True), columns=["feature","weight"])
imp_df.head(10)



# %%
# Save metadata
with open("cardiocare_metadata.json", "w") as f:
    json.dump({
        "project": "CardioCare Multi-Agent Demo",
        "model_path": model_path,
        "notes": "This notebook uses a deterministic mock LLM for the Clinician Agent. Replace mock_llm_* functions with real LLM calls when available."
    }, f, indent=2)
print("Saved metadata.")


