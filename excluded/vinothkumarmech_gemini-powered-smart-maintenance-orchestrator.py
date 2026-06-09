


!pip install -q scikit-learn pandas numpy joblib nest_asyncio
!pip install -q google-generativeai || true

import os
os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE"   # ← Replace with your key

import os, json, time, argparse, logging, asyncio
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import nest_asyncio
nest_asyncio.apply()

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "rul_model.pkl")
STATS_PATH = os.path.join(MODEL_DIR, "feature_stats.json")
MEMORY_PATH = "memory_store.json"
TRACE_LOG = "agent_trace.json"
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("mapmis")

import os, json, time, argparse, logging, asyncio
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import nest_asyncio
nest_asyncio.apply()

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "rul_model.pkl")
STATS_PATH = os.path.join(MODEL_DIR, "feature_stats.json")
MEMORY_PATH = "memory_store.json"
TRACE_LOG = "agent_trace.json"
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("mapmis")

def generate_synthetic_data(n_samples=3000, random_state=42):
    rng = np.random.RandomState(random_state)
    temp = rng.normal(loc=70, scale=8, size=n_samples)
    vibration = np.abs(rng.normal(loc=2.0, scale=1.2, size=n_samples))
    pressure = rng.normal(loc=5.5, scale=0.8, size=n_samples)
    rpm = rng.normal(loc=1500, scale=200, size=n_samples)
    load = rng.uniform(30, 100, size=n_samples)
    age = rng.exponential(scale=300, size=n_samples)

    baseline = 1000 - 0.6*age - 2.5*(load/10)
    temp_penalty = (temp - 65).clip(min=0) * 3.5
    vib_penalty  = vibration * 40
    rpm_penalty  = ((rpm - 1500)/100).clip(min=0)*6
    pres_penalty = ((pressure - 5.5)*10).clip(min=0)
    noise = rng.normal(scale=40, size=n_samples)

    rul = baseline - temp_penalty - vib_penalty - rpm_penalty - pres_penalty + noise
    rul = np.maximum(10, rul)

    return pd.DataFrame({
        "temperature": temp.round(2),
        "vibration": vibration.round(3),
        "pressure": pressure.round(3),
        "rpm": rpm.round(1),
        "load_pct": load.round(1),
        "age_cycles": age.round(1),
        "RUL": rul.round(1)
    })

def train_and_save_model(df):
    features = [c for c in df.columns if c != "RUL"]
    X = df[features].values
    y = df["RUL"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    joblib.dump(model, MODEL_PATH)

    stats = {
        "features": features,
        "means": np.mean(X, axis=0).tolist(),
        "stds": np.std(X, axis=0).tolist(),
        "importances": model.feature_importances_.tolist()
    }
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Trained. RMSE={rmse:.2f}, R2={r2:.3f}")
    return metrics

def generate_synthetic_data(n_samples=3000, random_state=42):
    rng = np.random.RandomState(random_state)
    temp = rng.normal(loc=70, scale=8, size=n_samples)
    vibration = np.abs(rng.normal(loc=2.0, scale=1.2, size=n_samples))
    pressure = rng.normal(loc=5.5, scale=0.8, size=n_samples)
    rpm = rng.normal(loc=1500, scale=200, size=n_samples)
    load = rng.uniform(30, 100, size=n_samples)
    age = rng.exponential(scale=300, size=n_samples)

    baseline = 1000 - 0.6*age - 2.5*(load/10)
    temp_penalty = (temp - 65).clip(min=0) * 3.5
    vib_penalty  = vibration * 40
    rpm_penalty  = ((rpm - 1500)/100).clip(min=0)*6
    pres_penalty = ((pressure - 5.5)*10).clip(min=0)
    noise = rng.normal(scale=40, size=n_samples)

    rul = baseline - temp_penalty - vib_penalty - rpm_penalty - pres_penalty + noise
    rul = np.maximum(10, rul)

    return pd.DataFrame({
        "temperature": temp.round(2),
        "vibration": vibration.round(3),
        "pressure": pressure.round(3),
        "rpm": rpm.round(1),
        "load_pct": load.round(1),
        "age_cycles": age.round(1),
        "RUL": rul.round(1)
    })

def train_and_save_model(df):
    features = [c for c in df.columns if c != "RUL"]
    X = df[features].values
    y = df["RUL"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    joblib.dump(model, MODEL_PATH)

    stats = {
        "features": features,
        "means": np.mean(X, axis=0).tolist(),
        "stds": np.std(X, axis=0).tolist(),
        "importances": model.feature_importances_.tolist()
    }
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Trained. RMSE={rmse:.2f}, R2={r2:.3f}")
    return metrics


class SensorAgent:
    def _init_(self):
        self.stats = json.load(open(STATS_PATH))
        self.keys = self.stats["features"]

    async def run(self, raw):
        out = {}
        for i, k in enumerate(self.keys):
            v = raw.get(k, self.stats["means"][i])
            out[k] = float(v)
        return out


class MLAgent:
    def _init_(self):
        self.model = joblib.load(MODEL_PATH)
        self.stats = json.load(open(STATS_PATH))
        self.features = self.stats["features"]

    async def predict(self, feats):
        x = [feats[f] for f in self.features]
        pred = float(self.model.predict([x])[0])

        contribs = []
        for i, f in enumerate(self.features):
            z = (x[i] - self.stats["means"][i]) / (self.stats["stds"][i] + 1e-6)
            contribs.append({"feature": f, "value": x[i], "score": round(-z*self.stats["importances"][i],4)})

        contribs = sorted(contribs, key=lambda c: abs(c["score"]), reverse=True)
        return {"RUL": round(pred,2), "contribs": contribs}



class LLMAgent:
    def _init_(self):
        self.use_gemini = GENAI_AVAILABLE and os.environ.get("GOOGLE_API_KEY")
        if self.use_gemini:
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    async def explain_failure(self, ml_out, sensors):
        prompt = f"""
You are an expert maintenance engineer.

ML Output:
{json.dumps(ml_out, indent=2)}

Sensor Input:
{json.dumps(sensors, indent=2)}

Explain:
- Why is RUL this value?
- Top 3 contributing features
- First maintenance check

Return JSON with keys:
explanation, top_features, first_check
"""

        if self.use_gemini:
            try:
                res = genai.generate_text(model="gemini-1.0", prompt=prompt)
                txt = res.text
                start, end = txt.find("{"), txt.rfind("}")
                return txt[start:end+1]
            except:
                pass

        # Fallback offline mode
        top = ml_out["contribs"][:3]
        return json.dumps({
            "explanation": f"Predicted RUL {ml_out['RUL']}. Strong influence from {[t['feature'] for t in top]}.",
            "top_features": [t["feature"] for t in top],
            "first_check": f"Inspect {top[0]['feature']} subsystem."
        }, indent=2)



class AdvisorAgent:
    def _init_(self, llm):
        self.llm = llm

    async def advise(self, ml_out, sensors):
        llm_json = await self.llm.explain_failure(ml_out, sensors)
        data = json.loads(llm_json)
        plan = []

        plan.append("Immediate check: " + data["first_check"])
        if ml_out["RUL"] < 100:
            plan.append("Urgent: replace critical components.")
        elif ml_out["RUL"] < 300:
            plan.append("Schedule maintenance in 2 weeks.")
        else:
            plan.append("Normal: continue operation with monitoring.")

        plan.append("Engineer note: " + data["explanation"])
        return plan


class Orchestrator:
    def _init_(self):
        self.sensor = SensorAgent()
        self.ml = MLAgent()
        self.llm = LLMAgent()
        self.advisor = AdvisorAgent(self.llm)
        self.memory = MemoryAgent()
        self.trace = TraceLogger()

    async def run(self, raw):
        sid = f"session-{int(time.time())}"

        feats = await self.sensor.run(raw)
        self.trace.record({"event": "sensor", "sid": sid})

        ml_out = await self.ml.predict(feats)
        self.trace.record({"event": "ml", "sid": sid})

        explanation = await self.llm.explain_failure(ml_out, feats)
        self.trace.record({"event": "llm", "sid": sid})

        plan = await self.advisor.advise(ml_out, feats)

        record = {"session_id": sid, "input": feats, "ml": ml_out, "explanation": explanation, "plan": plan}
        self.memory.add_session(record)

        return record


df = generate_synthetic_data(2500)
df.head()

metrics = train_and_save_model(df)
metrics


orch = Orchestrator()

sample = {
    "temperature": 85,
    "vibration": 5,
    "pressure": 6.5,
    "rpm": 1600,
    "load_pct": 90,
    "age_cycles": 500
}

result = await orch.run(sample)
print(json.dumps(result, indent=2))


print("Recent sessions:")
for s in orch.memory.query_recent(3):
    print(s["session_id"], s["ml"]["RUL"])

print("\nTrace log:")
print(json.load(open(TRACE_LOG)))

