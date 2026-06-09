# GreenGuard Capstone Project - FULL Kaggle Notebook (No PDF, Markdown reports)
# Paste into a new Kaggle notebook. Cells are separated by "# %%"

# %%
# Title and description
# GreenGuard — AI Energy Waste Detection & Reduction Agent System
# Single-file notebook prepared for Kaggle Agents Intensive capstone.
# Uses: multi-agent structure, custom tools, session memory, context compaction, observability, evaluation.

# %%
# 1) Imports & Setup
import os
import json
import math
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
import pickle
import shutil

# Configuration
PROJECT_ROOT = "/kaggle/working/greenguard"
os.makedirs(PROJECT_ROOT, exist_ok=True)
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_PATH, exist_ok=True)
LOG_PATH = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_PATH, exist_ok=True)
REPORTS_PATH = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(REPORTS_PATH, exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# %%
# 2) Logging & tracing helpers
def log_event(event_type: str, payload: Dict[str, Any]):
    rec = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event_type": event_type,
        "payload": payload,
    }
    fname = os.path.join(LOG_PATH, "events.log")
    with open(fname, "a") as f:
        f.write(json.dumps(rec) + "\n")


def new_trace_id() -> str:
    return hex(random.getrandbits(64))[2:]


# %%
# 3) Sample dataset generator (per-hour readings)
def generate_sample_data(days: int = 45, freq_minutes: int = 60) -> pd.DataFrame:
    start = datetime.now() - timedelta(days=days)
    periods = int(days * 24 * 60 / freq_minutes)
    timestamps = pd.date_range(start=start, periods=periods, freq=f"{freq_minutes}min")
    rows = []
    for t in timestamps:
        base = 200 + 50 * math.sin(2 * math.pi * (t.hour) / 24)
        ac = 0.0
        if 11 <= t.hour <= 17 and random.random() < 0.6:
            ac = random.uniform(800, 2000)
        heater = 0.0
        if (6 <= t.hour <= 8 or 18 <= t.hour <= 20) and random.random() < 0.4:
            heater = random.uniform(1000, 3000)
        fridge = 100 + 50 * (1 if random.random() < 0.8 else 0)
        waste = 0.0
        if random.random() < 0.01:
            waste = random.uniform(300, 1500)
        total = base + ac + heater + fridge + waste + np.random.normal(0, 20)
        rows.append(
            {
                "timestamp": t,
                "base": float(base),
                "ac": float(ac),
                "heater": float(heater),
                "fridge": float(fridge),
                "waste": float(waste),
                "total_w": float(total),
            }
        )
    df = pd.DataFrame(rows)
    return df


sample_csv = os.path.join(DATA_PATH, "sample_energy_data.csv")
if not os.path.exists(sample_csv):
    df = generate_sample_data(days=45, freq_minutes=60)
    df.to_csv(sample_csv, index=False)
    print("Generated sample data:", df.shape)
else:
    df = pd.read_csv(sample_csv)
    print("Loaded sample data:", df.shape)

# %%
# 4) Memory implementation (InMemorySessionService + MemoryBank)
class InMemorySessionService:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def start_session(self, session_id: str, meta: Dict[str, Any] = None):
        self.sessions[session_id] = {
            "id": session_id,
            "meta": meta or {},
            "created": datetime.utcnow().isoformat() + "Z",
            "state": {},
        }
        log_event("session_started", {"session_id": session_id})

    def get(self, session_id: str):
        return self.sessions.get(session_id)

    def update_state(self, session_id: str, key: str, value: Any):
        if session_id in self.sessions:
            self.sessions[session_id]["state"][key] = value
            log_event("session_update", {"session_id": session_id, "key": key})


class MemoryBank:
    def __init__(self, filename: str):
        self.filename = filename
        if os.path.exists(filename):
            with open(filename, "r") as f:
                self.mem = json.load(f)
        else:
            self.mem = {"profiles": {}, "summaries": {}}
            self._save()

    def _save(self):
        with open(self.filename, "w") as f:
            json.dump(self.mem, f, default=str, indent=2)

    def get_profile(self, user_id: str) -> Dict[str, Any]:
        return self.mem["profiles"].get(user_id, {})

    def set_profile(self, user_id: str, profile: Dict[str, Any]):
        self.mem["profiles"][user_id] = profile
        self._save()
        log_event("memory_profile_set", {"user_id": user_id})

    def append_summary(self, user_id: str, key: str, value: Any):
        self.mem["summaries"].setdefault(user_id, {})
        self.mem["summaries"][user_id][key] = value
        self._save()
        log_event("memory_summary_update", {"user_id": user_id, "key": key})


memory_file = os.path.join(PROJECT_ROOT, "memory_bank.json")
memory = MemoryBank(memory_file)
session_service = InMemorySessionService()
memory.set_profile(
    "user_001", {"home_type": "2BHK", "preferred_ac_off_time": "22:00", "timezone": "Asia/Kolkata"}
)

# %%
# 5) Context compaction
def context_compactor(df_local: pd.DataFrame) -> Dict[str, Any]:
    df_local = df_local.copy()
    df_local["date"] = pd.to_datetime(df_local["timestamp"]).dt.date
    summary = {}
    summary["mean_total"] = float(df_local["total_w"].mean())
    summary["median_total"] = float(df_local["total_w"].median())
    summary["max_total"] = float(df_local["total_w"].max())
    df_local["hour"] = pd.to_datetime(df_local["timestamp"]).dt.hour
    hourly = df_local.groupby("hour")["total_w"].mean().to_dict()
    summary["hourly_avg"] = {int(k): float(v) for k, v in hourly.items()}
    return summary


# %%
# 6) Tools: CSV loader, Markdown report writer, tips fetcher
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def save_markdown_report(filename: str, title: str, body: str, tables: Dict[str, pd.DataFrame] = None):
    """
    Save a simple markdown report. tables is a dict with name->DataFrame
    """
    lines = []
    lines.append(f"# {title}\n")
    lines.append(body + "\n")
    if tables:
        for name, df_table in tables.items():
            lines.append(f"## {name}\n")
            # write small table as markdown
            lines.append(df_table.head(10).to_markdown(index=False))
            lines.append("\n")
    with open(filename, "w") as f:
        f.write("\n".join(lines))
    log_event("report_saved_md", {"path": filename})


def fetch_efficiency_tips(query: str, limit: int = 5) -> List[str]:
    tips = [
        "Set AC temperature to 24-26°C and use a timer",
        "Use LED bulbs; they consume up to 80% less than incandescent",
        "Unplug idle chargers and devices",
        "Schedule heavy-duty appliances during off-peak hours",
        "Maintain refrigerator seals and clean coils",
    ]
    return tips[:limit]


# %%
# 7) Simple anomaly detection / label preparation and model training
edf = load_csv(sample_csv)
edf["timestamp"] = pd.to_datetime(edf["timestamp"])
edf["rolling_med"] = edf["total_w"].rolling(window=24, min_periods=1).median()
edf["is_spike"] = (edf["total_w"] > 3 * edf["rolling_med"]).astype(int)
edf["is_waste"] = ((edf["waste"] > 200) | (edf["is_spike"] == 1)).astype(int)

edf["hour"] = edf["timestamp"].dt.hour
features = ["base", "ac", "heater", "fridge", "hour"]
X = edf[features].fillna(0)
y = edf["is_waste"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
clf = RandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
print("Model metrics on test set:")
print("Precision:", precision_score(y_test, preds))
print("Recall:", recall_score(y_test, preds))
print("F1:", f1_score(y_test, preds))

model_path = os.path.join(PROJECT_ROOT, "anomaly_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(clf, f)
log_event("model_trained", {"model_path": model_path})

# %%
# 8) Agents (Analyzer, Predictor, Reporter, Notify)
class Agent:
    def __init__(self, name: str):
        self.name = name

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class AnalyzerAgent(Agent):
    def __init__(self, model_path: str):
        super().__init__("AnalyzerAgent")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        trace_id = new_trace_id()
        start = time.time()
        df_recent = context["df_recent"].copy()
        Xf = df_recent[["base", "ac", "heater", "fridge", "hour"]].fillna(0)
        preds_local = self.model.predict(Xf)
        df_recent["pred_is_waste"] = preds_local
        waste_events = df_recent[df_recent["pred_is_waste"] == 1]
        analysis = {
            "waste_event_count": int(waste_events.shape[0]),
            "examples": waste_events.head(5).to_dict(orient="records"),
        }
        log_event(
            "analyzer_run",
            {"trace_id": trace_id, "duration_s": time.time() - start, "events": analysis["waste_event_count"]},
        )
        return {"analysis": analysis, "trace_id": trace_id}


class PredictorAgent(Agent):
    def __init__(self):
        super().__init__("PredictorAgent")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        trace_id = new_trace_id()
        df_recent = context["df_recent"]
        avg_w = float(df_recent["total_w"].mean())
        threshold = 800.0
        risk = "HIGH" if avg_w > threshold else ("MEDIUM" if avg_w > threshold * 0.7 else "LOW")
        prediction = {"avg_recent_w": avg_w, "risk_level": risk}
        log_event("predictor_run", {"trace_id": trace_id, "prediction": prediction})
        return {"prediction": prediction, "trace_id": trace_id}


class ReporterAgent(Agent):
    def __init__(self):
        super().__init__("ReporterAgent")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        trace_id = new_trace_id()
        analysis = context["analysis"]
        prediction = context["prediction"]
        tips = fetch_efficiency_tips("energy saving")
        title = "GreenGuard Energy Report"
        body = (
            f"GreenGuard Report - Generated {datetime.utcnow().isoformat()}\n\n"
            f"Summary:\n- Waste events detected: {analysis['waste_event_count']}\n- Risk level: {prediction['risk_level']}\n\n"
            "Recommendations:\n" + "\n".join(["- " + t for t in tips])
        )
        # prepare table examples if present
        examples_df = pd.DataFrame(analysis["examples"]) if analysis["examples"] else pd.DataFrame()
        if not examples_df.empty:
            examples_df = examples_df.copy()
            examples_df["timestamp"] = examples_df["timestamp"].astype(str)
        report_path = os.path.join(REPORTS_PATH, f"report_{int(time.time())}.md")
        save_markdown_report(report_path, title, body, {"examples": examples_df} if not examples_df.empty else None)
        log_event("reporter_run", {"trace_id": trace_id, "report_path": report_path})
        return {"report_path": report_path, "trace_id": trace_id}


class NotifyAgent(Agent):
    def __init__(self):
        super().__init__("NotifyAgent")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        trace_id = new_trace_id()
        prediction = context["prediction"]
        analysis = context["analysis"]
        user_profile = context.get("user_profile", {})
        msg = (
            "Hello! GreenGuard detected "
            + str(analysis["waste_event_count"])
            + " possible waste events. Your risk level is "
            + str(prediction["risk_level"])
            + ". Recommendation: "
            + fetch_efficiency_tips("energy saving")[0]
        )
        print("\n[NOTIFICATION]")
        print(msg)
        log_event("notify_run", {"trace_id": trace_id, "message": msg, "user": user_profile})
        return {"message": msg, "trace_id": trace_id}


# %%
# 9) Orchestrator (sequential + loop)
class OrchestratorAgent(Agent):
    def __init__(
        self,
        analyzer: AnalyzerAgent,
        predictor: PredictorAgent,
        reporter: ReporterAgent,
        notifier: NotifyAgent,
        memory_obj: MemoryBank,
        session_svc: InMemorySessionService,
    ):
        super().__init__("Orchestrator")
        self.analyzer = analyzer
        self.predictor = predictor
        self.reporter = reporter
        self.notifier = notifier
        self.memory = memory_obj
        self.session_service = session_svc
        self.running = False

    def run_once(self, user_id: str, df_recent: pd.DataFrame) -> Dict[str, Any]:
        trace_id = new_trace_id()
        session_id = f"session_{user_id}_{int(time.time())}"
        self.session_service.start_session(session_id, {"user_id": user_id, "trace_id": trace_id})
        analyzer_res = self.analyzer.run({"df_recent": df_recent})
        predictor_res = self.predictor.run({"df_recent": df_recent})
        reporter_res = self.reporter.run({"analysis": analyzer_res["analysis"], "prediction": predictor_res["prediction"]})
        user_profile = self.memory.get_profile(user_id)
        notify_res = self.notifier.run({"analysis": analyzer_res["analysis"], "prediction": predictor_res["prediction"], "user_profile": user_profile})
        comp = context_compactor(df_recent)
        self.memory.append_summary(user_id, "last_run_summary", comp)
        result = {
            "analyzer": analyzer_res,
            "predictor": predictor_res,
            "reporter": reporter_res,
            "notify": notify_res,
            "session_id": session_id,
            "trace_id": trace_id,
        }
        log_event("orchestrator_run", {"trace_id": trace_id})
        return result

    def start_loop(self, user_id: str, df_stream_func, interval_seconds: int = 5, runs: int = 3):
        self.running = True
        run_count = 0
        while self.running and run_count < runs:
            df_recent = df_stream_func()
            print(f"--- Loop run {run_count+1} at {datetime.utcnow().isoformat()} ---")
            _ = self.run_once(user_id, df_recent)
            run_count += 1
            time.sleep(interval_seconds)
        print("Loop finished")

    def stop_loop(self):
        self.running = False


# %%
# 10) Demo wiring: create agents and run one orchestration + short loop
analyzer = AnalyzerAgent(model_path=model_path)
predictor = PredictorAgent()
reporter = ReporterAgent()
notifier = NotifyAgent()
orch = OrchestratorAgent(analyzer, predictor, reporter, notifier, memory, session_service)


def get_last_48h() -> pd.DataFrame:
    df_local = pd.read_csv(sample_csv)
    df_local["timestamp"] = pd.to_datetime(df_local["timestamp"])
    cutoff = datetime.now() - timedelta(hours=48)
    df_recent = df_local[df_local["timestamp"] >= cutoff].copy()
    df_recent["hour"] = df_recent["timestamp"].dt.hour
    return df_recent


# Run orchestrator once
result = orch.run_once("user_001", get_last_48h())
print("\nOrchestrator trace:", result["trace_id"])

# Short loop (2 iterations) - safe in Kaggle
orch.start_loop("user_001", get_last_48h, interval_seconds=1, runs=2)

# %%
# 11) Evaluation (metrics saved to logs)
def evaluate_model(model, X_test_local, y_test_local):
    preds_local = model.predict(X_test_local)
    return {
        "precision": float(precision_score(y_test_local, preds_local)),
        "recall": float(recall_score(y_test_local, preds_local)),
        "f1": float(f1_score(y_test_local, preds_local)),
    }


metrics = evaluate_model(clf, X_test, y_test)
print("Evaluation metrics:", metrics)
log_event("evaluation_metrics", metrics)

# %%
# 12) Visualizations and examples
plt.figure(figsize=(10, 4))
edf_local = pd.read_csv(sample_csv)
edf_local["timestamp"] = pd.to_datetime(edf_local["timestamp"])
edf_local["hour"] = edf_local["timestamp"].dt.hour
hr = edf_local.groupby("hour")["total_w"].mean()
plt.plot(hr.index, hr.values)
plt.title("Hourly Average Power Consumption (W)")
plt.xlabel("Hour")
plt.ylabel("Average W")
plt.grid(True)
plt.show()

examples_local = edf[edf["is_waste"] == 1].head(10)
if not examples_local.empty:
    display_df = examples_local[["timestamp", "total_w", "waste"]].head(10)
    print("Example waste events:")
    print(display_df.to_string(index=False))
else:
    print("No example waste events in dataset head.")

# %%
# 13) Packaging and submission notes (files saved in /kaggle/working/greenguard)
readme_text = """# GreenGuard - Capstone Project

This notebook is the single-file Kaggle Notebook submission for the GreenGuard Capstone.

Features included:
- Multi-agent system (Analyzer, Predictor, Reporter, Notifier, Orchestrator)
- Custom tools (minimal Markdown report generator, CSV loader)
- Sessions & Memory (InMemorySessionService, MemoryBank)
- Context compaction
- Observability: structured logging
- Simple ML model for anomaly detection

Follow steps in this notebook to run the demo and reproduce results.
"""

with open(os.path.join(PROJECT_ROOT, "README_NOTEBOOK.txt"), "w") as f:
    f.write(readme_text)

final_notes = """Submission notes (for Kaggle):
Title: GreenGuard — AI Energy Waste Detection & Reduction Agent System
Track: Agents for Good
Short pitch: GreenGuard detects energy waste using a multi-agent system that analyzes smart meter data, forecasts bill risk, and generates actionable recommendations and Markdown reports. The system demonstrates multi-agent orchestration, custom tools, sessions & memory, context engineering, and observability.

How to run:
1. Open this notebook on Kaggle
2. Run all cells in order
3. Download generated artifacts (reports & logs) from the /kaggle/working/greenguard directory

Do not include any API keys in the notebook.
"""

with open(os.path.join(PROJECT_ROOT, "SUBMISSION_NOTES.txt"), "w") as f:
    f.write(final_notes)

# Make ZIP of artifacts
shutil.make_archive(os.path.join(PROJECT_ROOT, "greenguard_artifacts"), "zip", PROJECT_ROOT)
print("Packaged artifacts:", os.path.join(PROJECT_ROOT, "greenguard_artifacts.zip"))

# %%
# End of notebook


