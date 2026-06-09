# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


!pip install -q pandas scikit-learn joblib loguru prometheus-client seaborn matplotlib


# Cell 1: Imports & plotting style (seaborn-like)
import json
import requests
import subprocess
import time
import uuid
import threading
import os
from datetime import datetime
from pathlib import Path

# ADK-style imports (use these where needed; we use deterministic fallbacks for offline runs)
from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import (
    RemoteA2aAgent,
    AGENT_CARD_WELL_KNOWN_PATH,
)
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Data & ML
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Logging & metrics
from loguru import logger
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid", palette="viridis")  # seaborn-like look

# Silence warnings in notebook
import warnings
warnings.filterwarnings("ignore")

print("âœ… Imports and plotting style set.")



# Cell 2: Prepare example folder & synthetic generator
Path("examples").mkdir(exist_ok=True)
Path("submission").mkdir(exist_ok=True)

def generate_synthetic_trace(path="examples/sample_traces.csv", n_flows=250, duration_s=600, mean_pkts=20, attack_at=None):
    """
    Creates a synthetic SDN flow event trace.
    Fields: timestamp, src, dst, proto, bytes, pkt_count, flow_id
    attack_at: (sec) optional time to inject attack spike (higher events/sec)
    """
    rng = np.random.default_rng(seed=42)
    rows = []
    start = int(time.time())
    flow_ids = [f"f{idx}" for idx in range(1, n_flows+1)]
    for sec in range(duration_s):
        # base event rate
        base_rate = 5
        if attack_at and abs(sec - attack_at) <= 10:  # spike window
            k = rng.poisson(base_rate * 5)
        else:
            k = rng.poisson(base_rate)
        for _ in range(max(0, k)):
            flow = rng.choice(flow_ids)
            pkt_count = int(max(1, rng.poisson(mean_pkts)))
            b = pkt_count * int(rng.integers(60,1500))
            rows.append({
                "timestamp": datetime.utcfromtimestamp(start + sec).isoformat(),
                "src": f"10.0.0.{rng.integers(1,255)}",
                "dst": f"10.0.1.{rng.integers(1,255)}",
                "proto": rng.choice(["TCP","UDP"]),
                "bytes": int(b),
                "pkt_count": pkt_count,
                "flow_id": flow
            })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"âœ… Synthetic trace written to {path} ({len(df)} rows)")
    return path

# Generate a sample trace with an attack spike at 300s
sample_path = generate_synthetic_trace(attack_at=300)



# Cell 3: Tools
def load_trace(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception:
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
    logger.info(f"Loaded trace {path} rows={len(df)}")
    return df

def aggregate_flow_stats(df: pd.DataFrame, window_seconds: int = 10) -> pd.DataFrame:
    df = df.copy()
    ts = df['timestamp'].astype('int64') // (10**9)
    df['window'] = (ts // window_seconds).astype(int)
    agg = df.groupby(['window','flow_id']).agg({'bytes':'sum','pkt_count':'sum'}).reset_index()
    occupancy = agg.groupby('window').agg({'flow_id':'nunique'}).rename(columns={'flow_id':'flow_count'}).reset_index()
    # bring back a readable time index
    occupancy['time'] = occupancy['window'] * window_seconds
    return occupancy

def detect_anomalies_zscore(occupancy_df, z_thresh=3.0):
    vals = occupancy_df['flow_count'].to_numpy()
    mu = vals.mean()
    sigma = vals.std(ddof=0) if vals.std(ddof=0) > 0 else 1.0
    z = (vals - mu) / sigma
    occupancy_df['zscore'] = z
    occupancy_df['anomaly'] = (np.abs(z) > z_thresh)
    return occupancy_df

def simulate_eviction_policy(occupancy_series: pd.Series, timeout_s=30):
    avg_occ = float(occupancy_series.mean() * (1.0 if timeout_s >= 30 else 0.9))
    eviction_rate = float(max(0.0, min(1.0, 1.0 - (timeout_s / 600.0))))
    return {"avg_occupancy": avg_occ, "eviction_rate": eviction_rate}



# Cell 4: Memory
import sqlite3, json

class SimpleInMemorySessionService:
    def __init__(self):
        self.sessions = {}
    def new_session(self, user_id: str):
        sid = f"{user_id}:{int(time.time())}:{uuid.uuid4().hex[:6]}"
        self.sessions[sid] = {"created": datetime.utcnow().isoformat(), "user_id": user_id, "context": []}
        return sid
    def append(self, sid, entry):
        if sid not in self.sessions:
            raise KeyError("session missing")
        self.sessions[sid]["context"].append({"ts": datetime.utcnow().isoformat(), "entry": entry})
    def get(self, sid):
        return self.sessions.get(sid, None)

class MemoryBankSQLite:
    def __init__(self, db_path="memory.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._ensure()
    def _ensure(self):
        cur = self.conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS experiments
                       (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, user_id TEXT, config TEXT, results TEXT)''')
        self.conn.commit()
    def log_experiment(self, user_id, config: dict, results: dict):
        cur = self.conn.cursor()
        cur.execute('INSERT INTO experiments (ts,user_id,config,results) VALUES (?,?,?,?)',
                    (datetime.utcnow().isoformat(), user_id, json.dumps(config), json.dumps(results)))
        self.conn.commit()
    def fetch_recent(self, limit=10):
        cur = self.conn.cursor()
        cur.execute('SELECT id, ts, user_id, config, results FROM experiments ORDER BY id DESC LIMIT ?', (limit,))
        rows = cur.fetchall()
        return [{"id":r[0], "ts":r[1], "user_id":r[2], "config":json.loads(r[3]), "results":json.loads(r[4])} for r in rows]

# instantiate
session_svc = SimpleInMemorySessionService()
mem_bank = MemoryBankSQLite()
print("âœ… Memory services ready.")



# Cell 5: Agents (plus data buffers used for plotting)
class AnalyzerAgent:
    def analyze(self, trace_path: str, window_seconds: int = 10):
        df = load_trace(trace_path)
        occupancy = aggregate_flow_stats(df, window_seconds=window_seconds)
        occupancy = detect_anomalies_zscore(occupancy)
        return occupancy

class PredictorAgent:
    def train(self, occupancy_df: pd.DataFrame, save_path="model.joblib"):
        X = occupancy_df['flow_count'].shift(1).fillna(occupancy_df['flow_count'].mean()).to_numpy().reshape(-1,1)
        y = occupancy_df['flow_count'].to_numpy()
        model = LinearRegression()
        model.fit(X, y)
        joblib.dump(model, save_path)
        return model
    def predict(self, model, last_value:int):
        return float(model.predict([[last_value]])[0])

class RecommenderAgent:
    def recommend(self, predicted_occ: float, target_capacity: int):
        ratio = predicted_occ / max(1, target_capacity)
        if ratio < 0.5:
            timeout = 60
        elif ratio < 0.9:
            timeout = 30
        elif ratio < 1.05:
            timeout = 15
        else:
            timeout = 5
        priority_policy = 'raise' if ratio > 1.0 else 'normal'
        return {"idle_timeout": timeout, "hard_timeout": 0, "priority_policy": priority_policy, "predicted_ratio": ratio}

class TutorAgent:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
    def explain(self, recommendation: dict, predicted_occ: float, target_capacity: int):
        return (f"Set idle_timeout={recommendation['idle_timeout']}s. Priority: {recommendation['priority_policy']}. "
                f"Predicted occupancy {predicted_occ:.1f} vs capacity {target_capacity}. "
                "Lower timeouts free space faster; raise priority for heavy flows if ratio >1.")

class LoopAgent:
    def __init__(self, analyzer, predictor, recommender, tutor, mem_bank, session_svc):
        self.analyzer = analyzer
        self.predictor = predictor
        self.recommender = recommender
        self.tutor = tutor
        self.mem_bank = mem_bank
        self.session_svc = session_svc
        self.jobs = {}
        # visualization buffers
        self.occupancy_history = []
        self.prediction_history = []
        self.anomaly_flags = []
        self.mitigation_actions = []
        # eviction matrix tracking (priority_levels x timeout_bins)
        self.eviction_matrix = np.zeros((5,6), dtype=int)

    def start_tuning(self, user_id: str, trace_path: str, target_capacity: int=100, max_iters:int=5):
        job_id = f"job-{uuid.uuid4().hex[:8]}"
        control = {"pause_event": threading.Event(), "stop": False, "thread": None, "result": None}
        control["pause_event"].set()
        def worker():
            sid = self.session_svc.new_session(user_id)
            occ = self.analyzer.analyze(trace_path)
            model = self.predictor.train(occ)
            last = int(occ['flow_count'].iloc[-1])
            iter_results = []
            for it in range(max_iters):
                if control["stop"]:
                    break
                while not control["pause_event"].is_set():
                    time.sleep(0.2)
                    if control["stop"]:
                        break
                predicted = self.predictor.predict(model, last)
                rec = self.recommender.recommend(predicted, target_capacity)
                sim = simulate_eviction_policy(occ['flow_count'], timeout_s=rec['idle_timeout'])
                explanation = self.tutor.explain(rec, predicted, target_capacity)
                # store to memory & session
                self.mem_bank.log_experiment(user_id, {"iter": it, "config": rec}, sim)
                self.session_svc.append(sid, {"iter": it, "predicted": predicted, "rec": rec, "sim": sim})
                # append visualization buffers
                self.occupancy_history.append(float(occ['flow_count'].mean()))
                self.prediction_history.append(predicted)
                self.anomaly_flags.append(bool(occ['anomaly'].any()))
                self.mitigation_actions.append(rec['idle_timeout'])
                # update eviction matrix: map priority (0-4) and timeout bins (0-5)
                pr_idx = 0 if rec['priority_policy']=='normal' else 4
                timeout_bin = min(5, int(rec['idle_timeout'] // 12))  # timeout mapped to 0..5
                self.eviction_matrix[pr_idx, timeout_bin] += int(round(sim['eviction_rate']*100))
                iter_results.append({"iter": it, "predicted": predicted, "rec": rec, "sim": sim, "explain": explanation})
                last = int(round((last*0.6 + predicted*0.4)))
                time.sleep(0.6)
            control["result"] = {"session": sid, "results": iter_results}
        t = threading.Thread(target=worker, daemon=True)
        control["thread"] = t
        self.jobs[job_id] = control
        t.start()
        return job_id

    def pause(self, job_id): 
        if job_id in self.jobs:
            self.jobs[job_id]["pause_event"].clear()
            return True
        return False
    def resume(self, job_id):
        if job_id in self.jobs:
            self.jobs[job_id]["pause_event"].set()
            return True
        return False
    def stop(self, job_id):
        if job_id in self.jobs:
            self.jobs[job_id]["stop"] = True
            self.jobs[job_id]["pause_event"].set()
            return True
        return False
    def status(self, job_id):
        if job_id not in self.jobs: return None
        ctl = self.jobs[job_id]
        alive = ctl["thread"].is_alive()
        paused = not ctl["pause_event"].is_set()
        return {"alive": alive, "paused": paused, "has_result": ctl.get("result") is not None}

# instantiate
an = AnalyzerAgent()
pr = PredictorAgent()
re = RecommenderAgent()
tu = TutorAgent()
loop_mgr = LoopAgent(an, pr, re, tu, mem_bank, session_svc)
print("âœ… Agents and loop manager instantiated.")



# Cell 6: Start a job and demo pause/resume
user = "saanu"
job_id = loop_mgr.start_tuning(user, sample_path, target_capacity=170, max_iters=6)
print("Started job:", job_id)

# monitor progress briefly
time.sleep(2)
print("Status after 2s:", loop_mgr.status(job_id))
time.sleep(2)
print("Pausing job for 1s...")
loop_mgr.pause(job_id)
time.sleep(1)
print("Status while paused:", loop_mgr.status(job_id))
print("Resuming job...")
loop_mgr.resume(job_id)

# wait for completion (timeout to avoid long blocking)
wait_for = 20
t0 = time.time()
while True:
    st = loop_mgr.status(job_id)
    if st is None:
        print("Job id missing.")
        break
    if not st["alive"] and st["has_result"]:
        break
    if time.time() - t0 > wait_for:
        print("Timed out waiting for job â€” continuing to visualization (partial data shown).")
        break
    time.sleep(0.5)

res = loop_mgr.jobs[job_id].get("result")
print("Job result session id:", res["session"])
print("Iterations recorded:", len(res["results"]))



# Cell 7: Occupancy trend & predicted vs predicted_history overlay
occ_hist = loop_mgr.occupancy_history
pred_hist = loop_mgr.prediction_history
iters = list(range(len(occ_hist)))

plt.figure(figsize=(12,5))
sns.lineplot(x=iters, y=occ_hist, marker="o", label="Observed mean occupancy")
sns.lineplot(x=iters, y=pred_hist, marker="X", linestyle="--", label="Predicted occupancy")
plt.fill_between(iters, np.array(occ_hist)-np.array(occ_hist)*0.1, np.array(occ_hist)+np.array(occ_hist)*0.1, alpha=0.08)
plt.title("Flow Table Occupancy: Observed vs Predicted")
plt.xlabel("Iteration")
plt.ylabel("Occupancy (flow count)")
plt.legend()
plt.tight_layout()
plt.show()



# Cell 8: Anomaly flags + mitigation (bar + scatter)
anoms = loop_mgr.anomaly_flags
mit_actions = loop_mgr.mitigation_actions
iters = list(range(len(anoms)))

fig, ax1 = plt.subplots(figsize=(12,4))
sns.barplot(x=iters, y=[1 if a else 0 for a in anoms], palette=["#d62728"], ax=ax1)
ax1.set_ylabel("Anomaly (1=yes)")
ax1.set_xlabel("Iteration")
ax1.set_ylim(-0.1, 1.4)
# overlay mitigation actions as scatter (timeout secs)
ax2 = ax1.twinx()
sns.scatterplot(x=iters, y=mit_actions, s=100, ax=ax2, color="#1f77b4")
ax2.set_ylabel("Suggested idle_timeout (s)")
plt.title("Anomaly Flags and Mitigation Actions per Iteration")
ax1.set_xticks(iters)
plt.tight_layout()
plt.show()



# Cell 9: Eviction heatmap
ev_mat = loop_mgr.eviction_matrix
plt.figure(figsize=(8,5))
ax = sns.heatmap(ev_mat, annot=True, fmt="d", cmap="viridis", cbar_kws={'label':'Eviction Score'})
ax.set_title("Eviction Heatmap (Priority rows -> Timeout bins cols)")
ax.set_ylabel("Priority Index (0=normal,4=raised)")
ax.set_xlabel("Timeout bin (0..5)")
plt.tight_layout()
plt.show()



# Cell 10: Agent timeline visualization (simulated times for demo)
agents = ["Analyzer", "Predictor", "Recommender", "Tutor"]
# create mock start/end times based on iterations to visualize sequencing/parallel behavior
agent_activity = []
base = time.time()
for i, a in enumerate(agents):
    start = base + i*0.2
    end = start + 0.6 + (i%2)*0.2
    agent_activity.append((a, start, end))

fig, ax = plt.subplots(figsize=(10, 2.5))
yticks = []
ylabels = []
for i, (agent, s, e) in enumerate(agent_activity):
    ax.broken_barh([(s, e - s)], (i - 0.4, 0.8), facecolors=sns.color_palette("viridis", len(agents))[i])
    yticks.append(i)
    ylabels.append(agent)
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels)
ax.set_xlabel("Simulated time")
ax.set_title("Agent Activity Timeline (simulated)")
plt.tight_layout()
plt.show()



# Cell 11: Quick evaluation
from sklearn.metrics import mean_squared_error

def evaluate_predictor_local(trace_path):
    occ = an.analyze(trace_path)
    model = pr.train(occ)
    X = occ['flow_count'].shift(1).fillna(occ['flow_count'].mean()).to_numpy().reshape(-1,1)
    y_true = occ['flow_count'].to_numpy()
    y_pred = model.predict(X)
    mse = mean_squared_error(y_true, y_pred)
    return {"mse": float(mse), "n_windows": len(occ)}

eval_res = evaluate_predictor_local(sample_path)
print("Predictor MSE:", eval_res["mse"])
# Mitigation summary
print("Mitigation actions observed (timeouts):", loop_mgr.mitigation_actions)
print("Anomaly flags:", loop_mgr.anomaly_flags)



# Cell 12: Save artifacts for submission
# README
readme_text = f"""
SDN Flow Table Anomaly Detection Assistant (Kaggle Notebook)
Author: Saanu ðŸ¤º

This notebook demonstrates:
- Multi-agent flow: Analyzer -> Predictor -> Recommender -> Tutor
- Tools: custom trace loader, aggregator, anomaly detector, simulator
- Memory: InMemory session + SQLite MemoryBank (memory.db)
- Long-running op: LoopAgent with pause/resume
- Visualizations: occupancy vs prediction, anomaly timeline, eviction heatmap, agent timeline

Predictor MSE: {eval_res['mse']:.4f}
"""
with open("submission/README.txt","w") as f:
    f.write(readme_text)

# save model if exists
if os.path.exists("model.joblib"):
    Path("submission/model.joblib").write_bytes(Path("model.joblib").read_bytes())
# copy memory DB
if os.path.exists("memory.db"):
    Path("submission/memory.db").write_bytes(Path("memory.db").read_bytes())

print("Saved submission artifacts in ./submission/")


