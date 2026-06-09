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





# run this cell if needed (Kaggle usually has these)
!pip install -q scikit-learn matplotlib


# create data/sample_sales.csv used in demo
import os
os.makedirs("data", exist_ok=True)
csv_text = """date,amount,customer
2025-10-01,100,Acme
2025-10-02,120,Acme
2025-10-03,110,Globex
2025-10-10,150,Initech
2025-10-20,5000,FraudCo
2025-10-25,130,Acme
2025-11-01,140,Globex
2025-11-02,160,Initech
2025-11-10,170,Acme
2025-11-15,180,Globex
"""
with open("data/sample_sales.csv","w") as f:
    f.write(csv_text)
print("Sample CSV written to data/sample_sales.csv")


# ===========================
# Core: sessions, memory, observability, tools, agents
# ===========================
import time, json, uuid, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# ---------------------------
# Sessions & Memory
# ---------------------------
class InMemorySessionService:
    """Per-job session state (demo)."""
    def __init__(self):
        self._store = {}

    def save_session(self, session_id, data):
        self._store[session_id] = {
            "ts": time.time(),
            "data": data
        }

    def load_session(self, session_id):
        return self._store.get(session_id)

    def all_sessions(self):
        return self._store

class MemoryBank:
    """Simple long-term memory with compaction (keep last N entries)."""
    def __init__(self, path="memory.json", max_items=50):
        self.path = path
        self.max_items = max_items
        self._mem = deque(maxlen=max_items)
        # load if exists
        try:
            with open(self.path) as f:
                arr = json.load(f)
                for e in arr[-max_items:]:
                    self._mem.append(e)
        except Exception:
            pass

    def remember(self, key, value):
        entry = {"ts": time.time(), "key": key, "value": value}
        self._mem.append(entry)
        self._persist()

    def query(self, key_contains=None):
        arr = list(self._mem)
        if key_contains:
            return [e for e in arr if key_contains in e["key"] or key_contains in str(e["value"])]
        return arr

    def compact_context(self, keep_last_n=10):
        """Return compacted text-like context for LLM prompt (demo: join last n summaries)."""
        arr = list(self._mem)[-keep_last_n:]
        texts = []
        for e in arr:
            v = e["value"]
            if isinstance(v, str):
                texts.append(v)
            else:
                texts.append(json.dumps(v)[:400])  # short
        return "\n".join(texts)

    def _persist(self):
        try:
            with open(self.path,"w") as f:
                json.dump(list(self._mem), f, indent=2)
        except Exception:
            pass

# ---------------------------
# Observability
# ---------------------------
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("EIA")

class Metrics:
    def __init__(self):
        self.counters = {}
    def inc(self, name, n=1):
        self.counters[name] = self.counters.get(name,0) + n
    def report(self):
        return dict(self.counters)

class Tracer:
    def start(self, name):
        logger.info(f"[TRACE] start {name}")
        return time.time()
    def end(self, name, start_ts):
        logger.info(f"[TRACE] end {name} dur={time.time()-start_ts:.2f}s")

metrics = Metrics()
tracer = Tracer()

# ---------------------------
# LLM wrapper (mocked)
# ---------------------------
def llm_call(prompt, mock=True, max_tokens=512):
    """Mocked LLM call for reproducible demo. Set mock=False and implement to use real API."""
    if mock:
        # produce concise summary from analyzers-like input
        base = "Executive Summary (mock):\n"
        base += "- Key metrics computed; anomalies flagged.\n"
        base += "- 2 high-priority items require investigation.\n"
        base += "Prioritized Actions:\n1) Investigate anomalies\n2) Create ticket\n3) Monitor next period\n"
        return base
    else:
        raise NotImplementedError("Plug your LLM here using env var keys (do not commit keys).")

# ---------------------------
# Tools (custom & built-in examples)
# ---------------------------
def search_tool(query):
    logger.info(f"[Tool:Search] query='{query}'")
    return {"query": query, "top_result": f"Simulated result for {query}"}

def code_exec_tool(code):
    logger.info("[Tool:CodeExec] running code snippet")
    # VERY LIMITED safe exec sandbox (demo only)
    safe_globals = {"np": np, "pd": pd}
    try:
        exec(code, safe_globals)
        return safe_globals.get("output", "No 'output' variable set by code")
    except Exception as e:
        return f"Code execution error: {e}"

def ticket_tool(summary, meta=None):
    logger.info("[Tool:Ticket] create ticket (mocked)")
    ticket_id = f"TCK-{random_hex(6)}"
    return {"ticket_id": ticket_id, "status":"created", "summary": summary, "meta": meta}

def email_tool(subject, body, to):
    logger.info("[Tool:Email] draft (mocked)")
    return {"to": to, "subject": subject, "body": body, "status":"drafted"}

def random_hex(n=6):
    return uuid.uuid4().hex[:n]

# ---------------------------
# Agents (A2A style)
# ---------------------------
class BaseAgent:
    def __init__(self, name):
        self.name = name
    def send(self, other, message):
        logger.info(f"[A2A] {self.name} -> {other.name} : {message if isinstance(message,str) else 'payload'}")
        return other.receive(message)
    def receive(self, message):
        raise NotImplementedError

# Ingest Agent
class IngestAgent(BaseAgent):
    def receive(self, message):
        tracer_ts = tracer.start("Ingest")
        # message can be path or DataFrame
        if isinstance(message, pd.DataFrame):
            df = message.copy()
        else:
            df = pd.read_csv(message)
        logger.info(f"[Ingest] rows={len(df)} cols={df.shape[1]}")
        tracer.end("Ingest", tracer_ts)
        return df

# KPI Agent (fast)
class KPIAgent(BaseAgent):
    def receive(self, df):
        tracer_ts = tracer.start("KPI")
        res = {}
        if 'amount' in df.columns:
            res['total'] = float(df['amount'].sum())
            res['avg'] = float(df['amount'].mean())
        else:
            res['total'] = None; res['avg'] = None
        tracer.end("KPI", tracer_ts)
        metrics.inc("kpi_runs")
        return res

# Anomaly Agent (uses IsolationForest)
class AnomalyAgent(BaseAgent):
    def detect(self, series, contamination=0.02):
        if len(series) < 3:
            return []
        clf = IsolationForest(contamination=contamination, random_state=42)
        vals = series.values.reshape(-1,1)
        pred = clf.fit_predict(vals)
        idxs = np.where(pred==-1)[0].tolist()
        return idxs

    def receive(self, df):
        tracer_ts = tracer.start("Anomaly")
        if 'amount' not in df.columns:
            out = {"anomalies": [], "count": 0}
        else:
            idxs = self.detect(df['amount'])
            out = {"anomalies": idxs, "count": len(idxs)}
        tracer.end("Anomaly", tracer_ts)
        metrics.inc("anomaly_runs")
        return out

# Trend Agent
class TrendAgent(BaseAgent):
    def receive(self, df):
        tracer_ts = tracer.start("Trend")
        if 'date' in df.columns and 'amount' in df.columns:
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values('date', inplace=True)
            df['rolling'] = df['amount'].rolling(window=3, min_periods=1).mean()
            y = df['rolling'].values
            x = np.arange(len(y))
            slope = float(np.polyfit(x, y, 1)[0]) if len(x) >=2 else 0.0
            out = {"slope": slope, "points": len(y)}
        else:
            out = {"slope": None, "points": 0}
        tracer.end("Trend", tracer_ts)
        metrics.inc("trend_runs")
        return out

# Summary Agent (LLM powered)
class SummaryAgent(BaseAgent):
    def __init__(self, name, memory: MemoryBank):
        super().__init__(name)
        self.memory = memory

    def receive(self, analyzers_output):
        tracer_ts = tracer.start("Summary")
        # Build prompt using context compaction
        compacted = self.memory.compact_context(keep_last_n=5)
        prompt = {
            "compacted_memory": compacted,
            "analyzers": analyzers_output
        }
        logger.info("[Summary] building prompt (mocked)")
        summary_text = llm_call(prompt, mock=True)
        # store into memory
        self.memory.remember("summary_"+str(int(time.time())), summary_text)
        tracer.end("Summary", tracer_ts)
        metrics.inc("summary_runs")
        return summary_text

# Action Agent (suggest + optionally execute)
class ActionAgent(BaseAgent):
    def receive(self, summary_text, execute=False):
        tracer_ts = tracer.start("Action")
        # Simple suggestion logic based on keywords (demo)
        suggestions = []
        if "anomal" in summary_text.lower():
            suggestions.append({"priority":"High", "action":"create_ticket", "note":"Investigate anomalies"})
        suggestions.append({"priority":"Medium", "action":"draft_email", "note":"Notify owner"})
        executed = []
        if execute:
            # call mocked tools
            t = ticket_tool(summary_text, meta={"source":"EIA"})
            e = email_tool("Action Required", summary_text, to="owner@example.com")
            executed = [t,e]
            metrics.inc("actions_executed")
        tracer.end("Action", tracer_ts)
        return {"suggestions": suggestions, "executed": executed}

# Monitor / Loop Agent (Long-running demo)
class MonitorAgent(BaseAgent):
    def __init__(self, name, coordinator, interval_sec=2):
        super().__init__(name)
        self.coordinator = coordinator
        self.interval = interval_sec
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return "already running"
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return "started"

    def pause(self):
        self._running = False
        return "paused"

    def _loop(self):
        logger.info("[Monitor] loop started")
        while self._running:
            # demo: call coordinator with sample csv
            try:
                out = self.coordinator.run_job(csv_path="data/sample_sales.csv", dry_run=True)
                logger.info("[Monitor] run_job produced session %s", out.get("job_id"))
            except Exception as e:
                logger.error("[Monitor] loop error: %s", e)
            time.sleep(self.interval)
        logger.info("[Monitor] loop stopped")

# ---------------------------
# Coordinator (orchestrator)
# ---------------------------
class Coordinator:
    def __init__(self):
        self.sessions = InMemorySessionService()
        self.memory = MemoryBank()
        # instantiate agents
        self.ingest = IngestAgent("IngestAgent")
        self.kpi = KPIAgent("KPIAgent")
        self.anom = AnomalyAgent("AnomalyAgent")
        self.trend = TrendAgent("TrendAgent")
        self.summary = SummaryAgent("SummaryAgent", self.memory)
        self.action = ActionAgent("ActionAgent")
        self.monitor = MonitorAgent("MonitorAgent", self)
        self.metrics = metrics
        self.tracer = tracer

    def run_job(self, csv_path=None, df=None, dry_run=True):
        job_id = str(uuid.uuid4())
        start = self.tracer.start("Coordinator.run_job")
        # A2A: ingest
        df = self.ingest.receive(df if df is not None else csv_path)
        # Parallelize analyzers
        results = {}
        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = {
                ex.submit(self.kpi.receive, df): "kpi",
                ex.submit(self.anom.receive, df): "anomaly",
                ex.submit(self.trend.receive, df): "trend"
            }
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    results[key] = fut.result()
                except Exception as e:
                    results[key] = {"error": str(e)}
        # sequential: summary after analyzers
        analyzers_output = results
        summary_text = self.summary.receive(analyzers_output)
        # action recommendations
        actions = self.action.receive(summary_text, execute=not dry_run)
        # save session
        self.sessions.save_session(job_id, {"kpi":results.get("kpi"), "anomaly":results.get("anomaly"), "trend":results.get("trend"), "summary":summary_text, "actions":actions})
        # update global metrics
        self.metrics.inc("jobs_processed")
        self.tracer.end("Coordinator.run_job", start)
        return {"job_id": job_id, "summary": summary_text, "actions": actions}

# End of core cell
print("Core modules loaded: Ingest, KPI, Anomaly, Trend, Summary, Action, Monitor, Coordinator")


# Run the coordinator demo
coord = Coordinator()
out = coord.run_job(csv_path="data/sample_sales.csv", dry_run=True)
print("\n=== JOB OUTPUT ===")
print("Job ID:", out["job_id"])
print("\n--- Summary (mock LLM) ---\n", out["summary"])
print("\n--- Actions suggested ---\n", out["actions"])
print("\n--- Sessions stored (preview) ---")
import pprint
pp = pprint.PrettyPrinter(indent=2)
pp.pprint(coord.sessions.all_sessions())
print("\n--- Metrics ---")
print(coord.metrics.report())

# Show compacted memory context (for context engineering demo)
print("\n--- Compacted memory context preview ---")
print(coord.memory.compact_context(keep_last_n=3))


# Start monitor loop (demo will run in background, auto-stop after a few seconds)
print("Starting monitor (background thread) for 4 seconds demo...")
coord.monitor.start()
time.sleep(4)   # let it run a couple of iterations
coord.monitor.pause()
print("Monitor paused. Sessions now:", len(coord.sessions.all_sessions()))
print("Metrics after monitor run:", coord.metrics.report())


# Basic tests / assertions to demonstrate agent evaluation
import pandas as pd
df = pd.read_csv("data/sample_sales.csv")
# KPI test
kres = coord.kpi.receive(df)
assert kres["total"] == float(df['amount'].sum()), "KPI total mismatch"
# Anomaly test (should flag FraudCo row index)
ares = coord.anom.receive(df)
assert "count" in ares, "Anomaly output missing count"
# Trend test
tres = coord.trend.receive(df)
assert "slope" in tres, "Trend missing slope"
print("Unit checks passed (demo assertions).")


# Save a small report JSON and a basic chart image
import matplotlib.pyplot as plt
# prepare simple timeseries plot and mark anomalies (if any)
df = pd.read_csv("data/sample_sales.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
plt.figure(figsize=(8,3))
plt.plot(df['date'], df['amount'], marker='o')
plt.title("Sales amount over time (sample)")
plt.xlabel("date"); plt.ylabel("amount")
plt.tight_layout()
plt.savefig("assets_sales_plot.png")
print("Saved assets_sales_plot.png (attach this to Kaggle submission).")

# Save JSON summary file
with open("submission_summary.json","w") as f:
    json.dump({"job": out["job_id"], "summary": out["summary"], "actions": out["actions"]}, f, indent=2)
print("Saved submission_summary.json (attach this to Kaggle submission).")

