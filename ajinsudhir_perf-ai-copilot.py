# Install all required libraries
!pip install -q pandas matplotlib google-generativeai markdown2 fpdf2



# General Imports
import os
import json
import uuid
import base64
import io
import logging
from typing import Dict, Any, Union
from threading import RLock

# Core Libraries
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt
from jinja2 import Template
import markdown2
from fpdf import FPDF
import google.generativeai as genai

# Kaggle-specific import for secrets
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

# Set the API key from Kaggle Secrets
os.environ['GOOGLE_API_KEY'] = user_secrets.get_secret("GOOGLE_API_KEY")

print("Libraries imported and API key configured.")



# General Imports
import os
import json
import uuid
import base64
import io
import logging
from typing import Dict, Any, Union
from threading import RLock
import tempfile

# Core Libraries
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt
from jinja2 import Template
import markdown2
from fpdf import FPDF
import google.generativeai as genai

# Kaggle-specific import for secrets
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

# Set the API key from Kaggle Secrets
# Make sure you have a secret named "GOOGLE_API_KEY" in your notebook
os.environ['GOOGLE_API_KEY'] = user_secrets.get_secret("GOOGLE_API_KEY")

# ==============================================================================
# Storage Functions (formerly app/storage.py)
# ==============================================================================
SESSIONS: Dict[str, Dict[str, Any]] = {}
MEMORY_BANK: Dict[str, Dict[str, Any]] = {}
_lock = RLock()

def create_session(metadata: dict) -> str:
    sid = str(uuid.uuid4())
    with _lock:
        SESSIONS[sid] = {"id": sid, "created_at": "now", "metadata": metadata, "artifacts": {}, "status": "created"}
    return sid

def get_session(sid: str):
    with _lock:
        return SESSIONS.get(sid)

def update_session_status(sid: str, status: str):
    with _lock:
        session = get_session(sid)
        if session: session["status"] = status

def store_artifact(sid: str, name: str, data: Any):
    with _lock:
        session = get_session(sid)
        if session: session["artifacts"][name] = data

def push_to_memory(key: str, item: dict):
    with _lock:
        if key not in MEMORY_BANK: MEMORY_BANK[key] = {"history": []}
        MEMORY_BANK[key]["history"].append(item)

def get_last_run_from_memory(key: str) -> Union[Dict[str, Any], None]:
    with _lock:
        memory_entry = MEMORY_BANK.get(key)
        if memory_entry and memory_entry.get("history"): return memory_entry["history"][-1]
    return None

# ==============================================================================
# Base Agent (formerly app/agents/base_agent.py)
# ==============================================================================
logger = logging.getLogger("perf_ai")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class BaseAgent:
    def __init__(self, session_id: str, context: Dict[str, Any]):
        self.session_id = session_id
        self.context = context
    def run(self) -> Dict[str, Any]:
        raise NotImplementedError("Agent must implement run()")

# ==============================================================================
# JMeter Parser (formerly app/tools/jmeter_parser.py)
# ==============================================================================
def parse_jmeter_csv(path: str) -> Dict[str, Any]:
    logger.info(f"PARSER: Starting to read CSV file at {path}...")
    df = pd.read_csv(path)
    logger.info(f"PARSER: Successfully read {len(df)} rows.")
    df.columns = [c.strip() for c in df.columns]
    if '# Samples' in df.columns and '95% Line' in df.columns:
        logger.info("PARSER: Detected JMeter Aggregate Report format.")
        df.rename(columns={'Label': 'label', 'Average': 'elapsed_mean', '95% Line': 'elapsed_p95', 'Error %': 'error_rate_pct'}, inplace=True)
        df['error_rate'] = pd.to_numeric(df['error_rate_pct'].str.replace('%', '')) / 100
        return {"type": "summary", "df": df}
    else:
        logger.info("PARSER: Detected raw JMeter JTL format.")
        if "timeStamp" in df.columns:
            df["timeStamp"] = pd.to_datetime(df["timeStamp"], unit='ms')
        summary_df = df.describe(include='all')
        json_safe_summary = json.loads(summary_df.to_json(default_handler=str))
        return {"type": "raw", "df": df, "summary": json_safe_summary}

# ==============================================================================
# Log Ingestion Agent (formerly app/agents/ingestion_agent.py)
# ==============================================================================
class LogIngestionAgent(BaseAgent):
    def __init__(self, session_id: str, context: Dict[str, Any], file_path: str):
        super().__init__(session_id, context)
        self.file_path = file_path
    def run(self) -> Dict[str, Any]:
        logger.info(f"Ingesting file for session {self.session_id}: {self.file_path}")
        if not self.file_path.lower().endswith('.csv'):
            message = "Invalid file type. Please upload a CSV file."
            logger.error(f"Ingestion failed for session {self.session_id}: {message}")
            return {"status": "error", "message": message}
        try:
            parsed = parse_jmeter_csv(self.file_path)
            # FIX: Removed 'storage.' prefix
            store_artifact(self.session_id, "raw_dataset", parsed)
            update_session_status(self.session_id, "ingested")
            logger.info(f"INGESTION AGENT [{self.session_id}]: Ingestion successful.")
            if parsed.get("type") == "summary":
                return {"status": "ok", "rows": len(parsed["df"]), "summary": "Aggregate Report"}
            else:
                return {"status": "ok", "rows": len(parsed["df"]), "summary": parsed["summary"]}
        except Exception as e:
            logger.error(f"Failed to ingest file {self.file_path}: {e}")
            # FIX: Removed 'storage.' prefix
            update_session_status(self.session_id, "ingestion_failed")
            return {"status": "error", "message": str(e)}

# ==============================================================================
# Metric Analyzer Agent (formerly app/agents/metric_agent.py)
# ==============================================================================
def save_fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

class MetricAnalyzerAgent(BaseAgent):
    def run(self) -> Dict[str, Any]:
        logger.info(f"Analyzing metrics for session {self.session_id}")
        # FIX: Removed 'storage.' prefix
        session = get_session(self.session_id)
        parsed = session["artifacts"].get("raw_dataset")
        if not parsed:
            return {"status": "error", "message": "No raw dataset found in session."}
        
        if parsed.get("type") == "summary":
            logger.info("Analyzing metrics from Aggregate Report summary.")
            df_summary = parsed["df"]
            results = {}
            total_row = df_summary[df_summary['label'].str.contains('TOTAL', case=False)]
            if not total_row.empty:
                total_metrics = total_row.iloc[0]
                results["elapsed_mean"] = float(total_metrics.get("elapsed_mean", 0))
                results["elapsed_p95"] = float(total_metrics.get("elapsed_p95", 0))
                results["error_rate"] = float(total_metrics.get("error_rate", 0))
                results["tps_mean"] = float(total_metrics.get("Throughput", 0))
            slowest_transactions = df_summary.sort_values(by='elapsed_p95', ascending=False).head(5)
            results["slowest_transactions"] = [{"label": row["label"], "p95_ms": int(row["elapsed_p95"])} for index, row in slowest_transactions.iterrows()]
            # FIX: Removed 'storage.' prefix
            store_artifact(self.session_id, "metrics", results)
        else:
            df = parsed["df"].copy()
            results = {}
            if "elapsed" in df.columns:
                fig = plt.figure(figsize=(10, 6))
                df["elapsed"].hist(bins=50)
                plt.title("Response Time Distribution (ms)")
                img_b64 = save_fig_to_base64(fig)
                # FIX: Removed 'storage.' prefix
                store_artifact(self.session_id, "plot_elapsed_hist_b64", img_b64)
                results["elapsed_mean"] = round(float(df["elapsed"].mean()), 2)
                results["elapsed_p95"] = round(float(df["elapsed"].quantile(0.95)), 2)
            if "elapsed" in df.columns and "label" in df.columns:
                slowest_transactions = df.groupby('label')['elapsed'].quantile(0.95).sort_values(ascending=False).head(5)
                results["slowest_transactions"] = [{"label": label, "p95_ms": round(p95_time)} for label, p95_time in slowest_transactions.items()]
            if "timeStamp" in df.columns:
                df.set_index("timeStamp", inplace=True)
                tps = df.resample("1s").size().rename("tps")
                results["tps_mean"] = round(float(tps.mean()), 2)
                fig = plt.figure(figsize=(10, 4))
                tps.plot()
                plt.title("Throughput (requests/sec)")
                img_tps_b64 = save_fig_to_base64(fig)
                # FIX: Removed 'storage.' prefix
                store_artifact(self.session_id, "plot_tps_b64", img_tps_b64)
            if "success" in df.columns:
                df["success_bool"] = df["success"].astype(str).str.lower().isin(["true", "1", "t"])
                results["error_rate"] = round(1.0 - df["success_bool"].mean(), 4)
            # FIX: Removed 'storage.' prefix
            store_artifact(self.session_id, "metrics", results)

        logger.info(f"Comparing with previous run for session {self.session_id}")
        # FIX: Removed 'storage.' prefix
        previous_run = get_last_run_from_memory("default")
        comparison_results = {}
        if previous_run and "metrics" in previous_run:
            prev_metrics = previous_run["metrics"]
            current_metrics = get_session(self.session_id)["artifacts"]["metrics"]
            comparison_results = {
                "previous_p95": prev_metrics.get("elapsed_p95"),
                "p95_delta": current_metrics.get("elapsed_p95", 0) - prev_metrics.get("elapsed_p95", 0),
                "previous_error_rate": prev_metrics.get("error_rate"),
                "error_rate_delta": current_metrics.get("error_rate", 0) - prev_metrics.get("error_rate", 0),
                "previous_tps_mean": prev_metrics.get("tps_mean"),
                "tps_mean_delta": current_metrics.get("tps_mean", 0) - prev_metrics.get("tps_mean", 0)
            }
        # FIX: Removed 'storage.' prefix
        store_artifact(self.session_id, "comparison", comparison_results)
        return {"status": "ok", "metrics": get_session(self.session_id)["artifacts"]["metrics"]}

# ==============================================================================
# Root Cause Agent (formerly app/agents/rc_agent.py)
# ==============================================================================
def call_llm(prompt: str) -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "LLM_DISABLED: Mocked root cause."
    try:
        genai.configure(api_key=api_key)
        # FIX: Corrected model name
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"LLM_ERROR: {e}"

class RootCauseAgent(BaseAgent):
    def run(self) -> Dict[str, Any]:
        logger.info(f"Computing root cause for session {self.session_id}")
        # FIX: Removed 'storage.' prefix
        session = get_session(self.session_id)
        metrics = session["artifacts"].get("metrics", {})
        comparison = session["artifacts"].get("comparison", {})
        parsed = session["artifacts"].get("raw_dataset")
        df = parsed.get("df")
        findings = []
        if metrics.get("elapsed_p95", 0) > 2000:
            findings.append(f"High p95 latency of {metrics['elapsed_p95']:.0f}ms detected.")
        if metrics.get("error_rate", 0) > 0.05:
            findings.append(f"High error rate of {metrics['error_rate']:.2%} detected.")
        prompt = f"""
You are an expert Performance Engineer. Analyze the following test results.
**Key Automated Findings:**
- {"- ".join(findings) if findings else "No significant issues detected."}
**Detailed Metrics Summary:**
```json
{json.dumps(metrics, indent=2)}
```

**Analysis Request:**
1.  Synthesize the findings and metrics into a concise executive summary (2-3 sentences).
2.  Identify the top 2 most likely root causes for any performance issues.
3.  For each root cause, provide a brief, actionable recommendation for a developer or SRE to investigate.
4. Specifically analyze the 'Top 5 Slowest Transactions' and suggest possible reasons for their high response times (e.g., inefficient database query, external API call, complex business logic).
5. Based on the 'Comparison with Previous Run' data, comment on any significant performance regressions (e.g., increased p95 latency, higher error rate) or improvements. If a regression is detected, suggest it as a potential primary root cause.

Format your response in Markdown.
"""
        logger.info("Sending prompt to LLM for synthesis.")
        
        # Run the synchronous, blocking LLM call in a threadpool to avoid freezing the server
        llm_output = call_llm(prompt)

        result = {
            "heuristic_findings": findings,
            "llm_synthesis": llm_output
        }
       
        return {"status": "ok", "root_cause": result}



from IPython.display import display, HTML, FileLink

# --- Configuration ---
# This path has been updated to point to your uploaded data file.
FILE_PATH = '/kaggle/input/dataset/aggregate_3010_500.csv' 

# --- Main Execution Logic ---
def run_full_pipeline():
    # 1. Create a session
    session_id = create_session(metadata={"source": "Kaggle Notebook"})
    print(f"PIPELINE: Session created with ID: {session_id}\n")

    # 2. Run Ingestion Agent
    update_session_status(session_id, "ingesting")
    ingestion_agent = LogIngestionAgent(session_id, {}, FILE_PATH)
    ingestion_agent.run()
    print(f"PIPELINE: Ingestion complete. Status: {get_session(session_id)['status']}\n")

    # 3. Run Metric Analyzer Agent
    update_session_status(session_id, "analyzing_metrics")
    metric_agent = MetricAnalyzerAgent(session_id, {})
    metric_agent.run()
    print(f"PIPELINE: Metric analysis complete.\n")

    # 4. Run Root Cause Agent
    update_session_status(session_id, "analyzing_root_cause")
    rc_agent = RootCauseAgent(session_id, {})
    rc_agent.run()
    print(f"PIPELINE: Root cause analysis complete.\n")

    # 5. Run Report Writer Agent
    update_session_status(session_id, "generating_report")
    report_agent = ReportWriterAgent(session_id, {})
    report_agent.run()
    print(f"PIPELINE: HTML report generation complete.\n")
    
    # 6. Run PDF Generator Agent
    update_session_status(session_id, "generating_pdf")
    pdf_agent = PdfGeneratorAgent(session_id, {})
    pdf_agent.run()
    print(f"PIPELINE: PDF generation complete.\n")

    # 7. Finalize
    update_session_status(session_id, "completed")
    # Store the final metrics in the memory bank for the next run
    push_to_memory("default", get_session(session_id)['artifacts'])
    print(f"PIPELINE: Analysis successfully completed!")
    
    return session_id

# --- Execute and Display ---
try:
    final_session_id = run_full_pipeline()
    final_session = get_session(final_session_id)
    final_html_report = final_session['artifacts']['report_html']
    pdf_path = final_session['artifacts']['report_pdf_path']

    print("\n\n=================================================")
    print("           PERF-AI COPILOT REPORT              ")
    print("=================================================")
    
    # Provide a downloadable link for the PDF in the output directory
    print("\nPDF report has been saved to the output directory.")
    display(FileLink(pdf_path))
    
    print("\nDisplaying HTML report below:")

    # Display the final HTML report in the notebook output
    display(HTML(final_html_report))

except FileNotFoundError:
    print("\n\n--- ERROR ---")
    print(f"Could not find the data file at: {FILE_PATH}")
    print("Please make sure you have uploaded the data and that the FILE_PATH variable is correct.")
except Exception as e:
    print(f"\n\n--- An unexpected error occurred ---")
    print(e)



