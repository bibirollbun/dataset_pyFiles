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


!pip install python-dotenv google-adk google-generativeai


!pip install google-adk


import asyncio
import json
import math
import uuid
import os
import dotenv
from typing import Any, Dict


from google.adk.agents import Agent, SequentialAgent, LlmAgent
from google.adk.models import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


ARTIFACTS_DIR = "session_artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


MODEL_NAME = "gemini-2.5-flash-lite"
APP_NAME = "log_ingestion_app"
USER_ID = "test_user"
SESSION_NAME = "ingestion_session"
LOG_FILE_PATH = "/kaggle/input/logs-data/synthetic_nodejs_logs.json"
PIPELINE_STATE = {
    "latest_log_artifact": None,
    "latest_feature_artifact": None
}


retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)


def ingest_logs(log_file: str) -> str:
    """
    Ingests logs from a source file, saves them to session storage, 
    and returns the storage path.
    """
    # Handle the input path
    base_path = os.getcwd() # Gets the current folder where python is running
    full_artifact_dir = os.path.join(base_path, ARTIFACTS_DIR)
    source_path = LOG_FILE_PATH if "synthetic_nodejs_logs.json" in log_file else log_file
    
    try:
        with open(source_path, 'r') as f:
            raw_data = json.load(f)

        if isinstance(raw_data, list):
            logs = raw_data
        else:
            logs = [hit['_source'] for hit in raw_data.get('hits', {}).get('hits', [])]

        output_filename = f"ingested_logs_output.json"
        output_path = os.path.join(full_artifact_dir, output_filename)
    
        
        with open(output_path, 'w') as f_out:
            json.dump(logs, f_out, indent=2)

        PIPELINE_STATE["latest_log_artifact"] = output_path
        count = len(logs)
        print(f"DEBUG: ğŸ“‚ File saved to -> {output_path}")
        return f"âœ… Success. Ingested {count} logs. Data saved to artifact: '{output_path}'"

    except FileNotFoundError:
        return f"Error: Source file not found at {source_path}"
    except Exception as e:
        return f"Error processing logs: {str(e)}"


def create_log_ingestor_agent(model: Gemini) -> Agent:
    """An agent that ingests and processes log data from various sources."""
    return Agent(
        model=model,
        name="log_ingestor_agent",
        description="Ingests and processes log data from various sources.",
        tools=[FunctionTool(ingest_logs)],
    )


def extract_features() -> str:
    """
    Reads ingested logs and calculates features (Error Rate, 5xx count).
    Args: None
    Returns: str - Status message with feature file path or error.
    """
    target_path = PIPELINE_STATE.get("latest_log_artifact")
    print(f"DEBUG: ğŸ“¥ Preprocessor reading global state -> {target_path}")
    if not target_path:
        return "â�Œ Error: 'latest_log_artifact' is missing from global state. Did the Ingestor run?"
    
    if not os.path.exists(target_path):
        return f"â�Œ Error: File found in state but missing on disk at {target_path}"
    
    try:

        with open(target_path, 'r') as f:
            logs = json.load(f)

        total_logs = len(logs)
        error_count = 0
        warning_count = 0
        http_5xx = 0

        for log in logs:
            level = log.get("log", {}).get("level", "INFO")
            status = log.get("http", {}).get("response", {}).get("status_code", 200)

            if level == "ERROR":
                error_count += 1
            if level == "WARN":
                warning_count += 1
            if status and int(status) >= 500:
                http_5xx += 1

        error_rate = (error_count / total_logs) * 100 if total_logs > 0 else 0

        features = {
            "total_logs": total_logs,
            "error_count": error_count,
            "warning_count": warning_count,
            "http_5xx_count": http_5xx,
            "error_rate_percent": round(error_rate, 2),
            "is_anomaly_suspect": error_rate > 5.0
        }
        base_dir = os.path.dirname(target_path)
        feature_file = os.path.join(base_dir, "features.json")
        print(f"DEBUG: Calculated Features: {features}")
        print(f"DEBUG: Saving features to -> {feature_file}")
        PIPELINE_STATE["latest_feature_artifact"] = feature_file
        with open(feature_file, 'w') as f_out:
            json.dump(features, f_out, indent=2)

        PIPELINE_STATE["latest_feature_artifact"] = feature_file
        print(f"DEBUG: ğŸ“Š Features saved to -> {feature_file}")
        return f"Feature extraction complete. Metrics saved to: {feature_file}"

    except Exception as e:
        return f"Error during preprocessing: {str(e)}"


def create_preprocessor_agent(model: Gemini) -> Agent:
    """An agent that takes raw logs and outputs feature vectors."""
    return Agent(
        model=model,
        name="preprocessor_agent",
        description="Reads file paths, calculates error rates/stats, and saves feature files.",
        tools=[FunctionTool(extract_features)],
    )


def fetch_metrics_for_analysis() -> str:
    """
    Retrieves the calculated feature set from the global state so the LLM can analyze it.
    """
    feature_path = PIPELINE_STATE.get("latest_feature_artifact")
    print(f"DEBUG: ğŸ”� Fetching metrics from -> {feature_path}")

    if not feature_path or not os.path.exists(feature_path):
        return "â�Œ Error: No feature file found."

    try:
        with open(feature_path, 'r') as f:
            features = json.load(f)
        return json.dumps(features, indent=2)

    except Exception as e:
        return f"â�Œ Error fetching metrics: {str(e)}"


def create_anomaly_detector_agent(model: Gemini) -> Agent:
    """
    Creates an agent acting as a Site Reliability Engineer (SRE).
    It uses the LLM to interpret metrics rather than hard-coded rules.
    """
    sre_instructions = """
    You are a Senior Site Reliability Engineer (SRE). 
    Your job is to analyze system metrics and decide if the system is healthy or failing.
    
    1. Call the tool 'fetch_metrics_for_analysis' to get the latest data.
    2. Analyze the JSON output, paying close attention to 'error_rate_percent' and 'http_5xx_count'.
    3. Use your judgment:
       - If error rates are negligible (near 0%) and 5xx count is 0, report: "âœ… SYSTEM NORMAL".
       - If you see critical failures (any 5xx errors) or high error rates (>3%), report: "ğŸš¨ ANOMALY DETECTED".
    4. Briefly explain your reasoning based on the data provided.
    """
    
    return LlmAgent(
        model=model,
        name="anomaly_detector",
        description="Analyzes metrics using LLM reasoning to detect complex anomalies.",
        instruction=sre_instructions,
        tools=[FunctionTool(fetch_metrics_for_analysis)],
    )


def predict_failure_risk() -> str:
    """
    Uses a Logistic Regression simulation to calculate failure probability
    based on weighted features (Error Rate, 5xx Counts, Warnings).
    """
    # 1. Load Features
    feature_path = PIPELINE_STATE.get("latest_feature_artifact")
    
    if not feature_path or not os.path.exists(feature_path):
        return "â�Œ Error: No feature artifacts found."

    try:
        with open(feature_path, 'r') as f:
            features = json.load(f)

        # 2. Extract Metrics
        error_rate = features.get("error_rate_percent", 0)
        http_5xx = features.get("http_5xx_count", 0)
        warning_count = features.get("warning_count", 0)

        # 3. WEIGHTED SCORING (The "Model")
        # These weights mimic a trained ML model's coefficients
        # Bias: Negative value ensures base probability is low when errors are 0
        w_bias = -4.0  
        w_error_rate = 0.6  # High impact per percentage point
        w_5xx = 1.5         # Very high impact per occurrence
        w_warning = 0.05    # Low impact

        # Calculate Logit (Raw Risk Score)
        logit = w_bias + (w_error_rate * error_rate) + (w_5xx * http_5xx) + (w_warning * warning_count)

        # Formula: P = 1 / (1 + e^-logit)
        failure_prob = 1 / (1 + math.exp(-logit))
        failure_prob = round(failure_prob, 4)


        risk_contributors = {
            "Cascading Service Failure": (w_error_rate * error_rate),
            "Database Connection Timeout": (w_5xx * http_5xx),
            "Performance Degradation": (w_warning * warning_count)
        }
        # Find the max contributor
        predicted_mode = max(risk_contributors, key=risk_contributors.get)
        
        # If the risk is very low, the mode is "None"
        if failure_prob < 0.2:
            predicted_mode = "Stable State"

        # 6. Create Artifact
        prediction_result = {
            "timestamp": "2025-11-23T09:30:00Z",
            "model_type": "LogisticRegression_Sim",
            "metrics": {
                "error_rate": error_rate,
                "5xx_count": http_5xx
            },
            "failure_probability": failure_prob,
            "risk_score_logit": round(logit, 2),
            "risk_level": "CRITICAL" if failure_prob > 0.7 else ("WARNING" if failure_prob > 0.4 else "HEALTHY"),
            "predicted_failure_mode": predicted_mode,
            "time_horizon": "15 minutes"
        }

        # Save and Return
        base_dir = os.path.dirname(feature_path)
        prediction_file = os.path.join(base_dir, "prediction.json")
        
        with open(prediction_file, 'w') as f_out:
            json.dump(prediction_result, f_out, indent=2)

        PIPELINE_STATE["latest_prediction_artifact"] = prediction_file
        
        return json.dumps(prediction_result, indent=2)

    except Exception as e:
        return f"â�Œ Prediction model failed: {str(e)}" 


def create_predictor_agent(model: Gemini) -> Agent:
    instructions = """
        You are the 'Predictor'. Your role is to forecast system stability.
        1. Call 'predict_failure_risk' to run the ML inference.
        2. Report the 'failure_probability' and 'predicted_failure_mode'.
        3. If probability > 0.7, warn that a critical incident is imminent.
        """
    return LlmAgent(
        model=model,
        name="predictor_agent",
        description="Forecasting agent that uses ML models to predict future system states.",
        instruction=instructions,
        tools=[FunctionTool(predict_failure_risk)],
    )


def perform_root_cause_analysis() -> str:
    """
    Aggregates all incident artifacts and retrieves similar past incidents 
    to facilitate Root Cause Analysis.
    """
    print("DEBUG: ğŸ•µï¸�â€�â™€ï¸� RCA Agent gathering evidence...")

    artifacts = {}
    evidence_sources = [
        ("logs", "latest_log_artifact"),
        ("features", "latest_feature_artifact"),
        ("prediction", "latest_prediction_artifact")
    ]

    for label, key in evidence_sources:
        path = PIPELINE_STATE.get(key)
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                # Load data (truncate logs if too large for context window)
                data = json.load(f)
                if label == "logs":
                    # Take only error logs or last 10 logs to save tokens
                    artifacts[label] = [l for l in data if l.get('log', {}).get('level') == 'ERROR'][:10]
                else:
                    artifacts[label] = data
    
    # Simulate "Memory Bank" Retrieval (Vector DB Search) 
    # In production, this would be: vector_db.query(embedding=current_error_embedding)
    past_incidents = [
        {
            "id": "INC-2023-001",
            "similarity": "98%",
            "description": "High 5xx errors caused by Database Connection Pool exhaustion.",
            "resolution": "Increased max_pool_size in payment-service config."
        },
        {
            "id": "INC-2024-045",
            "similarity": "85%",
            "description": "Latency spike due to unindexed MongoDB query on 'user_id'.",
            "resolution": "Added index to user collection."
        }
    ]

    # 3. Construct the Analysis Payload for the LLM
    analysis_payload = {
        "current_incident_evidence": artifacts,
        "similar_past_incidents": past_incidents,
        "instructions": "Correlate current evidence with past incidents to determine root cause."
    }

    return json.dumps(analysis_payload, indent=2)


def generate_remediation_plan(previous_output: str = "") -> str:
    """
    Reads the RCA report and maps the root cause to specific remediation actions.
    """
    print("DEBUG: ğŸ› ï¸� Remediation Agent generating action plan...")

    # 1. Get RCA Output
    # (In a real app, we'd save RCA to a file artifact, but for now we trust the flow or use state)
    # Let's assume the previous agent passed the JSON string, or we use a new state key if we saved it.
    # For this PoC, we'll simulate reading the "latest_rca_context" if we had saved it, 
    # but relying on the 'previous_output' string from the RCA agent is also fine for the Sequential chain.
    rca_text = previous_output
    
    # 2. Define the Action Map (The "Runbook")
    # This maps specific failure keywords to concrete commands
    action_map = {
        "Database Connection Pool": {
            "action": "scale_up_replicas",
            "target": "payment-db-shard-01",
            "command": "kubectl scale statefulset payment-db --replicas=5",
            "risk": "HIGH"
        },
        "Cache": {
            "action": "flush_cache",
            "target": "redis-cluster",
            "command": "redis-cli flushall",
            "risk": "LOW"
        },
        "Latency": {
            "action": "restart_pod",
            "target": "payment-service",
            "command": "kubectl rollout restart deployment/payment-service",
            "risk": "MEDIUM"
        }
    }
    
    # 3. Determine Actions
    plan = []
    
    # Check if RCA identified a specific cause
    detected_cause = "Unknown"
    for key in action_map:
        if key.lower() in rca_text.lower():
            detected_cause = key
            step = action_map[key]
            
            # Logic: Auto-execute Low Risk, Flag High Risk
            status = "AUTO_EXECUTED" if step["risk"] == "LOW" else "PENDING_APPROVAL"
            
            plan.append({
                "step": 1,
                "action": step["action"],
                "command": step["command"],
                "risk_level": step["risk"],
                "status": status
            })
    
    if not plan:
        # Fallback for unknown issues
        plan.append({
            "step": 1,
            "action": "notify_on_call",
            "command": "pagerduty trigger --title 'Unknown System Failure'",
            "risk_level": "LOW",
            "status": "AUTO_EXECUTED"
        })

    # 4. Construct Final Report
    remediation_report = {
        "timestamp": "2025-11-23T09:35:00Z",
        "root_cause_category": detected_cause,
        "action_plan": plan
    }

    # Save to state for the Notifier
    # (We can create a mock file or just print it)
    print(f"DEBUG: ğŸ“� Remediation Plan Created: {json.dumps(plan, indent=2)}")
    
    return json.dumps(remediation_report, indent=2)



def create_remediation_agent(model: Gemini) -> Agent:
    devops_instructions = """
    You are a Lead DevOps Engineer. Your goal is to fix the system stability issue identified by the RCA Agent.
    
    1. Call 'generate_remediation_plan' to map the root cause to technical commands.
    2. Review the plan.
    3. Output a structured response that lists:
       - The Actions to be taken.
       - Which actions were Auto-Executed (Low Risk).
       - Which actions require Human Approval (High Risk).
    """

    return LlmAgent(
        model=model,
        name="remediation_agent",
        description="Maps root causes to specific infrastructure remediation commands.",
        instruction=devops_instructions,
        tools=[FunctionTool(generate_remediation_plan)],
    )


def create_rca_agent(model: Gemini) -> Agent:
    rca_instructions = """
    You are a Principal Incident Commander. Your goal is to identify the Root Cause of the ongoing system failure.
    
    1. Call 'perform_root_cause_analysis' to get the Evidence Bag.
    2. Review the 'current_incident_evidence' (Logs, Metrics, Prediction).
    3. Compare it against 'similar_past_incidents' to find patterns.
    4. Output a structured RCA Report containing in text format:
       - **Root Cause Hypothesis**: What exactly broke?
       - **Evidence**: Which specific log/metric proves it?
       - **Confidence Score**: 0-100% based on past incident similarity.
       - **Recommended Fix**: What should the engineer do? (Base this on the past resolutions).
    """

    return LlmAgent(
        model=model,
        name="rca_agent",
        description="Correlates logs, metrics, and past incidents to determine root cause.",
        instruction=rca_instructions,
        tools=[FunctionTool(perform_root_cause_analysis)],
    )


async def get_orchestrator_agent() -> SequentialAgent:
    """An agent that orchestrates various sub-agents to perform complex tasks."""
    
    # Initialize Model
    orchestrator_model = Gemini(
        model=MODEL_NAME,
        retry_options=retry_config,
    )

    # Initialize Sub-Agent
    log_ingestor = create_log_ingestor_agent(model=orchestrator_model)
    preprocessor = create_preprocessor_agent(model=orchestrator_model)
    detector = create_anomaly_detector_agent(model=orchestrator_model)
    predictor = create_predictor_agent(model=orchestrator_model)
    rca = create_rca_agent(model=orchestrator_model)
    remediation = create_remediation_agent(model=orchestrator_model)

    return SequentialAgent(
        name="capstone_orchestrator",
        sub_agents=[log_ingestor, preprocessor, detector,predictor, rca, remediation]
    )


async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str = None,
    session_name: str = "default",
    session_service: InMemorySessionService = None,
):
    print(f"\n ### Session: {session_name}")

    # Get app name from the Runner
    app_name = runner_instance.app_name

    # Attempt to create a new session or retrieve an existing one
    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    except Exception as e:
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )

    if user_queries:
        if isinstance(user_queries, str):
            user_queries = [user_queries]

        for query in user_queries:
            print(f"\nğŸ‘¤ User > {query}")
            query_content = types.Content(role="user", parts=[types.Part(text=query)])

            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query_content
            ):
                # Handle the content parts safely
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        # Case 1: It's a Function Call (The Agent is acting)
                        if part.function_call:
                            print(f"   ğŸ› ï¸�  [Agent Tool Call] Action: {part.function_call.name}")
                        
                        # Case 2: It's a Text Response (The Agent is speaking)
                        elif part.text:
                             # Filter out empty/system strings
                            if part.text != "None" and part.text.strip():
                                print(f"ğŸ¤– {MODEL_NAME} > {part.text}")
    else:
        print("No queries!")


async def get_orchestrator_agent() -> SequentialAgent:
    """An agent that orchestrates various sub-agents to perform complex tasks."""
    
    # Initialize Model
    orchestrator_model = Gemini(
        model=MODEL_NAME,
        retry_options=retry_config,
    )

    # Initialize Sub-Agent
    log_ingestor = create_log_ingestor_agent(model=orchestrator_model)
    preprocessor = create_preprocessor_agent(model=orchestrator_model)
    detector = create_anomaly_detector_agent(model=orchestrator_model)
    predictor = create_predictor_agent(model=orchestrator_model)
    rca = create_rca_agent(model=orchestrator_model)
    remediation = create_remediation_agent(model=orchestrator_model)

    return SequentialAgent(
        name="capstone_orchestrator",
        sub_agents=[log_ingestor, preprocessor, detector,predictor, rca, remediation]
    )


print("ğŸ§ª Initializing Log Ingestor System...")

session_service = InMemorySessionService()    

root_agent = await get_orchestrator_agent()

runner = Runner(
    app_name=APP_NAME,
    agent=root_agent, 
    session_service=session_service
)

# user_query = ""
# user_query = "Please Extract key features like error rates and 5xx counts."
# user_query = (
#     "Please ingest the logs from 'synthetic_nodejs_logs.json' and extract key features like error rates and 5xx counts without asking to continue"
#     "and then analyze those metrics for anomalies "
#     "Execute the full pipeline immediately without asking for confirmation between steps."
# )
user_query = "Please ingest the logs from 'synthetic_nodejs_logs.json'"


await run_session(
    runner,
    user_queries=[user_query],
    session_name=SESSION_NAME,
    session_service=session_service,
)

print("âœ… Test completed.")







