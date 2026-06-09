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


# ============================================
# ASTRA GUARD - INCIDENT RESPONSE DEMO PROJECT
# Multi-"Agent" SRE / DevOps Troubleshooter
# ============================================

# --------- 1. Imports ---------
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
import json

import yaml

from textwrap import indent


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


# --------- 2. Create Synthetic Demo Data ---------
# Simple Incident: DB connection issue (logs + config + internal guidelines)

simple_log = """
2025-11-25 10:01:01,123 INFO  Starting web application...
2025-11-25 10:01:02,456 INFO  Connecting to database at db_host=localhost, db_port=5432
2025-11-25 10:01:05,789 ERROR psycopg2.OperationalError: could not connect to server: Connection refused
        Is the server running on host "localhost" (127.0.0.1) and accepting
        TCP/IP connections on port 5432?
2025-11-25 10:01:06,001 WARNING Retrying database connection (1/3)
2025-11-25 10:01:09,230 ERROR psycopg2.OperationalError: could not connect to server: Connection refused
2025-11-25 10:01:12,450 WARNING Retrying database connection (2/3)
2025-11-25 10:01:15,701 ERROR psycopg2.OperationalError: could not connect to server: Connection refused
2025-11-25 10:01:18,900 CRITICAL Application startup failed due to database connectivity issues.
"""

simple_config_yaml = """
app:
  name: sample-web-service
  env: production

database:
  host: localhost
  port: 5432
  name: prod_db
  user: app_user
  password: secret_password

logging:
  level: INFO
"""

infra_guidelines_md = """
# Database & Infrastructure Guidelines

For production deployments, applications **must not** connect to a local database instance.
Instead, they must use the managed database endpoint.

- Production DB host: `prod-db.mycompany.internal`
- Port: `5432`
- Connections must go over the private VPC only.

OOMKilled troubleshooting:

- OOMKilled events typically indicate that the container's memory limit is too low.
- Recommended mitigation: increase memory limits and/or tune application heap usage.
"""

Path("simple_incident_logs.log").write_text(simple_log)
Path("simple_incident_config.yaml").write_text(simple_config_yaml)
Path("infra_guidelines.md").write_text(infra_guidelines_md)


# Complex Incident: OOM + CrashLoopBackOff (app log + k8s events + deployment config)

complex_app_log = """
2025-11-25T09:00:01Z INFO  Starting service order-matcher...
2025-11-25T09:00:12Z INFO  Warm-up complete, accepting traffic.
2025-11-25T09:05:34Z ERROR java.lang.OutOfMemoryError: Java heap space
2025-11-25T09:05:34Z ERROR at com.mycompany.matching.Engine.run(Engine.java:123)
2025-11-25T09:05:34Z ERROR at java.base/java.lang.Thread.run(Thread.java:834)
2025-11-25T09:05:35Z WARN  Instance becoming unresponsive, latency p95 > 5000ms
2025-11-25T09:05:40Z ERROR Healthcheck failed: /health returned 500
"""

k8s_events_log = """
LAST SEEN   TYPE      REASON             MESSAGE
9m          Normal    Scheduled          Successfully assigned prod/order-matcher-7db9c6bd4d-h7x9z to node-3
4m          Warning   OOMKilled          Container order-matcher terminated by OOMKiller
4m          Normal    Killing            Stopping container order-matcher
3m          Normal    Pulling            Pulling image "registry.myco.internal/order-matcher:v2.1.0"
2m          Normal    Started            Started container order-matcher
1m          Warning   BackOff            Back-off restarting failed container
"""

complex_config_yaml = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-matcher
spec:
  replicas: 2
  template:
    metadata:
      labels:
        app: order-matcher
    spec:
      containers:
        - name: order-matcher
          image: registry.myco.internal/order-matcher:v2.1.0
          resources:
            requests:
              memory: "128Mi"
              cpu: "100m"
            limits:
              memory: "128Mi"
              cpu: "200m"
          env:
            - name: JAVA_OPTS
              value: "-Xms64m -Xmx96m"
"""

Path("complex_app.log").write_text(complex_app_log)
Path("k8s_events.log").write_text(k8s_events_log)
Path("complex_deployment.yaml").write_text(complex_config_yaml)


# --------- 3. Generic Helpers & "Tools" ---------

def load_text(path: str) -> str:
    return Path(path).read_text()


def load_yaml(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def extract_errors_from_log(log_text: str) -> List[str]:
    lines = log_text.splitlines()
    error_lines = [line for line in lines if "ERROR" in line or "CRITICAL" in line]
    return error_lines


def detect_common_error_signatures(log_text: str) -> Dict[str, int]:
    """
    Very simple 'signature frequency' counter, just to illustrate pattern detection.
    """
    signatures = {
        "connection refused": 0,
        "OutOfMemoryError": 0,
        "OOMKilled": 0,
        "CrashLoopBackOff": 0,
        "timeout": 0,
        "healthcheck failed": 0,
        " 500": 0,  # crude check for ' 500'
    }
    lower = log_text.lower()
    for key in signatures.keys():
        signatures[key] = lower.count(key.lower())
    return {k: v for k, v in signatures.items() if v > 0}


def extract_time_window(log_text: str, context_lines: int = 3) -> str:
    """
    Extract some context around the first ERROR/CRITICAL in the log.
    """
    lines = log_text.splitlines()
    idx = None
    for i, line in enumerate(lines):
        if "ERROR" in line or "CRITICAL" in line:
            idx = i
            break
    if idx is None:
        return ""
    start = max(0, idx - context_lines)
    end = min(len(lines), idx + context_lines + 1)
    return "\n".join(lines[start:end])


def check_db_config(config: dict) -> Dict[str, str]:
    """
    For the simple incident: check if DB host is localhost, which is wrong for prod.
    """
    issues: Dict[str, str] = {}
    db = config.get("database") or {}
    host = db.get("host")
    if host in ("localhost", "127.0.0.1"):
        issues["db_host"] = (
            f"Database host is '{host}' in production; should use a managed remote endpoint."
        )
    return issues


def check_k8s_resources(config: dict) -> Dict[str, str]:
    """
    For the complex incident: check if memory limits look suspiciously low.
    """
    issues: Dict[str, str] = {}
    try:
        containers = config["spec"]["template"]["spec"]["containers"]
        for c in containers:
            name = c.get("name", "unknown")
            mem_limit = c.get("resources", {}).get("limits", {}).get("memory")
            if mem_limit and mem_limit.endswith("Mi"):
                val = int(mem_limit.replace("Mi", ""))
                if val <= 128:
                    issues[f"{name}_memory"] = (
                        f"Memory limit {mem_limit} may be too low for a JVM-based service."
                    )
    except Exception as e:
        issues["config_parse"] = f"Error while checking k8s resources: {e}"
    return issues


def simple_rag_search(query: str, paths: List[str]) -> str:
    """
    Extremely simple local 'RAG': returns lines containing the query terms
    from a set of local markdown/text docs.
    """
    query_lower = query.lower()
    results: List[str] = []
    for path in paths:
        text = Path(path).read_text()
        for line in text.splitlines():
            if query_lower in line.lower():
                results.append(f"[{path}] {line}")
    return "\n".join(results[:10])



# --------- 4. "Agent" Data Classes ---------

@dataclass
class LogAnalysisResult:
    error_lines: List[str]
    signatures: Dict[str, int]
    context_snippet: str
    summary: str


@dataclass
class ConfigAnalysisResult:
    issues: Dict[str, str]
    summary: str


@dataclass
class RAGResult:
    query: str
    matches: str
    summary: str




# --------- 5. "Agent" Implementations ---------

def log_analyzer_agent(log_path: str) -> LogAnalysisResult:
    text = load_text(log_path)
    errors = extract_errors_from_log(text)
    sigs = detect_common_error_signatures(text)
    context = extract_time_window(text)

    # crude textual summary based on signatures
    sig_keys_lower = [k.lower() for k in sigs.keys()]
    if any("connection refused" in k for k in sig_keys_lower):
        summary = "Frequent database connection failures (connection refused)."
    elif any("outofmemoryerror" in k for k in sig_keys_lower):
        summary = "Application appears to be running out of memory (OutOfMemoryError)."
    elif any("oomkilled" in k for k in sig_keys_lower):
        summary = "Kubernetes OOMKilled events observed for this container."
    elif any(" 500" in k for k in sig_keys_lower):
        summary = "HTTP 500 errors detected, service health is degraded."
    else:
        summary = "Errors detected, but no dominant known pattern."

    return LogAnalysisResult(
        error_lines=errors,
        signatures=sigs,
        context_snippet=context,
        summary=summary,
    )


def config_auditor_agent(config_path: str, kind: str = "simple") -> ConfigAnalysisResult:
    if config_path.endswith((".yaml", ".yml")):
        cfg = load_yaml(config_path)
    elif config_path.endswith(".json"):
        cfg = load_json(config_path)
    else:
        return ConfigAnalysisResult(issues={}, summary="Unsupported config format.")

    if kind == "simple":
        issues = check_db_config(cfg)
    else:
        issues = check_k8s_resources(cfg)

    if not issues:
        summary = "No obvious misconfiguration found."
    else:
        summary = "Potential misconfigurations detected: " + "; ".join(issues.values())

    return ConfigAnalysisResult(issues=issues, summary=summary)


def rag_agent(query: str, docs: List[str]) -> RAGResult:
    matches = simple_rag_search(query, docs)
    if matches:
        summary = f"Retrieved guidance lines related to: '{query}'."
    else:
        summary = f"No direct matches found in docs for: '{query}'."
    return RAGResult(query=query, matches=matches, summary=summary)


# --------- 6. "Planner" / Orchestration Flows ---------

def simple_incident_flow():
    print("=== Simple Incident: DB connection failure ===")

    # 1. Log analysis
    log_result = log_analyzer_agent("simple_incident_logs.log")
    print("\n[Log Analyzer] Summary:")
    print(" ", log_result.summary)
    print("\n[Log Analyzer] Error signatures:", log_result.signatures)

    # 2. Config analysis
    cfg_result = config_auditor_agent("simple_incident_config.yaml", kind="simple")
    print("\n[Config Auditor] Summary:")
    print(" ", cfg_result.summary)

    # 3. RAG support (infra guidelines)
    rag_result = rag_agent("Production DB host", ["infra_guidelines.md"])
    print("\n[RAG Agent] Summary:")
    print(" ", rag_result.summary)
    if rag_result.matches:
        print("\n[RAG Agent] Retrieved guidance:")
        print(indent(rag_result.matches, "  "))

    # 4. "Final" RCA synthesis (this is where a higher-level agent would reason)
    rca_parts: List[str] = []
    rca_parts.append("The application fails to connect to the database (connection refused).")

    if "db_host" in cfg_result.issues:
        rca_parts.append(
            "Configuration uses 'localhost' as database host in a production environment."
        )

    if rag_result.matches:
        rca_parts.append(
            "Internal guidelines require using a managed DB endpoint instead of localhost."
        )

    final_rca = "\n- ".join(rca_parts)

    fix_plan = """
1. Update `simple_incident_config.yaml` to use the managed production DB host, for example:
   database:
     host: prod-db.mycompany.internal
     port: 5432

2. Redeploy the application with the updated configuration.

3. Verify connectivity and monitor logs to ensure there are no further connection refused errors.
"""

    prevention = """
- Avoid using `localhost` for production databases; always use managed DB endpoints.
- Add configuration linting or CI checks to block localhost DB host in production.
- Implement alerts on repeated database connection failures.
"""

    print("\n=== ROOT CAUSE ANALYSIS (Simple Incident) ===")
    print("- " + final_rca)
    print("\n=== FIX PLAN (Simple Incident) ===")
    print(fix_plan)
    print("=== PREVENTION RECOMMENDATIONS (Simple Incident) ===")
    print(prevention)


def complex_incident_flow():
    print("=== Complex Incident: OOM & CrashLoopBackOff ===")

    # 1. Log analysis (application logs)
    log_result = log_analyzer_agent("complex_app.log")
    print("\n[Log Analyzer] Summary:")
    print(" ", log_result.summary)
    print("\n[Log Analyzer] Error signatures:", log_result.signatures)
    print("\n[Log Analyzer] Context snippet:\n")
    print(indent(log_result.context_snippet, "  "))

    # 2. K8s events inspection (we reuse detect_common_error_signatures)
    k8s_text = load_text("k8s_events.log")
    k8s_sigs = detect_common_error_signatures(k8s_text)
    print("\n[K8s Events] Detected signatures:")
    print(" ", k8s_sigs)
    print("\n[K8s Events] Raw log:\n")
    print(indent(k8s_text.strip(), "  "))

    # 3. Config analysis (deployment YAML)
    cfg_result = config_auditor_agent("complex_deployment.yaml", kind="complex")
    print("\n[Config Auditor] Summary:")
    print(" ", cfg_result.summary)
    if cfg_result.issues:
        print("\n[Config Auditor] Issues:")
        for k, v in cfg_result.issues.items():
            print(f"  - {k}: {v}")

    # 4. RAG for OOMKilled / memory guidance
    rag_result = rag_agent("OOMKilled", ["infra_guidelines.md"])
    print("\n[RAG Agent] Summary:")
    print(" ", rag_result.summary)
    if rag_result.matches:
        print("\n[RAG Agent] Retrieved guidance for OOMKilled:")
        print(indent(rag_result.matches, "  "))

    # 5. Final RCA synthesis
    rca_parts: List[str] = []
    rca_parts.append("Application logs show OutOfMemoryError and degraded latency.")
    if any("oomkilled" in k.lower() for k in k8s_sigs.keys()):
        rca_parts.append("Kubernetes events indicate containers being OOMKilled and restarted.")
    if cfg_result.issues:
        rca_parts.append(
            "Deployment spec sets low memory limits, likely insufficient for this JVM-based service."
        )

    final_rca = "\n- ".join(rca_parts)

    fix_plan = """
1. Increase memory requests/limits in `complex_deployment.yaml`, for example:
   resources:
     requests:
       memory: "512Mi"
     limits:
       memory: "1Gi"

2. Update `JAVA_OPTS` to align with new limits (e.g., `-Xmx768m` or similar).
3. Redeploy the `order-matcher` deployment.
4. Monitor:
   - Pod restarts
   - JVM heap usage
   - P95 latency
   - Error rates (especially OutOfMemoryError and HTTP 500s).
"""

    prevention = """
- Establish baseline memory usage and tune memory limits with sufficient headroom.
- Add dashboards for JVM heap usage, GC pauses, and pod restarts.
- Configure alerts for OOMKilled events and repeated pod CrashLoopBackOff patterns.
- Load-test new versions of the service before rolling out to production.
"""

    print("\n=== ROOT CAUSE ANALYSIS (Complex Incident) ===")
    print("- " + final_rca)
    print("\n=== FIX PLAN (Complex Incident) ===")
    print(fix_plan)
    print("=== PREVENTION RECOMMENDATIONS (Complex Incident) ===")
    print(prevention)


simple_incident_flow()


complex_incident_flow()


# ============================================
# ADK AGENT LAYER: REAL INCIDENT COMMANDER AGENT
# ============================================

# 1) Install/import ADK 
try:
    import google.adk  # type: ignore
except ImportError:
    %pip install -q -U google-adk google-genai

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner

# --------------------------------------------
# 2) Wrap above analysis logic as TOOLS
# --------------------------------------------

def analyze_simple_incident_tool() -> dict:
    """
    Analyze the SIMPLE DB-connection incident and return a structured summary.

    This is a FUNCTION TOOL for the agent.
    It does NOT print anything; it returns structured data.

    Returns:
        {
          "incident_type": "simple-db-connection",
          "log_summary": str,
          "config_issues": [str],
          "doc_evidence": [str],
          "root_cause_hypotheses": [str],
          "suggested_fix_steps": [str],
          "prevention_ideas": [str],
          "status": "success" | "error",
          "error_message": str | None
        }
    """
    try:
        # 1. Use existing log / config / RAG helpers
        log_result = log_analyzer_agent("simple_incident_logs.log")
        cfg_result = config_auditor_agent("simple_incident_config.yaml", kind="simple")
        rag_result = rag_agent("Production DB host", ["infra_guidelines.md"])

        config_issues = list(cfg_result.issues.values())
        doc_evidence = rag_result.matches.splitlines() if rag_result.matches else []

        root_cause_hypotheses = [
            "The application cannot reach the database because the host is set to localhost in production.",
        ]
        if not config_issues:
            root_cause_hypotheses.append(
                "There may also be a networking or firewall issue if host configuration is corrected but the issue persists."
            )

        suggested_fix_steps = [
            "Change the database host from 'localhost' to the managed production endpoint (for example: prod-db.mycompany.internal).",
            "Redeploy the application with the updated configuration.",
            "Verify that the application can establish a DB connection and that the startup proceeds successfully.",
        ]

        prevention_ideas = [
            "Add CI/CD checks to prevent using 'localhost' as a DB host in production configs.",
            "Create monitoring alerts for repeated DB connection failures.",
        ]

        return {
            "incident_type": "simple-db-connection",
            "log_summary": log_result.summary,
            "config_issues": config_issues,
            "doc_evidence": doc_evidence,
            "root_cause_hypotheses": root_cause_hypotheses,
            "suggested_fix_steps": suggested_fix_steps,
            "prevention_ideas": prevention_ideas,
            "status": "success",
            "error_message": None,
        }
    except Exception as e:
        return {
            "incident_type": "simple-db-connection",
            "log_summary": "",
            "config_issues": [],
            "doc_evidence": [],
            "root_cause_hypotheses": [],
            "suggested_fix_steps": [],
            "prevention_ideas": [],
            "status": "error",
            "error_message": str(e),
        }


def analyze_complex_incident_tool() -> dict:
    """
    Analyze the COMPLEX OOM / CrashLoopBackOff incident and return structured data.

    Returns:
        {
          "incident_type": "complex-oom-k8s",
          "log_summary": str,
          "k8s_signatures": dict,
          "config_issues": [str],
          "doc_evidence": [str],
          "root_cause_hypotheses": [str],
          "suggested_fix_steps": [str],
          "prevention_ideas": [str],
          "status": "success" | "error",
          "error_message": str | None
        }
    """
    try:
        # 1. App log analysis
        app_log_result = log_analyzer_agent("complex_app.log")

        # 2. K8s events signatures
        k8s_text = load_text("k8s_events.log")
        k8s_sigs = detect_common_error_signatures(k8s_text)

        # 3. Deployment config analysis
        cfg_result = config_auditor_agent("complex_deployment.yaml", kind="complex")

        # 4. "RAG" over infra guidelines
        rag_result = rag_agent("OOMKilled", ["infra_guidelines.md"])
        doc_evidence = rag_result.matches.splitlines() if rag_result.matches else []

        config_issues = list(cfg_result.issues.values())

        root_cause_hypotheses = [
            "The service is running out of memory (OutOfMemoryError), causing Kubernetes OOMKilled events and restarts.",
            "Low memory limits in the deployment configuration likely contribute to the repeated failures.",
        ]

        suggested_fix_steps = [
            "Increase memory requests and limits in the deployment (e.g., requests: 512Mi, limits: 1Gi).",
            "Update JVM heap settings (JAVA_OPTS) to align with new memory limits.",
            "Redeploy the service and monitor pod restarts, latency and error rates.",
        ]

        prevention_ideas = [
            "Establish performance and memory baselines before pushing new versions to production.",
            "Add dashboards for JVM heap and pod restart counts.",
            "Add alerts for OOMKilled events and CrashLoopBackOff patterns.",
        ]

        return {
            "incident_type": "complex-oom-k8s",
            "log_summary": app_log_result.summary,
            "k8s_signatures": k8s_sigs,
            "config_issues": config_issues,
            "doc_evidence": doc_evidence,
            "root_cause_hypotheses": root_cause_hypotheses,
            "suggested_fix_steps": suggested_fix_steps,
            "prevention_ideas": prevention_ideas,
            "status": "success",
            "error_message": None,
        }
    except Exception as e:
        return {
            "incident_type": "complex-oom-k8s",
            "log_summary": "",
            "k8s_signatures": {},
            "config_issues": [],
            "doc_evidence": [],
            "root_cause_hypotheses": [],
            "suggested_fix_steps": [],
            "prevention_ideas": [],
            "status": "error",
            "error_message": str(e),
        }


# --------------------------------------------
# 3) Define INCIDENT COMMANDER AGENT (Gemini 2.5 Pro)
# --------------------------------------------

incident_commander_agent = Agent(
    name="incident_commander",
    model="gemini-2.5-pro",  
    description=(
        "An SRE/DevOps incident commander agent that investigates incidents using tools "
        "for log analysis, config inspection, and documentation lookup, then produces a "
        "clear Root Cause Analysis, Fix Plan, and Prevention recommendations."
    ),
    instruction="""
You are an expert Site Reliability Engineer and DevOps incident commander.

You have access to the following TOOLS:

1) analyze_simple_incident_tool()
   - Use this for the 'simple DB connection failure' incident.
   - It returns structured fields such as:
     - incident_type
     - log_summary
     - config_issues
     - doc_evidence
     - root_cause_hypotheses
     - suggested_fix_steps
     - prevention_ideas
     - status, error_message

2) analyze_complex_incident_tool()
   - Use this for the 'complex OOM / CrashLoopBackOff' incident.
   - It returns similar structured fields, plus k8s_signatures.

YOUR JOB:
- Decide which tool to call based on the user request.
- Inspect the returned JSON.
- Then write a professional incident report with the following sections in MARKDOWN:

## Incident Summary
- One short paragraph summarizing what is going on.

## Root Cause Analysis
- Bullet points describing the most likely root cause(s).
- Reference evidence from logs, configs, and docs when possible.

## Fix Plan
- Numbered list of clear, actionable steps the SRE team should take.

## Prevention Recommendations
- Bulleted list of recommendations to avoid similar issues in the future.

CONSTRAINTS:
- If the tool returns status="error", explain that the analysis failed and surface the error_message.
- Never invent tools; only use the tools you are given.
- Prefer concise, technically detailed writing.
""",
    tools=[
        analyze_simple_incident_tool,
        analyze_complex_incident_tool,
    ],
)

incident_runner = InMemoryRunner(agent=incident_commander_agent)

print("ADK Incident Commander agent is ready!")


# Simple incident with the real agent
response_simple = await incident_runner.run_debug(
    "Analyze the simple DB connection incident and produce the full report.",
    verbose=False,
)


await incident_runner.run_debug(
    "Analyze the complex OOM / CrashLoopBackOff incident and produce the full report.",
    verbose=False,
)

