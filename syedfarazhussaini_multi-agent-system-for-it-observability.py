# @title ğŸ› ï¸� Install core dependencies
# This mirrors the setup pattern used in the Google/Kaggle course notebooks.
# %pip install --quiet google-adk google-cloud-aiplatform pandas gradio #uncomment to install


# @title ğŸ”‘ Load API key (Kaggle Secrets or local .env)
import os

# Try Kaggle Secrets first (when running on Kaggle)
try:
    from kaggle_secrets import UserSecretsClient
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key loaded from Kaggle Secrets.")
except ImportError:
    # Fallback to .env file (when running locally)
    try:
        from dotenv import load_dotenv
        load_dotenv()
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if GOOGLE_API_KEY:
            print("âœ… Gemini API key loaded from .env file.")
        else:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
    except ImportError:
        # Manual environment variable check
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if GOOGLE_API_KEY:
            print("âœ… Gemini API key loaded from environment variables.")
        else:
            print("â�Œ Please set GOOGLE_API_KEY environment variable or install python-dotenv")
except Exception as e:
    print(f"ğŸ”‘ Authentication Error: {e}")
    print("Please ensure GOOGLE_API_KEY is available in Kaggle Secrets or environment variables.")


# @title Authenticate and initialize Vertex AI
import os
import vertexai

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "l3-multi-agent-system-4-IT")  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}
MODEL_NAME = "gemini-2.5-flash-lite"

print(f"Using project: {PROJECT_ID} | region: {LOCATION}")
vertexai.init(project=PROJECT_ID, location=LOCATION)


# @title Generate synthetic logs, metrics, and incidents
import pandas as pd
import random
from datetime import datetime, timedelta
from typing import Literal

random.seed(42)

SEVERITIES = ["CRITICAL", "ERROR", "WARN", "INFO"]
INCIDENT_TYPES = ["Network", "Database", "Application", "Infrastructure"]


def generate_mock_logs(server_id: str, *, window_minutes: int = 240) -> str:
    """Create timestamped log entries with realistic error bursts."""
    now = datetime.utcnow()
    entries = []
    for minute in range(window_minutes // 5):
        timestamp = now - timedelta(minutes=minute * 5)
        level = random.choices(SEVERITIES, weights=[0.05, 0.15, 0.3, 0.5])[0]
        if level in {"CRITICAL", "ERROR"}:
            message = random.choice([
                "Latency spike detected on API Gateway",
                "Database connection timeout",
                "Disk saturation beyond 95%",
                "Service mesh circuit breaker open",
            ])
        elif level == "WARN":
            message = random.choice([
                "Retrying connection to cache cluster",
                "CPU utilization approaching threshold",
                "Replica lag increasing",
            ])
        else:
            message = random.choice([
                "Health check passed",
                "Autoscaler polling",
                "Background job completed",
            ])
        entries.append(f"{timestamp.isoformat()}Z [{level}] {server_id}: {message}")
    return "\n".join(reversed(entries))


def generate_mock_metrics(hours: int = 24) -> pd.DataFrame:
    """Return hourly CPU/memory stats with spikes to trigger SLA alerts."""
    now = datetime.utcnow()
    return pd.DataFrame(
        {
            "timestamp": [now - timedelta(hours=h) for h in range(hours)][::-1],
            "cpu_pct": [max(10, min(99, random.gauss(55, 18))) for _ in range(hours)],
            "memory_pct": [max(20, min(95, random.gauss(63, 12))) for _ in range(hours)],
        }
    )


def generate_incident_email(severity: Literal["SEV1", "SEV2", "SEV3"]) -> str:
    incident = random.choice(INCIDENT_TYPES)
    window = random.choice(["00:00-02:00 UTC", "02:00-04:00 UTC", "Maintenance window TBD"])
    return (
        f"Subject: {severity} {incident} Incident Update\n"
        f"From: it-operations@company.com\n"
        f"Body: {incident} team reports anomalies impacting customer latency."
        f" Suggested remediation window: {window}."
    )


# @title Register ADK tools
from google.adk.tools import FunctionTool


def fetch_logs_tool(server_id: str = "prod-app-01") -> str:
    """Return recent log entries for a server."""
    return generate_mock_logs(server_id)


def summarize_utilization(time_range: str = "last_24h") -> dict:
    """Provide aggregate CPU/Memory stats for the requested window."""
    df = generate_mock_metrics()
    return {
        "time_range": time_range,
        "average_cpu_pct": round(df["cpu_pct"].mean(), 2),
        "peak_cpu_pct": round(df["cpu_pct"].max(), 2),
        "average_memory_pct": round(df["memory_pct"].mean(), 2),
    }


def fetch_latest_incident() -> str:
    """Return the latest synthetic incident email for context."""
    return generate_incident_email("SEV2")


fetch_server_logs = FunctionTool(fetch_logs_tool)
get_cpu_utilization = FunctionTool(summarize_utilization)
read_incident_emails = FunctionTool(fetch_latest_incident)


# @title Build supervisor and specialist agents
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner

log_agent = Agent(
    name="log_analyst",
    model=MODEL_NAME,
    instruction=(
        "You inspect raw infrastructure logs to detect anomalies, downtime, and root causes."
        " Summarize key findings and cite log fragments."
    ),
    tools=[fetch_server_logs],
)

metric_agent = Agent(
    name="metric_analyst",
    model=MODEL_NAME,
    instruction=(
        "You analyze time-series metrics to explain utilization trends, SLA breaches, and capacity risks."
        " Produce concise stats and recommendations."
    ),
    tools=[get_cpu_utilization],
)

operations_agent = Agent(
    name="operations_planner",
    model=MODEL_NAME,
    instruction=(
        "You coordinate remediation windows, patching schedules, and scaling plans using inputs from peers."
        " Recommend low-impact execution windows and stakeholder messaging."
    ),
    tools=[get_cpu_utilization, read_incident_emails],
)

supervisor_agent = Agent(
    name="it_ops_supervisor",
    model=MODEL_NAME,
    instruction=(
        "You orchestrate specialists to answer executive questions about reliability and performance."
        " Decide when to call sub-agents and synthesize a single actionable response."
    ),
    sub_agents=[log_agent, metric_agent, operations_agent],
)

runner = InMemoryRunner(agent=supervisor_agent)
print("âœ… Multi-agent system ready")


# @title Investigate a slowdown
query = "We had a customer-facing latency spike overnight. Explain root cause, summarize metrics, and propose a mitigation plan."
print(f"User > {query}\n")

# run_debug prints a formatted trace similar to Kaggle notebooks.
await runner.run_debug(query, user_id="exec", session_id="it-ops-session", verbose=True)


### Uncomment locally to run chatbot UI
# # @title Start Gradio chat prototype
# import gradio as gr
# from google.genai import types

# chat_runner = InMemoryRunner(agent=supervisor_agent)

# async def respond(message: str, history: list[tuple[str, str]]):
#     user_content = types.Content(
#         role="user",
#         parts=[types.Part.from_text(text=message)],
#     )
#     transcript: list[str] = []
#     async for event in chat_runner.run_async(
#         user_id="dashboard",
#         session_id="mgmt-briefing",
#         new_message=user_content,
#     ):
#         if event.author == "it_ops_supervisor" and event.content and event.content.parts:
#             transcript.extend(part.text or "" for part in event.content.parts if part.text)
#     return "\n".join(transcript)

# iface = gr.ChatInterface(
#     fn=respond,
#     title="IT Ops Reliability Copilot",
#     description="Ask about outages, patch windows, or capacity trends.",
# )
# iface.launch(share=False, debug=True)


# @title Scaffold ADK agent package (run once per session)
!adk create it_ops_observability --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY


# @title Write agent module for Try ADK
from pathlib import Path
import textwrap

agent_dir = Path("it_ops_observability")
agent_dir.mkdir(parents=True, exist_ok=True)

agent_source = textwrap.dedent('''
    Multi-agent IT observability system for ADK Web.

    from __future__ import annotations

    import random
    from datetime import datetime, timedelta
    from typing import Literal

    import pandas as pd

    from google.adk.agents import Agent
    from google.adk.tools import FunctionTool

    MODEL_NAME = "gemini-2.5-flash-lite"

    SEVERITIES = ["CRITICAL", "ERROR", "WARN", "INFO"]
    INCIDENT_TYPES = ["Network", "Database", "Application", "Infrastructure"]


    def generate_mock_logs(server_id: str, window_minutes: int = 240) -> str:
        """Create timestamped log entries with realistic error bursts."""
        now = datetime.utcnow()
        entries: list[str] = []
        for minute in range(window_minutes // 5):
            timestamp = now - timedelta(minutes=minute * 5)
            level = random.choices(SEVERITIES, weights=[0.05, 0.15, 0.3, 0.5])[0]
            if level in {"CRITICAL", "ERROR"}:
                message = random.choice(
                    [
                        "Latency spike detected on API Gateway",
                        "Database connection timeout",
                        "Disk saturation beyond 95%",
                        "Service mesh circuit breaker open",
                    ]
                )
            elif level == "WARN":
                message = random.choice(
                    [
                        "Retrying connection to cache cluster",
                        "CPU utilization approaching threshold",
                        "Replica lag increasing",
                    ]
                )
            else:
                message = random.choice(
                    [
                        "Health check passed",
                        "Autoscaler polling",
                        "Background job completed",
                    ]
                )
            entries.append(f"{timestamp.isoformat()}Z [{level}] {server_id}: {message}")
        return "\n".join(reversed(entries))


    def generate_mock_metrics(hours: int = 24) -> pd.DataFrame:
        """Return hourly CPU/memory stats with spikes to trigger SLA alerts."""
        now = datetime.utcnow()
        cpu = [max(10, min(99, random.gauss(55, 18))) for _ in range(hours)]
        memory = [max(20, min(95, random.gauss(63, 12))) for _ in range(hours)]
        return pd.DataFrame(
            {
                "timestamp": [now - timedelta(hours=h) for h in range(hours)][::-1],
                "cpu_pct": cpu,
                "memory_pct": memory,
            }
        )


    def generate_incident_email(severity: Literal["SEV1", "SEV2", "SEV3"]) -> str:
        incident = random.choice(INCIDENT_TYPES)
        window = random.choice(["00:00-02:00 UTC", "02:00-04:00 UTC", "Maintenance window TBD"])
        return (
            f"Subject: {severity} {incident} Incident Update\n"
            f"From: it-operations@company.com\n"
            f"Body: {incident} team reports anomalies impacting customer latency."
            f" Suggested remediation window: {window}."
        )


    def fetch_logs_tool(server_id: str = "prod-app-01") -> str:
        """Return recent log entries for a server."""
        return generate_mock_logs(server_id)


    def summarize_utilization(time_range: str = "last_24h") -> dict:
        """Provide aggregate CPU/Memory stats for the requested window."""
        df = generate_mock_metrics()
        return {
            "time_range": time_range,
            "average_cpu_pct": round(df["cpu_pct"].mean(), 2),
            "peak_cpu_pct": round(df["cpu_pct"].max(), 2),
            "average_memory_pct": round(df["memory_pct"].mean(), 2),
        }


    def fetch_latest_incident() -> str:
        """Return the latest synthetic incident email for context."""
        return generate_incident_email("SEV2")


    def create_agent() -> Agent:
        """Expose the supervisor agent for ADK web."""
        fetch_server_logs = FunctionTool(fetch_logs_tool)
        get_cpu_utilization = FunctionTool(summarize_utilization)
        read_incident_emails = FunctionTool(fetch_latest_incident)

        log_agent = Agent(
            name="log_analyst",
            model=MODEL_NAME,
            instruction=(
                "You inspect raw infrastructure logs to detect anomalies, downtime, and root causes."
                " Summarize key findings and cite log fragments."
            ),
            tools=[fetch_server_logs],
        )

        metric_agent = Agent(
            name="metric_analyst",
            model=MODEL_NAME,
            instruction=(
                "You analyze time-series metrics to explain utilization trends, SLA breaches, and capacity risks."
                " Produce concise stats and recommendations."
            ),
            tools=[get_cpu_utilization],
        )

        operations_agent = Agent(
            name="operations_planner",
            model=MODEL_NAME,
            instruction=(
                "You coordinate remediation windows, patching schedules, and scaling plans using inputs from peers."
                " Recommend low-impact execution windows and stakeholder messaging."
            ),
            tools=[get_cpu_utilization, read_incident_emails],
        )

        supervisor_agent = Agent(
            name="it_ops_supervisor",
            model=MODEL_NAME,
            instruction=(
                "You orchestrate specialists to answer executive questions about reliability and performance."
                " Decide when to call sub-agents and synthesize a single actionable response."
            ),
            sub_agents=[log_agent, metric_agent, operations_agent],
        )

        return supervisor_agent
    '''
)

agent_path = agent_dir / "agent.py"
agent_path.write_text(agent_source)

init_path = agent_dir / "__init__.py"
init_path.write_text("from .agent import create_agent\n\n__all__ = [\"create_agent\"]\n")

print(f"âœ… ADK web agent written to {agent_path}")


# # @title Compute optional Kaggle proxy URL #uncomment locally to run ADK UI
# import os
# from IPython.core.display import display, HTML
# from jupyter_server.serverapp import list_running_servers


# def get_adk_proxy_url() -> str:
#     """Resolve the proxied URL that Kaggle notebooks expect."""
#     proxy_host = "https://kkb-production.jupyter-proxy.kaggle.net"
#     adk_port = "8000"
#     servers = list(list_running_servers())
#     if not servers:
#         raise RuntimeError("No running Jupyter servers detected.")
#     base_url = servers[0]["base_url"]
#     parts = base_url.split("/")
#     try:
#         kernel, token = parts[2], parts[3]
#     except IndexError as exc:
#         raise RuntimeError(f"Could not parse kernel/token from base URL: {base_url}") from exc
#     url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{adk_port}"
#     button_html = f"""
#     <div style="padding: 15px; border: 2px solid #f0ad4e; border-radius: 8px; background-color: #fef9f0; margin: 20px 0;">
#       <div style=\"font-family: sans-serif; margin-bottom: 12px; color: #333; font-size: 1.1em;\"><strong>âš ï¸� When running on Kaggle:</strong></div>
#       <div style=\"font-family: sans-serif; margin-bottom: 15px; color: #333; line-height: 1.5;\">
#         Run the web server cell first, leave it running, then click the button below to open the Try ADK UI in a new tab.
#       </div>
#       <a href='{proxy_host}{url_prefix}' target='_blank' style=\"display:inline-block; background-color:#1a73e8; color:white; padding:10px 20px; text-decoration:none; border-radius:25px; font-family:sans-serif; box-shadow:0 2px 5px rgba(0,0,0,0.2);\">
#         Open Try ADK UI â†—
#       </a>
#     </div>
#     """
#     display(HTML(button_html))
#     return url_prefix


# try:
#     url_prefix
# except NameError:
#     url_prefix = ""

# if not url_prefix:
#     try:
#         url_prefix = get_adk_proxy_url()
#     except Exception as exc:
#         url_prefix = ""
#         print("âš ï¸� Running outside Kaggle. Continue to the next cell and omit --url_prefix if you are local.")
#         print(f"Details: {exc}")


# # @title Launch ADK web (keeps running) #uncomment locally to run ADK UI
# !adk web it_ops_observability --url_prefix {url_prefix}

