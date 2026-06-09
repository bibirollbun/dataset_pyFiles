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


# ============================================================
# 0. Install ADK and nest-asyncio (for Jupyter notebook compatibility)
# ============================================================
!pip install -q google-adk nest-asyncio

# ============================================================
# 1. Authenticate Gemini via Kaggle Secrets
# ============================================================
import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' "
        f"to your Kaggle secrets. Details: {e}"
    )

# ============================================================
# 2. Imports: ADK + helpers
# ============================================================
import warnings
# Suppress FutureWarning from traitlets/nbconvert (Kaggle notebook conversion)
warnings.filterwarnings('ignore', category=FutureWarning, module='traitlets')

import asyncio
import json
import textwrap
from datetime import datetime
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from google.adk.agents import Agent           # core ADK Agent (LLM-based)
from google.adk.runners import InMemoryRunner # simple runner for dev/notebooks
from google.genai import types                # Content, Part, HttpRetryOptions

# Import and apply nest_asyncio for Jupyter notebook compatibility
import nest_asyncio
nest_asyncio.apply()  # Allows nested event loops in Jupyter notebooks

# Matplotlib defaults
plt.switch_backend("agg")  # For non-interactive env like Kaggle

# ============================================================
# 3. Retry options for Gemini calls
# ============================================================
retry_config = types.HttpRetryOptions(
    attempts=5,          # max retry attempts
    exp_base=2,          # exponential backoff base
    initial_delay=1,     # initial delay (seconds)
    http_status_codes=[429, 500, 503, 504],
)

print("âœ… Retry configuration ready.")

# ============================================================
# 4. ADK Agents: UI Test Agent + API Test Agent
# ============================================================

APP_NAME = "qa_automation_orchestrator"

print("ğŸ”„ Creating ADK agents...")

# ---- UI Test Agent (Playwright-style) ----
ui_test_agent = Agent(
    name="ui_test_generator_agent",
    model="gemini-2.0-flash",
    description=(
        "An enterprise QA automation agent that converts investigated, "
        "fixed Sev0/Sev1 CRM UI incidents into deterministic Playwright tests."
    ),
    instruction=textwrap.dedent(
        """
        You are a senior QA automation engineer for a SaaS CRM.

        You receive JSON describing a **single UI incident** that is:
        - Sev0 or Sev1
        - Investigated, root cause identified
        - Fixed and closed
        - Reproducible

        Your job:
        - Convert it into a **single deterministic, reproducible UI automated test**.
        - Use **Python + Playwright** style pseudo-code (assume sync style).
        - Focus on clarity, explicit selectors, and robust assertions.
        - Do NOT add random sleeps; if needed, mention waits in comments.
        - Assume a web-based CRM UI (browser automation).

        Return ONLY Python code inside a single code block:

        ```python
        # code here
        ```
        """
    ),
)

# ---- API Test Agent (PyTest + requests) ----
api_test_agent = Agent(
    name="api_test_generator_agent",
    model="gemini-2.0-flash",
    description=(
        "An enterprise QA automation agent that converts investigated, "
        "fixed Sev0/Sev1 API incidents into deterministic PyTest API tests."
    ),
    instruction=textwrap.dedent(
        """
        You are a senior QA automation engineer for a SaaS CRM.

        You receive JSON describing a **single API incident** that is:
        - Sev0 or Sev1
        - Investigated, root cause identified
        - Fixed and closed
        - Reproducible

        The JSON includes:
        - HTTP method (GET/POST/PUT/DELETE)
        - Endpoint path (e.g., /api/v1/contacts)
        - Sample request payload (if applicable)
        - Expected status code
        - Expected response fields

        Your job:
        - Convert it into a **single deterministic, reproducible API automated test**.
        - Use **PyTest + Python `requests` library**.
        - Include clear assertions for status code and important fields.
        - DO NOT add random sleeps.
        - Assume a base URL variable like `BASE_URL` exists.

        Return ONLY Python code inside a single code block:

        ```python
        # code here
        ```
        """
    ),
)

# ---- Runners for each agent ----
ui_runner = InMemoryRunner(agent=ui_test_agent, app_name=f"{APP_NAME}_ui")
api_runner = InMemoryRunner(agent=api_test_agent, app_name=f"{APP_NAME}_api")

print("âœ… UI & API ADK Agents created.")


# ============================================================
# 5. Session helpers
# ============================================================

async def _run_with_cleanup(coro):
    """
    Run coroutine with minimal cleanup delay.
    """
    result = await coro
    # Small delay to allow immediate cleanup tasks to start
    # Don't wait for all tasks as it can cause hangs
    await asyncio.sleep(0.1)
    return result


def run_async(coro):
    """
    Run async function in both regular Python and Jupyter notebooks.
    Uses nest_asyncio to handle nested event loops (applied at module level).
    Ensures proper cleanup of async resources by allowing cleanup tasks to complete.
    """
    return asyncio.run(_run_with_cleanup(coro))


def create_session(runner: InMemoryRunner, app_name: str) -> str:
    """Create an ADK session and return session_id."""
    print(f"ğŸ”„ Creating session for {app_name}...")
    async def _create():
        try:
            session = await asyncio.wait_for(
                runner.session_service.create_session(
                    app_name=app_name,
                    user_id="user",
                ),
                timeout=30.0  # 30 second timeout
            )
            return session
        except asyncio.TimeoutError:
            raise RuntimeError(f"Session creation timed out for {app_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to create session for {app_name}: {e}")
    
    try:
        session = run_async(_create())
        print(f"âœ… Created session for {app_name}: {session.id}")
        return session.id
    except Exception as e:
        print(f"â�Œ Error creating session for {app_name}: {e}")
        raise

# Sessions will be created on-demand when agents are first called
# This prevents blocking at startup
UI_SESSION_ID = None
API_SESSION_ID = None
print("â„¹ï¸� Sessions will be created on-demand when agents are called.\n")


async def call_adk_agent_async(runner: InMemoryRunner, session_id: str, payload: dict, app_name: str = None) -> str:
    """
    Call an ADK agent with the given payload as JSON text (async version).
    Returns the final text response (expected to be Python code block).
    
    Note: runner.run() returns a regular generator, not an async generator.
    If session_id is None, creates a session on-demand.
    """
    # Create session on-demand if not provided
    if session_id is None:
        print("ğŸ”„ Creating session on-demand...")
        try:
            # Use provided app_name or default
            session_app_name = app_name or getattr(runner, 'app_name', APP_NAME)
            session = await asyncio.wait_for(
                runner.session_service.create_session(
                    app_name=session_app_name,
                    user_id="user",
                ),
                timeout=30.0
            )
            session_id = session.id
            print(f"âœ… Created session: {session_id}")
        except Exception as e:
            raise RuntimeError(f"Failed to create session on-demand: {e}")
    
    user_message = json.dumps(payload, indent=2)

    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)],
    )

    final_text = ""
    # Runner yields events; we take the last text part as final response
    # runner.run() returns a regular generator (not async)
    for event in runner.run(
        user_id="user",
        session_id=session_id,
        new_message=content,
    ):
        if getattr(event, "content", None) and event.content.parts:
            part = event.content.parts[0]
            if getattr(part, "text", None):
                final_text = part.text

    return final_text.strip()


def call_adk_agent(runner: InMemoryRunner, session_id: str, payload: dict, app_name: str = None) -> str:
    """
    Synchronous wrapper for call_adk_agent_async.
    Works in both regular Python and Jupyter notebooks.
    """
    return run_async(call_adk_agent_async(runner, session_id, payload, app_name))


# ============================================================
# 6. Sample Incidents with test_type (ui / api)
# ============================================================

SAMPLE_INCIDENTS = [
    {
        "id": "INC-1001",
        "title": "Sev0 - Production login failure for all users",
        "severity": "Sev0",
        "status": "Closed",
        "investigation_status": "Root Cause Identified",
        "fix_status": "Fix Deployed",
        "reproducible": True,
        "closed_in_tool": True,
        "root_cause": "Expired OAuth client secret not rotated in config.",
        "steps_to_reproduce": [
            "Navigate to https://crm.example.com",
            "Enter valid username and password",
            "Click Login",
            "Observe 500 error and 'Authentication service unavailable'"
        ],
        "expected_behavior": "User successfully logs in to CRM home page.",
        "actual_behavior": "Login fails with 500 error for all users.",
        "affected_module": "Authentication / Login",
        "env": "Production",
        "test_type": "ui"   # UI flow
    },
    {
        "id": "INC-1002",
        "title": "Sev1 - Contact save intermittently fails",
        "severity": "Sev1",
        "status": "Closed",
        "investigation_status": "Root Cause Identified",
        "fix_status": "Fix Deployed",
        "reproducible": True,
        "closed_in_tool": True,
        "root_cause": "Null pointer when phone number is missing and WhatsApp flag is enabled.",
        "steps_to_reproduce": [
            "Login as standard CRM user",
            "Go to Contacts tab",
            "Click New Contact",
            "Fill First Name and Last Name only",
            "Enable 'WhatsApp Opt-in' checkbox",
            "Click Save"
        ],
        "expected_behavior": "Contact is saved even if phone number is empty (or validation message is shown).",
        "actual_behavior": "Server error 500 displayed and contact is not saved.",
        "affected_module": "Contacts / Data Entry",
        "env": "Production",
        "test_type": "ui"   # UI flow
    },
    # Example API incident (should generate API test)
    {
        "id": "INC-2001",
        "title": "Sev1 - Create Contact API returns 500 when WhatsApp opt-in without phone",
        "severity": "Sev1",
        "status": "Closed",
        "investigation_status": "Root Cause Identified",
        "fix_status": "Fix Deployed",
        "reproducible": True,
        "closed_in_tool": True,
        "root_cause": "Server-side validation incorrectly rejects payload without phone.",
        "steps_to_reproduce": [
            "Send POST /api/v1/contacts with WhatsAppOptIn=true and phone=null",
            "Observe 500 status code."
        ],
        "expected_behavior": "API should return 400 with validation error or accept and create contact per spec.",
        "actual_behavior": "API returns 500 internal server error.",
        "affected_module": "Contacts / API",
        "env": "Production",
        "test_type": "api",      # API flow
        "api_method": "POST",
        "api_path": "/api/v1/contacts",
        "api_request_payload": {
            "firstName": "Test",
            "lastName": "User",
            "phone": None,
            "whatsAppOptIn": True
        },
        "expected_status_code": 200,
        "expected_response_assertions": {
            "has_fields": ["id", "firstName", "lastName"],
            "field_values": {
                "firstName": "Test",
                "lastName": "User"
            }
        }
    },
    # Non-eligible examples (config issue, in-progress, etc.)
    {
        "id": "INC-1003",
        "title": "Sev1 - Incorrect dashboard widget due to misconfiguration",
        "severity": "Sev1",
        "status": "Closed",
        "investigation_status": "Config Issue",
        "fix_status": "No Code Change",
        "reproducible": True,
        "closed_in_tool": True,
        "root_cause": "Customer modified report filter; no product bug.",
        "steps_to_reproduce": [
            "Login as admin",
            "Open Sales Performance dashboard",
            "Observe missing region filter"
        ],
        "expected_behavior": "Widget should show data based on correct filters.",
        "actual_behavior": "Data filtered incorrectly due to misconfiguration.",
        "affected_module": "Analytics / Dashboards",
        "env": "Production",
        "test_type": "ui"
    },
    {
        "id": "INC-1004",
        "title": "Sev0 - Intermittent timeout on nightly batch job",
        "severity": "Sev0",
        "status": "In Progress",
        "investigation_status": "Under Investigation",
        "fix_status": "No Fix Yet",
        "reproducible": False,
        "closed_in_tool": False,
        "root_cause": None,
        "steps_to_reproduce": [
            "Run nightly data sync job",
            "Observe that sometimes it times out after 30 minutes"
        ],
        "expected_behavior": "Job completes within SLA.",
        "actual_behavior": "Job intermittently times out.",
        "affected_module": "Batch / Integration",
        "env": "Production",
        "test_type": "api"
    }
]

print("ğŸ“‚ Sample incidents loaded:", len(SAMPLE_INCIDENTS))


# ============================================================
# 7. Agent 1: Incident Detector (simulated)
# ============================================================

def detect_incidents():
    """
    In real life:
      - Pull from Jira / ServiceNow / CRM / Slack exports.
    For this capstone:
      - Return the predefined SAMPLE_INCIDENTS.
    """
    print("ğŸ”� Incident Detector: returning simulated incidents.")
    return SAMPLE_INCIDENTS


# ============================================================
# 8. Agent 2: Eligibility Filter
# ============================================================

def is_eligible(incident: dict) -> bool:
    """
    Incident is eligible for automated regression test if:
      - Severity is Sev0 or Sev1
      - Status Closed
      - Investigation complete (Root Cause Identified)
      - Fix deployed as code
      - Reproducible == True
      - Closed in tracking tool
    """
    return (
        incident.get("severity") in ["Sev0", "Sev1"]
        and incident.get("status") == "Closed"
        and incident.get("investigation_status") == "Root Cause Identified"
        and incident.get("fix_status") == "Fix Deployed"
        and incident.get("reproducible") is True
        and incident.get("closed_in_tool") is True
    )


def filter_eligible_incidents(incidents):
    eligible, skipped = [], []
    for inc in incidents:
        (eligible if is_eligible(inc) else skipped).append(inc)
    print(f"âœ… Eligibility Filter: {len(eligible)} eligible, {len(skipped)} skipped.")
    return eligible, skipped


# ============================================================
# 9. Agent 3: Test Generators (UI + API) via ADK Agents
# ============================================================

TEST_OUTPUT_DIR = Path("/kaggle/working/generated_tests")
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_code_from_markdown(raw: str) -> str:
    """
    Extract Python code from a ```python ... ``` block if present.
    Fallback: return raw.
    """
    code = raw
    if "```" in raw:
        parts = raw.split("```")
        if len(parts) >= 3:
            candidate = parts[1]
            lines = candidate.splitlines()
            if lines and lines[0].strip().lower().startswith("python"):
                lines = lines[1:]
            code = "\n".join(lines)
    return textwrap.dedent(code).strip()


def generate_ui_test(incident: dict) -> str:
    payload = {
        "incident_id": incident.get("id"),
        "title": incident.get("title"),
        "severity": incident.get("severity"),
        "steps_to_reproduce": incident.get("steps_to_reproduce"),
        "expected_behavior": incident.get("expected_behavior"),
        "actual_behavior": incident.get("actual_behavior"),
        "affected_module": incident.get("affected_module"),
        "env": incident.get("env"),
    }
    raw = call_adk_agent(ui_runner, UI_SESSION_ID, payload, app_name=f"{APP_NAME}_ui")
    return extract_code_from_markdown(raw)


def generate_api_test(incident: dict) -> str:
    payload = {
        "incident_id": incident.get("id"),
        "title": incident.get("title"),
        "severity": incident.get("severity"),
        "api_method": incident.get("api_method"),
        "api_path": incident.get("api_path"),
        "request_payload": incident.get("api_request_payload"),
        "expected_status_code": incident.get("expected_status_code"),
        "expected_response_assertions": incident.get("expected_response_assertions"),
        "env": incident.get("env"),
    }
    raw = call_adk_agent(api_runner, API_SESSION_ID, payload, app_name=f"{APP_NAME}_api")
    return extract_code_from_markdown(raw)


def generate_tests(eligible_incidents):
    generated_files = []

    for incident in eligible_incidents:
        test_type = incident.get("test_type", "ui").lower()
        print(f"\nğŸ§ª Generating {test_type.upper()} test for {incident['id']}: {incident['title']}")

        if test_type == "api":
            code = generate_api_test(incident)
            suffix = "api"
        else:
            code = generate_ui_test(incident)
            suffix = "ui"

        safe_id = incident["id"].lower().replace("-", "_")
        filename = TEST_OUTPUT_DIR / f"test_{safe_id}_{suffix}.py"

        with open(filename, "w", encoding="utf-8") as f:
            f.write("# Auto-generated by Gemini (ADK) from incident details\n")
            f.write(f"# Incident: {incident['id']} - {incident['title']}\n\n")
            f.write(code + "\n")

        generated_files.append(str(filename))
        print("  ğŸ’¾ Saved test:", filename)

    print("\nğŸ§© Total generated tests:", len(generated_files))
    return generated_files


# ============================================================
# 10. Agent 4: Automation Integrator (simulated)
# ============================================================

def integrate_tests_into_suite(test_files):
    """
    In a real system:
      - Commit these to a repo, create PR, hook into CI.
    For this demo:
      - Write integration metadata JSON.
    """
    integration_record = {
        "integrated_at": datetime.utcnow().isoformat() + "Z",
        "test_files": test_files,
        "ci_pipeline": "simulated-regression-suite",
    }

    record_path = TEST_OUTPUT_DIR / "integration_record.json"
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(integration_record, f, indent=2)

    print("ğŸ”— Automation Integrator: recorded integration metadata at", record_path)
    return record_path


# ============================================================
# 11. Agent 5: Weekly Validator (simulated execution)
# ============================================================

RESULTS_CSV = Path("/kaggle/working/test_results.csv")

def run_weekly_tests(test_files):
    """
    Simulate a weekly regression run.
    Currently marks all tests as PASS (extend as needed).
    """
    rows = []
    run_date = datetime.utcnow().date().isoformat()

    for tf in test_files:
        rows.append(
            {
                "date": run_date,
                "test_name": Path(tf).name,
                "status": "PASS",
                "duration_seconds": 5.2,
                "error_log": "",
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print("ğŸ“Š Weekly Validator: wrote simulated results to", RESULTS_CSV)
    return df


# ============================================================
# 12. Agent 6: Reporting Agent (simple dashboard)
# ============================================================

def generate_dashboard(results_df: pd.DataFrame):
    if results_df.empty:
        print("No results to report.")
        return

    status_counts = results_df["status"].value_counts()

    print("\n=== Test Run Summary ===")
    print(status_counts)

    plt.figure(figsize=(4, 3))
    status_counts.plot(kind="bar")
    plt.title("Test Status Counts")
    plt.xlabel("Status")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


# ============================================================
# 13. Orchestrator: Full Enterprise Agent Pipeline
# ============================================================

def run_pipeline():
    print("\nğŸš€ Multi-Agent Quality Automation Orchestrator (ADK + Gemini)")

    # 1) Detect incidents
    incidents = detect_incidents()

    # 2) Filter eligible ones
    eligible, skipped = filter_eligible_incidents(incidents)

    print("\nâœ… Eligible incidents to convert into tests:")
    for inc in eligible:
        print(f"  - {inc['id']} : {inc['title']} ({inc.get('test_type','ui')})")

    print("\nâš ï¸� Skipped incidents (config/no fix/non-repro/etc.):")
    for inc in skipped:
        print(f"  - {inc['id']} : {inc['title']}")

    if not eligible:
        print("\nâ�Œ No eligible incidents. Nothing to generate.")
        return

    # 3) Generate tests using appropriate agent (UI or API)
    test_files = generate_tests(eligible)

    # 4) Integrate into suite (simulated)
    integrate_tests_into_suite(test_files)

    # 5) Run weekly regression (simulated)
    results_df = run_weekly_tests(test_files)

    # 6) Reporting
    generate_dashboard(results_df)

    print("\nâœ… Pipeline completed successfully.")


# ============================================================
# 14. Execute
# ============================================================

# comment the line below to run the pipeline in another cell
# Or call run_pipeline() manually in a separate cell
run_pipeline()

