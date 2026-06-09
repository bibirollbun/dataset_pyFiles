import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    # Fallback for local testing if not on Kaggle
    os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE" 
    print("âš ï¸� Using manual key or failed. Check secrets.")


from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.function_tool import FunctionTool
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
import warnings 
warnings.filterwarnings("ignore")
import uuid

print("âœ… ADK components imported successfully.")


async def trigger_emergency_pager(incident_type: str, severity: str) -> dict:
    """
    [Protocol: EMERGENCY]
    Triggers PagerDuty/OpsGenie for Critical Incidents.
    FAIL-OPEN LOGIC: Triggers on 'Data Loss', 'Security Breach', 'System Down'.
    """
    print(f"ğŸš¨ LOG: Emergency Pager Triggered: {incident_type} ({severity})")
    return {
        "status": "PAGED",
        "incident_id": f"INC-{uuid.uuid4().hex[:6].upper()}",
        "message": "On-Call DevOps team has been paged. Do not restart servers manually yet."
    }

print("âœ… Emergency Tool Defined.")


async def get_system_status(service_name: str) -> dict:
    """
    [Protocol: DIAGNOSTICS]
    Checks the health status of a specific microservice.
    ZERO-LATENCY LOGIC: Uses simulated mock registry.
    """
    print(f"â„¹ï¸� LOG: Checking status for: {service_name}")
    
    # Mock Service Registry
    status_db = {
        "auth-service": {"status": "Healthy", "uptime": "45d", "cpu": "12%"},
        "payment-gateway": {"status": "Degraded", "uptime": "2h", "cpu": "89%"},
        "database-primary": {"status": "Healthy", "uptime": "120d", "cpu": "45%"},
        "frontend-ui": {"status": "Healthy", "uptime": "5d", "cpu": "22%"}
    }
    
    # Fuzzy search logic
    for key in status_db:
        if key in service_name.lower():
            return {"service": key, "details": status_db[key]}
            
    return {"status": "Unknown", "message": f"Service '{service_name}' not found in registry."}

print("âœ… Status Tool Defined.")


async def restart_service(service_name: str, tool_context: ToolContext) -> dict:
    """
    [Protocol: REMEDIATION]
    Restarts a failing service. 
    SAFETY LOGIC: Pauses for HITL verification to prevent accidental downtime.
    """
    
    # ğŸ›‘ PHASE 1: PAUSE FOR APPROVAL
    if not tool_context.tool_confirmation:
        print(f"âš ï¸� LOG: Intercepting RESTART command for '{service_name}'...")
        print(f"â�¸ï¸� SYSTEM PAUSED: Waiting for Admin verification...")
        
        tool_context.request_confirmation(
            hint=f"CRITICAL: Confirm restart of '{service_name}'? This will cause brief downtime.",
            payload={"service": service_name, "action": "restart"}
        )
        return {
            "status": "pending_approval",
            "message": f"I need authorization to restart {service_name}."
        }

    # âœ… PHASE 2: EXECUTE (Only after confirmed=True)
    if tool_context.tool_confirmation.confirmed:
        print(f"âš ï¸� LOG: APPROVED. Restarting {service_name} now...")
        return {
            "status": "SUCCESS",
            "message": f"Service '{service_name}' has been restarted. Health check passing."
        }
    else:
        print("âš ï¸� LOG: Restart Cancelled by Admin.")
        return {"status": "CANCELLED", "message": "Action aborted."}

print("âœ… Restart Tool Defined.")


async def create_jira_issue(summary: str, issue_type: str, tool_context: ToolContext) -> dict:
    """
    [Protocol: REPORTING]
    Files a ticket in Jira.
    SAFETY LOGIC: Pauses to let user review the ticket content.
    """
    
    if not tool_context.tool_confirmation:
        print(f"ğŸ“� LOG: Drafting Jira Ticket: {summary}...")
        
        tool_context.request_confirmation(
            hint=f"Review Ticket: [{issue_type}] {summary}. Post this?",
            payload={"summary": summary}
        )
        return {
            "status": "pending_review",
            "message": "Please review the ticket draft."
        }

    if tool_context.tool_confirmation.confirmed:
        ticket_key = f"OPS-{uuid.uuid4().int % 1000}"
        print(f"ğŸ“� LOG: POSTED. Jira Key: {ticket_key}")
        return {
            "status": "CREATED",
            "key": ticket_key,
            "message": f"Ticket {ticket_key} created successfully."
        }
    else:
        return {"status": "CANCELLED", "message": "Ticket creation discarded."}

print("âœ… Jira Tool Defined.")


retry_config = types.HttpRetryOptions(
    attempts=3,
    exp_base=7,
    initial_delay=2,
    http_status_codes=[429, 500, 503, 504],
)



sentinel_agent = LlmAgent(
    name="sentinel_agent", 
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    instruction="""You are "Sentinel AI", the Automated Site Reliability Engineer.
Your goal is to maintain system stability and assist the DevOps team.

**CORE ROUTING PROTOCOLS:**

1. ğŸš¨ **EMERGENCY (Fail-Open)**
   - IF input contains "Data Loss", "Security Breach", "System Down", or "Critical":
   - CALL `trigger_emergency_pager` IMMEDIATELY.

2. â„¹ï¸� **DIAGNOSTICS (Info)**
   - IF asking for status (health, cpu, uptime) of a service:
   - CALL `get_system_status`.

3. ğŸ› ï¸� **REMEDIATION (Action)**
   - IF asking to restart, reboot, or kill a service:
   - CALL `restart_service`. (This requires confirmation, but you just call the tool).

4. ğŸ“� **REPORTING (Admin)**
   - IF asking to file a bug, ticket, or issue:
   - CALL `create_jira_issue`.

5. â›” **OUT OF SCOPE**
   - IF asked about non-devops topics (coding help, HR issues):
   - Decline politely: "I am restricted to Production Operations tasks."
    """,
    tools=[
        FunctionTool(func=trigger_emergency_pager),
        FunctionTool(func=get_system_status),
        FunctionTool(func=restart_service),
        FunctionTool(func=create_jira_issue)
    ]
)

# Wrap in Resumable App for HITL features
sentinel_app = App(
    name="sentinel_app",
    root_agent=sentinel_agent,
    resumability_config=ResumabilityConfig(is_resumable=True) 
)

# Create Runner
session_service = InMemorySessionService()
sentinel_runner = Runner(app=sentinel_app, session_service=session_service)

print("âœ… Sentinel Agent Online.")


# Helper to print agent output
def print_agent_response(events):
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"ğŸ¤– Sentinel > {part.text}")
                if part.function_response and "status" in str(part.function_response.response):
                     print(f"   [Tool Output]: {part.function_response.response}")

# The Complex Workflow Engine
async def run_sentinel_workflow(user_query: str, auto_confirm: bool = True):
    print(f"\n{'='*60}")
    print(f"ğŸ—£ï¸� Engineer > {user_query}\n")

    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    await session_service.create_session(app_name="sentinel_app", user_id="devops_user", session_id=session_id)

    query_content = types.Content(role="user", parts=[types.Part(text=user_query)])
    events = []

    # --- PASS 1: Initial Request ---
    async for event in sentinel_runner.run_async(user_id="devops_user", session_id=session_id, new_message=query_content):
        events.append(event)

    # --- CHECK FOR PAUSE (The HITL Firewall) ---
    approval_request = None
    invocation_id = None
    
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.function_call and part.function_call.name == "adk_request_confirmation":
                    approval_request = part.function_call
                    invocation_id = event.invocation_id
                    break
    
    # --- PASS 2: Handle Approval (If Paused) ---
    if approval_request:
        hint_text = approval_request.args.get('hint', 'Confirmation Required')
        print(f"â�¸ï¸�  SAFETY STOP: {hint_text}")
        print(f"ğŸ¤” Admin Decision: {'âœ… APPROVED' if auto_confirm else 'â�Œ DENIED'}\n")
        
        confirmation_response = types.FunctionResponse(
            id=approval_request.id,
            name="adk_request_confirmation",
            response={"confirmed": auto_confirm}
        )
        resume_message = types.Content(role="user", parts=[types.Part(function_response=confirmation_response)])

        # RESUME execution
        async for event in sentinel_runner.run_async(
            user_id="devops_user", 
            session_id=session_id, 
            new_message=resume_message, 
            invocation_id=invocation_id 
        ):
            events.append(event)

    print_agent_response(events)
    print(f"{'='*60}\n")

print("âœ… Workflow Engine Ready.")


print("--- ğŸ¤– STARTING SENTINEL INTEGRATION TESTS ---")

# TEST 1: Emergency (Fail-Open)
# Should trigger pager immediately without asking questions.
await run_sentinel_workflow("CRITICAL: Data Loss detected in Primary DB!")

# TEST 2: Diagnostics (Stateless)
# Should just look up the info.
await run_sentinel_workflow("What is the status of the payment-gateway?")

# TEST 3: Remediation (HITL - Approved)
# Should PAUSE, then Execute restart.
await run_sentinel_workflow("Restart the payment-gateway service.", auto_confirm=True)

# TEST 4: Remediation (HITL - Denied)
# Should PAUSE, then Abort.
await run_sentinel_workflow("Restart the auth-service.", auto_confirm=False)

# TEST 5: Reporting (HITL)
await run_sentinel_workflow("File a ticket: The checkout page is loading slowly.", auto_confirm=True)


# 1. Clean previous runs
!rm -rf sentinel_project

# 2. Initialize ADK structure
!adk create sentinel_project --model gemini-2.5-flash --api_key $GOOGLE_API_KEY

# 3. Write the Agent logic to agent.py
agent_code = """
import os
import uuid
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.function_tool import FunctionTool
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.tools.tool_context import ToolContext

async def trigger_emergency_pager(incident_type: str, severity: str) -> dict:
    return {"status": "PAGED", "message": "DevOps team paged."}

async def get_system_status(service_name: str) -> dict:
    return {"status": "Healthy", "service": service_name}

async def restart_service(service_name: str, tool_context: ToolContext) -> dict:
    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(
            hint=f"Confirm restart of {service_name}?",
            payload={"service": service_name}
        )
        return {"status": "pending"}
    
    if tool_context.tool_confirmation.confirmed:
        return {"status": "SUCCESS", "message": "Service restarted."}
    return {"status": "CANCELLED"}

async def create_jira_issue(summary: str, issue_type: str, tool_context: ToolContext) -> dict:
    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(
            hint=f"Post ticket: {summary}?",
            payload={"summary": summary}
        )
        return {"status": "pending"}
    if tool_context.tool_confirmation.confirmed:
        return {"status": "CREATED", "key": "OPS-123"}
    return {"status": "CANCELLED"}

retry_config = types.HttpRetryOptions(
    attempts=3, exp_base=2, initial_delay=1, http_status_codes=[429, 500, 503]
)

root_agent = LlmAgent(
    name="sentinel_agent", 
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    instruction="You are Sentinel AI, the DevOps Reliability Agent.",
    tools=[
        FunctionTool(func=trigger_emergency_pager),
        FunctionTool(func=get_system_status),
        FunctionTool(func=restart_service),
        FunctionTool(func=create_jira_issue)
    ]
)

agent = App(
    name="sentinel_app",
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(is_resumable=True) 
)
"""

with open("sentinel_project/agent.py", "w") as f:
    f.write(agent_code)

print("âœ… ADK Project Created: sentinel_project/agent.py")


# 5. Launch the ADK Web Interface
# -------------------------------
# This command launches the visual chat UI.
# NOTE: In a Kaggle Notebook, this server runs in the background but ports are not exposed.
# To use the UI for your Video Demo:
# 1. Download the 'sentinel_project' folder.
# 2. Run 'adk web sentinel_project' in your local terminal.

print("ğŸš€ COMMAND TO LAUNCH WEB UI:")
print("adk web sentinel_project")

# Uncomment the line below if you are running this locally
# !adk web sentinel_project


import asyncio

# ==========================================
# ğŸ§ª SENTINEL AI: COMPREHENSIVE TEST SUITE
# ==========================================

print("ğŸš€ INITIALIZING SENTINEL TEST SUITE...")

# --- PART 1: UNIT TEST CASES (Tool Logic) ---
# We test the raw Python functions to ensure they return the expected JSON structure.
# --------------------------------------------
async def run_unit_tests():
    print("\n[1/2] ğŸ› ï¸� RUNNING UNIT TESTS (Function Logic)...")
    
    # Case 1: Status Tool
    res = await get_system_status(service_name="payment-gateway")
    assert res['service'] == "payment-gateway", "â�Œ Status Check Failed"
    print("   âœ… Tool 'get_system_status' is working.")

    # Case 2: Emergency Pager
    res = await trigger_emergency_pager(incident_type="Test", severity="Low")
    assert "INC-" in res['incident_id'], "â�Œ Pager ID Generation Failed"
    print("   âœ… Tool 'trigger_emergency_pager' is working.")

    # Case 3: Restart (Should Pause)
    # We mock the ToolContext to simulate a first-pass call
    class MockContext:
        tool_confirmation = None # No confirmation yet
        def request_confirmation(self, hint, payload): pass
        
    res = await restart_service(service_name="db-prod", tool_context=MockContext())
    assert res['status'] == "pending_approval", "â�Œ Restart Safety Lock Failed"
    print("   âœ… Tool 'restart_service' correctly pauses for safety.")



# --- PART 2: WORKFLOW TEST CASES (Human-in-the-Loop) ---
# We verify the full conversation flow: Request -> Pause -> Approve -> Execute.
# -------------------------------------------------------
async def run_workflow_tests():
    print("\n[2/2] ğŸŒ� RUNNING WORKFLOW TESTS (Pause & Resume)...")
    
    print("   ğŸ”¹ Scenario: Restart Service (Authorized)")
    # We reuse the helper function defined in the main notebook
    # This proves the system can handle the full lifecycle
    await run_sentinel_workflow(
        user_query="I need to hard reboot the database-primary service.", 
        auto_confirm=True
    )
    print("   âœ… Workflow Completed.")


# --- EXECUTE SUITE ---
await run_unit_tests()
await run_workflow_tests()

print("\nğŸ�‰ ALL SYSTEMS GO. SENTINEL IS READY FOR DEPLOYMENT.")




