# Install the Google Agent Development Kit
!pip install google-adk


import os
from kaggle_secrets import UserSecretsClient

# Setup API Key from Kaggle Secrets
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(f"ğŸ”‘ Authentication Error: {e}. Please ensure 'GOOGLE_API_KEY' is added to Secrets.")


import sqlite3
import uuid
from datetime import datetime

DB_NAME = "tech_support.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Create tickets table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tickets (
            ticket_id TEXT PRIMARY KEY,
            device_id TEXT,
            issue_summary TEXT,
            solution TEXT,
            category TEXT,
            status TEXT,
            timestamp DATETIME
        )
    ''')
    conn.commit()
    conn.close()
    print("âœ… SQLite Database Initialized: tech_support.db")

init_db()


from typing import Dict, Any

def record_ticket_tool(device_id: str, issue_summary: str, solution: str, category: str) -> Dict[str, Any]:
    """
    Records a verified technical issue into the database.
    Use this ONLY after the Manager Agent has APPROVED the solution.
    
    Args:
        device_id: The hardware ID (e.g., PRINTER-X99).
        issue_summary: A concise summary of the problem.
        solution: The trusted solution steps.
        category: 'Hardware' or 'Software'.
    """
    ticket_id = f"TKT-{uuid.uuid4().hex[:6].upper()}"
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO tickets (ticket_id, device_id, issue_summary, solution, category, status, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (ticket_id, device_id, issue_summary, solution, category, "TRUSTED", datetime.now())
    )
    conn.commit()
    conn.close()
    
    print(f"   ğŸ“� [DB Record] Ticket {ticket_id} saved for {device_id}")
    return {"status": "success", "ticket_id": ticket_id, "message": "Ticket recorded."}

def get_device_issue_count(device_id: str) -> int:
    """Counts total issues for a specific device to identify recurring problems."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM tickets WHERE device_id = ?", (device_id,))
    count = cursor.fetchone()[0]
    conn.close()
    return count

def send_email_mcp_simulation(recipient: str, subject: str, body: str) -> Dict[str, str]:
    """
    Simulates sending a high-priority report via MCP Email Protocol.
    Use this when a device exceeds the failure threshold (>= 5 issues).
    """
    print(f"\nğŸš¨ [MCP REPORT] Sending Critical Alert to {recipient}")
    print(f"   Subject: {subject}")
    print(f"   Body: {body}\n")
    return {"status": "sent", "timestamp": str(datetime.now())}

print("âœ… Agent Tools Defined.")


from google.adk.agents import Agent, LlmAgent
from google.adk.models.google_llm import Gemini
from google.genai import types
from google.adk.tools import AgentTool, load_memory

# Rate limit handling config
retry_config = types.HttpRetryOptions(attempts=5, exp_base=2, initial_delay=2)
model_config = Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config)

# 1. Analyst: Compresses context
analyst_agent = LlmAgent(
    name="AnalystAgent",
    model=model_config,
    instruction="""
    You are a Technical Analyst. 
    1. Receive a raw technical issue description.
    2. Perform 'context compaction': Summarize the problem concisely.
    3. Classify: 'Hardware' or 'Software'.
    4. Extract: Device ID.
    Output a structured summary for the Manager.
    """
)

# 2. Manager: Approval Authority (Simulation)
manager_agent = LlmAgent(
    name="ManagerAgent",
    model=model_config,
    instruction="""
    You are the Technical Support Manager.
    Review the solution from the Analyst.
    - If safe and technical: Reply 'APPROVED'.
    - If dangerous or vague: Reply 'REJECTED'.
    """
)

# 3. Recorder: Database Writer
recorder_agent = LlmAgent(
    name="RecorderAgent",
    model=model_config,
    instruction="""
    You are the Database Recorder.
    IF the Manager APPROVED:
      Use `record_ticket_tool` to save the issue.
    ELSE:
      Do not record.
    """,
    tools=[record_ticket_tool]
)

# 4. Reporter: Observability & Alerting
reporter_agent = LlmAgent(
    name="ReporterAgent",
    model=model_config,
    instruction="""
    You are the Monitoring System.
    1. Check `get_device_issue_count` for the device.
    2. IF count >= 5: Use `send_email_mcp_simulation` to alert 'manager@techco.com'.
    """,
    tools=[get_device_issue_count, send_email_mcp_simulation]
)

# 5. Root: Orchestrator
root_agent = LlmAgent(
    name="TechSupportOrchestrator",
    model=model_config,
    instruction="""
    You manage the Tech Support Pipeline.
    
    [MODE A: NEW TICKET]
    1. Analyst -> Analyze issue
    2. Manager -> Approve/Reject
    3. Recorder -> Save if approved
    4. Reporter -> Check thresholds
    
    [MODE B: QUERY]
    If user asks about past issues (e.g. "How did we fix..."), 
    use `load_memory` to retrieve knowledge.
    """,
    tools=[
        AgentTool(analyst_agent), 
        AgentTool(manager_agent), 
        AgentTool(recorder_agent), 
        AgentTool(reporter_agent),
        load_memory # Enable Long-Term Memory Access
    ]
)

print("âœ… Multi-Agent System Architected.")


from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.plugins.logging_plugin import LoggingPlugin

# Configuration
APP_NAME = "TechSupportSystem"
USER_ID = "tech_admin"
SESSION_ID = "daily_ops_session"

# 1. Create App with Context Compaction AND Plugins
app = App(
    name=APP_NAME,
    root_agent=root_agent,
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=5, 
        overlap_size=2
    ),
    plugins=[LoggingPlugin()] # âœ… Plugins MUST be attached to the App
)

# 2. Initialize Services
# Sessions: Persisted to DB
session_service = DatabaseSessionService(db_url="sqlite:///sessions.db")
# Memory: Knowledge Store
memory_service = InMemoryMemoryService()

# 3. Create Runner (without plugins here)
runner = Runner(
    app=app,
    session_service=session_service,
    memory_service=memory_service
)

print("âœ… Application Runtime Ready.")


import asyncio

async def run_flow(input_text, session_name=SESSION_ID, delay=20):
    """Runs a single interaction flow with safety checks and rate limiting."""
    
    # 1. Session Safety: Ensure session exists
    try:
        session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_name)
        if not session:
             session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_name)
             print(f"   [System] Created new session: {session_name}")
    except Exception as e:
        # If any error occurs during retrieval, force create
        print(f"   [System] Session check failed ({e}), creating new session...")
        session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_name)

    print(f"\n{'='*60}\nğŸ“¥ INPUT: {input_text}\n{'='*60}")
    
    message = types.Content(role="user", parts=[types.Part(text=input_text)])
    
    # 2. Execution
    try:
        async for event in runner.run_async(
            user_id=USER_ID, 
            session_id=session_name, 
            new_message=message
        ):
            # 3. Safe Parsing
            if event.is_final_response() and event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(f"ğŸ¤– AGENT: {part.text}")
    except Exception as e:
        print(f"â�Œ Execution Error: {e}")

    # 4. Rate Limiting (Crucial for Free Tier)
    print(f"\nâ�³ Rate Limit Pause ({delay}s)...")
    await asyncio.sleep(delay)

print("âœ… Workflow Helper Defined.")


issues = [
    "Device PRINTER-X99 is jamming. Solution: Cleared paper path.",
    "Device PRINTER-X99 low toner. Solution: Replaced cartridge.",
    "Device PRINTER-X99 network error. Solution: Reset IP config.",
    "Device PRINTER-X99 overheating. Solution: Cleaned fan filters.",
    "Device PRINTER-X99 not responding. Solution: Hard reboot performed."
]

# Execute the batch
for issue in issues:
    await run_flow(issue)

# --- MEMORY INGESTION ---
# Save the session context to Long-Term Memory so we can query it later
final_session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
await memory_service.add_session_to_memory(final_session)
print("\nğŸ§  Session Knowledge saved to Long-Term Memory.")


# 1. Verify Database
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()
print("\n--- ğŸ—„ï¸� VERIFYING DB RECORDS ---")
count = cursor.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
print(f"Total Tickets Recorded: {count}")
conn.close()

# 2. Test Memory Retrieval
print("\n--- ğŸ§  TESTING MEMORY RETRIEVAL ---")
# We use 'query_session_01' - the agent has NO context here except what it pulls from Memory.
await run_flow(
    "What was the solution for the overheating issue on PRINTER-X99?", 
    session_name="query_session_01"
)


await run_flow(
    "What was the solution for the overheating issue ?", 
    session_name="query_session_01"
)

