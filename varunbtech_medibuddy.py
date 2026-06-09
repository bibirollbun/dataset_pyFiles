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
print("NOTE: If you want Gemini LLM responses, add your GOOGLE_API_KEY to Kaggle Secrets with the label 'GOOGLE_API_KEY'.")

# Kaggle secrets usage (works in Kaggle)
try:
    from kaggle_secrets import UserSecretsClient
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("✅ GOOGLE_API_KEY loaded from Kaggle Secrets.")
except Exception as e:
    # not fatal: will run demo in mock/local mode
    print("⚠️ No GOOGLE_API_KEY found in Kaggle Secrets. Running in mock/limited mode.", e)



# ADK imports
from typing import Any, Dict, List
import logging
import sqlite3
import json
import uuid
import time

from google.adk.agents import LlmAgent, Agent
from google.adk.runners import Runner, InMemoryRunner
from google.adk.sessions import DatabaseSessionService, InMemorySessionService
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.tools.tool_context import ToolContext
from google.adk.models.google_llm import Gemini
from google.genai import types

# Logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Retry config for LLM calls
retry_config = types.HttpRetryOptions(
    attempts=3, exp_base=3, initial_delay=1, http_status_codes=[429, 500, 503, 504]
)

print("✅ Imports and retry_config ready.")



# Mock medication database
MED_DB = {
    "atorvastatin": {"dose": "20 mg", "notes": "Take once daily at night", "interactions": ["warfarin"]},
    "metformin": {"dose": "500 mg", "notes": "Take twice daily with meals", "interactions": []},
    "ibuprofen": {"dose": "200-400 mg", "notes": "As needed", "interactions": ["aspirin"]},
}

def get_med_info(med_name: str) -> dict:
    """Return structured med info. Function tool best-practice: return dict with status/data."""
    name = med_name.lower().strip()
    if name in MED_DB:
        return {"status": "success", "med": MED_DB[name], "name": name}
    else:
        # give available list
        return {"status": "error", "error_message": f"Unknown medication: {med_name}. Available: {', '.join(MED_DB.keys())}"}

# Quick test
print(get_med_info("Atorvastatin"))



# Use DatabaseSessionService (SQLite) for persistence
db_url = "sqlite:///my_agent_data.db"
session_service = DatabaseSessionService(db_url=db_url)

# small helper to inspect SQLite events table (for demo)
def inspect_events():
    import sqlite3
    with sqlite3.connect("my_agent_data.db") as conn:
        cur = conn.cursor()
        cur.execute("select app_name, session_id, author, content from events limit 20")
        rows = cur.fetchall()
        for r in rows:
            print(r)

print("✅ Session service configured with SQLite. DB file: my_agent_data.db")



# We'll store reminders in a simple 'reminders' table inside the sqlite DB for demo persistence.

def ensure_reminder_table():
    import sqlite3
    with sqlite3.connect("my_agent_data.db") as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                op_id TEXT PRIMARY KEY,
                user_id TEXT,
                med_name TEXT,
                scheduled_time TEXT,
                status TEXT,
                created_at REAL,
                acknowledged_at REAL
            )
        """)
        conn.commit()

ensure_reminder_table()

def schedule_reminder(tool_context: ToolContext, user_id: str, med_name: str, scheduled_time: str) -> dict:
    """Schedule a reminder (a long-running operation pattern).
    Returns: {"status":"success","operation_id": "..."} immediately.
    """
    op_id = str(uuid.uuid4())
    created = time.time()
    with sqlite3.connect("my_agent_data.db") as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO reminders (op_id, user_id, med_name, scheduled_time, status, created_at, acknowledged_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (op_id, user_id, med_name, scheduled_time, "scheduled", created, None),
        )
        conn.commit()
    return {"status":"success", "operation_id": op_id, "message": f"Reminder scheduled for {med_name} at {scheduled_time}"}

def check_reminder_status(tool_context: ToolContext, operation_id: str) -> dict:
    """Check or update reminder status. Simulates 'resume' of long-running task."""
    with sqlite3.connect("my_agent_data.db") as conn:
        cur = conn.cursor()
        cur.execute("SELECT op_id, user_id, med_name, scheduled_time, status, created_at, acknowledged_at FROM reminders WHERE op_id=?", (operation_id,))
        row = cur.fetchone()
        if not row:
            return {"status":"error", "error_message": "Operation not found"}
        op = dict(op_id=row[0], user_id=row[1], med_name=row[2], scheduled_time=row[3], status=row[4], created_at=row[5], acknowledged_at=row[6])
        # For demo: if scheduled_time is in past, mark acknowledged automatically (simulate user clicked)
        try:
            import datetime
            scheduled_ts = float(op["scheduled_time"])
            if time.time() >= scheduled_ts and op["status"] == "scheduled":
                cur.execute("UPDATE reminders SET status=?, acknowledged_at=? WHERE op_id=?", ("completed", time.time(), operation_id))
                conn.commit()
                op["status"] = "completed"
                op["acknowledged_at"] = time.time()
        except Exception:
            pass
        return {"status":"success", "operation": op}

# Example usage (schedule for 10 seconds from now)
op = schedule_reminder(None, user_id="demo_user", med_name="atorvastatin", scheduled_time=str(time.time()+10))
print(op)
print(check_reminder_status(None, op["operation_id"]))



import sqlite3
with sqlite3.connect("my_agent_data.db") as conn:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print("Tables in DB:", [r[0] for r in cur.fetchall()])



import sqlite3

def ensure_demo_tables():
    with sqlite3.connect("my_agent_data.db") as conn:
        cur = conn.cursor()
        # reminders table (same as before)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                op_id TEXT PRIMARY KEY,
                user_id TEXT,
                med_name TEXT,
                scheduled_time TEXT,
                status TEXT,
                created_at REAL,
                acknowledged_at REAL
            )
        """)
        # adherence_events table (create before compute runs)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS adherence_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                session_id TEXT,
                med_name TEXT,
                taken INTEGER,
                timestamp REAL
            )
        """)
        conn.commit()

# call it now
ensure_demo_tables()
print("✅ Demo tables ensured (reminders, adherence_events).")



import time
import sqlite3

def compute_adherence_rate(user_id: str, window_seconds: int = 7*24*3600) -> dict:
    """
    Compute adherence rate for user_id over the given time window.
    Ensures the adherence_events table exists and returns a safe result (None) when no events.
    """
    ensure_demo_tables()  # call the initializer to be absolutely safe

    cutoff = time.time() - window_seconds
    with sqlite3.connect("my_agent_data.db") as conn:
        cur = conn.cursor()
        # total events (within window)
        cur.execute("SELECT COUNT(*) FROM adherence_events WHERE user_id=? AND timestamp>=?", (user_id, cutoff))
        total = cur.fetchone()[0] or 0
        # taken events
        cur.execute("SELECT COUNT(*) FROM adherence_events WHERE user_id=? AND timestamp>=? AND taken=1", (user_id, cutoff))
        taken = cur.fetchone()[0] or 0

    rate = (taken / total * 100) if total > 0 else None
    return {"status": "success", "total_events": total, "taken": taken, "adherence_rate_percent": rate}



ensure_demo_tables()
print(compute_adherence_rate("demo_user"))



# We'll store adherence events in the session.state for simplicity plus events DB (events table via ADK).
def record_adherence_event(user_id: str, session_id: str, med_name: str, taken: bool):
    # store as an event in our own table for evaluation
    with sqlite3.connect("my_agent_data.db") as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS adherence_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                session_id TEXT,
                med_name TEXT,
                taken INTEGER,
                timestamp REAL
            )
        """)
        cur.execute("INSERT INTO adherence_events (user_id, session_id, med_name, taken, timestamp) VALUES (?, ?, ?, ?, ?)",
                    (user_id, session_id, med_name, 1 if taken else 0, time.time()))
        conn.commit()

def compute_adherence_rate(user_id: str, window_seconds: int = 7*24*3600) -> dict:
    cutoff = time.time() - window_seconds
    with sqlite3.connect("my_agent_data.db") as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM adherence_events WHERE user_id=? AND timestamp>=?", (user_id, cutoff))
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM adherence_events WHERE user_id=? AND timestamp>=? AND taken=1", (user_id, cutoff))
        taken = cur.fetchone()[0]
    rate = (taken/total*100) if total > 0 else None
    return {"status":"success", "total_events": total, "taken": taken, "adherence_rate_percent": rate}

# Test evaluation (no events yet)
print(compute_adherence_rate("demo_user"))



# Tools list: function tools require just the function object
tools = [get_med_info, schedule_reminder, check_reminder_status]

# Agent
medi_agent = LlmAgent(
    name="medi_buddy_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config) if os.environ.get("GOOGLE_API_KEY") else Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="Medication adherence assistant.",
    instruction="""
You are MediBuddy, a medication adherence coach.

When users register meds:
- Ask for medication name, dose, times per day.
- Use get_med_info(med_name) to fetch structured info.
- Use schedule_reminder(user_id, med_name, scheduled_time) to schedule reminders.
- Do NOT perform calculations yourself if precise math required; ask to use tools.

When users confirm they took meds:
- Record adherence events and confirm.

If a dangerous interaction is detected (get_med_info returns interactions), warn and recommend contacting caregiver.
""",
    tools=tools
)

# Runner with LoggingPlugin and DatabaseSessionService
runner = InMemoryRunner(agent=medi_agent, plugins=[LoggingPlugin()])
print("✅ MediBuddy agent and runner configured. Use run_debug for quick prototyping.")



# Quick prototyping run (run_debug) - in production use Runner + sessions
print("Starting prototyping run (example).")

response = await runner.run_debug("Hi, I'm Sam. I take Atorvastatin 20mg nightly and Metformin 500mg twice daily. Can you help schedule reminders?")
# The LoggingPlugin will print trace logs in the notebook output.

# After scheduling, list reminders from DB
print("\nReminders currently in DB:")
with sqlite3.connect("my_agent_data.db") as conn:
    cur = conn.cursor()
    cur.execute("SELECT op_id, user_id, med_name, scheduled_time, status FROM reminders")
    for row in cur.fetchall():
        print(row)



# Full session demo
from google.adk.sessions import DatabaseSessionService
from google.adk.runners import Runner

APP_NAME = "medi_app"
USER_ID = "demo_user"
SESSION_ID = "demo_session_01"

# ensure session_service is the DatabaseSessionService used earlier
session_service = DatabaseSessionService(db_url=db_url)

# Create session (or retrieve)
try:
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
except Exception:
    session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

runner_persistent = Runner(agent=medi_agent, app_name=APP_NAME, session_service=session_service, plugins=[LoggingPlugin()])

# 1) Register user meds (simulate user message)
test_content = types.Content(parts=[types.Part(text="My name is Sam. I take Atorvastatin nightly at 21:00 and Metformin 500mg at 08:00 and 20:00. Please remind me.")])
async for event in runner_persistent.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=test_content):
    if event.is_final_response() and event.content:
        print("Agent:", event.content.parts[0].text)

# 2) Simulate user confirming they took a dose (record_adherence_event)
record_adherence_event(USER_ID, SESSION_ID, "atorvastatin", taken=True)
print("Recorded adherence event for Atorvastatin.")

# 3) Compute adherence metric
print("Adherence summary:", compute_adherence_rate(USER_ID))

# 4) Inspect events DB (quick)
inspect_events()



# If you want to start fresh (warning: will delete db)
import os
if os.path.exists("my_agent_data.db"):
    os.remove("my_agent_data.db")
    print("Deleted my_agent_data.db - next run will recreate.")


