# Remark: Code Cell 1 â€“ License Information
# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



# Remark: Code Cell 2 â€“ Environment & API Key Setup

import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete. GOOGLE_API_KEY loaded into environment.")
except Exception as e:
    print(
        "ğŸ”‘ Authentication Error:\n"
        "- Please add 'GOOGLE_API_KEY' in Kaggle > Account > Secrets.\n"
        f"- Details: {e}"
    )



# Remark: Code Cell 3 â€“ Core Imports

import json
import logging
import time
import uuid
import subprocess
import requests

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("workflow_autopilot")

print("âœ… Imports ready.")



# Remark: Code Cell 4 â€“ Retry Config

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

print("âœ… Retry config ready.")



# Remark: Code Cell 5 â€“ BusinessMemoryBank

class BusinessMemoryBank:
    def __init__(self):
        self.entries = []

    def add_entry(self, entry: dict):
        self.entries.append(entry)

    def get_recent(self, limit: int = 5):
        return self.entries[-limit:]

memory_bank = BusinessMemoryBank()
print("âœ… BusinessMemoryBank initialized.")



# Remark: Code Cell 6 â€“ Tool: parse_workday_dump()

def parse_workday_dump(text: str) -> str:
    """
    Extracts tasks from daily blob of meetings, emails, notes.
    Returns JSON string with structure:
    [
      {"task": "...", "context": "..."},
      ...
    ]
    """
    lines = text.split("\n")
    tasks = []

    for line in lines:
        if any(word in line.lower() for word in ["email", "send", "follow", "prepare", "meeting", "schedule", "update", "invoice", "client"]):
            tasks.append({"task": line.strip()})

    return json.dumps(tasks)

print("âœ… Tool ready: parse_workday_dump")



# Remark: Code Cell 7 â€“ Intake Agent

intake_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="intake_agent",
    description="Extracts actionable tasks from end-of-day text.",
    instruction="""
    Convert input text into discrete actionable tasks.
    Use the parse_workday_dump tool to extract task items.
    Output JSON only â€” do not add commentary.
    """,
    tools=[parse_workday_dump]
)

print("âœ… Intake Agent ready.")



# Remark: Code Cell 8 â€“ Tool: score_task_json()

def score_task_json(task_json: str) -> str:
    """
    Takes JSON list of tasks and assigns priority.
    Adds: urgency (1-5), impact (1-5)
    """
    tasks = json.loads(task_json)
    for t in tasks:
        text = t["task"].lower()
        t["urgency"] = 5 if "client" in text else 3
        t["impact"] = 5 if "revenue" in text or "contract" in text else 3
    return json.dumps(tasks)

print("âœ… Tool ready: score_task_json")



# Remark: Code Cell 9 â€“ Prioritization Agent

priority_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="priority_agent",
    description="Assigns urgency + impact to tasks.",
    instruction="""
    Input: JSON list of tasks.
    Call score_task_json to add urgency + impact fields.
    Output JSON only.
    """,
    tools=[score_task_json]
)

print("âœ… Prioritization Agent ready.")



# Remark: Code Cell 10 â€“ Tool: suggest_calendar_slots_json()

def suggest_calendar_slots_json(task_json: str) -> str:
    """
    Suggests calendar blocks for each task.
    Adds calendar_start and duration
    """
    tasks = json.loads(task_json)
    base_hour = 9

    for idx, t in enumerate(tasks):
        t["calendar_start"] = f"{base_hour+idx}:00"
        t["duration_hr"] = 1

    return json.dumps(tasks)

print("âœ… Tool ready: suggest_calendar_slots_json")



# Remark: Code Cell 11 â€“ Tool: format_email_reply_json()

def format_email_reply_json(task_json: str) -> str:
    """
    Generates short professional draft email for each task.
    Returning structure:
    {task: ..., email_draft: ...}
    """
    tasks = json.loads(task_json)

    for t in tasks:
        t["email_draft"] = f"Hi â€” regarding '{t['task']}', I'm on it and will follow up shortly."

    return json.dumps(tasks)

print("âœ… Tool ready: format_email_reply_json")



# Remark: Code Cell 12 â€“ Planner Agent

planner_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="planner_agent",
    description="Turns tasks into scheduled items + email drafts.",
    instruction="""
    You receive JSON tasks with urgency & impact.
    1. Call suggest_calendar_slots_json
    2. Call format_email_reply_json
    Output JSON-only final task list.
    """,
    tools=[suggest_calendar_slots_json, format_email_reply_json]
)

print("âœ… Planner Agent ready.")



# Remark: Code Cell 13 â€“ Create CRM Agent A2A file

crm_agent_code = '''
import json
from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.genai import types

crm_db = []

def register_task_in_crm(task_json: str) -> str:
    task = json.loads(task_json)
    crm_db.append(task)
    return "CRM OK"

crm_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite"),
    name="crm_agent",
    description="External enterprise CRM storing structured task logs.",
    instruction=\"\"\"
    When called with register_task_in_crm,
    store incoming task entries into persistent CRM storage.
    Return 'CRM OK' upon success.
    \"\"\",
    tools=[register_task_in_crm]
)

app = to_a2a(crm_agent, port=9001)
'''

with open("/tmp/crm_server.py", "w") as f:
    f.write(crm_agent_code)

print("ğŸ“� CRM Agent code saved.")



# Remark: Code Cell 14 â€“ Launch A2A CRM Agent

crm_server_process = subprocess.Popen(
    ["uvicorn", "crm_server:app", "--host", "localhost", "--port", "9001"],
    cwd="/tmp",
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env={**os.environ}
)

print("ğŸš€ Starting CRM Agent server...")

for _ in range(30):
    try:
        resp = requests.get(f"http://localhost:9001{AGENT_CARD_WELL_KNOWN_PATH}")
        if resp.status_code == 200:
            print("âœ… CRM Agent running:", f"http://localhost:9001")
            break
    except:
        time.sleep(1)



# Remark: Code Cell 15 â€“ Create Remote CRM Agent proxy

remote_crm_agent = RemoteA2aAgent(
    name="crm_agent",
    description="Remote CRM Agent storing task results.",
    agent_card=f"http://localhost:9001{AGENT_CARD_WELL_KNOWN_PATH}",
)

print("âœ… Remote CRM Agent proxy ready.")



# Remark: Code Cell 16 â€“ Orchestrator Agent

orchestrator_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash"),
    name="orchestrator_agent",
    description="Top-level enterprise workflow automation agent.",
    instruction="""
    You are the central conductor agent.

    Steps:
    1) Send the raw input text to intake_agent
    2) Send extracted tasks to priority_agent
    3) Send scored tasks to planner_agent
    4) Send each finalized task to crm_agent via A2A
    5) Return final JSON summary of completed actions

    Output JSON only.
    """,
    sub_agents=[intake_agent, priority_agent, planner_agent, remote_crm_agent]
)

print("âœ… Orchestrator Agent ready.")



# Remark: Code Cell 17 â€“ Full Workflow Test

async def run_full_test(workday_text: str):
    session_service = InMemorySessionService()
    app_name = "workflow_app"
    user_id = "demo_user"
    session_id = f"session_{uuid.uuid4().hex[:8]}"

    session = await session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )

    runner = Runner(
        agent=orchestrator_agent,
        app_name=app_name,
        session_service=session_service
    )

    content = types.Content(parts=[types.Part(text=workday_text)])

    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                print(part.text)


print("ğŸ§ª Ready to run tests.")



# Remark: Code Cell 18 â€“ Run Test

await run_full_test("""
Email client about contract renewal.
Follow up with accounting regarding invoice.
Schedule meeting with Sarah.
Update CRM with vendor notes.
""")


