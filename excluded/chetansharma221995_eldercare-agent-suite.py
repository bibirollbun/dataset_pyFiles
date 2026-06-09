# !pip install google-adk
# !pip install --quiet google-genai nest_asyncio


# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import json
import logging
import asyncio
from datetime import datetime, timedelta

# Patch jupyter event loop so run_until_complete works
import nest_asyncio
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ElderCareAgentSuite")

from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )




# Paste this single cell into your Kaggle notebook (replace previous demo cell)
import asyncio
import uuid
import logging
from typing import Any, Optional

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.function_tool import FunctionTool
from google.genai import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eldercare-capstone")

# --- Tools ---
def health_check(patient_name: str) -> str:
    return f"{patient_name} is stable and healthy."

def medication_reminder(patient_name: str) -> str:
    return f"Next medication for {patient_name} is at 8 PM."

def schedule_appointment(patient_name: str, date: str, doctor: Optional[str] = None) -> str:
    doc = f" with Dr. {doctor}" if doctor else ""
    ref = str(uuid.uuid4())[:8]
    return f"Appointment scheduled for {patient_name}{doc} on {date}. Ref: {ref}"

def emergency_alert(patient_name: str, contact_number: str) -> str:
    return f"Emergency alert sent for {patient_name} to {contact_number}."

tools = [
    FunctionTool(func=health_check),
    FunctionTool(func=medication_reminder),
    FunctionTool(func=schedule_appointment),
    FunctionTool(func=emergency_alert),
]

# --- Agent + App + Runner ---
agent = LlmAgent(
    name="eldercare_agent",
    model=Gemini(model="gemini-2.5-flash-lite"),
    instruction="""
You are an ElderCare assistant. Use available tools for external actions:
- health_check(patient_name)
- medication_reminder(patient_name)
- schedule_appointment(patient_name, date, doctor)
- emergency_alert(patient_name, contact_number)
Extract parameters and call tools when appropriate.
Keep replies concise and clear.
""",
    tools=tools,
)

app = App(name="eldercare_app", root_agent=agent, resumability_config=ResumabilityConfig(is_resumable=True))
session_service = InMemorySessionService()
runner = Runner(app=app, session_service=session_service)

# --- Helpers to extract text and tool outputs ---
def extract_text_parts(event: Any) -> str:
    if not getattr(event, "content", None):
        return ""
    parts = getattr(event.content, "parts", []) or []
    texts = []
    for p in parts:
        if hasattr(p, "text") and p.text:
            texts.append(p.text)
    return " ".join(texts).strip()

def extract_function_call(event: Any) -> Optional[dict]:
    # Some ADK responses include function_call parts; find and return the dict
    if not getattr(event, "content", None):
        return None
    for p in getattr(event.content, "parts", []) or []:
        if hasattr(p, "function_call") and p.function_call:
            # p.function_call often is a simple object; convert to dict-like
            return {"name": getattr(p.function_call, "name", None), "args": getattr(p.function_call, "args", None)}
    return None

def extract_tool_output_from_event(event: Any) -> str:
    # After a function is executed, the tool output usually appears as text parts in later events.
    return extract_text_parts(event)

# --- Demo runner: prints both agent text and tool outputs clearly ---
async def run_demo(user_id: str = "demo_user"):
    session = await session_service.create_session(app_name=app.name, user_id=user_id)
    print("âœ… Session created:", session.id, "\n")
    demo_prompts = [
        "Check health for patient John Doe.",
        "When is John Doe's next medicine?",
        "Schedule appointment for John Doe with Dr. Sharma on 2025-12-05.",
        "Send emergency alert for patient John Doe to +911234567890."
    ]

    for prompt in demo_prompts:
        print("USER âžœ", prompt)
        content = types.Content(role="user", parts=[types.Part(text=prompt)])
        # Run and capture streaming events
        async for event in runner.run_async(user_id=user_id, session_id=session.id, new_message=content):
            # 1) natural language reply parts (assistant)
            assistant_text = extract_text_parts(event)
            if assistant_text:
                print("AGENT âžœ", assistant_text)

            # 2) function_call metadata
            fc = extract_function_call(event)
            if fc:
                print("AGENT â†’ FUNCTION_CALL:", fc)

            # 3) tool outputs (may appear in subsequent events as text)
            tool_out = extract_tool_output_from_event(event)
            # if tool outputs are different from assistant_text they will still appear; print if meaningful
            # to avoid duplicates, we already printed assistant_text; print tool_out only if it adds new info
            # (practical simple rule: if tool_out is non-empty and not same as assistant_text)
            if tool_out and tool_out != assistant_text:
                print("TOOL OUTPUT âžœ", tool_out)
        print("-" * 50)

    print("\nDemo finished.")

# Run in notebook
try:
    asyncio.run(run_demo())
except RuntimeError:
    # nested event loop case in some notebook environments:
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.get_event_loop().run_until_complete(run_demo())


