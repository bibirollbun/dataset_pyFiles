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


# Install correct LangChain split packages and minimal deps for the demo
!pip install -q langchain langchain-openai langchain-community openai python-dotenv httpx sqlalchemy pydantic



# === Concierge Agent (Kaggle-friendly, single-cell) ===
import os, json, datetime, asyncio
from typing import List, Dict, Any, Optional

# If you used Kaggle Secrets, OPENAI_API_KEY will be available as env var.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in Notebook Settings → Secrets, then re-run this cell.")

# LangChain new imports
from langchain_openai import OpenAI
from langchain.agents import Tool, initialize_agent, AgentType

# === LLM (LangChain wrapper for OpenAI) ===
llm = OpenAI(api_key=OPENAI_API_KEY, temperature=0.2, model="gpt-4o-mini")

# === Tools (mocked for Kaggle) ===
MOCK_CALENDAR: List[Dict[str, Any]] = []
MOCK_NOTES:    List[Dict[str, Any]] = []
MEMORY:        List[Dict[str, Any]] = []

async def _create_event(title: str, start_iso: str, end_iso: str, description: str = "") -> Dict[str, Any]:
    ev = {"id": f"ev-{len(MOCK_CALENDAR)+1}", "title": title, "start": start_iso, "end": end_iso, "description": description}
    MOCK_CALENDAR.append(ev)
    return ev

async def _create_note(title: str, content: str) -> Dict[str, Any]:
    note = {"id": f"note-{len(MOCK_NOTES)+1}", "title": title, "content": content, "created_at": datetime.datetime.utcnow().isoformat()}
    MOCK_NOTES.append(note)
    return note

async def _search_web(query: str, max_results: int = 3):
    # Kaggle often cannot reach arbitrary web pages; return mock structured results
    return [{"title": f"Mock result {i+1} for {query}", "snippet": "This is a mock result", "url": "https://example.com"} for i in range(max_results)]

async def _set_reminder(text: str, when_iso: str):
    return await _create_event(f"Reminder: {text}", when_iso, when_iso, description=text)

# helper to run async functions synchronously for LangChain Tools
def syncify(async_fn):
    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            # If inside running loop (rare in Kaggle), create new loop in thread via asyncio.run
            return asyncio.run(async_fn(*args, **kwargs))
        else:
            return asyncio.get_event_loop().run_until_complete(async_fn(*args, **kwargs))
    return wrapper

tools = [
    Tool(name="create_event", func=syncify(_create_event), description="Create a calendar event. Args: title, start_iso, end_iso, description"),
    Tool(name="create_note",  func=syncify(_create_note),  description="Create a note. Args: title, content"),
    Tool(name="search_web",   func=syncify(_search_web),   description="Search the web (mock). Args: query"),
    Tool(name="set_reminder", func=syncify(_set_reminder), description="Set a reminder by creating an event. Args: text, when_iso"),
]

# === Simple memory functions ===
def save_memory(user: str, key: str, value: str):
    MEMORY.append({"user": user, "key": key, "value": value, "time": datetime.datetime.utcnow().isoformat()})

def get_recent_memory(user: str, limit: int = 5):
    items = [m for m in MEMORY if m["user"] == user]
    return items[-limit:]

# === Initialize agent (LangChain) ===
agent_executor = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)

def run_agent(user_text: str, user_id: str = "default_user") -> str:
    # attach short memory context to prompt
    mems = get_recent_memory(user_id)
    mem_text = "\n".join([f"{m['time']}: {m['key']}={m['value']}" for m in mems])
    prompt = f"Memory:\n{mem_text}\n\nUser: {user_text}\n\nAs a concierge agent, decide whether to call tools to take action. Use available tools when appropriate and report outcomes."
    # Run agent synchronously
    try:
        out = agent_executor.run(prompt)
    except Exception as e:
        out = f"Agent error: {e}"
    # store interaction in memory
    save_memory(user_id, "interaction", f"Q: {user_text} -> A: {str(out)[:2000]}")
    return out

# === Quick helper to inspect mock stores ===
def show_state():
    return {"memory_count": len(MEMORY), "notes": MOCK_NOTES, "calendar": MOCK_CALENDAR}

print("Concierge Agent initialized. Call run_agent('your request here').")



print(run_agent("Create a reminder tomorrow at 10:00 UTC to submit the report."))



show_state()


