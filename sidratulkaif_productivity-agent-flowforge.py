Data model examples
Task object
{
"id": null,
"title": "Draft Q4 marketing plan",
"description": "Create Q4 marketing plan covering channels X,Y,Z. Reference:
slides at /link",
"priority": "High",
"due_date": "2025-12-01",
"assignee": "user:@jane",
"source_meeting_id": "meeting-20251114-1230",
"confidence": 0.92
}
Memory entry
{
"project_id":"proj-1234",
"summary":"Q4 goals focused on user acquisition; marketing lead assigned to
Jane",
"timestamp":"2025-11-14T12:45:00Z"
}


Code scaffold (Python, ADK-friendly)
Repository layout
flowforge/

├── README.md
├── requirements.txt
├── src/
│ ├── agents/
│ │ ├── master_agent.py
│ │ ├── summarizer_agent.py
│ │ ├── task_extractor_agent.py
│ │ ├── assignment_agent.py
│ │ └── monitor_agent.py
│ ├── tools/
│ │ ├── task_manager_tool.py
│ │ ├── org_directory_tool.py
│ │ └── calendar_tool.py
│ ├── memory/
│ │ └── memory_bank.py
│ ├── sessions/
│ │ └── session_service.py
│ ├── observability/
│ │ └── logger.py
│ └── evaluation/
│ └── eval_pipeline.py
├── tests/
│ └── test_task_extraction.py
├── dockerfile
└── deploy/
└── cloud_run.yaml
    
Example: task_extractor_agent.py (simplified)
    
# src/agents/task_extractor_agent.py
from typing import List, Dict
class TaskExtractorAgent:
def __init__(self, llm):
self.llm = llm
def extract_tasks(self, meeting_text: str) -> List[Dict]:
"""Returns list of task dicts with title, description, priority, and due
suggestion."""
prompt = f"Extract action items from the meeting notes. Return JSON
array of tasks with fields: title, description, priority, due_suggestion,
context."
f"\nMeeting:\n{meeting_text[:6000]}"
response = self.llm.call(prompt)
# parse response safely
tasks = parse_json_array(response)
return tasks

(Full code with comments included in repo.)


README.md

# FlowForge — Enterprise Productivity Agent

FlowForge automates meeting summarization, action-item extraction, task
assignment, and follow-ups for teams.

## Quickstart (local)
1. Clone repo.
2. Create Python venv and install: `pip install -r requirements.txt`.
3. Set environment variables: `LLM_API_KEY`, `TASK_MANAGER_API_URL`,
`TASK_MANAGER_API_TOKEN`.
4. Run demo server: `python -m src.server`

## Architecture
See docs and src/ for agents and tools.

## Tests
Run `pytest -q`.

## Deployment
Dockerfile and deploy/cloud_run.yaml included.

## Notes
- Replace LLM adapter in `src/llm_adapter.py` to swap models (Gemini
compatible).
- Do not commit API keys.


requirements.txt

fastapi
uvicorn
requests
pytest
pydantic
python-dotenv
# add the ADK / agent SDK dependency used in your environment


dockerfile

FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8080"]


deploy/cloud_run.yaml

apiVersion: serving.knative.dev/v1
kind: Service
metadata:
name: flowforge
spec:
template:
spec:
containers:
- image: gcr.io/YOUR_PROJECT/flowforge:latest
env:
- name: LLM_API_KEY
valueFrom:
secretKeyRef:
name: llm-secret
key: key
- name: TASK_MANAGER_API_TOKEN
valueFrom:
secretKeyRef:
name: task-manager-secret
key: token


src/llm_adapter.py

# Simple LLM adapter — abstract this to swap providers (Gemini, OpenAI, etc.)
import os
import requests
import json
class LLMAdapter:
def __init__(self):
self.api_key = os.getenv('LLM_API_KEY')
assert self.api_key, "Set LLM_API_KEY env var"
def call(self, prompt: str, max_tokens: int = 512) -> str:
# Minimal stub — replace with real provider call
# For local tests we echo the prompt summary
# In production, call Gemini or OpenAI here
return json.dumps([{"title": "[SIM] Example task", "description":
"Simulated task from prompt.", "priority":"Medium", "due_suggestion": null}])


src/agents/master_agent.py

from src.agents.summarizer_agent import SummarizerAgent
from src.agents.task_extractor_agent import TaskExtractorAgent
from src.agents.assignment_agent import AssignmentAgent
from src.tools.task_manager_tool import TaskManagerTool
from src.sessions.session_service import InMemorySessionService
from src.observability.logger import get_logger
logger = get_logger()
class MasterAgent:
def __init__(self, llm):
self.summarizer = SummarizerAgent(llm)
self.extractor = TaskExtractorAgent(llm)
self.assigner = AssignmentAgent()
self.task_tool = TaskManagerTool()
self.session_svc = InMemorySessionService()
def process_meeting(self, meeting_text: str, meeting_id: str = None):
session = self.session_svc.create_session(meeting_id)
logger.info({'action':'process_meeting','meeting_id':meeting_id})
summary = self.summarizer.summarize(meeting_text)
session['summary'] = summary
tasks = self.extractor.extract_tasks(meeting_text)
created = []
for t in tasks:
assignee = self.assigner.assign(t)
t['assignee'] = assignee
task_obj = self.task_tool.create_task(t)
created.append(task_obj)
# schedule monitor agent or return created tasks
return {'summary': summary, 'tasks': created}


src/agents/summarizer_agent.py

from src.llm_adapter import LLMAdapter
class SummarizerAgent:
def __init__(self, llm=None):
self.llm = llm or LLMAdapter()
def summarize(self, text: str) -> str:
prompt = f"Summarize the meeting notes into a short paragraph (3
sentences):
{text[:4000]}"
result = self.llm.call(prompt)
return result


src/agents/task_extractor_agent.py

import json
from typing import List, Dict
from src.llm_adapter import LLMAdapter
def parse_json_array(s: str):
try:
return json.loads(s)
except Exception:
# best-effort parse fallback
return []
class TaskExtractorAgent:
def __init__(self, llm=None):
self.llm = llm or LLMAdapter()
def extract_tasks(self, meeting_text: str) -> List[Dict]:
prompt = ("Extract action items from the meeting notes. Return a JSON
array of objects "
"with fields: title, description, priority, due_suggestion,
context. Only return JSON.")
response = self.llm.call(prompt + "
10
Meeting:
" + meeting_text[:6000])
tasks = parse_json_array(response)
return tasks


src/agents/assignment_agent.py

from src.tools.org_directory_tool import OrgDirectoryTool
class AssignmentAgent:
def __init__(self):
self.directory = OrgDirectoryTool()
def assign(self, task: dict) -> str:
# Very simple heuristic: match skill_tag in directory
required = task.get('context', '')
candidates = self.directory.search_by_skill(required)
if not candidates:
return 'unassigned'
# pick lowest load
candidates.sort(key=lambda c: c.get('current_load', 0))
return candidates[0]['username']


src/agents/monitor_agent.py

import time
from threading import Thread
from src.observability.logger import get_logger
logger = get_logger()
class MonitorAgent:
def __init__(self, task_tool, poll_interval=3600):
self.task_tool = task_tool
self.poll_interval = poll_interval
self._running = False
def start(self):
self._running = True
Thread(target=self._loop, daemon=True).start()
def _loop(self):
while self._running:
logger.info({'action':'monitor_tick'})
# query task tool for in-progress tasks and send followups if needed
time.sleep(self.poll_interval)


src/tools/task_manager_tool.py

import os
import requests
class TaskManagerTool:
def __init__(self):
self.base = os.getenv('TASK_MANAGER_API_URL')
self.token = os.getenv('TASK_MANAGER_API_TOKEN')
def create_task(self, task_obj: dict) -> dict:
# Example POST to task manager — replace with real API spec
if not self.base or not self.token:
# For demo return a local object
task_obj['id'] = 'local-' + task_obj.get('title','').replace('
','-')[:12]
return task_obj
resp = requests.post(
f"{self.base}/tasks", json=task_obj, headers={'Authorization':
f'Bearer {self.token}'}
)
return resp.json()


src/tools/org_directory_tool.py

# Simple JSON-backed org directory for demo
import json
from pathlib import Path
class OrgDirectoryTool:
def __init__(self, path: str = None):
p = path or 'data/org_directory.json'
if Path(p).exists():
self._data = json.loads(Path(p).read_text())
else:
# demo dataset
self._data = [
{"username":"@jane","role":"PM","skill_tags":
["marketing","strategy"],"current_load":1},
{"username":"@ali","role":"engineer","skill_tags":
["backend","api"],"current_load":2},
]
def search_by_skill(self, text: str):
# naive search: look for any skill tag in text
results = []
for u in self._data:
for tag in u.get('skill_tags',[]):
if tag in text.lower():
results.append(u)
break
return results


src/tools/calendar_tool.py

# Small wrapper to create calendar events (stub)
class CalendarTool:
def create_event(self, title, when, attendees=None):
return {'id': 'evt-local-' + title[:8], 'title': title, 'when': when}


src/memory/memory_bank.py

from collections import deque
class MemoryBank:
def __init__(self, max_entries=100):
self.store = deque(maxlen=max_entries)
def add(self, project_id: str, summary: str, timestamp: str):
self.store.append({'project_id':project_id,'summary':summary,'timestamp':timestamp})
def query(self, project_id: str):
return [s for s in self.store if s['project_id']==project_id]


src/sessions/session_service.py

class InMemorySessionService:
def __init__(self):
self._sessions = {}
def create_session(self, session_id: str = None):
sid = session_id or f"session-{len(self._sessions)+1}"
self._sessions[sid] = {'id': sid}
return self._sessions[sid]
def get_session(self, session_id: str):
return self._sessions.get(session_id)


src/observability/logger.py

import logging
import json
class JsonLogFormatter(logging.Formatter):
def format(self, record):
data = {'level':record.levelname,'msg':record.getMessage()}
if record.args:
data.update({'args': record.args})
return json.dumps(data)
def get_logger(name=__name__):
handler = logging.StreamHandler()
handler.setFormatter(JsonLogFormatter())
logger = logging.getLogger(name)
if not logger.handlers:
logger.addHandler(handler)
logger.setLevel(logging.INFO)
return logger


src/evaluation/eval_pipeline.py

# Simple evaluation harness: compare extracted tasks to gold tasks and print
precision/recall
from typing import List, Dict
def score(pred: List[Dict], gold: List[Dict]):
pred_titles = set([p['title'].lower() for p in pred])
gold_titles = set([g['title'].lower() for g in gold])
tp = len(pred_titles & gold_titles)
precision = tp / max(1, len(pred_titles))
recall = tp / max(1, len(gold_titles))
return {'precision': precision, 'recall': recall}


src/server.py

from fastapi import FastAPI
from pydantic import BaseModel
from src.llm_adapter import LLMAdapter
from src.agents.master_agent import MasterAgent
app = FastAPI()
llm = LLMAdapter()
master = MasterAgent(llm)
class MeetingPayload(BaseModel):
meeting_text: str
meeting_id: str | None = None
@app.post('/ingest')
async def ingest(payload: MeetingPayload):
result = master.process_meeting(payload.meeting_text, payload.meeting_id)
return result

