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


!pip install fastapi uvicorn[standard] python-dotenv markdown readability-lxml



import os
folders = ["agent", "tools", "memory", "logs"]
for f in folders:
    os.makedirs(f, exist_ok=True)
print("Folders created:", folders)



%%writefile agent/base_agent.py
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)

class BaseAgent:
    def __init__(self, name: str):
        self.name = name

    def call_llm(self, prompt: str) -> str:
        logging.info(f"{self.name}: call_llm() prompt len={len(prompt)}")
        return mock_llm(prompt)

def mock_llm(prompt: str) -> str:
    lower = prompt.lower()
    if 'research' in lower or 'facts' in lower:
        return "Top facts: walking improves cardiovascular health, aids mood, burns calories. Keywords: walking, exercise, health."
    if 'outline' in lower:
        return "1. Introduction\n2. Benefits of walking\n3. How to start\n4. Tips for consistency\n5. Conclusion"
    if 'draft' in lower:
        return (
            "Walking is a simple exercise with big benefits. Regular walking improves cardiovascular health, "
            "boosts mood, and supports weight management. Below are practical tips to incorporate walking into your daily routine."
        )
    if 'edit' in lower or 'improve' in lower:
        return "Edited: Improved clarity and added SEO-friendly phrases."
    return "(mock response)"



%%writefile agent/memory_service.py
import json
import os
from typing import Any, Dict

MEM_PATH = os.path.join('memory', 'memory_bank.json')

class MemoryBank:
    def __init__(self, path: str = MEM_PATH):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, 'w') as f:
                json.dump({'users': {}}, f)
        self._load()

    def _load(self):
        with open(self.path, 'r') as f:
            self.data = json.load(f)

    def _save(self):
        with open(self.path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def get_user_prefs(self, user_id: str):
        return self.data.get('users', {}).get(user_id, {}).get('prefs', {})

    def save_user_prefs(self, user_id: str, prefs: dict):
        user = self.data.setdefault('users', {}).setdefault(user_id, {})
        user['prefs'] = prefs
        self._save()

    def save_project(self, user_id: str, project_meta: dict):
        user = self.data.setdefault('users', {}).setdefault(user_id, {})
        user.setdefault('projects', []).append(project_meta)
        self._save()

class InMemorySessionService:
    def __init__(self):
        self.sessions = {}

    def create_session(self, session_id: str):
        self.sessions[session_id] = {'history': []}

    def append(self, session_id: str, msg: str):
        self.sessions.setdefault(session_id, {'history': []})['history'].append(msg)

    def get_history(self, session_id: str):
        return self.sessions.get(session_id, {}).get('history', [])



%%writefile agent/research_agent.py
from .base_agent import BaseAgent
from tools.search_tool import search_web

class ResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__("ResearchAgent")

    def research(self, topic: str):
        hits = search_web(topic)
        prompt = f"Research facts about: {topic}. Use hits: {hits}."
        summary = self.call_llm(prompt)
        return {"summary": summary, "hits": hits}



%%writefile agent/writer_agent.py
from .base_agent import BaseAgent

class WriterAgent(BaseAgent):
    def __init__(self):
        super().__init__("WriterAgent")

    def create_outline(self, topic: str, research_summary: str):
        prompt = f"Create an outline about {topic}. Research: {research_summary}"
        return self.call_llm(prompt)

    def write_draft(self, outline: str, prefs: dict):
        prompt = f"Write a draft using outline:\n{outline}\nPreferences:{prefs}"
        return self.call_llm(prompt)



%%writefile agent/editor_agent.py
from .base_agent import BaseAgent

class EditorAgent(BaseAgent):
    def __init__(self):
        super().__init__("EditorAgent")

    def edit(self, draft: str, keywords=None):
        prompt = f"Edit draft: {draft} | Improve clarity and SEO. Keywords:{keywords}"
        return self.call_llm(prompt)



%%writefile tools/search_tool.py
import logging

def search_web(query: str):
    logging.info(f"search_web stub for: {query}")
    return [
        {"title": "Walking and health", "snippet": "Walking improves cardiovascular health and mood."},
        {"title": "Calories burned", "snippet": "30 min brisk walk burns 150-200 calories."}
    ]



%%writefile demo.py
from agent.research_agent import ResearchAgent
from agent.writer_agent import WriterAgent
from agent.editor_agent import EditorAgent
from agent.memory_service import MemoryBank, InMemorySessionService
import os, json

def run_demo(topic, user_id="user1"):
    memory = MemoryBank()
    session = InMemorySessionService()
    session.create_session("session1")

    prefs = memory.get_user_prefs(user_id) or {"tone": "friendly", "length": "medium"}

    research = ResearchAgent()
    writer = WriterAgent()
    editor = EditorAgent()

    print("\n-- Research Phase --")
    r = research.research(topic)
    print(r["summary"])

    print("\n-- Outline Phase --")
    outline = writer.create_outline(topic, r["summary"])
    print(outline)

    print("\n-- Draft Phase --")
    draft = writer.write_draft(outline, prefs)
    print(draft)

    print("\n-- Edit Phase --")
    edited = editor.edit(draft, ["walking","health"])
    print(edited)

    memory.save_project(user_id, {"topic": topic})

    return {"research": r, "outline": outline, "draft": draft, "edited": edited}

if __name__ == "__main__":
    result = run_demo("Benefits of Walking Daily")
    os.makedirs("logs", exist_ok=True)
    with open("logs/last_run.json", "w") as f:
        json.dump(result, f, indent=2)



%%writefile memory/memory_bank.json
{"users": {}}



!ls -R .



!python demo.py



%%writefile server.py
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid
import os
import json
import threading

# Import the demo pipeline
from demo import run_demo

app = FastAPI(title="Automated Blog Writer Agent API")

JOB_STORE = {}
JOB_STORE_FILE = "logs/job_store.json"
JOB_STORE_LOCK = threading.Lock()

def _persist_job_store():
    try:
        os.makedirs(os.path.dirname(JOB_STORE_FILE), exist_ok=True)
        with open(JOB_STORE_FILE, "w") as f:
            json.dump(JOB_STORE, f, indent=2)
    except Exception:
        pass

def _load_job_store():
    try:
        if os.path.exists(JOB_STORE_FILE):
            with open(JOB_STORE_FILE, "r") as f:
                data = json.load(f)
                JOB_STORE.update(data)
    except Exception:
        pass

_load_job_store()

class GenerateRequest(BaseModel):
    topic: str
    tone: Optional[str] = "friendly"
    length: Optional[str] = "medium"
    user_id: Optional[str] = "user1"

class JobStatus(BaseModel):
    job_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None

@app.get("/health")
def health():
    return {"status": "ok", "app": "automated-blog-writer-agent"}

@app.post("/generate_blog")
def generate_blog(req: GenerateRequest):
    try:
        result = run_demo(req.topic, user_id=req.user_id)
        return {"status": "completed", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _job_worker(job_id: str, req: GenerateRequest):
    JOB_STORE[job_id] = {"status": "running", "result": None, "error": None}
    _persist_job_store()
    try:
        result = run_demo(req.topic, user_id=req.user_id)
        JOB_STORE[job_id]["status"] = "completed"
        JOB_STORE[job_id]["result"] = result
    except Exception as e:
        JOB_STORE[job_id]["status"] = "failed"
        JOB_STORE[job_id]["error"] = str(e)
    finally:
        _persist_job_store()

@app.post("/generate_blog_async", response_model=JobStatus)
def generate_blog_async(req: GenerateRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    JOB_STORE[job_id] = {"status": "queued", "result": None, "error": None}
    _persist_job_store()
    background_tasks.add_task(_job_worker, job_id, req)
    return JobStatus(job_id=job_id, status="queued", result=None)

@app.get("/job/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    job = JOB_STORE.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(job_id=job_id, status=job.get("status"), result=job.get("result"), error=job.get("error"))



# Start the server in the background and write logs to server.log
!nohup uvicorn server:app --host 0.0.0.0 --port 8080 &> server.log & disown
# Give it a moment and show startup log lines
!sleep 1
!tail -n 200 server.log



# Test the API (run from inside the notebook)
import requests, time, json

base = "http://127.0.0.1:8080"

# 1) Health check
try:
    print("Health:", requests.get(base + "/health", timeout=5).json())
except Exception as e:
    print("Health check failed:", e)

# 2) Synchronous generate_blog
payload = {
    "topic": "Benefits of Walking Daily",
    "tone": "friendly",
    "length": "medium",
    "user_id": "user1"
}
try:
    resp = requests.post(base + "/generate_blog", json=payload, timeout=120)
    print("Sync status code:", resp.status_code)
    print("Sync response (truncated):")
    print(json.dumps(resp.json(), indent=2)[:1500])
except Exception as e:
    print("Sync request failed:", e)

# 3) Async job example
try:
    resp2 = requests.post(base + "/generate_blog_async", json={"topic":"Quick home workout", "user_id":"user1"})
    job = resp2.json()
    print("Job queued:", job)
    job_id = job.get("job_id")
    # wait a bit, then check
    time.sleep(2)
    status = requests.get(base + f"/job/{job_id}", timeout=60).json()
    print("Job status:", status.get("status"))
    print("Job result (truncated):")
    print(json.dumps(status.get("result"), indent=2)[:1500])
except Exception as e:
    print("Async job test failed:", e)



import threading, time
from uvicorn import Config, Server

def start_server():
    config = Config("server:app", host="127.0.0.1", port=8080, log_level="info")
    server = Server(config)
    server.run()

thread = threading.Thread(target=start_server, daemon=True)
thread.start()

time.sleep(2)
print("Server thread alive:", thread.is_alive())



import requests

try:
    r = requests.get("http://127.0.0.1:8080/health", timeout=5)
    print("Health:", r.json())
except Exception as e:
    print("Health check failed:", e)



import requests, json

payload = {
    "topic": "Benefits of Walking Daily",
    "tone": "friendly",
    "length": "medium",
    "user_id": "user1"
}

try:
    r = requests.post("http://127.0.0.1:8080/generate_blog", json=payload, timeout=60)
    print("STATUS:", r.status_code)
    print(json.dumps(r.json(), indent=2)[:1500])
except Exception as e:
    print("Generate failed:", e)



!pip install --upgrade google-cloud-aiplatform



import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/kaggle/working/service-account.json"
os.environ["GCP_PROJECT"] = "your-project-id"
os.environ["GCP_REGION"] = "us-central1"
os.environ["GEMINI_MODEL"] = "text-bison@001"

# turn ON Gemini
os.environ["USE_GEMINI"] = "1"



%%writefile agent/base_agent.py
import logging
import os

logging.basicConfig(level=logging.INFO)

# Try import Vertex AI SDK
try:
    from google.cloud import aiplatform
    _HAS_AIP = True
except Exception:
    _HAS_AIP = False

class BaseAgent:
    def __init__(self, name: str):
        self.name = name

    def call_llm(self, prompt: str) -> str:
        logging.info(f"{self.name}: LLM call triggered")
        return call_gemini_or_mock(prompt)


def mock_llm(prompt: str) -> str:
    lower = prompt.lower()
    if 'research' in lower:
        return "Mock research summary."
    if 'outline' in lower:
        return "1. Intro\n2. Body\n3. Conclusion"
    if 'draft' in lower:
        return "Mock draft text."
    if 'edit' in lower:
        return "Mock edited version."
    return "(mock response)"


def gemini_llm(prompt: str):
    if not _HAS_AIP:
        raise RuntimeError("Vertex AI SDK not installed.")

    project = os.environ.get("GCP_PROJECT")
    region = os.environ.get("GCP_REGION", "us-central1")
    model_name = os.environ.get("GEMINI_MODEL", "text-bison@001")

    aiplatform.init(project=project, location=region)

    model = aiplatform.TextGenerationModel.from_pretrained(model_name)
    response = model.predict(prompt)
    return response.text


def call_gemini_or_mock(prompt: str):
    """
    If Gemini ready → use Gemini
    Else → fallback to mock
    """
    use_gemini = os.environ.get("USE_GEMINI", "0")
    if use_gemini == "1" and _HAS_AIP and os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        try:
            return gemini_llm(prompt)
        except Exception as e:
            logging.warning(f"Gemini failed → Using mock. Error: {e}")
            return mock_llm(prompt)
    return mock_llm(prompt)



!python demo.py


