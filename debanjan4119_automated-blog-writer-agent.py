# Kaggle secrets wiring for Gemini / ADK
import os
from kaggle_secrets import UserSecretsClient

try:
    _usc = UserSecretsClient()
    _gkey = _usc.get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = _gkey
    print("✅ GOOGLE_API_KEY loaded into environment.")
except Exception as e:
    # If this raises, Kaggle secrets may not be configured; ADK will expect GOOGLE_API_KEY in env.
    raise RuntimeError(f"Failed to load GOOGLE_API_KEY from Kaggle secrets: {e}")



# Configuration
import os
from pathlib import Path

ROOT = Path.cwd()
MEMORY_DB = ROOT / "blog_memory.db"
METRICS_FILE = ROOT / "blog_metrics.json"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")  # override via env if needed
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]  # required for ADK/Gemini calls
print("Notebook root:", ROOT)
print("Memory DB:", MEMORY_DB)
print("Metrics file:", METRICS_FILE)
print("Gemini model:", GEMINI_MODEL)



# Gemini client wrapper (ADK / google.genai usage)
# The wrapper constructs a Gemini model instance and provides a generate_text() helper.
from google.adk.models.google_llm import Gemini as _GeminiModel

def create_gemini(model_name=GEMINI_MODEL, api_key=GOOGLE_API_KEY, retry=retry_config):
    # Construct a Gemini client (ADK model wrapper). Exact ADK initialization may vary by ADK version.
    return _GeminiModel(model=model_name, api_key=api_key, retry=retry)

_gemini = create_gemini()

def generate_text(prompt: str, temperature: float = 0.2, max_tokens: int = 512):
    # Use Gemini client to generate text. Result handling follows ADK/GenAI response shape.
    resp = _gemini.generate_text(prompt=prompt, temperature=temperature, max_tokens=max_tokens)
    # ADK response may differ by version; prefer common attributes
    if hasattr(resp, "text"):
        return resp.text
    # fallback to serialized content if object shape differs
    try:
        return json.dumps(resp)  # will rarely be ideal, but avoids crashes if response shape varies
    except Exception:
        return str(resp)



# Session service (in-memory) and simple SQLite for long-term drafts/preferences
import sqlite3

session_service = InMemorySessionService()

def init_db(path=MEMORY_DB):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS prefs (user_id TEXT PRIMARY KEY, prefs TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS drafts (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, content TEXT, metadata TEXT, created_at REAL)""")
    conn.commit()
    conn.close()

def save_pref(user_id: str, prefs: dict, path=MEMORY_DB):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO prefs (user_id, prefs) VALUES (?, ?)", (user_id, json.dumps(prefs)))
    conn.commit()
    conn.close()

def get_pref(user_id: str, path=MEMORY_DB):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("SELECT prefs FROM prefs WHERE user_id=?", (user_id,))
    r = c.fetchone()
    conn.close()
    return json.loads(r[0]) if r else {}

def save_draft(title: str, content: str, metadata: dict=None, path=MEMORY_DB):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO drafts (title, content, metadata, created_at) VALUES (?, ?, ?, ?)",
              (title, content, json.dumps(metadata or {}), time.time()))
    conn.commit()
    conn.close()

def list_drafts(limit=10, path=MEMORY_DB):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("SELECT id, title, metadata, created_at FROM drafts ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return rows

init_db()
print("✅ Session service and long-term memory initialized.")



# Observability helpers
import logging, time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("automated_blog_agent")

def record_metric(name: str, value, tags: dict=None, path=METRICS_FILE):
    entry = {"metric": name, "value": value, "tags": tags or {}, "ts": time.time()}
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")

logger.info("Observability ready.")



# Web search tool (Google Custom Search / SerpAPI style)
# Replace `SEARCH_ENDPOINT` and usage with your chosen search provider; uses requests and GOOGLE_API_KEY.
import requests
import urllib.parse

def web_search(query: str, top_k: int = 5):
    # Example: Google Custom Search JSON API (requires enabling in Google Cloud and a Search Engine ID)
    # If using SerpAPI, adapt endpoint and params accordingly.
    SEARCH_API_KEY = os.getenv("GOOGLE_API_KEY")  # reuse same key if appropriate
    SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID", "")  # set as Kaggle secret if using Custom Search
    if not SEARCH_ENGINE_ID:
        raise RuntimeError("SEARCH_ENGINE_ID not set. Provide a search engine id via env 'SEARCH_ENGINE_ID'.")
    base = "https://www.googleapis.com/customsearch/v1"
    params = {"key": SEARCH_API_KEY, "cx": SEARCH_ENGINE_ID, "q": query, "num": min(10, top_k)}
    r = requests.get(base, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    items = data.get("items", [])
    snippets = []
    for it in items[:top_k]:
        snippets.append({"title": it.get("title"), "snippet": it.get("snippet"), "url": it.get("link")})
    record_metric("search.requests", 1, {"query": query})
    return snippets



# Simple in-process messaging for agents (A2A style)
from collections import defaultdict
import queue

INBOXES = defaultdict(queue.Queue)

def a2a_send(to_agent: str, message: dict):
    INBOXES[to_agent].put(message)

def a2a_receive(agent_name: str, timeout: float = 0.2):
    try:
        return INBOXES[agent_name].get(timeout=timeout)
    except queue.Empty:
        return None



# Agent implementations: Coordinator orchestrates; Research, Writer, Editor, Scheduler are subagents.
# Each agent runs a loop in a background thread and communicates via a2a_send / a2a_receive.

class CoordinatorAgent:
    def __init__(self, user_id="user1"):
        self.user_id = user_id

    def submit_request(self, request: dict, wait_seconds: int = 30):
        logger.info("Coordinator: submit request %s", request.get("title"))
        record_metric("coordinator.requests", 1, {"user": self.user_id})
        prefs = get_pref(self.user_id) or {"tone": "neutral", "audience": "general"}
        job = {"job_id": str(uuid.uuid4()), "request": request, "prefs": prefs}
        # publish job to research and writer (parallel start)
        a2a_send("research", {"type":"job","job":job})
        a2a_send("writer", {"type":"job","job":job})
        # wait for final draft from editor
        start = time.time()
        while time.time() - start < wait_seconds:
            m = a2a_receive("coordinator", timeout=0.5)
            if m and m.get("type") == "final_draft":
                record_metric("coordinator.completed", 1, {"job_id": job["job_id"]})
                return m
        logger.warning("Coordinator: timeout waiting for final_draft")
        return {"error": "timeout"}

class ResearchAgent:
    def run(self):
        while True:
            m = a2a_receive("research", timeout=1.0)
            if not m:
                continue
            if m.get("type") == "job":
                job = m["job"]
                topic = job["request"].get("topic")
                snippets = web_search(topic, top_k=5)
                context = " ".join([s["snippet"] for s in snippets])[:2000]
                job["research"] = {"snippets": snippets, "context": context}
                a2a_send("writer", {"type":"research_result","job":job})

class WriterAgent:
    def run(self):
        while True:
            m = a2a_receive("writer", timeout=1.0)
            if not m:
                continue
            if m.get("type") == "job" or m.get("type") == "research_result":
                job = m["job"] if m.get("type") == "job" else m["job"]
                prompt = f"""
You are an expert blog writer. Produce a concise outline and a first draft for the topic below.
Topic: {job['request'].get('topic')}
Preferences: {job['prefs']}
Research context: {job.get('research', {}).get('context', '')}
Write an outline first, then a first draft.
"""
                draft = generate_text(prompt, temperature=0.2, max_tokens=1200)
                a2a_send("editor", {"type":"draft","job":job,"draft":draft})

class EditorAgent:
    def run(self):
        while True:
            m = a2a_receive("editor", timeout=1.0)
            if not m:
                continue
            if m.get("type") == "draft":
                draft = m["draft"]
                # LLM-as-judge: score and improve
                score_prompt = f"Score this draft from 1-10 for clarity, factuality, and SEO. Suggest improvements.\n\n{draft}"
                score = generate_text(score_prompt, temperature=0.0, max_tokens=200)
                improve_prompt = f"Apply the suggestions to improve the draft. Draft:\n\n{draft}\n\nSuggestions:\n{score}"
                improved = generate_text(improve_prompt, temperature=0.2, max_tokens=1600)
                save_draft(m["job"]["request"].get("title","untitled"), improved, metadata={"score_eval": score})
                a2a_send("coordinator", {"type":"final_draft","job_id": m["job"]["job_id"], "draft": improved, "score": score})

class SchedulerAgent:
    def run(self):
        while True:
            m = a2a_receive("scheduler", timeout=1.0)
            if not m:
                continue
            if m.get("type") == "publish":
                # placeholder for scheduling / publishing integration (WordPress / API)
                payload = m.get("payload")
                record_metric("scheduler.publish", 1, {"title": payload.get("title")})
                a2a_send(m.get("reply_to"), {"type":"published","payload": {"status": "published", "title": payload.get("title")}})

# Agent thread starters
def start_agents():
    threads = []
    ra = ResearchAgent(); wa = WriterAgent(); ea = EditorAgent(); sa = SchedulerAgent()
    threads.append(threading.Thread(target=ra.run, daemon=True)); threads.append(threading.Thread(target=wa.run, daemon=True))
    threads.append(threading.Thread(target=ea.run, daemon=True)); threads.append(threading.Thread(target=sa.run, daemon=True))
    for t in threads:
        t.start()
    logger.info("All subagents started.")
    return threads



# Demo: submit a blog request and get the final draft
start_agents()
coord = CoordinatorAgent(user_id="author_1")
# sample preferences persisted
save_pref("author_1", {"tone":"conversational","audience":"technical","keywords":["automation","blogs","content"]})

request = {
    "title": "Automating Blog Production with AI",
    "topic": "How small teams can automate blog writing workflows using LLMs and agent pipelines",
    "require_approval": False
}

result = coord.submit_request(request, wait_seconds=60)
if result.get("error"):
    print("Error:", result)
else:
    print("=== FINAL DRAFT PREVIEW ===\n")
    print(result.get("draft")[:3000])
    print("\n=== METADATA ===")
    print("Score output (editor):", result.get("score"))
    # show saved drafts
    print("\nSaved drafts (recent):", list_drafts(5))


