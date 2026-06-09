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


#“Orion Assistant is a Python-based personal AI agent that routes natural language tasks to specialized skills (research, coding, notes, meetings, automation). 
#It’s hardware-aware and can integrate with local tools for hands-on actions, while still running on free/accessible models and APIs. 
#Unlike typical chat-only bots, Orion is built for end-to-end execution: it plans, calls the right agent, keeps lightweight memory, and can interact with your system when permitted. 
#This is a prototype concept—details and code are on the website and GitHub (https://orion-assistant.com/index.html), running on free-tier resources because we’re bootstrapping.”
# Install deps
!pip install -q --upgrade google-generativeai requests beautifulsoup4

from kaggle_secrets import UserSecretsClient
import google.generativeai as genai
import os, json, time, requests, xml.etree.ElementTree as ET
from google.api_core.exceptions import ResourceExhausted

# --- API key from Kaggle secrets ---
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("GEMINI_API_KEY")

# Option 1: configure Gemini directly (recommended)
genai.configure(api_key=api_key)

# (optional) also expose as env var for consistency
os.environ["GOOGLE_API_KEY"] = api_key

# Model config
GEMINI_MODEL = "models/gemini-2.5-flash"   # or whatever model the hackathon allows

def ask_gemini(prompt: str, system: str | None = None, attempts: int = 3, base_delay: float = 5.0) -> str:
    for i in range(attempts):
        try:
            m = genai.GenerativeModel(GEMINI_MODEL, system_instruction=system) if system else genai.GenerativeModel(GEMINI_MODEL)
            res = m.generate_content(prompt)
            return res.text or ""
        except ResourceExhausted as e:
            delay = base_delay * (i + 1)
            print(f"Quota hit, retrying in {delay}s...")
            time.sleep(delay)
        except Exception as e:
            print(f"Gemini error: {e}")
            break
    return "Quota exhausted or error."

# --- Planner ---
def plan(command: str):
    system = (
        "You are a router. Return JSON with keys: agent, info. "
        "Agents: code, papers, notes_add, notes_read, notes_clear, meeting, unknown. "
        "Only return JSON."
    )
    raw = ask_gemini(f"Command: {command}", system=system)
    try:
        if "{" in raw:
            raw = raw[raw.index("{"):]
        return json.loads(raw)
    except Exception:
        # Fallback keyword routing
        lc = command.lower()
        if "paper" in lc or "arxiv" in lc or "research" in lc:
            return {"agent": "papers", "info": {"topic": command}}
        if "note" in lc and "read" in lc:
            return {"agent": "notes_read", "info": {"topic": command}}
        if "note" in lc and ("add" in lc or "save" in lc):
            return {"agent": "notes_add", "info": {"topic": "general", "content": command}}
        if "note" in lc and ("clear" in lc or "delete" in lc):
            return {"agent": "notes_clear", "info": {"topic": "general"}}
        if "meeting" in lc:
            if "start" in lc:
                return {"agent": "meeting", "info": {"action": "start", "topic": command}}
            if "stop" in lc or "end" in lc or "summarize" in lc:
                return {"agent": "meeting", "info": {"action": "stop"}}
            if "add" in lc:
                return {"agent": "meeting", "info": {"action": "add", "content": command}}
        if "code" in lc or "python" in lc or "function" in lc:
            return {"agent": "code", "info": {"prompt": command}}
        return {"agent": "unknown", "info": {}}


# --- Papers (ArXiv) ---
def search_papers(topic, max_results=5):
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{topic}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("atom:entry", ns)[:max_results]:
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        link = ""
        for l in entry.findall("atom:link", ns):
            if l.attrib.get("type") == "text/html":
                link = l.attrib.get("href", "")
                break
        authors = [a.findtext("atom:name", default="", namespaces=ns) or "" for a in entry.findall("atom:author", ns)]
        papers.append({"title": title, "link": link, "authors": authors})
    return papers

# --- Notes (in-memory) ---
NOTES = {}
def notes_add(topic, content):
    NOTES.setdefault(topic, []).append(content)
    return f"Note added to {topic}."
def notes_read(topic):
    items = NOTES.get(topic, [])
    return "\n".join(items) if items else f"No notes for {topic}."
def notes_clear(topic):
    NOTES.pop(topic, None)
    return f"Cleared notes for {topic}."

# --- Meeting (text) ---
MEETING = {"active": False, "topic": "", "log": []}
def meeting_start(topic):
    MEETING.update({"active": True, "topic": topic, "log": []})
    return f"Meeting started on '{topic}'. Say 'add' notes, then 'stop' to summarize."
def meeting_add(text):
    if not MEETING["active"]:
        return "No active meeting."
    MEETING["log"].append(text)
    return "Captured."
def meeting_stop():
    if not MEETING["active"]:
        return "No active meeting."
    MEETING["active"] = False
    transcript = "\n".join(MEETING["log"])
    prompt = f"Summarize this meeting with 3-7 bullets and action items.\nTopic: {MEETING['topic']}\nTranscript:\n{transcript}"
    summary = ask_gemini(prompt, system="You summarize meetings concisely.")
    notes_add(MEETING["topic"], summary)
    return summary or "Empty summary."

# --- Dispatcher ---
def handle(command: str):
    p = plan(command)
    agent = p.get("agent")
    info = p.get("info", {})
    if agent == "code":
        prompt = info.get("prompt") or command
        sys = "You are a coding copilot. Output concise code with minimal explanation."
        return ask_gemini(prompt, system=sys)
    if agent == "papers":
        topic = info.get("topic") or command
        papers = search_papers(topic)
        if not papers: return "No papers found."
        return "\n\n".join(
            f"{i+1}. {p['title']}\n   Authors: {', '.join(p['authors'])}\n   Link: {p['link']}"
            for i, p in enumerate(papers)
        )
    if agent == "notes_add":
        return notes_add(info.get("topic") or "general", info.get("content") or command)
    if agent == "notes_read":
        return notes_read(info.get("topic") or "general")
    if agent == "notes_clear":
        return notes_clear(info.get("topic") or "general")
    if agent == "meeting":
        action = info.get("action", "")
        if action == "start": return meeting_start(info.get("topic") or "meeting")
        if action == "add":   return meeting_add(info.get("content") or command)
        if action == "stop":  return meeting_stop()
    return "I couldn't route that command."

# --- Minimal demo (run one at a time to avoid 429) ---
print(handle("Find recent papers on diffusion models"))
#print(handle("Write a Python function for quicksort"))
#print(handle("Start meeting about roadmap"))
#print(handle("Add this to the meeting: prioritize latency"))
#print(handle("Stop meeting and summarize"))



!ls -R /kaggle/input/gemini-3
!head -100 /kaggle/input/gemini-3/README



import pandas as pd

# Replace with the actual columns/values per the competition rules
sub = pd.DataFrame({
    "id": [...],
    "prediction": [...]
})
sub.to_csv("/kaggle/working/submission.csv", index=False)
print(sub.head())





