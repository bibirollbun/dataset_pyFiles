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


# ConciergeLite: Single-file Kaggle demo for a Concierge (Campus) Multi-Agent Assistant
# -------------------------------------------------------------------------
from dataclasses import dataclass
from typing import List, Dict, Any
import time, uuid, threading, logging, random

# --------------------------
# Observability & Metrics
# --------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("ConciergeLite")

class Metrics:
    def __init__(self):
        self.counters = {}
        self.lock = threading.Lock()
    def incr(self, k, n=1):
        with self.lock:
            self.counters[k] = self.counters.get(k, 0) + n
    def snapshot(self):
        with self.lock:
            return dict(self.counters)

metrics = Metrics()

# --------------------------
# Utilities & Message type
# --------------------------
@dataclass
class Message:
    id: str
    sender: str
    text: str
    ts: float = None
    def __post_init__(self):
        if self.ts is None:
            self.ts = time.time()

def make_msg(sender: str, text: str) -> Message:
    return Message(id=str(uuid.uuid4()), sender=sender, text=text)

def pretty(conv: List[Message]):
    for m in conv:
        t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(m.ts))
        print(f"[{t}] {m.sender.upper():10}: {m.text}")

# --------------------------
# Simple Tools (databases)
# --------------------------
CAMPUS_FAQ = {
    "library hours": "Library is open 8am-10pm on weekdays and 9am-6pm on weekends.",
    "cafeteria menu": "Today's menu: rice bowl, salad, soup. Check campus portal for full menu.",
    "exam dates cs101": "CS101 final: 2025-12-14, 9:00 AM, Hall A."
}

CAMPUS_RESOURCES = [
    {"name":"Main Library", "type":"library", "floor":"2", "id":"L1"},
    {"name":"Computer Lab A", "type":"lab", "floor":"1", "id":"LAB1"},
    {"name":"Counseling Center", "type":"support", "floor":"G", "id":"S1"},
]

# --------------------------
# Session & Memory
# --------------------------
class SessionService:
    def __init__(self):
        self.sessions = {}  # session_id -> list[Message]
        self.lock = threading.Lock()
    def create(self, sid):
        with self.lock:
            self.sessions.setdefault(sid, [])
    def append(self, sid, msg: Message):
        with self.lock:
            self.sessions.setdefault(sid, []).append(msg)
    def get(self, sid):
        with self.lock:
            return list(self.sessions.get(sid, []))

class SimpleMemory:
    """Tiny long-term memory: user_id -> dict"""
    def __init__(self):
        self.store = {}
        self.lock = threading.Lock()
    def read(self, user):
        with self.lock:
            return dict(self.store.get(user, {}))
    def write(self, user, key, value):
        with self.lock:
            self.store.setdefault(user, {})[key] = value

sessions = SessionService()
memory = SimpleMemory()

# --------------------------
# Tools wrappers
# --------------------------
class CalendarTool:
    def __init__(self):
        self.events = {}  # user -> list of events
        self.lock = threading.Lock()
    def add(self, user, title, when):
        with self.lock:
            ev = {"id": str(uuid.uuid4()), "title": title, "when": when}
            self.events.setdefault(user, []).append(ev)
            return ev
    def list(self, user):
        with self.lock:
            return list(self.events.get(user, []))

calendar = CalendarTool()

class ResourceFinder:
    def find(self, qtype):
        return [r for r in CAMPUS_RESOURCES if r["type"] == qtype] or []

resource_finder = ResourceFinder()

# --------------------------
# Agents (each returns a dict)
# --------------------------
class AgentBase:
    def __init__(self, name):
        self.name = name
    def handle(self, user_id: str, text: str, session_id: str) -> Dict[str,Any]:
        raise NotImplementedError

class FAQAgent(AgentBase):
    def __init__(self): super().__init__("FAQAgent")
    def handle(self, user_id, text, session_id):
        metrics.incr("faq_requests")
        # simple keyword matches
        for k,v in CAMPUS_FAQ.items():
            if k in text.lower():
                resp = v
                sessions.append(session_id, make_msg(self.name, resp))
                return {"agent": self.name, "answer": resp, "score": 1.0}
        # fallback: short web-like answer simulation
        resp = "I couldn't find an exact FAQ match. Try 'library hours' or ask about course dates."
        sessions.append(session_id, make_msg(self.name, resp))
        return {"agent": self.name, "answer": resp, "score": 0.2}

class ScheduleAgent(AgentBase):
    def __init__(self): super().__init__("ScheduleAgent")
    def handle(self, user_id, text, session_id):
        metrics.incr("schedule_requests")
        if "add event" in text.lower() or "remind" in text.lower() or "schedule" in text.lower():
            # naive parse: look for 'tomorrow', 'today' or time tokens
            when = "tomorrow 6pm"
            ev = calendar.add(user_id, "Study session", when)
            resp = f"Added event '{ev['title']}' at {ev['when']} (id:{ev['id']})."
            sessions.append(session_id, make_msg(self.name, resp))
            return {"agent": self.name, "answer": resp, "score": 1.0}
        else:
            evs = calendar.list(user_id)
            resp = f"You have {len(evs)} upcoming events."
            sessions.append(session_id, make_msg(self.name, resp))
            return {"agent": self.name, "answer": resp, "score": 0.8}

class ResourceAgent(AgentBase):
    def __init__(self): super().__init__("ResourceAgent")
    def handle(self, user_id, text, session_id):
        metrics.incr("resource_requests")
        # look for type keywords
        if "lab" in text.lower():
            found = resource_finder.find("lab")
            if found:
                resp = f"Found labs: {', '.join([f['name'] for f in found])}."
                sessions.append(session_id, make_msg(self.name, resp))
                return {"agent": self.name, "answer": resp, "score": 0.95}
        # fallback
        resp = "I can help find labs, libraries, or support centers. Try 'find lab'."
        sessions.append(session_id, make_msg(self.name, resp))
        return {"agent": self.name, "answer": resp, "score": 0.2}

# --------------------------
# Intent Router (parallel ensemble)
# --------------------------
class IntentRouter:
    def __init__(self, agents: List[AgentBase], combine_strategy: str="max_score"):
        self.agents = agents
        self.combine_strategy = combine_strategy

    def route(self, user_id: str, text: str, session_id: str):
        # Store incoming message
        sessions.append(session_id, make_msg(user_id, text))
        metrics.incr("messages_in")
        # Query all agents in parallel (threads)
        results = []
        threads = []
        def run_agent(agent):
            try:
                res = agent.handle(user_id, text, session_id)
                results.append(res)
            except Exception as e:
                logger.exception("Agent failed: %s", agent.name)
        for a in self.agents:
            t = threading.Thread(target=run_agent, args=(a,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=2.0)
        metrics.incr("messages_routed")
        # Combine responses (simple: pick highest score, or merge)
        if not results:
            final = {"agent":"router", "answer":"Sorry, I couldn't handle that right now."}
            sessions.append(session_id, make_msg("Router", final["answer"]))
            return final
        # pick highest score
        best = max(results, key=lambda r: r.get("score",0))
        # but also attach short ensemble summary
        ensemble = {"agent":"ensemble", "selected": best, "candidates": results}
        sessions.append(session_id, make_msg("Router", best["answer"]))
        return ensemble

# --------------------------
# Build system helper
# --------------------------
def build_system():
    faq = FAQAgent()
    sched = ScheduleAgent()
    res = ResourceAgent()
    router = IntentRouter([faq, sched, res])
    return router

# --------------------------
# Evaluation utilities
# --------------------------
def run_dialog(router, user_id, session_id, turns: List[str]):
    for t in turns:
        out = router.route(user_id, t, session_id)
        # pretty-print a short summarised response
        selected = out.get("selected") if isinstance(out, dict) else out
        if selected and isinstance(selected, dict):
            print(f"USER: {t}\nBOT: {selected['answer']}\n")
        else:
            print(f"USER: {t}\nBOT: {out}\n")
    print("Conversation log:")
    pretty(sessions.get(session_id))
    print("Metrics:", metrics.snapshot())

def auto_evaluate(router):
    # Small testcases and expected keywords
    tests = [
        ("Who teaches CS101?", "CS101"),
        ("What are the library hours?", "Library"),
        ("Please add event study tomorrow", "Added event"),
        ("Find me a lab", "Found labs")
    ]
    results = []
    for q, expect in tests:
        sid = "eval_" + str(uuid.uuid4())[:6]
        sessions.create(sid)
        out = router.route("student_x", q, sid)
        # get selected answer
        selected = out.get("selected") if isinstance(out, dict) else None
        answer = selected["answer"] if selected else str(out)
        ok = expect.lower() in answer.lower()
        results.append(ok)
    score = sum(results) / len(results)
    return score, results

# --------------------------
# Demo run (single-file)
# --------------------------
if __name__ == "__main__":
    print("=== ConciergeLite: Single-file demo ===")
    router = build_system()

    # Create session
    sid = "session_demo_1"
    sessions.create(sid)
    user = "priyanka"

    # Example conversation (show features)
    dialog = [
        "Hi, what are library hours?",
        "Can you add event: study group tomorrow evening?",
        "Find me a lab to practice code",
        "Who teaches CS101?"
    ]
    run_dialog(router, user, sid, dialog)

    # Automated evaluation
    print("\nRunning quick automated evaluation...")
    score, detail = auto_evaluate(router)
    print(f"Auto-eval score: {score*100:.1f}%  (details: {detail})")
    print("\nFinal metrics snapshot:", metrics.snapshot())

# -------------------------------------------------------------------------
# End of single-file ConciergeLite


