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


# ===============================================================
# GlucoCare Assistant – Multi-Agent Diabetes Demo (Modified)
# ===============================================================
# Key features:
# - Multi-agent system (UserMsg, MedTracker, RiskCheck, Reports, Alerts)
# - Sequential + parallel execution
# - Dummy LLM stub for offline Kaggle demo
# - Session & Memory handled internally
# - Simple observability (logs + metrics)
# ===============================================================

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import datetime
import json
import textwrap
import uuid
from concurrent.futures import ThreadPoolExecutor

# -----------------------------
# Observability
# -----------------------------
class Logger:
    def __init__(self):
        self.entries: List[Dict[str, Any]] = []
        self.metrics: Dict[str, int] = {}

    def log(self, level: str, msg: str, **kw):
        self.entries.append({
            "time": datetime.datetime.utcnow().isoformat(),
            "level": level,
            "msg": msg,
            "extra": kw
        })

    def inc(self, metric: str, count: int = 1):
        self.metrics[metric] = self.metrics.get(metric, 0) + count

    def snapshot(self):
        return {"logs": self.entries[-50:], "metrics": self.metrics}

logger = Logger()

# -----------------------------
# Session & Memory
# -----------------------------
@dataclass
class UserSession:
    sid: str
    profile: Dict[str, Any]
    recent_msgs: List[str]

class SessionStore:
    def __init__(self):
        self.store: Dict[str, UserSession] = {}

    def get_session(self, sid: Optional[str]) -> UserSession:
        if sid is None:
            sid = str(uuid.uuid4())
        if sid not in self.store:
            self.store[sid] = UserSession(sid=sid, profile={}, recent_msgs=[])
            logger.log("INFO", "SessionCreated", sid=sid)
        return self.store[sid]

session_store = SessionStore()

@dataclass
class MemoryItem:
    type: str
    data: Dict[str, Any]
    time: str

class LongTermMemory:
    def __init__(self):
        self.items: List[MemoryItem] = []
        self.summaries: List[Dict[str, Any]] = []

    def record(self, type_: str, data: Dict[str, Any]):
        mi = MemoryItem(type=type_, data=data, time=datetime.datetime.utcnow().isoformat())
        self.items.append(mi)
        logger.inc("events_total")
        return mi

    def fetch(self, days: Optional[int] = None):
        if days is None:
            return list(self.items)
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=days)
        return [i for i in self.items if datetime.datetime.fromisoformat(i.time) >= cutoff]

    def compact_old(self, older_than_days: int = 7):
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=older_than_days)
        old = [i for i in self.items if datetime.datetime.fromisoformat(i.time) < cutoff]
        self.items = [i for i in self.items if datetime.datetime.fromisoformat(i.time) >= cutoff]
        if old:
            summary = {
                "from": "MemoryBank",
                "type": "compact",
                "count": len(old),
                "note": "Old events summarized"
            }
            self.summaries.append(summary)
            logger.log("INFO", "MemoryCompacted", summary=summary)

    def export(self):
        return json.dumps({
            "events": [asdict(i) for i in self.items],
            "summaries": self.summaries
        }, indent=2)

memory = LongTermMemory()

# -----------------------------
# Dummy LLM
# -----------------------------
USE_DUMMY = True

def llm_stub(sys_prompt: str, usr_prompt: str) -> str:
    if USE_DUMMY:
        combined = (sys_prompt + " " + usr_prompt).lower()
        if "risk" in sys_prompt.lower():
            risk = "RED" if "chest" in combined else "AMBER" if "shaky" in combined else "GREEN"
            return f"Risk level: {risk} (demo)"
        if "summarize" in sys_prompt.lower():
            return "Weekly summary (demo): few missed meds, check schedule."
        return "Demo response from dummy LLM."
    raise NotImplementedError("Real LLM not integrated")

# -----------------------------
# Agent Message
# -----------------------------
@dataclass
class Msg:
    sender: str
    receiver: str
    type: str
    payload: Dict[str, Any]
    trace: str
    time: str

    @staticmethod
    def new(sender: str, receiver: str, type_: str, payload: Dict[str, Any]) -> "Msg":
        return Msg(sender=sender, receiver=receiver, type=type_, payload=payload,
                   trace=str(uuid.uuid4()), time=datetime.datetime.utcnow().isoformat())

# -----------------------------
# Base Agent
# -----------------------------
class AgentBase:
    def __init__(self, name: str):
        self.name = name

    def process(self, msg: Msg, session: UserSession) -> Optional[Msg]:
        raise NotImplementedError

# --- Tools ---
def run_code_tool(code: str) -> str:
    try:
        env = {}
        exec(code, {"__builtins__": {"len": len, "sum": sum, "max": max}}, env)
        return f"Success. Locals: {env}"
    except Exception as e:
        return f"Error: {e}"

def fake_search_tool(query: str) -> str:
    return f"(Demo search) Pretended to search for '{query}'"

# -----------------------------
# Concrete Agents (med, risk, report)
# -----------------------------
class MedTracker(AgentBase):
    def __init__(self):
        super().__init__("MedTracker")

    def process(self, msg: Msg, session: UserSession) -> Optional[Msg]:
        if msg.type == "ADD_MED":
            memory.record("med_schedule", {"session": session.sid, **msg.payload})
            return Msg.new(self.name, msg.sender, "ADD_MED_OK", {"text": f"Saved {msg.payload}"})
        if msg.type == "LOG_MED":
            memory.record("med_taken", {"session": session.sid, **msg.payload})
            return Msg.new(self.name, msg.sender, "LOG_MED_OK", {"text": f"Logged {msg.payload}"})
        if msg.type == "MED_SUM":
            logs = [i.data for i in memory.fetch(1) if i.type == "med_taken" and i.data.get("session") == session.sid]
            text = "No meds today." if not logs else "\n".join(str(l) for l in logs)
            return Msg.new(self.name, msg.sender, "MED_SUM_RESULT", {"text": text})
        return None

class RiskChecker(AgentBase):
    def __init__(self):
        super().__init__("RiskChecker")

    def process(self, msg: Msg, session: UserSession) -> Optional[Msg]:
        if msg.type != "RISK":
            return None
        result = llm_stub("Assess risk", msg.payload.get("text", ""))
        memory.record("risk_event", {"session": session.sid, "assessment": result})
        return Msg.new(self.name, msg.sender, "RISK_RESULT", {"assessment": result})

# -----------------------------
# Conversation / Orchestrator
# -----------------------------
class ConversationAgent(AgentBase):
    def __init__(self):
        super().__init__("ConversationAgent")
        self.med = MedTracker()
        self.risk = RiskChecker()
        self.executor = ThreadPoolExecutor(max_workers=3)

    def handle_message(self, msg_text: str, sid: Optional[str] = None) -> str:
        session = session_store.get_session(sid)
        session.recent_msgs.append(msg_text)
        text = msg_text.lower()

        # Medication add / log
        if "add med" in text:
            msg = Msg.new(self.name, "MedTracker", "ADD_MED", {"name": "Metformin", "dose": "500mg"})
            reply = self.med.process(msg, session)
            return reply.payload["text"]
        if "took med" in text:
            msg = Msg.new(self.name, "MedTracker", "LOG_MED", {"name": "Metformin", "status": "taken"})
            reply = self.med.process(msg, session)
            return reply.payload["text"]

        # Medication summary
        if "med summary" in text:
            msg = Msg.new(self.name, "MedTracker", "MED_SUM", {})
            reply = self.med.process(msg, session)
            return reply.payload["text"]

        # Risk assessment
        if "feeling" in text or "symptom" in text:
            msg = Msg.new(self.name, "RiskChecker", "RISK", {"text": msg_text})
            reply = self.risk.process(msg, session)
            return reply.payload["assessment"]

        # Code / Search tools
        if text.startswith("run code:"):
            return run_code_tool(msg_text.split(":", 1)[1])
        if text.startswith("search:"):
            return fake_search_tool(msg_text.split(":", 1)[1])

        # Generic fallback
        return llm_stub("Generic assistant", msg_text)

# -----------------------------
# Demo Run
# -----------------------------
conv_agent = ConversationAgent()
demo_msgs = [
    "Add med Metformin 500mg",
    "I took my med",
    "Med summary",
    "I am feeling shaky and sweaty",
    "run code: x = 5 + 7",
    "search: diabetes tips"
]

sid = "demo123"
for m in demo_msgs:
    print(f"User: {m}")
    r = conv_agent.handle_message(m, sid)
    print(f"Assistant: {r}\n{'-'*60}")


