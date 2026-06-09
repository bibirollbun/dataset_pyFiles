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
# DiaCare Pro â€“ Advanced Multi-Agent Diabetes Assistant (Capstone)
# ===============================================================
# Features shown:
# - Multi-agent system (Conversation, Medication, Risk, Report, Notification, Scheduler)
# - LLM-powered reasoning stub with easy Gemini integration
# - Sequential + parallel agent execution
# - Loop-style SchedulerAgent (long-running / pause-resume pattern)
# - A2A-like message protocol between agents
# - Custom tools (code execution, toy search)
# - Sessions & Memory (SessionService + MemoryBank)
# - Context compaction (simple summarization of old events)
# - Observability: logging, tracing, simple metrics
# - Simple agent evaluation harness
# - Deployment stub (FastAPI-style handler example)
# ===============================================================

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Callable
import datetime
import json
import textwrap
import uuid
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------------
# 0. Observability: Logger, Tracing & Metrics
# ---------------------------------------------------------------

class Observability:
    def __init__(self):
        self.logs: List[Dict[str, Any]] = []
        self.metrics: Dict[str, int] = {}

    def log(self, level: str, message: str, **kwargs):
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            "extra": kwargs,
        }
        self.logs.append(entry)

    def info(self, message: str, **kwargs):
        self.log("INFO", message, **kwargs)

    def error(self, message: str, **kwargs):
        self.log("ERROR", message, **kwargs)

    def incr(self, metric_name: str, amount: int = 1):
        self.metrics[metric_name] = self.metrics.get(metric_name, 0) + amount

    def dump(self):
        return {
            "logs": self.logs[-50:],          # last 50 log entries
            "metrics": self.metrics,
        }

obs = Observability()


# ---------------------------------------------------------------
# 1. Sessions & Memory (SessionService + MemoryBank)
# ---------------------------------------------------------------

@dataclass
class SessionState:
    session_id: str
    user_profile: Dict[str, Any]
    short_context: List[str]    # recent messages or summaries

class SessionService:
    """Simple in-memory session/state management."""
    def __init__(self):
        self.sessions: Dict[str, SessionState] = {}

    def get_or_create(self, session_id: Optional[str]) -> SessionState:
        if session_id is None:
            session_id = str(uuid.uuid4())
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(
                session_id=session_id,
                user_profile={},
                short_context=[]
            )
            obs.info("SessionCreated", session_id=session_id)
        return self.sessions[session_id]

session_service = SessionService()

@dataclass
class MemoryEvent:
    event_type: str
    payload: Dict[str, Any]
    timestamp: str

class MemoryBank:
    """Long-term memory with simple context compaction."""
    def __init__(self):
        self.events: List[MemoryEvent] = []
        self.compacted_summaries: List[Dict[str, Any]] = []

    def add_event(self, event_type: str, payload: Dict[str, Any]) -> MemoryEvent:
        e = MemoryEvent(
            event_type=event_type,
            payload=payload,
            timestamp=datetime.datetime.utcnow().isoformat()
        )
        self.events.append(e)
        obs.incr("events_total")
        return e

    def get_events(self, since_days: Optional[int] = None) -> List[MemoryEvent]:
        if since_days is None:
            return list(self.events)
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=since_days)
        result = []
        for e in self.events:
            ts = datetime.datetime.fromisoformat(e.timestamp)
            if ts >= cutoff:
                result.append(e)
        return result

    def compact_old_events(self, older_than_days: int = 7):
        """Context compaction â€“ summarize older events and drop details."""
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=older_than_days)
        old_events = []
        keep_events = []
        for e in self.events:
            ts = datetime.datetime.fromisoformat(e.timestamp)
            if ts < cutoff:
                old_events.append(e)
            else:
                keep_events.append(e)
        self.events = keep_events

        if not old_events:
            return

        # Simple heuristic summarization (could use LLM)
        summary = {
            "from": "MemoryBank",
            "summary_type": "compact",
            "event_count": len(old_events),
            "time_range": f"older than {older_than_days} days",
            "note": "Compact summary for long-term history (no details)."
        }
        self.compacted_summaries.append(summary)
        obs.info("ContextCompacted", summary=summary)

    def to_json(self) -> str:
        return json.dumps({
            "events": [asdict(e) for e in self.events],
            "compacted": self.compacted_summaries,
        }, indent=2)

memory_bank = MemoryBank()


# ---------------------------------------------------------------
# 2. LLM Stub (Dummy + Gemini integration point)
# ---------------------------------------------------------------

USE_DUMMY_LLM = True  # Set to False when wiring Gemini / Vertex AI

def llm_chat(system_prompt: str, user_prompt: str) -> str:
    """
    Dummy LLM; safe for Kaggle offline demo.
    Replace with Gemini / Vertex AI call for real reasoning.
    """
    if USE_DUMMY_LLM:
        text = (system_prompt + "\n\n" + user_prompt).lower()

        # Risk triage behavior
        if "classify risk" in system_prompt.lower():
            if any(k in text for k in ["shaky", "sweaty", "blurred", "confused", "faint"]):
                risk = "AMBER"
            elif any(k in text for k in ["chest pain", "severe", "unconscious"]):
                risk = "RED"
            else:
                risk = "GREEN"
            return textwrap.dedent(f"""
            Risk level: {risk}
            Explanation: Rule-based demo only. NOT medical advice.
            Next step (non-medical guidance): If symptoms persist or worsen, contact your clinician.
            """).strip()

        # Weekly summary behavior
        if "summarize the following health events" in system_prompt.lower():
            return (
                "Doctor Summary (Demo):\n"
                "- Some episodes with symptoms and missed medications.\n"
                "- Recommend reviewing routine and adherence.\n\n"
                "Patient Summary (Demo):\n"
                "You had a few days where you did not feel well or missed medicines.\n"
                "Try to keep a regular schedule and discuss this with your doctor."
            )

        # Generic fallback
        return (
            "This is a demo response from the dummy LLM. "
            "In production, this would be powered by Gemini."
        )

    # ------------ REAL GEMINI HOOK (pseudo) --------------------
    # from google import genai
    # client = genai.Client(api_key="YOUR_KEY")
    # response = client.responses.create(
    #     model="gemini-1.5-pro",
    #     contents=[
    #         {"role": "system", "parts": system_prompt},
    #         {"role": "user", "parts": user_prompt},
    #     ]
    # )
    # return response.text
    # -----------------------------------------------------------
    raise NotImplementedError("LLM integration not configured.")


# ---------------------------------------------------------------
# 3. A2A-style Message Protocol
# ---------------------------------------------------------------

@dataclass
class AgentMessage:
    sender: str
    receiver: str
    msg_type: str
    payload: Dict[str, Any]
    trace_id: str
    timestamp: str

    @staticmethod
    def create(sender: str, receiver: str, msg_type: str, payload: Dict[str, Any]) -> "AgentMessage":
        return AgentMessage(
            sender=sender,
            receiver=receiver,
            msg_type=msg_type,
            payload=payload,
            trace_id=str(uuid.uuid4()),
            timestamp=datetime.datetime.utcnow().isoformat()
        )


# ---------------------------------------------------------------
# 4. BaseAgent + Tools
# ---------------------------------------------------------------

class BaseAgent:
    def __init__(self, name: str):
        self.name = name

    def handle(self, msg: AgentMessage, session: SessionState) -> Optional[AgentMessage]:
        raise NotImplementedError

# --- Tools (as callable utilities an agent could use) ---

def tool_code_execute(code: str) -> str:
    """
    Simple, sandboxed-ish code execution tool.
    For demo only â€“ do NOT expose to untrusted input in production.
    """
    allowed_builtins = {"__builtins__": {"len": len, "max": max, "min": min, "sum": sum}}
    local_env = {}
    try:
        exec(code, allowed_builtins, local_env)
        return f"Execution success. Locals: {local_env}"
    except Exception as e:
        return f"Execution error: {e}"

def tool_fake_search(query: str) -> str:
    """
    Stub to represent a search tool. Replace with a real API call (e.g. Google Search, OpenAPI) if desired.
    """
    return f"(Demo search) I pretended to search the web for: '{query}' and found some general info."


# ---------------------------------------------------------------
# 5. Concrete Agents
# ---------------------------------------------------------------

class MedicationAgent(BaseAgent):
    def __init__(self):
        super().__init__("MedicationAgent")

    def handle(self, msg: AgentMessage, session: SessionState) -> Optional[AgentMessage]:
        obs.info("MedicationAgent.handle", msg_type=msg.msg_type, trace_id=msg.trace_id)

        if msg.msg_type == "ADD_MED":
            name = msg.payload.get("name", "Metformin")
            dose = msg.payload.get("dose", "500 mg")
            times = msg.payload.get("times", ["08:00"])
            memory_bank.add_event("med_schedule", {
                "session_id": session.session_id,
                "name": name,
                "dose": dose,
                "times": times
            })
            return AgentMessage.create(
                sender=self.name,
                receiver=msg.sender,
                msg_type="ADD_MED_RESULT",
                payload={"text": f"Saved schedule for {name} {dose} at {', '.join(times)}."}
            )

        if msg.msg_type == "LOG_MED":
            name = msg.payload.get("name", "Metformin")
            status = msg.payload.get("status", "taken")
            memory_bank.add_event("med_intake", {
                "session_id": session.session_id,
                "name": name,
                "status": status
            })
            return AgentMessage.create(
                sender=self.name,
                receiver=msg.sender,
                msg_type="LOG_MED_RESULT",
                payload={"text": f"Logged that you {status} {name}."}
            )

        if msg.msg_type == "MED_SUMMARY":
            events = [
                e for e in memory_bank.get_events(since_days=1)
                if e.event_type == "med_intake" and e.payload.get("session_id") == session.session_id
            ]
            if not events:
                text = "No medication intake logged today."
            else:
                lines = [
                    f"- {e.payload.get('name')} : {e.payload.get('status')} at {e.timestamp}"
                    for e in events
                ]
                text = "Today's medication log:\n" + "\n".join(lines)

            return AgentMessage.create(
                sender=self.name,
                receiver=msg.sender,
                msg_type="MED_SUMMARY_RESULT",
                payload={"text": text}
            )

        return None


class RiskAgent(BaseAgent):
    def __init__(self):
        super().__init__("RiskAgent")

    def handle(self, msg: AgentMessage, session: SessionState) -> Optional[AgentMessage]:
        obs.info("RiskAgent.handle", msg_type=msg.msg_type, trace_id=msg.trace_id)
        if msg.msg_type != "RISK_ASSESS":
            return None

        symptom_text = msg.payload.get("text", "")
        system_prompt = (
            "You are a diabetes risk triage assistant. "
            "Classify risk as GREEN, AMBER or RED based on text. "
            "Do NOT change medication or give diagnosis."
        )
        result = llm_chat(system_prompt, f"Patient says: {symptom_text}")
        memory_bank.add_event("risk_assessment", {
            "session_id": session.session_id,
            "symptoms": symptom_text,
            "assessment": result
        })
        return AgentMessage.create(
            sender=self.name,
            receiver=msg.sender,
            msg_type="RISK_ASSESS_RESULT",
            payload={"assessment": result}
        )


class NotificationAgent(BaseAgent):
    def __init__(self):
        super().__init__("NotificationAgent")

    def handle(self, msg: AgentMessage, session: SessionState) -> Optional[AgentMessage]:
        obs.info("NotificationAgent.handle", msg_type=msg.msg_type, trace_id=msg.trace_id)
        if msg.msg_type != "NOTIFY_FROM_RISK":
            return None

        risk_output = msg.payload.get("assessment", "").lower()
        if "red" in risk_output:
            text = ("âš ï¸� High risk pattern detected (RED). "
                    "If you feel very unwell, seek urgent medical evaluation.")
        elif "amber" in risk_output:
            text = ("âš ï¸� Some concerning signals detected (AMBER). "
                    "Monitor closely and contact your clinician if it continues.")
        else:
            text = "âœ… Current pattern appears low risk (GREEN) in this simple demo."

        memory_bank.add_event("notification", {
            "session_id": session.session_id,
            "text": text
        })
        return AgentMessage.create(
            sender=self.name,
            receiver=msg.sender,
            msg_type="NOTIFY_RESULT",
            payload={"text": text}
        )


class ReportAgent(BaseAgent):
    def __init__(self):
        super().__init__("ReportAgent")

    def handle(self, msg: AgentMessage, session: SessionState) -> Optional[AgentMessage]:
        obs.info("ReportAgent.handle", msg_type=msg.msg_type, trace_id=msg.trace_id)
        if msg.msg_type != "WEEKLY_REPORT":
            return None

        events = [
            asdict(e) for e in memory_bank.get_events(since_days=msg.payload.get("days", 7))
            if e.payload.get("session_id") == session.session_id
        ]
        system_prompt = (
            "Summarize the following health events into a short doctor-facing weekly note "
            "and a simple patient-facing summary."
        )
        user_prompt = "Events:\n" + json.dumps(events, indent=2)
        report_text = llm_chat(system_prompt, user_prompt)
        memory_bank.add_event("weekly_report", {
            "session_id": session.session_id,
            "report": report_text
        })
        return AgentMessage.create(
            sender=self.name,
            receiver=msg.sender,
            msg_type="WEEKLY_REPORT_RESULT",
            payload={"text": report_text}
        )


class SchedulerAgent(BaseAgent):
    """
    Loop-style agent that would periodically scan for due reminders.
    In this demo, we simulate ticks instead of real-time scheduling.
    """
    def __init__(self):
        super().__init__("SchedulerAgent")
        self.paused = False

    def pause(self):
        self.paused = True
        obs.info("SchedulerPaused")

    def resume(self):
        self.paused = False
        obs.info("SchedulerResumed")

    def tick(self, session: SessionState) -> Optional[AgentMessage]:
        if self.paused:
            return None
        # For demo, we just emit a â€œcheck remindersâ€� message
        obs.info("SchedulerTick", session_id=session.session_id)
        return AgentMessage.create(
            sender=self.name,
            receiver="ConversationAgent",
            msg_type="SCHEDULER_REMINDER",
            payload={"text": "Time to check today's medication status."}
        )

    def handle(self, msg: AgentMessage, session: SessionState) -> Optional[AgentMessage]:
        # Not used for now; tick() is our loop entry.
        return None


# ---------------------------------------------------------------
# 6. ConversationAgent (Orchestrator, with parallelism)
# ---------------------------------------------------------------

class ConversationAgent(BaseAgent):
    def __init__(self):
        super().__init__("ConversationAgent")
        self.med_agent = MedicationAgent()
        self.risk_agent = RiskAgent()
        self.report_agent = ReportAgent()
        self.notify_agent = NotificationAgent()
        self.scheduler = SchedulerAgent()
        self.executor = ThreadPoolExecutor(max_workers=4)

    def handle_user_message(self, message: str, session_id: Optional[str] = None) -> str:
        session = session_service.get_or_create(session_id)
        session.short_context.append(message)
        obs.incr("user_messages")

        text = message.lower().strip()

        # 1) Add medication schedule
        if "add medicine" in text or "add medication" in text:
            msg = AgentMessage.create(
                sender=self.name,
                receiver="MedicationAgent",
                msg_type="ADD_MED",
                payload={"name": "Metformin", "dose": "500 mg", "times": ["08:00", "20:00"]}
            )
            reply = self.med_agent.handle(msg, session)
            return reply.payload["text"]

        # 2) Log taken / missed
        if "i took" in text or "tablet taken" in text or "medicine taken" in text:
            msg = AgentMessage.create(
                sender=self.name,
                receiver="MedicationAgent",
                msg_type="LOG_MED",
                payload={"name": "Metformin", "status": "taken"}
            )
            reply = self.med_agent.handle(msg, session)
            return reply.payload["text"]

        if "i missed" in text or "i forgot" in text:
            msg = AgentMessage.create(
                sender=self.name,
                receiver="MedicationAgent",
                msg_type="LOG_MED",
                payload={"name": "Metformin", "status": "missed"}
            )
            reply = self.med_agent.handle(msg, session)
            return reply.payload["text"]

        # 3) Ask for today's summary
        if "what did i take today" in text or "medication summary" in text:
            msg = AgentMessage.create(
                sender=self.name,
                receiver="MedicationAgent",
                msg_type="MED_SUMMARY",
                payload={}
            )
            reply = self.med_agent.handle(msg, session)
            return reply.payload["text"]

        # 4) Symptom description -> Risk + Notification (sequential, but could be parallel)
        if "i feel" in text or "symptom" in text or "not feeling well" in text:
            risk_msg = AgentMessage.create(
                sender=self.name,
                receiver="RiskAgent",
                msg_type="RISK_ASSESS",
                payload={"text": message}
            )
            # sequential
            risk_reply = self.risk_agent.handle(risk_msg, session)
            notify_msg = AgentMessage.create(
                sender=self.name,
                receiver="NotificationAgent",
                msg_type="NOTIFY_FROM_RISK",
                payload={"assessment": risk_reply.payload["assessment"]}
            )
            notify_reply = self.notify_agent.handle(notify_msg, session)
            return (
                "Risk interpretation (demo only):\n\n"
                + risk_reply.payload["assessment"]
                + "\n\n"
                + notify_reply.payload["text"]
                + "\n\nReminder: This is not medical advice."
            )

        # 5) Weekly report: run Risk + Medication summary in PARALLEL, then generate report
        if "weekly report" in text or "summary for doctor" in text:
            # Parallel tasks: get medication summary + compact old memory
            def task_med():
                med_msg = AgentMessage.create(
                    sender=self.name,
                    receiver="MedicationAgent",
                    msg_type="MED_SUMMARY",
                    payload={}
                )
                return self.med_agent.handle(med_msg, session)

            def task_compact():
                memory_bank.compact_old_events(older_than_days=7)
                return "compacted"

            futures = [
                self.executor.submit(task_med),
                self.executor.submit(task_compact)
            ]
            med_reply = futures[0].result()

            report_msg = AgentMessage.create(
                sender=self.name,
                receiver="ReportAgent",
                msg_type="WEEKLY_REPORT",
                payload={"days": 7}
            )
            report_reply = self.report_agent.handle(report_msg, session)
            return (
                med_reply.payload["text"]
                + "\n\nWeekly summary (demo text):\n\n"
                + report_reply.payload["text"]
            )

        # 6) Scheduler tick (loop-style simulation)
        if "run scheduler" in text or "check reminders" in text:
            sched_msg = self.scheduler.tick(session)
            med_msg = AgentMessage.create(
                sender=self.name,
                receiver="MedicationAgent",
                msg_type="MED_SUMMARY",
                payload={}
            )
            med_reply = self.med_agent.handle(med_msg, session)
            return f"{sched_msg.payload['text']}\n\n{med_reply.payload['text']}"

        # 7) Use custom tools: code execution or fake search
        if text.startswith("run code:"):
            code = message.split(":", 1)[1]
            result = tool_code_execute(code)
            return "Code tool result:\n" + result

        if text.startswith("search:"):
            query = message.split(":", 1)[1]
            result = tool_fake_search(query)
            return result

        # 8) Generic fallback
        system_prompt = (
            "You are a friendly diabetes self-management assistant. "
            "You NEVER modify treatment and always advise calling a clinician "
            "for concerning symptoms."
        )
        generic = llm_chat(system_prompt, message)
        return generic + "\n\n(General support reply â€“ demo only.)"

    # Not used as a pure A2A receiver here, but implemented for completeness
    def handle(self, msg: AgentMessage, session: SessionState) -> Optional[AgentMessage]:
        if msg.msg_type == "SCHEDULER_REMINDER":
            text = "Scheduler says: " + msg.payload.get("text", "")
            return AgentMessage.create(
                sender=self.name,
                receiver=msg.sender,
                msg_type="SCHEDULER_ACK",
                payload={"text": text}
            )
        return None


# ---------------------------------------------------------------
# 7. Agent Evaluation (simple harness)
# ---------------------------------------------------------------

def evaluate_agent():
    """
    Tiny evaluation suite showing that the agent behaves sensibly.
    Each test is a simple heuristic check (pass/fail).
    """
    conv = ConversationAgent()
    session_id = "test-session"
    tests = []

    # Test 1: Risk detection
    msg = "I feel very shaky and sweaty."
    out1 = conv.handle_user_message(msg, session_id=session_id)
    tests.append(("risk_amber", "AMBER" in out1 or "amber" in out1.lower()))

    # Test 2: Medication schedule + logging
    conv.handle_user_message("Add medication Metformin", session_id=session_id)
    conv.handle_user_message("I took my tablet", session_id=session_id)
    out2 = conv.handle_user_message("What did I take today?", session_id=session_id)
    tests.append(("med_log", "Metformin" in out2))

    # Test 3: Weekly report generation
    out3 = conv.handle_user_message("Generate my weekly report for the doctor.", session_id=session_id)
    tests.append(("weekly_report", "Weekly summary" in out3 or "Doctor Summary" in out3))

    # Aggregate results
    passed = sum(1 for _, ok in tests if ok)
    print("=== Evaluation Results ===")
    for name, ok in tests:
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    print(f"Total: {passed}/{len(tests)} tests passed\n")


# ---------------------------------------------------------------
# 8. Deployment Stub (FastAPI-style handler example)
# ---------------------------------------------------------------

DEPLOYMENT_EXAMPLE = """
# Example FastAPI deployment (not executed in this notebook):

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
conv_agent = ConversationAgent()

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    reply: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    reply = conv_agent.handle_user_message(req.message, session_id=req.session_id)
    return ChatResponse(session_id=req.session_id, reply=reply)

# Run with: uvicorn main:app --reload
"""
# ---------------------------------------------------------------
# 9. Demo Run
# ---------------------------------------------------------------

conversation_agent = ConversationAgent()

demo_messages = [
    "Add medication Metformin 500mg after breakfast and dinner.",
    "I feel shaky and sweaty, and I skipped breakfast today.",
    "What did I take today?",
    "Generate my weekly report for the doctor.",
    "Run scheduler",
    "search: diabetes foot care tips"
]

print("=== DiaCare Pro Demo Conversation ===\n")
sid = "demo-session-1"
for msg in demo_messages:
    print(f"ğŸ‘¤ User: {msg}")
    reply = conversation_agent.handle_user_message(msg, session_id=sid)
    print(f"ğŸ¤– DiaCare:\n{reply}\n")
    print("-" * 70)

print("\n=== MemoryBank Snapshot (compact view) ===")
print(memory_bank.to_json())

print("\n=== Observability Snapshot ===")
print(json.dumps(obs.dump(), indent=2))

print("\n=== Running quick evaluation suite ===")
evaluate_agent()

print("\n=== Deployment Example (for README) ===")
print(DEPLOYMENT_EXAMPLE)

