# Agents Intensive - Capstone Project: TeamOps Assistant
# ====================================================
# Single-file scaffold for a multi-agent capstone project.
# - Domain: Team operations (technical debt, feature requests, incident tracking)
# - Features: Observability, Sessions & Memory, LLM stub, A2A protocol,
#   multiple agents (IssueAgent, DebtAgent, NotificationAgent, ReportAgent, SchedulerAgent),
#   demo run, and deployment stub.
#
# Usage: open this file in Kaggle / Jupyter / VSCode and run top-to-bottom. Replace the
# LLM stub with a real model (Gemini, OpenAI, etc.) when ready.
#
# NOTE: This scaffold is intentionally self-contained and uses a dummy LLM for offline demos.

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import datetime
import json
import uuid
import textwrap
from concurrent.futures import ThreadPoolExecutor

# -------------------------------
# Observability
# -------------------------------
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
        return {"logs": self.logs[-100:], "metrics": self.metrics}

obs = Observability()

# -------------------------------
# Sessions & Memory
# -------------------------------
@dataclass
class SessionState:
    session_id: str
    user_profile: Dict[str, Any]
    short_context: List[str]

class SessionService:
    def __init__(self):
        self.sessions: Dict[str, SessionState] = {}

    def get_or_create(self, session_id: Optional[str]) -> SessionState:
        if session_id is None:
            session_id = str(uuid.uuid4())
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(session_id=session_id, user_profile={}, short_context=[])
            obs.info("SessionCreated", session_id=session_id)
        return self.sessions[session_id]

session_service = SessionService()

@dataclass
class MemoryEvent:
    event_type: str
    payload: Dict[str, Any]
    timestamp: str

class MemoryBank:
    def __init__(self):
        self.events: List[MemoryEvent] = []
        self.compacted_summaries: List[Dict[str, Any]] = []

    def add_event(self, event_type: str, payload: Dict[str, Any]) -> MemoryEvent:
        e = MemoryEvent(event_type=event_type, payload=payload, timestamp=datetime.datetime.utcnow().isoformat())
        self.events.append(e)
        obs.incr("events_total")
        return e

    def get_events(self, since_days: Optional[int] = None) -> List[MemoryEvent]:
        if since_days is None:
            return list(self.events)
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=since_days)
        return [e for e in self.events if datetime.datetime.fromisoformat(e.timestamp) >= cutoff]

    def compact_old_events(self, older_than_days: int = 7):
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=older_than_days)
        old, new = [], []
        for e in self.events:
            ts = datetime.datetime.fromisoformat(e.timestamp)
            if ts < cutoff:
                old.append(e)
            else:
                new.append(e)
        self.events = new
        if not old:
            return
        summary = {"from": "MemoryBank", "summary_type": "compact", "event_count": len(old), "note": "Compact summary for long-term history."}
        self.compacted_summaries.append(summary)
        obs.info("ContextCompacted", summary=summary)

    def to_json(self) -> str:
        return json.dumps({"events": [asdict(e) for e in self.events], "compacted": self.compacted_summaries}, indent=2)

memory_bank = MemoryBank()

# -------------------------------
# LLM Stub (dummy for demo)
# -------------------------------
USE_DUMMY_LLM = True

def llm_chat(system_prompt: str, user_prompt: str) -> str:
    """
    Dummy LLM for offline demo. Replace with real API integration for production.
    """
    if USE_DUMMY_LLM:
        prompt = (system_prompt + "\n" + user_prompt).lower()
        # Basic heuristics for demo
        if "summarize" in system_prompt.lower():
            return "(Demo) Short report: Several technical debt items and two high-priority incidents."
        if "classify" in system_prompt.lower():
            if "blocking" in user_prompt.lower() or "down" in user_prompt.lower() or "500" in user_prompt.lower():
                return "PRIORITY: HIGH\nReason: Blocking incident detected."
            return "PRIORITY: MEDIUM\nReason: Needs attention."
        # Generic fallback
        return "(Demo LLM reply) This is a placeholder. Replace with real model integration."
    raise NotImplementedError("LLM integration not configured")

# -------------------------------
# A2A Message
# -------------------------------
@dataclass
class AgentMessage:
    sender: str
    receiver: str
    msg_type: str
    payload: Dict[str, Any]
    trace_id: str
    timestamp: str

    @staticmethod
    def create(sender: str, receiver: str, msg_type: str, payload: Dict[str, Any]) -> 'AgentMessage':
        return AgentMessage(sender=sender, receiver=receiver, msg_type=msg_type, payload=payload, trace_id=str(uuid.uuid4()), timestamp=datetime.datetime.utcnow().isoformat())

# -------------------------------
# Base Agent + Tools
# -------------------------------
class BaseAgent:
    def __init__(self, name: str):
        self.name = name

    def handle(self, msg: AgentMessage, session: SessionState) -> Optional[AgentMessage]:
        raise NotImplementedError

def tool_search_internal_issues(query: str) -> str:
    return f"(Internal search) Found simulated issues for: {query}"

# -------------------------------
# Concrete Agents
# -------------------------------
class IssueAgent(BaseAgent):
    def __init__(self):
        super().__init__("IssueAgent")

    def handle(self, msg: AgentMessage, session: SessionState) -> Optional[AgentMessage]:
        obs.info("IssueAgent.handle", msg_type=msg.msg_type, trace_id=msg.trace_id)
        if msg.msg_type == "REPORT_ISSUE":
            memory_bank.add_event("issue_reported", {"session_id": session.session_id, **msg.payload})
            return AgentMessage.create(self.name, msg.sender, "ISSUE_REPORTED", {"text": "Issue recorded."})
        if msg.msg_type == "SEARCH_ISSUES":
            res = tool_search_internal_issues(msg.payload.get("query", ""))
            return AgentMessage.create(self.name, msg.sender, "SEARCH_RESULT", {"text": res})
        return None

class DebtAgent(BaseAgent):
    def __init__(self):
        super().__init__("DebtAgent")

    def handle(self, msg: AgentMessage, session: SessionState) -> Optional[AgentMessage]:
        obs.info("DebtAgent.handle", msg_type=msg.msg_type, trace_id=msg.trace_id)
        if msg.msg_type == "LOG_DEBT":
            memory_bank.add_event("technical_debt", {"session_id": session.session_id, **msg.payload})
            return AgentMessage.create(self.name, msg.sender, "DEBT_LOGGED", {"text": "Technical debt item logged."})
        if msg.msg_type == "PRIORITIZE_DEBT":
            system_prompt = "Classify debt items into priority buckets HIGH/MEDIUM/LOW based on impact and effort."
            user_prompt = json.dumps(msg.payload)
            result = llm_chat(system_prompt, user_prompt)
            memory_bank.add_event("debt_prioritization", {"session_id": session.session_id, "result": result})
            return AgentMessage.create(self.name, msg.sender, "DEBT_PRIORITIZED", {"text": result})
        return None

class NotificationAgent(BaseAgent):
    def __init__(self):
        super().__init__("NotificationAgent")

    def handle(self, msg: AgentMessage, session: SessionState) -> Optional[AgentMessage]:
        obs.info("NotificationAgent.handle", msg_type=msg.msg_type, trace_id=msg.trace_id)
        if msg.msg_type == "NOTIFY":
            memory_bank.add_event("notification", {"session_id": session.session_id, "text": msg.payload.get("text")})
            return AgentMessage.create(self.name, msg.sender, "NOTIFY_ACK", {"text": "Notification sent (demo)."})
        return None

class ReportAgent(BaseAgent):
    def __init__(self):
        super().__init__("ReportAgent")

    def handle(self, msg: AgentMessage, session: SessionState) -> Optional[AgentMessage]:
        obs.info("ReportAgent.handle", msg_type=msg.msg_type, trace_id=msg.trace_id)
        if msg.msg_type == "GENERATE_REPORT":
            events = [asdict(e) for e in memory_bank.get_events(since_days=msg.payload.get("days", 7)) if e.payload.get("session_id") == session.session_id]
            system_prompt = "Summarize the following events into an executive note and a developer-friendly action list."
            user_prompt = "Events:\n" + json.dumps(events, indent=2)
            report_text = llm_chat(system_prompt, user_prompt)
            memory_bank.add_event("report_generated", {"session_id": session.session_id, "report": report_text})
            return AgentMessage.create(self.name, msg.sender, "REPORT_READY", {"text": report_text})
        return None

class SchedulerAgent(BaseAgent):
    def __init__(self):
        super().__init__("SchedulerAgent")
        self.paused = False

    def tick(self, session: SessionState) -> Optional[AgentMessage]:
        if self.paused:
            return None
        obs.info("SchedulerTick", session_id=session.session_id)
        return AgentMessage.create(self.name, "ConversationAgent", "SCHEDULER_REMINDER", {"text": "Reminder: review high-priority debt items."})

    def pause(self):
        self.paused = True
        obs.info("SchedulerPaused")

    def resume(self):
        self.paused = False
        obs.info("SchedulerResumed")

# -------------------------------
# Conversation / Orchestrator
# -------------------------------
class ConversationAgent(BaseAgent):
    def __init__(self):
        super().__init__("ConversationAgent")
        self.issue_agent = IssueAgent()
        self.debt_agent = DebtAgent()
        self.report_agent = ReportAgent()
        self.notify_agent = NotificationAgent()
        self.scheduler = SchedulerAgent()
        self.executor = ThreadPoolExecutor(max_workers=4)

    def handle_user_message(self, message: str, session_id: Optional[str] = None) -> str:
        session = session_service.get_or_create(session_id)
        session.short_context.append(message)
        obs.incr("user_messages")
        text = message.lower().strip()

        # Report an issue:
        if text.startswith("report:"):
            # content after 'report:' or fallback to full message
            after = message[len("report:"):].strip()
            payload = {"title": after or message, "description": after or message}
            msg = AgentMessage.create(self.name, "IssueAgent", "REPORT_ISSUE", payload)
            reply = self.issue_agent.handle(msg, session)
            return reply.payload["text"]

        # Log a technical debt item:
        if text.startswith("log debt:"):
            after = message[len("log debt:"):].strip()
            payload = {"title": after or "Unnamed Debt", "impact": "medium", "effort": "unknown", "reported_by": session.session_id}
            msg = AgentMessage.create(self.name, "DebtAgent", "LOG_DEBT", payload)
            reply = self.debt_agent.handle(msg, session)
            return reply.payload["text"]

        # Prioritize debt:
        if "prioritize debt" in text or text.startswith("prioritize debt"):
            events = [asdict(e) for e in memory_bank.get_events() if e.event_type == "technical_debt"]
            msg = AgentMessage.create(self.name, "DebtAgent", "PRIORITIZE_DEBT", {"items": events})
            reply = self.debt_agent.handle(msg, session)
            return reply.payload["text"]

        # Generate report:
        if "generate report" in text or "weekly report" in text:
            msg = AgentMessage.create(self.name, "ReportAgent", "GENERATE_REPORT", {"days": 7})
            reply = self.report_agent.handle(msg, session)
            return reply.payload["text"]

        # Scheduler tick:
        if "run scheduler" in text or "check reminders" in text:
            tick_msg = self.scheduler.tick(session)
            if tick_msg:
                return tick_msg.payload["text"]
            return "Scheduler paused."

        # Search issues:
        if text.startswith("search:"):
            query = message.split(":", 1)[1].strip() if ":" in message else ""
            msg = AgentMessage.create(self.name, "IssueAgent", "SEARCH_ISSUES", {"query": query})
            reply = self.issue_agent.handle(msg, session)
            return reply.payload["text"]

        # Fallback to LLM stub:
        system_prompt = "You are a TeamOps assistant. Help users track issues and technical debt."
        return llm_chat(system_prompt, message)

    def handle(self, msg: AgentMessage, session: SessionState) -> Optional[AgentMessage]:
        # Handles messages sent to ConversationAgent (e.g., scheduler reminders)
        if msg.msg_type == "SCHEDULER_REMINDER":
            return AgentMessage.create(self.name, msg.sender, "SCHED_ACK", {"text": "Ack: " + msg.payload.get("text", "")})
        return None

# -------------------------------
# Agent Evaluation (simple harness)
# -------------------------------
def evaluate_agent():
    conv = ConversationAgent()
    session_id = "test-session"
    tests = []

    # Test 1: Log debt then check memory
    conv.handle_user_message("Log debt: Refactor legacy auth module", session_id=session_id)
    debts = [e for e in memory_bank.get_events() if e.event_type == "technical_debt" and e.payload.get("session_id") == session_id]
    tests.append(("log_debt", len(debts) >= 1))

    # Test 2: Report an issue
    conv.handle_user_message("Report: Payment API is returning 500 intermittently", session_id=session_id)
    issues = [e for e in memory_bank.get_events() if e.event_type == "issue_reported" and e.payload.get("session_id") == session_id]
    tests.append(("report_issue", len(issues) >= 1))

    # Test 3: Prioritize debt (stub returns a string)
    out = conv.handle_user_message("Prioritize debt", session_id=session_id)
    tests.append(("prioritize_debt", isinstance(out, str) and len(out) > 0))

    # Aggregate results
    passed = sum(1 for _, ok in tests if ok)
    print("=== Evaluation Results ===")
    for name, ok in tests:
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    print(f"Total: {passed}/{len(tests)} tests passed\n")

# -------------------------------
# Deployment Stub (README-style)
# -------------------------------
DEPLOYMENT_EXAMPLE = """
# Example FastAPI deployment (not executed in this script):

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

# -------------------------------
# Demo run
# -------------------------------
if __name__ == "__main__":
    conv = ConversationAgent()
    sid = "capstone-demo-session"
    demo_msgs = [
        "Log debt: Refactor legacy auth module",
        "Report: Payment API is returning 500 intermittently",
        "Log debt: Outdated dependencies in billing service",
        "Prioritize debt",
        "Generate report",
        "Run scheduler",
        "Search: payment api 500"
    ]

    print("=== Agents Intensive: TeamOps Assistant Demo ===\n")
    for m in demo_msgs:
        print(f"ðŸ‘¤ User: {m}")
        reply = conv.handle_user_message(m, session_id=sid)
        print(f"ðŸ¤– Assistant:\n{reply}\n")
        print("-" * 70)

    print("\n=== Memory Snapshot ===")
    print(memory_bank.to_json())

    print("\n=== Observability Snapshot ===")
    print(json.dumps(obs.dump(), indent=2))

    print("\n=== Running quick evaluation suite ===")
    evaluate_agent()

    print("\n=== Deployment Example (for README) ===")
    print(DEPLOYMENT_EXAMPLE)


