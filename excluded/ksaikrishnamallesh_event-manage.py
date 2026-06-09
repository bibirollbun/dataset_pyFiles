# %% [markdown]
# # Kaggle Capstone: Multilevel Event Planner Concierge Agent
# Step 1: Core primitives (observability, sessions, memory), Mock LLM, MCP, Tools, Agent base class
# Supports event types: Wedding, Birthday, College/School Event
# Paste cells into a Kaggle notebook and run sequentially.

# %%
# Basic imports
import threading
import time
import uuid
import queue
import json
import random
from collections import defaultdict, deque
from typing import Any, Dict, Optional, List, Callable

# %% [markdown]
# ## Observability primitives: Logger, Tracer, Metrics

# %%
class SimpleLogger:
    def info(self, msg: str):
        print(f"[INFO] {time.strftime('%H:%M:%S')} {msg}")

    def debug(self, msg: str):
        print(f"[DEBUG] {time.strftime('%H:%M:%S')} {msg}")

    def error(self, msg: str):
        print(f"[ERROR] {time.strftime('%H:%M:%S')} {msg}")

class Tracer:
    def start_span(self, name: str) -> str:
        return str(uuid.uuid4())[:8]

class Metrics:
    def __init__(self):
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)

    def incr(self, key: str, amount: int = 1):
        self.counters[key] += amount

    def time_it(self, key: str, value: float):
        self.timers[key].append(value)

    def snapshot(self):
        return {
            "counters": dict(self.counters),
            "timers": {k: {"count": len(v), "avg_ms": (sum(v)/len(v))*1000 if v else None} for k, v in self.timers.items()}
        }

logger = SimpleLogger()
tracer = Tracer()
metrics = Metrics()

# %% [markdown]
# ## Sessions & Memory
# - Minimal InMemorySessionService
# - MemoryBank with simple compaction

# %%
class InMemorySessionService:
    def __init__(self):
        self.sessions = {}

    def create_session(self, user_id: str, event_type: str = "birthday") -> str:
        sid = str(uuid.uuid4())
        self.sessions[sid] = {"user_id": user_id, "created": time.time(), "event_type": event_type}
        logger.info(f"session created: {sid} user={user_id} event_type={event_type}")
        return sid

    def get_session(self, sid: str):
        return self.sessions.get(sid)

class MemoryBank:
    def __init__(self, compaction_threshold: int = 6):
        self.memories = defaultdict(list)  # session_id -> list of messages
        self.compaction_threshold = compaction_threshold

    def add(self, session_id: str, message: str):
        self.memories[session_id].append({"ts": time.time(), "text": message})
        if len(self.memories[session_id]) >= self.compaction_threshold:
            self.compact(session_id)

    def retrieve(self, session_id: str, k: int = 5) -> List[str]:
        return [m["text"] for m in self.memories[session_id][-k:]]

    def compact(self, session_id: str):
        items = self.memories[session_id]
        if len(items) < 2:
            return
        # naive compaction: merge oldest two entries into a short summary
        oldest = items.pop(0)
        second = items.pop(0)
        summarized = f"(COMPACTED) {oldest['text']} | {second['text']}"
        items.insert(0, {"ts": time.time(), "text": summarized})
        logger.info(f"compacted memory for {session_id}; new length {len(items)}")

session_service = InMemorySessionService()
memory_bank = MemoryBank(compaction_threshold=6)

# %% [markdown]
# ## Mock LLM (deterministic, useful for Kaggle without API keys)
# - Produces different templates for Wedding, Birthday, College events

# %%
class MockLLM:
    def __init__(self):
        pass

    def respond(self, prompt: str, event_type: str = "birthday"):
        p = prompt.lower()
        # wedding templates
        if event_type == "wedding" or "wedding" in p:
            if "venue" in p:
                return ("Venue Suggestions:\n"
                        "1) Riverside Banquet Hall — capacity 150, est. cost ₹80,000 — romantic riverside setting\n"
                        "2) Garden Villa — capacity 120, est. cost ₹70,000 — outdoor garden, decorations included\n")
            if "budget" in p:
                return ("Budget Plan:\nVenue: ₹80,000\nCatering: ₹60,000\nDecorator: ₹15,000\nPhotography: ₹20,000\nTotal est: ₹175,000")
            if "checklist" in p:
                return ("Wedding Checklist:\n- Book venue (6-9 months before)\n- Send invitations (2 months before)\n- Confirm catering (1 month before)\n- Arrange photography (1 month before)\n")
            return "MockLLM (wedding): I can help find venues, build a budget, and generate checklists."
        # birthday templates
        if event_type == "birthday" or "birthday" in p:
            if "venue" in p:
                return ("Venue Suggestions:\n1) Rooftop Cafe — capacity 50, est. cost ₹8,000 — casual vibe\n2) Community Hall — capacity 80, est. cost ₹10,000 — affordable\n")
            if "budget" in p:
                return ("Budget Plan:\nVenue: ₹8,000\nFood: ₹6,000\nDecor: ₹2,000\nEntertainment: ₹4,000\nTotal est: ₹20,000")
            if "checklist" in p:
                return ("Birthday Checklist:\n- Book venue (1 month before)\n- Order cake (2 weeks before)\n- Send invites (2 weeks before)\n- Arrange DJ (1 week before)\n")
            return "MockLLM (birthday): I can pick venues, estimate budgets, and produce checklists."
        # college/school event templates
        if event_type == "college" or "college" in p or "school" in p:
            if "venue" in p:
                return ("Venue Suggestions:\n1) University Auditorium — capacity 300, est. cost Free/low — best for seminars\n2) College Lawn — capacity 200, est. cost ₹5,000 — open-air\n")
            if "budget" in p:
                return ("Budget Plan:\nVenue: ₹0-5,000\nAV Equipment: ₹3,000\nRefreshments: ₹6,000\nSecurity/Staff: ₹2,000\nTotal est: ₹16,000")
            if "checklist" in p:
                return ("College Event Checklist:\n- Reserve auditorium (2 months before)\n- Arrange AV (1 month before)\n- Coordinate with faculty (3 weeks before)\n- Promote event on campus (2 weeks before)\n")
            return "MockLLM (college): I can recommend campus venues, budget lines, and event checklists."
        # fallback
        return "MockLLM: generic response. Ask for venue, budget, or checklist."

mock_llm = MockLLM()

# %% [markdown]
# ## A2A Message format & MCP (Message/Tool Broker)

# %%
class Message:
    def __init__(self, from_agent: str, to_agent: str, msg_type: str, payload: Any, trace_id: Optional[str] = None):
        self.id = str(uuid.uuid4())
        self.from_agent = from_agent
        self.to_agent = to_agent
        self.type = msg_type
        self.payload = payload
        self.trace_id = trace_id or tracer.start_span("msg")

    def to_dict(self):
        return {
            "id": self.id,
            "from": self.from_agent,
            "to": self.to_agent,
            "type": self.type,
            "payload": self.payload,
            "trace_id": self.trace_id
        }

class MCP:
    def __init__(self):
        self.queues = defaultdict(queue.Queue)  # agent_id -> Queue
        self.tools = {}

    def register_agent(self, agent_id: str):
        _ = self.queues[agent_id]
        logger.info(f"MCP: registered agent {agent_id}")

    def send(self, msg: Message):
        logger.debug(f"MCP send {msg.from_agent} -> {msg.to_agent} ({msg.type})")
        metrics.incr("mcp.messages_sent")
        if msg.to_agent in self.queues:
            self.queues[msg.to_agent].put(msg)
        else:
            logger.error(f"MCP: unknown recipient {msg.to_agent}")

    def recv(self, agent_id: str, timeout: Optional[float] = None) -> Optional[Message]:
        try:
            msg = self.queues[agent_id].get(timeout=timeout)
            metrics.incr("mcp.messages_received")
            return msg
        except queue.Empty:
            return None

    def register_tool(self, name: str, fn: Callable):
        self.tools[name] = fn
        logger.info(f"tool registered: {name}")

    def call_tool(self, name: str, *args, **kwargs):
        metrics.incr(f"tool.{name}.calls")
        if name in self.tools:
            return self.tools[name](*args, **kwargs)
        else:
            raise RuntimeError(f"Tool {name} not found")

mcp = MCP()

# %% [markdown]
# ## Tools (toy data / KB) - registered in MCP

# %%
# small knowledge base for venue hints
KB = {
    "wedding": "Consider guest count, season, and cultural preferences for wedding venues.",
    "birthday": "Consider age group, vibe (casual/formal), and budget for birthdays.",
    "college": "Consider capacity, AV needs, and campus approvals for college events."
}

def kb_search(event_type: str, query: str = ""):
    base = KB.get(event_type, "General event planning tips.")
    # naive result
    return f"{base} (query matched: {query})"

def simple_compute(x: int, y: int):
    time.sleep(0.05)
    return {"sum": x+y, "product": x*y}

mcp.register_tool("kb_search", kb_search)
mcp.register_tool("compute", simple_compute)

# %% [markdown]
# ## Agent base class (threaded agents to allow parallelism)

# %%
class Agent(threading.Thread):
    def __init__(self, agent_id: str, mcp: MCP, behavior: Callable[['Agent', Message], None]):
        super().__init__(daemon=True)
        self.agent_id = agent_id
        self.mcp = mcp
        self.behavior = behavior
        self.running = False
        self._stop_event = threading.Event()
        self.mcp.register_agent(agent_id)

    def stop(self):
        self.running = False
        self._stop_event.set()

    def run(self):
        logger.info(f"Agent {self.agent_id} starting")
        self.running = True
        while not self._stop_event.is_set():
            msg = self.mcp.recv(self.agent_id, timeout=0.5)
            if msg:
                try:
                    start = time.time()
                    self.behavior(self, msg)
                    duration = time.time() - start
                    metrics.time_it(f"agent.{self.agent_id}.proc_time", duration)
                except Exception as e:
                    logger.error(f"Agent {self.agent_id} behavior error: {e}")
            else:
                time.sleep(0.05)
        logger.info(f"Agent {self.agent_id} stopped")


