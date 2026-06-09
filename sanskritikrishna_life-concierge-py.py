"""
life_concierge.py
Full Python demo of a Life Celebrations Concierge AI with:
- Supervisor agent, parallel/sequential/loop agents
- Tools (MCP, custom tools, built-in search & code execution)
- OpenAPI tool mock
- Sessions & Memory (InMemorySessionService, MemoryBank)
- Long-running operations (pause/resume)
- Context compaction
- Observability (logging, tracing, metrics counters)
- Agent evaluation harness
- FastAPI endpoints for demo
"""

import asyncio
import logging
import uuid
import datetime
import json
import sqlite3
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

# -------------------------
# Observability (logging/tracing/metrics)
# -------------------------
logger = logging.getLogger("life_concierge")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

class Metrics:
    counters = {
        "ideas_generated": 0,
        "bookings_attempted": 0,
        "bookings_succeeded": 0,
        "reminders_sent": 0,
        "llm_calls": 0,
    }

    @classmethod
    def inc(cls, name, n=1):
        cls.counters[name] = cls.counters.get(name, 0) + n
        logger.debug(f"metric {name} = {cls.counters[name]}")

    @classmethod
    def snapshot(cls):
        return dict(cls.counters)

# -------------------------
# Persistence (SQLite simple demo)
# -------------------------
DB = "life_concierge_demo.db"

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
                 id TEXT PRIMARY KEY, name TEXT, email TEXT, timezone TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS events (
                 id TEXT PRIMARY KEY, user_id TEXT, title TEXT, date TEXT, recurrence TEXT, context TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS bookings (
                 id TEXT PRIMARY KEY, user_id TEXT, event_id TEXT, provider TEXT, payload TEXT, status TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS pending_ops (
                 id TEXT PRIMARY KEY, user_id TEXT, op_type TEXT, payload TEXT, status TEXT, created_at TEXT)""")
    conn.commit(); conn.close()

def db_insert(table: str, row: Dict[str, Any]):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    keys = ",".join(row.keys())
    q = ",".join("?" for _ in row)
    c.execute(f"INSERT INTO {table} ({keys}) VALUES ({q})", tuple(row.values()))
    conn.commit(); conn.close()

def db_query(table: str, where: Optional[str] = None):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    q = f"SELECT * FROM {table}" + (f" WHERE {where}" if where else "")
    rows = c.execute(q).fetchall()
    cols = [desc[0] for desc in c.description]
    conn.close()
    return [dict(zip(cols, r)) for r in rows]

def db_get(table: str, id_value: str):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    rows = c.execute(f"SELECT * FROM {table} WHERE id=?", (id_value,)).fetchall()
    cols = [desc[0] for desc in c.description]
    conn.close()
    return dict(zip(cols, rows[0])) if rows else None

# -------------------------
# LLM Client abstraction (mock)
# -------------------------
class LLMClient:
    """
    Replace the mock methods with real LLM API calls.
    The mock is deterministic for demo.
    """
    async def generate_ideas(self, user_profile: Dict, event: Dict, count: int = 3):
        Metrics.inc("llm_calls")
        # Mock creative outputs
        base = [
            {
                "title": f"Memory Lane Montage for {event['title']}",
                "description": "Put together old photos and short clips into a heartfelt montage. Play during dinner.",
                "estimated_budget": "$10-$50",
                "actions": ["diy", "order_prints"]
            },
            {
                "title": f"Chef's Table Experience for {event['title']}",
                "description": "Reserve an intimate tasting menu at a nearby boutique restaurant and request a personalized note.",
                "estimated_budget": "$60-$200",
                "actions": ["book_restaurant"]
            },
            {
                "title": f"Custom Gift Box + Book for {event['title']}",
                "description": "Curate favorite snacks, a book, and a handwritten note in a themed box.",
                "estimated_budget": "$30-$100",
                "actions": ["buy_gift"]
            }
        ]
        await asyncio.sleep(0.2)  # simulate latency
        Metrics.inc("ideas_generated")
        return base[:count]

    async def plan(self, instruction: str) -> Dict:
        Metrics.inc("llm_calls")
        # Mock planner returns structured steps
        await asyncio.sleep(0.05)
        return {"steps": [{"tool":"ideas","action":"generate_ideas","params":{"count":3}}]}

# -------------------------
# Tools Layer (MCP wrapper + custom + builtins)
# -------------------------
@dataclass
class ToolResult:
    success: bool
    payload: Dict[str, Any]
    raw: Any = None

class Tool:
    name: str
    def __init__(self, name: str):
        self.name = name
    async def run(self, **kwargs) -> ToolResult:
        raise NotImplementedError

class MCPWrapper:
    """
    A minimal Model Context Protocol wrapper ensuring structured tool calls.
    """
    def __init__(self, tool: Tool):
        self.tool = tool

    async def call(self, **kwargs) -> ToolResult:
        logger.info(f"[MCP] Calling tool {self.tool.name} with {kwargs}")
        try:
            res = await self.tool.run(**kwargs)
            logger.info(f"[MCP] Tool {self.tool.name} returned success={res.success}")
            return res
        except Exception as e:
            logger.exception("Tool error")
            return ToolResult(success=False, payload={"error": str(e)})

# --- Custom Tools ---
class CalendarTool(Tool):
    def __init__(self):
        super().__init__("calendar")

    async def run(self, action: str, user_id: str, **kwargs):
        # For demo, stores reminder in pending_ops or events
        if action == "add_event":
            ev = {"id": str(uuid.uuid4()), "user_id": user_id, "title": kwargs.get("title"), "date": kwargs.get("date"), "recurrence": kwargs.get("recurrence","yearly"), "context": kwargs.get("context","")}
            db_insert("events", ev)
            return ToolResult(success=True, payload={"event": ev})
        if action == "list_upcoming":
            # simple upcoming within 7 days
            items = db_query("events", where=f"user_id='{user_id}'")
            return ToolResult(success=True, payload={"events": items})
        return ToolResult(success=False, payload={"error":"unknown action"})

class RestaurantBookingTool(Tool):
    def __init__(self):
        super().__init__("restaurant_booking")
    async def run(self, action: str, user_id: str, event_id: str, details: Dict):
        Metrics.inc("bookings_attempted")
        # Mock external booking (simulate success)
        await asyncio.sleep(0.2)
        booking = {"id": str(uuid.uuid4()), "user_id": user_id, "event_id": event_id, "provider":"mock_restaurant", "payload": details, "status":"confirmed"}
        db_insert("bookings", {"id": booking["id"], "user_id": user_id, "event_id": event_id, "provider": booking["provider"], "payload": json.dumps(booking["payload"]), "status": booking["status"]})
        Metrics.inc("bookings_succeeded")
        return ToolResult(success=True, payload={"booking": booking})

class GiftOrderTool(Tool):
    def __init__(self):
        super().__init__("gift_order")
    async def run(self, action: str, user_id: str, event_id: str, details: Dict):
        # Mock order
        await asyncio.sleep(0.15)
        order = {"id": str(uuid.uuid4()), "user_id": user_id, "event_id": event_id, "provider":"mock_shop", "payload": details, "status":"ordered"}
        db_insert("bookings", {"id": order["id"], "user_id": user_id, "event_id": event_id, "provider": order["provider"], "payload": json.dumps(order["payload"]), "status": order["status"]})
        return ToolResult(success=True, payload={"order": order})

# Built-in search tool (mock)
class SearchTool(Tool):
    def __init__(self):
        super().__init__("search")
    async def run(self, q: str, limit: int = 3):
        # Mock results
        await asyncio.sleep(0.1)
        results = [{"title": f"Result for {q} #{i+1}", "url": f"https://example.com/search/{i+1}", "snippet":"Short summary"} for i in range(limit)]
        return ToolResult(success=True, payload={"results": results})

# Code execution tool (very limited & safe)
class CodeExecTool(Tool):
    def __init__(self):
        super().__init__("code_exec")
    async def run(self, code: str):
        # Extremely restricted: only allow simple arithmetic or date math using eval on a safe dict
        safe_globals = {"__builtins__": {}}
        safe_locals = {"datetime": datetime, "uuid": uuid}
        try:
            # Warning: in production replace with safe sandbox
            result = eval(code, safe_globals, safe_locals)
            return ToolResult(success=True, payload={"result": str(result)})
        except Exception as e:
            return ToolResult(success=False, payload={"error": str(e)})

# OpenAPI Tool mock (calls external REST)
class OpenAPITool(Tool):
    def __init__(self, spec_name: str):
        super().__init__(f"openapi:{spec_name}")
        self.spec_name = spec_name
    async def run(self, operation: str, payload: Dict):
        # Mock behavior: pretend to call external API and return success
        await asyncio.sleep(0.15)
        return ToolResult(success=True, payload={"spec": self.spec_name, "operation": operation, "payload": payload, "status":"ok"})

# -------------------------
# Sessions & Memory
# -------------------------
@dataclass
class SessionState:
    session_id: str
    user_id: str
    conversation: List[Dict[str,Any]] = field(default_factory=list)
    created_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    last_active: datetime.datetime = field(default_factory=datetime.datetime.utcnow)

class InMemorySessionService:
    def __init__(self):
        self.sessions: Dict[str, SessionState] = {}

    def create(self, user_id: str) -> SessionState:
        sid = str(uuid.uuid4())
        s = SessionState(session_id=sid, user_id=user_id)
        self.sessions[sid] = s
        logger.info(f"session created: {sid} for user {user_id}")
        return s

    def get(self, session_id: str) -> Optional[SessionState]:
        return self.sessions.get(session_id)

    def append_message(self, session_id: str, role: str, content: str):
        s = self.get(session_id)
        if s:
            s.conversation.append({"role": role, "content": content, "ts": datetime.datetime.utcnow().isoformat()})
            s.last_active = datetime.datetime.utcnow()

    def compact(self, session_id: str, max_tokens: int = 1000):
        # simple compaction: keep only last N messages
        s = self.get(session_id)
        if s and len(s.conversation) > 20:
            removed = len(s.conversation) - 20
            s.conversation = s.conversation[-20:]
            logger.debug(f"compacted session {session_id}, removed {removed} messages")

class MemoryBank:
    """
    Long-term memory storing user profiles, events, preferences, celebration_history.
    For demo this wraps sqlite events table + a JSON memory store.
    """
    def __init__(self):
        self.memory_store: Dict[str, Dict] = {}  # user_id -> memory dict

    def load_user(self, user_id: str):
        if user_id not in self.memory_store:
            # initialize with DB events
            events = db_query("events", where=f"user_id='{user_id}'")
            self.memory_store[user_id] = {"important_dates": events, "preferences": {}, "history": []}
        return self.memory_store[user_id]

    def add_event(self, user_id: str, event: Dict):
        mem = self.load_user(user_id)
        mem["important_dates"].append(event)

    def log_history(self, user_id: str, entry: Dict):
        mem = self.load_user(user_id)
        mem["history"].append({"ts": datetime.datetime.utcnow().isoformat(), **entry})

    def query(self, user_id: str, key: str):
        mem = self.load_user(user_id)
        return mem.get(key)

# -------------------------
# A2A Protocol (simple)
# -------------------------
@dataclass
class A2AMessage:
    from_agent: str
    to_agent: str
    payload: Dict
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ts: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())

class A2AChannel:
    """
    Very simple pub-sub between agents (in memory).
    """
    def __init__(self):
        self.queues: Dict[str, asyncio.Queue] = {}

    def ensure(self, agent_name: str):
        if agent_name not in self.queues:
            self.queues[agent_name] = asyncio.Queue()

    async def send(self, msg: A2AMessage):
        self.ensure(msg.to_agent)
        await self.queues[msg.to_agent].put(msg)
        logger.debug(f"A2A sent {msg.msg_id} from {msg.from_agent} to {msg.to_agent}")

    async def recv(self, agent_name: str, timeout: Optional[float] = None) -> Optional[A2AMessage]:
        self.ensure(agent_name)
        try:
            if timeout:
                msg = await asyncio.wait_for(self.queues[agent_name].get(), timeout=timeout)
            else:
                msg = await self.queues[agent_name].get()
            return msg
        except asyncio.TimeoutError:
            return None

# -------------------------
# Agents
# -------------------------
class BaseAgent:
    def __init__(self, name: str, llm: LLMClient, a2a: A2AChannel, memory: MemoryBank, sessions: InMemorySessionService):
        self.name = name
        self.llm = llm
        self.a2a = a2a
        self.memory = memory
        self.sessions = sessions

    async def handle(self, *args, **kwargs):
        raise NotImplementedError

class IdeaAgent(BaseAgent):
    def __init__(self, *args, tools: Dict[str, Tool] = None, **kwargs):
        super().__init__("idea_agent", *args, **kwargs)
        self.tools = tools or {}

    async def handle(self, user_id: str, event_id: str, count: int = 3):
        event = db_get("events", event_id)
        if not event:
            return {"error":"event_not_found"}
        user_mem = self.memory.load_user(user_id)
        prefs = user_mem.get("preferences", {})
        ideas = await self.llm.generate_ideas(prefs, event, count=count)
        # Log to memory
        self.memory.log_history(user_id, {"type":"ideas_generated","event_id": event_id, "ideas": ideas})
        logger.info(f"{self.name} generated {len(ideas)} ideas for event {event_id}")
        return {"ideas": ideas}

class BookingAgent(BaseAgent):
    def __init__(self, *args, tools: Dict[str, Tool] = None, **kwargs):
        super().__init__("booking_agent", *args, **kwargs)
        self.tools = tools or {}
        self.mcp_wrappers = {k: MCPWrapper(v) for k,v in self.tools.items()}

    async def handle(self, user_id: str, event_id: str, idea: Dict):
        # Based on idea.actions call appropriate tool
        for action in idea.get("actions", []):
            if action == "book_restaurant" and "restaurant_booking" in self.mcp_wrappers:
                res = await self.mcp_wrappers["restaurant_booking"].call(action="book", user_id=user_id, event_id=event_id, details={"idea": idea})
                return {"booking_result": res.payload if res.success else {"error":"booking_failed"}}
            if action == "buy_gift" and "gift_order" in self.mcp_wrappers:
                res = await self.mcp_wrappers["gift_order"].call(action="order", user_id=user_id, event_id=event_id, details={"idea": idea})
                return {"order_result": res.payload if res.success else {"error":"order_failed"}}
        return {"message":"no_actionable_tool_found"}

class ReminderAgent(BaseAgent):
    def __init__(self, *args, tools: Dict[str, Tool] = None, **kwargs):
        super().__init__("reminder_agent", *args, **kwargs)
        self.tools = tools or {}
        self.mcp_wrappers = {k: MCPWrapper(v) for k,v in self.tools.items()}

    async def schedule_all_upcoming(self, user_id: str):
        # find upcoming within 7 days and "send" a reminder (mock)
        rows = db_query("events", where=f"user_id='{user_id}'")
        today = datetime.date.today()
        soon = []
        for r in rows:
            try:
                ev_date = datetime.datetime.strptime(r["date"], "%Y-%m-%d").date()
                ev_this_year = ev_date.replace(year=today.year)
                delta = (ev_this_year - today).days
                if 0 <= delta <= 7:
                    soon.append(r)
            except Exception:
                continue
        # simulate sending reminders
        for ev in soon:
            logger.info(f"ReminderAgent: sending reminder for {ev['title']} to user {user_id}")
            Metrics.inc("reminders_sent")
            self.memory.log_history(user_id, {"type":"reminder_sent", "event_id": ev["id"]})
        return {"reminders": len(soon)}

# Supervisor orchestrates agents, supports parallel, sequential, loop, pause/resume
class SupervisorAgent:
    def __init__(self, llm: LLMClient, tools: Dict[str, Tool], a2a: A2AChannel, memory: MemoryBank, sessions: InMemorySessionService):
        self.llm = llm
        self.a2a = a2a
        self.tools = tools
        self.memory = memory
        self.sessions = sessions
        # instantiate agents
        self.idea_agent = IdeaAgent("idea_agent", llm, a2a, memory, sessions)
        self.booking_agent = BookingAgent("booking_agent", llm, a2a, memory, sessions, tools=tools)
        self.reminder_agent = ReminderAgent("reminder_agent", llm, a2a, memory, sessions, tools=tools)

    async def run_sequential(self, user_id: str, event_id: str):
        # Sequential: generate ideas, then ask to book one (mock user pick)
        ideas_res = await self.idea_agent.handle(user_id, event_id, count=3)
        # In a real flow we'd send options to user; here we simulate picking the second idea
        ideas = ideas_res.get("ideas", [])
        if not ideas:
            return {"error":"no_ideas"}
        chosen = ideas[1]
        booking_res = await self.booking_agent.handle(user_id, event_id, chosen)
        # also add a calendar reminder
        await self.reminder_agent.schedule_all_upcoming(user_id)
        return {"ideas": ideas, "booking": booking_res}

    async def run_parallel(self, user_id: str, event_id: str):
        # Parallel: run idea generation and upcoming reminders concurrently
        t1 = asyncio.create_task(self.idea_agent.handle(user_id, event_id, count=3))
        t2 = asyncio.create_task(self.reminder_agent.schedule_all_upcoming(user_id))
        res1, res2 = await asyncio.gather(t1, t2)
        return {"ideas": res1.get("ideas",[]), "reminders": res2}

    async def loop_check(self, user_id: str, interval_seconds: int = 60*60*24):
        # Long-running loop agent that checks events periodically
        logger.info("Supervisor loop_check started")
        while True:
            try:
                await self.reminder_agent.schedule_all_upcoming(user_id)
            except Exception as e:
                logger.exception("loop_check error")
            await asyncio.sleep(interval_seconds)

    async def pause_for_user(self, pending_op: Dict):
        # Save pending operation in DB so it can resume later
        pid = str(uuid.uuid4())
        db_insert("pending_ops", {"id": pid, "user_id": pending_op.get("user_id"), "op_type": pending_op.get("op_type"), "payload": json.dumps(pending_op.get("payload")), "status":"paused", "created_at": datetime.datetime.utcnow().isoformat()})
        logger.info(f"paused op saved {pid}")
        return pid

    async def resume_pending(self, pending_id: str):
        row = db_get("pending_ops", pending_id)
        if not row:
            return {"error":"not_found"}
        payload = json.loads(row["payload"])
        # interpret op_type and resume
        if row["op_type"] == "booking_confirm":
            # call booking agent
            res = await self.booking_agent.handle(payload["user_id"], payload["event_id"], payload["idea"])
            # mark completed (in demo, we don't update status)
            return {"res": res}
        return {"error":"unknown_op"}

# -------------------------
# Agent evaluation harness
# -------------------------
class AgentEvaluator:
    def __init__(self, supervisor: SupervisorAgent):
        self.supervisor = supervisor

    async def run_tests(self, user_id: str, event_id: str):
        # simple test suite
        results = {"memory_recall": None, "idea_generation": None, "booking_flow": None}
        # memory recall: can memory load event
        mem = self.supervisor.memory.load_user(user_id)
        results["memory_recall"] = len(mem.get("important_dates", [])) > 0
        # idea generation
        ideas = await self.supervisor.idea_agent.handle(user_id, event_id, count=2)
        results["idea_generation"] = isinstance(ideas.get("ideas", None), list) and len(ideas["ideas"])>0
        # booking flow (mock)
        if ideas.get("ideas"):
            booking = await self.supervisor.booking_agent.handle(user_id, event_id, ideas["ideas"][0])
            results["booking_flow"] = "booking_result" in booking or "order_result" in booking or "message" in booking
        return results

# -------------------------
# FastAPI demo endpoints
# -------------------------
app = FastAPI(title="Life Celebrations Concierge (Demo)")

# init components
init_db()
llm = LLMClient()
memory = MemoryBank()
sessions = InMemorySessionService()
a2a = A2AChannel()
tools = {
    "calendar": CalendarTool(),
    "restaurant_booking": RestaurantBookingTool(),
    "gift_order": GiftOrderTool(),
    "search": SearchTool(),
    "code_exec": CodeExecTool(),
    "openapi:mock": OpenAPITool("mock")
}
supervisor = SupervisorAgent(llm, tools, a2a, memory, sessions)
evaluator = AgentEvaluator(supervisor)

# Pydantic models for API
class CreateUser(BaseModel):
    name: str
    email: str

class CreateEvent(BaseModel):
    user_id: str
    title: str
    date: str
    recurrence: Optional[str] = "yearly"
    context: Optional[str] = ""

@app.post("/users", status_code=201)
def api_create_user(u: CreateUser):
    user_id = str(uuid.uuid4())
    db_insert("users", {"id": user_id, "name": u.name, "email": u.email, "timezone": "Asia/Kolkata"})
    # seed memory
    memory.load_user(user_id)
    logger.info(f"user created {user_id}")
    return {"user_id": user_id}

@app.post("/events", status_code=201)
def api_add_event(e: CreateEvent):
    # use calendar tool
    res = asyncio.get_event_loop().run_until_complete(MCPWrapper(tools["calendar"]).call(action="add_event", user_id=e.user_id, title=e.title, date=e.date, recurrence=e.recurrence, context=e.context))
    if not res.success:
        raise HTTPException(status_code=500, detail="failed to add event")
    # also add to long-term memory
    memory.add_event(e.user_id, res.payload["event"])
    return {"event": res.payload["event"]}

@app.get("/events/{user_id}")
def api_list_events(user_id: str):
    rows = db_query("events", where=f"user_id='{user_id}'")
    return {"events": rows}

@app.post("/generate_ideas/{event_id}")
async def api_generate_ideas(event_id: str):
    ev = db_get("events", event_id)
    if not ev:
        raise HTTPException(status_code=404, detail="event not found")
    res = await supervisor.idea_agent.handle(ev["user_id"], event_id, count=3)
    return res

@app.post("/book_idea/{event_id}")
async def api_book_idea(event_id: str, idea_index: int = 0):
    ev = db_get("events", event_id)
    if not ev:
        raise HTTPException(status_code=404, detail="event not found")
    ideas_res = await supervisor.idea_agent.handle(ev["user_id"], event_id, count=3)
    ideas = ideas_res.get("ideas", [])
    if idea_index >= len(ideas):
        raise HTTPException(status_code=400, detail="invalid idea index")
    chosen = ideas[idea_index]
    # For safety: pause and create pending_op for user confirmation in demo
    pending_op = {"user_id": ev["user_id"], "op_type":"booking_confirm", "payload": {"user_id": ev["user_id"], "event_id": event_id, "idea": chosen}}
    pid = await supervisor.pause_for_user(pending_op)
    return {"status":"paused_for_confirmation","pending_id": pid, "idea": chosen}

@app.post("/confirm_pending/{pending_id}")
async def api_confirm_pending(pending_id: str):
    res = await supervisor.resume_pending(pending_id)
    return res

@app.post("/parallel_run/{event_id}")
async def api_parallel_run(event_id: str):
    ev = db_get("events", event_id)
    if not ev:
        raise HTTPException(status_code=404, detail="event not found")
    res = await supervisor.run_parallel(ev["user_id"], event_id)
    return res

@app.get("/metrics")
def api_metrics():
    return Metrics.snapshot()

@app.get("/evaluate/{user_id}/{event_id}")
async def api_evaluate(user_id: str, event_id: str):
    res = await evaluator.run_tests(user_id, event_id)
    return res

@app.get("/pending_ops")
def api_pending_ops():
    return db_query("pending_ops")

# Simple health endpoint
@app.get("/health")
def health():
    return {"status":"ok"}

# -------------------------
# Demo runner
# -------------------------
if __name__ == "__main__":
    import uvicorn
    print("Run with: uvicorn life_concierge:app --reload --port 8000")
    uvicorn.run("life_concierge:app", host="0.0.0.0", port=8000, reload=True)


