import threading
import time
import uuid
import queue
import json
import random
from collections import defaultdict, deque
from typing import Any, Dict, Optional, List, Callable, Tuple

# %% [markdown]
# ## Observability primitives: Logger, Tracer, Metrics

# %%
class SimpleLogger:
    def info(self, msg: str):
        print(f"[INFO] {time.strftime('%H:%M:%S')} {msg}")

    def debug(self, msg: str):
        print(f"[DEBUG]{time.strftime('%H:%M:%S')} {msg}")

    def error(self, msg: str):
        print(f"[ERROR]{time.strftime('%H:%M:%S')} {msg}")

class Tracer:
    def start_span(self, name: str) -> str:
        span_id = str(uuid.uuid4())[:8]
        return span_id

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
# - `InMemorySessionService` stores session metadata
# - `MemoryBank` stores long-term memory per session; includes a simple context compaction

# %%
class InMemorySessionService:
    def __init__(self):
        self.sessions = {}

    def create_session(self, user_id: str) -> str:
        sid = str(uuid.uuid4())
        self.sessions[sid] = {"user_id": user_id, "created": time.time()}
        logger.info(f"session created: {sid} for user {user_id}")
        return sid

    def get_session(self, sid: str):
        return self.sessions.get(sid)

class MemoryBank:
    def __init__(self, compaction_threshold: int = 5):
        self.memories = defaultdict(list)  # session_id -> list of messages
        self.compaction_threshold = compaction_threshold

    def add(self, session_id: str, message: str):
        self.memories[session_id].append({"ts": time.time(), "text": message})
        if len(self.memories[session_id]) >= self.compaction_threshold:
            self.compact(session_id)

    def retrieve(self, session_id: str, k: int = 5) -> List[str]:
        return [m["text"] for m in self.memories[session_id][-k:]]

    def compact(self, session_id: str):
        # naive compaction: join oldest two messages into a summary string
        items = self.memories[session_id]
        if len(items) < 2:
            return
        oldest = items.pop(0)
        second = items.pop(0)
        summarized = f"(COMPACTED) {oldest['text']} | {second['text']}"
        items.insert(0, {"ts": time.time(), "text": summarized})
        logger.info(f"compacted memory for {session_id}; new length {len(items)}")

session_service = InMemorySessionService()
memory_bank = MemoryBank(compaction_threshold=4)

# %% [markdown]
# ## A2A Protocol & MCP (Message/Tool Broker)
# - Messages are JSON objects with `id`, `from`, `to`, `type`, `payload`, `trace_id`
# - MCP routes messages and exposes tools

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
        self.tools = {}  # tool_name -> callable

    def register_agent(self, agent_id: str):
        # ensure a message queue
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
# ## Tools
# - `search_tool` simulates a small knowledge base search
# - `compute_tool` performs a toy computation
# Register them into MCP.

# %%
# Toy knowledge base
KB = {
    "agents": "An agent is an autonomous program that perceives and acts in an environment.",
    "mcp": "MCP is the message and tool broker providing routing and tool access.",
    "memory": "MemoryBank stores session messages and supports compaction."
}

def search_tool(query: str, top_k: int = 3):
    # naive matching
    results = []
    for k, v in KB.items():
        score = sum(1 for tok in query.lower().split() if tok in v.lower())
        if score > 0:
            results.append((score, k, v))
    results.sort(reverse=True)
    return [r[2] for r in results[:top_k]]

def compute_tool(x: int, y: int):
    time.sleep(0.1)  # simulate work
    return {"sum": x + y, "product": x * y}

mcp.register_tool("search", search_tool)
mcp.register_tool("compute", compute_tool)

# %% [markdown]
# ## Agent base class
# - Agents run in separate threads (for parallel behaviour)
# - Agents can be "sequential" by chaining messages through agents in a pipeline

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
                # idle action hook
                time.sleep(0.05)
        logger.info(f"Agent {self.agent_id} stopped")

# %% [markdown]
# ## Agent Behaviors
# We'll define a few behaviors:
# - `explainer_agent`: uses `search_tool` to explain concepts and stores in memory
# - `pipeline_agent`: transforms payload and forwards to next agent (sequential)
# - `loop_agent`: periodically polls queue and emits heartbeat or tasks

# %%
def explainer_behavior(agent: Agent, msg: Message):
    payload = msg.payload
    trace = msg.trace_id
    logger.info(f"{agent.agent_id} received {msg.type} from {msg.from_agent} trace={trace}")
    if msg.type == "explain_request":
        query = payload.get("query", "")
        # use tool via MCP
        start = time.time()
        results = agent.mcp.call_tool("search", query)
        metrics.time_it("tool.search.latency", time.time() - start)
        explanation = results[0] if results else f"No info for '{query}'"
        # save to memory
        memory_bank.add(payload["session_id"], explanation)
        # respond
        resp = Message(agent.agent_id, msg.from_agent, "explain_response", {"explanation": explanation, "session_id": payload["session_id"]}, trace_id=trace)
        agent.mcp.send(resp)

def pipeline_behavior_factory(next_agent: str):
    def behavior(agent: Agent, msg: Message):
        logger.info(f"{agent.agent_id} pipeline got {msg.type}")
        payload = msg.payload
        # transform: append agent id to text
        text = payload.get("text", "")
        new_text = f"{text} -> processed_by_{agent.agent_id}"
        # forward to next agent
        forward = Message(agent.agent_id, next_agent, "pipeline_transform", {"text": new_text, "session_id": payload.get("session_id")}, trace_id=msg.trace_id)
        agent.mcp.send(forward)
    return behavior

def loop_behavior(agent: Agent, msg: Message):
    # loop agent listens for "start_loop" or "stop_loop"
    if msg.type == "start_loop":
        logger.info(f"{agent.agent_id} starting internal loop for session {msg.payload.get('session_id')}")
        # spawn a background loop (not blocking the thread)
        def background_loop():
            for i in range(3):
                time.sleep(1.0)
                ping = Message(agent.agent_id, msg.payload.get("notify_to"), "heartbeat", {"count": i+1, "session_id": msg.payload.get("session_id")}, trace_id=msg.trace_id)
                agent.mcp.send(ping)
                metrics.incr("loop_agent.heartbeats")
            # end notification
            done = Message(agent.agent_id, msg.payload.get("notify_to"), "loop_done", {"session_id": msg.payload.get("session_id")}, trace_id=msg.trace_id)
            agent.mcp.send(done)
        threading.Thread(target=background_loop, daemon=True).start()
    elif msg.type == "stop_loop":
        logger.info(f"{agent.agent_id} received stop_loop (no op in this simple demo)")

# %% [markdown]
# ## Build Agents: pipeline A -> B -> explainer; plus loop agent and a controller
# Controller will send tasks and collect responses for evaluation.

# %%
# agent IDs
AG_A = "agent.pipeline.A"
AG_B = "agent.pipeline.B"
AG_EXPLAIN = "agent.explainer"
AG_LOOP = "agent.loop"
AG_CONTROLLER = "agent.controller"

# create agents
agent_A = Agent(AG_A, mcp, pipeline_behavior_factory(AG_B))
agent_B = Agent(AG_B, mcp, pipeline_behavior_factory(AG_EXPLAIN))
agent_explainer = Agent(AG_EXPLAIN, mcp, explainer_behavior)
agent_loop = Agent(AG_LOOP, mcp, loop_behavior)

# controller behavior implemented inline (it will run in main thread and also register for messages)
mcp.register_agent(AG_CONTROLLER)

# start agents
for a in [agent_A, agent_B, agent_explainer, agent_loop]:
    a.start()

# %% [markdown]
# ## Controller: orchestrates tasks, keeps session, evaluates responses

# %%
class Controller:
    def __init__(self, controller_id: str, mcp: MCP):
        self.id = controller_id
        self.mcp = mcp
        self.responses = []
        self.eval_results = []
        self.session_id = session_service.create_session("kaggle_user_1")

    def send_pipeline_task(self, text: str):
        msg = Message(self.id, AG_A, "pipeline_input", {"text": text, "session_id": self.session_id})
        self.mcp.send(msg)

    def ask_explain(self, query: str):
        trace = tracer.start_span("ctrl.explain")
        msg = Message(self.id, AG_EXPLAIN, "explain_request", {"query": query, "session_id": self.session_id}, trace_id=trace)
        self.mcp.send(msg)

    def start_loop(self):
        msg = Message(self.id, AG_LOOP, "start_loop", {"notify_to": self.id, "session_id": self.session_id})
        self.mcp.send(msg)

    def receive(self, timeout=5.0):
        # collect until timeout
        start = time.time()
        while time.time() - start < timeout:
            msg = self.mcp.recv(self.id, timeout=0.5)
            if msg:
                logger.info(f"Controller received {msg.type} from {msg.from_agent}")
                self.handle(msg)
            else:
                time.sleep(0.1)

    def handle(self, msg: Message):
        if msg.type == "explain_response":
            explanation = msg.payload["explanation"]
            # evaluate: does explanation contain any KB keywords? (toy eval)
            correct = any(k in explanation.lower() for k in ["agent", "mcp", "memory"])
            latency = random.uniform(0.01, 0.05)  # simulated response latency
            self.eval_results.append({"msg_id": msg.id, "correct": correct, "latency": latency})
            self.responses.append(explanation)
            metrics.incr("controller.responses")
        elif msg.type == "pipeline_transform":
            # final pipeline output arrived at B but forwarded to explainer; we may get intermediate messages
            logger.debug(f"controller sees pipeline transform: {msg.payload}")
        elif msg.type in ("heartbeat", "loop_done"):
            logger.info(f"controller got loop message: {msg.type} payload={msg.payload}")
        else:
            logger.debug(f"controller unhandled message {msg.type}")

controller = Controller(AG_CONTROLLER, mcp)

# %% [markdown]
# ## Demo run: send pipeline tasks, ask explain, start loop, collect responses

# %%
# 1) pipeline task -> flows A -> B -> explainer (via messaging)
controller.send_pipeline_task("Start with: hello world")
# 2) direct explain query
controller.ask_explain("What is an agent?")

# 3) start loop agent to emit heartbeats
controller.start_loop()

# Let system run and controller collect messages
controller.receive(timeout=6.0)

# Wait briefly to allow background threads to finish sending
time.sleep(1.0)

# %% [markdown]
# ## Evaluation & Metrics snapshot

# %%
print("\n--- Responses collected ---")
for r in controller.responses:
    print("-", r)

print("\n--- Eval results ---")
for ev in controller.eval_results:
    print("-", ev)

print("\n--- Observability snapshot ---")
print(json.dumps(metrics.snapshot(), indent=2))

# %% [markdown]
# ## Clean up: stop agents

# %%
for a in [agent_A, agent_B, agent_explainer, agent_loop]:
    a.stop()
# join to ensure clean exit
for a in [agent_A, agent_B, agent_explainer, agent_loop]:
    a.join(timeout=1.0)

logger.info("Demo finished.")

