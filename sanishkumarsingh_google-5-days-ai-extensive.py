!pip install --quiet google-genai


import os
import time
import uuid
import json
import sqlite3
import logging
import threading
from queue import Queue, Empty
from typing import Any, Dict, List, Optional


# Kaggle secrets for safe API handling
try:
    from kaggle_secrets import UserSecretsClient
    _use_kaggle_secrets = True
except Exception:
    _use_kaggle_secrets = False


# GenAI client import (wrapped to avoid hard error if not present)
try:
    import google.genai as genai
    _have_genai = True
except Exception:
    _have_genai = False


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("enterprise_agent")


METRICS = {
    "queries_received": 0,
    "agents_invoked": 0,
    "long_tasks_started": 0,
    "long_tasks_completed": 0,
    "evaluations_done": 0,
}


def inc_metric(name: str, n: int = 1):
    if name in METRICS:
        METRICS[name] += n


class MockLLM:
    def __init__(self, name="mock-llm"):
        self.name = name
    
    
    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        logger.info("MockLLM.generate called")
        p = prompt.lower()
        if "order" in p and "ord" in p:
            # return a fake order status mention
            return "Your order appears to be processing. Estimated delivery in 5 days."
        if "refund" in p:
            return "Our refund policy: refunds processed within 5-7 business days."
        if "summarize" in p:
            return "Summary: simulated summary."
        return f"[Mock] I would respond to: {prompt[:120]}"


class GenAIWrapper:
    def __init__(self, client: Any, model: str = "gemini-2.0-flash"):
        self.client = client
        self.model = model
        
    
    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        # The google-genai client expects structured inputs; use generate_content
        try:
            # Use the content API; the exact call may vary by SDK version
            resp = self.client.models.generate_content(model=self.model, contents=prompt)
            # resp may have .text or .candidates; adapt to what's available
            if hasattr(resp, "text"):
                return resp.text
            if hasattr(resp, "candidates") and len(resp.candidates) > 0:
                return resp.candidates[0].content
            return str(resp)
        except Exception as e:
            logger.exception("GenAI call failed, falling back to string repr")
            return str(e)



llm_adapter = None


if _have_genai and _use_kaggle_secrets:
    try:
        user_secrets = UserSecretsClient()
        api_key = user_secrets.get_secret("GEMINI_API_KEY")
        # Work around some async cleanup issues in certain environments
        client = genai.Client(api_key=api_key, http_options={"follow_redirects": True})
        llm_adapter = GenAIWrapper(client)
        logger.info("Connected to GenAI client (real).")
    except Exception as e:
        logger.warning("Could not initialize real GenAI client: %s", e)
        llm_adapter = MockLLM()
else:
    # No real genai client available; use mock adapter
    llm_adapter = MockLLM()
    logger.info("Using MockLLM adapter.")


class MemoryDB:
    def __init__(self, filename: str = "memory.db"):
        self.conn = sqlite3.connect(filename, check_same_thread=False)
        self._init()
    def _init(self):
        c = self.conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            role TEXT,
            content TEXT,
            created_at REAL
            )
        """
        )
        self.conn.commit()
    
    
    def store(self, session_id: str, role: str, content: str):
        _id = str(uuid.uuid4())
        ts = time.time()
        c = self.conn.cursor()
        c.execute("INSERT INTO memory (id,session_id,role,content,created_at) VALUES (?,?,?,?,?)",
            (_id, session_id, role, content, ts))
        self.conn.commit()


    def fetch_recent(self, session_id: str, limit: int = 10):
        c = self.conn.cursor()
        c.execute("SELECT role,content,created_at FROM memory WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
            (session_id, limit))
        rows = c.fetchall()
        return [{"role": r[0], "content": r[1], "created_at": r[2]} for r in rows]
        
    
# Simple in-memory session manager
class Session:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history: List[Dict[str, str]] = []
        self.created_at = time.time()
        self.last_active = time.time()
    
    
    def add(self, role: str, text: str):
        self.history.append({"role": role, "content": text, "ts": time.time()})
        self.last_active = time.time()


class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        self.lock = threading.Lock()
    
    
    def get(self, session_id: Optional[str] = None) -> Session:
        with self.lock:
            if not session_id:
                session_id = str(uuid.uuid4())
            if session_id not in self.sessions:
                self.sessions[session_id] = Session(session_id)
            return self.sessions[session_id]
        
        
memory_db = MemoryDB(":memory:")
session_mgr = SessionManager()



SIM_ORDERS = {
    "ORD1001": {"status": "Shipped", "eta": "2025-12-05", "items": ["widget A"]},
    "ORD1002": {"status": "Processing", "eta": "2025-12-08", "items": ["gadget X"]},
}


SIM_FAQ = {
    "refund": "Refunds are processed within 5-7 business days.",
    "shipping": "Standard shipping 3-5 business days.",
}



def get_order_details(order_id: str) -> Dict[str, Any]:
    logger.info("Tool: get_order_details %s", order_id)
    inc_metric("agents_invoked")
    return SIM_ORDERS.get(order_id, {"error": "not_found"})



def fetch_faq(query: str) -> str:
    logger.info("Tool: fetch_faq %s", query)
    for k, v in SIM_FAQ.items():
        if k in query.lower():
           return v
    return "No matching FAQ found."




def simulated_search(query: str) -> List[str]:
    logger.info("Tool: simulated_search %s", query)
    return [f"Snippet {i} for '{query}'" for i in range(1, 4)]


def code_execute_stub(code: str) -> str:
    logger.info("Tool: code_execute_stub")
    # DO NOT EXECUTE untrusted code. This is a stub that simulates execution.
    return f"Executed code (simulated). len={len(code)}"



class AgentBus:
    def __init__(self):
        self.inboxes: Dict[str, Queue] = {}
        self.lock = threading.Lock()
        
    
    def register(self, agent_id: str):
        with self.lock:
            self.inboxes.setdefault(agent_id, Queue())
            logger.info("A2A register %s", agent_id)
    
    
    def send(self, to_agent: str, message: Dict[str, Any]):
        q = self.inboxes.get(to_agent)
        if q:
            q.put(message)
            logger.info("A2A sent to %s", to_agent)
        else:
            logger.warning("A2A no inbox for %s", to_agent)


    def receive(self, agent_id: str, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        q = self.inboxes.get(agent_id)
        if not q:
           return None
        try:
           return q.get(timeout=timeout)
        except Empty:
           return None


bus = AgentBus()


class BaseAgent(threading.Thread):
    def __init__(self, agent_id: str):
        super().__init__(daemon=True)
        self.agent_id = agent_id
        self.inbox = Queue()
        self.running = True
        bus.register(self.agent_id)
        logger.info("Agent %s initialized", agent_id)
    
    
    def send_a2a(self, to_agent: str, payload: Dict[str, Any]):
        pkg = {"from": self.agent_id, "payload": payload}
        bus.send(to_agent, pkg)
    
    
    def receive_a2a(self, timeout: float = 0.05) -> Optional[Dict[str, Any]]:
        return bus.receive(self.agent_id, timeout)
    
    
    def stop(self):
        self.running = False


    def handle(self, message: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError()
    
    
    def run(self):
        while self.running:
            try:
                item = self.inbox.get(timeout=0.1)
                try:
                    self.handle(item)
                except Exception:
                    logger.exception("Error in handling local inbox item")
            except Empty:
                pass
            a2a_msg = self.receive_a2a(timeout=0.05)
            if a2a_msg:
                try:
                    self.handle(a2a_msg.get("payload", {}))
                except Exception:
                    logger.exception("Error handling A2A message")
            time.sleep(0.01)

class IntakeAgent(BaseAgent):
    def handle(self, message: Dict[str, Any]) -> Dict[str, Any]:
        inc_metric("agents_invoked")
        session_id = message.get("session_id")
        query = message.get("query", "")
        logger.info("Intake handling: %s", query)
        session = session_mgr.get(session_id)
        session.add("user", query)
        memory_db.store(session_id, "user", query)
        qlower = query.lower()
        if "ord" in qlower or "order" in qlower:
            target = "data_retrieval"
        elif "refund" in qlower:
            target = "faq_agent"
        else:
            target = "response_agent"
        self.send_a2a(target, {"session_id": session_id, "query": query})
        return {"routed_to": target}



class DataRetrievalAgent(BaseAgent):
    def handle(self, message: Dict[str, Any]) -> Dict[str, Any]:
        inc_metric("agents_invoked")
        session_id = message.get("session_id")
        query = message.get("query", "")
        logger.info("DataRetrieval handling: %s", query)
        # extract order id naive
        import re
        m = re.search(r"(ORD\d+)", query.upper())
        if m:
            order_id = m.group(1)
            result = get_order_details(order_id)
        else:
            result = {"search": simulated_search(query)}
        memory_db.store(session_id, "retrieval", json.dumps(result))
        self.send_a2a("response_agent", {"session_id": session_id, "query": query, "retrieval": result})
        return result


class FAQAgent(BaseAgent):
    def handle(self, message: Dict[str, Any]) -> Dict[str, Any]:
        inc_metric("agents_invoked")
        session_id = message.get("session_id")
        query = message.get("query", "")
        logger.info("FAQAgent handling: %s", query)
        ans = fetch_faq(query)
        memory_db.store(session_id, "faq", ans)
        self.send_a2a("response_agent", {"session_id": session_id, "query": query, "retrieval": {"faq": ans}})
        return {"faq": ans}


class ResponseAgent(BaseAgent):
    def handle(self, message: Dict[str, Any]) -> Dict[str, Any]:
        inc_metric("agents_invoked")
        session_id = message.get("session_id")
        query = message.get("query", "")
        retrieval = message.get("retrieval", {})
        logger.info("ResponseAgent building reply for: %s", query)
        session = session_mgr.get(session_id)
        # Build prompt from memory + retrieval
        recent = memory_db.fetch_recent(session_id, limit=6)
        prompt_parts = [f"You are a helpful support assistant.", f"Session history: {session.history[-6:]}", f"Memory: {recent}", f"Retrieval: {retrieval}", f"User query: {query}", "Provide a concise empathetic reply."]
        prompt = "\n\n".join([str(p) for p in prompt_parts])
        # Use LLM adapter
        response_text = llm_adapter.generate(prompt)
        session.add("assistant", response_text)
        memory_db.store(session_id, "assistant", response_text)
        score = Evaluator.evaluate_response(query, response_text, retrieval)
        inc_metric("evaluations_done")
        logger.info("ResponseAgent produced reply (score=%s)", score)
        # For demo: put result in a simple 'last_reply' memory key
        memory_db.store(session_id, "last_reply", response_text)
        return {"response": response_text, "score": score}




class Evaluator:
    @staticmethod
    def evaluate_response(query: str, response: str, retrieval: Dict[str, Any]) -> float:
        score = 0.1
        q = query.lower()
        r = response.lower()
        if "order" in q:
            for kw in ["shipped", "processing", "delivered", "eta"]:
                if kw in r:
                    score += 0.5
                    break
        if "refund" in q and ("refund" in r or "return" in r):
            score += 0.4
        if "[mock]" in r:
            score -= 0.05
        return max(0.0, min(1.0, score))


class LongTask:
    def __init__(self, task_id: str, session_id: str, payload: Dict[str, Any]):
        self.task_id = task_id
        self.session_id = session_id
        self.payload = payload
        self._pause = threading.Event()
        self._pause.set()
        self._cancel = False
        self.status = "created"
        self.progress = 0.0
    
    
    def pause(self):
        self._pause.clear()
        self.status = "paused"
    
    
    def resume(self):
        self._pause.set()
        self.status = "running"
    

    def cancel(self):
        self._cancel = True
        self.status = "cancelled"
    
    
    def run(self, on_complete=None):
        inc_metric("long_tasks_started")
        self.status = "running"
        for i in range(20):
            if self._cancel:
                return
            self._pause.wait()
            time.sleep(0.2)
            self.progress = (i + 1) / 20.0
        self.status = "completed"
        inc_metric("long_tasks_completed")
        if on_complete:
            on_complete(self)


class LongTaskManager:
    def __init__(self):
        self.tasks: Dict[str, LongTask] = {}
        self.lock = threading.Lock()
    
    
    def start_export(self, session_id: str, payload: Dict[str, Any]) -> str:
        task_id = str(uuid.uuid4())
        task = LongTask(task_id, session_id, payload)
        with self.lock:
            self.tasks[task_id] = task
        t = threading.Thread(target=task.run, args=(self._complete_cb,), daemon=True)
        t.start()
        return task_id
    
    
    def _complete_cb(self, task: LongTask):
        logger.info("Task %s completed callback", task.task_id)


    def get_status(self, task_id: str) -> Dict[str, Any]:
        task = self.tasks.get(task_id)
        if not task:
            return {"error": "not_found"}
        return {"task_id": task.task_id, "status": task.status, "progress": task.progress}
    
    
    def pause(self, task_id: str) -> bool:
        t = self.tasks.get(task_id)
        if t:
            t.pause()
            return True
        return False
        
    
    def resume(self, task_id: str) -> bool:
        t = self.tasks.get(task_id)
        if t:
            t.resume()
            return True
        return False
    
    
long_task_mgr = LongTaskManager()


intake_agent = IntakeAgent("intake_agent")
data_agent = DataRetrievalAgent("data_retrieval")
faq_agent = FAQAgent("faq_agent")
response_agent = ResponseAgent("response_agent")


for a in [intake_agent, data_agent, faq_agent, response_agent]:
    a.start()



def submit_query(session_id: Optional[str], query: str, wait_seconds: float = 2.0):
    inc_metric("queries_received")
    if not session_id:
        session_id = str(uuid.uuid4())
    pkg = {"session_id": session_id, "query": query}
    intake_agent.inbox.put(pkg)
    # wait briefly for pipeline to produce a reply
    start = time.time()
    last_reply = None
    while time.time() - start < wait_seconds:
        rows = memory_db.fetch_recent(session_id, limit=5)
        for r in rows:
            if r["role"] == "assistant" if isinstance(r, dict) and False else False:
                pass
        # simpler: check last_reply key
        recent = memory_db.fetch_recent(session_id, limit=10)
        for item in recent:
            if item["role"] == "last_reply":
                last_reply = item["content"]
                break
        # fallback: look in 'assistant' items
        for item in recent:
            if item["role"] == "assistant":
                last_reply = item["content"]
                break
        if last_reply:
            break
        time.sleep(0.1)
    return {"session_id": session_id, "reply": last_reply}


if __name__ == "__main__":
    print("Running a quick demo (this block runs when executing the script)")
    sid = "demo-session-1"
    print("Submitting: Where is my order ORD1002?")
    submit_query(sid, "Where is my order ORD1002?")
    time.sleep(1.2)
    print("Memory recent:", memory_db.fetch_recent(sid, limit=10))
    print("Starting a long task...")
    tid = long_task_mgr.start_export(sid, {"type": "export_data"})
    print("Task id:", tid)
    time.sleep(0.8)
    long_task_mgr.pause(tid)
    print("Paused task")
    time.sleep(0.5)
    long_task_mgr.resume(tid)
    print("Resumed task")
    time.sleep(1.5)
    print("Final metrics:", METRICS)

