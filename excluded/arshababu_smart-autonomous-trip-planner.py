
import os
import time
import json
import threading
import uuid
import logging
from typing import Any, Dict, List, Optional, Callable

# Optional rich print for nicer console output
try:
    from rich import print as rprint
except Exception:
    rprint = print

# --------------------------- Observability ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
)
logger = logging.getLogger("autotravel")

# --------------------------- Memory Bank (Long-term storage) ---------------------------
class MemoryBank:
    def __init__(self, filepath: str = "autotravel_memory.json"):
        self.filepath = filepath
        self._lock = threading.Lock()
        self._mem: Dict[str, Any] = {}
        self._load()

    def _load(self):
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                self._mem = json.load(f)
            logger.info("Memory loaded (keys=%d)", len(self._mem))
        except FileNotFoundError:
            self._mem = {}

    def _save(self):
        with self._lock:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self._mem, f, indent=2)

    def get(self, key: str, default=None):
        return self._mem.get(key, default)

    def set(self, key: str, value: Any):
        self._mem[key] = value
        self._save()

    def append(self, key: str, value: Any):
        arr = self._mem.setdefault(key, [])
        if not isinstance(arr, list):
            arr = [arr]
            self._mem[key] = arr
        arr.append(value)
        self._save()

    def snapshot_keys(self):
        return list(self._mem.keys())


# --------------------------- Session (Ephemeral state) ---------------------------
class Session:
    def __init__(self, user_id: str, memory: MemoryBank):
        self.session_id = str(uuid.uuid4())
        self.user_id = user_id
        self.memory = memory
        self.state: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []

    def push(self, event: Dict[str, Any]):
        event_record = {"time": time.time(), **event}
        self.history.append(event_record)
        logger.debug("Session %s event: %s", self.session_id, event)


# --------------------------- Tools (stubs) ---------------------------

def web_search_stub(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    logger.info("web_search_stub: %s", query)
    # Replace with real web search integration
    return [{"title": f"{query} - result {i}", "url": f"https://example.com/{i}", "snippet": "sample"} for i in range(1, max_results+1)]


def booking_api_stub(resource: str, params: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("booking_api_stub: %s %s", resource, params)
    return {"status": "ok", "resource": resource, "params": params, "booking_id": str(uuid.uuid4())}


def safe_eval(expression: str) -> Dict[str, Any]:
    logger.info("safe_eval called")
    try:
        allowed = {"__builtins__": {"len": len, "sum": sum, "min": min, "max": max}}
        result = eval(expression, allowed, {})
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# --------------------------- LLM Client Wrapper (Replaceable) ---------------------------
class LLMClient:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("VERTEX_API_KEY")
        # try to import openai as optional
        try:
            import openai
            self.openai = openai
            if self.api_key:
                self.openai.api_key = self.api_key
        except Exception:
            self.openai = None
            logger.warning("No LLM package available; using stub responses")

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        logger.debug("LLM.generate prompt len=%d", len(prompt))
        if self.openai:
            try:
                resp = self.openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.6,
                )
                return resp.choices[0].text.strip()
            except Exception as e:
                logger.exception("LLM call failed: %s", e)
                return ""
        # stubbed reply
        return f"[LLM stub] Summarize: {prompt[:120]}..."


# --------------------------- Agents ---------------------------
class Agent:
    def __init__(self, name: str, llm: LLMClient, memory: MemoryBank):
        self.name = name
        self.llm = llm
        self.memory = memory
        self.logger = logging.getLogger(f"agent.{self.name}")

    def act(self, session: Session, user_input: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class PlannerAgent(Agent):
    def act(self, session: Session, user_input: Dict[str, Any]) -> Dict[str, Any]:
        dest = user_input.get("destination")
        days = user_input.get("days", 3)
        prefs = user_input.get("preferences", {})
        prompt = (
            f"Create a {days}-day personalized itinerary for {dest}. Preferences: {prefs}. "
            "Output as JSON with day-by-day activities (3 per day) and short notes."
        )
        plan_text = self.llm.generate(prompt)
        # In production, parse JSON. Here we store LLM text as-is.
        itinerary = {"destination": dest, "days": days, "plan": plan_text}
        self.memory.append("itineraries", {"user": session.user_id, "itinerary": itinerary})
        session.push({"agent": self.name, "action": "generated_itinerary"})
        self.logger.info("Itinerary created for %s", dest)
        return {"itinerary": itinerary}


class BudgetAgent(Agent):
    def act(self, session: Session, user_input: Dict[str, Any]) -> Dict[str, Any]:
        budget = user_input.get("budget", 1000)
        dest = user_input.get("destination")
        days = user_input.get("days", 3)
        # Use simple heuristics combined with LLM suggestions
        heuristic = {
            "flights": round(budget * 0.35, 2),
            "accommodation": round(budget * 0.35, 2),
            "food": round(budget * 0.15, 2),
            "transport": round(budget * 0.08, 2),
            "activities": round(budget * 0.07, 2),
        }
        prompt = f"Given a budget of {budget} for {days} days in {dest}, provide cost tips and possible savings." 
        notes = self.llm.generate(prompt)
        estimate = {"heuristic": heuristic, "llm_notes": notes}
        self.memory.append("budget_estimates", {"user": session.user_id, "estimate": estimate})
        session.push({"agent": self.name, "action": "estimated_budget"})
        self.logger.info("Budget estimated: %s", heuristic)
        return {"estimate": estimate}


class BookingAgent(Agent):
    def act(self, session: Session, user_input: Dict[str, Any]) -> Dict[str, Any]:
        # In production, call real booking APIs. Here, simulate booking suggestions.
        dest = user_input.get("destination")
        dates = user_input.get("dates", {})
        flights = web_search_stub(f"flights to {dest}")
        hotels = web_search_stub(f"hotels in {dest}")
        session.push({"agent": self.name, "action": "suggested_bookings"})
        # return simulated booking links
        return {"flights": flights, "hotels": hotels}


class CarbonAgent(Agent):
    def act(self, session: Session, user_input: Dict[str, Any]) -> Dict[str, Any]:
        # Simple carbon footprint heuristic
        dest = user_input.get("destination")
        transport_mode = user_input.get("transport_mode", "flight")
        days = user_input.get("days", 3)
        per_day_activity_emission = 5  # kg CO2 per day placeholder
        transport_emission = 300 if transport_mode == "flight" else 50
        total = transport_emission + days * per_day_activity_emission
        suggestion = "Choose trains or direct flights where possible to reduce CO2." 
        session.push({"agent": self.name, "action": "carbon_estimated"})
        self.memory.append("carbon_reports", {"user": session.user_id, "carbon_kg": total})
        return {"carbon_kg": total, "suggestion": suggestion}


# --------------------------- Long-running Price Monitor (pause/resume) ---------------------------
class PriceMonitor:
    def __init__(self, memory: MemoryBank, check_interval: int = 60, callback: Optional[Callable] = None):
        self.memory = memory
        self.check_interval = check_interval
        self.callback = callback
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._thread and self._thread.is_alive():
            logger.warning("PriceMonitor already running")
            return
        self._stop_event.clear()
        self._pause_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("PriceMonitor started")

    def pause(self):
        self._pause_event.set()
        logger.info("PriceMonitor paused")

    def resume(self):
        self._pause_event.clear()
        logger.info("PriceMonitor resumed")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("PriceMonitor stopped")

    def _run_loop(self):
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(1)
                continue
            # In production: fetch prices via APIs, compute deltas, store events
            # Here: simulate a price check and store a synthetic event
            sample_event = {"time": time.time(), "route": "NYC-Bali", "price": 500 + int(time.time()) % 50}
            logger.info("PriceMonitor check: %s", sample_event)
            self.memory.append("price_checks", sample_event)
            if self.callback:
                try:
                    self.callback(sample_event)
                except Exception:
                    logger.exception("PriceMonitor callback failed")
            time.sleep(self.check_interval)


# --------------------------- Coordinator / Orchestrator ---------------------------
class Coordinator:
    def __init__(self, agents: List[Agent], llm: LLMClient, memory: MemoryBank, price_monitor: PriceMonitor):
        self.agents = {a.name: a for a in agents}
        self.llm = llm
        self.memory = memory
        self.price_monitor = price_monitor
        self.logger = logging.getLogger("coordinator")

    def run_parallel(self, session: Session, user_input: Dict[str, Any], timeout: float = 10.0) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        threads: List[threading.Thread] = []

        def run_agent(agent: Agent):
            try:
                results[agent.name] = agent.act(session, user_input)
            except Exception as e:
                results[agent.name] = {"error": str(e)}
                logger.exception("Agent %s failed", agent.name)

        for agent in self.agents.values():
            t = threading.Thread(target=run_agent, args=(agent,))
            t.start()
            threads.append(t)

        start = time.time()
        for t in threads:
            elapsed = time.time() - start
            remaining = max(0, timeout - elapsed)
            t.join(remaining)

        for t, name in zip(threads, list(self.agents.keys())):
            if t.is_alive():
                results[name] = {"error": "timeout"}
                logger.warning("Agent %s timed out", name)
        return results

    def run_sequential(self, session: Session, user_input: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for name, agent in self.agents.items():
            self.logger.info("Running %s sequentially", name)
            out[name] = agent.act(session, user_input)
        return out


# --------------------------- Simple Evaluator ---------------------------
class Evaluator:
    def evaluate(self, session: Session, results: Dict[str, Any]) -> Dict[str, Any]:
        score = 0
        reasons = []
        if "PlannerAgent" in results and results["PlannerAgent"].get("itinerary"):
            score += 40
        if "BudgetAgent" in results and results["BudgetAgent"].get("estimate"):
            score += 25
        if "BookingAgent" in results and results["BookingAgent"].get("flights"):
            score += 20
        if "CarbonAgent" in results:
            co2 = results["CarbonAgent"].get("carbon_kg")
            if co2 is not None and co2 < 500:
                score += 10
        mem_items = len(session.memory.get("itineraries", []))
        if mem_items > 0:
            reasons.append(f"{mem_items} itineraries in memory")
        return {"score": score, "reasons": reasons}


# --------------------------- CLI / Demo Flow ---------------------------
def demo_flow():
    rprint("[bold]Autonomous Travel Concierge — Demo[/bold]")
    memory = MemoryBank("autotravel_demo_memory.json")
    llm = LLMClient()

    planner = PlannerAgent("PlannerAgent", llm, memory)
    budgeter = BudgetAgent("BudgetAgent", llm, memory)
    booker = BookingAgent("BookingAgent", llm, memory)
    carbon = CarbonAgent("CarbonAgent", llm, memory)

    # Price monitor with simple callback to notify when price < threshold
    def price_callback(event):
        if event.get("price", 9999) < 520:
            logger.info("Price alert: %s", event)

    price_monitor = PriceMonitor(memory, check_interval=5, callback=price_callback)

    coord = Coordinator([planner, budgeter, booker, carbon], llm, memory, price_monitor)
    session = Session(user_id="user_demo", memory=memory)

    # start long-running price monitor
    price_monitor.start()

    user_input = {
        "destination": "Bali, Indonesia",
        "days": 5,
        "budget": 1500,
        "preferences": {"activities": ["beach", "temple", "hiking"]},
        "transport_mode": "flight",
    }

    session.push({"user_input": user_input})

    # Run agents in parallel
    results = coord.run_parallel(session, user_input, timeout=8.0)

    rprint("\n[green]Results (parallel):[/green]")
    for k, v in results.items():
        rprint(f"[blue]{k}[/blue]: {json.dumps(v, indent=2)[:800]}")

    evaluator = Evaluator()
    score = evaluator.evaluate(session, results)
    rprint(f"\n[bold]Evaluation:[/bold] {score}")

    rprint("\nSession history (last 5 events):")
    for ev in session.history[-5:]:
        rprint(ev)

    rprint("\nMemory keys:", memory.snapshot_keys())

    # Demonstrate pause/resume of price monitor
    time.sleep(6)
    price_monitor.pause()
    rprint("Price monitor paused — sleeping 3s")
    time.sleep(3)
    price_monitor.resume()
    rprint("Price monitor resumed — stopping in 4s")
    time.sleep(4)
    price_monitor.stop()

    rprint("Demo complete.")


# --------------------------- If used as a module: provide helpers for packaging ---------------------------
if __name__ == "__main__":
    demo_flow()







