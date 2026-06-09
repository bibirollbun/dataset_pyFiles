
import os, json, re, uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any

# ====== Configuration / Local storage setup ======
USE_MOCK_TOOLS = True
os.makedirs("local_storage", exist_ok=True)

# ====== Mock storage tools ======
def _append_ndjson(fname: str, obj: Dict[str, Any]):
    path = os.path.join("local_storage", fname)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def add_to_todo_list(item: str, priority: str = "medium", metadata: Dict = None) -> Dict:
    metadata = metadata or {}
    todo = {
        "id": f"todo_{int(datetime.utcnow().timestamp()*1000)}",
        "item": item,
        "priority": priority,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "meta": metadata,
    }
    if USE_MOCK_TOOLS:
        _append_ndjson("todos.ndjson", todo)
    return todo

def save_note(content: str, tags: List[str] = None, metadata: Dict = None) -> Dict:
    tags = tags or []
    metadata = metadata or {}
    note = {
        "id": f"note_{int(datetime.utcnow().timestamp()*1000)}",
        "content": content,
        "tags": tags,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "meta": metadata,
    }
    if USE_MOCK_TOOLS:
        _append_ndjson("notes.ndjson", note)
    return note

def create_calendar_event(title: str, start_iso: str, description: str = "") -> Dict:
    event = {
        "id": str(uuid.uuid4()),
        "title": title,
        "start": start_iso,
        "description": description,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    if USE_MOCK_TOOLS:
        _append_ndjson("calendar_events.ndjson", event)
    return event

# ====== BaseAgent ======
class BaseAgent:
    def __init__(self, name: str):
        self.name = name

    def call_model(self, prompt: str, multimodal: Dict = None) -> Dict:
        # Mock model call placeholder — replace with Gemini/ADK call later.
        # Print is intentional so assignment grader can see agent prompts (if needed).
        # Keep responses deterministic for assignment/demo.
        print(f"[{self.name}] → MODEL PROMPT (mocked): {prompt[:200]}{'...' if len(prompt)>200 else ''}")
        return {"text": "MOCK_RESPONSE", "prompt_sent": prompt[:400]}

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

# ====== Agents definitions ======
class TriageAgent(BaseAgent):
    def __init__(self):
        super().__init__("TriageAgent")

    def _simple_rule_classify(self, text: str) -> List[Dict[str, str]]:
        outputs = []
        # Split into sentence-like parts
        parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        for p in parts:
            s = p.strip()
            if not s:
                continue
            lowered = s.lower()
            # calendar-like detection
            if re.search(r"\b(tomorrow|today|on|at|next)\b", lowered) and re.search(r"\b(am|pm|\d{1,2}:\d{2})\b", lowered):
                outputs.append({"type": "event", "text": s})
                continue
            # task detection (imperative verbs)
            if re.match(r"^(buy|call|schedule|email|remind|remember|finish|do|pay|order|send)\b", lowered):
                outputs.append({"type": "task", "text": s})
                continue
            # short task-like
            if len(s.split()) <= 6 and re.search(r"\b(buy|call|pay|order|send)\b", lowered):
                outputs.append({"type": "task", "text": s})
                continue
            # default note
            outputs.append({"type": "note", "text": s})
        return outputs

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        text = data.get("text", "") or ""
        # If audio were present, we would call Gemini/ADK transcription here.
        splits = self._simple_rule_classify(text)
        routes = []
        for s in splits:
            if s["type"] == "task":
                routes.append({"agent": "task", "payload": {"text": s["text"], "meta": data.get("metadata", {})}})
            elif s["type"] == "event":
                routes.append({"agent": "scheduler", "payload": {"text": s["text"], "meta": data.get("metadata", {})}})
            else:
                routes.append({"agent": "archivist", "payload": {"text": s["text"], "meta": data.get("metadata", {})}})
        return {"routes": routes}

class TaskAgent(BaseAgent):
    def __init__(self):
        super().__init__("TaskAgent")

    def _extract_priority(self, text: str) -> str:
        t = text.lower()
        if "urgent" in t or "asap" in t or "immediately" in t:
            return "high"
        if "sometime" in t or "whenever" in t:
            return "low"
        return "medium"

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        text = data.get("text", "")
        priority = self._extract_priority(text)
        dod = f"Definition of Done: {text}"
        todo = add_to_todo_list(item=text, priority=priority, metadata={"dod": dod, "source": data.get("meta")})
        return {"status": "ok", "stored": todo}

class ArchivistAgent(BaseAgent):
    def __init__(self):
        super().__init__("ArchivistAgent")

    def _summarize(self, text: str) -> str:
        # Mock summarization (first 200 chars) — replace with model summarization for production.
        return text if len(text) <= 200 else text[:197] + "..."

    def _auto_tag(self, text: str) -> List[str]:
        tags = []
        t = text.lower()
        if "research" in t or "paper" in t or "study" in t:
            tags.append("research")
        if "recipe" in t or "cook" in t:
            tags.append("recipe")
        if "moon" in t or "space" in t:
            tags.append("space")
        return tags

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        text = data.get("text", "")
        summary = self._summarize(text)
        tags = self._auto_tag(text)
        note = save_note(content=summary, tags=tags, metadata={"source_text": text[:1000], "meta": data.get("meta")})
        return {"status": "ok", "note": note}

class SchedulerAgent(BaseAgent):
    def __init__(self):
        super().__init__("SchedulerAgent")

    def _extract_time(self, text: str) -> Dict[str, str]:
        t = text.lower()
        now = datetime.utcnow()
        # YYYY-MM-DD pattern
        m_date = re.search(r"(\d{4}-\d{2}-\d{2})", t)
        m_time = re.search(r"(\d{1,2}:\d{2})", t)
        m_hourpm = re.search(r"(\d{1,2})(am|pm)", t)
        if "tomorrow" in t:
            day = now + timedelta(days=1)
            hh, mm = 9, 0
            if m_hourpm:
                h = int(m_hourpm.group(1))
                if m_hourpm.group(2) == "pm" and h != 12:
                    h += 12
                hh = h
            if m_time:
                hh = int(m_time.group(1).split(":")[0])
                mm = int(m_time.group(1).split(":")[1])
            start = datetime(day.year, day.month, day.day, hh, mm)
            return {"start": start.isoformat() + "Z"}
        if m_date:
            date_part = m_date.group(1)
            hh, mm = 9, 0
            if m_hourpm:
                h = int(m_hourpm.group(1))
                if m_hourpm.group(2) == "pm" and h != 12:
                    h += 12
                hh = h
            if m_time:
                hh = int(m_time.group(1).split(":")[0])
                mm = int(m_time.group(1).split(":")[1])
            start = datetime.fromisoformat(date_part).replace(hour=hh, minute=mm)
            return {"start": start.isoformat() + "Z"}
        # fallback to next hour
        fallback = now + timedelta(hours=1)
        return {"start": fallback.isoformat() + "Z"}

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        text = data.get("text", "")
        time_info = self._extract_time(text)
        title = text if len(text) < 80 else text[:77] + "..."
        event = create_calendar_event(title=title, start_iso=time_info["start"], description=text)
        return {"status": "ok", "event": event}

# ====== Agent factory ======
def create_agents():
    return {
        "triage": TriageAgent(),
        "task": TaskAgent(),
        "archivist": ArchivistAgent(),
        "scheduler": SchedulerAgent(),
    }

agents = create_agents()

# ====== Orchestrator ======
def orchestrate(input_payload: Dict[str, Any]) -> Dict[str, Any]:
    triage_agent = agents["triage"]
    triage_out = triage_agent.process(input_payload)
    results = []
    for route in triage_out.get("routes", []):
        agent_name = route["agent"]
        payload = route["payload"]
        # attach metadata if present
        payload.setdefault("meta", input_payload.get("metadata", {}))
        print(f"\n[Orchestrator] Routing to '{agent_name}' with text: {payload['text']}")
        agent = agents.get(agent_name)
        if not agent:
            results.append({"agent": agent_name, "error": "unknown agent"})
            continue
        res = agent.process(payload)
        results.append({"agent": agent_name, "result": res})
    return {"triage": triage_out, "results": results}

# ====== Demo / sample run (this executes when the cell runs) ======
if __name__ == "__main__" or True:
    samples = [
        "Buy milk tomorrow at 9am. Also note that the moon is 384,000 km away.",
        "Call Anita and schedule dentist appointment on 2025-12-03 at 14:00. Remember to send recipe.",
        "Idea: Create a 2-week study plan for GATE. Research: MPC design papers and summarize.",
        "Pay electricity bill ASAP. Book flight next Friday at 11am."
    ]
    print("\n===== MindOverflow Cleaner Demo (single-cell) =====")
    for s in samples:
        print("\n" + "-"*72)
        print("INPUT:", s)
        out = orchestrate({"text": s, "metadata": {"source": "single_cell_demo"}})
        print("\nRESULTS SUMMARY:")
        for r in out["results"]:
            agent = r["agent"]
            res = r["result"]
            print(f" - Agent: {agent} -> {res}")
    print("\nLocal storage files written under ./local_storage/: todos.ndjson, notes.ndjson, calendar_events.ndjson")


