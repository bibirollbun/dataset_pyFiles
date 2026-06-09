# SmartLife Planner - Single-Cell Kaggle Notebook (MOCK LLM)
# Track: Concierge Agents
import json
import re
import logging
import uuid
from datetime import datetime, timedelta, time
from dateutil.parser import parse as parse_dt
from typing import List, Dict, Any, Optional
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("SmartLifePlanner")

# -------------------------
# Memory services
# -------------------------
class InMemorySessionService:
    def __init__(self):
        self.sessions = {}  # session_id -> dict

    def create_session(self, user_id: str) -> str:
        sid = str(uuid.uuid4())
        self.sessions[sid] = {
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "tasks": [],
            "preferences": {},
            "history": []
        }
        logger.info(f"Session created: {sid} for user {user_id}")
        return sid

    def get(self, session_id: str) -> dict:
        return self.sessions.get(session_id, {})

    def update(self, session_id: str, key: str, value):
        if session_id not in self.sessions:
            raise KeyError("session not found")
        self.sessions[session_id][key] = value

    def append_task(self, session_id: str, task):
        self.sessions[session_id]["tasks"].append(task)
        self.sessions[session_id]["history"].append({"time": datetime.utcnow().isoformat(), "event": f"task_added: {task.get('title')}"})

class MemoryBank:
    """Simple long-term memory for user preferences and learned heuristics."""
    def __init__(self, filepath: Optional[str] = None):
        self.store = {}
        self.filepath = filepath

    def set(self, key: str, value):
        self.store[key] = value
        if self.filepath:
            with open(self.filepath, "w") as f:
                json.dump(self.store, f, indent=2)

    def get(self, key: str, default=None):
        return self.store.get(key, default)

    def load(self):
        if self.filepath:
            try:
                with open(self.filepath, "r") as f:
                    self.store = json.load(f)
            except FileNotFoundError:
                self.store = {}

# -------------------------
# Tools (stubs)
# -------------------------
class SearchTool:
    """A stubbed search tool. Returns canned suggestions."""
    def search(self, query: str, k: int = 3) -> List[str]:
        logger.info(f"SearchTool: searching for '{query}' (stub)")
        lower_q = query.lower()
        if "study interval" in lower_q or "pomodoro" in lower_q:
            return [
                "Pomodoro technique: 25 min focus, 5 min break.",
                "Long study blocks: 50-90 minutes for deep work.",
                "Active recall and spaced repetition recommended for study."
            ][:k]
        if "productivity" in lower_q:
            return [
                "Morning energy peaks are common (9-11AM).",
                "Schedule hardest tasks in peak cognitive hours.",
                "Break tasks into 25-90 minute focused blocks."
            ][:k]
        return [
            f"Search result for '{query}' - idea 1",
            f"Search result for '{query}' - idea 2",
            f"Search result for '{query}' - idea 3",
        ][:k]

class CodeExecutionTool:
    """Executes small Python expressions with sandboxing. Use only for safe computations."""
    def run(self, code_str: str) -> dict:
        logger.info("CodeExecutionTool: executing code snippet (safe mode)")
        safe_locals = {}
        try:
            result = eval(code_str, {"__builtins__": {}}, safe_locals)
            return {"ok": True, "result": result}
        except Exception as e:
            logger.exception("Code execution failed")
            return {"ok": False, "error": str(e)}

# -------------------------
# Mock LLM
# -------------------------
USE_LLM = False  # keep False for mock/demo mode

class LLMClient:
    def __init__(self, provider="mock", api_key=None):
        self.provider = provider
        self.api_key = api_key

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        logger.info(f"LLMClient.generate called (provider={self.provider})")
        if not USE_LLM:
            return self._mock_response(prompt)
        # If you want to enable a real LLM, implement API calls here and set USE_LLM=True (not done in this demo)
        raise NotImplementedError("LLM integration not configured. Set USE_LLM=False for offline demo or implement API calls.")

    def _mock_response(self, prompt: str) -> str:
        low = prompt.lower()
        if "create a daily study plan" in low or "break down" in low or "daily study plan" in low:
            return "Divide the goal into daily sessions. Start with fundamentals, then exercises. Use 4 sessions of 45 minutes each for 4 days, then a mock test."
        if "prioritise" in low or "prioritize" in low or "schedule" in low:
            return "Prioritise by deadline then estimated effort. Put highest priority tasks in the user's peak hours."
        return "Understood. (mock LLM reply) " + prompt[:120]

# -------------------------
# Agent base classes and agents
# -------------------------
class AgentResult:
    def __init__(self, success: bool, output: Any, metadata: Optional[dict] = None):
        self.success = success
        self.output = output
        self.metadata = metadata or {}

class Agent:
    def __init__(self, llm: LLMClient = None, tools: dict = None, memory: InMemorySessionService = None, long_term: MemoryBank = None):
        self.llm = llm or LLMClient()
        self.tools = tools or {}
        self.memory = memory
        self.long_term = long_term

    def run(self, input_text: str, session_id: str) -> AgentResult:
        raise NotImplementedError

class IntakeAgent(Agent):
    TASK_REGEX = re.compile(r"(?P<title>.+?)(?: by| due| on| - |: )(?P<deadline>.+)", re.IGNORECASE)

    def run(self, input_text: str, session_id: str) -> AgentResult:
        logger.info("IntakeAgent.run")
        lines = [l.strip() for l in re.split(r"\n|;|\.", input_text) if l.strip()]
        tasks = []
        for line in lines:
            # detect 'due ...' first
            m = re.search(r"due (on )?(?P<date>[\w\s\d,]+)", line, re.I)
            if m:
                deadline_str = m.group("date")
                try:
                    dt = parse_dt(deadline_str, fuzzy=True)
                    title = line[:m.start()].strip() or "Task"
                    tasks.append({"title": title, "deadline": dt.isoformat(), "est_minutes": 60, "priority": "medium"})
                    continue
                except Exception:
                    pass
            # detect 'on ...'
            m2 = re.search(r"on (?P<date>[\w\s\d]+)", line, re.I)
            if m2:
                try:
                    dt = parse_dt(m2.group("date"), fuzzy=True)
                    title = line[:m2.start()].strip() or "Task"
                    tasks.append({"title": title, "deadline": dt.isoformat(), "est_minutes": 90, "priority": "high"})
                    continue
                except Exception:
                    pass
            # fallback to LLM mock to parse
            parse_prompt = f"Parse task: '{line}' -> return title, deadline, est_minutes, priority"
            parsed = self.llm.generate(parse_prompt)
            tasks.append({"title": line, "deadline": None, "est_minutes": 60, "priority": "low", "note_parsed": parsed[:120]})
        # append to session memory
        for t in tasks:
            self.memory.append_task(session_id, t)
        return AgentResult(True, tasks)

class StudyAgent(Agent):
    def run(self, input_text: str, session_id: str) -> AgentResult:
        logger.info("StudyAgent.run")
        session = self.memory.get(session_id)
        tasks = session.get("tasks", [])
        study_plan = []
        for t in tasks:
            if any(word in t["title"].lower() for word in ["learn", "study", "exam", "prepare"]):
                prompt = f"Create a daily study plan for: {t['title']} with deadline {t.get('deadline')}"
                reply = self.llm.generate(prompt)
                # days left heuristic
                try:
                    if t.get("deadline"):
                        days_left = max(1, (parse_dt(t["deadline"]) - datetime.utcnow()).days)
                    else:
                        days_left = 7
                except Exception:
                    days_left = 7
                per_day_minutes = min(240, max(30, t.get("est_minutes", 60)))
                daily = []
                for i in range(days_left):
                    day = {
                        "day_index": i+1,
                        "date": (datetime.utcnow() + timedelta(days=i)).date().isoformat(),
                        "focus_minutes": per_day_minutes
                    }
                    daily.append(day)
                study_plan.append({"task": t["title"], "daily": daily, "llm_note": reply[:200]})
        if study_plan and self.long_term:
            self.long_term.set(f"study_plan_{session_id}", study_plan)
        return AgentResult(True, study_plan)

class PlannerAgent(Agent):
    def run(self, input_text: str, session_id: str) -> AgentResult:
        logger.info("PlannerAgent.run")
        session = self.memory.get(session_id)
        tasks = session.get("tasks", [])
        now = datetime.utcnow()
        day_start = datetime.combine(now.date(), time(9,0))
        cursor = day_start
        end_of_day = datetime.combine(now.date(), time(22,0))
        sorted_tasks = sorted(tasks, key=lambda t: {"high":0,"medium":1,"low":2}.get(t.get("priority","medium"),1))
        schedule = []
        for t in sorted_tasks:
            est = int(t.get("est_minutes", 60))
            slot_start = cursor
            slot_end = slot_start + timedelta(minutes=est)
            if slot_end > end_of_day:
                # move to next day start
                cursor = day_start + timedelta(days=1)
                slot_start = cursor
                slot_end = slot_start + timedelta(minutes=est)
                cursor = slot_end + timedelta(minutes=10)
            else:
                cursor = slot_end + timedelta(minutes=10)
            schedule.append({
                "title": t["title"],
                "start": slot_start.isoformat(),
                "end": slot_end.isoformat(),
                "est_minutes": est,
                "priority": t.get("priority","medium")
            })
        # store schedule
        self.memory.update(session_id, "today_schedule", schedule)
        return AgentResult(True, schedule)

class NotifierAgent(Agent):
    """Improved Notifier: shows upcoming tasks in next 0-60 minutes and returns local-time summary (IST default)."""
    def __init__(self, *args, local_tz_offset_hours=5, local_tz_offset_mins=30, **kwargs):
        super().__init__(*args, **kwargs)
        self.tz_hours = local_tz_offset_hours
        self.tz_mins = local_tz_offset_mins

    def _utc_to_local(self, dt_utc: datetime) -> datetime:
        return dt_utc + timedelta(hours=self.tz_hours, minutes=self.tz_mins)

    def run(self, input_text: str, session_id: str) -> AgentResult:
        logger.info("NotifierAgent.run (improved)")
        session = self.memory.get(session_id)
        schedule = session.get("today_schedule", [])
        summary_lines = []
        now = datetime.utcnow()
        for s in schedule:
            try:
                start = parse_dt(s["start"])
            except Exception:
                continue
            delta = start - now
            minutes_left = int(delta.total_seconds() // 60)
            # Only show events that are upcoming in the next hour
            if 0 <= minutes_left <= 60:
                local_start = self._utc_to_local(start)
                summary_lines.append(f"Upcoming within hour: {s['title']} at {local_start.time().isoformat()} (in {minutes_left} mins)")
        if not summary_lines:
            summary_lines.append("No urgent tasks in the next hour.")
        return AgentResult(True, "\n".join(summary_lines))

# -------------------------
# Pipeline orchestration
# -------------------------
class SequentialPipeline:
    def __init__(self, agents: List[Agent]):
        self.agents = agents

    def run(self, input_text: str, session_id: str) -> dict:
        data = input_text
        outputs = {}
        for a in self.agents:
            res = a.run(data, session_id)
            outputs[type(a).__name__] = res.output
        return outputs

class ParallelRunner:
    def __init__(self, agents: List[Agent]):
        self.agents = agents

    def run(self, input_text: str, session_id: str) -> dict:
        results = {}
        with ThreadPoolExecutor(max_workers=len(self.agents)) as ex:
            futures = {ex.submit(a.run, input_text, session_id): a for a in self.agents}
            for fut in futures:
                a = futures[fut]
                try:
                    res = fut.result()
                    results[type(a).__name__] = res.output
                except Exception as e:
                    results[type(a).__name__] = {"error": str(e)}
        return results

# -------------------------
# Helpers for display (IST)
# -------------------------
def utciso_to_local_str(utc_iso: str, hours=5, mins=30) -> str:
    try:
        dt = parse_dt(utc_iso)
        local = dt + timedelta(hours=hours, minutes=mins)
        return local.strftime('%Y-%m-%d %H:%M')
    except Exception:
        return utc_iso

def pretty_schedule_table(session_id: str, session_service: InMemorySessionService = None):
    if session_service is None:
        print("No session service provided")
        return
    sched = session_service.get(session_id).get('today_schedule', [])
    rows = []
    for s in sched:
        start_local = utciso_to_local_str(s['start'])
        end_local = utciso_to_local_str(s['end'])
        duration = s.get('est_minutes', None)
        rows.append({
            'title': s['title'],
            'start_local': start_local,
            'end_local': end_local,
            'duration_min': duration,
            'priority': s.get('priority')
        })
    df = pd.DataFrame(rows)
    if df.empty:
        print('No scheduled items')
    else:
        display(df)

# -------------------------
# Demo run (create session, run pipeline)
# -------------------------
def demo_run(user_input: str):
    # Setup memory and tools
    session_service = InMemorySessionService()
    memory_bank = MemoryBank()
    tools = {"search": SearchTool(), "code_exec": CodeExecutionTool()}
    llm = LLMClient()

    # create session
    session_id = session_service.create_session(user_id="ravi_capstone")

    # instantiate agents
    intake = IntakeAgent(llm=llm, tools=tools, memory=session_service, long_term=memory_bank)
    study = StudyAgent(llm=llm, tools=tools, memory=session_service, long_term=memory_bank)
    planner = PlannerAgent(llm=llm, tools=tools, memory=session_service, long_term=memory_bank)
    notifier = NotifierAgent(llm=llm, tools=tools, memory=session_service, long_term=memory_bank)

    # Run sequential: Intake -> (Study + Planner) in parallel -> Notifier
    seq1 = SequentialPipeline([intake])
    out1 = seq1.run(user_input, session_id)
    print("=== Intake output ===")
    print(json.dumps(out1, indent=2, default=str))

    par = ParallelRunner([study, planner])
    out_par = par.run(user_input, session_id)
    print("\n=== Parallel outputs (StudyAgent + PlannerAgent) ===")
    print(json.dumps(out_par, indent=2, default=str))

    noti = notifier.run(user_input, session_id)
    print("\n=== Notifier summary (local IST) ===")
    print(noti.output)

    # Show pretty schedule with local times
    print("\n=== Schedule table (local IST) ===")
    pretty_schedule_table(session_id, session_service)

    return {
        "session_service": session_service,
        "session_id": session_id,
        "intake": out1,
        "parallel": out_par,
        "notifier": noti
    }

# -------------------------
# Dynamic Evaluation Engine (replaces simple evaluation)
# -------------------------

def evaluate_pipeline():
    print("\n\n================ DYNAMIC EVALUATION ================")

    # Run a controlled test input
    test_input = "Prepare for math exam due next Monday. Finish assignment due tomorrow."
    r = demo_run(test_input)

    intake = r["intake"].get("IntakeAgent", [])
    study = r["parallel"].get("StudyAgent", [])
    planner = r["parallel"].get("PlannerAgent", [])
    notifier_text = r["notifier"].output

    total_score = 0
    breakdown = {}

    # ----------------------------------------------------
    # 1️⃣ Intake Agent Evaluation (0–30)
    # ----------------------------------------------------
    intake_score = 0

    if isinstance(intake, list) and len(intake) >= 2:
        intake_score += 10  # detected tasks

    deadlines_found = sum(1 for t in intake if t.get("deadline"))
    if deadlines_found >= 1:
        intake_score += 10

    titles_good = sum(1 for t in intake if len(t.get("title", "")) > 3)
    if titles_good >= 1:
        intake_score += 10

    breakdown["Intake Accuracy"] = intake_score
    total_score += intake_score

    # ----------------------------------------------------
    # 2️⃣ Study Plan Evaluation (0–25)
    # ----------------------------------------------------
    study_score = 0

    if isinstance(study, list) and len(study) > 0:
        study_score += 10

    # Check daily plan length
    long_plans = [s for s in study if len(s.get("daily", [])) >= 3]
    if long_plans:
        study_score += 10

    # Check daily structure
    good_daily = 0
    for s in study:
        for day in s.get("daily", []):
            if "focus_minutes" in day:
                good_daily += 1

    if good_daily > 3:
        study_score += 5

    breakdown["Study Plan Quality"] = study_score
    total_score += study_score

    # ----------------------------------------------------
    # 3️⃣ PlannerAgent Scheduling Evaluation (0–25)
    # ----------------------------------------------------
    planner_score = 0

    if isinstance(planner, list) and len(planner) >= 2:
        planner_score += 10

    # Check chronological order (no backward time)
    chrono_ok = True
    prev_end = None
    for item in planner:
        try:
            st = parse_dt(item["start"])
            en = parse_dt(item["end"])
            if prev_end and st < prev_end:
                chrono_ok = False
            prev_end = en
        except:
            chrono_ok = False

    if chrono_ok:
        planner_score += 10

    # Check reasonable durations
    dur_ok = sum(1 for p in planner if 10 <= p.get("est_minutes", 0) <= 300)
    if dur_ok >= 1:
        planner_score += 5

    breakdown["Schedule Planning"] = planner_score
    total_score += planner_score

    # ----------------------------------------------------
    # 4️⃣ Notifier Evaluation (0–20)
    # ----------------------------------------------------
    notifier_score = 0

    if isinstance(notifier_text, str):
        notifier_score += 5

    # Reward detection of "Upcoming" alerts
    if "Upcoming" in notifier_text:
        notifier_score += 10

    # Reward clean summary
    if len(notifier_text.split("\n")) <= 5:
        notifier_score += 5

    breakdown["Notifier Quality"] = notifier_score
    total_score += notifier_score

    # ----------------------------------------------------
    # Final report
    # ----------------------------------------------------
    print("\n========= SCORE BREAKDOWN =========")
    # Print with expected maxima
    print(f"{'Intake Accuracy':25s}: {breakdown.get('Intake Accuracy',0)}/30")
    print(f"{'Study Plan Quality':25s}: {breakdown.get('Study Plan Quality',0)}/25")
    print(f"{'Schedule Planning':25s}: {breakdown.get('Schedule Planning',0)}/25")
    print(f"{'Notifier Quality':25s}: {breakdown.get('Notifier Quality',0)}/20")

    print("\nFINAL SCORE:", total_score, "/ 100")
    print("====================================\n")

    return total_score

# -------------------------
# Run the demo and evaluation
# -------------------------
USER_INPUT = """
I have an exam on Sunday and a project due Friday. Also, learn Python for interviews in two weeks.
Finish the project by Friday; study for exam on Sunday morning.
"""
results = demo_run(USER_INPUT)
eval_score = evaluate_pipeline()

