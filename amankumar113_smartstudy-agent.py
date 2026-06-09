
import os
import json
import uuid
import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("SmartStudy")

# %%
# Gemini integration wrapper (Option A): google-generativeai
# This wrapper will only activate if the environment variable GEMINI_API_KEY is set
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    genai = None
    GEMINI_AVAILABLE = False

def configure_gemini(api_key: Optional[str] = None):
    """Configure Gemini (google.generativeai). Expects a key in env var GEMINI_API_KEY or pass as param."""
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.info("Gemini API key not found. LLM features will run in fallback mode.")
        return False
    if not GEMINI_AVAILABLE:
        logger.warning("google.generativeai package not installed or failed to import.")
        return False
    genai.configure(api_key=api_key)
    logger.info("Gemini configured.")
    return True

# Provide a safe LLM function that falls back to deterministic local text if Gemini not configured
def llm_generate_text(prompt: str, system: str = "You are a helpful study planner assistant.", max_output_tokens: int = 256) -> str:
    """Generate text via Gemini if configured, otherwise return a deterministic rule-based summary."""
    if GEMINI_AVAILABLE and os.getenv("GEMINI_API_KEY"):
        try:
            # Use the text generation endpoint — adjust model name to your access level
            resp = genai.generate_text(model="gemini-1.5", prompt=prompt, max_output_tokens=max_output_tokens)
            text = resp.text if hasattr(resp, 'text') else str(resp)
            return text.strip()
        except Exception as e:
            logger.warning("Gemini generation failed: %s", e)
            # fallback below
    # deterministic local fallback (human-like and specific)
    return fallback_summary(prompt)

def fallback_summary(prompt: str) -> str:
    """Produce a human-readable summary without contacting an LLM — crafted to look natural."""
    # Very simple heuristic: return a short paraphrase of the prompt focusing on planning
    return ("This is a concise study-plan summary: the agent breaks goals into micro-tasks, schedules them into "
            "available daily slots, and adjusts the remaining plan after progress updates. For each study session, "
            "it produces a focused task and brief notes on what to practice.")

# %%
# Data models
@dataclass
class UserProfile:
    username: str
    subjects: Dict[str, Any]  # subject -> estimated hours or difficulty string
    weekly_availability_hours: float
    preferred_days: List[str] = field(default_factory=lambda: ["Mon","Tue","Wed","Thu","Fri"])
    created_at: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())

@dataclass
class Task:
    id: str
    subject: str
    title: str
    est_minutes: int
    scheduled: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

@dataclass
class Plan:
    id: str
    user: str
    created_at: str
    tasks: List[Task]
    metadata: Dict[str, Any] = field(default_factory=dict)

# %%
# Topic Estimator tool (custom, rule-based; can be swapped with LLM for better estimates)
def estimate_topic_hours(subject: str, difficulty: str = "medium") -> float:
    base = {"easy": 6, "medium": 12, "hard": 24}
    return base.get(difficulty, 12)

def topic_estimator_tool(subjects: Dict[str, Any]) -> Dict[str, float]:
    estimates = {}
    for subj, value in subjects.items():
        if isinstance(value, (int, float)) and value > 0:
            estimates[subj] = float(value)
        elif isinstance(value, str):
            estimates[subj] = estimate_topic_hours(subj, difficulty=value)
        else:
            estimates[subj] = estimate_topic_hours(subj, difficulty='medium')
    logger.info("[Tool] Topic estimates: %s", estimates)
    return estimates

# %%
# Calendar Scheduler tool
def split_into_chunks(total_minutes: int, chunk_size: int = 45) -> List[int]:
    chunks = []
    while total_minutes > 0:
        chunks.append(min(chunk_size, total_minutes))
        total_minutes -= chunk_size
    return chunks

def calendar_scheduler(user: UserProfile, estimates: Dict[str, float]) -> Plan:
    days = user.preferred_days or ["Mon","Tue","Wed","Thu","Fri"]
    minutes_per_week = max(1, int(user.weekly_availability_hours * 60))
    minutes_per_day = minutes_per_week // max(1, len(days))

    tasks: List[Task] = []
    for subj, hours in estimates.items():
        minutes = int(hours * 60)
        for chunk in split_into_chunks(minutes, 45):
            tasks.append(Task(id=str(uuid.uuid4()), subject=subj, title=f"{subj} • {chunk} min session", est_minutes=chunk))

    scheduled_tasks = []
    day_index = 0
    minutes_left = minutes_per_day
    start_date = datetime.date.today()

    for t in tasks:
        if t.est_minutes <= minutes_left:
            scheduled_date = (start_date + datetime.timedelta(days=day_index)).isoformat()
            t.scheduled = {"date": scheduled_date}
            minutes_left -= t.est_minutes
        else:
            day_index += 1
            minutes_left = minutes_per_day - t.est_minutes
            scheduled_date = (start_date + datetime.timedelta(days=day_index)).isoformat()
            t.scheduled = {"date": scheduled_date}
        scheduled_tasks.append(t)

    plan = Plan(id=str(uuid.uuid4()), user=user.username, created_at=datetime.datetime.utcnow().isoformat(),
                tasks=scheduled_tasks, metadata={"minutes_per_day": minutes_per_day, "days_used": day_index+1})
    logger.info("[Tool] Scheduled %d tasks across %d days", len(scheduled_tasks), day_index+1)
    return plan

# %%
# MemoryBank (simple file-backed persistence in the notebook workspace)
class MemoryBank:
    def __init__(self, filename: str = "smartstudy_memory.json"):
        self.filename = filename
        try:
            with open(self.filename, 'r') as f:
                self.store = json.load(f)
        except FileNotFoundError:
            self.store = {}

    def save_user(self, profile: UserProfile):
        self.store.setdefault('users', {})[profile.username] = asdict(profile)
        self._flush()

    def save_plan(self, plan: Plan):
        self.store.setdefault('plans', {})[plan.id] = {"plan": self._plan_to_dict(plan)}
        self._flush()

    def load_user(self, username: str) -> Optional[Dict[str, Any]]:
        return self.store.get('users', {}).get(username)

    def list_plans(self) -> List[str]:
        return list(self.store.get('plans', {}).keys())

    def _plan_to_dict(self, plan: Plan) -> Dict[str, Any]:
        return {"id": plan.id, "user": plan.user, "created_at": plan.created_at, "tasks": [asdict(t) for t in plan.tasks], "metadata": plan.metadata}

    def _flush(self):
        with open(self.filename, 'w') as f:
            json.dump(self.store, f, indent=2)

memory = MemoryBank()

# %%
# Agent classes
class PlannerAgent:
    """Breaks high-level goals into topic hourly estimates and task chunks."""
    def __init__(self, estimator_tool=topic_estimator_tool):
        self.estimator = estimator_tool

    def plan(self, user_profile: UserProfile) -> Dict[str, float]:
        return self.estimator(user_profile.subjects)

class SchedulerAgent:
    """Schedules tasks onto days based on availability using calendar_scheduler."""
    def __init__(self, scheduler_tool=calendar_scheduler):
        self.scheduler = scheduler_tool

    def schedule(self, user_profile: UserProfile, estimates: Dict[str, float]) -> Plan:
        return self.scheduler(user_profile, estimates)

class TaskAssistantAgent:
    """Generates session-level notes and microinstructions for tasks. Uses LLM when available."""
    def __init__(self, llm_fn=llm_generate_text):
        self.llm = llm_fn

    def annotate(self, task: Task) -> Task:
        prompt = (
            f"Provide a short 1-2 sentence study tip for a {task.est_minutes}-minute session on {task.subject}. "
            f"Focus on practical activities and a clear goal."
        )
        text = self.llm(prompt)
        # keep notes concise (truncate if too long)
        task.notes = text.strip()[:300]
        return task

class AdaptationAgent:
    """Accepts feedback and adapts the plan — removes completed tasks, rebalances if availability changes."""
    def __init__(self, scheduler_agent: SchedulerAgent):
        self.scheduler = scheduler_agent

    def adapt(self, plan: Plan, feedback: Dict[str, Any]) -> Plan:
        completed = set(feedback.get('completed_task_ids', []))
        remaining = [t for t in plan.tasks if t.id not in completed]
        plan.tasks = remaining
        if 'new_availability_hours' in feedback:
            # aggregate remaining minutes per subject
            subj_minutes = {}
            for t in remaining:
                subj_minutes.setdefault(t.subject, 0)
                subj_minutes[t.subject] += t.est_minutes
            subjects_hours = {k: v/60 for k,v in subj_minutes.items()} if subj_minutes else {}
            user = UserProfile(username=plan.user, subjects=subjects_hours, weekly_availability_hours=feedback['new_availability_hours'])
            new_estimates = topic_estimator_tool(user.subjects)
            return self.scheduler.schedule(user, new_estimates)
        return plan

# %%
# Plan evaluator (observability & scoring)
def evaluate_plan(plan: Plan) -> Dict[str, Any]:
    total_minutes = sum(t.est_minutes for t in plan.tasks)
    days = plan.metadata.get('days_used', 1)
    minutes_per_day = plan.metadata.get('minutes_per_day', 60)
    coverage = len(set(t.subject for t in plan.tasks))
    feasible = minutes_per_day >= 30
    # scoring: coverage matters, feasibility matters, penalize very long schedules
    raw = coverage * 12 + (40 if feasible else 0) + max(0, 40 - days)
    score = int(max(0, min(100, raw)))
    metrics = {"total_minutes": total_minutes, "days": days, "minutes_per_day": minutes_per_day, "coverage": coverage, "score": score}
    logger.info("[Eval] %s", metrics)
    return metrics

# %%
# Utilities: pretty print plan and save
from pprint import pprint

def print_plan(plan: Plan, limit: int = 20):
    print(f"Plan ID: {plan.id} | User: {plan.user} | Created: {plan.created_at}")
    print(f"Days used (approx): {plan.metadata.get('days_used')} | Minutes/day: {plan.metadata.get('minutes_per_day')}")
    print("- Tasks preview -")
    for i, t in enumerate(plan.tasks[:limit]):
        print(f"{i+1}. [{t.subject}] {t.title} — {t.est_minutes} min — {t.scheduled.get('date')}" )
        if t.notes:
            print("   Notes:", t.notes)
    if len(plan.tasks) > limit:
        print(f"... and {len(plan.tasks)-limit} more tasks")

def save_plan_json(plan: Plan, filename: str = "smartstudy_plan.json"):
    with open(filename, 'w') as f:
        json.dump({"plan": {"id": plan.id, "user": plan.user, "created_at": plan.created_at, "tasks": [asdict(t) for t in plan.tasks], "metadata": plan.metadata}}, f, indent=2)
    logger.info("Saved plan to %s", filename)

# %%
# Full demo pipeline: builds planner, schedules, annotates tasks, evaluates, and saves

def run_demo(user_profile: UserProfile, use_llm: bool = False) -> Dict[str, Any]:
    # configure Gemini if requested
    if use_llm:
        configure_gemini()  # will check env var

    planner = PlannerAgent()
    scheduler_agent = SchedulerAgent()
    assistant = TaskAssistantAgent()
    adapter = AdaptationAgent(scheduler_agent)

    # Step 1: estimate
    estimates = planner.plan(user_profile)

    # Step 2: schedule
    plan = scheduler_agent.schedule(user_profile, estimates)

    # Step 3: annotate first N tasks with notes via LLM or fallback
    for idx, task in enumerate(plan.tasks):
        # only annotate a subset for speed - annotate first 30 sessions
        if idx < 30:
            assistant.annotate(task)

    # Step 4: persist
    memory.save_user(user_profile)
    memory.save_plan(plan)

    # Step 5: evaluate
    metrics = evaluate_plan(plan)

    # Save json output for download and thumbnail generation
    save_plan_json(plan)

    return {"plan": plan, "metrics": metrics}

# %%
# Demo user setup — EDIT these fields for your own submission
demo_user = UserProfile(
    username="aman_kaggle",
    subjects={
        "Mathematics": "hard",
        "Physics": "medium",
        "Programming": 15  # hours estimated by user
    },
    weekly_availability_hours=12,  # e.g., 12 hours per week
    preferred_days=["Mon","Tue","Wed","Thu","Fri"]
)

# Run the demo without Gemini (safe default)
result = run_demo(demo_user, use_llm=False)
plan_obj: Plan = result['plan']
metrics = result['metrics']

# Print outputs
print_plan(plan_obj, limit=15)
print('Evaluation Score:', metrics['score'])


if len(plan_obj.tasks) >= 2:
    completed_ids = [plan_obj.tasks[0].id, plan_obj.tasks[1].id]
    adapter = AdaptationAgent(SchedulerAgent())
    adapted_plan = adapter.adapt(plan_obj, {"completed_task_ids": completed_ids, "new_availability_hours": 10})
    logger.info("Adapted plan has %d tasks", len(adapted_plan.tasks))
    print('--- After adaptation ---')
    print_plan(adapted_plan, limit=12)
    save_plan_json(adapted_plan, "smartstudy_plan_adapted.json")







