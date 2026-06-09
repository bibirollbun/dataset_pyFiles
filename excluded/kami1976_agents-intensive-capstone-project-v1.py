"""
FlowPilot: AI-Powered Task Manager with Multi-Agent Automation
Core skeleton for Agents Intensive – Capstone Project.

Notes:
- No API keys or secrets.
- Tools are implemented as pure Python, easily portable to ADK tools.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import json
import uuid
import datetime as dt
import logging
import os

# -----------------------------------------------------------------------------
# Config & Logging (Observability)
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)

logger = logging.getLogger("flowpilot")

DATA_DIR = "./flowpilot_data"
os.makedirs(DATA_DIR, exist_ok=True)

TASK_DB_PATH = os.path.join(DATA_DIR, "tasks.json")
PROFILE_PATH = os.path.join(DATA_DIR, "user_profile.json")

# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------

@dataclass
class Task:
    id: str
    title: str
    description: str = ""
    priority: str = "medium"  # low / medium / high
    status: str = "todo"      # todo / in_progress / done / dropped
    due_date: Optional[str] = None  # ISO date string
    created_at: str = dt.datetime.utcnow().isoformat()
    updated_at: str = dt.datetime.utcnow().isoformat()
    tags: List[str] = None
    project: Optional[str] = None
    estimated_minutes: Optional[int] = None
    scheduled_slot: Optional[str] = None  # ISO datetime interval string

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["tags"] is None:
            d["tags"] = []
        return d

# -----------------------------------------------------------------------------
# Tools (Pure Python – stand-ins for ADK tools)
# -----------------------------------------------------------------------------

class TaskStoreTool:
    """
    Simple JSON-backed task store.

    This is your long-term memory for tasks. In ADK, this would be exposed as
    a tool with operations like list/create/update/delete.
    """

    def __init__(self, path: str = TASK_DB_PATH):
        self.path = path
        self._ensure_db()

    def _ensure_db(self) -> None:
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump([], f)

    def _load(self) -> List[Dict[str, Any]]:
        with open(self.path, "r") as f:
            return json.load(f)

    def _save(self, raw: List[Dict[str, Any]]) -> None:
        with open(self.path, "w") as f:
            json.dump(raw, f, indent=2)

    def list_tasks(self, status: Optional[str] = None) -> List[Task]:
        raw = self._load()
        tasks = [Task(**item) for item in raw]
        if status:
            tasks = [t for t in tasks if t.status == status]
        return tasks

    def get_task(self, task_id: str) -> Optional[Task]:
        for t in self.list_tasks():
            if t.id == task_id:
                return t
        return None

    def create_task(self, **kwargs) -> Task:
        t = Task(
            id=str(uuid.uuid4()),
            title=kwargs["title"],
            description=kwargs.get("description", ""),
            priority=kwargs.get("priority", "medium"),
            due_date=kwargs.get("due_date"),
            tags=kwargs.get("tags", []),
            project=kwargs.get("project"),
            estimated_minutes=kwargs.get("estimated_minutes"),
        )
        raw = self._load()
        raw.append(t.to_dict())
        self._save(raw)
        logger.info(f"Created task: {t.title} ({t.id})")
        return t

    def update_task(self, task_id: str, **updates) -> Optional[Task]:
        raw = self._load()
        updated = None
        for item in raw:
            if item["id"] == task_id:
                item.update(updates)
                item["updated_at"] = dt.datetime.utcnow().isoformat()
                updated = Task(**item)
                break
        if updated:
            self._save(raw)
            logger.info(f"Updated task: {updated.title} ({updated.id})")
        return updated


class UserProfileTool:
    """
    Simple user profile + preferences store.
    Used to remember working hours, priorities, etc.
    """

    def __init__(self, path: str = PROFILE_PATH):
        self.path = path
        if not os.path.exists(self.path):
            self._save({
                "work_hours": {"start": "09:00", "end": "17:00"},
                "timezone": "UTC",
                "default_priority": "medium",
            })

    def _load(self) -> Dict[str, Any]:
        with open(self.path, "r") as f:
            return json.load(f)

    def _save(self, data: Dict[str, Any]) -> None:
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def get_profile(self) -> Dict[str, Any]:
        return self._load()

    def update_profile(self, **updates) -> Dict[str, Any]:
        data = self._load()
        data.update(updates)
        self._save(data)
        logger.info(f"Updated user profile: {updates}")
        return data


class PlanningTool:
    """
    Very simple planner that assigns tasks to today if they have no schedule.
    In a real system, you'd respect calendar, work hours, etc.
    """

    def create_daily_plan(self, tasks: List[Task], date: Optional[str] = None) -> List[Task]:
        if date is None:
            date = dt.date.today().isoformat()
        planned = []
        for t in tasks:
            if t.scheduled_slot is None and t.status == "todo":
                slot = f"{date}T09:00/{date}T10:00"
                t.scheduled_slot = slot
                planned.append(t)
        return planned


class NotificationTool:
    """
    Simulation of notifications (stdout logs + internal records).
    """

    def send_notification(self, message: str) -> None:
        logger.info(f"[NOTIFICATION] {message}")


# -----------------------------------------------------------------------------
# Agents (simplified; in ADK, these become LLM agents + tools)
# -----------------------------------------------------------------------------

class IntakeAgent:
    """
    Converts natural language into structured Task objects.

    Here we simulate behavior; in the real capstone, this is an LLM agent
    (e.g. Gemini) with a prompt that calls TaskStoreTool.
    """

    def __init__(self, store: TaskStoreTool, profile: UserProfileTool):
        self.store = store
        self.profile = profile

    def parse_and_create_task(self, user_text: str) -> Task:
        # TODO: replace with LLM-based parsing.
        # For now, use heuristic: everything is title, due date unknown.
        logger.info(f"[IntakeAgent] Parsing text: {user_text}")
        task = self.store.create_task(
            title=user_text.strip(),
            description="Created from free-form input.",
            priority=self.profile.get_profile().get("default_priority", "medium"),
        )
        return task


class PlannerAgent:
    """
    Builds a daily plan from tasks using the PlanningTool.
    """

    def __init__(self, store: TaskStoreTool, planner: PlanningTool):
        self.store = store
        self.planner = planner

    def plan_today(self) -> List[Task]:
        tasks = self.store.list_tasks(status="todo")
        new_plan = self.planner.create_daily_plan(tasks)
        for t in new_plan:
            self.store.update_task(t.id, scheduled_slot=t.scheduled_slot)
        logger.info(f"[PlannerAgent] Planned {len(new_plan)} tasks for today.")
        return new_plan


class AutomationAgent:
    """
    Loop-style agent for daily reviews and overdue checks.

    In production, this might run on a cron job or long-running agent runtime.
    """

    def __init__(self, store: TaskStoreTool, notifier: NotificationTool):
        self.store = store
        self.notifier = notifier

    def run_daily_review(self) -> Dict[str, Any]:
        tasks = self.store.list_tasks()
        today = dt.date.today()
        overdue = []
        done = 0

        for t in tasks:
            if t.status == "done":
                done += 1
            elif t.due_date:
                try:
                    due = dt.date.fromisoformat(t.due_date)
                    if due < today and t.status != "done":
                        overdue.append(t)
                except ValueError:
                    pass

        if overdue:
            msg = f"{len(overdue)} tasks are overdue."
            self.notifier.send_notification(msg)

        summary = {
            "total_tasks": len(tasks),
            "completed_tasks": done,
            "overdue_tasks": len(overdue),
        }
        logger.info(f"[AutomationAgent] Daily review summary: {summary}")
        return summary


class OrchestratorAgent:
    """
    Receives high-level commands and routes them to sub-agents.
    """

    def __init__(
        self,
        intake: IntakeAgent,
        planner: PlannerAgent,
        automation: AutomationAgent,
    ):
        self.intake = intake
        self.planner = planner
        self.automation = automation

    def handle_command(self, command: str) -> Dict[str, Any]:
        command = command.strip().lower()

        if command.startswith("add "):
            text = command[len("add "):]
            task = self.intake.parse_and_create_task(text)
            return {"type": "task_created", "task": task.to_dict()}

        if "plan" in command and "today" in command:
            plan = self.planner.plan_today()
            return {"type": "plan_created", "tasks": [t.to_dict() for t in plan]}

        if "daily review" in command:
            summary = self.automation.run_daily_review()
            return {"type": "daily_review", "summary": summary}

        return {"type": "unknown_command", "message": "Command not recognized."}


# -----------------------------------------------------------------------------
# Example usage (for Kaggle Notebook demo)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    store = TaskStoreTool()
    profile = UserProfileTool()
    planner_tool = PlanningTool()
    notifier = NotificationTool()

    intake_agent = IntakeAgent(store, profile)
    planner_agent = PlannerAgent(store, planner_tool)
    automation_agent = AutomationAgent(store, notifier)

    orchestrator = OrchestratorAgent(intake_agent, planner_agent, automation_agent)

    # Simulated session:
    print(orchestrator.handle_command("add finish the Agents Intensive capstone writeup"))
    print(orchestrator.handle_command("add prepare slides for demo"))
    print(orchestrator.handle_command("plan today"))
    print(orchestrator.handle_command("run daily review"))


