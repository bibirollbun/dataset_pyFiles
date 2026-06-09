# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# 1. Imports and basic setup

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

import pandas as pd


# 2. Mock data: calendar, tasks, and notes

calendar_data = [
    {"date": "2025-11-30", "time": "09:00", "title": "DSA Study Session", "duration_min": 120},
    {"date": "2025-11-30", "time": "13:00", "title": "College Project Meeting", "duration_min": 60},
    {"date": "2025-12-01", "time": "10:00", "title": "Aptitude Practice", "duration_min": 90},
]

calendar_df = pd.DataFrame(calendar_data)

tasks_data = [
    {"task": "Watch Durga Sir Java OOPs", "priority": "high", "estimate_min": 60},
    {"task": "Revise automata theory notes", "priority": "medium", "estimate_min": 45},
    {"task": "Practice data structure And algorithm question ", "priority": "high", "estimate_min": 30},
    {"task": "Practice 20 aptitude questions", "priority": "medium", "estimate_min": 40},
]

tasks_df = pd.DataFrame(tasks_data)

notes_data = [
    {"topic": "concierge agents", "content": "Multi-agent personal assistant that coordinates daily tasks using tools."},
    {"topic": "study routine", "content": "Morning: DSA, Afternoon: College work, Evening: Content creation."},
]

notes_df = pd.DataFrame(notes_data)

# 3. Tools (custom Python functions)

def get_calendar_for_date(date_str: str) -> List[Dict[str, Any]]:
    return calendar_df[calendar_df["date"] == date_str].to_dict(orient="records")


def list_tasks(priority: Optional[str] = None) -> List[Dict[str, Any]]:
    df = tasks_df
    if priority:
        df = df[df["priority"] == priority]
    return df.to_dict(orient="records")


def search_notes(keyword: str) -> List[Dict[str, Any]]:
    mask = notes_df["topic"].str.contains(keyword, case=False) | \
           notes_df["content"].str.contains(keyword, case=False)
    return notes_df[mask].to_dict(orient="records")


def plan_focus_block(available_minutes: int) -> List[Dict[str, Any]]:
    remaining = available_minutes
    plan = []
    for priority in ["high", "medium"]:
        for _, row in tasks_df[tasks_df["priority"] == priority].iterrows():
            if row["estimate_min"] <= remaining:
                plan.append(row.to_dict())
                remaining -= row["estimate_min"]
    return plan


# Simple context compaction: truncate long note text
def compact_text(text: str, max_len: int = 160) -> str:
    return text if len(text) <= max_len else text[:max_len] + "..."

# 4. Session & logging (memory + observability)

@dataclass
class SessionState:
    user_id: str = "default_user"
    default_focus_minutes: int = 90
    last_plan: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = field(default_factory=list)


session_state = SessionState()
logs: List[Dict[str, Any]] = []  # for observability


def log_event(user_request: str, agent_name: str, tools_used: List[str], extra: Dict[str, Any] = None):
    entry = {
        "user_request": user_request,
        "agent": agent_name,
        "tools_used": tools_used,
        "extra": extra or {},
    }
    logs.append(entry)


# 5. AgentResponse & agents

@dataclass
class AgentResponse:
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


class SchedulingAgent:
    def summarize_day(self, date_str: str) -> AgentResponse:
        events = get_calendar_for_date(date_str)
        log_event(f"schedule {date_str}", "SchedulingAgent", ["get_calendar_for_date"], {"events_found": len(events)})

        if not events:
            return AgentResponse(
                message=f"No events found for {date_str}. You have a free day!",
                details={"events": []},
            )

        lines = [f"{e['time']} – {e['title']} ({e['duration_min']} min)" for e in events]
        msg = "Your schedule for " + date_str + ":\n" + "\n".join(lines)
        return AgentResponse(message=msg, details={"events": events})


class TaskAgent:
    def list_high_priority(self) -> AgentResponse:
        high_tasks = list_tasks(priority="high")
        log_event("list high priority tasks", "TaskAgent", ["list_tasks"], {"tasks_found": len(high_tasks)})

        if not high_tasks:
            return AgentResponse(message="No high-priority tasks found.", details={"tasks": []})

        lines = [f"- {t['task']} ({t['estimate_min']} min)" for t in high_tasks]
        msg = "High-priority tasks:\n" + "\n".join(lines)
        return AgentResponse(message=msg, details={"tasks": high_tasks})

    def create_focus_plan(self, available_minutes: int) -> AgentResponse:
        plan = plan_focus_block(available_minutes)
        log_event(f"focus plan {available_minutes}", "TaskAgent", ["plan_focus_block"], {"tasks_in_plan": len(plan)})

        if not plan:
            return AgentResponse(
                message=f"No tasks fit into a {available_minutes}-minute block.",
                details={"plan": []},
            )

        lines = [f"- {p['task']} ({p['estimate_min']} min)" for p in plan]
        msg = f"Focus plan for the next {available_minutes} minutes:\n" + "\n".join(lines)

        # store in session memory
        session_state.last_plan = {
            "minutes": available_minutes,
            "items": plan,
        }

        return AgentResponse(message=msg, details={"plan": plan})


class NotesAgent:
    def summarize_topic(self, topic: str) -> AgentResponse:
        results = search_notes(topic)
        log_event(f"notes {topic}", "NotesAgent", ["search_notes"], {"notes_found": len(results)})

        if not results:
            return AgentResponse(
                message=f"No notes found for topic '{topic}'.",
                details={"notes": []},
            )

        contents = [compact_text(n["content"]) for n in results]
        summary = " | ".join(contents)
        msg = f"Notes on '{topic}': {summary}"
        return AgentResponse(message=msg, details={"notes": results})


class PlannerAgent:
    """
    Combines schedule + tasks to propose a simple day plan.
    Sequential multi-agent pattern: Concierge -> SchedulingAgent + TaskAgent -> PlannerAgent.
    """

    def create_day_plan(self, date_str: str) -> AgentResponse:
        events = get_calendar_for_date(date_str)
        all_tasks = list_tasks()

        log_event(f"day plan {date_str}", "PlannerAgent",
                  ["get_calendar_for_date", "list_tasks"],
                  {"events": len(events), "tasks": len(all_tasks)})

        if not events:
            msg = f"Day plan for {date_str}:\nNo fixed events. Use a focus block to work on high priority tasks."
            return AgentResponse(message=msg, details={"events": [], "slots": [], "plan": []})

        # simple: after last event, create a focus slot
        last_event = max(events, key=lambda e: e["time"])
        focus_start = "15:00"
        focus_minutes = session_state.default_focus_minutes

        focus_plan = plan_focus_block(focus_minutes)
        lines = [
            f"Fixed events on {date_str}:",
            *[f"- {e['time']} – {e['title']} ({e['duration_min']} min)" for e in events],
            "",
            f"Suggested focus block starting {focus_start} ({focus_minutes} min):",
            *[f"- {p['task']} ({p['estimate_min']} min)" for p in focus_plan],
        ]
        msg = "\n".join(lines)

        session_state.last_plan = {
            "date": date_str,
            "focus_start": focus_start,
            "minutes": focus_minutes,
            "items": focus_plan,
        }

        return AgentResponse(
            message=msg,
            details={"events": events, "focus_plan": focus_plan},
        )


# 6. Concierge Agent (orchestrator with session & routing)

class ConciergeAgent:
    def __init__(self):
        self.scheduling_agent = SchedulingAgent()
        self.task_agent = TaskAgent()
        self.notes_agent = NotesAgent()
        self.planner_agent = PlannerAgent()

    def handle_request(self, request: str) -> AgentResponse:
        text = request.lower()

        # update simple conversational history in session
        session_state.history.append({"user": request})

        if "plan my day" in text:
            date = self._extract_date(text)
            resp = self.planner_agent.create_day_plan(date)
            session_state.history.append({"agent": resp.message})
            return resp

        if "schedule" in text or "today" in text or "tomorrow" in text:
            date = self._extract_date(text)
            resp = self.scheduling_agent.summarize_day(date)
            session_state.history.append({"agent": resp.message})
            return resp

        if "high priority" in text or "important tasks" in text:
            resp = self.task_agent.list_high_priority()
            session_state.history.append({"agent": resp.message})
            return resp

        if "focus plan" in text or "study plan" in text:
            minutes = self._extract_minutes(text) or session_state.default_focus_minutes
            resp = self.task_agent.create_focus_plan(minutes)
            session_state.history.append({"agent": resp.message})
            return resp

        if "last plan" in text:
            if session_state.last_plan is None:
                msg = "There is no previous plan stored in this session."
                return AgentResponse(message=msg, details={})
            plan = session_state.last_plan
            lines = [f"Last plan ({plan.get('minutes', '?')} minutes):"]
            for item in plan.get("items", []):
                lines.append(f"- {item['task']} ({item['estimate_min']} min)")
            msg = "\n".join(lines)
            return AgentResponse(message=msg, details={"plan": plan})

        if "notes" in text or "summarize" in text:
            topic = self._extract_topic(text)
            resp = self.notes_agent.summarize_topic(topic)
            session_state.history.append({"agent": resp.message})
            return resp

        if "set default focus" in text:
            minutes = self._extract_minutes(text)
            if minutes:
                session_state.default_focus_minutes = minutes
                msg = f"Default focus block updated to {minutes} minutes."
            else:
                msg = "Please specify the number of minutes, e.g., 'Set default focus to 120 minutes'."
            return AgentResponse(message=msg, details={"default_focus_minutes": session_state.default_focus_minutes})

        # fallback
        msg = (
            "I can help with:\n"
            "- Show my schedule for today/tomorrow\n"
            "- Show my high priority tasks\n"
            "- Create a 120 minute study focus plan\n"
            "- Plan my day for tomorrow\n"
            "- Summarize my notes about concierge agents\n"
            "- Show my last plan\n"
            "- Set default focus to 120 minutes"
        )
        return AgentResponse(message=msg, details={})

    def _extract_date(self, text: str) -> str:
        today = datetime(2025, 11, 30)
        if "tomorrow" in text:
            return (today + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            return today.strftime("%Y-%m-%d")

    def _extract_minutes(self, text: str) -> Optional[int]:
        for tok in text.split():
            if tok.isdigit():
                return int(tok)
        return None

    def _extract_topic(self, text: str) -> str:
        if "concierge" in text:
            return "concierge agents"
        if "study" in text:
            return "study routine"
        return "concierge agents"


# 7. Demo: End-to-end examples

concierge = ConciergeAgent()

demo_requests = [
    "Show my schedule for today",
    "Show my schedule for tomorrow",
    "Show my high priority tasks",
    "Create a 120 minute study focus plan",
    "Plan my day for tomorrow",
    "Summarize my notes about concierge agents",
    "Show my last plan",
]

for req in demo_requests:
    resp = concierge.handle_request(req)
    print("=" * 70)
    print("User:", req)
    print("Concierge:\n", resp.message)

# Show simple logs and session summary
print("\n\n==== Logs (Observability) ====")
for entry in logs:
    print(entry)

print("\n==== Session State ====")
print(session_state)


