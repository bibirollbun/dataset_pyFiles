!pip install google-adk google-genai




from __future__ import annotations

import os
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, Any, List

from google.adk.agents.llm_agent import Agent  # ADK Agent class :contentReference[oaicite:1]{index=1}


# ======================================================
# JSON MEMORY STORE
# ======================================================

class JSONMemoryStore:
    """
    Very simple JSON file store for StudyFlow Agent.
    Keeps notes, assignments, study_sessions and study_plans.
    """

    def __init__(self, path: str = "studyflow_memory.json"):
        self.path = Path(path)
        self.data: Dict[str, Any] = {
            "notes": [],
            "assignments": [],
            "study_sessions": [],
            "study_plans": [],
        }
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception as e:
                print("âš ï¸� Failed to load memory, starting fresh:", e)
                self._save()
        else:
            self._save()

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def add_item(self, key: str, item: Dict[str, Any]) -> None:
        self.data.setdefault(key, [])
        self.data[key].append(item)
        self._save()

    def get_items(self, key: str) -> List[Dict[str, Any]]:
        return list(self.data.get(key, []))

    def save_items(self, key: str, items: List[Dict[str, Any]]) -> None:
        self.data[key] = items
        self._save()


# Single shared memory instance used by all tools
memory = JSONMemoryStore("studyflow_memory.json")


# ======================================================
# DATA MODELS (optional but nice)
# ======================================================

@dataclass
class Note:
    id: str
    subject: str
    title: str
    content: str
    created_at: str
    updated_at: str


@dataclass
class Assignment:
    id: str
    subject: str
    title: str
    description: str
    due_date: str  # YYYY-MM-DD
    status: str
    priority: int
    created_at: str
    updated_at: str


# ======================================================
# STUDYFLOW TOOLS (plain Python functions)
# ======================================================

def add_note(subject: str, title: str, content: str) -> Dict[str, Any]:
    """
    Tool: Add a study note.
    """
    now = datetime.utcnow().isoformat()
    note = Note(
        id=str(uuid.uuid4()),
        subject=subject.strip(),
        title=title.strip(),
        content=content.strip(),
        created_at=now,
        updated_at=now,
    )
    memory.add_item("notes", asdict(note))
    return {"status": "ok", "note": asdict(note)}


def search_notes(query: str, subject: str | None = None) -> Dict[str, Any]:
    """
    Tool: Search notes by keyword, optionally filtering by subject.
    """
    q = query.lower()
    notes = memory.get_items("notes")

    def matches(n: Dict[str, Any]) -> bool:
        if subject and n["subject"].lower() != subject.lower():
            return False
        return q in (n["title"] + " " + n["content"]).lower()

    results = [n for n in notes if matches(n)]
    return {"status": "ok", "count": len(results), "results": results}


def add_assignment(
    subject: str,
    title: str,
    description: str,
    due_date: str,
    priority: int = 3,
) -> Dict[str, Any]:
    """
    Tool: Add a new assignment with due date.
    due_date should be in YYYY-MM-DD format.
    """
    now = datetime.utcnow().isoformat()
    assignment = Assignment(
        id=str(uuid.uuid4()),
        subject=subject.strip(),
        title=title.strip(),
        description=description.strip(),
        due_date=due_date,
        status="pending",
        priority=int(priority),
        created_at=now,
        updated_at=now,
    )
    memory.add_item("assignments", asdict(assignment))
    return {"status": "ok", "assignment": asdict(assignment)}


def upcoming_assignments(within_days: int = 7) -> Dict[str, Any]:
    """
    Tool: Get assignments due within the next N days.
    """
    today = date.today()
    cutoff = today + timedelta(days=within_days)
    upcoming: List[Dict[str, Any]] = []

    for a in memory.get_items("assignments"):
        try:
            due = datetime.strptime(a["due_date"], "%Y-%m-%d").date()
        except ValueError:
            continue
        if today <= due <= cutoff and a["status"] != "completed":
            upcoming.append(a)

    upcoming.sort(key=lambda x: (x["due_date"], x["priority"]))
    return {"status": "ok", "upcoming": upcoming}


# ======================================================
# ROOT AGENT (ADK)
# ======================================================

ROOT_INSTRUCTION = """
You are StudyFlow Agent, an AI-powered personal study assistant.

You help students by:
- Creating and searching study notes
- Tracking assignments and due dates
- Highlighting upcoming deadlines

When helpful, call the available tools to:
- Store data (notes, assignments)
- Fetch or filter data (search_notes, upcoming_assignments)

Always explain what you did and summarize the results for the user.
"""

root_agent = Agent(
    model=os.getenv("GEN_MODEL_NAME", "gemini-2.5-flash"),  # Gemini model :contentReference[oaicite:2]{index=2}
    name="studyflow_root_agent",
    description="StudyFlow Agent â€“ manages notes and assignments for students.",
    instruction=ROOT_INSTRUCTION,
    tools=[
        add_note,
        search_notes,
        add_assignment,
        upcoming_assignments,
    ],
)



# -------------------------------
# NOTES TOOLS
# -------------------------------

def add_note(subject: str, title: str, content: str, tags: list[str] | None = None) -> dict:
    """
    Add a study note.
    - subject: Subject name (e.g. 'OOPS', 'Data Communication')
    - title: Short title of the note
    - content: Full text of the note
    - tags: Optional list of tags/keywords
    """
    now = datetime.utcnow().isoformat()
    note = {
        "id": str(uuid.uuid4()),
        "subject": subject.strip(),
        "title": title.strip(),
        "content": content.strip(),
        "tags": [t.strip().lower() for t in (tags or [])],
        "created_at": now,
        "updated_at": now,
    }
    memory.add_item("notes", note)
    return {"status": "ok", "note": note}


def search_notes(query: str, subject: str | None = None) -> dict:
    """
    Search notes by keyword and optional subject filter.
    """
    q = query.lower()
    notes = memory.get_items("notes")

    def matches(n):
        if subject and n["subject"].lower() != subject.lower():
            return False
        return q in (n["title"] + " " + n["content"]).lower()

    results = [n for n in notes if matches(n)]
    return {"status": "ok", "count": len(results), "results": results}


# -------------------------------
# ASSIGNMENTS TOOLS
# -------------------------------

def add_assignment(subject: str, title: str, description: str, due_date: str, priority: int = 3) -> dict:
    """
    Add a new assignment.
    - due_date format: 'YYYY-MM-DD'
    - priority: 1 (high) to 5 (low)
    """
    now = datetime.utcnow().isoformat()
    assignment = {
        "id": str(uuid.uuid4()),
        "subject": subject.strip(),
        "title": title.strip(),
        "description": description.strip(),
        "due_date": due_date,
        "status": "pending",
        "priority": int(priority),
        "created_at": now,
        "updated_at": now,
    }
    memory.add_item("assignments", assignment)
    return {"status": "ok", "assignment": assignment}


def list_assignments(include_completed: bool = True) -> dict:
    """
    List assignments, optionally excluding completed ones.
    """
    items = memory.get_items("assignments")
    if not include_completed:
        items = [a for a in items if a["status"] != "completed"]

    items = sorted(items, key=lambda a: (a["due_date"], a["priority"]))
    return {"status": "ok", "assignments": items}


def upcoming_assignments(within_days: int = 7) -> dict:
    """
    Get assignments due within the next N days (default: 7).
    """
    today = date.today()
    cutoff = today + timedelta(days=within_days)
    upcoming = []

    for a in memory.get_items("assignments"):
        try:
            due = datetime.strptime(a["due_date"], "%Y-%m-%d").date()
        except ValueError:
            continue
        if today <= due <= cutoff and a["status"] != "completed":
            upcoming.append(a)

    upcoming = sorted(upcoming, key=lambda a: (a["due_date"], a["priority"]))
    return {"status": "ok", "upcoming": upcoming}


def update_assignment_status(assignment_id: str, status: str) -> dict:
    """
    Update assignment status.
    Allowed statuses: 'pending', 'in_progress', 'completed'
    """
    status = status.lower()
    if status not in {"pending", "in_progress", "completed"}:
        return {"status": "error", "message": "Invalid status"}

    assignments = memory.get_items("assignments")
    updated = None
    for a in assignments:
        if a["id"] == assignment_id:
            a["status"] = status
            a["updated_at"] = datetime.utcnow().isoformat()
            updated = a
            break

    memory.save_items("assignments", assignments)
    if not updated:
        return {"status": "error", "message": "Assignment not found"}

    return {"status": "ok", "assignment": updated}


# -------------------------------
# STUDY PLAN TOOLS
# -------------------------------

def create_study_plan(
    title: str,
    start_date: str,
    end_date: str,
    goals: str,
    daily_subjects: list[str],
    daily_minutes: int = 120,
) -> dict:
    """
    Create a multi-day study plan.
    The plan splits daily_minutes equally across daily_subjects.
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError:
        return {"status": "error", "message": "Invalid date format, use YYYY-MM-DD"}

    if end < start:
        return {"status": "error", "message": "end_date must be >= start_date"}

    plan_id = str(uuid.uuid4())
    tasks: list[dict] = []
    per_subject = max(30, daily_minutes // max(1, len(daily_subjects)))
    current = start

    while current <= end:
        for subj in daily_subjects:
            tasks.append(
                {
                    "id": str(uuid.uuid4()),
                    "plan_id": plan_id,
                    "date": current.isoformat(),
                    "subject": subj,
                    "topic": f"{subj} â€“ Study session",
                    "estimated_minutes": per_subject,
                    "completed": False,
                }
            )
        current += timedelta(days=1)

    plan = {
        "id": plan_id,
        "title": title.strip(),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "goals": goals.strip(),
        "tasks": tasks,
    }
    memory.add_item("study_plans", plan)
    return {"status": "ok", "plan": plan}


# -------------------------------
# STUDY SESSION & STATS TOOLS
# -------------------------------

def log_study_session(subject: str, topic: str, duration_minutes: int, notes: str = "", session_date: str | None = None) -> dict:
    """
    Log a single study session.
    If session_date is None, uses today's date.
    """
    if session_date is None:
        d = date.today()
    else:
        try:
            d = datetime.strptime(session_date, "%Y-%m-%d").date()
        except ValueError:
            return {"status": "error", "message": "Invalid session_date format"}

    session = {
        "id": str(uuid.uuid4()),
        "date": d.isoformat(),
        "subject": subject.strip(),
        "topic": topic.strip(),
        "duration_minutes": int(duration_minutes),
        "notes": notes.strip(),
    }
    memory.add_item("study_sessions", session)
    return {"status": "ok", "session": session}


def get_study_stats(days: int = 7) -> dict:
    """
    Compute total study time and subject-wise breakdown for last N days.
    """
    today = date.today()
    cutoff = today - timedelta(days=days - 1)
    sessions = memory.get_items("study_sessions")

    filtered: list[dict] = []
    for s in sessions:
        try:
            d = datetime.strptime(s["date"], "%Y-%m-%d").date()
        except ValueError:
            continue
        if cutoff <= d <= today:
            filtered.append(s)

    total_minutes = sum(s["duration_minutes"] for s in filtered)
    per_subject: Dict[str, int] = {}
    for s in filtered:
        subj = s["subject"]
        per_subject[subj] = per_subject.get(subj, 0) + s["duration_minutes"]

    return {
        "status": "ok",
        "days": days,
        "total_minutes": total_minutes,
        "hours": round(total_minutes / 60, 2),
        "per_subject_minutes": per_subject,
        "session_count": len(filtered),
    }


def daily_summary() -> dict:
    """
    Generate a structured daily summary:
    - today's sessions
    - upcoming assignments (3 days)
    - active study plans and today's tasks
    """
    today_str = date.today().isoformat()

    # Sessions today
    sessions_today = [
        s for s in memory.get_items("study_sessions") if s["date"] == today_str
    ]

    # Upcoming assignments (3 days)
    upcoming = upcoming_assignments(within_days=3)["upcoming"]

    # Active plans + today's tasks
    plans = memory.get_items("study_plans")
    active_plans = []
    today_date = date.today()
    for p in plans:
        try:
            start = datetime.strptime(p["start_date"], "%Y-%m-%d").date()
            end = datetime.strptime(p["end_date"], "%Y-%m-%d").date()
        except ValueError:
            continue
        if start <= today_date <= end:
            tasks_today = [t for t in p["tasks"] if t["date"] == today_str]
            active_plans.append(
                {
                    "id": p["id"],
                    "title": p["title"],
                    "goals": p["goals"],
                    "tasks_today": tasks_today,
                }
            )

    return {
        "status": "ok",
        "date": today_str,
        "sessions_today": sessions_today,
        "upcoming_assignments": upcoming,
        "active_plans": active_plans,
    }



import os

# Make sure GOOGLE_API_KEY is set in your environment or .env file.
# In local dev you might do:
# os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE"

root_agent = Agent(
    name="studyflow_agent",
    model=os.getenv("GEN_MODEL_NAME", "gemini-2.5-flash"),  # configurable model :contentReference[oaicite:3]{index=3}
    description="AI-powered personal study assistant for notes, assignments, and planning.",
    instruction=(
        "You are StudyFlow Agent, an AI-powered personal study assistant for students. "
        "You help with:\n"
        "- creating and searching study notes\n"
        "- tracking assignments and due dates\n"
        "- building multi-day study plans\n"
        "- logging study sessions\n"
        "- summarizing daily progress and study statistics\n\n"
        "Use the provided tools whenever they are helpful. "
        "Prefer structured, concise answers with explanations when needed."
    ),
    tools=[
        add_note,
        search_notes,
        add_assignment,
        list_assignments,
        upcoming_assignments,
        update_assignment_status,
        create_study_plan,
        log_study_session,
        get_study_stats,
        daily_summary,
    ],
)



prompt = """
I am a student preparing for OOPS and Data Communication exams.

1. Create one assignment for each subject with realistic due dates.
2. Create a 3-day study plan with both subjects, 2 hours per day.
3. Then show me a short summary of what you set up.
"""

response = root_agent.run(prompt)  # ADK's standard call pattern :contentReference[oaicite:4]{index=4}
print(response)



# Manually add a note
note_result = add_note(
    subject="OOPS",
    title="Polymorphism overview",
    content="Polymorphism allows methods to have different implementations under a common interface.",
    tags=["oops", "polymorphism", "exam"],
)
pretty(note_result)

# Add a manual assignment
assignment_result = add_assignment(
    subject="Data Communication",
    title="Line Coding Lab Report",
    description="Explain Unipolar, Polar, Bipolar line coding with waveforms.",
    due_date=(date.today() + timedelta(days=3)).isoformat(),
    priority=2,
)
pretty(assignment_result)

# Check memory file
print("\nMemory file exists:", Path("studyflow_memory.json").exists())
print("First 40 lines of JSON:")
txt = Path("studyflow_memory.json").read_text(encoding="utf-8").splitlines()
for line in txt[:40]:
    print(line)



# Log a couple of sessions
pretty(
    log_study_session(
        subject="OOPS",
        topic="Use case diagrams",
        duration_minutes=60,
        notes="Practice actor and include/extend relationships.",
    )
)

pretty(
    log_study_session(
        subject="Data Communication",
        topic="Line coding (Unipolar/Polar/Bipolar)",
        duration_minutes=90,
        notes="Understood basic waveforms.",
    )
)

# Study statistics for last 7 days
stats = get_study_stats(days=7)
print("\nStudy stats (last 7 days):")
pretty(stats)

# Daily summary
summary = daily_summary()
print("\nDaily summary:")
pretty(summary)



class ReminderAgent:
    """Very small helper using StudyFlow's tools/results."""

    def build_reminders(self, days: int = 3) -> list[str]:
        data = upcoming_assignments(within_days=days)
        messages = []
        for a in data["upcoming"]:
            msg = (
                f"Reminder: '{a['title']}' for {a['subject']} is due on {a['due_date']} "
                f"(status: {a['status']}, priority: {a['priority']})."
            )
            messages.append(msg)
        return messages


rem_agent = ReminderAgent()
print("ğŸ”” Reminder messages:")
for line in rem_agent.build_reminders(days=3):
    print("-", line)


