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


import os
import json
from copy import deepcopy

class MemoryStore:
    """
    Very small JSON-based memory.
    Stores user preferences and task history per session_id.
    File is saved in /kaggle/working so it persists between runs in this session.
    """
    def __init__(self, path="/kaggle/working/organizer_memory.json"):
        self.path = path
        self._data = None
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}
        else:
            self._data = {}

    def _save(self):
        try:
            with open(self.path, "w") as f:
                json.dump(self._data, f, indent=2)
        except Exception:
            # we don't want memory failures to break the agent
            pass

    def get_session(self, session_id="default"):
        if session_id not in self._data:
            # default preferences
            self._data[session_id] = {
                "preferences": {
                    "work_start": 10,
                    "work_end": 18,
                    "baseline_minutes": 30,
                    "avoid_weekends": True
                },
                "history": []
            }
        return deepcopy(self._data[session_id])

    def update_session(self, session_id, session_state):
        self._data[session_id] = session_state
        self._save()



# Cell 1 — Imports & basic helpers
from datetime import datetime, timedelta, time as dtime
import json
import math
import uuid

# Small helper for human-readable durations
def minutes_to_iso(minutes:int):
    hours = minutes // 60
    mins = minutes % 60
    if hours:
        return f"{hours}h {mins}m"
    return f"{mins}m"


# Cell 2 — Task splitter
def determine_steps(task_text: str):
    """
    Heuristic-driven splitter:
    - If user lists steps with commas/semicolon, respect them.
    - Otherwise produce a typical decomposition for knowledge-work tasks.
    """
    text = task_text.strip()
    # if user supplied an obvious step list, use it
    if ";" in text or "\n" in text:
        parts = [p.strip() for p in text.replace("\n",";").split(";") if p.strip()]
        return parts
    if "," in text and len(text.split(",")) > 2:
        return [p.strip() for p in text.split(",") if p.strip()]
    # Otherwise fallback to a sensible list
    return [
        "Clarify goal and success criteria",
        "Gather required materials/data",
        "Outline structure / plan",
        "Create first draft / implementation",
        "Review and refine",
        "Finalize and deliver"
    ]


# Cell 3 — Time estimator 
def estimate_subtask_durations(subtask_list, baseline_minutes=25):
    """
    Return a dict mapping subtask -> estimated_minutes.
    - baseline_minutes is the default for a short task (e.g., 25 minutes)
    - heuristics bump estimates for words like 'draft', 'research', 'review'
    """
    estimates = {}
    for s in subtask_list:
        s_low = s.lower()
        minutes = baseline_minutes
        if any(k in s_low for k in ("research", "gather", "collect", "analyze")):
            minutes *= 2  # research needs more time
        if any(k in s_low for k in ("draft", "write", "implement")):
            minutes *= 1.5
        if "review" in s_low or "refine" in s_low or "proof" in s_low:
            minutes *= 1.25
        # round to nearest 5
        minutes = int(5 * round(minutes/5))
        estimates[s] = max(5, minutes)
    return estimates


# Cell 4 — Deadline-aware priority function
def compute_priority(task_text, deadline_iso=None, now=None):
    """
    Returns 'High', 'Medium', or 'Low' based on:
    - presence of explicit 'urgent' words
    - how close deadline is vs estimated total duration
    """
    if now is None:
        now = datetime.now()
    text = task_text.lower()
    # immediate keywords
    if any(k in text for k in ("urgent", "asap", "immediately", "right now", "today")):
        return "High"
    # parse deadline if provided
    if deadline_iso:
        try:
            deadline = datetime.fromisoformat(deadline_iso)
            delta = (deadline - now).total_seconds() / 3600.0  # hours
            if delta <= 6:
                return "High"
            elif delta <= 48:
                return "Medium"
            else:
                return "Low"
        except Exception:
            pass
    # default fallback
    return "Medium"


# Cell 5 — Schedule builder (working hours, breaks, optional Pomodoro)
def build_timeblocked_schedule(estimated_minutes_map,
                               work_start_hour=10,
                               work_end_hour=18,
                               lunch_break=(13, 14),   # hour range (13:00 to 14:00)
                               use_pomodoro=False,
                               pomodoro_length=25,
                               short_break=5,
                               long_break=15):
    """
    Create a list of dicts: {id, task, start: iso, end: iso, minutes}
    Very conservative: will not place work during lunch; pushes tasks to next day if needed.
    """
    today = datetime.now().date()
    current_dt = datetime.combine(today, dtime(hour=work_start_hour, minute=0))
    schedule = []
    # helper to advance time to next allowed slot if currently in break or after hours
    def advance_to_working_hours(dt):
        if dt.time() >= dtime(hour=work_end_hour):
            # move to next day start
            return datetime.combine(dt.date() + timedelta(days=1), dtime(hour=work_start_hour))
        if lunch_break and (lunch_break[0] <= dt.hour < lunch_break[1]):
            return datetime.combine(dt.date(), dtime(hour=lunch_break[1]))
        if dt.time() < dtime(hour=work_start_hour):
            return datetime.combine(dt.date(), dtime(hour=work_start_hour))
        return dt

    current_dt = advance_to_working_hours(current_dt)

    for task, mins in estimated_minutes_map.items():
        remaining = mins
        while remaining > 0:
            current_dt = advance_to_working_hours(current_dt)
            block_mins = remaining
            # if using pomodoro, only allow pomodoro_length blocks
            if use_pomodoro:
                block_mins = min(block_mins, pomodoro_length)
            # ensure we don't cross work_end_hour or lunch
            end_dt_candidate = current_dt + timedelta(minutes=block_mins)
            # if crosses lunch or end hour, shorten block
            if (lunch_break and current_dt.hour < lunch_break[1] and end_dt_candidate.hour >= lunch_break[1]):
                # shorten to lunch start
                block_mins = int((datetime.combine(current_dt.date(), dtime(hour=lunch_break[0])) - current_dt).total_seconds() / 60)
                if block_mins <= 0:
                    current_dt = datetime.combine(current_dt.date(), dtime(hour=lunch_break[1]))
                    continue
            if end_dt_candidate.time() > dtime(hour=work_end_hour):
                block_mins = int((datetime.combine(current_dt.date(), dtime(hour=work_end_hour)) - current_dt).total_seconds() / 60)
                if block_mins <= 0:
                    current_dt = datetime.combine(current_dt.date() + timedelta(days=1), dtime(hour=work_start_hour))
                    continue
            start_iso = current_dt.isoformat()
            end_iso = (current_dt + timedelta(minutes=block_mins)).isoformat()
            schedule.append({
                "id": str(uuid.uuid4())[:8],
                "task": task,
                "start": start_iso,
                "end": end_iso,
                "minutes": block_mins
            })
            current_dt = current_dt + timedelta(minutes=block_mins)
            remaining -= block_mins
            # if pomodoro, insert short break after each pomodoro segment
            if use_pomodoro and remaining > 0:
                current_dt += timedelta(minutes=short_break)
    return schedule



# Cell 6 — Reminder generator and iCal exporter (simple)
def make_reminders_from_schedule(schedule, reminder_offset_minutes=30):
    """
    Create reminders placed at (start - reminder_offset_minutes).
    Returns list of {"task","remind_at_iso","note"}.
    """
    reminders = []
    for item in schedule:
        start = datetime.fromisoformat(item["start"])
        remind_at = start - timedelta(minutes=reminder_offset_minutes)
        reminders.append({
            "task": item["task"],
            "remind_at": remind_at.isoformat(),
            "note": f"Upcoming: {item['task']} at {start.strftime('%H:%M')}"
        })
    return reminders

def export_schedule_to_ical(schedule, title_prefix="Organizer"):
    """
    Very small iCal export building minimal VEVENT blocks. Returns string.
    You can save to file and import to Google Calendar.
    """
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//PersonalOrganizer//EN"
    ]
    for ev in schedule:
        start_dt = datetime.fromisoformat(ev["start"])
        end_dt = datetime.fromisoformat(ev["end"])
        uid = f"{ev['id']}@organizer.local"
        lines += [
            "BEGIN:VEVENT",
            f"UID:{uid}",
            f"DTSTAMP:{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}",
            f"DTSTART:{start_dt.strftime('%Y%m%dT%H%M%S')}",
            f"DTEND:{end_dt.strftime('%Y%m%dT%H%M%S')}",
            f"SUMMARY:{title_prefix} - {ev['task']}",
            "END:VEVENT"
        ]
    lines.append("END:VCALENDAR")
    return "\n".join(lines)


class TaskDecomposerAgent:
    """
    Agent that only worries about breaking a task into subtasks.
    Uses the determine_steps() tool.
    """
    def run(self, task_text, session_state):
        subtasks = determine_steps(task_text)
        return {"subtasks": subtasks}


class EstimationAgent:
    """
    Agent that assigns time estimates to each subtask.
    Uses estimate_subtask_durations() tool.
    """
    def __init__(self, default_baseline=25):
        self.default_baseline = default_baseline

    def run(self, subtasks, session_state):
        baseline = session_state["preferences"].get("baseline_minutes", self.default_baseline)
        estimates = estimate_subtask_durations(subtasks, baseline_minutes=baseline)
        return {"estimates": estimates}


class SchedulingAgent:
    """
    Agent that builds a time-blocked schedule.
    Uses build_timeblocked_schedule() tool.
    """
    def run(self, estimates, session_state, use_pomodoro=False):
        prefs = session_state["preferences"]
        schedule = build_timeblocked_schedule(
            estimates,
            work_start_hour=prefs.get("work_start", 10),
            work_end_hour=prefs.get("work_end", 18),
            lunch_break=(13, 14),
            use_pomodoro=use_pomodoro
        )
        return {"schedule": schedule}


class ReminderAgent:
    """
    Agent that turns the schedule into reminders and calendar text.
    Uses make_reminders_from_schedule() and export_schedule_to_ical().
    """
    def run(self, schedule, session_state):
        reminders = make_reminders_from_schedule(schedule)
        ical_text = export_schedule_to_ical(schedule, title_prefix="Smart Organizer")
        return {"reminders": reminders, "ical_text": ical_text}



class OrganizerOrchestrator:
    """
    Top-level multi-agent controller.
    - Loads session memory
    - Calls decomposer -> estimation -> scheduler -> reminders
    - Updates memory with the new task
    """
    def __init__(self, memory_store=None):
        self.memory = memory_store or MemoryStore()
        self.decomposer = TaskDecomposerAgent()
        self.estimator = EstimationAgent()
        self.scheduler = SchedulingAgent()
        self.reminder_agent = ReminderAgent()

    def run(self, user_text, session_id="default", deadline_iso=None, use_pomodoro=False):
        # 1. Load session memory
        state = self.memory.get_session(session_id)

        # 2. Priority uses your existing compute_priority() tool
        priority = compute_priority(user_text, deadline_iso=deadline_iso)

        # 3. Sub-agent: decomposer
        decomp_out = self.decomposer.run(user_text, state)
        subtasks = decomp_out["subtasks"]

        # 4. Sub-agent: estimator
        est_out = self.estimator.run(subtasks, state)
        estimates = est_out["estimates"]

        # 5. Sub-agent: scheduler
        sched_out = self.scheduler.run(estimates, state, use_pomodoro=use_pomodoro)
        schedule = sched_out["schedule"]

        # 6. Sub-agent: reminders
        rem_out = self.reminder_agent.run(schedule, state)
        reminders = rem_out["reminders"]
        ical_text = rem_out["ical_text"]

        total_mins = sum(estimates.values())

        # 7. Update memory (history)
        state["history"].append({
            "task": user_text,
            "priority": priority,
            "deadline": deadline_iso,
            "total_minutes": total_mins,
            "created_at": datetime.now().isoformat()
        })
        self.memory.update_session(session_id, state)

        # 8. Final structured response
        return {
            "session_id": session_id,
            "task": user_text,
            "priority": priority,
            "deadline": deadline_iso,
            "subtasks": subtasks,
            "estimates_minutes": estimates,
            "total_estimated_time": total_mins,
            "total_estimated_time_human": minutes_to_iso(total_mins),
            "schedule": schedule,
            "reminders": reminders,
            "ical_text": ical_text,
            "user_preferences": state["preferences"],
            "tasks_completed_so_far": len(state["history"])
        }



# Example usage of the multi-agent orchestrator

orchestrator = OrganizerOrchestrator()

example_task = "Prepare project review document and send it to the manager"
deadline = (datetime.now() + timedelta(hours=24)).isoformat()

final_output = orchestrator.run(
    user_text=example_task,
    session_id="user_1",
    deadline_iso=deadline,
    use_pomodoro=True
)

# This is what you'd normally write to submission.json
print(json.dumps(final_output, indent=2))



import json

with open("/kaggle/working/submission.json", "w") as f:
    json.dump(final_output, f, indent=2)

print("submission.json created successfully!")


