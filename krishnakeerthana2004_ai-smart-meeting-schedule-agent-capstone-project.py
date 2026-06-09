# Cell A — install deps (Colab)
!pip install pandas python-dateutil --quiet
print("Deps installed.")


# Cell B — agent core (save as agent.py in runtime)
agent_code = r'''
"""
Smart Meeting Scheduler - minimal agent
Demonstrates:
1) Tool use (mock calendar read/write)
2) Multi-step planning
3) Memory (JSON file)
"""

import json
import os
from datetime import datetime, timedelta
from dateutil import parser, tz

MEMORY_FILE = "user_memory.json"
CALENDAR_FILE = "mock_calendar.json"

# ---------- Memory helpers ----------
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_memory(mem):
    with open(MEMORY_FILE, "w") as f:
        json.dump(mem, f, indent=2)

# ---------- Mock Calendar "Tool" ----------
def load_calendar():
    if os.path.exists(CALENDAR_FILE):
        with open(CALENDAR_FILE, "r") as f:
            return json.load(f)
    return []

def save_calendar(cal):
    with open(CALENDAR_FILE, "w") as f:
        json.dump(cal, f, indent=2, default=str)

def add_event(start_iso, end_iso, title="Meeting"):
    cal = load_calendar()
    cal.append({"start": start_iso, "end": end_iso, "title": title})
    save_calendar(cal)
    return {"status": "ok", "event": {"start": start_iso, "end": end_iso, "title": title}}

def find_free_slots(start_window_iso, end_window_iso, duration_minutes=30):
    start_win = parser.isoparse(start_window_iso)
    end_win = parser.isoparse(end_window_iso)
    cal = load_calendar()
    events = []
    for e in cal:
        events.append((parser.isoparse(e["start"]), parser.isoparse(e["end"])))
    events.sort()
    candidates = []
    cursor = start_win
    delta = timedelta(minutes=duration_minutes)
    while cursor + delta <= end_win:
        slot_ok = True
        for s,e in events:
            if not (cursor + delta <= s or cursor >= e):
                slot_ok = False
                break
        if slot_ok:
            candidates.append((cursor.isoformat(), (cursor+delta).isoformat()))
        cursor += timedelta(minutes=30)
    return candidates

# ---------- Simple Planner + Parser ----------
def parse_request(text):
    text = text.lower()
    duration = 30
    if "60" in text or "60 minutes" in text or "1 hour" in text:
        duration = 60
    now = datetime.now(tz=tz.tzlocal())
    if "tomorrow" in text:
        target_day = now + timedelta(days=1)
        start = target_day.replace(hour=10, minute=0, second=0, microsecond=0)
    elif "next week" in text:
        target_day = now + timedelta(days=7)
        start = target_day.replace(hour=10, minute=0, second=0, microsecond=0)
    else:
        try:
            start = parser.parse(text, default=now)
            if start < now:
                start = now + timedelta(days=1)
                start = start.replace(hour=10, minute=0)
        except Exception:
            start = now + timedelta(days=1)
            start = start.replace(hour=10, minute=0, second=0, microsecond=0)
    end = start + timedelta(minutes=duration)
    return {"start": start.isoformat(), "end": end.isoformat(), "duration": duration}

def rank_slots(slots, prefs):
    if not slots:
        return []
    if not prefs:
        return [slots[0]]
    preferred = []
    for s,e in slots:
        sdt = parser.isoparse(s)
        if prefs.get("preferred_start_hour") is not None:
            if prefs["preferred_start_hour"] <= sdt.hour < prefs.get("preferred_end_hour", 24):
                preferred.append((s,e))
    return preferred if preferred else [slots[0]]

# ---------- High-level agent flow ----------
def run_agent(user_id, user_text):
    mem = load_memory()
    user_prefs = mem.get(user_id, {"preferred_start_hour":10, "preferred_end_hour":17, "timezone": str(tz.tzlocal())})
    parse = parse_request(user_text)
    start_window = datetime.now(tz=tz.tzlocal()).isoformat()
    end_window = (datetime.now(tz=tz.tzlocal()) + timedelta(days=3)).isoformat()
    slots = find_free_slots(start_window, end_window, duration_minutes=parse["duration"])
    candidates = rank_slots(slots, user_prefs)
    chosen = candidates[0] if candidates else None
    if chosen:
        add_event(chosen[0], chosen[1], title="Auto-scheduled meeting")
        mem[user_id] = user_prefs
        mem[user_id]["last_scheduled"] = chosen[0]
        save_memory(mem)
        return {"status":"scheduled", "slot": chosen}
    else:
        return {"status":"no-slots", "message":"No free slots found in the next 3 days."}
'''
with open("agent.py","w") as f:
    f.write(agent_code)
print("agent.py created.")



# Cell C — init calendar & run demo
from datetime import datetime, timedelta
from dateutil import tz
import json, os
from agent import run_agent

# helper to create iso times
now = datetime.now(tz=tz.tzlocal()).replace(hour=0, minute=0, second=0, microsecond=0)
def iso(hour, mins=0, days=0):
    dt = (now + timedelta(days=days)).replace(hour=hour, minute=mins)
    return dt.isoformat()

# create initial mock calendar (shows BEFORE state)
mock_cal = [
  {"start": iso(10,0,0), "end": iso(11,0,0), "title":"Existing Meeting A"},
  {"start": iso(14,0,0), "end": iso(14,30,0), "title":"Existing Meeting B"}
]
with open("mock_calendar.json","w") as f:
    json.dump(mock_cal, f, indent=2)
# clear memory
open("user_memory.json","w").write("{}")
print("Mock calendar & memory initialized.\n")

def show_calendar():
    with open("mock_calendar.json","r") as f:
        cal = json.load(f)
    print("Calendar events:")
    for e in cal:
        print("-", e["title"], e["start"], "→", e["end"])

print("BEFORE:")
show_calendar()

user_id = "user_1"
requests = [
    "Schedule a 30 minute meeting tomorrow morning",
    "Schedule a 60 minute meeting next week",
    "Schedule a quick 30 minutes today at 10am"
]

for q in requests:
    print("\nUser:", q)
    res = run_agent(user_id, q)
    print("Agent result:", res)

print("\nAFTER:")
show_calendar()
print("\nMemory file contents:")
print(open("user_memory.json").read())


