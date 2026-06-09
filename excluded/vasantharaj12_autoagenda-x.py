# Cell 1 — safe conditional installs (Kaggle-friendly)
import importlib, subprocess, sys

def ensure_pkg(pkg_name, pip_name=None, no_deps=False):
    pip_name = pip_name or pkg_name
    try:
        importlib.import_module(pkg_name)
        print(f"{pkg_name} present")
    except Exception:
        cmd = [sys.executable, "-m", "pip", "install", pip_name]
        if no_deps:
            cmd.append("--no-deps")
        print("Installing", pip_name)
        subprocess.check_call(cmd)

# Only install what is likely missing. Avoid large upgrades on Kaggle.
ensure_pkg("pandas")
ensure_pkg("pytz")
ensure_pkg("googleapiclient")   # Google API client - if missing it'll install
ensure_pkg("google.oauth2", "google-auth")   # google-auth
ensure_pkg("icalendar", "icalendar", no_deps=True)  # optional robust ICS lib (no deps)
print("Packages ready.")



# Cell 2 — list uploaded files so you can find your JSON and other uploads.
import glob, os
print("Known uploaded image (from your session):")
UPLOADED_IMAGE = "/mnt/data/818b973c-f8ac-4393-8936-0dfdc9755906.png"
print(" ->", UPLOADED_IMAGE, "exists:", os.path.exists(UPLOADED_IMAGE))
print("\nFiles under /kaggle/input (uploaded datasets):")
for p in sorted(glob.glob('/kaggle/input/**', recursive=True)):
    print(p)
print("\nFiles under /kaggle/working (if you uploaded there):")
for p in sorted(glob.glob('/kaggle/working/**', recursive=True)):
    print(p)

# Auto-detect a service-account JSON if present:
json_candidates = sorted(glob.glob('/kaggle/input/**/*.json', recursive=True) + glob.glob('/kaggle/working/**/*.json', recursive=True))
print("\nDetected .json files (possible service account keys):")
for j in json_candidates:
    print(" -", j)
if not json_candidates:
    print("No JSON found. If you uploaded your service-account key, re-check Add data -> Upload.")



# Cell 3 — config & load tasks (edit CSV_PATH if you uploaded a tasks CSV)
import pandas as pd
from datetime import datetime, timedelta, time
import uuid, os, pytz, json

OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Planner config (edit to taste)
WORK_START_HOUR = 9
WORK_END_HOUR = 17
BUFFER_MIN = 10
PLANNING_DAYS = 5
DEFAULT_ESTIMATE_MIN = 30
DEFAULT_IMPORTANCE = 3
WEIGHTS = {"wImportance": 0.6, "wUrgency": 0.3, "wDependency": 0.1}
MAX_SPLIT_CHUNK = 120
TIMEZONE = "UTC"   # change to your timezone, e.g. "Asia/Kolkata"
tz = pytz.timezone(TIMEZONE)

# If you have a tasks CSV uploaded, set CSV_PATH to its path (example: '/kaggle/input/mytasks/tasks.csv')
CSV_PATH = None

sample_tasks = [
    {"id":"t1","title":"Finish Q3 report","description":"Finalize numbers","importance":5,"deadline":(datetime.utcnow().date()+timedelta(days=2)).isoformat(),"estimate_min":90,"dependencies":""},
    {"id":"t2","title":"Call John","description":"Discuss onboarding","importance":3,"deadline":(datetime.utcnow().date()+timedelta(days=1)).isoformat(),"estimate_min":20,"dependencies":""},
    {"id":"t3","title":"Code review","description":"Review PR #432","importance":4,"deadline":"","estimate_min":45,"dependencies":""},
    {"id":"t4","title":"Plan presentation","description":"Slides for client","importance":4,"deadline":(datetime.utcnow().date()+timedelta(days=4)).isoformat(),"estimate_min":120,"dependencies":""},
    {"id":"t5","title":"Inbox cleanup","description":"Clear low-priority emails","importance":2,"deadline":"","estimate_min":30,"dependencies":""},
]

def parse_iso_date(s):
    try:
        return datetime.fromisoformat(s).date() if s and str(s).strip() else None
    except Exception:
        return None

# Load tasks
if CSV_PATH and os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    tasks = []
    for _, r in df.iterrows():
        tasks.append({
            "id": str(r.get("id") or str(uuid.uuid4())),
            "title": str(r.get("title") or "Untitled"),
            "description": str(r.get("description") or ""),
            "importance": int(r.get("importance") or DEFAULT_IMPORTANCE),
            "deadline": parse_iso_date(r.get("deadline")),
            "estimate_min": int(r.get("estimate_min") or DEFAULT_ESTIMATE_MIN),
            "dependencies": [d.strip() for d in str(r.get("dependencies") or "").split(",") if d.strip()]
        })
else:
    tasks = []
    for t in sample_tasks:
        tasks.append({
            "id": t.get("id") or str(uuid.uuid4()),
            "title": t.get("title","Untitled"),
            "description": t.get("description",""),
            "importance": int(t.get("importance") or DEFAULT_IMPORTANCE),
            "deadline": parse_iso_date(t.get("deadline")) if t.get("deadline") else None,
            "estimate_min": int(t.get("estimate_min") or DEFAULT_ESTIMATE_MIN),
            "dependencies": [d.strip() for d in (t.get("dependencies") or "").split(",") if d.strip()]
        })

print("Loaded", len(tasks), "tasks. Example:")
for t in tasks[:3]:
    print(" -", t["id"], t["title"], "est", t["estimate_min"], "min", "deadline", t["deadline"])



# Cell 4 — scoring & scheduling
from datetime import datetime
today_date = datetime.now(tz).date()

def days_until(deadline_date, from_date=None):
    if not deadline_date: return None
    if from_date is None: from_date = today_date
    delta = (deadline_date - from_date).days
    return max(0, delta)

# scoring
for t in tasks:
    d_until = days_until(t["deadline"], today_date) if t["deadline"] else None
    urgency = (1 / (1 + d_until)) if d_until is not None else 0.0
    dep_penalty = 0.1 if t["dependencies"] else 0.0
    score = WEIGHTS["wImportance"] * t["importance"] + WEIGHTS["wUrgency"] * urgency + WEIGHTS["wDependency"] * dep_penalty
    t["urgency"] = urgency; t["score"] = score

tasks_sorted = sorted(tasks, key=lambda x: x["score"], reverse=True)

# prepare day slots
from datetime import time, timedelta
day_slots = []
for i in range(PLANNING_DAYS):
    day_date = today_date + timedelta(days=i)
    work_start = tz.localize(datetime.combine(day_date, time(WORK_START_HOUR,0)))
    work_end = tz.localize(datetime.combine(day_date, time(WORK_END_HOUR,0)))
    total_minutes = int((work_end - work_start).total_seconds() // 60)
    day_slots.append({"date":day_date, "start":work_start, "end":work_end, "events":[], "free_minutes": total_minutes})

def find_gap_for_day(slot, required_minutes):
    busy = sorted(slot["events"], key=lambda e: e["start"])
    cursor = slot["start"]
    for ev in busy:
        gap_minutes = int((ev["start"] - cursor).total_seconds() // 60)
        if gap_minutes >= required_minutes:
            return cursor
        cursor = ev["end"] + timedelta(minutes=BUFFER_MIN)
    gap_minutes = int((slot["end"] - cursor).total_seconds() // 60)
    if gap_minutes >= required_minutes:
        return cursor
    return None

scheduled_rows = []
for t in tasks_sorted:
    remaining = t["estimate_min"]
    part = 1
    while remaining > 0:
        chunk = min(remaining, MAX_SPLIT_CHUNK)
        placed = False
        for slot in day_slots:
            if slot["free_minutes"] < (chunk + BUFFER_MIN):
                continue
            start_time = find_gap_for_day(slot, chunk)
            if start_time:
                end_time = start_time + timedelta(minutes=chunk)
                ev_title = f"{t['title']}" + (f" (Part {part})" if remaining > chunk else "")
                slot["events"].append({"task_id":t["id"], "title":ev_title, "start":start_time, "end":end_time, "minutes":chunk})
                slot["free_minutes"] -= (chunk + BUFFER_MIN)
                scheduled_rows.append({"task_id":t["id"], "title":t["title"], "part":part, "start":start_time, "end":end_time, "minutes":chunk, "date":slot["date"], "unscheduled":False})
                remaining -= chunk
                part += 1
                placed = True
                break
        if not placed:
            # no more room in 5-day window
            scheduled_rows.append({"task_id":t["id"], "title":t["title"], "part":part, "start":None, "end":None, "minutes":remaining, "date":None, "unscheduled":True})
            remaining = 0

schedule_df = pd.DataFrame(scheduled_rows)
# sort for display
schedule_df["start_sort"] = schedule_df["start"].apply(lambda x: x if x is not None else pd.Timestamp.max.tz_localize(None))
schedule_df = schedule_df.sort_values(["date","start_sort"]).drop(columns=["start_sort"])
print("Schedule created. Preview:")
display(schedule_df.head(50))
# Save CSV + ICS will be in next cell



# Cell 5 — export CSV & ICS
csv_path = os.path.join(OUTPUT_DIR, "five_day_schedule.csv")
schedule_df.to_csv(csv_path, index=False)
print("Saved CSV to:", csv_path)

# Build simple ICS (timezone-aware -> UTC)
def dt_to_ics(dt):
    return dt.astimezone(pytz.UTC).strftime("%Y%m%dT%H%M%SZ")

ics_lines = ["BEGIN:VCALENDAR","VERSION:2.0","PRODID:-//5DayPlanner//EN"]
for idx, row in schedule_df.iterrows():
    if row["unscheduled"]:
        continue
    uid = str(uuid.uuid4())
    dtstart = row["start"]; dtend = row["end"]
    ics_lines += [
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}",
        f"DTSTART:{dt_to_ics(dtstart)}",
        f"DTEND:{dt_to_ics(dtend)}",
        f"SUMMARY:{row['title']}",
        f"DESCRIPTION:Auto-scheduled task {row['task_id']} (Part {row['part']})",
        "END:VEVENT"
    ]
ics_lines.append("END:VCALENDAR")
ics_path = os.path.join(OUTPUT_DIR, "five_day_schedule.ics")
with open(ics_path, "w") as f:
    f.write("\n".join(ics_lines))
print("Saved ICS to:", ics_path)



# Cell 6 — detect a service-account JSON in /kaggle/input or /kaggle/working and print client_email
import glob, json, os
cands = sorted(glob.glob('/kaggle/input/json-data/gen-lang-client-0380668532-83f83f22f022.json', recursive=True) + glob.glob('/kaggle/working/**/*.json', recursive=True))
print("JSON candidates found:")
for i,c in enumerate(cands):
    print(i+1, "-", c)
    
if not cands:
    print("No JSON found. Upload your service account JSON via Add data -> Upload.")
else:
    # choose first by default, but show all
    SERVICE_ACCOUNT_FILE = cands[0]
    print("\nUsing (default) SERVICE_ACCOUNT_FILE =", SERVICE_ACCOUNT_FILE)
    try:
        with open(SERVICE_ACCOUNT_FILE,'r') as f:
            data = json.load(f)
        print("client_email:", data.get("client_email"))
        print("project_id:", data.get("project_id"))
    except Exception as e:
        print("Failed to read JSON:", e)
    
print("\nIf this is not the correct JSON, upload the correct file and re-run this cell.")



# Cell 7 — Create a single test event (to confirm permissions)
SERVICE_ACCOUNT_FILE = "/kaggle/input/json-data/gen-lang-client-0380668532-83f83f22f022.json"

# You shared your personal Google Calendar with the service account:
TARGET_CALENDAR_ID = "vasantharajb12@gmail.com"

from google.oauth2 import service_account
from googleapiclient.discovery import build
import pytz
from datetime import datetime, timedelta

# Load service account credentials
creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=['https://www.googleapis.com/auth/calendar']
)
service = build('calendar', 'v3', credentials=creds)

# Create a test event for 10 minutes from now
start = datetime.utcnow() + timedelta(minutes=10)
end = start + timedelta(minutes=5)

event = {
    'summary': 'Kaggle SA Test Event',
    'description': 'Test event created by service account',
    'start': {'dateTime': start.astimezone(pytz.UTC).isoformat()},
    'end': {'dateTime': end.astimezone(pytz.UTC).isoformat()},
}

created = service.events().insert(
    calendarId=TARGET_CALENDAR_ID,
    body=event
).execute()

print("Created test event id:", created.get('id'))
print("✔ Check Google Calendar now — it should appear.")



# Cell 8 — Bulk push scheduled events into Google Calendar
SERVICE_ACCOUNT_FILE = "/kaggle/input/json-data/gen-lang-client-0380668532-83f83f22f022.json"
TARGET_CALENDAR_ID = "vasantharajb12@gmail.com"

from google.oauth2 import service_account
from googleapiclient.discovery import build
import pytz, os
import pandas as pd

# Ensure schedule_df exists
try:
    schedule_df
except NameError:
    raise RuntimeError("schedule_df not found. Run scheduling cells (Cell 3 & 4).")

# Build Calendar API client
creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=['https://www.googleapis.com/auth/calendar']
)
service = build('calendar', 'v3', credentials=creds)

created_ids = []

for idx, row in schedule_df.iterrows():
    if row.get('unscheduled', False):
        continue

    start_dt = row['start']
    end_dt = row['end']

    # Ensure timezone-aware
    if start_dt.tzinfo is None:
        start_dt = pytz.UTC.localize(start_dt)
        end_dt = pytz.UTC.localize(end_dt)

    event_body = {
        'summary': row['title'],
        'description': f"Auto-scheduled task {row['task_id']} (Part {row['part']})",
        'start': {'dateTime': start_dt.astimezone(pytz.UTC).isoformat()},
        'end':   {'dateTime': end_dt.astimezone(pytz.UTC).isoformat()},
        'reminders': {'useDefault': False}
    }

    created = service.events().insert(
        calendarId=TARGET_CALENDAR_ID,
        body=event_body
    ).execute()

    created_ids.append(created.get('id'))

print("✔ Created events:", len(created_ids))

# Save created IDs for undo
out_path = "/kaggle/working/created_event_ids.csv"
pd.DataFrame({"event_id": created_ids}).to_csv(out_path, index=False)
print("✔ Saved created_event_ids.csv to", out_path)



# Cell 9 — undo created events using saved created_event_ids.csv
import pandas as pd, os
ids_path = os.path.join(OUTPUT_DIR,"created_event_ids.csv")
if not os.path.exists(ids_path):
    print("No created_event_ids.csv found at", ids_path)
else:
    ids = pd.read_csv(ids_path)['event_id'].tolist()
    deleted = 0
    for eid in ids:
        try:
            service.events().delete(calendarId=TARGET_CALENDAR_ID or 'primary', eventId=eid).execute()
            deleted += 1
        except Exception as e:
            print("Delete failed for", eid, e)
    print("Deleted events (attempted):", deleted)


