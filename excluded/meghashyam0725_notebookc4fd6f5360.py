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


import pandas as pd


calendar_columns = [
    "meeting_id",     
    "date",            
    "start_time",     
    "end_time",        
    "title",          
    "participants"    
]

calendar_df = pd.DataFrame(columns=calendar_columns)

print("Empty calendar initialized:")
print(calendar_df)


memory = {
    "preferred_duration_minutes": 30,     
    "working_hours": {
        "start": "09:00",                 
        "end": "17:00"                    
    },
    "meeting_history": []                 
}

print("Agent memory initialized:")
print(memory)


from datetime import datetime, timedelta

def to_datetime(date_str, time_str):
    return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")

def time_range_overlaps(s1, e1, s2, e2):
    return not (e1 <= s2 or e2 <= s1)

def list_meetings(user_name):
    mask = calendar_df["participants"].str.contains(user_name, case=False, na=False)
    return calendar_df[mask].sort_values(["date", "start_time"])

def find_free_slot(date_str, participants, duration_minutes=None):
    if duration_minutes is None:
        duration_minutes = memory["preferred_duration_minutes"]
    work_start = to_datetime(date_str, memory["working_hours"]["start"])
    work_end = to_datetime(date_str, memory["working_hours"]["end"])
    slot_delta = timedelta(minutes=duration_minutes)
    day_meetings = calendar_df[calendar_df["date"] == date_str].copy()
    day_meetings = day_meetings[
        day_meetings["participants"].apply(lambda p: any(name.lower() in p.lower() for name in participants))
    ]
    busy_ranges = [
        (to_datetime(row["date"], row["start_time"]), to_datetime(row["date"], row["end_time"]))
        for _, row in day_meetings.iterrows()
    ]
    current_start = work_start
    while current_start + slot_delta <= work_end:
        current_end = current_start + slot_delta
        if not any(time_range_overlaps(current_start, current_end, s, e) for s, e in busy_ranges):
            return current_start, current_end
        current_start += timedelta(minutes=15)
    return None, None

def schedule_meeting(date_str, start_dt, end_dt, title, participants):
    global calendar_df
    new_id = len(calendar_df) + 1
    new_row = {
        "meeting_id": new_id,
        "date": date_str,
        "start_time": start_dt.strftime("%H:%M"),
        "end_time": end_dt.strftime("%H:%M"),
        "title": title,
        "participants": ", ".join(participants)
    }
    calendar_df = pd.concat([calendar_df, pd.DataFrame([new_row])], ignore_index=True)
    memory["meeting_history"].append(new_id)
    return new_id

def cancel_meeting(meeting_id):
    global calendar_df
    before = len(calendar_df)
    calendar_df = calendar_df[calendar_df["meeting_id"] != meeting_id].reset_index(drop=True)
    after = len(calendar_df)
    return before != after

def update_preferences(duration_minutes=None, work_start=None, work_end=None):
    if duration_minutes is not None:
        memory["preferred_duration_minutes"] = duration_minutes
    if work_start is not None:
        memory["working_hours"]["start"] = work_start
    if work_end is not None:
        memory["working_hours"]["end"] = work_end
    return memory

update_preferences(duration_minutes=45, work_start="10:00", work_end="18:00")

today = datetime.today().strftime("%Y-%m-%d")
participants = ["Megha Shyam", "Krishna"]

start_dt, end_dt = find_free_slot(today, participants)
print("First free slot:", start_dt, end_dt)

if start_dt and end_dt:
    meeting_id = schedule_meeting(today, start_dt, end_dt, "Project Discussion", participants)
    print("Scheduled meeting id:", meeting_id)
else:
    print("No free slot found.")

print("Meetings for Megha Shyam:")
print(list_meetings("Megha Shyam"))

print("Meetings for Krishna:")
print(list_meetings("Krishna"))


from datetime import datetime, timedelta

def extract_participants(user_message):
    msg = user_message.lower()
    if "with" in msg:
        after_with = msg.split("with", 1)[1]
        if "," in after_with:
            names = [n.strip().title() for n in after_with.split(",") if n.strip()]
        else:
            names = [name.strip().title() for name in after_with.split() if name.strip(",").isalpha()]
        return names
    return ["Megha Shyam", "Krishna"]

def extract_title(user_message):
    user_message = user_message.lower()
    for kw in ["about", "for"]:
        if kw in user_message:
            title = user_message.split(kw, 1)[1].strip().title()
            return title[:50]
    return "Project Discussion"

def extract_date(user_message):
    msg = user_message.lower()
    if "tomorrow" in msg:
        dt = datetime.today() + timedelta(days=1)
        return dt.strftime("%Y-%m-%d")
    return datetime.today().strftime("%Y-%m-%d")

def agent_handle_request(user_message):
    msg_lower = user_message.lower().strip()

    if "set my preferences" in msg_lower:
        words = msg_lower.split()
        duration = None
        work_start = None
        work_end = None
        for w in words:
            if "minute" in w:
                try:
                    duration = int(w.split("-")[0])
                except:
                    pass
            if ":" in w:
                if work_start is None:
                    work_start = w
                else:
                    work_end = w
        update_preferences(duration_minutes=duration, work_start=work_start, work_end=work_end)
        return f"Preferences updated: duration {duration} minutes, working hours {work_start}–{work_end}."

    if "list my meetings" in msg_lower or "show" in msg_lower:
        name = None
        for possible in ["megha shyam", "krishna"]:
            if possible in msg_lower:
                name = possible.title()
        if name is None:
            name = "Megha Shyam"
        return list_meetings(name)

    if "schedule" in msg_lower:
        date_str = extract_date(user_message)
        participants = extract_participants(user_message)
        title = extract_title(user_message)
        start_dt, end_dt = find_free_slot(date_str, participants)
        if start_dt is None or end_dt is None:
            return "No free slot available for all participants."
        meeting_id = schedule_meeting(date_str, start_dt, end_dt, title, participants)
        return f"Scheduled '{title}' on {date_str} from {start_dt.strftime('%H:%M')} to {end_dt.strftime('%H:%M')} (id={meeting_id})."

    if "cancel" in msg_lower:
        words = msg_lower.split()
        meeting_id = None
        for w in words:
            if w.isdigit():
                meeting_id = int(w)
                break
        if meeting_id:
            success = cancel_meeting(meeting_id)
            if success:
                return f"Cancelled meeting {meeting_id}."
            else:
                return f"No meeting found with id {meeting_id}."
        else:
            return "Please specify the meeting id to cancel."

    return "Sorry, I didn't understand. Try: 'schedule', 'list my meetings Megha Shyam', 'cancel meeting 1', or 'set my preferences'."

print(agent_handle_request("Schedule a meeting with Krishna, Megha Shyam tomorrow about AI Review"))
print(agent_handle_request("List my meetings Megha Shyam"))
print(agent_handle_request("Cancel meeting 1"))
print(agent_handle_request("Set my preferences to 30-minute meetings from 10:00 to 17:00"))



!pip install dateparser


import pandas as pd
from datetime import datetime, timedelta

def to_datetime(date_str, time_str):
    return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")

def time_range_overlaps(s1, e1, s2, e2):
    return not (e1 <= s2 or e2 <= s1)

calendar_data = [
    {"meeting_id": 1, "date": "2025-11-15", "start_time": "09:00", "end_time": "10:00", "title": "Team Sync", "participants": "Megha Shyam, Krishna"},
    {"meeting_id": 2, "date": "2025-11-15", "start_time": "10:45", "end_time": "11:30", "title": "Project Discussion", "participants": "Megha Shyam, Krishna"},
]

calendar_df = pd.DataFrame(calendar_data)

def check_conflict(date_str, start_time, end_time, participants):
    for _, row in calendar_df.iterrows():
        if row['date'] != date_str:
            continue
        if any(name.strip().lower() in row['participants'].lower() for name in participants):
            existing_start = to_datetime(row['date'], row['start_time'])
            existing_end = to_datetime(row['date'], row['end_time'])
            new_start = to_datetime(date_str, start_time)
            new_end = to_datetime(date_str, end_time)
            if time_range_overlaps(existing_start, existing_end, new_start, new_end):
                print("Conflict: Overlaps with meeting:", row['title'])
                return True
    print("No conflict detected.")
    return False

check_conflict("2025-11-15", "10:00", "10:45", ["Megha Shyam", "Krishna"])
check_conflict("2025-11-15", "09:30", "10:15", ["Megha Shyam", "Krishna"])



import pandas as pd
from datetime import datetime, timedelta


calendar_columns = ["meeting_id", "date", "start_time", "end_time", "title", "participants"]
calendar_df = pd.DataFrame(columns=calendar_columns)

memory = {
    "preferred_duration_minutes": 30,
    "working_hours": {"start": "09:00", "end": "17:00"},
    "meeting_history": []
}

def to_datetime(date_str, time_str):
    return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")

def time_range_overlaps(s1, e1, s2, e2):
    return not (e1 <= s2 or e2 <= s1)

def list_meetings(user_name):
    mask = calendar_df["participants"].str.contains(user_name, case=False, na=False)
    return calendar_df[mask].sort_values(["date", "start_time"])

def find_free_slot(date_str, participants, duration_minutes=None):
    if duration_minutes is None:
        duration_minutes = memory["preferred_duration_minutes"]
    work_start = to_datetime(date_str, memory["working_hours"]["start"])
    work_end = to_datetime(date_str, memory["working_hours"]["end"])
    slot_delta = timedelta(minutes=duration_minutes)
    day_meetings = calendar_df[calendar_df["date"] == date_str].copy()
    day_meetings = day_meetings[
        day_meetings["participants"].apply(lambda p: any(name.lower() in p.lower() for name in participants))
    ]
    busy_ranges = [
        (to_datetime(row["date"], row["start_time"]), to_datetime(row["date"], row["end_time"]))
        for _, row in day_meetings.iterrows()
    ]
    current_start = work_start
    while current_start + slot_delta <= work_end:
        current_end = current_start + slot_delta
        if not any(time_range_overlaps(current_start, current_end, s, e) for s, e in busy_ranges):
            return current_start, current_end
        current_start += timedelta(minutes=15)
    return None, None

def schedule_meeting(date_str, start_dt, end_dt, title, participants):
    global calendar_df
    new_id = len(calendar_df) + 1
    new_row = {
        "meeting_id": new_id,
        "date": date_str,
        "start_time": start_dt.strftime("%H:%M"),
        "end_time": end_dt.strftime("%H:%M"),
        "title": title,
        "participants": ", ".join(participants)
    }
    calendar_df = pd.concat([calendar_df, pd.DataFrame([new_row])], ignore_index=True)
    memory["meeting_history"].append(new_id)
    return new_id

def cancel_meeting(meeting_id):
    global calendar_df
    before = len(calendar_df)
    calendar_df = calendar_df[calendar_df["meeting_id"] != meeting_id].reset_index(drop=True)
    after = len(calendar_df)
    return before != after

def update_preferences(duration_minutes=None, work_start=None, work_end=None):
    if duration_minutes is not None:
        memory["preferred_duration_minutes"] = duration_minutes
    if work_start is not None:
        memory["working_hours"]["start"] = work_start
    if work_end is not None:
        memory["working_hours"]["end"] = work_end
    return memory

def extract_participants(user_message):
    msg = user_message.lower()
    if "with" in msg:
        after_with = msg.split("with", 1)[1]
        if "," in after_with:
            names = [n.strip().title() for n in after_with.split(",") if n.strip()]
        else:
            names = [name.strip().title() for name in after_with.split() if name.strip(",").isalpha()]
        return names
    return ["Megha Shyam", "Krishna"]

def extract_title(user_message):
    user_message = user_message.lower()
    for kw in ["about", "for"]:
        if kw in user_message:
            title = user_message.split(kw, 1)[1].strip().title()
            return title[:50]
    return "Project Discussion"

def extract_date(user_message):
    msg = user_message.lower()
    if "tomorrow" in msg:
        dt = datetime.today() + timedelta(days=1)
        return dt.strftime("%Y-%m-%d")
    return datetime.today().strftime("%Y-%m-%d")

def agent_handle_request(user_message):
    msg_lower = user_message.lower().strip()
    if "set my preferences" in msg_lower:
        words = msg_lower.split()
        duration = None
        work_start = None
        work_end = None
        for w in words:
            if "minute" in w:
                try:
                    duration = int(w.split("-")[0])
                except:
                    pass
            if ":" in w:
                if work_start is None:
                    work_start = w
                else:
                    work_end = w
        update_preferences(duration_minutes=duration, work_start=work_start, work_end=work_end)
        return f"Preferences updated: duration {duration} minutes, working hours {work_start}–{work_end}."
    if "list my meetings" in msg_lower or "show" in msg_lower:
        name = None
        for possible in ["megha shyam", "krishna"]:
            if possible in msg_lower:
                name = possible.title()
        if name is None:
            name = "Megha Shyam"
        return list_meetings(name)
    if "schedule" in msg_lower:
        date_str = extract_date(user_message)
        participants = extract_participants(user_message)
        title = extract_title(user_message)
        start_dt, end_dt = find_free_slot(date_str, participants)
        if start_dt is None or end_dt is None:
            return "No free slot available for all participants."
        meeting_id = schedule_meeting(date_str, start_dt, end_dt, title, participants)
        return f"Scheduled '{title}' on {date_str} from {start_dt.strftime('%H:%M')} to {end_dt.strftime('%H:%M')} (id={meeting_id})."
    if "cancel" in msg_lower:
        words = msg_lower.split()
        meeting_id = None
        for w in words:
            if w.isdigit():
                meeting_id = int(w)
                break
        if meeting_id:
            success = cancel_meeting(meeting_id)
            if success:
                return f"Cancelled meeting {meeting_id}."
            else:
                return f"No meeting found with id {meeting_id}."
        else:
            return "Please specify the meeting id to cancel."
    return "Sorry, I didn't understand. Try: 'schedule', 'list my meetings Megha Shyam', 'cancel meeting 1', or 'set my preferences'."

print(agent_handle_request("Schedule a meeting with Krishna, Megha Shyam today about Team Update"))
print(agent_handle_request("Schedule a meeting with Megha Shyam, Krishna tomorrow about Planning"))
print(agent_handle_request("List my meetings Megha Shyam"))
print(agent_handle_request("Cancel meeting 1"))
print(agent_handle_request("Schedule a meeting with Megha Shyam today about Conflict Test"))
print(agent_handle_request("List my meetings Krishna"))

