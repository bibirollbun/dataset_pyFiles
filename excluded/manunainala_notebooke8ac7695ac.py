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


# AI Health Navigator - Agents for good

import re
import json
import uuid
import os
from datetime import datetime, timedelta
from pprint import pprint

# ------------------- Memory / Persistence -------------------
MEMORY_FILE = "/kaggle/working/health_navigator_memory.json"

# Ensure directory exists
os.makedirs("/kaggle/working/", exist_ok=True)

# Create file if missing
if not os.path.exists(MEMORY_FILE):
    MEMORY = {"sessions": {}, "patients": {}}
    with open(MEMORY_FILE, "w") as f:
        json.dump(MEMORY, f, indent=2)
else:
    with open(MEMORY_FILE, "r") as f:
        MEMORY = json.load(f)

def save_memory():
    with open(MEMORY_FILE, "w") as f:
        json.dump(MEMORY, f, indent=2)

def new_session(user_id=None):
    sess_id = str(uuid.uuid4())[:8]
    MEMORY["sessions"][sess_id] = {
        "id": sess_id,
        "user_id": user_id or f"user_{sess_id}",
        "created_at": datetime.utcnow().isoformat(),
        "history": [],
        "triage": None
    }
    save_memory()
    return MEMORY["sessions"][sess_id]

# ------------------- Symptom Parser Agent -------------------
def symptom_parser_agent(text):
    text_l = text.lower()
    
    # Age extraction
    age = None
    m = re.search(r'(\b[0-9]{1,3})\s*(?:year|yr|y)\b', text_l)
    if m:
        age = int(m.group(1))

    # Duration extraction
    duration = None
    m = re.search(r'(\d+)\s*(day|days|hour|hours|hr|hrs)', text_l)
    if m:
        val = int(m.group(1))
        unit = m.group(2)
        duration = val * 24 if "day" in unit else val

    # Severity estimation
    severity = 3
    if any(w in text_l for w in ["severe", "extreme", "worst"]):
        severity = 8
    elif any(w in text_l for w in ["high", "bad", "strong"]):
        severity = 6
    elif any(w in text_l for w in ["mild", "light", "slight"]):
        severity = 2

    # Symptom detection
    symptoms = []
    keywords = ["fever","cough","breathing","pain","headache","vomiting","bleeding","cut","chest","dizzy","faint","unconscious","rash","stomach"]
    for k in keywords:
        if k in text_l:
            symptoms.append(k)

    # Red flags
    red_flags = {
        "breathing_difficulty": "difficulty breathing" in text_l or "hard to breathe" in text_l,
        "severe_bleeding": "severe bleeding" in text_l or "bleeding heavily" in text_l,
        "unconscious": "unconscious" in text_l or "fainted" in text_l,
        "chest_pain": "chest pain" in text_l
    }

    return {
        "raw_text": text,
        "chief_complaint": text[:100],
        "duration_hours": duration,
        "severity_est": severity,
        "symptoms_list": symptoms,
        "vital_red_flags": red_flags,
        "age": age,
        "timestamp": datetime.utcnow().isoformat()
    }

# ------------------- Triage Agent -------------------
def triage_agent(symptom_json):
    rf = symptom_json["vital_red_flags"]
    severity = symptom_json["severity_est"]
    duration = symptom_json["duration_hours"]

    if any(rf.values()):
        return {
            "level": "High",
            "reason": f"Emergency signs detected: {rf}",
            "actions": ["Call emergency services immediately.", "Go to the nearest hospital."],
            "explainability": "Red flag triggered"
        }

    if severity >= 7:
        level = "High"
    elif severity >= 5:
        level = "Moderate"
    else:
        level = "Low"

    if duration and duration > 7*24 and level == "Low":
        level = "Moderate"

    actions = {
        "High": ["Immediate hospital care required."],
        "Moderate": ["Consult a doctor within 24–48 hours.", "Monitor symptoms closely."],
        "Low": ["Home care recommended.", "Re-check if symptoms worsen."]
    }

    return {
        "level": level,
        "reason": "Based on severity & duration",
        "actions": actions[level],
        "explainability": f"Severity={severity}, Duration={duration}"
    }

# ------------------- Healthcare Locator Agent -------------------
MOCK_CLINICS = [
    {"name": "Primary Health Center A", "distance_km": 5, "open_24x7": False},
    {"name": "Community Clinic B", "distance_km": 12, "open_24x7": False},
    {"name": "Emergency Hospital C", "distance_km": 30, "open_24x7": True},
]

def healthcare_locator_agent(triage_level):
    if triage_level == "High":
        return [c for c in MOCK_CLINICS if c["open_24x7"]]
    return MOCK_CLINICS

# ------------------- Follow-Up Agent -------------------
def followup_agent(session_id, hours=24):
    sess = MEMORY["sessions"][session_id]
    next_time = (datetime.utcnow() + timedelta(hours=hours)).isoformat()
    sess["followup_time"] = next_time
    save_memory()
    return {"followup_scheduled_for": next_time}

# ------------------- Orchestrator -------------------
def health_navigator_agent(text):
    session = new_session()

    parsed = symptom_parser_agent(text)
    triage = triage_agent(parsed)
    clinics = healthcare_locator_agent(triage["level"])

    followup = None
    if triage["level"] != "High":
        followup = followup_agent(session["id"], 12 if triage["level"]=="Low" else 24)

    return {
        "session_id": session["id"],
        "parsed_symptoms": parsed,
        "triage": triage,
        "clinics": clinics,
        "followup": followup
    }

# ------------------- DEMO RUN -------------------
demo_cases = [
    "My father has chest pain and is finding it hard to breathe.",
    "My 6 year old has fever and rash for 2 days.",
    "I have mild headache and little fever."
]

for case in demo_cases:
    print("\n" + "="*70)
    print("USER:", case)
    pprint(health_navigator_agent(case))

print("\nMemory stored at:", MEMORY_FILE)


