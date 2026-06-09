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


# =========================
# 0. Imports & Setup
# =========================

import json
import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

# from adk import Agent, Tool, Orchestrator  # <- adapt to actual ADK imports

# -------------------------
# Global in-memory stores (for demo)
# -------------------------

USER_PROFILES: Dict[str, Dict[str, Any]] = {}
SYMPTOM_SESSIONS: Dict[str, List[Dict[str, Any]]] = {}
REMINDERS: List[Dict[str, Any]] = []

# Mock data, you can load from CSV/JSON instead
FACILITIES = [
    {
        "state": "Bihar",
        "district": "Gaya",
        "name": "Gaya District Hospital",
        "type": "District Hospital",
        "services": ["24x7 emergency", "maternal", "child"],
    },
    {
        "state": "Bihar",
        "district": "Gaya",
        "name": "Amas PHC",
        "type": "PHC",
        "services": ["basic OPD", "maternal"],
    },
    # add more...
]

RED_FLAGS = [
    "difficulty breathing",
    "severe chest pain",
    "cannot wake up",
    "confusion",
    "severe bleeding",
    "convulsions",
    "fits",
    "pregnant and heavy bleeding",
    "pregnant and severe headache",
]

HEALTH_TOPICS = {
    "fever": {
        "summary": "Fever is a sign that the body is fighting an infection. It can cause weakness and dehydration.",
        "advice": [
            "Drink plenty of clean water or ORS.",
            "Wear light, comfortable clothes.",
            "Avoid taking unknown medicines without a doctor's advice.",
        ],
    },
    "diarrhea": {
        "summary": "Diarrhea can quickly lead to dehydration, especially in children.",
        "advice": [
            "Give small sips of ORS frequently.",
            "Watch for blood in stool or very dry mouth and eyes.",
        ],
    },
    # add topics as needed
}



# =========================
# 1. Tool Definitions
# =========================

# In ADK you’ll wrap these with @tool decorators


def load_user_profile(user_id: str) -> Dict[str, Any]:
    return USER_PROFILES.get(user_id, {})


def save_user_profile(user_id: str, profile: Dict[str, Any]) -> str:
    USER_PROFILES[user_id] = profile
    return "profile_saved"


def save_symptom_session(user_id: str, session: Dict[str, Any]) -> str:
    SYMPTOM_SESSIONS.setdefault(user_id, []).append(session)
    return "session_saved"


def find_facilities(state: str, district: str, need_type: str = "emergency") -> List[Dict[str, Any]]:
    # Simple filtering logic, can be smarter
    matches = []
    for f in FACILITIES:
        if f["state"].lower() == state.lower() and f["district"].lower() == district.lower():
            matches.append(f)
    return matches


def save_reminder(user_id: str, message: str, hours_from_now: int) -> str:
    due_time = datetime.datetime.utcnow() + datetime.timedelta(hours=hours_from_now)
    REMINDERS.append(
        {
            "user_id": user_id,
            "message": message,
            "due_utc": due_time.isoformat(),
        }
    )
    return "reminder_saved"



# =========================
# 2. Symptom Risk Helper
# =========================

def classify_risk(symptom_text: str) -> Dict[str, Any]:
    text_lower = symptom_text.lower()
    red_flags_detected = [rf for rf in RED_FLAGS if rf in text_lower]

    if red_flags_detected:
        risk_level = "EMERGENCY"
    elif "fever" in text_lower and "3 days" in text_lower:
        risk_level = "URGENT"
    else:
        risk_level = "NON_URGENT"

    return {
        "symptom_summary": symptom_text,
        "red_flags_detected": red_flags_detected,
        "risk_level": risk_level,
    }



# =========================
# 3. Example System Prompts
# =========================

COMMUNITY_PROFILE_SYSTEM_PROMPT = """
You are a Community Profile Agent for a rural health navigator.
Ask short, simple questions to understand:
- user's state and district
- approximate village
- preferred language (e.g. Hindi or English)
- age and gender if relevant to symptoms

Return a concise JSON-style summary and use tools to save the profile.
"""

SYMPTOM_INTAKE_SYSTEM_PROMPT = """
You are a Symptom Intake & Safety Agent.
Your goal is to understand the main symptoms and check for danger signs.
Ask a few clear, short questions if needed.
DO NOT attempt to diagnose or name any disease.
DO NOT suggest specific medicines.
Be conservative: if symptoms sound serious or unclear, mark as EMERGENCY.

Use the `classify_risk` helper (wrapped as a tool) to decide risk_level.
Return a JSON with fields: symptom_summary, red_flags_detected, risk_level, advice_brief.
"""

HEALTH_EDU_SYSTEM_PROMPT = """
You are a Health Education Agent for rural users.
Based on a symptom summary and risk level, explain in very simple language what might be happening in general.
Use generic health topics only; do not name diseases or give drug doses.
Always remind the user:
'I am not a doctor. For serious or lasting problems, please see a qualified health worker.'
"""

FACILITY_FINDER_SYSTEM_PROMPT = """
You are a Facility Finder Agent.
Using the `find_facilities` tool, suggest 1–3 facilities in the user's state and district.
Indicate which facility is more suitable for emergencies and which for routine checkups.
If facilities are unknown, say so honestly and suggest contacting the nearest known government hospital.
"""

FOLLOWUP_SYSTEM_PROMPT = """
You are a Follow-up & Reminder Agent.
Help the user decide when to check symptoms again or re-visit a health worker.
Use the reminder tool to store a reminder with a human-readable message.
Keep language short and encouraging.
"""

ORCHESTRATOR_SYSTEM_PROMPT = """
You are the Rural Health Navigator.
You coordinate other agents and tools.
High-level steps:
1. Ensure you know the user's location and language (Community Profile Agent).
2. Ask about symptoms and risk (Symptom Intake & Safety Agent).
3. Provide explanation (Health Education Agent).
4. Suggest nearby facilities (Facility Finder Agent).
5. Offer a follow-up reminder (Follow-up Agent).
Always be clear that you cannot diagnose or replace a doctor.
"""



# =========================
# 4. Orchestrator Pseudo-flow
# =========================

def run_rural_health_navigator(user_id: str, user_message: str) -> str:
    # 1. Load profile or ask for it via Community Profile Agent (simplified here)
    profile = load_user_profile(user_id)
    if not profile:
        # In real ADK, you'd call the Community Profile Agent to interact.
        profile = {
            "state": "Bihar",
            "district": "Gaya",
            "language": "en",
        }
        save_user_profile(user_id, profile)

    # 2. Symptom classification
    risk_struct = classify_risk(user_message)
    save_symptom_session(user_id, risk_struct)

    # 3. Build simple health education text from HEALTH_TOPICS
    edu_parts = []
    text_lower = user_message.lower()
    if "fever" in text_lower:
        topic = HEALTH_TOPICS["fever"]
        edu_parts.append(topic["summary"])
        edu_parts.extend(topic["advice"])

    edu_text = "\n- ".join(edu_parts) if edu_parts else "Take rest, drink clean fluids, and monitor symptoms."

    # 4. Facility suggestions
    facilities = find_facilities(profile["state"], profile["district"], need_type=risk_struct["risk_level"])
    fac_lines = []
    for f in facilities:
        fac_lines.append(f"- {f['name']} ({f['type']}) – services: {', '.join(f['services'])}")

    facilities_text = "\n".join(fac_lines) if fac_lines else "No facility data is available for your area in this demo."

    # 5. Compose final message (in real ADK, this comes from Orchestrator Agent LLM)
    response = f"""
Here is a safety-focused summary based on what you told me:

1) Symptom & risk (not a diagnosis):
- Summary: {risk_struct['symptom_summary']}
- Red flags noticed: {', '.join(risk_struct['red_flags_detected']) or 'none detected'}
- Urgency (cautious estimate): {risk_struct['risk_level']}

2) General information and self-care:
- {edu_text}

3) Nearby facilities (for real medical help):
{facilities_text}

4) Important:
I am not a doctor or nurse. I can only give general guidance.
If the person becomes worse, very drowsy, confused, has trouble breathing, or you are worried,
please go to the nearest health facility or call emergency services immediately.
"""
    return response



print(
    run_rural_health_navigator(
        user_id="demo_user_1",
        user_message="My mother has high fever for 3 days and vomiting, she is very weak and sometimes confused."
    )
)


