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


# ADK is not available on Kaggle. We will use a simulated ADK architecture.
# This still demonstrates agent concepts required for the capstone.
print("Using simulated ADK (google-adk not available on Kaggle).")



# REAL_MODE disabled because ADK cannot be installed on Kaggle.
REAL_MODE = False
print("REAL_MODE set to False. Running in simulated agent mode.")



def calculate_bmi(weight_kg: float, height_m: float) -> dict:
    """Return BMI and category."""
    bmi = weight_kg / (height_m * height_m)
    if bmi < 18.5:
        category = "underweight"
    elif bmi < 25:
        category = "normal"
    elif bmi < 30:
        category = "overweight"
    else:
        category = "obese"
    return {"bmi": round(bmi, 2), "category": category}

tool_registry = {"calculate_bmi": calculate_bmi}

print("Tool registered:", list(tool_registry.keys()))



import datetime

class SimulatedSessionService:
    def __init__(self):
        self.sessions = {}

    def create_session(self, session_id="default"):
        self.sessions[session_id] = {
            "history": [],
            "state": {}
        }

    def append_event(self, session_id, event):
        self.sessions[session_id]["history"].append(event)

    def get_history(self, session_id):
        return self.sessions[session_id]["history"]

sim_session = SimulatedSessionService()
sim_session.create_session("default")

print("Session service initialized.")



class SimulatedAgent:
    def __init__(self, name="helper_agent"):
        self.name = name

    def run(self, user_input: str, tools: dict, session_id="default"):
        sim_session.append_event(session_id, {
            "role": "user",
            "text": user_input,
            "time": str(datetime.datetime.utcnow())
        })

        text = user_input.lower()

        # TOOL: BMI
        if "bmi" in text:
            import re
            pattern = r"weight\s*=?\s*(\d+(?:\.\d+)?)\s*.*height\s*=?\s*(\d+(?:\.\d+)?)"
            match = re.search(pattern, user_input)
            if match:
                w = float(match.group(1))
                h = float(match.group(2))
                out = tools["calculate_bmi"](w, h)
                sim_session.append_event(session_id, {
                    "role": "agent",
                    "text": str(out)
                })
                return {"result": out, "used_tool": "calculate_bmi"}
            else:
                return {"error": "To calculate BMI use: 'Calculate BMI weight=70 height=1.75'"}

        # SUMMARIZER
        if "summarize" in text:
            cleaned = user_input.replace("summarize", "").strip()
            summary = cleaned[:200] + ("..." if len(cleaned) > 200 else "")
            sim_session.append_event(session_id, {
                "role": "agent",
                "text": summary
            })
            return {"result": summary}

        # STUDY PLAN
        if "study plan" in text or "study" in text:
            plan = {
                "days": 7,
                "daily_hours": 2,
                "tasks": ["Revise notes", "Solve questions", "Mock tests"]
            }
            sim_session.append_event(session_id, {
                "role": "agent",
                "text": str(plan)
            })
            return {"result": plan}

        # DEFAULT ACTION
        reply = "I can summarize text, create study plans, or calculate BMI."
        sim_session.append_event(session_id, {
            "role": "agent",
            "text": reply
        })
        return {"result": reply}

sim_agent = SimulatedAgent()
print("Simulated agent ready.")



def ask_agent(text, session_id="default"):
    return sim_agent.run(text, tool_registry, session_id=session_id)



tests = [
    "Calculate BMI weight=70 height=1.75",
    "Summarize: Python is a popular programming language used for AI, ML and web development.",
    "Create a study plan for my compiler design exam."
]

for q in tests:
    print("USER:", q)
    print("AGENT:", ask_agent(q))
    print("-----")



sim_session.get_history("default")


