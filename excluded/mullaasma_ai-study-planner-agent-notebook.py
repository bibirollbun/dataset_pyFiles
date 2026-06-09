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


# AI Study Planner Agent (Kaggle Notebook)
#*Capstone (Google 5-Day AI Agents format)*

#Run cells in order.  
#This notebook:
#1. Loads sample input (or sample_inputs.json if uploaded).
#2. Computes subject priority scores.
#3. Allocates total hours across the schedule.
#4. Generates a day-by-day plan with weekly reviews and final revision days.
#5. Exports results to study_plan_output.json and study_plan_output.csv.


# Cell 2
import json
import csv
from datetime import datetime, timedelta
from pathlib import Path

print("Imports OK")


# Cell 3
# If you uploaded sample_inputs.json to Kaggle notebook (via + Add data or Upload),
# this code will try to load it. Otherwise it uses the built-in sample.
INPUT_FILE = "sample_inputs.json"

def load_input(file_path=INPUT_FILE):
    p = Path(file_path)
    if p.exists():
        with open(p, "r") as f:
            data = json.load(f)
        # support two formats: dict of users or single user object
        if isinstance(data, dict) and ("user1" in data or list(data.keys())[0].startswith("user")):
            # pick the first user entry
            first_key = list(data.keys())[0]
            return data[first_key]
        return data
    else:
        # fallback sample
        return {
            "name": "Asma",
            "subjects": [
                {"name": "Data Structures", "weight": 40, "difficulty": "hard", "current_level": 30},
                {"name": "Digital Electronics", "weight": 20, "difficulty": "medium", "current_level": 50},
                {"name": "Physics", "weight": 20, "difficulty": "medium", "current_level": 60},
                {"name": "English", "weight": 20, "difficulty": "easy", "current_level": 80}
            ],
            "hours_per_day": 4,
            "days_until_deadline": 30,
            "revision_days": 4
        }

input_data = load_input()
print("Loaded input for:", input_data.get("name", "Unknown"))
print("Subjects:", [s["name"] for s in input_data["subjects"]])


# Cell 4
def compute_priority_scores(subjects):
    # difficulty multipliers: tunable
    difficulty_factor = {"easy": 1.0, "medium": 1.2, "hard": 1.4}
    scores = []
    for s in subjects:
        need = max(0, 100 - s.get("current_level", 50))  # gap to 100%
        diff = difficulty_factor.get(s.get("difficulty","medium"), 1.2)
        weight = s.get("weight", 10)
        score = weight * need * diff
        scores.append(round(score, 2))
    return scores

def allocate_hours(subjects, hours_per_day, days):
    scores = compute_priority_scores(subjects)
    total_score = sum(scores) or 1.0
    allocation = []
    for s, sc in zip(subjects, scores):
        frac = sc / total_score
        hours_total = frac * hours_per_day * days
        allocation.append({
            "name": s["name"],
            "hours_total": round(hours_total, 1),
            "priority_score": sc,
            "difficulty": s.get("difficulty","medium"),
            "current_level": s.get("current_level", 0),
            "weight": s.get("weight", 0)
        })
    return allocation

def generate_daily_plan(allocation, days, revision_days, max_tasks_per_day=6, min_block_hr=0.25):
    # build per-day plan
    per_day = []
    for day in range(1, days+1):
        tasks = []
        for sub in allocation:
            per_day_hours = sub["hours_total"] / days
            # ensure a readable minimum chunk
            per_day_hours = max(per_day_hours, min_block_hr)
            # produce a readable label
            label = f"{sub['name']}: ~{round(per_day_hours,1)} hr"
            tasks.append(label)
        # weekly review checkpoint
        if day % 7 == 0:
            tasks.append("Weekly review: Solve 10 practice questions + quick revision")
        # limit tasks shown (keeps each day sane)
        per_day.append({"day": day, "tasks": tasks[:max_tasks_per_day]})
    # final revision days override
    for i in range(revision_days):
        idx = days - 1 - i
        if idx >= 0:
            per_day[idx]["tasks"] = ["Revision: High-yield topics + mock test (2 hrs)"]
    return per_day

# convenience wrapper that uses input_data
def run_agent(input_obj):
    subjects = input_obj["subjects"]
    days = int(input_obj["days_until_deadline"])
    hpd = float(input_obj["hours_per_day"])
    rev_days = int(input_obj.get("revision_days", 3))
    allocation = allocate_hours(subjects, hpd, days)
    plan = generate_daily_plan(allocation, days, rev_days)
    return allocation, plan

print("Agent logic loaded")


# Cell 5
allocation, plan = run_agent(input_data)

print("=== Hour Allocation (total hours across the period) ===")
for a in allocation:
    print(f"- {a['name']}: {a['hours_total']} hrs  (score={a['priority_score']}, difficulty={a['difficulty']})")

print("\n=== Plan preview (first 7 days) ===")
for day in plan[:7]:
    print(f"\nDay {day['day']}:")
    for t in day['tasks']:
        print("  -", t)


# Cell 6
output = {
    "metadata": {
        "name": input_data.get("name"),
        "generated_on": datetime.utcnow().isoformat() + "Z",
        "days": input_data.get("days_until_deadline"),
        "hours_per_day": input_data.get("hours_per_day")
    },
    "allocation": allocation,
    "plan_preview": plan[:min(30, len(plan))]
}

# JSON
with open("study_plan_output.json", "w") as f:
    json.dump(output, f, indent=2)
print("Saved study_plan_output.json")

# CSV: flat per-day view (day, tasks combined)
csv_file = "study_plan_output.csv"
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["day", "tasks"])
    for d in plan:
        writer.writerow([d["day"], " | ".join(d["tasks"])])
print("Saved", csv_file)


#Validation checklist:
#- study_plan_output.json exists and contains allocation + plan_preview.
#- study_plan_output.csv exists for easy preview.
#- Take screenshots of the notebook outputs (Hour Allocation + Plan preview) and add them to the Kaggle Write-Up gallery.
#- Download the notebook (.ipynb) and upload it to your GitHub repo under AI-Study-Planner-Agent folder.
#- Add README.md (you already did) and link to this notebook in README.
#
#Next steps available on demand:
#- Replace rule-based logic with LLM prompt (Gemini/Vertex/OpenAI) — requires internet/API keys.
#- Build a small web UI to display plan dynamically.

