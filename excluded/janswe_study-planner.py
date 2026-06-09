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




import math
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import textwrap
import os
import json


USE_LLM = False
try:
    import google.generativeai as genai
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY_HERE")
    if api_key:
        genai.configure(api_key=api_key)
        # using model name likely available on Kaggle; adjust if needed
        llm = genai.GenerativeModel("gemini-pro")
        USE_LLM = True
    else:
        USE_LLM = False
except Exception as e:
    
    USE_LLM = False


def wrap(text, width=80):
    return "\n".join(textwrap.wrap(text, width=width))

def format_time(dt):
    return dt.strftime("%Y-%m-%d %H:%M")


def estimate_topic_hours(topics):
    """
    If user supplies no estimates, we estimate based on length of topic name.
    topics: list of dicts or strings.
    Returns list of (topic, est_hours)
    """
    out = []
    for t in topics:
        if isinstance(t, dict):
            name = t.get("name")
            est = t.get("hours")
            if est is None:
                est = max(1, int(len(name.split())/2) + 1)
        else:
            name = str(t)
            est = max(1, int(len(name.split())/2) + 1)
        out.append({"topic": name, "hours": est})
    return out

def expand_topics(topics_with_hours):
    """
    Convert topic hours into study 'units' (1 unit = 1 hour)
    Returns list of units: [{'topic':..., 'unit_id':i, 'hours':1}]
    """
    units = []
    for t in topics_with_hours:
        for i in range(int(t["hours"])):
            units.append({"topic": t["topic"], "unit_hours": 1})
    return units

def schedule_units(units, start_date, days_available, hours_per_day, preferred_start_hour=9):
    """
    Basic scheduler:
    - Distribute units across days evenly (respecting hours_per_day)
    - Each 'unit' is one hour block. We allocate Pomodoro cycles (25+5) optionally in presentation.
    Returns schedule dict: date -> list of blocks
    """
    total_hours = len(units)
    capacity = days_available * hours_per_day
    if capacity < total_hours:

        hours_per_day = math.ceil(total_hours / days_available)
        capacity = days_available * hours_per_day

    schedule = {}
    unit_idx = 0
    for day in range(days_available):
        date = (start_date + timedelta(days=day)).date()
        schedule[str(date)] = []
        for h in range(hours_per_day):
            if unit_idx >= total_hours:
                break
            block_start = datetime.combine(date, datetime.min.time()) + timedelta(hours=preferred_start_hour + h)
            unit = units[unit_idx]

            schedule[str(date)].append({
                "start_time": format_time(block_start),
                "duration_hours": 1,
                "topic": unit["topic"],
                "pomodoro": {"work_min":25, "break_min":5, "cycles":2}  # example
            })
            unit_idx += 1
       
    return schedule

def generate_revision_sessions(schedule, review_days=[1,3,7,30]):
    """
    Generate spaced revision sessions for each scheduled block.
    For every block, create review tasks at +review_days.
    Returns list of reviews (date, topic, origin_date)
    """
    reviews = []
    for day_str, blocks in schedule.items():
        origin_date = datetime.strptime(day_str, "%Y-%m-%d").date()
        for b in blocks:
            for rd in review_days:
                review_date = origin_date + timedelta(days=rd)
                reviews.append({
                    "review_date": str(review_date),
                    "topic": b["topic"],
                    "origin_date": str(origin_date),
                    "type": f"revision+{rd}d"
                })
    
    return reviews

def compact_schedule_to_df(schedule, reviews=None):
    rows = []
    for day_str, blocks in schedule.items():
        for b in blocks:
            rows.append({
                "date": day_str,
                "start_time": b["start_time"],
                "duration_hours": b["duration_hours"],
                "topic": b["topic"],
                "type": "study",
                "pomodoro": json.dumps(b.get("pomodoro", {}))
            })
    if reviews:
        for r in reviews:
            rows.append({
                "date": r["review_date"],
                "start_time": f"{r['review_date']} 18:00", 
                "duration_hours": 0.5,
                "topic": r["topic"],
                "type": r["type"],
                "pomodoro": ""
            })
    df = pd.DataFrame(rows)
    df = df.sort_values(["date","start_time"]).reset_index(drop=True)
    return df


def create_study_plan(
    subject_list,
    start_date_str=None,
    days_available=14,
    hours_per_day=3,
    preferred_start_hour=9,
    include_revision=True,
    use_llm_for_polish=False
):
    """
    subject_list: list of strings or dicts {"name":..., "hours":...}
    start_date_str: "YYYY-MM-DD" or None for today
    days_available: int
    hours_per_day: int
    returns: dict with schedule, reviews, dataframe
    """
    if start_date_str:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    else:
        start_date = datetime.now()

    
    topics_with_hours = estimate_topic_hours(subject_list)
    units = expand_topics(topics_with_hours)
    schedule = schedule_units(units, start_date, days_available, hours_per_day, preferred_start_hour)

  
    reviews = generate_revision_sessions(schedule) if include_revision else []

   
    df = compact_schedule_to_df(schedule, reviews)


    summary = {
        "total_topics": len(topics_with_hours),
        "total_hours": len(units),
        "days_available": days_available,
        "hours_per_day": hours_per_day,
        "start_date": start_date.strftime("%Y-%m-%d")
    }

 
    polish_text = None
    if use_llm_for_polish and USE_LLM:
      
        prompt = f"Create a motivating summary for a student following this study plan starting {summary['start_date']} for {days_available} days, studying {hours_per_day} hours per day. Topics: " + \
                 ", ".join([f"{t['topic']}({t['hours']}h)" for t in topics_with_hours])
        try:
            res = llm.generate_content(prompt)
            polish_text = res.text
        except Exception as e:
            polish_text = f"(LLM failed: {e})"

    return {"schedule": schedule, "reviews": reviews, "df": df, "summary": summary, "polish": polish_text}


    
def print_summary(plan):
    s = plan["summary"]
    print("=== Study Plan Summary ===")
    print(f"Start date: {s['start_date']}")
    print(f"Days available: {s['days_available']}")
    print(f"Hours per day: {s['hours_per_day'] if 'hours_per_day' in s else 'N/A'}")
    print(f"Total topics: {s['total_topics']}")
    print(f"Estimated total hours: {s['total_hours']}")
    if plan.get("polish"):
        print("\n--- Motivational Note (LLM-polish) ---")
        print(wrap(plan["polish"]))
    print("\n")

def show_day(plan_df, date_str):
    subset = plan_df[plan_df["date"] == date_str]
    if subset.empty:
        print(f"No tasks scheduled for {date_str}")
        return
    print(f"Tasks for {date_str}:")
    for i, row in subset.iterrows():
        print(f" - {row['start_time'].split()[1]} | {row['topic']} | {row['type']} | {row['duration_hours']}h")
    print("")

def export_plan_csv(plan_df, filename="study_plan.csv"):
    plan_df.to_csv(filename, index=False)
    print(f"Exported plan to {filename}")


if __name__ == "__main__":
   
    subjects = [
        {"name":"DBMS - Normalization & ER Models", "hours":4},
        {"name":"Python - OOP & Modules", "hours":3},
        {"name":"Data Structures - Arrays & Linked Lists", "hours":4},
        {"name":"Operating Systems - Processes & Scheduling", "hours":3},
        "Probability & Statistics - basics"
    ]

    print("Creating study plan... (demo defaults: start today, 10 days, 3 hours/day)\n")
    plan = create_study_plan(subjects, start_date_str=None, days_available=10, hours_per_day=3, preferred_start_hour=9, include_revision=True, use_llm_for_polish=False)

    df = plan["df"]
    display_df = df.head(40).copy()
    
    print("First 20 scheduled tasks (study + revision):\n")
    print(display_df.to_string(index=False))

    
    export_plan_csv(df, "community_study_plan.csv")

  
    some_date = df.iloc[0]["date"]
    print("\nExample day view:\n")
    show_day(df, some_date)

    with open("study_plan_summary.json","w") as f:
        json.dump(plan["summary"], f)
    print("\nPlan summary saved to study_plan_summary.json")


