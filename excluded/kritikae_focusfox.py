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


# Sample Input for the Study Plan Intelligence Agent

inputs = {
    "start_date": "2025-11-25",
    "hours_per_day": 3,
    
    "subjects": [
        {
            "name": "Machine Learning",
            "estimated_hours": 20,
            "difficulty": 4,
            "exam_date": "2025-12-10"
        },
        {
            "name": "Statistics",
            "estimated_hours": 12,
            "difficulty": 3,
            "exam_date": "2025-12-05"
        }
    ],
    
    "preferences": {
        "session_length": 50,          # minutes
        "revision_ratio": 0.2          # 20% time for revision
    },
    
    "blackout_days": ["2025-11-29"]
}

inputs



from datetime import datetime, timedelta

# Helper function: generate a list of dates
def generate_date_range(start_date, end_date):
    """
    Returns a list of dates (YYYY-MM-DD strings) 
    from start_date to end_date inclusive.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    dates = []
    current = start
    
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    
    return dates

# Test: get date range from start date to latest exam date
latest_exam = max(sub["exam_date"] for sub in inputs["subjects"])
date_range = generate_date_range(inputs["start_date"], latest_exam)

date_range[:10]   # show the first 10 dates



# Prepare subject data for time allocation
import copy

def prepare_subjects(subjects):
    """
    Creates a clean, editable copy of subjects.
    Adds a 'remaining_hours' field for allocation.
    """
    processed = []
    
    for sub in subjects:
        item = copy.deepcopy(sub)
        item["remaining_hours"] = sub["estimated_hours"]
        processed.append(item)
    
    return processed

subjects_data = prepare_subjects(inputs["subjects"])

subjects_data



# Helper: sort subjects by exam date (earliest first)
from datetime import datetime

def sort_by_exam_date(subjects):
    """
    Returns subjects sorted by their exam_date (ascending).
    """
    return sorted(
        subjects,
        key=lambda s: datetime.strptime(s["exam_date"], "%Y-%m-%d")
    )

sorted_subjects = sort_by_exam_date(subjects_data)

sorted_subjects


def allocate_hours_for_day(subjects, available_hours, today):
    """
    Allocates study hours for a single day.
    
    subjects: list of subject dicts (with remaining_hours)
    available_hours: how many hours student can study today
    today: current date (string: YYYY-MM-DD)
    
    Returns:
        allocations: list of (subject_name, hours_assigned)
    """
    
    # 1. Prepare result
    allocations = []
    
    # 2. Sort subjects by exam date (earliest exam gets priority)
    ordered_subjects = sort_by_exam_date(subjects)
    
    # 3. For each subject, try to assign a fair portion
    for sub in ordered_subjects:
        if available_hours <= 0:
            break
        
        # Days left including today
        exam_date = datetime.strptime(sub["exam_date"], "%Y-%m-%d")
        today_date = datetime.strptime(today, "%Y-%m-%d")
        days_left = (exam_date - today_date).days + 1
        
        if days_left <= 0:
            continue  # exam passed
        
        # Daily fair share
        daily_need = sub["remaining_hours"] / days_left
        
        # Amount we can actually assign
        assign = min(daily_need, available_hours, sub["remaining_hours"])
        
        # Update trackers
        if assign > 0:
            allocations.append({
                "subject": sub["name"],
                "hours": round(assign, 2)
            })
            
            sub["remaining_hours"] -= assign
            available_hours -= assign
    
    return allocations



def build_initial_plan(inputs):
    """
    Creates a full day-by-day study plan.
    Returns a list of daily allocations.
    """
    
    # Extract from inputs
    start_date = inputs["start_date"]
    hours_per_day = inputs["hours_per_day"]
    subjects = prepare_subjects(inputs["subjects"])
    blackout = set(inputs.get("blackout_days", []))
    
    # Determine full date range
    latest_exam = max(sub["exam_date"] for sub in subjects)
    all_dates = generate_date_range(start_date, latest_exam)
    
    # Final plan
    full_plan = []
    
    for day in all_dates:
        
        # 0 hours on blackout days
        if day in blackout:
            full_plan.append({
                "date": day,
                "allocations": [],
                "note": "Blackout day â€“ no study"
            })
            continue
        
        # Default available hours
        available = hours_per_day
        
        # Allocate
        today_alloc = allocate_hours_for_day(subjects, available, day)
        
        # Save the day's plan
        full_plan.append({
            "date": day,
            "allocations": today_alloc
        })
    
    return full_plan

# Build the plan using your sample inputs
initial_plan = build_initial_plan(inputs)

# Show first 5 days
initial_plan[:5]



import pandas as pd

def plan_to_table(plan):
    """
    Converts the plan (list of days) into a clean DataFrame.
    Each row = one subject allocation for one day.
    """
    rows = []
    
    for day in plan:
        date = day["date"]
        
        # Blackout day
        if "note" in day:
            rows.append({
                "date": date,
                "subject": "---",
                "hours": 0,
                "note": day["note"]
            })
            continue
        
        # Normal day
        for alloc in day["allocations"]:
            rows.append({
                "date": date,
                "subject": alloc["subject"],
                "hours": alloc["hours"],
                "note": ""
            })
    
    return pd.DataFrame(rows)

# Convert your plan to a table
plan_df = plan_to_table(initial_plan)

# Show first 10 rows
plan_df.head(10)



def explain_daily_plan(day_entry):
    """
    Produces a simple natural-language explanation 
    for a single day's plan.
    """
    date = day_entry["date"]
    
    # Blackout day
    if "note" in day_entry:
        return f"{date}: No study scheduled â€” it's a blackout day."
    
    allocations = day_entry["allocations"]
    
    if not allocations:
        return f"{date}: No study assigned today."
    
    # Construct explanation
    parts = [f"{alloc['subject']} ({alloc['hours']} hrs)" 
             for alloc in allocations]
    
    subjects_list = ", ".join(parts)
    
    return f"{date}: Focus on {subjects_list} based on exam priority and remaining syllabus."
    

# Generate explanations for the first few days
for d in initial_plan[:5]:
    print(explain_daily_plan(d))



def generate_initial_plan_report(plan):
    """
    Prints a clear, readable report of the initial study plan
    with explanations for each day.
    """
    print("=== INITIAL STUDY PLAN ===\n")
    
    for day in plan:
        print(explain_daily_plan(day))
        
    print("\n=== END OF PLAN ===")

# Show the full plan
generate_initial_plan_report(initial_plan)



# Memory: Progress Log
progress_log = []

def log_progress(date, subject, planned, completed, mastery=None, notes=""):
    """
    Adds a daily progress entry to the memory.
    """
    entry = {
        "date": date,
        "subject": subject,
        "planned_hours": planned,
        "completed_hours": completed,
        "mastery_score": mastery,
        "notes": notes
    }
    progress_log.append(entry)



# Simulated student progress for demo purposes

log_progress("2025-11-25", "Statistics", 1.67, 1.67, mastery=80)
log_progress("2025-11-25", "Machine Learning", 1.33, 0.8, mastery=40, notes="Found regression difficult")

log_progress("2025-11-26", "Statistics", 1.67, 1.0, mastery=60)   # studied less
log_progress("2025-11-26", "Machine Learning", 1.33, 1.33, mastery=70)

progress_log[:4]


from collections import defaultdict

def calculate_risk(subjects, progress_log, today):
    """
    Calculates a risk score for each subject.
    """
    # Step 1 â€” Aggregate progress
    completed = defaultdict(float)
    planned = defaultdict(float)
    mastery = defaultdict(list)
    
    for entry in progress_log:
        sub = entry["subject"]
        planned[sub] += entry["planned_hours"]
        completed[sub] += entry["completed_hours"]
        if entry["mastery_score"] is not None:
            mastery[sub].append(entry["mastery_score"])
    
    risk_scores = {}
    
    for sub in subjects:
        name = sub["name"]
        
        # Completion ratio
        if planned[name] > 0:
            completion_ratio = completed[name] / planned[name]
        else:
            completion_ratio = 1.0   # no data yet, assume fine
        
        # Mastery ratio
        if mastery[name]:
            mastery_ratio = sum(mastery[name]) / (len(mastery[name]) * 100)
        else:
            mastery_ratio = 1.0   # default to safe if no mastery data
        
        # Time pressure
        exam_date = datetime.strptime(sub["exam_date"], "%Y-%m-%d")
        today_dt = datetime.strptime(today, "%Y-%m-%d")
        
        days_left = max((exam_date - today_dt).days, 1)
        time_pressure = sub["remaining_hours"] / days_left
        
        # Weighted risk (safe, balanced weights)
        w1, w2, w3 = 0.4, 0.4, 0.2
        
        risk = (
            w1 * (1 - completion_ratio) +
            w2 * (1 - mastery_ratio) +
            w3 * time_pressure
        )
        
        risk_scores[name] = round(risk, 3)
    
    return risk_scores



today = "2025-11-27"   # assume adaptation happens today

risk_scores = calculate_risk(subjects_data, progress_log, today)
risk_scores



def adapt_tomorrows_plan(original_day_plan, risk_scores, max_hours):
    """
    Adjusts the allocations for a single day based on risk scores.
    Keeps total hours <= max_hours.
    """
    if "note" in original_day_plan:
        return original_day_plan  # blackout stays blackout
    
    allocations = original_day_plan["allocations"]
    
    # If nothing planned, return as is
    if not allocations:
        return original_day_plan
    
    adapted = []
    
    # Step 1 â€” Adjust hours based on risk
    for alloc in allocations:
        sub = alloc["subject"]
        hours = alloc["hours"]
        
        risk = risk_scores.get(sub, 0)
        
        if risk > 0.6:
            new_hours = hours * 1.3     # +30%
        elif risk < 0.4:
            new_hours = hours * 0.8     # -20%
        else:
            new_hours = hours           # no change
        
        adapted.append({
            "subject": sub,
            "hours": round(new_hours, 2)
        })
    
    # Step 2 â€” Normalize total hours to max_hours
    total = sum(a["hours"] for a in adapted)
    
    if total > 0:
        scale = max_hours / total
        adapted = [
            {"subject": a["subject"], "hours": round(a["hours"] * scale, 2)}
            for a in adapted
        ]
    
    return {
        "date": original_day_plan["date"],
        "allocations": adapted
    }



# Find tomorrow's entry
tomorrow = "2025-11-27"
tomorrow_plan = None

for day in initial_plan:
    if day["date"] == tomorrow:
        tomorrow_plan = day
        break

adapted_plan = adapt_tomorrows_plan(
    tomorrow_plan, 
    risk_scores, 
    inputs["hours_per_day"]
)

adapted_plan



def explain_adaptation(original, adapted, risk_scores):
    """
    Produces a natural-language explanation 
    of why the plan changed for each subject.
    """
    explanations = []
    orig_map = {a["subject"]: a["hours"] for a in original["allocations"]}
    adap_map = {a["subject"]: a["hours"] for a in adapted["allocations"]}
    
    date = original["date"]
    
    for subject in orig_map:
        old = orig_map[subject]
        new = adap_map.get(subject, old)
        risk = risk_scores.get(subject, 0)
        
        if new > old:
            reason = f"Increased time for {subject} because its risk score is high ({risk})."
        elif new < old:
            reason = f"Reduced time for {subject}; it has lower risk ({risk})."
        else:
            reason = f"{subject} remains unchanged."
        
        explanations.append(reason)
    
    return {
        "date": date,
        "changes": explanations
    }



explain_adaptation(tomorrow_plan, adapted_plan, risk_scores)


def generate_adapted_plan_report(original_plan, adapted_plan, explanation):
    """
    Prints a clear, readable report comparing the 
    original and adapted plan for a specific day.
    """
    date = original_plan["date"]
    
    print(f"=== ADAPTED PLAN FOR {date} ===\n")
    
    print("Original Allocations:")
    for alloc in original_plan["allocations"]:
        print(f" - {alloc['subject']}: {alloc['hours']} hrs")
        
    print("\nAdapted Allocations:")
    for alloc in adapted_plan["allocations"]:
        print(f" - {alloc['subject']}: {alloc['hours']} hrs")
    
    print("\nReasons for Changes:")
    for line in explanation["changes"]:
        print(f" - {line}")
    
    print("\n=== END OF REPORT ===")


# Create explanation for the adapted plan
explanation = explain_adaptation(tomorrow_plan, adapted_plan, risk_scores)

# Generate the full report
generate_adapted_plan_report(tomorrow_plan, adapted_plan, explanation)



def calculate_completion(progress_log):
    """
    Returns the overall completion percentage 
    based on planned vs completed hours.
    """
    total_planned = sum(p["planned_hours"] for p in progress_log)
    total_completed = sum(p["completed_hours"] for p in progress_log)
    
    if total_planned == 0:
        return 0
    
    return round((total_completed / total_planned) * 100, 2)

completion_pct = calculate_completion(progress_log)
completion_pct



def subject_summary(progress_log):
    summary = {}
    
    for entry in progress_log:
        sub = entry["subject"]
        
        if sub not in summary:
            summary[sub] = {"planned": 0, "completed": 0, "mastery": []}
        
        summary[sub]["planned"] += entry["planned_hours"]
        summary[sub]["completed"] += entry["completed_hours"]
        
        if entry["mastery_score"] is not None:
            summary[sub]["mastery"].append(entry["mastery_score"])
    
    # compute final stats
    results = []
    for sub, data in summary.items():
        planned = data["planned"]
        completed = data["completed"]
        mastery_list = data["mastery"]
        
        mastery = sum(mastery_list)/len(mastery_list) if mastery_list else None
        
        completion_pct = (completed/planned)*100 if planned else 0
        
        results.append({
            "subject": sub,
            "planned_hours": round(planned,2),
            "completed_hours": round(completed,2),
            "completion_pct": round(completion_pct,2),
            "avg_mastery": round(mastery,2) if mastery else "N/A"
        })
    
    return pd.DataFrame(results)

subject_summary(progress_log)



def risk_overview(risk_scores):
    return pd.DataFrame([
        {"subject": sub, "risk_score": score}
        for sub, score in risk_scores.items()
    ]).sort_values("risk_score", ascending=False)

risk_overview(risk_scores)



import matplotlib.pyplot as plt

def plot_overall_completion(completion_pct):
    plt.figure(figsize=(5,4))
    
    plt.bar(["Completion"], [completion_pct])
    plt.ylim(0, 100)
    
    plt.title("Overall Study Completion (%)")
    plt.ylabel("Percentage")
    plt.show()

plot_overall_completion(completion_pct)



def plot_subject_completion(df):
    plt.figure(figsize=(6,4))
    
    subjects = df["subject"]
    values = df["completion_pct"]
    
    plt.bar(subjects, values)
    
    plt.title("Per-Subject Completion (%)")
    plt.ylabel("Percentage")
    plt.xticks(rotation=30)
    plt.ylim(0, 100)
    plt.show()

summary_df = subject_summary(progress_log)
plot_subject_completion(summary_df)



def plot_risk_scores(risk_scores):
    plt.figure(figsize=(6,4))
    
    subjects = list(risk_scores.keys())
    scores = list(risk_scores.values())
    
    plt.bar(subjects, scores)
    
    plt.title("Risk Score by Subject")
    plt.ylabel("Risk Level")
    plt.xticks(rotation=30)
    plt.show()

plot_risk_scores(risk_scores)



# === LLM SAFETY CHECK ===
import os

def can_use_llm():
    return "GEMINI_API_KEY" in os.environ and os.environ["GEMINI_API_KEY"].strip() != ""

if not can_use_llm():
    print("âš ï¸� LLM explanations skipped â€” No GEMINI_API_KEY detected.")
    print("To enable LLM features, set the GEMINI_API_KEY in the runtime environment.")
else:
    print("Gemini API key found â€” LLM explanations are enabled.")


from kaggle_secrets import UserSecretsClient
import os

# Load key from Kaggle Secrets
user_secrets = UserSecretsClient()
key = user_secrets.get_secret("GEMINI_API_KEY")

# Store in environment variable
os.environ["GEMINI_API_KEY"] = key

print("Gemini key loaded safely using Kaggle Secrets.")



import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

models = genai.list_models()

#for m in models:
    #print(m.name, " -- supports:", m.supported_generation_methods)


def llm_daily_explanation(plain_text):
    model = genai.GenerativeModel("models/gemini-2.5-flash")

    prompt = f"""
    Rewrite the following study plan instruction in a friendly,
    encouraging way for a student. Keep the meaning EXACTLY the same.

    {plain_text}
    """

    response = model.generate_content(prompt)
    return response.text.strip()


def llm_adaptation_explanation(changes_list):
    model = genai.GenerativeModel("models/gemini-2.5-flash")

    prompt = f"""
    The study agent made these adjustments to tomorrow's plan:

    {changes_list}

    Rewrite this into a short, simple, supportive explanation.
    Keep the meaning accurate.
    """

    response = model.generate_content(prompt)
    return response.text.strip()


# Test daily explanation on first day
plain = explain_daily_plan(initial_plan[0])
pretty = llm_daily_explanation(plain)

print("PLAIN:\n", plain)
print("\nLLM:\n", pretty)


plain_adapt = explain_adaptation(tomorrow_plan, adapted_plan, risk_scores)
pretty_adapt = llm_adaptation_explanation(plain_adapt["changes"])

print("PLAIN:\n", plain_adapt["changes"])
print("\nLLM:\n", pretty_adapt)

