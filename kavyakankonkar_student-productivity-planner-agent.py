import os
from kaggle_secrets import UserSecretsClient
import google.generativeai as genai

# Read Gemini key from Kaggle secret named "GEMINI_API_KEY"
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("GEMINI_API_KEY")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")


from datetime import datetime
from typing import List, Dict

def analyze_tasks(tasks: List[Dict]) -> List[Dict]:
    """
    Each task: {"name": str, "deadline": "YYYY-MM-DD", "hours": float}
    Returns same list with added fields: priority, reason.
    """
    today = datetime.today().date()
    result = []

    for t in tasks:
        d = datetime.strptime(t["deadline"], "%Y-%m-%d").date()
        days_left = (d - today).days

        if days_left <= 0:
            priority = "urgent"
            reason = "deadline is today or overdue"
        elif days_left == 1:
            priority = "urgent"
            reason = "deadline is tomorrow"
        elif days_left <= 3:
            priority = "important"
            reason = "deadline within 3 days"
        else:
             priority = "low"
             reason = "deadline more than 3 days away"

        result.append({
            **t,
            "days_left": days_left,
            "priority": priority,
            "reason": reason,
        })

    return result

# quick test
sample_tasks = [
    {"name": "DSA assignment", "deadline": "2025-11-30", "hours": 3},
    {"name": "DevOps exam revision", "deadline": "2025-12-02", "hours": 5},
]

analyze_tasks(sample_tasks)


def plan_schedule(tasks: List[Dict], user_context: str = "") -> str:
    analyzed = analyze_tasks(tasks)

    prompt = f"""
You are a helpful student productivity assistant.

User context (optional): {user_context}

You receive a list of tasks in JSON with fields:
name, deadline, hours, days_left, priority, reason.

1. First, briefly summarize how busy the next 2 days look.
2. Then create a plan for **today** and **tomorrow**:
   - For each day, list tasks in order with time estimates.
   - Respect priorities (urgent first) and keep total work realistic (around 6–8 hours per day).
3. End with 3 short tips for the student.

Here are the tasks (JSON):
{analyzed}
"""

    response = model.generate_content(prompt)
    return response.text
# demo
demo_tasks = [
    {"name": "Finish DevOps notes", "deadline": "2025-11-30", "hours": 2},
    {"name": "Work on Kaggle capstone", "deadline": "2025-11-30", "hours": 4},
    {"name": "DSA practice (10 problems)", "deadline": "2025-12-02", "hours": 3},
    {"name": "Clean room", "deadline": "2025-12-05", "hours": 1},
]

print(plan_schedule(demo_tasks, user_context="3rd year CS student with exams next week."))


