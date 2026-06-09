# ============================================
#  SMART STUDY PLANNER â€“ AI AGENT (FREESTYLE)
#  Created for Kaggle 5-Day AI Agents Intensive
#  This agent generates study plans for 12th CS
# ============================================

# No external installs needed (Kaggle runtime ok)

import random
from datetime import datetime, timedelta

# Subjects for 12th Computer Science group
subjects = ["Tamil", "English", "Maths", "Physics", "Chemistry", "Computer Science"]

print("ğŸ“˜ Smart Study Planner AI Agent\n")

# Ask user study hours
daily_hours = 4   # default fixed so Kaggle notebook runs without typing

print(f"Daily Study Hours: {daily_hours}\n")
print("Subjects:", subjects, "\n")

# Function to create study plan
def create_study_plan(hours):
    plan = {}
    per_subject = round(hours / len(subjects), 2)
    
    for sub in subjects:
        plan[sub] = per_subject
    return plan

# Generate plan
study_plan = create_study_plan(daily_hours)

print("ğŸ“… Today's Study Plan\n")
for subject, hrs in study_plan.items():
    print(f"{subject} : {hrs} hours")

# Weekly rotation plan
print("\nğŸ“† Weekly Rotation Plan\n")
week_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

for i, day in enumerate(week_days):
    subject_today = subjects[i % len(subjects)]
    print(f"{day}: Focus on {subject_today}")

# Revision schedule generator
print("\nğŸ”� Revision Schedule\n")

start_date = datetime.today()
for i in range(1, 6):
    rev_day = start_date + timedelta(days=i)
    print(f"Revision Day {i} ({rev_day.date()}): Revise {random.choice(subjects)}")

print("\nğŸ�“ AI Agent Successfully Generated Your Study Plan!")
print("Submit this notebook for Kaggle Freestyle Capstone Project.")

