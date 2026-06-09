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


import datetime

# Mock Holiday List (Simulating API Response)
holiday_list = ["2025-01-26", "2025-03-08", "2025-03-29"]  # Republic Day, Maha Shivratri etc.

def is_holiday(date):
    return date.strftime("%Y-%m-%d") in holiday_list

def generate_study_plan(student_name, subjects, hours_per_day, exam_start_date):
    study_plan = []
    
    # Calculate starting date (1 month before exams)
    exam_date = datetime.datetime.strptime(exam_start_date, "%Y-%m-%d")
    start_date = exam_date - datetime.timedelta(days=25)
    
    current_date = start_date
    
    subject_index = 0  # For rotating subjects
    
    while current_date < exam_date:
        if is_holiday(current_date):
            study_plan.append({
                "date": current_date.strftime("%d %b"),
                "subject": "Holiday",
                "duration": "No Study",
                "notes": "Holiday â€“ Rest day"
            })
        else:
            # Distribute hours among subjects
            for i in range(hours_per_day):
                study_plan.append({
                    "date": current_date.strftime("%d %b"),
                    "subject": subjects[(subject_index + i) % len(subjects)],
                    "duration": "1 hour",
                    "notes": "AI generated study session"
                })
            subject_index += 1
        
        current_date += datetime.timedelta(days=1)
    
    return study_plan


# Example input for testing
student_name = "Mohit"
subjects = ["Math", "Science", "English", "SST"]
hours_per_day = 3
exam_start_date = "2025-03-15"

# Generate plan
plan = generate_study_plan(student_name, subjects, hours_per_day, exam_start_date)

# Display results
print(f"\nğŸ“˜ AI Study Plan for: {student_name}")
print(f"ğŸ—“ Exam Starts: {exam_start_date}\n")
print("â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�")
print(" Date   | Subject     | Duration | Notes")
print("â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�")
for entry in plan[:15]:  # Show first 15 days only (sample view)
    print(f" {entry['date']:6} | {entry['subject']:10} | {entry['duration']:8} | {entry['notes']}")
print("â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�")

