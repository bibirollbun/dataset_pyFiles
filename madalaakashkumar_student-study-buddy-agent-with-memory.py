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


import random
import time
class SimpleLLM:
    def generate(self, prompt: str) -> str:
        templates = [
            "This is a focused and balanced study plan.",
            "The plan spreads time fairly across subjects.",
            "This schedule helps you revise regularly."
        ]
        return random.choice(templates)

llm = SimpleLLM()

class MemoryBank:
    def __init__(self):
        # Simple in-memory dictionary
        self.data = {}

    def save(self, user: str, info: dict):
        if user not in self.data:
            self.data[user] = {}
        self.data[user].update(info)

    def load(self, user: str):
        return self.data.get(user, {})

memory = MemoryBank()

def generate_quiz_questions(subject: str, num_questions: int = 3):
    """
    Very simple quiz generator tool (mock).
    In a real project, you could connect to a question bank or LLM.
    """
    questions = []
    for i in range(1, num_questions + 1):
        q = f"[{subject}] Question {i}: Write a short summary of today's topic."
        questions.append(q)
    return questions

logs = []

def log(message: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = f"{timestamp} - {message}"
    logs.append(entry)
    print("LOG:", entry)

def study_buddy_agent(user: str, subjects: list, hours_per_day: int):
    """
    Sequential flow:
    1) Log start
    2) Save subjects + hours in memory
    3) Build a simple time allocation plan
    4) Use quiz tool for each subject
    5) Return final structured result
    """
    log(f"StudyBuddy started for user={user}")
    memory.save(user, {"subjects": subjects, "hours_per_day": hours_per_day})


    overview_prompt = f"Generate a simple study guidance for {subjects} with {hours_per_day} hours."
    overview = llm.generate(overview_prompt)

    if len(subjects) == 0 or hours_per_day <= 0:
        return {"error": "Please provide at least one subject and positive hours."}

    hours_per_subject = max(1, hours_per_day // len(subjects))

    study_plan = []
    for subj in subjects:
        block = {
            "subject": subj,
            "allocated_hours": hours_per_subject,
            "quiz_questions": generate_quiz_questions(subj, num_questions=2)
        }
        study_plan.append(block)
        log(f"Allocated {hours_per_subject}h to {subj}")

    result = {
        "overview": overview,
        "hours_per_day": hours_per_day,
        "study_plan": study_plan
    }

    memory.save(user, {"last_plan": result})
    log("StudyBuddy finished and stored last_plan in memory")
    return result

demo_user = "student_1"
demo_subjects = ["Math", "Science", "English"]
demo_hours = 3

demo_result = study_buddy_agent(
    user=demo_user,
    subjects=demo_subjects,
    hours_per_day=demo_hours
)

demo_result



print("=== Memory for student_1 ===")
print(memory.load("student_1"))

print("\n=== Logs ===")
for line in logs:
    print(line)


