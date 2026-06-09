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
import json
from datetime import datetime, timedelta
import random

class SmartStudyAgent:
    def __init__(self, data_file="study_data.json"):
        self.data_file = data_file
        self.data = self.load_data()

    # ---------------------------------------
    # Load & Save
    # ---------------------------------------
    def load_data(self):
        try:
            with open(self.data_file, "r") as f:
                return json.load(f)
        except:
            return {"subjects": {}, "progress": {}}

    def save_data(self):
        with open(self.data_file, "w") as f:
            json.dump(self.data, f, indent=4)

    # ---------------------------------------
    # Add Subject and Topics
    # ---------------------------------------
    def add_subject(self, subject, topics):
        self.data["subjects"][subject] = topics
        self.data["progress"][subject] = {topic: 0 for topic in topics}
        self.save_data()
        print(f"Added subject {subject} with topics.")

    # ---------------------------------------
    # Create Study Plan
    # ---------------------------------------
    def create_study_plan(self, subject, days):
        topics = self.data["subjects"].get(subject)
        if not topics:
            return "Subject not found."

        per_day = max(1, len(topics) // days)
        plan = {}
        index = 0

        for d in range(days):
            plan[f"Day {d+1}"] = topics[index:index+per_day]
            index += per_day

        return plan

    # ---------------------------------------
    # Generate Quiz
    # ---------------------------------------
    def generate_quiz(self, subject, n=5):
        topics = self.data["subjects"].get(subject)
        if not topics:
            return "Subject not found."

        questions = []
        for _ in range(n):
            topic = random.choice(topics)
            q = f"Explain the concept of {topic}."
            questions.append(q)
        return questions

    # ---------------------------------------
    # Update Progress
    # ---------------------------------------
    def update_progress(self, subject, topic, percent):
        if subject in self.data["progress"] and topic in self.data["progress"][subject]:
            self.data["progress"][subject][topic] = percent
            self.save_data()
            return f"Updated progress for {topic}."
        return "Invalid subject or topic."

    # ---------------------------------------
    # View Progress
    # ---------------------------------------
    def get_progress(self, subject):
        return self.data["progress"].get(subject, "Subject not found.")

# ----------------------------------------------------
# Example Usage
# ----------------------------------------------------
if __name__ == "__main__":
    agent = SmartStudyAgent()

    # Add subject
    agent.add_subject("DBMS", [
        "ER Model",
        "Normalization",
        "Transactions",
        "Indexing",
        "SQL Joins",
        "Relational Algebra"
    ])

    # Generate plan
    plan = agent.create_study_plan("DBMS", days=3)
    print("\nStudy Plan:")
    print(json.dumps(plan, indent=4))

    # Quiz
    print("\nQuiz:")
    for q in agent.generate_quiz("DBMS"):
        print("- " + q)

    # Update progress
    print("\n" + agent.update_progress("DBMS", "ER Model", 80))

    # View progress
    print("\nProgress:")
    print(agent.get_progress("DBMS"))


