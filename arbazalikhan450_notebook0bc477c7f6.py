# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)
    for filename in filenames:
        print("   ", filename)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ============================================================
# 1. Load Hackathon Dataset (Agents Intensive)
# ============================================================
hackathon_path = "/kaggle/input/agents-intensive-capstone-project/Hackathon dataset.txt"

if os.path.exists(hackathon_path):
    with open(hackathon_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    print("Hackathon dataset loaded. Length (chars):", len(content))
else:
    print("Hackathon dataset not found at:", hackathon_path)


from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Task:
    task_id: int
    description: str
    category: str
    importance: int
    urgency: int
    estimated_minutes: int
    due_in_hours: float
    completed: bool = False

    def __repr__(self):
        return (
            f"Task({self.task_id}, '{self.description[:15]}', "
            f"{self.category}, imp={self.importance}, urg={self.urgency}, "
            f"eta={self.estimated_minutes}m, due={self.due_in_hours}h, "
            f"done={self.completed})"
        )



def create_sample_tasks(seed: int = 42) -> List[Task]:
    np.random.seed(seed)
    categories = ["Study", "Work", "Health", "Chore", "Personal"]
    descriptions = [
        "Prepare for SQL interview",
        "Revise ML concepts",
        "Apply to job postings",
        "Clean apartment",
        "Go for a walk",
        "Complete Kaggle capstone",
        "Read a book",
        "Update resume",
        "Practice LeetCode",
        "Weekly reflection"
    ]

    tasks = []
    for i, desc in enumerate(descriptions):
        task = Task(
            task_id=i,
            description=desc,
            category=np.random.choice(categories),
            importance=np.random.randint(2, 6),
            urgency=np.random.randint(1, 6),
            estimated_minutes=np.random.choice([15, 25, 30, 45, 60, 90]),
            due_in_hours=float(np.random.choice([2, 4, 8, 12, 24, 48]))
        )
        tasks.append(task)

    return tasks

tasks = create_sample_tasks()
tasks[:5]



class ProductivityAgent:
    def __init__(self, tasks: List[Task]):
        self.tasks = tasks
        self.log = []

    def score_task(self, task: Task) -> float:
        time_factor = 1.0 / (1.0 + task.due_in_hours)
        score = 0.5 * task.importance + 0.3 * task.urgency + 0.2 * time_factor
        return score

    def choose_next_task(self) -> Optional[Task]:
        pending = [t for t in self.tasks if not t.completed]
        if not pending:
            return None

        scored = [(self.score_task(t), t) for t in pending]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def work_on_task(self, task: Task) -> float:
        if task.completed:
            return 0.0

        task.completed = True
        reward = 2 * task.importance + 1.5 * task.urgency

        self.log.append({
            "task_id": task.task_id,
            "description": task.description,
            "category": task.category,
            "importance": task.importance,
            "urgency": task.urgency,
            "due_in_hours": task.due_in_hours,
            "estimated_minutes": task.estimated_minutes,
            "reward": reward
        })

        return reward

    def simulate_day(self, available_minutes: int = 480):
        time_left = available_minutes
        total_reward = 0.0

        while time_left > 0:
            next_task = self.choose_next_task()
            if next_task is None:
                break

            if next_task.estimated_minutes > time_left:
                break

            reward = self.work_on_task(next_task)
            total_reward += reward
            time_left -= next_task.estimated_minutes

        penalty = 0.0
        for t in self.tasks:
            if not t.completed and t.urgency >= 4:
                penalty -= 2.0

        return total_reward + penalty, penalty, time_left

    def get_log_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.log)



agent = ProductivityAgent(tasks)

total_reward, penalty, time_left = agent.simulate_day()

log_df = agent.get_log_df()

print("Total reward:", total_reward)
print("Penalty for missed urgent tasks:", penalty)
print("Time left:", time_left)
log_df



import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
plt.hist(log_df["reward"], bins=10)
plt.title("Reward Distribution")
plt.xlabel("Reward")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(log_df["importance"], log_df["urgency"], s=log_df["reward"]*5)
plt.xlabel("Importance")
plt.ylabel("Urgency")
plt.title("Completed Tasks Importance vs Urgency")
plt.grid(True)
plt.show()



def inspect_next_task(agent: ProductivityAgent):
    t = agent.choose_next_task()
    if t is None:
        print("All tasks completed.")
        return

    print("Next recommended task:")
    print(" Description:", t.description)
    print(" Category:", t.category)
    print(" Importance:", t.importance)
    print(" Urgency:", t.urgency)
    print(" ETA (min):", t.estimated_minutes)
    print(" Due in (hrs):", t.due_in_hours)

inspect_next_task(agent)


