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


!pip install pydantic



from typing import List, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import json



class StudyTask(BaseModel):
    label: str
    priority: str = Field(..., description="High, Medium, Low")
    estimate_minutes: int

class DeadlineInfo(BaseModel):
    title: str
    due_date: str  # YYYY-MM-DD

class PlanRequest(BaseModel):
    tasks: List[StudyTask]
    deadlines: List[DeadlineInfo] = []
    planner_mode: str = "daily"  # daily | weekly



def fetch_search_results(query: str) -> str:
    return f"This is where Google Search results for '{query}' would appear:"



class MotivationEngine:
    @staticmethod
    def boost_message(score: int) -> str:
        if score >= 90:
            return "ðŸ”¥ Amazing job! You're operating at peak performance."
        if score >= 70:
            return "ðŸ’ª Strong consistencyâ€”keep the energy going."
        if score >= 40:
            return "âœ¨ You're progressing. Every effort pays off."
        return "ðŸŒ± Tough day, but tomorrow is yours. Reset and rise again."



class AccountabilityMonitor:
    def __init__(self):
        self.day_streak = 0
        self.previous_score = 0

    def log_checkin(self, text: str) -> None:
        print(f"[CHECK-IN]: {text}")

    def calculate_score(self, completed: int, total: int) -> int:
        if total == 0:
            return 0
        score = int((completed / total) * 100)
        if score > 60:
            self.day_streak += 1
        else:
            self.day_streak = 0
        self.previous_score = score
        return score



class PlanningBrain:
    @staticmethod
    def summarize_deadlines(deadlines: List[DeadlineInfo]) -> List[str]:
        today = datetime.now().date()
        info = []
        for d in deadlines:
            due = datetime.strptime(d.due_date, "%Y-%m-%d").date()
            diff = (due - today).days
            info.append(f"{d.title}: {diff} days remaining")
        return info

    def create_daily_agenda(self, tasks: List[StudyTask]) -> List[str]:
        agenda = []
        clock = datetime.now().replace(second=0, microsecond=0)
        for t in tasks:
            end = clock + timedelta(minutes=t.estimate_minutes)
            agenda.append(f"{clock.strftime('%H:%M')}â€“{end.strftime('%H:%M')} | {t.label} ãƒ» {t.priority}")
            clock = end
        return agenda

    def create_weekly_outline(self, tasks: List[StudyTask]) -> Dict[str, Any]:
        week = {d: [] for d in ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]}
        idx = 0
        for t in tasks:
            day = list(week.keys())[idx % 7]
            week[day].append(f"{t.label} ({t.priority}) â€“ {t.estimate_minutes} mins")
            idx += 1
        return week



class IntelliStudyOrchestrator:
    def __init__(self):
        self.monitor = AccountabilityMonitor()
        self.motivation = MotivationEngine()
        self.planner = PlanningBrain()

    def produce_plan(self, req: PlanRequest) -> Dict[str, Any]:
        results = {}
        results["deadline_summary"] = self.planner.summarize_deadlines(req.deadlines)

        if req.planner_mode == "weekly":
            results["weekly"] = self.planner.create_weekly_outline(req.tasks)
        else:
            results["daily"] = self.planner.create_daily_agenda(req.tasks)

        return results

    def run_progress_cycle(self, completed: int, total: int) -> Dict[str, Any]:
        score = self.monitor.calculate_score(completed, total)
        return {
            "score": score,
            "streak": self.monitor.day_streak,
            "motivation": self.motivation.boost_message(score)
        }

    def search_resources(self, topic: str) -> str:
        return fetch_search_results(topic)



engine = IntelliStudyOrchestrator()

req1 = PlanRequest(
    planner_mode="daily",
    tasks=[
        StudyTask(label="Revise Probability", priority="High", estimate_minutes=50),
        StudyTask(label="Stats Practice Problems", priority="Medium", estimate_minutes=30),
        StudyTask(label="Watch Regression Tutorial", priority="Low", estimate_minutes=20),
    ],
    deadlines=[DeadlineInfo(title="Math Exam", due_date="2025-12-10")]
)

daily_plan = engine.produce_plan(req1)
print(json.dumps(daily_plan, indent=2, ensure_ascii=False))




req2 = PlanRequest(
    planner_mode="weekly",
    tasks=[
        StudyTask(label="Study Chapter 4", priority="High", estimate_minutes=60),
        StudyTask(label="Flashcards", priority="Medium", estimate_minutes=25),
        StudyTask(label="Practice Quiz", priority="High", estimate_minutes=30),
    ]
)

weekly_plan = engine.produce_plan(req2)
print(json.dumps(weekly_plan, indent=2, ensure_ascii=False))



engine.run_progress_cycle(completed=2, total=4)



print(engine.search_resources("k-nearest neighbors explanation"))


