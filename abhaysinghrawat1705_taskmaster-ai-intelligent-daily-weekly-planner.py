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


# AGENT MEMORY --------------------------------------------------
import json
from datetime import datetime, timedelta

class AgentMemory:
    def __init__(self):
        self.data = {
            "reminders": [],
            "tasks": []
        }

    def save_reminder(self, reminder):
        self.data["reminders"].append(reminder)

    def get_reminders(self):
        return self.data["reminders"]

    def save_task(self, task):
        self.data["tasks"].append(task)

    def get_tasks(self):
        return self.data["tasks"]

memory = AgentMemory()


# PLANNER --------------------------------------------------------
class Planner:
    def plan(self, user_input):
        plan = []

        ui = user_input.lower()

        if "day" in ui:
            plan.append({"step": "daily planning", "tool": "daily_planner"})

        if "week" in ui:
            plan.append({"step": "weekly planning", "tool": "weekly_scheduler"})

        if "remind" in ui:
            plan.append({"step": "set reminder", "tool": "reminder"})

        if "summarize" in ui or "report" in ui:
            plan.append({"step": "generate summary", "tool": "report"})

        # Every pipeline ends with reflection
        plan.append({"step": "reflection", "tool": "reflection"})

        return plan



# TOOL 1 â€” DAILY PLANNER -----------------------------------------
class DailyPlanner:
    def run(self, tasks):
        schedule = []
        start = datetime.now().replace(hour=9, minute=0)

        for task in tasks:
            end = start + timedelta(hours=1)
            schedule.append({
                "task": task.strip(),
                "start": start.strftime("%I:%M %p"),
                "end": end.strftime("%I:%M %p")
            })
            start = end

        return schedule


# TOOL 2 â€” WEEKLY SCHEDULER --------------------------------------
class WeeklyScheduler:
    def run(self, tasks):
        week = {}
        today = datetime.now()

        for i, task in enumerate(tasks):
            day = (today + timedelta(days=i)).strftime("%A")
            week[day] = task.strip()

        return week


# TOOL 3 â€” REMINDER TOOL -----------------------------------------
class ReminderTool:
    def run(self, reminder_text, time):
        reminder = {"reminder": reminder_text, "time": time}
        memory.save_reminder(reminder)
        return f"ğŸ”” Reminder saved: '{reminder_text}' at {time}"


# TOOL 4 â€” REPORT GENERATOR --------------------------------------
class ReportGenerator:
    def run(self, text):
        sentences = text.split(".")
        summary = [s.strip() for s in sentences if len(s.strip()) > 0]
        return summary


# TOOL 5 â€” REFLECTION TOOL ---------------------------------------
class ReflectionTool:
    def run(self, raw):
        if not raw:
            return "âš ï¸� No valid output generated."

        return f"âœ¨ **Refined Output**:\n{raw}"



class TaskMasterAgent:
    def __init__(self):
        self.planner = Planner()
        self.daily = DailyPlanner()
        self.weekly = WeeklyScheduler()
        self.remind = ReminderTool()
        self.report = ReportGenerator()
        self.reflect = ReflectionTool()

    def execute(self, user_input):
        plan = self.planner.plan(user_input)

        # Extract tasks after ":"
        tasks = []
        if ":" in user_input:
            try:
                tasks = user_input.split(":")[1].split(",")
            except:
                tasks = []

        output = None

        for step in plan:
            tool = step["tool"]

            if tool == "daily_planner":
                output = self.daily.run(tasks)

            elif tool == "weekly_scheduler":
                output = self.weekly.run(tasks)

            elif tool == "reminder":
                output = self.remind.run("Task Reminder", "Tomorrow 10 AM")

            elif tool == "report":
                output = self.report.run(user_input)

            elif tool == "reflection":
                output = self.reflect.run(output)

        return output


agent = TaskMasterAgent()



print("ğŸ“… Daily Plan Example:")
print(agent.execute("Plan my day with these tasks: gym, study, work"))

print("\nğŸ—“ Weekly Planner Example:")
print(agent.execute("Create a weekly schedule: workout, coding, reading, shopping"))

print("\nğŸ”” Reminder Example:")
print(agent.execute("Remind me about my meeting"))

print("\nğŸ“� Summary Example:")
print(agent.execute("Summarize these notes: AI is growing fast. Agents are the future."))


