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


# Import required modules for date/time and JSON data storage
import datetime
import json

# Initialize an empty log for this notebook session
session_logs = []

print("import successful")


# TaskAgent class manages your to-do list using Python objects and persistent storage
class TaskAgent:
    # Initialize: load tasks from previous sessions (if available), else start fresh
    def __init__(self, memory_file="tasks.json"):
        self.memory_file = memory_file
        try:
            with open(self.memory_file, "r") as f:
                self.tasks = json.load(f)  # Load tasks from file
        except FileNotFoundError:
            self.tasks = []  # If file doesn't exist, start with no tasks
    
    # Add a new task to memory with optional due date
    def add_task(self, text, due=None):
        self.tasks.append({"text": text, "due": due, "status": "pending"})
        self._sync_memory()  # Save updates to file
        self.log_action(f"New task added: {text}")

    # Sort all tasks by due date (earliest first)
    def prioritize_tasks(self):
        self.tasks.sort(key=lambda x: x['due'] or "")
        self.log_action("Tasks prioritized.")

    # Show which tasks are scheduled for today
    def remind_tasks(self):
        today = datetime.date.today().isoformat()
        due_today = [t for t in self.tasks if t['due'] == today]
        return due_today

    # Mark task as done by its position (index) in the list
    def mark_done(self, idx):
        self.tasks[idx]['status'] = "done"
        self._sync_memory()  # Save changes
        self.log_action(f"Task marked done: {self.tasks[idx]['text']}")

    # Log each action with timestamp for review and debugging
    def log_action(self, msg):
        log_entry = {"ts": datetime.datetime.now().isoformat(), "msg": msg}
        session_logs.append(log_entry)

    # Write updated task list to persistent memory (JSON file)
    def _sync_memory(self):
        with open(self.memory_file, "w") as f:
            json.dump(self.tasks, f)


# Display all logged actions from the current notebook session
def show_past_sessions():
    for log in session_logs:
        print(f"[{log['ts']}] {log['msg']}")


# Create the agent and test core features for the notebook
agent = TaskAgent()  # Create agent instance and load previous memory
agent.add_task("Complete mathematics assignment", "2025-11-20")      # Add assignment task
agent.add_task("Go for groceries", "2025-11-19")                    # Add shopping task
agent.prioritize_tasks()                                            # Organize tasks by deadline
agent.mark_done(1)                                                  # Mark groceries as completed

# Print today's reminders and show the session log history
print("Today's reminders:", agent.remind_tasks())
show_past_sessions()
print("All tasks:", agent.tasks)  # Optionally show detailed task status list

