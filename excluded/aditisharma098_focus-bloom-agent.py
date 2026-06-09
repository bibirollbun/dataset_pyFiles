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


# Tools 
import time
import threading
import random
import uuid
import subprocess

class FocusScoreTool:
    def calculate_focus(self, message: str) -> int:
        length = len(message)
        short_penalty = 10 if length < 5 else 0
        confusion_penalty = 15 if "idk" in message.lower() or "?" in message else 0
        base = min(length, 100)
        return max(base - short_penalty - confusion_penalty, 0)

class CodeExecutionTool:
    def run_python_code(self, code: str) -> str:
        try:
            result = subprocess.run(["python3","-c",code], capture_output=True, text=True, timeout=3)
            return result.stdout if result.stdout else result.stderr
        except Exception as e:
            return str(e)

# Sessions & Memory 
class InMemorySessionService:
    def __init__(self):
        self.sessions = {}
    def create_session(self, user_name: str) -> str:
        sid = str(uuid.uuid4())
        self.sessions[sid] = {"user": user_name, "state": {}}
        return sid
    def get_state(self, sid: str) -> dict:
        return self.sessions.get(sid, {}).get("state", {})

class MemoryBank:
    def __init__(self):
        self.store = []
    def add(self, owner, kind, text):
        self.store.append({"owner": owner, "kind": kind, "text": text})
    def query(self, owner, kind=None):
        return [m for m in self.store if m["owner"]==owner and (kind is None or m["kind"]==kind)]

# Agents 
class FocusDetectorAgent:
    def __init__(self, tool: FocusScoreTool):
        self.tool = tool
    def detect(self, message: str) -> int:
        return self.tool.calculate_focus(message)

class ConfusionDetectorAgent:
    def __init__(self, focus_tool: FocusScoreTool):
        self.focus_tool = focus_tool

    def detect(self, message: str) -> bool:
        # Keywords that indicate confusion
        keywords = ["?", "idk", "not getting", "confused"]
        confused_by_keywords = any(k in message.lower() for k in keywords)

        # Also consider very low focus as confusion
        focus_score = self.focus_tool.calculate_focus(message)
        confused_by_focus = focus_score < 20  # you can adjust threshold

        return confused_by_keywords or confused_by_focus


class MicroTaskAgent:
    def generate_task(self):
        tasks = [
            "Explain your topic in ONE emoji.",
            "Write a 10-word summary.",
            "Pick correct option: Deadlock happens due to (A) Mutual Exclusion (B) Ice Cream.",
            "Explain your topic in 1 sentence to a 5-year-old.",
            "List 3 keywords related to your topic."
        ]
        return tasks[int(time.time()) % len(tasks)]

class MiniLessonAgent:
    def create_lesson(self, topic: str):
        # Replace with GPT API if available
        return f"Mini-lesson on {topic}: \n- Break topic into small parts.\n- Relate to real-world examples.\n- Quick test yourself with a micro-question."

# Coordinator
class Coordinator:
    def __init__(self):
        self.focus_tool = FocusScoreTool()
        self.code_tool = CodeExecutionTool()
        self.detector = FocusDetectorAgent(self.focus_tool)
        self.confusion = ConfusionDetectorAgent(self.focus_tool)

        self.micro = MicroTaskAgent()
        self.lesson = MiniLessonAgent()
        self.memory = MemoryBank()
    def process_message(self, user_name: str, message: str):
        score = self.detector.detect(message)
        confused = self.confusion.detect(message)
        self.memory.add(user_name, "message", message)
        micro_task = None
        mini_lesson = None
        if score < 30 or confused:
            micro_task = self.micro.generate_task()
            mini_lesson = self.lesson.create_lesson(message)
        return score, confused, micro_task, mini_lesson

# Interactive Loop for Kaggle Notebook
coordinator = Coordinator()
print("ğŸŒ¼ FocusBloom Agent Started - Type 'exit' to quit\n")

while True:
    user_message = input("You: ")
    if user_message.lower() in ["exit", "quit"]:
        print("ğŸŒ± Ending session. Goodbye!")
        break
    score, confused, micro_task, mini_lesson = coordinator.process_message("user", user_message)
    print(f"[LOG] Focus Score: {score}, Confused: {confused}")
    if micro_task:
        print(f"ğŸŒŸ Micro Task: {micro_task}")
        print("(â�³ Pause for 20-40 seconds to complete task...)")
        time.sleep(2)  # simulate pause
    if mini_lesson:
        print(f"ğŸ“˜ Mini Lesson:\n{mini_lesson}")
    else:
        print("âœ¨ You're focused. Keep going!")





