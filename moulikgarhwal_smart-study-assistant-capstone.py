# Smart Study Assistant Agent (Offline, No-API version)
# This notebook is designed to run on Kaggle WITHOUT any external API.

import json
import os
import time
import threading
import random
from datetime import datetime

TASK_FILE = "tasks.json"

def load_tasks():
    """Load tasks from JSON file."""
    if os.path.exists(TASK_FILE):
        with open(TASK_FILE, "r") as f:
            return json.load(f)
    return []

def save_tasks(tasks):
    """Save tasks list to JSON file."""
    with open(TASK_FILE, "w") as f:
        json.dump(tasks, f, indent=2)



class TaskAgent:
    """Agent to manage study tasks with simple persistent memory."""
    
    def __init__(self):
        self.tasks = load_tasks()
    
    def add_task(self, description, deadline="no deadline"):
        task = {
            "description": description,
            "deadline": deadline,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "done": False,
        }
        self.tasks.append(task)
        save_tasks(self.tasks)
        return f"Task added: {description} (deadline: {deadline})"
    
    def list_tasks(self):
        if not self.tasks:
            return "No tasks stored yet."
        lines = []
        for i, t in enumerate(self.tasks, start=1):
            status = "✅ done" if t["done"] else "⏳ pending"
            lines.append(f"{i}. {t['description']}  [deadline: {t['deadline']}]  ({status})")
        return "\n".join(lines)
    
    def mark_done(self, index):
        if index < 1 or index > len(self.tasks):
            return "Invalid task number."
        self.tasks[index - 1]["done"] = True
        save_tasks(self.tasks)
        return f"Marked task {index} as done."
    
    def clear_all(self):
        self.tasks = []
        save_tasks(self.tasks)
        return "All tasks cleared."



class StudyPlanAgent:
    """Rule-based 7-day study planner (no external API)."""
    
    def generate_plan(self, subjects_text, hours_per_day=4):
        subjects = [s.strip() for s in subjects_text.split(",") if s.strip()]
        if not subjects:
            return "Please provide at least one subject (comma-separated)."
        
        days = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"]
        lines = []
        
        for i, day in enumerate(days):
            subject = subjects[i % len(subjects)]
            lines.append(
                f"{day}: Focus on {subject} – {hours_per_day} hours "
                f"(1h revision, 3h new topics)."
            )
        
        lines.append("\nGeneral tips:")
        lines.append("- Study in 45–60 minute blocks with 5–10 minute breaks.")
        lines.append("- Start each session by revising last session's notes.")
        lines.append("- Keep a small notebook for formulas and doubts.")
        
        return "\n".join(lines)



class SummarizerAgent:
    """
    Very simple offline 'summarizer':
    - Split text into sentences
    - Return first few as bullet points
    """
    
    def summarize(self, text, max_points=5):
        text = text.strip().replace("\n", " ")
        if not text:
            return "No text provided to summarize."
        
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        bullets = []
        for s in sentences[:max_points]:
            bullets.append(f"- {s}.")
        return "\n".join(bullets)



class MotivationAgent:
    def __init__(self):
        self.quotes = [
            "You’re doing great! Keep going!",
            "Small progress every day adds up.",
            "Focus on the next 30 minutes, not the whole syllabus.",
            "You are capable of more than you think.",
            "Consistency beats intensity. Just show up!"
        ]
    
    def motivate(self):
        return random.choice(self.quotes)


class PomodoroAgent:
    """
    Simple Pomodoro demo.
    For Kaggle demo we use seconds instead of real minutes.
    """
    
    def start(self, work_seconds=10, break_seconds=5, cycles=1):
        def run():
            for i in range(cycles):
                print(f"Cycle {i+1}: Work for {work_seconds} seconds.")
                time.sleep(work_seconds)
                print("Break time!")
                time.sleep(break_seconds)
            print("Pomodoro finished.")
        
        thread = threading.Thread(target=run)
        thread.start()
        return "Pomodoro started in background (watch cell output)."



class SmartStudyAgent:
    """
    Main orchestrator:
    - routes commands to different agents (multi-agent system)
    - also shows simple sequential behaviour
    """
    
    def __init__(self):
        self.task_agent = TaskAgent()
        self.plan_agent = StudyPlanAgent()
        self.summarizer_agent = SummarizerAgent()
        self.motivation_agent = MotivationAgent()
        self.pomodoro_agent = PomodoroAgent()
    
    def handle(self, query: str):
        q = query.strip()
        q_low = q.lower()
        
        # ---- Task commands ----
        if q_low.startswith("add task"):
            # Example: "add task Revise DBMS by tomorrow"
            body = q[8:].strip()  # remove 'add task'
            if " by " in body:
                desc, deadline = body.split(" by ", 1)
            else:
                desc, deadline = body, "no deadline"
            return self.task_agent.add_task(desc.strip(), deadline.strip())
        
        if q_low == "show tasks":
            return self.task_agent.list_tasks()
        
        if q_low.startswith("done task"):
            # Example: "done task 1"
            try:
                idx = int(q_low.split()[-1])
                return self.task_agent.mark_done(idx)
            except ValueError:
                return "Use: done task <number> (e.g. 'done task 1')."
        
        if q_low == "clear tasks":
            return self.task_agent.clear_all()
        
        # ---- Study plan ----
        if q_low.startswith("plan study for"):
            subjects = q[13:].strip()  # remove 'plan study for'
            return self.plan_agent.generate_plan(subjects)
        
        # ---- Summarizer ----
        if q_low.startswith("summarize"):
            text = q[9:].strip()
            return self.summarizer_agent.summarize(text)
        
        # ---- Motivation ----
        if "motivate" in q_low:
            return self.motivation_agent.motivate()
        
        # ---- Pomodoro (loop / long-running) ----
        if "start pomodoro" in q_low:
            return self.pomodoro_agent.start(work_seconds=5, break_seconds=3, cycles=1)
        
        # ---- Sequential example: daily recap ----
        if q_low == "daily recap":
            tasks_view = self.task_agent.list_tasks()
            recap_text = (
                "Today you worked on these tasks: \n" + tasks_view +
                "\nKeep going tomorrow with the highest priority ones!"
            )
            # Here one agent output flows into a 'summary' style message
            return recap_text
        
        # ---- Help ----
        return (
            "Commands I understand:\n"
            "- add task <description> by <deadline>\n"
            "- show tasks\n"
            "- done task <number>\n"
            "- clear tasks\n"
            "- plan study for <sub1, sub2, ...>\n"
            "- summarize <your text>\n"
            "- motivate me\n"
            "- start pomodoro\n"
            "- daily recap"
        )


# create a single global agent instance for the notebook
main_agent = SmartStudyAgent()



print("➡ Adding a task")
print(main_agent.handle("add task Revise DBMS chapter 3 by tomorrow"))
print()

print("➡ Showing tasks")
print(main_agent.handle("show tasks"))
print()

print("➡ Generating 7-day study plan")
print(main_agent.handle("plan study for DBMS, OS, CN"))
print()

print("➡ Summarizing example text")
sample_text = (
    "Artificial intelligence helps automate tasks. "
    "It can assist students with practice questions. "
    "It also supports teachers with grading. "
    "However, students must still understand core concepts. "
    "AI should be used as a helper, not a replacement for thinking."
)
print(main_agent.handle("summarize " + sample_text))
print()

print("➡ Getting motivation")
print(main_agent.handle("motivate me"))
print()

print("➡ Starting pomodoro demo (short)")
print(main_agent.handle("start pomodoro"))
print()

print("➡ Daily recap example")
print(main_agent.handle("daily recap"))



import os

file_path = "tasks.json"

if os.path.exists(file_path):
    os.remove(file_path)
    print("tasks.json deleted successfully!")
else:
    print("tasks.json not found (maybe already deleted).")


