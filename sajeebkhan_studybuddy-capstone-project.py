# Cell 1: Install ADK + genai libs
!pip install -q google-adk google-genai
print("Installed google-adk and google-genai")



# Cell 2: Load API key from Kaggle Secrets + Suppress ADK warnings

from kaggle_secrets import UserSecretsClient
import os
import logging   # <-- added

# Load API key
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("Secret GOOGLE_API_KEY not found in Kaggle Secrets.")

os.environ["GOOGLE_API_KEY"] = api_key

print("API key loaded successfully!")

# ---- Suppress noisy ADK warnings ----
logging.getLogger("google_genai.types").setLevel(logging.ERROR)




# Cell 3: imports & file helpers
import json
import csv
import os
from datetime import datetime, date, timedelta
from pathlib import Path

DATA_DIR = Path("studybuddy_data")
DATA_DIR.mkdir(exist_ok=True)

PLAN_PATH = DATA_DIR / "plan.json"
PROGRESS_PATH = DATA_DIR / "progress.csv"

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def append_progress(row):
    header = ["date", "task_id", "task_text", "done", "score"]
    write_header = not PROGRESS_PATH.exists()
    with open(PROGRESS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)



# Cell 4: Create an agent skeleton using google-adk
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types

# Root agent (acts as dispatcher / orchestrator)
root_agent = Agent(
    name="studybuddy_root_agent",
    model="gemini-2.5-flash",  # choose available model if needed
    instruction=(
        "You are StudyBuddy: brief, clear, encouraging assistant. "
        "Use tools when appropriate for planning, quizzing, and progress logging."
    ),
    description="Root agent for StudyBuddy Capstone."
)

runner = InMemoryRunner(agent=root_agent, app_name="studybuddy_capstone_app")
print("Agent + runner initialized")



# Cell 5: Tools (planner, get_today_plan, quiz, log_progress, summarize)

from typing import Optional

# ---- FIX: ADK's 'tool' decorator no longer exists, so we define our own ----
def tool(func):
    func.is_tool = True  # simple metadata for ADK
    return func


# ---------------------
# Tool: create study plan
# ---------------------
@tool
def plan_study_tool(goal_type: str, days_until_exam: int, hours_per_day: float) -> dict:
    today = date.today()
    plan = {"goal_type": goal_type, "created_at": str(today), "days": []}

    for i in range(days_until_exam):
        day_date = today + timedelta(days=i)
        plan["days"].append({
            "day": i + 1,
            "date": str(day_date),
            "tasks": [
                {"id": f"{i+1}-1", "type": "reading", "text": f"30 min focused reading ({goal_type})"},
                {"id": f"{i+1}-2", "type": "vocab", "text": "20 vocab flashcards + 10 min review"},
                {"id": f"{i+1}-3", "type": "practice", "text": "30 min practice questions / speaking prompts"}
            ]
        })

    save_json(PLAN_PATH, plan)
    return {
        "message": f"Plan created: {days_until_exam} days, {hours_per_day} hours/day",
        "plan_path": str(PLAN_PATH)
    }


# ---------------------
# Tool: get today's plan
# ---------------------
@tool
def get_today_plan_tool() -> dict:
    plan = load_json(PLAN_PATH)
    if not plan:
        return {"error": "No plan found. Create one using plan_study_tool."}

    today = str(date.today())
    for day in plan["days"]:
        if day["date"] == today:
            return {"today": day}
    return {"today": plan["days"][0]}


# ---------------------
# Tool: quiz generator
# ---------------------
SAMPLE_VOCAB = [
    ("coherent", "logical and consistent"),
    ("obscure", "not discovered or known about"),
    ("concise", "giving a lot of information clearly in few words"),
    ("mitigate", "make less severe"),
]

@tool
def quiz_tool(num_questions: int = 3) -> dict:
    questions = []
    for i in range(min(num_questions, len(SAMPLE_VOCAB))):
        word, meaning = SAMPLE_VOCAB[i]
        questions.append({
            "q_id": i + 1,
            "word": word,
            "prompt": f"Define or use '{word}' in a sentence."
        })
    return {"questions": questions}


# ---------------------
# Tool: log progress  ✅ FIXED VERSION
# ---------------------
@tool
def log_progress_tool(task_id: str, task_text: str, done: bool = True, score: Optional[float] = None) -> dict:
    row = {
        "date": str(date.today()),
        "task_id": task_id,
        "task_text": task_text,
        "done": "yes" if done else "no",
        "score": score if score is not None else ""
    }
    append_progress(row)
    return {"message": "Progress logged", "row": row}


# ---------------------
# Tool: summarize progress
# ---------------------
@tool
def summarize_progress_tool() -> dict:
    if not PROGRESS_PATH.exists():
        return {"summary": "No progress logged yet."}

    total = 0
    done_count = 0
    scores = []

    with open(PROGRESS_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            total += 1
            if r["done"].lower() in ("yes", "true", "1"):
                done_count += 1
            if r["score"]:
                try:
                    scores.append(float(r["score"]))
                except:
                    pass

    completion_rate = round(100 * done_count / total, 2) if total else 0.0
    avg_score = round(sum(scores) / len(scores), 2) if scores else None

    return {
        "total_tasks": total,
        "completed": done_count,
        "completion_rate_pct": completion_rate,
        "avg_quiz_score": avg_score,
    }


# ---------------------
# Register tools with root agent
# ---------------------
root_agent.tools = [
    plan_study_tool,
    get_today_plan_tool,
    quiz_tool,
    log_progress_tool,
    summarize_progress_tool
]

print("Tools attached to root_agent successfully!")



# Cell 6: create a real session + demonstrate tool calls

# Create real session using ADK SessionService
session = await runner.session_service.create_session(
    app_name=runner.app_name,
    user_id="sajeeb-user",
)

print("Session created:", session.id)

# --- Demonstrate tools ---
print(plan_study_tool("IELTS", 10, 2.0))
print(get_today_plan_tool())
print(quiz_tool(3))
print(log_progress_tool("1-1", "30 min focused reading (IELTS)", True, score=8.0))
print(summarize_progress_tool())



# Cleaned Cell 7 – Zero warnings, Zero 'None', Clean output

import logging
logging.getLogger("google_genai.types").setLevel(logging.ERROR)

from google.genai import types

def run_agent_simple(message: str):
    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=message)]
    )

    for event in runner.run(
        user_id="sajeeb-user",
        session_id=session.id,
        new_message=content,
    ):
        if event.content and event.content.parts:
            text = getattr(event.content.parts[0], "text", None)
            if text:
                print("Agent:", text)


# Example interactions
run_agent_simple("Create a 7-day IELTS plan for 2 hours per day.")
run_agent_simple("What is my plan for today?")
run_agent_simple("Give me a short vocabulary quiz of 2 words.")
run_agent_simple("I completed today's reading and scored 9 on the quiz.")



# Cell 8: Full demonstration of StudyBuddy Agent capabilities

print("---- FULL STUDYBUDDY AGENT DEMO ----\n")

# 1️⃣ Create a new 5-day IELTS plan
print(">>> User: Create a 5-day IELTS plan for reading + speaking.")
run_agent_simple("Create a 5-day IELTS plan for reading + speaking.")
print("\n")

# 2️⃣ Ask for today's tasks
print(">>> User: What is my study plan for today?")
run_agent_simple("What is my study plan for today?")
print("\n")

# 3️⃣ Generate a quiz
print(">>> User: Give me a vocabulary quiz with 3 words.")
run_agent_simple("Give me a vocabulary quiz with 3 words.")
print("\n")

# 4️⃣ Mark tasks completed + quiz score
print(">>> User: I completed today's tasks and scored 8 on the quiz.")
run_agent_simple("I completed today's tasks and scored 8 on the quiz.")
print("\n")

# 5️⃣ Summarize overall progress
print(">>> User: Summarize my progress so far.")
run_agent_simple("Summarize my progress so far.")
print("\n")

print("---- END OF DEMO ----")


