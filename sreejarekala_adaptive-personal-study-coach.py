!pip install --quiet tinydb python-dateutil



from tinydb import TinyDB, Query
from datetime import datetime
from dateutil import parser
import uuid
import os
import requests
import json



# ============================================================
# ğŸ”� Gemini API Key Authentication (Kaggle Secrets)
# ============================================================

from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please add 'GOOGLE_API_KEY' to Kaggle Secrets. Details: {e}"
    )



API_KEY = os.environ.get("GOOGLE_API_KEY")
MODEL = "gemini-2.5-flash"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"

def ask_gemini(prompt: str) -> str:
    """
    Sends a prompt to Gemini and returns the generated text.
    """
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    
    response = requests.post(URL, json=body)
    data = response.json()
    
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except:
        return f"[Gemini Error] {data}"



db = TinyDB("memory.json")
Memory = Query()

def save_memory(entry):
    entry["id"] = str(uuid.uuid4())
    entry["timestamp"] = datetime.now().isoformat()
    db.insert(entry)

def get_all_memory():
    return db.all()



db = TinyDB("memory.json")
Memory = Query()

def save_memory(entry):
    entry["id"] = str(uuid.uuid4())
    entry["timestamp"] = datetime.now().isoformat()
    db.insert(entry)

def get_all_memory():
    return db.all()



class MemoryAgent:
    def __init__(self, db):
        self.db = db

    def record_task(self, task, status, mood=None, energy=None):
        entry = {
            "type": "task_record",
            "task": task,
            "status": status,
            "mood": mood,
            "energy": energy,
        }
        save_memory(entry)

    def record_feedback(self, feedback):
        entry = {"type": "feedback", "feedback": feedback}
        save_memory(entry)

    def fetch_history(self):
        return get_all_memory()



memory_agent = MemoryAgent(db)
memory_agent.record_task("Study Operating Systems", "completed", mood="calm", energy="medium")



class PlannerAgent:
    def __init__(self):
        pass

    def decompose(self, task: str):
        prompt = (
            "Break the following study task into 5â€“8 actionable subtasks. "
            "Respond only with a numbered list.\n\n"
            f"Task: {task}"
        )
        output = ask_gemini(prompt)
        return output



planner = PlannerAgent()
subtasks = planner.decompose("Revise Data Structures")
print(subtasks)



class RecommenderAgent:
    def __init__(self):
        pass

    def recommend(self, subtasks: str, mood: str, energy: str):
        prompt = (
            f"You are an adaptive study recommender.\n"
            f"Given the user's mood: {mood} and energy: {energy},\n"
            f"reorder or enhance the following subtasks to create an optimal study sequence.\n\n"
            f"Subtasks:\n{subtasks}"
        )
        return ask_gemini(prompt)



recommender = RecommenderAgent()
rec = recommender.recommend(
    subtasks,
    mood="tired",
    energy="low"
)
print(rec)



class Controller:
    def __init__(self, memory_agent, planner_agent, recommender_agent):
        self.memory = memory_agent
        self.planner = planner_agent
        self.recommender = recommender_agent

    def process_task(self, task, mood, energy):
        subtasks = self.planner.decompose(task)
        plan = self.recommender.recommend(subtasks, mood, energy)

        self.memory.record_task(task, "planned", mood, energy)

        return {
            "subtasks": subtasks,
            "final_plan": plan,
            "history": self.memory.fetch_history()
        }



controller = Controller(memory_agent, planner, recommender)

result = controller.process_task(
    task="Prepare for DBMS exam",
    mood="neutral",
    energy="medium"
)

result



# Quick system test
controller = Controller(memory_agent, planner, recommender)

test_output = controller.process_task(
    task="Revise Computer Networks",
    mood="relaxed",
    energy="high"
)

test_output



# Inspect what the system remembers so far
memory_agent.fetch_history()



import gradio as gr

def interact_ui(task, mood, energy):
    result = controller.process_task(task, mood, energy)
    
    return (
        result["subtasks"],
        result["final_plan"],
        "Memory updated successfully âœ”ï¸�"
    )

ui = gr.Interface(
    fn=interact_ui,
    inputs=[
        gr.Textbox(label="Enter your study task"),
        gr.Dropdown(["calm", "stressed", "neutral"], label="Your current mood"),
        gr.Dropdown(["low", "medium", "high"], label="Your energy level"),
    ],
    outputs=[
        gr.Textbox(label="ğŸ§© Subtasks"),
        gr.Textbox(label="ğŸ�¯ Personalized Plan"),
        gr.Textbox(label="ğŸ—‚ Memory Status")
    ],
    title="ğŸ“Œ Adaptive Study Planner",
    description="Gemini-powered multi-agent study planner"
)

ui.launch(share=True)



def clear_memory(confirm=False):
    if confirm:
        db.truncate()
        print("ğŸ§¼ Memory wiped clean.")
    else:
        print("Set confirm=True to wipe memory.")

# Example: clear_memory(confirm=True)



print("âœ¨ Study Coach Notebook fully initialized.")


