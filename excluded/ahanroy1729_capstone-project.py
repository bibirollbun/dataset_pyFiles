# If needed on Kaggle, install dependencies
!pip install -q google-generativeai python-dotenv




import os
import google.generativeai as genai
from kaggle_secrets import UserSecretsClient


# Load API key from Kaggle Secrets (secure method)
user_secrets = UserSecretsClient()
API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in Kaggle Secrets!")

genai.configure(api_key=API_KEY)

def ask_gemini(prompt, model="models/gemini-2.5-flash"):
    """
    Simple wrapper around Gemini to send a text prompt and return the response text.
    """
    llm = genai.GenerativeModel(model)
    response = llm.generate_content(prompt)
    return response.text



class Memory:
    def __init__(self):
        self.notes = []
        self.tasks = []

    def add_task(self, task):
        self.tasks.append(task)

    def add_note(self, note):
        self.notes.append(note)

    def get_tasks(self):
        return self.tasks

    def get_notes(self):
        return self.notes

# Global memory instance
memory = Memory()



def planning_agent(user_input: str) -> str:
    """
    Planning agent: turns free-form user input into structured tasks.
    """
    prompt = f"""
You are a Planning Agent for a personal productivity / self-help system.

Convert the following input into structured tasks.

Input: {user_input}

Output format:
- Task list
- Deadlines (if any)
- Priority (High/Medium/Low)
"""
    tasks_text = ask_gemini(prompt)
    memory.add_note(tasks_text)
    return tasks_text



def scheduler_agent(tasks: str) -> str:
    """
    Scheduler agent: takes the task description and creates a realistic daily schedule.
    """
    prompt = f"""
Create a realistic daily schedule based on these tasks:
{tasks}

Consider time distribution, breaks, and priorities.
Return a clear, time-ordered schedule.
"""
    return ask_gemini(prompt)



def summarizer_agent(text: str) -> str:
    """
    Summarizer agent: simplifies the schedule into bullet points.
    """
    prompt = f"Summarize this in simple bullet points:\n{text}"
    return ask_gemini(prompt)



def assistant_controller(user_query: str):
    """
    Orchestrator: runs planning -> scheduling -> summarization pipeline.
    """
    # Step 1: Planning
    tasks = planning_agent(user_query)

    # Step 2: Scheduling
    schedule = scheduler_agent(tasks)

    # Step 3: Summary output
    summary = summarizer_agent(schedule)

    return {
        "tasks": tasks,
        "schedule": schedule,
        "summary": summary,
    }



print("=== Personal Concierge / Self-Help Agent ===\n")

# For Kaggle, avoid interactive input; instead, define a sample query.
# You can later turn this into a text widget or leave as a variable.
user_query = """
I feel overwhelmed this week. I need to finish my assignment,
revise for an exam, and still make time for a daily walk and journaling.
"""

print("User query:")
print(user_query)

result = assistant_controller(user_query)

print("\n\n=== TASKS IDENTIFIED ===")
print(result["tasks"])

print("\n\n=== GENERATED SCHEDULE ===")
print(result["schedule"])

print("\n\n=== SUMMARY ===")
print(result["summary"])


