!pip install -q -U "google-generativeai>=0.8.3"

import datetime
import google.generativeai as genai
from kaggle_secrets import UserSecretsClient

# Get the API key from Kaggle Secrets
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=api_key)

print("Gemini client configured successfully.")



# Simple in-memory task "database"
TASKS = []
NEXT_ID = 1

def add_task(title: str, due_date: str | None = None, category: str | None = None) -> str:
    """
    Add a new task to the list and return a confirmation message.
    """
    global NEXT_ID
    task = {
        "id": NEXT_ID,
        "title": title,
        "due_date": due_date,
        "category": category,
        "status": "todo",
        "created_at": datetime.date.today().isoformat(),
    }
    TASKS.append(task)
    NEXT_ID += 1

    return (
        f"Added task #{task['id']}: '{task['title']}' "
        f"(due: {due_date or 'no deadline'}, "
        f"category: {category or 'general'})."
    )

def list_tasks(status: str | None = None) -> str:
    """
    Return a human-readable summary of current tasks.
    If status is given, filter by 'todo' or 'done'.
    """
    if not TASKS:
        return "You have no tasks yet."

    filtered = [t for t in TASKS if status is None or t["status"] == status]
    if not filtered:
        return f"You have no tasks with status '{status}'."

    lines = []
    for t in filtered:
        lines.append(
            f"#{t['id']} - {t['title']} "
            f"[status: {t['status']}, "
            f"due: {t['due_date'] or 'no deadline'}, "
            f"category: {t['category'] or 'general'}]"
        )
    return "Here are your tasks:\n" + "\n".join(lines)

def mark_task_done(task_id: int) -> str:
    """
    Mark a task as done by its numeric id.
    """
    for t in TASKS:
        if t["id"] == task_id:
            t["status"] = "done"
            return f"Marked task #{task_id} as done."
    return f"I couldn't find a task with id {task_id}."



# Create the agent model with tools
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    tools=[add_task, list_tasks, mark_task_done],
    system_instruction=(
        "You are StudyBuddy, a friendly productivity and study-planning agent. "
        "Help the user break down their work into tasks. "
        "Use the tools to add, list, and complete tasks when helpful. "
        "Always explain what you did in simple language."
    ),
)

# Create a chat session with automatic function calling enabled
chat = model.start_chat(enable_automatic_function_calling=True)

def demo_conversation():
    """
    Run a fixed demo conversation to show the agent using its tools.
    """
    messages = [
        "Hi StudyBuddy, I have an AI exam next week and a project due in 3 days. Please help me plan.",
        "Can you show me all my tasks?",
        "Mark task 1 as done and then show only the remaining todo tasks.",
    ]

    for i, msg in enumerate(messages, start=1):
        print(f"\n--- User message {i} ---")
        print("User:", msg)

        response = chat.send_message(msg)
        print("\nStudyBuddy response:\n")
        print(response.text)

# Run the demo conversation
demo_conversation()



# Create a simple submission file
with open("/kaggle/working/submission.txt", "w") as f:
    f.write("StudyBuddy AI Agent submission file.")



import pandas as pd

# Create a simple CSV submission file
df = pd.DataFrame({
    "result": ["StudyBuddy AI Agent submission file."]
})

df.to_csv("/kaggle/working/submission.csv", index=False)

print("CSV file created: submission.csv")


