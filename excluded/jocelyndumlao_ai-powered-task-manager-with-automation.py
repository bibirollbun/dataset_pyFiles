!pip install google-adk


import os
import json
from typing import Any, Dict, List

from google.adk.agents import Agent, LlmAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import FunctionTool, google_search
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from kaggle_secrets import UserSecretsClient
import asyncio


# Configure your Gemini API Key
try:
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_Key")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


# Configure Retry Options
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# Constants
TASK_DB_FILE = "task_database.json"
USER_ID = "user123"
APP_NAME = "personalized_task_manager"


# Data Loading and Preprocessing

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Loads the dataset from a text file."""
    data = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {file_path}")
        return []
    return data


dataset = load_dataset("/kaggle/input/agents-intensive-capstone-project/Hackathon dataset.txt")
print(f"Loaded {len(dataset)} data entries.")



# Custom Tools (task DB interactions)

def load_tasks_from_db(user_id: str) -> List[Dict[str, Any]]:
    """Loads tasks from the task database."""
    try:
        with open(TASK_DB_FILE, "r") as f:
            task_data = json.load(f)
            return task_data.get(user_id, [])
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        print("Error: Invalid JSON in task database.")
        return []


def save_tasks_to_db(user_id: str, tasks: List[Dict[str, Any]]):
    """Saves tasks to the task database."""
    try:
        with open(TASK_DB_FILE, "r") as f:
            try:
                task_data = json.load(f)
            except json.JSONDecodeError:
                task_data = {}
    except FileNotFoundError:
        task_data = {}

    task_data[user_id] = tasks
    with open(TASK_DB_FILE, "w") as f:
        json.dump(task_data, f, indent=4)


def add_task(tool_context: ToolContext, task_description: str, priority: str = "medium") -> str:
    """Adds a new task to the task database."""
    user_id = tool_context.session_state.get("user_id", USER_ID)
    tasks = load_tasks_from_db(user_id)
    new_task = {"description": task_description, "priority": priority, "completed": False}
    tasks.append(new_task)
    save_tasks_to_db(user_id, tasks)
    return f"Added task: {task_description} with priority: {priority}"


def complete_task(tool_context: ToolContext, task_description: str) -> str:
    """Marks a task as complete in the task database."""
    user_id = tool_context.session_state.get("user_id", USER_ID)
    tasks = load_tasks_from_db(user_id)
    for task in tasks:
        if task["description"] == task_description:
            task["completed"] = True
            save_tasks_to_db(user_id, tasks)
            return f"Completed task: {task_description}"
    return f"Task not found: {task_description}"


def list_tasks(tool_context: ToolContext) -> str:
    """Lists all tasks from the task database."""
    user_id = tool_context.session_state.get("user_id", USER_ID)
    tasks = load_tasks_from_db(user_id)
    if not tasks:
        return "No tasks found."
    task_list = "\n".join([f"- {task['description']} (Priority: {task['priority']}, Completed: {task['completed']})" for task in tasks])
    return f"Tasks:\n{task_list}"


# Function Tools
add_task_tool = FunctionTool(add_task)
complete_task_tool = FunctionTool(complete_task)
list_tasks_tool = FunctionTool(list_tasks)



# Agent Architecture

task_understanding_agent = Agent(
    name="TaskUnderstandingAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a task understanding agent. Identify task description and priority."""
)

task_generation_agent = Agent(
    name="TaskGenerationAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""Generate helpful tasks based on user context.""",
    tools=[google_search],
)

root_agent = LlmAgent(
    name="TaskManagerAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="An agent to manage user tasks.",
    instruction="""Use the appropriate tools to add, complete, or list tasks.""",
    tools=[add_task_tool, complete_task_tool, list_tasks_tool, google_search],
)


# Sessions & Runner

session_service = InMemorySessionService()
runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)


# Session + Runner helper

async def run_session(
    runner_instance: InMemoryRunner,
    user_queries: list[str] | str = None,
    session_name: str = "default",
):
    print(f"\n ### Session: {session_name}")

    # Create or fetch session
    try:
        session = await session_service.create_session(
            app_name=runner_instance.app_name, user_id=USER_ID, session_id=session_name
        )
    except Exception:
        # If already exists, retrieve it
        session = await session_service.get_session(
            app_name=runner_instance.app_name, user_id=USER_ID, session_id=session_name
        )

    if user_queries:
        if isinstance(user_queries, str):
            user_queries = [user_queries]

        for query in user_queries:
            print(f"\nUser > {query}")

            query_content = types.Content(role="user", parts=[types.Part(text=query)])

            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query_content
            ):
                if event.content and event.content.parts:
                    text = event.content.parts[0].text
                    if text and text != "None":
                        print(f"{root_agent.name} > {text}")
    else:
        print("No queries!")



# Printed sample user/agent lines (print-only examples)

print("\nUser > Please add buy groceries to my tasks")
print("TaskManagerAgent > Added task: buy groceries with priority: medium")

print("\nUser > Please add wash the car to my tasks")
print("TaskManagerAgent > Added task: wash the car with priority: medium")

print("\nUser > Please add finish homework to my tasks")
print("TaskManagerAgent > Added task: finish homework with priority: medium")


# Evaluation set 

evaluation_set = {
    "eval_set_id": "task_manager_integration_suite",
    "eval_cases": [
        {
            "eval_id": "add_groceries_task",
            "conversation": [
                {
                    "user_content": {"parts": [{"text": "Please add buy groceries to my tasks"}]},
                    "final_response": {"parts": [{"text": "Added task: buy groceries with priority: medium"}]},
                    "intermediate_data": {
                        "tool_uses": [
                            {
                                "name": "add_task",
                                "args": {"task_description": "buy groceries", "priority": "medium"},
                            }
                        ]
                    },
                }
            ],
        },
        {
            "eval_id": "complete_groceries_task",
            "conversation": [
                {
                    "user_content": {"parts": [{"text": "Please complete buy groceries from my tasks"}]},
                    "final_response": {"parts": [{"text": "Completed task: buy groceries"}]},
                    "intermediate_data": {
                        "tool_uses": [
                            {"name": "complete_task", "args": {"task_description": "buy groceries"}}
                        ]
                    },
                }
            ],
        },
    ],
}

with open("integration.evalset.json", "w") as f:
    json.dump(evaluation_set, f, indent=2)

print("Created test data")


# Test sequence (uses run_session)

async def test_sequence():
    # these calls use the in-memory runner/session and will actually invoke the tools
    await run_session(runner, ["Please add buy groceries to my tasks"], "test_session")
    await run_session(runner, ["Please complete buy groceries from my tasks"], "test_session")
    await run_session(runner, ["What tasks do I have?"], "test_session")



# Main: ensure session is created BEFORE running tests

async def main():
    # Create test_session (if not exists). If already exists, get_session will succeed.
    try:
        await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id="test_session")
        print("Pre-created session: test_session")
    except Exception:
        # If already exists or creation failed, try to get it
        try:
            await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id="test_session")
            print("Session already exists: test_session")
        except Exception as e:
            print("Warning: could not create/get test_session:", e)
            # If session cannot be created, we still attempt to continue (will likely error)
    # Run the test sequence after ensuring session creation
    await test_sequence()


# Run main safely for both script and interactive (notebook) environments

def run_main_safely():
    try:
        # If no running loop, this raises a RuntimeError and we use asyncio.run(main())
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # Normal script execution
        asyncio.run(main())
    else:
        # Running inside an event loop (e.g., Jupyter / Kaggle notebook).
        # Schedule main as a task so it runs in the existing loop.
        # We also return the created task so callers (if any) may await it.
        task = asyncio.create_task(main())
        return task

# Execute
_run_task = run_main_safely()
# If in interactive environment and you want to wait for the task to finish synchronously,
# you can await `_run_task` from an async cell, or just let the environment run it.





