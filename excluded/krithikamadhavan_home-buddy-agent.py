import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")


from typing import Any, Dict

from google.adk.agents import Agent, LlmAgent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.models.google_llm import Gemini
from google.adk.sessions import DatabaseSessionService
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import load_memory, preload_memory
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext
from google.genai import types

print("âœ… ADK components imported successfully.")

APP_NAME = "MemoryDemoApp"
USER_ID = "demo_user"



from google.genai import types

USER_ID = "kaggle_user"

async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str,
    session_name: str = "default-session"
):
    print(f"\n### Session: {session_name}")

    session_service = runner_instance.session_service
    app_name = runner_instance.app_name

    # Create or fetch session
    try:
        session = await session_service.create_session(
            app_name=app_name,
            user_id=USER_ID,
            session_id=session_name
        )
    except:
        session = await session_service.get_session(
            app_name=app_name,
            user_id=USER_ID,
            session_id=session_name
        )

    if isinstance(user_queries, str):
        user_queries = [user_queries]

    for query in user_queries:
        print(f"\nUser > {query}")

        # Convert user text to ADK Content format
        adk_query = types.Content(
            role="user",
            parts=[types.Part(text=query)]
        )

        # Stream multi-agent response
        async for event in runner_instance.run_async(
            user_id=USER_ID,
            session_id=session.id,
            new_message=adk_query
        ):
            if event.content and event.content.parts:
                text = event.content.parts[0].text
                if text and text != "None":
                    print(f"HomeBuddy > {text}")

    return "âœ” Session complete"



retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


## Instructions for Intent Recognition Agent
intent_agent_instruction = """
You are the Intent Agent. Your job is to understand the user's natural language
and convert it into a clean, structured JSON intent.

You MUST classify the user request into one of the following categories:

CHORE INTENTS:
- "chore_add"
- "chore_update"
- "chore_delete"
- "chore_complete"
- "chore_list"

MEAL PLAN INTENTS:
- "meal_plan_new"
- "meal_plan_update"
- "meal_plan_view"

GROCERY INTENTS:
- "grocery_generate"
- "grocery_view"

ROUTINE & REMINDER INTENTS:
- "reminder_create"
- "reminder_delete"
- "reminder_view"
- "routine_create"

OTHER:
- "other"

You MUST extract:
- task (e.g., "water plants", "clean kitchen")
- time ("7 AM", "tonight", "tomorrow morning")
- frequency ("daily", "weekly", etc.)
- days (for meal plan creation)
- diet preferences ("veg", "non-veg", "healthy veg", etc.)
- target_day & meal_type (for meal plan updates)
- all grocery-related info
- raw = original user message

Output format (ALWAYS JSON):

{
  "category": "...",
  "task": "...",
  "time": "...",
  "frequency": "...",
  "days": null or number,
  "diet": null or string,
  "target_day": null or string,
  "meal_type": null or string,
  "new_meal": null or string,
  "raw": "<original user msg>"
}

DO NOT perform tasks.
DO NOT call tools.
Only interpret and classify intent.
"""



intent_agent = LlmAgent(
    name="intent_agent",
    description="Understands user instructions and converts them into structured intent. "
        "Extracts task type, time, frequency, category, and details needed to route "
        "the request to the appropriate agent.",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction= intent_agent_instruction
)
print("âœ… Intent Agent Created")


import sqlite3

def connect_to_homebuddy_db():
    return sqlite3.connect("homebuddy_data.db")

def init_db():
    cursor = connect_to_homebuddy_db().cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chore_name TEXT NOT NULL,
        time TEXT,
        frequency TEXT,
        status TEXT DEFAULT 'pending'
    )
    """)

    connect_to_homebuddy_db().commit()
    connect_to_homebuddy_db().close()

# Initialize DB on startup
init_db()
print("âœ… Chores table is ready.")



import sqlite3

# -----------------------
# Add a chore
# -----------------------
def add_chore_to_db(chore_name: str, time: str | None, frequency: str | None):
    cursor = connect_to_homebuddy_db().cursor()

    cursor.execute("""
        INSERT INTO chores (chore_name, time, frequency, status)
        VALUES (?, ?, ?, 'pending')
    """, (chore_name, time, frequency))

    connect_to_homebuddy_db().commit()
    connect_to_homebuddy_db().close()

    return f"Chore '{chore_name}' added successfully."


# -----------------------
# List chores
# -----------------------
def list_all_chores():
    cursor = connect_to_homebuddy_db().cursor()

    cursor.execute("""
        SELECT id, chore_name, time, frequency, status FROM chores
    """)
    rows = cursor.fetchall()

    connect_to_homebuddy_db().close()

    chores = []
    for row in rows:
        chores.append({
            "id": row[0],
            "chore_name": row[1],
            "time": row[2],
            "frequency": row[3],
            "status": row[4]
        })

    return chores


# -----------------------
# Update chore
# -----------------------
def update_chore_in_db(chore_id: int, updates: dict):
    cursor = connect_to_homebuddy_db().cursor()

    for key, value in updates.items():
        cursor.execute(f"""
            UPDATE chores SET {key} = ? WHERE id = ?
        """, (value, chore_id))

    connect_to_homebuddy_db().commit()
    connect_to_homebuddy_db().close()

    return "Chore updated successfully."


# -----------------------
# Delete chore
# -----------------------
def delete_chore_from_db(chore_id: int):
    cursor = connect_to_homebuddy_db().cursor()

    cursor.execute("DELETE FROM chores WHERE id = ?", (chore_id,))

    connect_to_homebuddy_db().commit()
    connect_to_homebuddy_db().close()

    return "Chore deleted successfully."



from pydantic import BaseModel
# -----------------------
# Add Chore Tool
# -----------------------

class AddChoreInput(BaseModel):
    chore_name: str
    time: str | None = None
    frequency: str | None = None

class AddChoreOutput(BaseModel):
    status: str
    message: str

def add_chore(input: AddChoreInput) -> AddChoreOutput:
    msg = add_chore_to_db(
        input.chore_name,
        input.time,
        input.frequency
    )
    return AddChoreOutput(status="success", message=msg)


# -----------------------
# List Chores Tool
# -----------------------

class ListChoresOutput(BaseModel):
    chores: list

def list_chores() -> ListChoresOutput:
    chores = list_all_chores()
    return ListChoresOutput(chores=chores)


# -----------------------
# Update Chore Tool
# -----------------------

class UpdateChoreInput(BaseModel):
    id: int
    updates: dict

class UpdateChoreOutput(BaseModel):
    status: str
    message: str

def update_chore(input: UpdateChoreInput) -> UpdateChoreOutput:
    msg = update_chore_in_db(input.id, input.updates)
    return UpdateChoreOutput(status="success", message=msg)


# -----------------------
# Delete Chore Tool
# -----------------------

class DeleteChoreInput(BaseModel):
    id: int

class DeleteChoreOutput(BaseModel):
    status: str
    message: str
    
def delete_chore(input: DeleteChoreInput) -> DeleteChoreOutput:
    msg = delete_chore_from_db(input.id)
    return DeleteChoreOutput(status="success", message=msg)


print("ğŸ”¥ Chore Management Agent Tools Created")



## Instruction for Chore Management Agent

chore_agent_instruction = """
You are the Chore Management Agent.

Tools available:
- `add_chore()`: Add a new chore to the database.
- `list_chores()`: Retrieve all chores from the database.
- `update_chore()`: Update or modify an existing chore.
- `delete_chore()`: Remove a chore from the database.

Your responsibilities:
1. Add new chores with name, time, and frequency.
2. Update existing chores when the user requests changes.
3. Mark chores as completed using the update_chore tool (status = 'completed').
4. Delete chores when the user no longer needs them.
5. Retrieve and return a clean list of all chores.
6. Support daily, weekly, or custom recurring schedules.

Tool usage rules:
- If the user asks to add a chore â†’ call `add_chore()`.
- If the user asks to view chores â†’ call `list_chores()`.
- If the user asks to update or complete a chore â†’ call `update_chore()`.
- If the user asks to delete a chore â†’ call `delete_chore()`.

Additional rules:
- NEVER guess missing time or frequency. If unclear, ask for clarification.
- NEVER hallucinate tool names. Use only the tools listed above.
- ALWAYS check the "status" field in tool responses.
- ONLY return structured JSON when interacting with tools.

Output structure (only when tool output applies):
{
  "status": "success" | "error",
  "message": "...",
  "chores": [...]   // Only included for list_chores()
}

Your goal:
Manage chores reliably using the tools above and maintain consistent structured output.
"""


chore_agent= LlmAgent(
    name="chore_agent",
    description=("Handles all chore-related tasks. Adds, updates, deletes, and lists chores. "
        "Supports recurring schedules such as daily, weekly, or custom frequencies."),
    instruction=chore_agent_instruction,
    model=Gemini(model="gemini-2.5-flash-lite",retry_options=retry_config),
    tools = [add_chore,
    list_chores,
    update_chore,
    delete_chore]
)
print("âœ… Chore Management Agent Created")


# -----------------------
# Database functions
# -----------------------

# ---------------------------------
# Save Meal Plan in Meal Plan Table
# ---------------------------------

def save_meal_plan_in_db(days, diet, plan: dict):
    cursor = connect_to_homebuddy_db().cursor()

    cursor.execute("""
    INSERT INTO meal_plans (days, diet, plan_json)
    VALUES (?, ?, ?)
    """, (days, diet, json.dumps(plan)))

    connect_to_homebuddy_db().commit()
    connect_to_homebuddy_db().close()

    return "Meal plan saved."
    
# ---------------------------------
# Fetch Latest Meal Plan
# ---------------------------------

def get_latest_meal_plan_from_db():
    cursor = connect_to_homebuddy_db().cursor()

    cursor.execute("SELECT id, days, diet, plan_json FROM meal_plans ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    connect_to_homebuddy_db().close()

    if not row:
        return None
    
    return {
        "id": row[0],
        "days": row[1],
        "diet": row[2],
        "plan": json.loads(row[3])
    }

# ---------------------------------
# Update Meal Plan
# ---------------------------------

def update_meal_plan_in_db(meal_plan_id: int, updated_plan: dict):
    cursor = connect_to_homebuddy_db().cursor()

    cursor.execute("""
        UPDATE meal_plans SET plan_json = ?
        WHERE id = ?
    """, (json.dumps(updated_plan), meal_plan_id))

    connect_to_homebuddy_db().commit()
    connect_to_homebuddy_db().close()

    return "Meal plan updated."




# ---------------------------------
# Meal Planning Agent Tools
# ---------------------------------
from pydantic import BaseModel

class MealPlanInput(BaseModel):
    days: int
    diet: str
    plan: dict

class MealPlanOutput(BaseModel):
    status: str
    message: str

class MealPlanUpdateOutput(BaseModel):
    status: str            # "success" / "error"
    message: str           # human-readable message
    updated_plan: dict | None = None  # the updated meal plan after edit

# ---------------------------------
# Create Meal Plan Tool
# ---------------------------------

def create_meal_plan(input: MealPlanInput) -> MealPlanOutput:
    msg = save_meal_plan_in_db(input.days, input.diet, input.plan)
    return MealPlanOutput(status="success", message=msg)

# ---------------------------------
# Fetch Meal Plan Tool
# ---------------------------------

def fetch_meal_plan() -> MealPlanOutput:
    existing = get_latest_meal_plan_from_db()
    return MealPlanOutput(
        status="success",
        meal_plan=existing
    )

# ---------------------------------
# Update Meal Plan Tool
# ---------------------------------

def update_meal_plan(input: MealPlanInput) -> MealPlanUpdateOutput:
    msg = update_meal_plan_in_db(input.id, input.updated_plan)
    return MealPlanUpdateOutput(status="success", message=msg)

print("ğŸ”¥ Meal Planning Agent Tools Created")



# ---------------------------------
# Meal Planning Agent instructions
# ---------------------------------

meal_planning_agent_instruction = """
You are the Meal Planning Agent.

Your tools:
- `fetch_meal_plan()`: Retrieve the latest saved meal plan from the database.
- `create_meal_plan()`: Save a complete meal plan structure into the database.
- `update_meal_plan()`: Update an existing meal plan in the database.

Your responsibilities:
1. Generate healthy meal plans (1â€“7 days) using the following structure:
   {
     "days": <number>,
     "diet": "<veg | non-veg | healthy-veg | healthy-non-veg>",
     "plan": {
        "day1": {"breakfast": "", "lunch": "", "dinner": ""},
        "day2": {...}
     },
     "summary": "<short human-readable explanation>"
   }

2. Meal planning rules:
   - Keep meals simple, balanced, Indian-home-friendly.
   - NEVER produce extreme diets, fasting plans, or medical advice.
   - Support: veg, non-veg, simple, healthy-veg, healthy-non-veg.

3. When creating a NEW meal plan:
   - FIRST create the structured meal plan JSON using the format above.
   - THEN call `save_meal_plan()` with the generated JSON.

4. When UPDATING an existing meal plan:
   - FIRST call `fetch_meal_plan()` to load the latest saved plan.
   - Modify only the requested day/meal.
   - THEN call `update_meal_plan()` with the updated plan JSON.

5. When the user wants to view the meal plan:
   - Call `fetch_meal_plan()` and return it.

6. Tool Usage Rules:
   - Only use the tools listed above.
   - Always check the "status" field in tool responses.
   - NEVER hallucinate or invent tool names.
   - NEVER output unrelated text during tool execution.

Your goal:
Provide accurate, structured meal plans and ensure all plans are saved or updated correctly using the tools.
"""



# ---------------------------------
# Meal Planning Agent
# ---------------------------------
meal_planning_agent = LlmAgent(
    name="meal_planning_agent",
    description=("Generates healthy meal plans based on dietary preference (veg/non-veg), "
        "time constraints, and simplicity. Outputs structured daily or weekly plans."),
    instruction=meal_planning_agent_instruction,
    model=Gemini(model="gemini-2.5-flash-lite",retry_options=retry_config),
    tools=[create_meal_plan, fetch_meal_plan, update_meal_plan]
)
print("âœ… Meal Planning Agent Created")


import sqlite3
import json

# ---------------------------------
# Save Grocery List
# ---------------------------------
def save_grocery_list(grocery_list: dict):
    cursor = connect_to_homebuddy_db().cursor()

    cursor.execute("""
    INSERT INTO grocery_lists (list_json)
    VALUES (?)
    """, (json.dumps(grocery_list),))

    connect_to_homebuddy_db().commit()
    connect_to_homebuddy_db().close()

    return "Grocery list saved successfully."


# ---------------------------------
# Fetch Grocery List
# ---------------------------------
def fetch_grocery_list():
    cursor = connect_to_homebuddy_db().cursor()

    cursor.execute("""
    SELECT id, list_json FROM grocery_lists
    ORDER BY id DESC LIMIT 1
    """)
    row = cursor.fetchone()
    connect_to_homebuddy_db().close()

    if not row:
        return None

    return {
        "id": row[0],
        "grocery_list": json.loads(row[1])
    }



# ---------------------------------
# Grocery Agent Tools
# ---------------------------------
from pydantic import BaseModel

# ----------------------------
# Generate Grocery List Tool
# ----------------------------

class GroceryGenerateInput(BaseModel):
    meal_plan: dict

class GroceryGenerateOutput(BaseModel):
    status: str
    grocery_list: dict
    message: str

def generate_grocery_list(input: GroceryGenerateInput) -> GroceryGenerateOutput:
    """
    The LLM will analyze the meal_plan and produce the grocery items.
    This tool ensures the correct schema structure.
    """
    # LLM will fill this during tool execution
    grocery_template = {
        "produce": [],
        "grains": [],
        "dairy": [],
        "spices": [],
        "other": []
    }

    return GroceryGenerateOutput(
        status="success",
        grocery_list=grocery_template,
        message="Template created. LLM must populate grocery_list."
    )


# -----------------------
# Save Grocery List Tool
# -----------------------

class GrocerySaveInput(BaseModel):
    grocery_list: dict

class GrocerySaveOutput(BaseModel):
    status: str
    message: str

def save_grocery(input: GrocerySaveInput) -> GrocerySaveOutput:
    msg = save_grocery_list(input.grocery_list)
    return GrocerySaveOutput(status="success", message=msg)


# -----------------------
# Fetch Grocery List Tool
# -----------------------

class GroceryFetchOutput(BaseModel):
    status: str
    grocery_list: dict | None
    message: str

def fetch_grocery() -> GroceryFetchOutput:
    result = fetch_grocery_list()

    if not result:
        return GroceryFetchOutput(
            status="error",
            grocery_list=None,
            message="No grocery list found."
        )

    return GroceryFetchOutput(
        status="success",
        grocery_list=result["grocery_list"],
        message="Grocery list loaded."
    )

print("ğŸ”¥ Grocery Agent Tools Created")



# ---------------------------------
# Grocery Agent instructions
# ---------------------------------

grocery_agent_instructions ="""
You are the Grocery Agent.

Tools available:
- `generate_grocery_list()`: Analyze a meal plan and create a categorized grocery list template.
- `save_grocery()`: Save the final grocery list to the database.
- `fetch_grocery()`: Retrieve the latest saved grocery list.

Rules for usage:
1. When a meal plan is provided, first call `generate_grocery_list()` to create the grocery list structure.
2. If the user wants the grocery list saved, call `save_grocery()`.
3. If the user wants to view or retrieve the grocery list, call `fetch_grocery()`.
4. Do NOT hallucinate tool calls. Only use the specific tool listed above.
5. Return only structured tool output, or a friendly summary if no tool call is required.
6. Always check the status field in tool responses.

Your goal:
Generate, save, and fetch grocery lists with accurate categorization.
"""


grocery_agent=LlmAgent(
    name="grocery_agent",
    description=( "Takes a structured meal plan and generates a consolidated grocery list. "
        "Removes duplicates and groups items by category such as produce, grains, dairy, etc."
),
    instruction=grocery_agent_instructions,
    model=Gemini(model="gemini-2.5-flash-lite",retry_options=retry_config),
    tools=[generate_grocery_list,save_grocery,fetch_grocery,]
)
print("âœ… Grocery Agent Created")


import sqlite3

# ---------------------------
# Save Reminder
# ---------------------------
def save_reminder(task, time, frequency):
    cursor = connect_to_homebuddy_db().cursor()

    cursor.execute("""
    INSERT INTO reminders (task, time, frequency)
    VALUES (?, ?, ?)
    """, (task, time, frequency))

    connect_to_homebuddy_db().commit()
    connect_to_homebuddy_db().close()

    return "Reminder scheduled."


# ---------------------------
# Fetch All Reminders
# ---------------------------
def fetch_all_reminders():
    cursor = connect_to_homebuddy_db().cursor()

    cursor.execute("""
    SELECT id, task, time, frequency FROM reminders
    ORDER BY id DESC
    """)
    rows = cursor.fetchall()

    connect_to_homebuddy_db().close()

    reminders = []
    for row in rows:
        reminders.append({
            "id": row[0],
            "task": row[1],
            "time": row[2],
            "frequency": row[3]
        })

    return reminders


# ---------------------------
# Delete Reminder
# ---------------------------
def delete_reminder_from_db(reminder_id):
    cursor = connect_to_homebuddy_db().cursor()

    cursor.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))

    connect_to_homebuddy_db().commit()
    connect_to_homebuddy_db().close()

    return "Reminder deleted."




from pydantic import BaseModel
# ---------------------------
# TOOL 1 â€” Schedule Reminder
# ---------------------------
class ScheduleReminderInput(BaseModel):
    task: str
    time: str
    frequency: str | None = None   # daily / weekly / once

class ScheduleReminderOutput(BaseModel):
    status: str
    message: str

def schedule_reminder(input: ScheduleReminderInput) -> ScheduleReminderOutput:
    msg = save_reminder(
        task=input.task,
        time=input.time,
        frequency=input.frequency
    )
    return ScheduleReminderOutput(status="success", message=msg)


# ---------------------------
# TOOL 2 â€” Fetch Reminders
# ---------------------------
class FetchRemindersOutput(BaseModel):
    status: str
    reminders: list

def fetch_reminders() -> FetchRemindersOutput:
    reminders = fetch_all_reminders()
    return FetchRemindersOutput(
        status="success",
        reminders=reminders
    )


# ---------------------------
# TOOL 3 â€” Delete Reminder
# ---------------------------
class DeleteReminderInput(BaseModel):
    id: int

class DeleteReminderOutput(BaseModel):
    status: str
    message: str

def delete_reminder(input: DeleteReminderInput) -> DeleteReminderOutput:
    msg = delete_reminder_from_db(input.id)
    return DeleteReminderOutput(status="success", message=msg)



routine_agent = LlmAgent(
    name="routine_and_reminder_agent",
    model=Gemini(model="gemini-2.5-flash-lite"),
    instruction="""
You are the Routine & Reminder Agent.

Tools available:
- `schedule_reminder()`: Create a new reminder with task, time, and frequency.
- `fetch_reminders()`: Retrieve all existing reminders from the database.
- `delete_reminder()`: Remove a reminder by its ID.

Your responsibilities:
1. Create recurring routines (morning, evening, cleaning, prep routines).
2. Schedule reminders for chores, tasks, and daily activities.
3. Modify existing reminders by deleting and re-creating them.
4. Fetch and display all reminders when the user asks.

Tool Usage Rules:
- When the user wants to create a reminder â†’ use `schedule_reminder()`.
- When the user wants to view reminders â†’ use `fetch_reminders()`.
- When the user wants to delete or edit a reminder â†’ use `delete_reminder()`.
- NEVER guess missing time or frequency. Ask the user if unclear.
- NEVER hallucinate tool names. Use only the tools listed above.
- ALWAYS check the 'status' field in tool responses.

Output Structure:
Return only structured tool outputs or a friendly summary of the result.

Your goal:
Manage all routines and reminders reliably using the provided tools.
""",
    tools=[
        schedule_reminder,
        fetch_reminders,
        delete_reminder
    ]
)

print("â�° Routine & Reminder Agent ready!")



planner_agent_instruction = """
You are the Planner Agent.

Your job:
- Receive the structured JSON intent produced by the Intent Agent.
- Decide which specialized agent should handle the request.
- Forward the entire intent object to the correct agent and wait for its response.
- NEVER perform the task yourself.
- NEVER call tools directly.
- ONLY route tasks.

Categories and routing rules:

CHORE:
- "chore_add" â†’ chore_agent
- "chore_update" â†’ chore_agent
- "chore_delete" â†’ chore_agent
- "chore_complete" â†’ chore_agent
- "chore_list" â†’ chore_agent

MEAL PLAN:
- "meal_plan_new" â†’ meal_planning_agent
- "meal_plan_update" â†’ meal_planning_agent
- "meal_plan_view" â†’ meal_planning_agent

GROCERY:
- "grocery_generate" â†’ grocery_agent
- "grocery_view" â†’ grocery_agent

ROUTINE & REMINDERS:
- "reminder_create" â†’ routine_agent
- "reminder_delete" â†’ routine_agent
- "reminder_view" â†’ routine_agent
- "routine_create" â†’ routine_agent

RULES:
- NEVER hallucinate categories.
- NEVER modify the intent JSON.
- Forward exactly what you received.
- If category = "other", return a message saying you cannot identify the request.

Your output:
- A natural language summary after the child agent completes the request.

Example:
â€œSure! Iâ€™ve asked the Chore Agent to add your new chore.â€�

Your sole job is ROUTING.
"""



planner_agent = LlmAgent(
    name="planner_agent",
    model=Gemini(model="gemini-2.5-flash-lite"),
    instruction=planner_agent_instruction
)

print("ğŸ§  Planner Agent created")


## Instructions for main agent

main_agent_instructions = """
You are the HomeBuddy Concierge Agent.

Your role:
- Act as the main orchestrator for the HomeBuddy system.
- Receive user messages and pass them to the Intent Agent first.
- Then route the structured intent to the Planner Agent.
- The Planner Agent will decide which child agent should perform the task.

Your workflow:
1. ALWAYS send the user message to the Intent Agent.
2. Take the structured JSON returned by the Intent Agent.
3. ALWAYS forward that JSON to the Planner Agent.
4. NEVER perform tasks yourself.
5. NEVER call tools directly.
6. After the Planner Agent responds, summarize the result to the user
   in a friendly, simple way.

Categories you will see from the Intent Agent:
- chore_add, chore_update, chore_delete, chore_complete, chore_list
- meal_plan_new, meal_plan_update, meal_plan_view
- grocery_generate, grocery_view
- reminder_create, reminder_delete, reminder_view, routine_create
- other

Routing rules:
- DO NOT directly route based on category yourself.
- ALWAYS send the intent JSON to the Planner Agent.
- The Planner Agent will pick the correct child agent.

Tone rules:
- Friendly, warm, simple, and helpful.
- DO NOT expose raw JSON to the user unless they ask.
- Summaries must be human and conversational.
- If Intent Agent output is missing info, politely ask the user for clarification.

Goal:
Provide a smooth multi-agent experience where:
User â†’ Concierge â†’ Intent Agent â†’ Planner Agent â†’ Child Agent â†’ DB â†’ Response.
"""


# -----------------------------
# 1. Create Your Main Agent
# -----------------------------
concierge_agent = LlmAgent(
    name="homebuddy_concierge_agent",
    description=(
        "A smart concierge agent that helps with daily chores, weekly routines, "
        "healthy meal planning, and reminders."
    ),
    instruction=main_agent_instructions,
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    )
)

print("âœ… Concierge Agent created successfully!")

print("âœ… Concierge Agent Initialized")



# -----------------------------
# 2. Enable Persistent Sessions
# -----------------------------
db_url = "sqlite:///homebuddy_data.db"
session_service = DatabaseSessionService(db_url=db_url)

print("âœ… Persistent database ready at:", db_url)


# -----------------------------
# 3. Create the Runner
# -----------------------------
APP_NAME = "HomeBuddyChoresAgent"

runner = Runner(
    agent=concierge_agent,
    app_name=APP_NAME,
    session_service=session_service
)

print("ğŸš€ HomeBuddy Agent is ready with persistent memory!")



async def test_homebuddy(msg):
    response = await run_session(
        runner,
        msg,
        "homebuddy-prod-session"
    )
    return response


await test_homebuddy("Plan meals for 3 days. Vegetarian.")


