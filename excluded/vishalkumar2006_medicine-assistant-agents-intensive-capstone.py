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


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.adk.agents import LlmAgent, SequentialAgent, BaseAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import load_memory, preload_memory, FunctionTool
from google.genai import types
print("âœ… ADK components imported successfully.")


import logging
import os

# Clean up any previous logs
for log_file in ["logger.log", "web.log", "tunnel.log"]:
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"ğŸ§¹ Cleaned up {log_file}")

# Configure logging with DEBUG log level.
logging.basicConfig(
    filename="logger.log",
    level=logging.DEBUG,
    format="%(filename)s:%(lineno)s %(levelname)s:%(message)s",
)

print("âœ… Logging configured")


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


import sqlite3
from typing import List, Dict, Any, Union

# --- Configuration ---
GEMINI_MODEL = Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config)
DB_PATH = "medication_assistant.db"

# --- SQLite Setup (Long-Term Memory) ---
def setup_database():
    """Creates the SQLite database and necessary tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Table for User's Medication and Profile Data
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS UserMeds (
            id INTEGER PRIMARY KEY,
            user_id TEXT NOT NULL,
            med_name TEXT NOT NULL,
            dosage TEXT,
            frequency TEXT
        )
    """)
    # Table for Drug-Drug Interaction Reference Data
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Interactions (
            id INTEGER PRIMARY KEY,
            med_a TEXT NOT NULL,
            med_b TEXT NOT NULL,
            interaction_level TEXT, -- e.g., 'Severe', 'Moderate', 'None'
            recommendation TEXT
        )
    """)
    
    # Insert some dummy interaction data
    cursor.execute("INSERT OR IGNORE INTO Interactions VALUES (1, 'Lisinopril', 'Advil', 'Moderate', 'Monitor blood pressure closely.')")
    cursor.execute("INSERT OR IGNORE INTO Interactions VALUES (2, 'Amoxicillin', 'None', 'None', 'No known interactions.')")

    conn.commit()
    conn.close()
    print(f"Database setup complete at {DB_PATH}")

# --- Tool Functions (Exposed to Agents) ---

def query_meds_db(query: str) -> str:
    """
    Executes a SELECT query against the Medication Assistant SQLite database.
    Use this to retrieve user's current meds or check for interactions.
    The query must be a valid SQL statement (SELECT * FROM...).
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        # Fetch column names for a structured result
        column_names = [description[0] for description in cursor.description]
        results = [dict(zip(column_names, row)) for row in cursor.fetchall()]
        return f"Query successful. Results: {results}"
    except Exception as e:
        return f"Query failed: {e}"
    finally:
        conn.close()

def update_meds_db(sql_statement: str) -> str:
    """
    Executes an INSERT, UPDATE, or DELETE statement against the database.
    Use this to set reminders or log new medications.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(sql_statement)
        conn.commit()
        return f"Update successful. Rows affected: {cursor.rowcount}"
    except Exception as e:
        return f"Update failed: {e}"
    finally:
        conn.close()

# Define the Function Tools for the ToolAgent
db_query_tool = FunctionTool(query_meds_db)
db_update_tool = FunctionTool(update_meds_db)

# Run the setup once
setup_database()

print("âœ… Tools and DB setup successful")


# --- 3. Specialized LlmAgents ---

# Agent that manages the persistent SQLite data (ToolAgent concept)
# NOTE: In ADK, tools are passed directly to the LLMAgent that needs them.
# The tool functions above act as the services a ToolAgent would wrap.

# 3.1. Task Decomposer Agent (MCP Step 1)
task_decomposer_agent = LlmAgent(
    name="TaskDecomposer",
    model=GEMINI_MODEL,
    instruction=(
        "You are the **Decomposer Agent**. Your task is to analyze the user's complex "
        "medical request and break it down into a list of discrete, structured JSON objects. "
        "The allowed task types are 'Set_Reminder' (needs med, time, freq) and "
        "'Check_Interaction' (needs med_A, med_B). Output ONLY the list of JSON tasks."
    ),
    output_key="decomposed_tasks" # Saves output to the Session State
)

# 3.2. Interaction Checker Agent (MCP Step 2)
interaction_checker = LlmAgent(
    name="InteractionChecker",
    model=GEMINI_MODEL,
    instruction=(
        "You are the **Interaction Checker**. Use the `query_meds_db` tool to find "
        "drug-drug interactions based on the given medicine names. Formulate the SQL query "
        "to check the 'Interactions' table. Report the interaction level and recommendation. "
        "Input format: ['MedA', 'MedB']. Output the findings clearly."
    ),
    tools=[db_query_tool], # Grants access to the SQLite tool
    output_key="interaction_results"
)

# 3.3. Schedule Planner Agent (MCP Step 3)
planner_agent = LlmAgent(
    name="SchedulePlanner",
    model=GEMINI_MODEL,
    instruction=(
        "You are the **Schedule Planner**. Use the `update_meds_db` tool to insert "
        "a new medication and schedule into the 'UserMeds' table based on the given JSON. "
        "The input is a JSON object with keys like: {med_name, dosage, frequency}. "
        "Formulate a correct SQL INSERT statement. Report success or failure."
    ),
    tools=[db_update_tool], # Grants access to the SQLite tool
    output_key="scheduling_results"
)

# 3.4. Final Synthesizer Agent (MCP Step 4)
synthesizer_agent = LlmAgent(
    name="ResultSynthesizer",
    model=GEMINI_MODEL,
    instruction=(
        "You are the **Final Assistant**. Your task is to gather all results from the "
        "Scheduler and Checker agents (available in the context) and combine them into a "
        "single, friendly, and comprehensive message for the user. Do not include any "
        "raw JSON or SQL. Focus on the reminders and interaction findings."
    ),
    # The output from previous agents is available in the shared Session State
    # and passed via ADK's context injection.
    output_key="final_response"
)
print("âœ… Component Agents configured")


# --- 4. The Health Coordinator (Sequential Agent) ---

health_coordinator = SequentialAgent(
    name="HealthCoordinator",
    description="Manages multi-step requests for medicine scheduling and interaction checking.",
    sub_agents=[
        # 1. Decompose the initial prompt
        task_decomposer_agent,        
        # 2. Check for interactions (requires data from decomposition)
        interaction_checker, 
        # 3. Set the schedule (requires data from decomposition)
        planner_agent,
        # 4. Synthesize the final answer
        synthesizer_agent
    ]
)

print("\n--- Medicine Assistant ADK System Defined ---")
print("Orchestrator: HealthCoordinator (SequentialAgent)")
print("Persistence: SQLite DB via Function Tools (Long-Term Memory)")
print("A2A/Context: Data passed via SequentialAgent's Session State")


# Define helper functions that will be reused throughout the notebook
async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str = None,
    session_name: str = "default"
):
    print(f"\n ### Session: {session_name}")

    # Get app name from the Runner
    app_name = runner_instance.app_name

    # Attempt to create a new session or retrieve an existing one
    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    except:
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )

    # Process queries if provided
    if user_queries:
        # Convert single query to list for uniform processing
        if type(user_queries) == str:
            user_queries = [user_queries]

        # Process each query in the list sequentially
        for query in user_queries:
            print(f"\nUser > {query}")

            # Convert the query string to the ADK Content format
            query = types.Content(role="user", parts=[types.Part(text=query)])

            # Stream the agent's response asynchronously
            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query
            ):
                # Check if the event contains valid content
                if event.content and event.content.parts:
                    # Filter out empty or "None" responses before printing
                    if (
                        event.content.parts[0].text != "None"
                        and event.content.parts[0].text
                    ):
                        result = event.content.parts[0].text
                        print(f"âœ… Final Response {result}")
    else:
        print("No queries!")


print("âœ… Helper functions defined.")


# --- 1. Setup ---

# Database and agents setup should be completed:
# - setup_database()
# - db_query_tool, db_update_tool
# - decomposer_agent, interaction_checker, planner_agent, synthesizer_agent
# - health_coordinator (SequentialAgent)

# --------------------------------------------------------
# 2. Define Runner Components
# --------------------------------------------------------

# Unique identifiers are crucial for memory/session management
APP_NAME = "MedicineAssistantApp"

# Session Service: Manages the short-term memory (conversation context/state)
# For production, this would be a persistent service (e.g., DatabaseSessionService)
session_service = InMemorySessionService()

# The Runner: Links the top-level agent (HealthCoordinator) to the services
runner = Runner(
    agent=health_coordinator,
    app_name=APP_NAME,
    session_service=session_service,
)


# --------------------------------------------------------
# 4. Execute the Application and Pass Input
# --------------------------------------------------------

# The initial input is passed via the `user_query` argument to the function.
user_request = (
    "I need to start taking my Lisinopril 10mg every day at 8:00 AM. "
    "Also, please check if I can take it safely with the Advil I sometimes use for headaches."
)

USER_ID="testuser"
print("\n\n=== Health Coordinator Final Output ===")
await run_session(
    runner, user_request, "test-session-01"
) 
print("=" * 50)

