import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("✅ Gemini API key setup complete.")
except Exception as e:
    print(
        f"🔑 Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


import json
import uuid
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

logging.getLogger("google_genai.types").setLevel(logging.ERROR)

from google.genai import types
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner, InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.agents import Agent, LlmAgent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.tools import AgentTool, FunctionTool, google_search, load_memory
from google.adk.plugins.logging_plugin import (
    LoggingPlugin,
) 

print("✅ ADK components imported successfully.")


APP_NAME = 'VillanovaFoodNews'
USER_ID = "test"


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


# Create Session Service
session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

print("✅ session service and memory service created.")


# Define helper functions that will be reused throughout the notebook
async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str = None,
    session_name: str = "default",
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
                        print(f"{MODEL_NAME} > ", event.content.parts[0].text)
    else:
        print("No queries!")


print("✅ Helper functions defined.")


# Save to memory after each update
async def mark_session_for_saving(callback_context):
    """Tags the session state to indicate this turn should be saved."""
    callback_context.state["should_save_history"] = True
    print("🚩 Session marked for saving.")

print("✅ Callback created.")


async def conditional_save_to_memory(callback_context):
    """Saves session ONLY if the specific flag was set during the turn."""
    
    # Check if our flag is True
    if callback_context.state.get("should_save_history") is True:
        print("💾 Saving food news update to memory...")
        
        # Perform the save
        await callback_context._invocation_context.memory_service.add_session_to_memory(
            callback_context._invocation_context.session
        )
        # CRITICAL: Reset the flag so the NEXT query isn't saved accidentally
        callback_context.state["should_save_history"] = False
    else:
        print("⏭️ Skipping save (not a food update).")


import sqlite3

DB_FILE = "food_news.db"

def init_db():
    """Creates the table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS food_news (
            id TEXT PRIMARY KEY,
            food TEXT,
            location TEXT,
            posted_time TEXT,
            expiry TEXT,
            last_updated TEXT,
            norm_food TEXT,
            norm_location TEXT
        )
    ''')
    conn.commit()
    conn.close()
    print("✅ Database initialized (SQLite).")

# Initialize
init_db()


def save_food_news_tool(food_type: str, location: str):
    """
    Saves food announcement to SQLite.
    Automatically handles Timezone (EST) and Deduplication.
    """
    # 1. Calculate Times
    est_zone = ZoneInfo("America/New_York")
    now_est = datetime.now(est_zone)
    expiry_est = now_est + timedelta(hours=2)

    current_time_str = now_est.strftime("%Y-%m-%d %I:%M %p")
    expiry_time_str = expiry_est.strftime("%Y-%m-%d %I:%M %p")

    # 2. Normalize
    norm_food = food_type.strip().lower()
    norm_loc = location.strip().lower()

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # 3. Check for existing record (Deduplication)
    # We query the DB instead of looping through a list
    cursor.execute('''
        SELECT id FROM food_news 
        WHERE norm_food = ? AND norm_location = ?
    ''', (norm_food, norm_loc))
    
    existing_record = cursor.fetchone()

    if existing_record:
        # --- UPDATE CASE ---
        record_id = existing_record[0]
        cursor.execute('''
            UPDATE food_news 
            SET expiry = ?, last_updated = ?
            WHERE id = ?
        ''', (expiry_time_str, current_time_str, record_id))
        conn.commit()
        conn.close()
        
        response = f"Updated existing record for {food_type} at {location}. New expiry: {expiry_time_str}."
        return response

    else:
        # --- INSERT CASE ---
        new_id = str(uuid.uuid4())[:8]
        cursor.execute('''
            INSERT INTO food_news 
            (id, food, location, posted_time, expiry, last_updated, norm_food, norm_location)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (new_id, food_type, location, current_time_str, expiry_time_str, current_time_str, norm_food, norm_loc))
        conn.commit()
        conn.close()
        
        response = f"Created new record (ID: {new_id}) for {food_type} at {location}."
        return response


def get_available_food_tool():
    """Reads all food from the SQLite DB."""
    conn = sqlite3.connect(DB_FILE)
    # We use row_factory to get dictionaries back instead of tuples
    conn.row_factory = sqlite3.Row 
    cursor = conn.cursor()
    
    cursor.execute("SELECT food, location, expiry FROM food_news")
    rows = cursor.fetchall()
    conn.close()
    
    # Convert SQLite rows to a Python list of dicts
    results = [dict(row) for row in rows]
    
    if not results:
        return "Database is empty."
    return json.dumps(results, indent=2)


# add food agent
# This agent will add the food based on the message send by an uploader. Each food_related message should
# have an expiry time of 2 hour, unlesss notified differently

food_news_update_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="FoodNewsUpdateAgent",
    tools=[FunctionTool(save_food_news_tool)], 
    instruction="""
    You are a food update agent. 
    1. Extract food_type and location from the user message.
    2. If missing, clarify.
    3. CRITICAL: You must call the `save_food_news_tool` with the extracted info. 
    
    Return a confirmation message to the user after the tool executes.
    """,
    after_agent_callback=mark_session_for_saving
)

print("✅ Food news updator created with conditional memory saving!")


food_news_search_agent = LlmAgent(
    name = "FoodNewsSearchAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction = """
    You are a food search agent.
    1. CALL the `get_available_food_tool` to see what food is listed.
    2. Return the data as a dictionary where keys are locations and values are lists of food.
    
    If the tool returns "No food...", you MUST return an empty dictionary

    Your output should be formatted as:
    {
        "<location_1>": ["food_1","food_2",....],
        "<location_2>": ["food_1","food_2",....]
    }
    """,
    tools = [FunctionTool(get_available_food_tool)], 
    output_key = "food_dict"
)


print("✅ Database Search Agent created.")


# add allergen search agent
allergen_search_agent = LlmAgent(
    model = Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name = "AllergenSearchAgent",
    instruction = """
    You are a food ingredient researcher. 
    You have access to the variable {food_dict} which contains the list of available food.

    STEP 1: Check {food_dict}.
    - If it is empty, "[]", or says "No food", DO NOT SEARCH. return an empty dictionary.
    
    STEP 2: If food exists, use `Google Search` to find common allergens for each item.
    
    Output Format (save to output_key):
    {
        "<location>": { "<food_name>": ["allergen1", "allergen2"] }
    }
    """,
    tools=[google_search],
    output_key = "food_allergen_dict"
)

print("✅ Allergen Search Agent created.")


# Response agent that would answer to the query
response_agent = LlmAgent(
    name="ResponseAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""Provides a short list of available food location, with caution of allergen as
    mentioned in {food_allergen_dict}

    Inform users that no food is available if {food_allergen_dict} is empty. DO NOT STAY QUIET.

    The answer can be formatted as:
    - <location_1> has [food_1], [food_2], .... may contain [list of allergen]...
    - <location_2> has [food_1], [food_2], .... may contain [list of allergen]...
    
    """,
    output_key="summary",
)

print("✅ aggregator_agent created.")


food_search_pipeline = SequentialAgent(
    name= "FoodSearchPipeline",
    sub_agents = [food_news_search_agent, allergen_search_agent, response_agent]
)

print("✅ Food Search pipeline created.")


root_agent = LlmAgent(
    name = 'Coordianator',
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction = """
    You are the coordinator for Villanova's food sustainability. Your job is to answer the users' 
    query by orchestrating the workflow.
    1. If the query is an annoucement of new available food on campus, you MUST call the `food_news_update_agent`
    2. If the query is asking for food availability on campus, you MUST call the `food_search_pipeline`.
    3. You MUST deny any request that does not fall into these 2 cases.

    DO NOT STAY SILENT
    """,
    tools = [
        AgentTool(food_news_update_agent),
        AgentTool(food_search_pipeline)
    ],
    after_agent_callback=conditional_save_to_memory
)

print("✅ root_agent created.")


auto_runner = Runner(
    agent=root_agent, 
    app_name=APP_NAME,
    session_service=session_service, 
    memory_service=memory_service,
    plugins=[
        LoggingPlugin()
    ],
)

print("✅ Runner created.")


# Test when calling with no food news
response = await auto_runner.run_debug("Is there any food on campus today?")

# Test with updating food
response = await auto_runner.run_debug("There is free tacos in Tolene hall")

# Test with updating food
response = await auto_runner.run_debug("There are free bagel, milk tea and Wawa roll in Vasey hall")

# Test with updating food (duplicate food and location)
response = await auto_runner.run_debug("There is free tacos in Tolene hall")

# Test with updating food (duplicate location)
response = await auto_runner.run_debug("There is free coffee and bagel in Tolene hall")

# Test with query for free food after updating 
response = await auto_runner.run_debug("Are there any available food today?")

