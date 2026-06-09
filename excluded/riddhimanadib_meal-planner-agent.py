!pip install google-adk --quiet


# APP GLOBAL VARIABLES
APP_NAME = "meal-planner-app"  # Application
USER_ID = "default"  # User
SESSION_NAME = "random-session-name"
MODEL_NAME = "gemini-2.5-flash-lite"


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")


from google.adk.runners import Runner

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


print("âœ… Helper function: Run Session defined.")


from google.adk.agents import LlmAgent, Agent, SequentialAgent, ParallelAgent, LoopAgent

from google.adk.models.google_llm import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.google_search_tool import google_search

from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.sessions import DatabaseSessionService, InMemorySessionService
from google.adk.plugins.logging_plugin import LoggingPlugin

from google.adk.runners import InMemoryRunner

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from google.genai import types
from typing import List


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


# ---------------------------------------------------------
#   MEAL PLANNER AGENT SYSTEM  (NO TIME MCP)
# ---------------------------------------------------------

# 1. MealIntentAgent: Only proceed if user says â€œplan a mealâ€�
meal_intent_agent = Agent(
    name="MealIntentAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
You are a meal planner assistant.

Task:
- Determine if the user wants help planning a meal.
- Accept ANY phrasing meaning "plan a meal".
- If yes: output {"plan_meal": true}
- Otherwise: output {"plan_meal": false, "message": "You can only ask me to plan a meal."}

Output only valid JSON.
""",
    output_key="meal_intent",
)
print("âœ… meal_intent_agent created.")

# 2. LoadMemoryAgent: Loads last meal prepared
load_memory_agent = Agent(
    name="LoadMemoryAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
Retrieve the last saved meal from memory.
Output JSON:
{"last_meal": "<meal name or null>"}
""",
    output_key="last_meal_info",
)
print("âœ… load_memory_agent created.")

# 3. MealPlannerAgent: No longer depends on time, just suggests a meal
meal_planner_agent = Agent(
    name="MealPlannerAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
Given the last meal: {last_meal_info}

Plan a simple new meal that is:
- easy to cook,
- different from the last meal (if any).

Return JSON: {"planned_meal": "<meal name>"}.
""",
    output_key="planned_meal",
)
print("âœ… meal_planner_agent created.")

# ---------------------------------------------------------
# PARALLEL AGENTS (Calories + Prep Time)
# ---------------------------------------------------------

calorie_agent = Agent(
    name="CalorieAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
Search online to estimate calories for: {planned_meal}.
Return JSON: {"calories": "<approx calories>"}.
""",
    tools=[google_search],
    output_key="calorie_info",
)
print("âœ… calorie_agent created.")

prep_time_agent = Agent(
    name="PrepTimeAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
Search online for typical preparation time for: {planned_meal}.
Assume 2 servings.
Return JSON: {"prep_time": "<minutes>"}.
""",
    tools=[google_search],
    output_key="prep_time_info",
)
print("âœ… prep_time_agent created.")

parallel_info_agents = ParallelAgent(
    name="ParallelInfoAgents",
    sub_agents=[calorie_agent, prep_time_agent],
)
print("âœ… parallel_info_agents created.")

# ---------------------------------------------------------
# FINAL SUMMARY AGENT
# ---------------------------------------------------------

summary_agent = Agent(
    name="SummaryAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
Create a friendly summary using:

Last meal: {last_meal_info}
New meal: {planned_meal}
Calories: {calorie_info}
Prep time: {prep_time_info}

Ask the user: â€œDoes this meal, calorie estimate, and prep time look good?â€�
Return JSON:
{"summary": "<text>"}.
""",
    output_key="meal_summary",
)
print("âœ… summary_agent created.")

# ---------------------------------------------------------
# OPTIONAL: Saving meal after confirmation
# ---------------------------------------------------------

save_meal_agent = Agent(
    name="SaveMealAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
If user confirms with yes/yep/sure/ok/etc:
Save {planned_meal} into memory under key 'last_meal'.
Output JSON {"saved": true}.
Otherwise: {"saved": false}.
""",
    output_key="save_status",
)
print("âœ… save_meal_agent created.")

# ---------------------------------------------------------
# FULL WORKFLOW: Sequence
# ---------------------------------------------------------

root_agent = SequentialAgent(
    name="MealPlannerSystem",
    sub_agents=[
        meal_intent_agent,
        load_memory_agent,
        meal_planner_agent,
        parallel_info_agents,
        summary_agent,
        save_meal_agent,
    ],
)
print("âœ… MealPlannerSystem created.")


# Run
runner = InMemoryRunner(agent=root_agent)
response = await runner.run_debug("help me plan a meal")
print(response)


print(await runner.run_debug("sure, looks good"))


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


# Re-define our app with Events Compaction enabled
app_compacting = App(
    name=APP_NAME,
    root_agent=root_agent,
    plugins=[LoggingPlugin()],
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=3,  # Trigger compaction every 3 invocations
        overlap_size=1,  # Keep 1 previous turn for context
    ),
)

db_url = "sqlite:///my_agent_data.db"  # Local SQLite file
session_service = DatabaseSessionService(db_url=db_url)
# session_service = InMemorySessionService()

# Create a new runner for our upgraded app
runner_compacting = Runner(app=app_compacting, session_service=session_service)

print("âœ… Research App upgraded with Events Compaction!")


print("âœ… Stateful agent initialized!")
print(f"   - Application: {APP_NAME}")
print(f"   - User: {USER_ID}")
print(f"   - Using: {session_service.__class__.__name__}")


await run_session(
    runner_compacting,
    ["hello",
     "okay lets plan a meal"],
    SESSION_NAME,
)


import sqlite3

def check_data_in_db():
    with sqlite3.connect("my_agent_data.db") as connection:
        cursor = connection.cursor()
        result = cursor.execute(
            "select app_name, session_id, author, content from events"
        )
        print([_[0] for _ in result.description])
        for each in result.fetchall():
            print(each)


check_data_in_db()


# Get the final session state
final_session = await session_service.get_session(
    app_name=runner_compacting.app_name,
    user_id=USER_ID,
    session_id=SESSION_NAME,
)

print("--- Searching for Compaction Summary Event ---")
found_summary = False
for event in final_session.events:
    # Compaction events have a 'compaction' attribute
    if event.actions and event.actions.compaction:
        print("\nâœ… SUCCESS! Found the Compaction Event:")
        print(f"  Author: {event.author}")
        print(f"\n Compacted information: {event}")
        found_summary = True
        break

if not found_summary:
    print(
        "\nâ�Œ No compaction event found. Try increasing the number of turns in the demo."
    )


import json

# Create evaluation configuration with basic criteria
eval_config = {
    "criteria": {
        "tool_trajectory_avg_score": 1.0,  # Perfect tool usage required
        "response_match_score": 0.8,  # 80% text similarity threshold
    }
}

with open("meal-planner-agent/test_config.json", "w") as f:
    json.dump(eval_config, f, indent=2)

print("âœ… Evaluation configuration created!")


# Create evaluation test cases that reveal tool usage and response quality problems
test_cases = {
    "eval_set_id": "home_automation_integration_suite",
    "eval_cases": [
        {
            "eval_id": "living_room_light_on",
            "conversation": [
                {
                    "user_content": {
                        "parts": [
                            {"text": "Please turn on the floor lamp in the living room"}
                        ]
                    },
                    "final_response": {
                        "parts": [
                            {
                                "text": "Successfully set the floor lamp in the living room to on."
                            }
                        ]
                    },
                    "intermediate_data": {
                        "tool_uses": [
                            {
                                "name": "set_device_status",
                                "args": {
                                    "location": "living room",
                                    "device_id": "floor lamp",
                                    "status": "ON",
                                },
                            }
                        ]
                    },
                }
            ],
        },
        {
            "eval_id": "kitchen_on_off_sequence",
            "conversation": [
                {
                    "user_content": {
                        "parts": [{"text": "Switch on the main light in the kitchen."}]
                    },
                    "final_response": {
                        "parts": [
                            {
                                "text": "Successfully set the main light in the kitchen to on."
                            }
                        ]
                    },
                    "intermediate_data": {
                        "tool_uses": [
                            {
                                "name": "set_device_status",
                                "args": {
                                    "location": "kitchen",
                                    "device_id": "main light",
                                    "status": "ON",
                                },
                            }
                        ]
                    },
                }
            ],
        },
    ],
}


with open("meal-planner-agent/integration.evalset.json", "w") as f:
    json.dump(test_cases, f, indent=2)
print("âœ… Evaluation test cases created")

print("\nğŸ§ª Test scenarios:")
for case in test_cases["eval_cases"]:
    user_msg = case["conversation"][0]["user_content"]["parts"][0]["text"]
    print(f"â€¢ {case['eval_id']}: {user_msg}")


# !adk eval meal-planner-agent meal-planner-agent/integration.evalset.json --config_file_path=meal-planner-agent/test_config.json --print_detailed_results


# Define helper functions that will be reused throughout the notebook

from IPython.core.display import display, HTML
from jupyter_server.serverapp import list_running_servers

# Gets the proxied URL in the Kaggle Notebooks environment
def get_adk_proxy_url():
    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    ADK_PORT = "8000"

    servers = list(list_running_servers())
    if not servers:
        raise Exception("No running Jupyter servers found.")

    baseURL = servers[0]['base_url']

    try:
        path_parts = baseURL.split('/')
        kernel = path_parts[2]
        token = path_parts[3]
    except IndexError:
        raise Exception(f"Could not parse kernel/token from base URL: {baseURL}")

    url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{ADK_PORT}"
    url = f"{PROXY_HOST}{url_prefix}"

    styled_html = f"""
    <div style="padding: 15px; border: 2px solid #f0ad4e; border-radius: 8px; background-color: #fef9f0; margin: 20px 0;">
        <div style="font-family: sans-serif; margin-bottom: 12px; color: #333; font-size: 1.1em;">
            <strong>âš ï¸� IMPORTANT: Action Required</strong>
        </div>
        <div style="font-family: sans-serif; margin-bottom: 15px; color: #333; line-height: 1.5;">
            The ADK web UI is <strong>not running yet</strong>. You must start it in the next cell.
            <ol style="margin-top: 10px; padding-left: 20px;">
                <li style="margin-bottom: 5px;"><strong>Run the next cell</strong> (the one with <code>!adk web ...</code>) to start the ADK web UI.</li>
                <li style="margin-bottom: 5px;">Wait for that cell to show it is "Running" (it will not "complete").</li>
                <li>Once it's running, <strong>return to this button</strong> and click it to open the UI.</li>
            </ol>
            <em style="font-size: 0.9em; color: #555;">(If you click the button before running the next cell, you will get a 500 error.)</em>
        </div>
        <a href='{url}' target='_blank' style="
            display: inline-block; background-color: #1a73e8; color: white; padding: 10px 20px;
            text-decoration: none; border-radius: 25px; font-family: sans-serif; font-weight: 500;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.2s ease;">
            Open ADK Web UI (after running cell below) â†—
        </a>
    </div>
    """

    display(HTML(styled_html))

    return url_prefix

print("âœ… Helper function: get_adk_proxy_url() defined.")

url_prefix = get_adk_proxy_url()


!adk web --log_level DEBUG --url_prefix {url_prefix}


# Check the DEBUG logs from the broken agent
# print("ğŸ”� Examining web server logs for debugging clues...\n")
# !cat logger.log

