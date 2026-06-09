# @title 1. Setup & Installation
# Install the official Google Agent Development Kit (ADK) and GenAI SDK
#!pip install -q -U google-adk google-genai

import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


import uuid
from google.genai import types

# from google.adk.agents import LlmAgent
# from google.adk.models.google_llm import Gemini
# from google.adk.runners import Runner
# from google.adk.sessions import InMemorySessionService

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from google.adk.apps.app import App, ResumabilityConfig
from google.adk.tools.function_tool import FunctionTool

print("âœ… ADK components imported successfully.")


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


# @title 2. Define Tools
#from google.adk.tools import tool
import random

# --- TOOL 1: METEOROLOGIST'S TOOL (Custom Function) ---
def get_weather_forecast(city: str, travel_date: str) -> str:
    """
    Retrieves the weather forecast for a specific city and date.
    Use this to validate if outdoor activities are possible.
    
    Args:
        city: The name of the city (e.g., "Paris", "Tokyo").
        travel_date: The date of travel (e.g., "2025-10-12").
    
    Returns:
        A string describing the weather conditions.
    """
    # MOCK RESPONSE (Replace with real OpenWeatherMap API in production)
    print(f"\n[Tool] Checking weather for {city} on {travel_date}...")
    
    # Simulated weather logic
    if "London" in city or "Seattle" in city:
        return "Rainy, 12Â°C. High chance of precipitation."
    elif "Tokyo" in city:
        return "Clear skies, 22Â°C. Perfect for walking."
    elif "Chicago" in city:
        return "Windy and Rainy, 10Â°C."
    else:
        return "Partly cloudy, 18Â°C."

# --- TOOL 2: GOOGLE SEARCH ---
# We will import the built-in tool in the next cell


# @title 2. Define Tools (Updated with MCP)
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

# --- MCP TOOL: WEATHER SERVER (via NPX) ---
# We connect to the 'open-meteo-mcp' server package.
# Note: 'tool_filter' belongs in McpToolset, not StdioServerParameters.
weather_mcp_tool = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=[
                "-y",                # Auto-confirm installation
                "open-meteo-mcp",    # The package name for cmer81's server
            ]
        ),
        timeout=30
    ),
    # tool_filter=["get_forecast", "get_current_weather"]
)

print("âœ… MCP Tool created: Open-Meteo Weather Server")


async def auto_save_to_memory(callback_context):
    """Automatically save session to memory after each agent turn."""
    await callback_context._invocation_context.memory_service.add_session_to_memory(
        callback_context._invocation_context.session
    )


print("âœ… Callback created.")


from google.adk.agents import Agent, ParallelAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search
from google.adk.tools import load_memory
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService

MODEL_NAME = "gemini-2.5-flash-lite"

# Session Service: Handles immediate conversation history (RAM)
session_service = InMemorySessionService()
# Memory Service: Handles long-term storage (RAM for demo, but persists across sessions)
# Using VertexAiMemoryBankService will do the consolidation of memory automatically before storing. Can be used in production
memory_service = InMemoryMemoryService()

# --- 1. THE WORKER AGENTS (Leaf Nodes) ---

# Scout: Finds the raw data
scout_agent = Agent(
    name="scout_agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    tools=[google_search],
    description="Finds flight options, hotel prices, and local event schedules.",
    instruction="""
    You are the Scout.
    - Search for 3 distinct flight options with prices.
    - Search for 3 hotels with prices.
    - Search for 3 specific events matching the user's interests.
    - OUTPUT MUST BE DETAILED text with prices clearly listed.
    """
)

# Meteorologist: Checks the weather
meteorologist_agent = Agent(
    name="meteorologist_agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    tools=[get_weather_forecast], 
    description="Checks weather forecasts for the destination.",
    instruction="""
    You are the Meteorologist.
    - Check the weather for the user's destination and dates.
    - Use `get_weather_forecast()` to get weather information
    - Provide a "Go/No-Go" recommendation for outdoor activities.
    """
)

# CFO: Calculates the budget
cfo_agent = Agent(
    name="cfo_agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    tools=[load_memory],
    description="Analyzes costs against the budget.",
    instruction="""
    You are the CFO.
    - Review the flight and hotel options found by the Scout.
    - Calculate the Total Trip Cost (Flight + Hotel + $50/day food).
    - Compare against the user's stated budget.
    - If over budget, suggest which item to cut.
    """
)

# Writer: Compiles the final plan
writer_agent = Agent(
    name="writer_agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    tools=[load_memory],
    description="Writes the final itinerary.",
    after_agent_callback=auto_save_to_memory,
    instruction="""
    You are the Concierge.
    - Synthesize the Scout's options, the Meteorologist's warnings, and the CFO's budget check.
    - Write a beautiful, day-by-day itinerary for the user.
    - If the Meteorologist warned about rain, prioritize indoor events.
    """
)

# --- 2. THE ORCHESTRATION AGENTS (Day 1 Concepts) ---

# Parallel Agent: Runs Scout and Meteorologist at the same time
# This saves time because weather checking doesn't depend on flight prices.
context_gatherer = ParallelAgent(
    name="context_gatherer",
    sub_agents=[scout_agent, meteorologist_agent],
    description="Gathers all context (prices, events, weather) simultaneously."
)

# Sequential Agent: The Main Pipeline
# We force a strict order: Gather Info -> Check Budget -> Write Itinerary
nomad_navigator = SequentialAgent(
    name="nomad_navigator",
    sub_agents=[context_gatherer, cfo_agent, writer_agent],
    description="The main travel planning workflow."
)

print("âœ… Agent Architecture Created: Parallel(Scout+Weather) -> CFO -> Writer")


#For testing purpose
from google.adk.runners import InMemoryRunner
runner = InMemoryRunner(agent=scout_agent)
response = await runner.run_debug("Looking to go to London from 2025-10-09", verbose=True)


#For testing purpose
from google.adk.runners import InMemoryRunner
runner = InMemoryRunner(agent=meteorologist_agent)
response = await runner.run_debug("Looking for the weather in London for 2025-10-09", verbose=True)


from google.adk.runners import Runner
USER_ID = "default"
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


print("âœ… Helper functions defined.")


from google.adk.plugins.logging_plugin import (
    LoggingPlugin,
)

APP_NAME = "NomadNavigator"
runner = Runner(
    agent=nomad_navigator,
    app_name=APP_NAME,
    session_service=session_service, 
    memory_service=memory_service,
    plugins=[
        LoggingPlugin()  #common approach to logging
    ],
)

print("âœ… Runner created.")


user_query = """
I want to go to Chicago for a 3-day trip starting 2025-10-9.
My budget is strict: $1500 total.
I love architecture tours and river cruises.
Please build me a plan, but check the weather first!
"""
await run_session(
    runner,
    user_query,
    "session-01",
)


# import json

# # Create evaluation configuration with basic criteria
# eval_config = {
#     "criteria": {
#         "tool_trajectory_avg_score": 1.0,  # Perfect tool usage required
#         "response_match_score": 0.8,  # 80% text similarity threshold
#     }
# }

# with open("test_config.json", "w") as f:
#     json.dump(eval_config, f, indent=2)

# print("âœ… Evaluation configuration created!")
# print("\nðŸ“Š Evaluation Criteria:")
# print("â€¢ tool_trajectory_avg_score: 1.0 - Requires exact tool usage match")
# print("â€¢ response_match_score: 0.8 - Requires 80% text similarity")


# Create evaluation test cases that reveal tool usage and response quality problems
# test_cases = {
#     "eval_set_id": "home_automation_integration_suite",
#     "eval_cases": [
#         {
#             "eval_id": "living_room_light_on",
#             "conversation": [
#                 {
#                     "user_content": {
#                         "parts": [
#                             {"text": "Please turn on the floor lamp in the living room"}
#                         ]
#                     },
#                     "final_response": {
#                         "parts": [
#                             {
#                                 "text": "Successfully set the floor lamp in the living room to on."
#                             }
#                         ]
#                     },
#                     "intermediate_data": {
#                         "tool_uses": [
#                             {
#                                 "name": "set_device_status",
#                                 "args": {
#                                     "location": "living room",
#                                     "device_id": "floor lamp",
#                                     "status": "ON",
#                                 },
#                             }
#                         ]
#                     },
#                 }
#             ],
#         },
#         {
#             "eval_id": "kitchen_on_off_sequence",
#             "conversation": [
#                 {
#                     "user_content": {
#                         "parts": [{"text": "Switch on the main light in the kitchen."}]
#                     },
#                     "final_response": {
#                         "parts": [
#                             {
#                                 "text": "Successfully set the main light in the kitchen to on."
#                             }
#                         ]
#                     },
#                     "intermediate_data": {
#                         "tool_uses": [
#                             {
#                                 "name": "set_device_status",
#                                 "args": {
#                                     "location": "kitchen",
#                                     "device_id": "main light",
#                                     "status": "ON",
#                                 },
#                             }
#                         ]
#                     },
#                 }
#             ],
#         },
#     ],
# }


# with open("integration.evalset.json", "w") as f:
#     json.dump(test_cases, f, indent=2)


#!adk eval home_automation_agent integration.evalset.json --config_file_path=test_config.json --print_detailed_results

