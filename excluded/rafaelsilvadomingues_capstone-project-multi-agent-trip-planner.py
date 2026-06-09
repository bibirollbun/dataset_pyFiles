# Load API Key from Kaggle Secrets
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


import json
from typing import List, Dict, Union, Any
from google.genai import types
from google.adk.agents import Agent, LlmAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search, AgentTool, ToolContext, google_maps_grounding, FunctionTool, load_memory, preload_memory
from google.adk.runners import InMemoryRunner, Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.tools.tool_context import ToolContext
from google.adk.apps.app import App, EventsCompactionConfig
#from google.adk.sessions import DatabaseSessionService
from google import genai
from google.genai.errors import APIError
from google.adk.plugins.logging_plugin import LoggingPlugin

print("âœ… ADK components imported successfully.")


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


print("âœ… Helper functions defined.")


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)
print("âœ… Retry configuration defined.")


# Define the aplication name
APP_NAME = "default" 

# Define the session user ID
USER_ID = "default"  

#Define the session name
SESSION = "default"  

#Define Model to be used in this notebook
MODEL_NAME = "gemini-2.5-flash-lite"

# Create Memory Service
memory_service = InMemoryMemoryService()  # ADK's built-in Memory Service for development and testing

# Create Session Service
session_service = InMemorySessionService() # InMemorySessionService stores conversations in RAM (temporary)


async def auto_save_to_memory(callback_context):
    """Automatically save session to memory after each agent turn."""
    await callback_context._invocation_context.memory_service.add_session_to_memory(
        callback_context._invocation_context.session
    )


print("âœ… Callback created.")


# --- Configuration ---
# Use the gemini-2.5-flash-lite model for faster, cost-effective search/tool-use in a specialist agent
DISCOVERY_MODEL = "gemini-2.5-flash-lite" 

# --- 1. Define the Discovery Agent ---
discovery_agent = LlmAgent(
    # A clear, unique name for the agent
    name="DiscoveryAgent",
    
    # The LLM model powering the agent
    model=Gemini(model=DISCOVERY_MODEL,
                retry_options=retry_config),
    
    # Crucial: The instruction sets the agent's goal and persona
    instruction="""
        You are the 'Discovery Agent,' a highly focused travel researcher. 
        Your sole task is to find and research points of interest (POIs) and activities 
        based on the user's destination, duration, and interests.
        
        CRITICAL STEPS:
        1. Use the 'google_search' tool to find relevant information.
        2. Compile the search results into a structured, concise list of 12-16 POIs.
        3. For each POI, include its exact name, a brief description, and an estimated time 
           (e.g., '2 hours') needed for a visit.
        4. Do NOT plan the itinerary or determine the order. Just discover and format the data.
        5. Return ONLY the final structured list, do not add conversational text or commentary.
    """,
    
    # The tool the agent can use to perform its core function
    tools=[google_search],
    
    # A short description for the Coordinator Agent to use when calling this agent as a tool
    description="A specialist agent that uses Google Search to find relevant points of interest (POIs), activities, and their estimated visit duration for a specific trip."
)

print(f"âœ… Discovery Agent defined using {DISCOVERY_MODEL} with the google_search tool.")



# --- Configuration ---
# Use the gemini-2.5-flash-lite model for better complex reasoning required for optimization tasks
ROUTING_MODEL = "gemini-2.5-flash-lite" 

# --- 1. Define the Routing Agent ---
routing_agent = LlmAgent(
    # A clear, unique name for the agent
    name="RoutingAgent",
    
    # The LLM model powering the agent
    model=Gemini(model=ROUTING_MODEL,
                retry_options=retry_config),
    
    # Crucial: The instruction sets the agent's goal and persona
    instruction="""
        You are the 'Routing Agent,' an expert logistical and itinerary planner. 
        Your task is to take a list of unsequenced Points of Interest (POIs) 
        and determine the most efficient, logical, and time-optimized daily route.
        
        CRITICAL STEPS:
        1. Input will be a structured list of POIs (name, location/address, estimated visit time).
        2. Use the 'google_maps_grounding' tool to calculate the shortest travel time 
           (assuming walking or public transit) between all POIs to solve the routing problem.
        3. Sequence the POIs within the total trip duration (e.g., Day 1, Day 2) to minimize 
           total travel time and avoid back-tracking.
        4. For the final output, provide a structured JSON or list showing each day, the 
           sequenced POIs, the estimated time for the visit, and the estimated travel time 
           to the *next* location.
        5. Return ONLY the final structured itinerary data for the Coordinator Agent to format.
    """,
    
    # The tool the agent can use to perform its core function
    tools=[google_maps_grounding],
    
    # A short description for the Coordinator Agent to use when calling this agent as a tool
    description="A specialist agent that uses Google Maps to calculate the optimal, time-efficient route between a list of POIs and groups them into logical, sequenced daily itineraries."
)

print(f"âœ… Routing Agent defined using {ROUTING_MODEL} with the google_maps_grounding tool.")

# --- Example of How it Processes Data ---
# This agent takes raw POI data from the Discovery Agent (input) 
# and returns an optimized daily schedule (output).


# --- Configuration ---
COORDINATOR_MODEL = "gemini-2.5-flash-lite" 

# --- 1. Define the Coordinator Agent ---
coordinator_agent = LlmAgent(
    name="TripCoordinator",
    model=Gemini(model=COORDINATOR_MODEL,
                retry_options=retry_config),
    instruction="""
        You are the 'Trip Coordinator,' the main interface for the user. 
        Your goal is to manage the trip planning workflow using your specialist agents.
        
        CRITICAL 3-STEP WORKFLOW for the Lite model:
        1. DELEGATE DISCOVERY: Use the 'DiscoveryAgent' (your first tool) to get a structured 
           list of POIs based on the user's request (Name, destination, interests, duration).
        2. DELEGATE ROUTING: Pass the exact original POI list that the Discovery Agent returns
           (and only the list) to the 'RoutingAgent' (your second tool) to sequence and optimize the daily routes based on location.
        3. FORMAT FINAL PLAN: Take the sequenced output from the Routing Agent and present it 
           ALWAYS to the user as a clear, easy-to-read, day-by-day travel itinerary, with the estimated time for each visit and the expected price. 
           
           Do not add commentary until the final result is ready.
           If you are requested to present a previous trip plan from your memory,don't make any questions, just PRESENT IT EXACTLY AS IT WAS
    """,
    # The specialists are defined as tools for the Coordinator
    tools=[
        AgentTool(discovery_agent),  # Step 1
        AgentTool(routing_agent),    # Step 2
        preload_memory
        ],
        after_agent_callback=auto_save_to_memory,
        
    description="The main agent for orchestrating trip planning, delegating research and routing tasks to specialist agents."
)

print(f"âœ… Coordinator Agent defined using {COORDINATOR_MODEL} and its specialist agents as tools.")


runner = Runner(agent=coordinator_agent, 
                app_name=APP_NAME, 
                session_service=session_service, 
                memory_service=memory_service,
                plugins=[
                LoggingPlugin()
                ]
               )

print("âœ… Stateful agent with logging initialized!")



#Test 1: Tell the agent about a visit to London
# The callback will automatically save this to memory when the turn completes
await run_session(
    runner,
    [
        "Plan a 4-day itinerary in London focused on historical landmarks and museums.",    
    ],
    "London-Trip-session",
)


# Test 2: Ask about the trip to London in a NEW session (second conversation)
# The agent should retrieve the memory using preload_memory and answer correctly
await run_session(
    runner,
    [
        "What was the 4-day trip plan for London trip you provided before?",  
    ],
    "London-Trip-session-2", # Different session ID - proves memory works across sessions!
)

