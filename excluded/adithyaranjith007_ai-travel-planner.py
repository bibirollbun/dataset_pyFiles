# ----------- Install Agent Development Kit(ADK) ----------
!pip install google-adk


!pip install amadeus -q


# Handling warnings

import warnings
warnings.filterwarnings('ignore',module = 'google_genai')

import logging

# Set the google_genai logger level to ERROR, suppressing warnings
logging.getLogger('google_genai.types').setLevel(logging.ERROR)

# Also clear all warnings temporarily (less surgical)
warnings.filterwarnings('ignore', module='google_genai')


# ----------- Import ADK Components ----------

# --- Utilities ---
import json
import os

# Core ADK
from google import genai
from google.adk.models.google_llm import Gemini

# Core Agent & Runner
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner

# Tools
from google.adk.tools import FunctionTool
from google.adk.tools import google_search
from amadeus import Client, ResponseError

# Memory + Sessions
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService     

# Warnings
import warnings
warnings.filterwarnings('ignore',module = 'google_genai')


# ---- IMPLEMENTING OBSERVABILITY ----
import logging
import sys

# Setup logging to print to the notebook output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Get a logger instance for your application
log = logging.getLogger('TravelPlanner')


# ----------- Configure your Gemini API key ----------
from kaggle_secrets import UserSecretsClient
try:
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_Key")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


import requests
# --- Creating a Weather Tool (Manual API Call) ---
def get_current_weather(city: str) -> str:
    """
    Retrieves the current weather and temperature for a specified city. 
    Use this tool when the user asks about the weather in a destination city.
    Args:
        city (str): The name of the city (e.g., "Paris", "Tokyo").
    Returns:
        str: A JSON string containing the weather details or an error message.
    """
    # Replace with your actual key or use os.environ
    OPENWEATHER_KEY = "OPENWEATHER_API_KEY"
    if OPENWEATHER_KEY == "OPENWEATHER_API_KEY":
        return f'{{"city": "{city}", "temp": 22, "condition": "sunny", "note": "Using mock data."}}'

    # Example API Call Structure (requires proper key setup)
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_KEY}&units=metric"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status() # Raises an exception for bad status codes
        data = response.json()
        temp = data['main']['temp']
        condition = data['weather'][0]['description']
        return f'{{"city": "{city}", "temp": {temp}, "condition": "{condition}"}}'
    except Exception as e:
        return f'{{"error": "Failed to retrieve weather data: {str(e)}"}}'


# ---- Setting up Amadeus Client ---
# It automatically handles the authentication token exchange
def get_amadeus_client():
    try:
        api_key = os.environ.get("AMADEUS_API_KEY")
        api_secret = os.environ.get("AMADEUS_API_SECRET")
        if not api_key or not api_secret:
            return None
        return Client(client_id=api_key, client_secret=api_secret)
    except Exception as e:
        print(f"Amadeus Client Error: {e}")
        return None


# --- Creating the Flight Search Function ---
def search_flights(origin: str, destination: str, outbound_date: str, return_date: str) -> str:
    """
    Searches for real flight offers using Amadeus API.
    Args:
        origin (str): IATA code for origin city (e.g., "NYC", "LON", "DEL").
        destination (str): IATA code for destination (e.g., "PAR", "DXB").
        departure_date (str): Date in YYYY-MM-DD format (e.g., "2025-05-01").
    """
    # ğŸŸ¢ LOG ENTRY: Log the exact inputs received by the tool
    log.info(f"Tool called: Searching R/T flights. {origin} -> {destination}. Outbound: {outbound_date}, Return: {return_date}")
    
    client = get_amadeus_client()
    
    # Fallback to mock if keys are missing (prevents crashes)
    if not client:
        # ğŸŸ  LOG ENTRY: Log when the fallback is triggered
        log.warning("Amadeus API client is NOT initialized. Falling back to mock data.")
        return "Error: Amadeus API keys not found. Using Mock: Flight found for $450."

    try:
        # ğŸŸ¢ LOG ENTRY: Indicate the real API call is starting
        log.info("Initiating REAL Amadeus API call for flight offers.")

        # REAL API CALL
        response = client.shopping.flight_offers_search.get(
            originLocationCode=origin,
            destinationLocationCode=destination,
            departureDate=departure_date,
            adults=1,
            max=3 # Limit to top 3 cheapest flights to save tokens
        )
        
        # Parse the complex JSON response
        offers = response.data
        if not offers:
            return f"No flights found from {origin} to {destination} on {departure_date}."
            
        # ğŸŸ¢ LOG ENTRY: Log the number of successful results
        log.info(f"Successfully retrieved {len(offers)} flight offers.")
        
        results = []
        for offer in offers:
            price = offer['price']['total']
            currency = offer['price']['currency']
            # Extract airline from the first segment
            airline = offer['itineraries'][0]['segments'][0]['carrierCode']
            results.append(f"- Flight via {airline}: {price} {currency}")

        # ğŸŸ¢ LOG ENTRY: Log the final, simplified string being returned to the LLM
        log.debug(f"Returning simplified flight results to Planner Agent: {results[0]}")

        return "\n".join(results)

    except ResponseError as error:
        # ğŸ”´ LOG ENTRY: Critical failure from the API
        log.error(f"Amadeus API Response Error (Code {error.response.status_code}): {error}")
        return f"Amadeus API Error: {error}"
    
    except Exception as e:
        # ğŸ”´ LOG ENTRY: Catch all unexpected errors
        log.critical(f"UNEXPECTED ERROR in search_flights: {str(e)}", exc_info=True)
        return f"Unexpected Error: {str(e)}"
        
        
   


# Wrap the functions into ADK FunctionTools
weather_tool = FunctionTool(get_current_weather)
flight_tool = FunctionTool(search_flights)


# --- Planner Agent (Corrected for Amadeus IATA Codes) ---
planner_agent = LlmAgent(
    name="PlannerAgent",
    model="gemini-2.0-flash-001",
    instruction="""
    You are a Senior Travel Planner. Your goal is to create a complete, detailed, and ready-to-book 3-day itinerary.

    1. GATHER DATA: Receive the destination, origin, and travel dates from the user.
    
    2. REQUIRED TOOL USE: 
        a. Use 'weather_tool' to check the forecast.
        b. CRITICAL STEP: Before calling 'search_flights', you MUST first convert the Origin and Destination city names into their standard 3-letter IATA airport codes (e.g., London -> LHR, Paris -> CDG).
        c. Use 'search_flights' with the IATA codes and dates.
    
    3. FINAL OUTPUT CREATION:
        a. You MUST integrate the exact **flight price** and **weather conditions** into the final plan.
        b. Generate a detailed, day-by-day itinerary based on the weather and flight info.
        c. Include popular tourist recommendations and local dining suggestions
    """,
    tools=[weather_tool, flight_tool] # Assuming flight_tool wraps the Amadeus code
)


# ---  Creating The Reviewer Agent (The Quality Control) ---
reviewer_agent = LlmAgent(
    name="ReviewerAgent",
    model="gemini-2.0-flash-001",
    instruction="""
    You are a Strict Travel Critic.
    1. Review the itinerary provided by the Planner Agent.
    2. Check for LOGIC ERRORS (e.g., "Dinner is planned for 3 PM").
    3. Check for MISSING INFO (e.g., "Did they check the weather? Is the flight price included?").
    4. If the plan is good, output: "FINAL PLAN APPROVED: [Summary of Plan]"
    5. If the plan is bad, output: "REJECTED: [Specific feedback]"
    """,
    tools=[] # Reviewer needs no tools, just its brain
)


# --- Creating The Sequential Workflow ---
planning_workflow = SequentialAgent(
    name="PlanningWorkflow",
    description="A workflow that plans a trip and then reviews it for errors.",
    # This list defines the order of execution
    sub_agents=[planner_agent, reviewer_agent] 
)


import asyncio
from google.genai.types import Content, Part

# ---  Define globals ---
APP_NAME = "TravelPlannerApp"
USER_ID = "local_user"
session_service = InMemorySessionService()

# --- Define the tool function as ASYNC ---
async def call_planning_workflow(user_request: str) -> str:  # â†� Made async
    """
    Triggers the planning team (Planner + Reviewer).
    """
    # LOG ENTRY: Start of the nested workflow
    log.info(f"Tool: Planning Workflow started. Request: '{user_request[:50]}...'")
    
    print(f" [Tool Triggered] Starting Planning Workflow for: {user_request[:30]}...")

    sub_runner = Runner(
        agent=planning_workflow,
        app_name=APP_NAME,
        session_service=session_service
    )
    
    sub_session_id = f"sub_task_{hash(user_request)}"
    
    # Use await instead of asyncio.run (already in async context!)
    await session_service.create_session(
        session_id=sub_session_id,
        user_id=USER_ID,
        app_name=APP_NAME
    )

    sub_input = Content(role="user", parts=[Part(text=user_request)])
    
    # Use run_async instead of run (since we're in async context)
    events = sub_runner.run_async(
        user_id=USER_ID,
        session_id=sub_session_id,
        new_message=sub_input
    )
    
    result_text = ""
    async for event in events:  # â†� Use async for
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, 'text') and part.text:
                    result_text += part.text
            
    print(f"[Tool Complete] Plan generated.")
    
    # LOG ENTRY: End of the nested workflow
    log.info("Tool: Planning Workflow successfully executed the Sequential Agent.")
    return result_text
    


# --- Create orchestrator ---
orchestrator = LlmAgent(
    name="Orchestrator",
    model="gemini-2.0-flash-001",
    instruction="""
    You are the Head Concierge.
    1. Greeting: Welcome the user.
    2. Analysis: If the user wants a trip plan, use the 'call_planning_workflow' tool.
    3. Output: Present the final approved plan to the user clearly.
    """,
    tools=[FunctionTool(call_planning_workflow)]
)

# --- Create main runner ---
main_runner = Runner(
    agent=orchestrator,
    app_name=APP_NAME,
    session_service=session_service
)

print("Setup complete!")


from google.adk.runners import Runner
from google.genai import types

session_service = InMemorySessionService()

# --- Initialize the main Runner with the Orchestrator as the entry point ---
main_runner = Runner(agent=orchestrator, app_name="TravelPlannerApp", session_service=session_service)

await session_service.create_session(
    session_id="local_session",
    user_id="local_user",
    app_name="TravelPlannerApp"
)

# Ask the question inside the runner's .run() method.
user_query = "Plan a 3-day trip to Paris. I need flights from London on May 10th."

# LOG ENTRY: Start of the main orchestration
log.info(f"--- STARTING ORCHESTRATOR RUN --- Query: '{user_query}'")
print(f"--- Running Orchestrator with Query: '{user_query}' ---")

user_content = types.Content(
    role="user",
    parts=[types.Part(text=user_query)]
)

# Pass required keyword arguments: user_id, session_id, new_message
# Capture the generator returned by .run()
response_generator = main_runner.run_async(
    user_id="local_user",
    session_id="local_session",
    new_message=user_content
)

print("\n FINAL SYSTEM OUTPUT (From Orchestrator) ")

# Iterate over the generator to extract text from events
async for event in response_generator:
 # Check if the event has text content
    if event.content and event.content.parts:
        for part in event.content.parts:
            if hasattr(part, 'text') and part.text:
                print(part.text, end="", flush=True)

# LOG ENTRY: Final completion
log.info("--- ORCHESTRATOR RUN COMPLETE --- Check output for final itinerary.")
print() # Print a final newline


# --- AGENT EVALUATION SCRIPT ---
import asyncio
from google.genai.types import Content, Part

# 1. Define the Test Case
test_query = "Plan a 3-day trip to London starting October 5th, flying from Paris."
print(f"ğŸ§ª TESTING AGENT WITH QUERY: '{test_query}'")

# 2. Run the Agent
# We need to capture the text output first
final_response_text = ""

try:
    # Setup inputs
    eval_input = Content(parts=[Part(text=test_query)])
    
    # Run Async (The method we know works)
    # Note: Ensure you use a unique session ID for the test to avoid history pollution
    eval_session_id = "eval_session_01"
    await session_service.create_session(
        session_id=eval_session_id, 
        user_id=USER_ID, 
        app_name=APP_NAME
    )
    
    response_generator = main_runner.run_async(
        user_id=USER_ID,
        session_id=eval_session_id,
        new_message=eval_input
    )
    
    # Extract text from the stream
    print("   ... Agent is thinking ...")
    async for event in response_generator:
        if hasattr(event, 'content') and event.content:
            if hasattr(event.content, 'parts'):
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        final_response_text += part.text
            elif isinstance(event.content, str):
                final_response_text += event.content

except Exception as e:
    print(f"TEST FAILED TO RUN: {e}")

# 3. Manual Evaluation Logic (Your Logic)
# Now that we have the text, we check it against our criteria
print("\n ANALYZING RESPONSE...")

response_lower = final_response_text.lower()

# Check for Weather (Did it call the weather tool?)
has_weather = "weather" in response_lower or "temperature" in response_lower or "degree" in response_lower

# Check for Flights (Did it call the flight tool?)
has_flights = "flight" in response_lower or "airline" in response_lower or "airport" in response_lower

# Check for Mock/Real Data (Did it actually give a price?)
has_price = "$" in response_lower or "usd" in response_lower

# 4. Generate Report
print("\n EVALUATION REPORT ")
print(f"1. Checked Weather?      {'PASS' if has_weather else 'FAIL'}")
print(f"2. Checked Flights?      {'PASS' if has_flights else 'FAIL'}")
print(f"3. Included Price?       {'PASS' if has_price else 'FAIL'}")
print(f"4. Response Length:      {len(final_response_text)} characters")

if has_weather and has_flights:
    print("\n RESULT: AGENT PASSED ALL CORE CHECKS")
else:
    print("\n RESULT: AGENT MISSED SOME REQUIREMENTS")




