import os
import asyncio
import time
import logging
import random
from threading import Thread
import httpx
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

from kaggle_secrets import UserSecretsClient
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner, Runner
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.adk.tools import google_search


print("âœ… All import completed successfully.")


# --- LOGGING SETUP ---
def setup_logging():
    # Create a custom logger
    logger = logging.getLogger("TravelSystem")
    logger.setLevel(logging.DEBUG) # Capture everything

    # 1. File Handler (Records EVERYTHING to a file)
    file_handler = logging.FileHandler('travel_agent_debug.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    file_handler.setFormatter(file_fmt)

    # 2. Console Handler (Shows only INFO/IMPORTANT stuff to screen)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter('ğŸ”¹ %(message)s') # Simple format for console
    console_handler.setFormatter(console_fmt)

    # Add handlers
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()
logger.info("âœ… Logging initialized. Detailed logs in 'travel_agent_debug.log'")


try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    logger.error(f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


# Model Name
MODEL_NAME = "gemini-2.5-flash-lite"


# --- MOCK DATABASES ---

# Structure: Key is (Origin, Destination) tuple
FLIGHT_DB = {
    ("BOM", "DXB"): [
        {"id": "EK-501", "airline": "Emirates", "price": 450, "time": "10:00 AM", "duration": "3h 15m"},
        {"id": "AI-910", "airline": "Air India", "price": 320, "time": "01:00 PM", "duration": "3h 30m"},
        {"id": "6E-088", "airline": "IndiGo", "price": 280, "time": "11:45 PM", "duration": "3h 20m"},
    ],
    ("BOM", "LHR"): [
        {"id": "BA-198", "airline": "British Airways", "price": 850, "time": "08:15 AM", "duration": "9h 40m"},
        {"id": "VS-355", "airline": "Virgin Atlantic", "price": 920, "time": "02:30 AM", "duration": "9h 50m"},
    ],
    ("DEL", "DXB"): [
        {"id": "SG-12", "airline": "SpiceJet", "price": 250, "time": "04:00 PM", "duration": "3h 45m"},
        {"id": "FZ-44", "airline": "FlyDubai", "price": 310, "time": "06:20 PM", "duration": "3h 50m"},
    ],
    ("JFK", "LHR"): [
        {"id": "AA-100", "airline": "American Airlines", "price": 600, "time": "06:00 PM", "duration": "7h 00m"},
        {"id": "DL-404", "airline": "Delta", "price": 650, "time": "08:00 PM", "duration": "6h 50m"},
    ]
}

# Structure: Key is City Code
HOTEL_DB = {
    "DXB": [
        {"id": "DXB-01", "name": "Atlantis The Royal", "price_night": 900, "rating": "5-Star", "amenities": "Luxury, Beach"},
        {"id": "DXB-02", "name": "Rove Downtown", "price_night": 120, "rating": "3-Star", "amenities": "City View, WiFi"},
        {"id": "DXB-03", "name": "Grand Hyatt Dubai", "price_night": 250, "rating": "5-Star", "amenities": "Pool, Spa"},
    ],
    "LHR": [ # London
        {"id": "LHR-99", "name": "The Savoy", "price_night": 750, "rating": "5-Star", "amenities": "Historic, Luxury"},
        {"id": "LHR-55", "name": "Premier Inn Heathrow", "price_night": 85, "rating": "3-Star", "amenities": "Airport Shuttle"},
    ],
    "BOM": [
        {"id": "BOM-11", "name": "The Taj Mahal Palace", "price_night": 350, "rating": "5-Star", "amenities": "Sea View, History"},
        {"id": "BOM-22", "name": "Trident Nariman Point", "price_night": 180, "rating": "5-Star", "amenities": "Business Center"},
    ],
    "NYC": [
        {"id": "NYC-01", "name": "The Plaza", "price_night": 950, "rating": "5-Star", "amenities": "Central Park View"},
        {"id": "NYC-02", "name": "Pod 51", "price_night": 110, "rating": "3-Star", "amenities": "Budget, Rooftop"},
    ]
}


# --- FASTAPI MOCK SERVER ---
app = FastAPI()

# Data Models (kept for documentation, though mostly unused by these specific endpoints)
class Flight(BaseModel):
    id: str
    origin: str
    dest: str
    price: float
    airline: str

class Hotel(BaseModel):
    id: str
    name: str
    price_night: float
    city: str

@app.get("/flights")
def get_flights(origin: str, dest: str):
    # Try to find the exact route in our DB
    key = (origin.upper(), dest.upper())
    
    if key in FLIGHT_DB:
        logger.info(f"API: Found {len(FLIGHT_DB[key])} flights for {key}")
        return FLIGHT_DB[key]
    
    # Fallback: Generate generic data if route not known
    logger.warning(f"API: Route {key} not in DB. Generating generic data.")
    return [
        {"id": f"GEN-101", "airline": "Generic Air", "origin": origin, "dest": dest, "price": random.randint(300, 800)},
        {"id": f"GEN-202", "airline": "Budget Gen", "origin": origin, "dest": dest, "price": random.randint(150, 400)},
    ]

@app.get("/hotels")
def get_hotels(city: str):
    #  Try to find the city in our DB
    city_code = city.upper()
    
    if city_code in HOTEL_DB:
        logger.info(f"API: Found {len(HOTEL_DB[city_code])} hotels in {city_code}")
        return HOTEL_DB[city_code]

    # Fallback: Generate generic data if city not known
    logger.warning(f"API: City {city_code} not in DB. Generating generic data.")
    return [
        {"id": f"HT-GEN-1", "name": "Generic City Hotel", "price_night": random.randint(100, 300), "city": city},
        {"id": f"HT-GEN-2", "name": "Generic Luxury Stay", "price_night": random.randint(400, 900), "city": city},
    ]

@app.post("/book")
def create_booking(type: str, id: str, user: str):
    # Simulate a booking transaction ID
    ref_id = f"{type.upper()}-{id}-{int(time.time())}"
    logger.info(f"ğŸ’° MOCK API: Booking Confirmed | {type} | {id} | {user} | Ref: {ref_id}")
    return {"status": "confirmed", "booking_ref": ref_id, "user": user}

# --- THREADED SERVER RUNNER ---
def run_server():
    # Disable access log to keep kaggle output clean
    config = uvicorn.Config(app, host="127.0.0.1", port=8001, log_level="warning")
    server = uvicorn.Server(config)
    server.run()

thread = Thread(target=run_server, daemon=True)
thread.start()
time.sleep(2)
logger.info("âœ… Mock Server running on port 8001")


API_BASE = "http://127.0.0.1:8001"

# --- CUSTOM TOOLS ---
async def search_flights(origin: str, destination: str):
    """Search for flights between origin and destination."""
    logger.info(f" TOOL CALL: search_flights({origin} -> {destination})")
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{API_BASE}/flights", params={"origin": origin, "dest": destination})
        return resp.json()

async def search_hotels(city: str):
    """Search for hotels in a specific city."""
    logger.info(f" TOOL CALL: search_hotels({city})")
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{API_BASE}/hotels", params={"city": city})
        return resp.json()

async def execute_booking(resource_type: str, resource_id: str, user_name: str):
    """Book a resource. resource_type must be 'flight' or 'hotel'."""
    logger.info(f"TOOL CALL: execute_booking({resource_type}, {resource_id}, {user_name})")
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{API_BASE}/book", params={"type": resource_type, "id": resource_id, "user": user_name})
        return resp.json()


# A. PLANNER AGENT (Uses Google Search)
planner_agent = Agent(
    name="travel_planner",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    tools=[google_search],
    instruction="""
    You are a TravelPlanner.
    
    INPUT DATA:
    You will see a text block starting with "user_prompt:".
    
    INSTRUCTIONS:
    1. Read the "User Name" line and address the user personally (e.g. "Hello Paplu!").
    2. Read "Origin", "Destination", and "Dates".
    3. Search for weather and create a plan.
    """
)


# B. PARALLEL AGENTS (Flight, Hotel)
flight_agent = Agent(
    name="flight_agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    tools=[search_flights],
    instruction="""
    You are a Flight Data Extractor.
    
    DATA SOURCE: 
    Look at the VERY FIRST message in the conversation history (the user_prompt).
    
    INSTRUCTIONS:
    1. Find the line that says "Origin: [Code]". Extract that code (e.g. BOM).
    2. Find the line that says "Destination: [Code]". Extract that code (e.g. DXB).
    3. CALL search_flights(origin=..., destination=...) immediately.
    """
)

hotel_agent = Agent(
    name="hotel_agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    tools=[search_hotels],
    instruction="""
    You are a Hotel Data Extractor.
    
    DATA SOURCE: 
    Look at the VERY FIRST message in the conversation history.
    
    INSTRUCTIONS:
    1. Find the line that says "Destination: [Code]". Extract that code.
    2. CALL search_hotels(city=...) immediately.
    """
)

sourcing_parallel = ParallelAgent(
    name="sourcing_parallel",
    sub_agents=[flight_agent, hotel_agent]
)


# C. EVALUATOR AGENT
evaluator_agent = Agent(
    name="evaluator_agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction="""
    You are the Evaluator.
    
    STEP 1: GET BUDGET
    - Look at the FIRST message in history.
    - Find the line "Budget: [Amount]". Parse it as a number.
    - Find the line "Passengers: [Count]". Parse it as a number.
    
    STEP 2: CALCULATE
    - Review the flight and hotel options found by the tools.
    - Total Cost = (Flight Price * Passengers) + (Hotel Price * Nights).
    
    STEP 3: OUTPUT
    - Select the best options.
    - Output strictly valid JSON with the structure:
      {
        "itinerary_proposal": {
           "status": "within_budget" or "over_budget",
           "total_cost": 0.0,
           "selected_flight": { "id": "...", "price": 0.0 },
           "selected_hotel": { "id": "...", "price": 0.0 }
        }
      }
    """
)


# D. BOOKING AGENT
booking_agent = Agent(
    name="booking_agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    tools=[execute_booking],
    instruction="""
    You are the Transaction Manager.
    
    YOUR GOAL: Finalize the booking using the correct User Name.
    
    STEP 1: RETRIEVE NAME
    - Look at the VERY FIRST message in the history.
    - Find the line that says "User Name: [Name]".
    - Extract that name (e.g. "Paplu").
    
    STEP 2: RETRIEVE SELECTIONS
    - Read the 'itinerary_proposal' JSON from the conversation history.
    - Extract the flight and hotel IDs.

    STEP 3: EXECUTE
    - Call execute_booking(resource_type="flight", ..., user_name="Paplu") 
    - Call execute_booking(resource_type="hotel", ..., user_name="Paplu")
    
    OUTPUT:
    - "Booking confirmed for Paplu! Ref IDs: ..."
    """
)


# E. TRACKER AGENT 
# Note: This agent will hallucinate status checks as it has no actual tools to check against the mock API.
tracker_agent = LoopAgent(
    name="tracker_agent",
    sub_agents=[
        Agent(
            name="status_checker",
            model=Gemini(model=MODEL_NAME, retry_options=retry_config),
            instruction="Report success."
        )
    ],
    max_iterations=2
)


# --- ORCHESTRATION FLOW ---

# PART 1: PLANNING & EVALUATION
planning_flow = SequentialAgent(
    name="planning_phase",
    sub_agents=[
        planner_agent,      # 1. Plan context
        sourcing_parallel,  # 2. Get Options concurrently
        evaluator_agent     # 3. Select best options and check budget
    ]
)

# PART 2: EXECUTION
execution_flow = SequentialAgent(
    name="execution_phase",
    sub_agents=[
        booking_agent,      # 4. Book selected items
        tracker_agent       # 5. Monitor
    ]
)


def get_user_travel_details():
    print("\nğŸ“� ENTER TRAVEL DETAILS (Press Enter to use defaults):")
    
    name = input("Name of the traveller: ").strip() or "Paplu"
    origin = input("Origin [e.g. BOM]: ").strip() or "BOM"
    dest = input("Destination [e.g. DXB, SGN]: ").strip() or "DXB"
    start = input("Depart Date [e.g. 2026-01-10]: ").strip() or "2026-01-10"
    end = input("Return Date [e.g. 2026-01-15]: ").strip() or "2026-01-15"
    
    budget_in = input("Budget [e.g. 800]: ").strip() or "800"
    passengers_in = input("Passengers [e.g. 1]: ").strip() or "1"
    
    # Construct the dictionary structure
    return {
        "name": name,
        "origin": origin,
        "destination": dest,
        "depart_date": start,
        "return_date": end,
        "budget": float(budget_in),
        "passengers": int(passengers_in)
    }


# --- MAIN EXECUTION ---
async def run_demo():
    # user_data = {
    #     "name" : "Paplu",
    #     "origin" : "BOM",
    #     "destination": "DXB", # Dubai
    #     "depart_date": "2026-01-10",
    #     "return_date" : "2026-01-15",
    #     "budget": 800.00,
    #     "passengers" : 1
    # }

    # STEP 1: Get Input Dynamically 
    user_data = get_user_travel_details()

    # 2. Formulate Prompt string (Grounding)
    prompt_string = f"""
    user_prompt:
    User Name: {user_data['name']}
    Origin: {user_data['origin']}
    Destination: {user_data['destination']}
    Dates: {user_data['depart_date']} to {user_data['return_date']}
    Budget: {user_data['budget']}
    Passengers: {user_data['passengers']}
    """
    
    MY_APP_NAME = "travel_booking_v1"
    # ----- Create a shared session service and ID -----
    session_service = InMemorySessionService()
    session_id = f"travel_session_{int(time.time())}"
    
    print(f"\n STARTING PHASE 1: PLANNING & SOURCING for {user_data['destination']}")
    runner1 = Runner(app_name=MY_APP_NAME, agent=planning_flow, session_service=session_service)
    # Pass user request combined into the initial state context
    await runner1.run_debug(prompt_string, session_id=session_id)

    # --- User Confirmation ---
    print("\n" + "="*60)
    print(" SYSTEM PROPOSAL (FROM EVALUATOR AGENT)")
    # In a real app, you would parse the JSON from the state here nicely.
    # For this console demo, we rely on the debug output above to see the JSON.
    print("Review the 'evaluator_agent' JSON output in the logs above.")
    print("="*60 + "\n")

    confirm = input("Do you authorize these bookings based on the proposal? (yes/no): ")

    if confirm.lower().strip() in ["yes", "y"]:
        print("\n STARTING PHASE 2: EXECUTION (BOOKING)")
        # CRITICAL: Pass the state from Phase 1 into Phase 2 to maintain context
        runner2 = Runner(app_name=MY_APP_NAME, agent=execution_flow, session_service=session_service)
        
        trigger_message = "User has confirmed the proposal. Proceed with bookings."
        await runner2.run_debug(trigger_message, session_id=session_id)
        
        print("\n WORKFLOW FINISHED")
    else:
        print("\n Workflow Terminated by User. No bookings made.")

# Execute in Jupyter/Kaggle environment
try:
    await run_demo()
except KeyboardInterrupt:
    print("\nProgram interrupted by user.")
finally:
    print("Shutting down.")

