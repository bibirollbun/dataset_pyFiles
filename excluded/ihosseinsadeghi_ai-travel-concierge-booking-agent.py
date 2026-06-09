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


!pip install google-adk


# Ensure you have added your 'GOOGLE_API_KEY' in the Kaggle Secrets Add-on.
import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(f"ğŸ”‘ Authentication Error: {e}")
    print("Hint: Go to 'Add-ons' -> 'Secrets' and add a secret named 'GOOGLE_API_KEY'.")


import uuid
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.function_tool import FunctionTool
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext

print("âœ… ADK components imported successfully.")


# --- TOOL 1: Flight Lookup (Stateless/Info Protocol) ---
async def get_flight_options(origin: str, destination: str) -> dict:
    """
    [Protocol: DISCOVERY]
    Searches for available flights. Returns a list of options.
    ZERO-LATENCY LOGIC: Uses local dictionary lookup.
    """
    print(f"âœˆï¸� LOG: Discovery Protocol Triggered for: {origin} -> {destination}")
    
    # Mock Database for stable testing
    flights_db = [
        {"id": "FL-101", "airline": "SkyWays", "time": "08:00 AM", "price": "$120", "route": f"{origin}-{destination}"},
        {"id": "FL-202", "airline": "EagleJet", "time": "04:30 PM", "price": "$145", "route": f"{origin}-{destination}"},
        {"id": "FL-303", "airline": "NightOwl", "time": "11:00 PM", "price": "$90",  "route": f"{origin}-{destination}"}
    ]
    
    return {
        "status": "Found",
        "options": flights_db,
        "message": f"I found 3 flights from {origin} to {destination}."
    }

# --- TOOL 2: Hotel Search (Stateless/Info Protocol) ---
async def get_hotel_options(city: str, tier: str = "standard") -> dict:
    """
    [Protocol: DISCOVERY]
    Searches for hotels in a specific city and tier (budget/luxury).
    """
    print(f"ğŸ�¨ LOG: Hotel Lookup Triggered for: {city} ({tier})")
    
    # Mock Data
    if tier.lower() == "luxury":
        return {"status": "Found", "hotels": ["Grand Plaza (5*)", "Royal Suites (5*)"], "price_range": "$300+"}
    else:
        return {"status": "Found", "hotels": ["City Inn (3*)", "Travelodge (3*)"], "price_range": "$80-$120"}

print("âœ… Info Tools Defined.")


# --- TOOL 3: Ticket Booking (Stateful/Booking Protocol) ---
async def book_ticket(flight_id: str, passenger_name: str, tool_context: ToolContext) -> dict:
    """
    [Protocol: BOOKING]
    Reserves a specific flight. 
    STATEFUL LOGIC: Pauses for HITL (Human-in-the-Loop) verification before charging money.
    """
    
    # ğŸ›‘ PHASE 1: PAUSE FOR APPROVAL (HITL)
    if not tool_context.tool_confirmation:
        print(f"ğŸ’³ LOG: Intercepting Booking Request for '{flight_id}'...")
        print(f"â�¸ï¸� SYSTEM PAUSED: Waiting for user verification...")
        
        # This triggers the pause. The Runner will stop here and return control to the user.
        tool_context.request_confirmation(
            hint=f"Please confirm booking for flight {flight_id} under name {passenger_name}.",
            payload={"flight_id": flight_id, "passenger": passenger_name}
        )
        return {
            "status": "pending_confirmation",
            "message": f"I need your approval to charge the card for flight {flight_id}."
        }

    # âœ… PHASE 2: RESUME & EXECUTE
    if tool_context.tool_confirmation.confirmed:
        booking_ref = f"REF-{uuid.uuid4().hex[:6].upper()}"
        print(f"ğŸ’³ LOG: CONFIRMED. Transaction processed. Ref: {booking_ref}")
        return {
            "status": "BOOKED",
            "reference": booking_ref,
            "message": f"Booking confirmed! Your reference number is {booking_ref}."
        }
    else:
        print("ğŸ’³ LOG: Booking cancelled by user.")
        return {"status": "CANCELLED", "message": "Booking request cancelled."}

print("âœ… Booking Tool Defined.")


print("--- ğŸ§ª TEST: Unit Testing 'book_ticket' (Pause & Resume) ---")

# Mock Classes for Testing
class MockToolConfirmation:
    def __init__(self, confirmed=False):
        self.confirmed = confirmed
        
class MockToolContext:
    def __init__(self, confirmed=False):
        self.tool_confirmation = MockToolConfirmation(confirmed) if confirmed is not None else None
    def request_confirmation(self, hint, payload):
        print(f"   [Mock Engine] Pausing... Hint: '{hint}'")

# Scenario A: Pause
print("\nğŸ”¹ SCENARIO A: Initial Request (Should Pause)")
ctx_pause = MockToolContext(confirmed=None)
res_pause = await book_ticket("FL-101", "John Doe", ctx_pause)
if res_pause["status"] == "pending_confirmation":
    print("âœ… PASS: Tool paused for verification.")

# Scenario B: Resume/Confirm
print("\nğŸ”¹ SCENARIO B: User Confirmed (Should Book)")
ctx_resume = MockToolContext(confirmed=True)
res_resume = await book_ticket("FL-101", "John Doe", ctx_resume)
if res_resume["status"] == "BOOKED":
    print("âœ… PASS: Tool executed booking.")
    print(f"   Ref: {res_resume['reference']}")


# Configure Retry
retry_config = types.HttpRetryOptions(
    attempts=3, exp_base=2, initial_delay=1, http_status_codes=[429, 500, 503]
)

# Define the Agent
travel_agent = LlmAgent(
    name="travel_agent",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    instruction="""You are the "TravelConnect Concierge".
    **YOUR CORE PROTOCOL:**
    1. âœˆï¸� **DISCOVERY:** If user asks for flights, call `get_flight_options`.
    2. ğŸ�¨ **HOTELS:** If user asks for hotels, call `get_hotel_options`.
    3. ğŸ’³ **BOOKING:** If user explicitly asks to BOOK or RESERVE a flight:
       - You MUST extract the `flight_id` and `passenger_name`.
       - Call `book_ticket`. This will trigger a confirmation pause.
    4. â›” **OUT OF SCOPE:** Do not answer general questions about history or math.
    """,
    tools=[
        FunctionTool(func=get_flight_options),
        FunctionTool(func=get_hotel_options),
        FunctionTool(func=book_ticket)
    ]
)

# Wrap in Resumable App
travel_app = App(
    name="travel_app",
    root_agent=travel_agent,
    resumability_config=ResumabilityConfig(is_resumable=True)
)

# Create Runner with Memory
session_service = InMemorySessionService()
travel_runner = Runner(app=travel_app, session_service=session_service)

print("âœ… Travel Agent & Runner Ready!")


def print_agent_response(events):
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"ğŸ¤– Agent > {part.text}")
                if part.function_response:
                    print(f"   [Tool Output]: {part.function_response.response}")

async def run_travel_workflow(user_query: str, auto_confirm: bool = True):
    print(f"\n{'='*60}\nğŸ—£ï¸� User > {user_query}\n")
    
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    await session_service.create_session(app_name="travel_app", user_id="traveler", session_id=session_id)
    
    query_content = types.Content(role="user", parts=[types.Part(text=user_query)])
    events = []

    # --- PASS 1: Run Agent ---
    async for event in travel_runner.run_async(user_id="traveler", session_id=session_id, new_message=query_content):
        events.append(event)

    # --- CHECK FOR PAUSE ---
    approval_req = None
    invocation_id = None
    
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.function_call and part.function_call.name == "adk_request_confirmation":
                    approval_req = part.function_call
                    invocation_id = event.invocation_id

    # --- PASS 2: Handle Approval ---
    if approval_req:
        hint = approval_req.args.get('hint', 'Confirm?')
        print(f"â�¸ï¸� SYSTEM PAUSED: {hint}")
        print(f"ğŸ¤” Human Decision: {'âœ… CONFIRM' if auto_confirm else 'â�Œ REJECT'}\n")
        
        conf_response = types.FunctionResponse(
            id=approval_req.id,
            name="adk_request_confirmation",
            response={"confirmed": auto_confirm}
        )
        resume_msg = types.Content(role="user", parts=[types.Part(function_response=conf_response)])
        
        async for event in travel_runner.run_async(user_id="traveler", session_id=session_id, new_message=resume_msg, invocation_id=invocation_id):
            events.append(event)

    print_agent_response(events)
    print(f"{'='*60}\n")


print("--- ğŸ¤– STARTING INTEGRATION TESTS ---")

# Test 1: Info Lookups (Stateless)
await run_travel_workflow("Find flights from Tehran to Shiraz")
await run_travel_workflow("Show me luxury hotels in Kish")

# Test 2: Booking (Stateful - Confirmed)
await run_travel_workflow("Book flight FL-101 for Sarah Connor", auto_confirm=True)

# Test 3: Booking (Stateful - Rejected)
await run_travel_workflow("Book flight FL-202 for John Smith", auto_confirm=False)


print("--- ğŸ“Š STARTING GOLDEN DATASET EVALUATION ---")

# 1. Define the Golden Dataset (COMPLETE inputs)
# We add explicit 'Origin' and 'Passenger Name' so the agent calls the tool immediately.
golden_dataset = [
    {"input": "Find flights from Tehran to Isfahan", "expected_tool": "get_flight_options"}, # Added Origin
    {"input": "I need a hotel in Tabriz.", "expected_tool": "get_hotel_options"},
    {"input": "Book flight FL-101 for Sarah Connor.", "expected_tool": "book_ticket"}, # Added Name
    {"input": "What is the capital of France?", "expected_tool": "None"}, # Out of scope
]

# 2. The Evaluation Loop
score = 0
total = len(golden_dataset)

for test_case in golden_dataset:
    user_query = test_case["input"]
    expected_tool = test_case["expected_tool"]
    
    print(f"\nğŸ“� Testing: '{user_query}'")
    
    # Create a fresh session for isolation
    session_id = f"eval_{uuid.uuid4().hex[:8]}"
    await session_service.create_session(app_name="travel_app", user_id="eval_user", session_id=session_id)
    
    query_content = types.Content(role="user", parts=[types.Part(text=user_query)])
    
    # Run the agent
    tool_detected = "None"
    async for event in travel_runner.run_async(user_id="eval_user", session_id=session_id, new_message=query_content):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.function_call:
                    tool_detected = part.function_call.name
                    break
        if tool_detected != "None": break

    # Score the result
    if tool_detected == expected_tool:
        print(f"   âœ… PASS: Routed to '{tool_detected}'")
        score += 1
    else:
        print(f"   â�Œ FAIL: Expected '{expected_tool}', got '{tool_detected}'")

print(f"\n{'-'*40}")
print(f"ğŸ�† FINAL SCORE: {score}/{total} ({(score/total)*100:.0f}%)")
print(f"{'-'*40}")

