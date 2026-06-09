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


import os
import uuid
import asyncio
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.tools import FunctionTool, ToolContext
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import preload_memory
from google.adk.apps.app import App, ResumabilityConfig
from google.genai import types
from kaggle_secrets import UserSecretsClient


try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Authentication Complete")
except:
    print("âš ï¸� Please set your GOOGLE_API_KEY in Kaggle Secrets")
    


def get_weather_forecast(city: str) -> dict:
    """Checks the weather forecast for a specific city."""
    #Mock data for demonstration
    weather_db = {
        "paris": "Sunny, 20Â°C",
        "london": "Rainy, 15Â°C",
        "tokyo": "Cloudy, 22Â°C",
        "new york": "Sunny, 25Â°C"
    }
    weather = weather_db.get(city.lower(), "Sunny, 20Â°C (Default)")
    return {"status": "success", "forecast": weather}

#This tool handles the "booking" and requires approval for expensive items
def book_itinerary_item(activity_name: str, price: int, tool_context: ToolContext) -> dict:
    """Books an activity. Requires approval if price > $200."""
    
    #Human in the loop logic 
    if price > 200:
        if not tool_context.tool_confirmation:
            tool_context.request_confirmation(
                hint=f"âš ï¸� This activity ({activity_name}) costs ${price}. Approve booking?",
                payload={"activity": activity_name, "price": price}
            )
            return {"status": "pending", "message": "Waiting for user approval for expensive item."}
        
    
        if tool_context.tool_confirmation.confirmed:
            return {"status": "success", "message": f"âœ… Booked {activity_name} for ${price}."}
        else:
            return {"status": "rejected", "message": f"â�Œ Booking for {activity_name} cancelled by user."}
            
    #Cheap items book immediately
    return {"status": "success", "message": f"âœ… Booked {activity_name} for ${price}."}

print("âœ… Tools Created: Weather & Booking")


# 2.1. Initialize Services
# Initialize Memory and Session Services
memory_service = InMemoryMemoryService()
session_service = InMemorySessionService()

# 2.2. Define a Seeding Function
# This creates a temporary agent just to "hear" your preferences and save them.
async def seed_memory_preferences():
    # Create a basic agent for the seeding phase
    seeding_agent = LlmAgent(
        name="SeedingAgent",
        model=Gemini(model="gemini-2.5-flash-lite")
    )

    app_name = "TripPlanner"
    user_id = "user1"
    seed_session_id = "seed_session_01"

    # Create a runner connecting the services [cite: 735]
    seeding_runner = Runner(
        agent=seeding_agent,
        app_name=app_name,
        session_service=session_service,
        memory_service=memory_service
    )

    # The fact we want to plant in memory
    fact = "I prefer outdoor activities and I am allergic to peanuts."
    print(f"ğŸŒ± Seeding Memory with: '{fact}'")

    #Run a single turn conversation
    #We use a specific session ID for this seeding event
    #Create the session explicitly first
    try:
        await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=seed_session_id
        )
    except Exception:
        # If it already exists (e.g. from a previous run), just get it
        await session_service.get_session(
            app_name=app_name,
            user_id=user_id,
            session_id=seed_session_id
        )
    #Run a single turn conversation to put the fact into the session history
    user_msg = types.Content(role="user", parts=[types.Part(text=fact)])
    
    async for event in seeding_runner.run_async(
        user_id=user_id, 
        session_id=seed_session_id, 
        new_message=user_msg
    ):
        pass 
 
    # This commits the "fact" to the agent's long-term storage
    session = await session_service.get_session(
        app_name=app_name, 
        user_id=user_id, 
        session_id=seed_session_id
    )
    await memory_service.add_session_to_memory(session)
    print("âœ… Memory Seeded Successfully!")

# 2.3. Run the seeder
await seed_memory_preferences()


# 3.1. Create the Agent
travel_agent = LlmAgent(
    name="TripPlannerAgent",
    model=Gemini(model="gemini-2.5-flash-lite"),
    instruction="""You are an expert travel concierge.
    1. ALWAYS check your memory (using `preload_memory`) first to see user preferences, allergies, and interests.
    2. When a user asks for a plan, check the weather using `get_weather_forecast`.
    3. Suggest an itinerary based on the weather and their preferences (e.g., if they like outdoors, suggest parks).
    4. If the user agrees to an activity, use `book_itinerary_item` to book it.
    5. If a tool returns 'pending', tell the user you need their approval.
    """,
    
    tools=[get_weather_forecast, book_itinerary_item, preload_memory]
)

# 3.2. Wrap in App for Resumability
travel_app = App(
    name="TripPlanner",
    root_agent=travel_agent,
    resumability_config=ResumabilityConfig(is_resumable=True)
)

# 3.3. Create the Runner
runner = Runner(
    app=travel_app,
    session_service=session_service,
    memory_service=memory_service
)

# 3.4. Define the Chat Workflow Logic
def check_for_approval(events):
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.function_call and part.function_call.name == "adk_request_confirmation":
                    return {
                        "approval_id": part.function_call.id, 
                        "invocation_id": event.invocation_id
                    }
    return None

def create_approval_response(approval_info, approved: bool):
    confirmation_response = types.FunctionResponse(
        id=approval_info["approval_id"],
        name="adk_request_confirmation",
        response={"confirmed": approved},
    )
    return types.Content(role="user", parts=[types.Part(function_response=confirmation_response)])

async def chat_with_planner(user_input, session_id="trip_session_1"):
    print(f"\nğŸ‘¤ User: {user_input}")
    
    # Ensure session exists
    try:
        await session_service.create_session(app_name="TripPlanner", user_id="user1", session_id=session_id)
    except:
        pass 

    user_msg = types.Content(role="user", parts=[types.Part(text=user_input)])
    events = []
    
    # Initial Run
    print("ğŸ¤– Agent is thinking...")
    async for event in runner.run_async(user_id="user1", session_id=session_id, new_message=user_msg):
        events.append(event)
        # Filter out function calls and only print text responses
        if event.content and event.content.parts:
             for part in event.content.parts:
                if part.text: 
                    print(f"ğŸ¤– Agent: {part.text}")

    # Check for Pause
    approval_info = check_for_approval(events)
    
    if approval_info:
        print("\nâš ï¸�  SYSTEM PAUSE: Agent is requesting approval for a high-value booking.")
        print("    (Simulating User clicking 'Approve' button...)")
        
        decision = True 
        print(f"ğŸ‘‰ User Action: {'Approved âœ…' if decision else 'Rejected â�Œ'}")
        
        print("ğŸ¤– Resuming Agent...")
        async for event in runner.run_async(
            user_id="user1", 
            session_id=session_id, 
            new_message=create_approval_response(approval_info, decision),
            invocation_id=approval_info["invocation_id"]
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text: print(f"ğŸ¤– Agent (Resumed): {part.text}")

print("âœ… Agent, Runner, and Workflow fully re-configured.")


# Helper: Check if the agent paused for approval
def check_for_approval(events):
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                # We look for the specific function call that requests confirmation
                if part.function_call and part.function_call.name == "adk_request_confirmation":
                    return {
                        "approval_id": part.function_call.id, 
                        "invocation_id": event.invocation_id
                    }
    return None

# Helper: Create the response packet to send back to the agent
def create_approval_response(approval_info, approved: bool):
    confirmation_response = types.FunctionResponse(
        id=approval_info["approval_id"],
        name="adk_request_confirmation",
        response={"confirmed": approved}, # This matches the structure expected by the ToolContext
    )
    return types.Content(role="user", parts=[types.Part(function_response=confirmation_response)])

# Main Chat Loop
# Improved Chat Loop with Debugging Prints
async def chat_with_planner(user_input, session_id="trip_session_1"):
    print(f"\nğŸ‘¤ User: {user_input}")
    print("-" * 40)
    
    # 1. Ensure Session Exists
    try:
        await session_service.create_session(app_name="TripPlanner", user_id="user1", session_id=session_id)
    except:
        pass 

    user_msg = types.Content(role="user", parts=[types.Part(text=user_input)])
    events = []
    
    # 2. Run the Agent (Initial Turn)
    # We use a loop that stays alive to catch the text AFTER the tool runs
    print("ğŸ¤– Agent is thinking...", end="", flush=True)
    
    async for event in runner.run_async(user_id="user1", session_id=session_id, new_message=user_msg):
        events.append(event)
        
        # Check for Tool Calls (Debugging)
        if event.content and event.content.parts:
            for part in event.content.parts:
                # If it's a tool call, print it so we know it's working
                if part.function_call:
                    print(f"\n   âš™ï¸�  [Tool Call] Using: {part.function_call.name}...")
                
                # If it's the final text, print it!
                if part.text: 
                    print(f"\n\nğŸ¤– Agent: {part.text}")

    # 3. Check for Pause (Approval Logic for Scenario B)
    approval_info = check_for_approval(events)
    
    if approval_info:
        print("\n\nâš ï¸�  SYSTEM PAUSE: Agent requested approval for high-value item.")
        print("    (Auto-clicking 'Approve' for demo...)")
        decision = True 
        print(f"ğŸ‘‰ User Action: {'Approved âœ…' if decision else 'Rejected â�Œ'}")
        
        print("ğŸ¤– Resuming Agent to finalize booking...")
        # Resume execution with the approval decision
        async for event in runner.run_async(
            user_id="user1", 
            session_id=session_id, 
            new_message=create_approval_response(approval_info, decision),
            invocation_id=approval_info["invocation_id"]
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text: print(f"\nğŸ¤– Agent (Resumed): {part.text}")

print("âœ… Debugging Chat Loop Ready")


await chat_with_planner("I want to go to Paris tomorrow. Suggest a plan.", session_id="trip_session_1")


await chat_with_planner("That sounds great. Please book the VIP Private Tour for $350.", session_id="trip_session_1")

