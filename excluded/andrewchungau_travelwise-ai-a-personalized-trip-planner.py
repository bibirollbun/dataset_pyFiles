# --- INSTALL DEPENDENCIES ---
# This must run first to ensure libraries exist for imports below
!pip install mcp amadeus python-dotenv


# --- 1. STANDARD LIBRARY IMPORTS ---
import os
import sys
import json
import time
import random
import base64
import asyncio
import inspect
import logging
import subprocess
import contextvars
from datetime import date, datetime, timedelta
from typing import Optional, Dict, Any, List

# --- 2. KAGGLE & ENVIRONMENT ---
from kaggle_secrets import UserSecretsClient
import requests

# --- 3. GOOGLE GEN AI & ADK CORE ---
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext, AgentTool

# --- 4. MCP & TOOLING ---
import mcp.shared.session
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.types import CallToolRequestParams
from amadeus import Client, ResponseError
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, StdioServerParameters 
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams, SseConnectionParams

# --- 5. UI & DISPLAY ---
from IPython.display import display, HTML, Markdown

print("âœ… All Dependencies Imported & Ready.")


# --- âš™ï¸� GLOBAL CONFIGURATION ---

# LLM Retry Logic
# Protects against 429 (Rate Limit) and 503 (Service Busy) errors
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

print("âœ… Global Configuration Loaded (Gemini Retries active).")


print("ğŸ”‘ LOADING SECRETS...")

try:
    # Load Google key (used by Gemini)
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    # Map to the name the Maps server expects
    os.environ["GOOGLE_MAPS_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("   - Google API Key: âœ… Loaded and mapped to GOOGLE_MAPS_API_KEY")

    # Amadeus
    AMADEUS_CLIENT_ID = UserSecretsClient().get_secret("AMADEUS_CLIENT_ID")
    os.environ["AMADEUS_CLIENT_ID"] = AMADEUS_CLIENT_ID
    AMADEUS_CLIENT_SECRET = UserSecretsClient().get_secret("AMADEUS_CLIENT_SECRET")
    os.environ["AMADEUS_CLIENT_SECRET"] = AMADEUS_CLIENT_SECRET
    print("   - Amadeus Keys: âœ… Loaded")

    # AWS checks
    AWS_HOST_IP = UserSecretsClient().get_secret("AWS_HOST_IP")
    AWS_PEM_KEY = UserSecretsClient().get_secret("AWS_PEM_KEY")
    print("   - AWS Credentials: âœ… Found")

    print("\nâœ… ALL SYSTEMS GO: Credentials are ready.")

except Exception as e:
    print(f"\nâ�Œ SECRET ERROR: {e}")
    print("   Please check 'Add-ons -> Secrets' and ensure all keys (GOOGLE, AMADEUS, AWS) are set.")

# Reconfigure Maps tool using the mapped env var
print("ğŸ”„ RE-CONFIGURING MAPS TOOL WITH CORRECT KEY NAME...")

mcp_maps_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-google-maps"],
            env={
                # Use the mapped variable so the server sees GOOGLE_MAPS_API_KEY
                "GOOGLE_MAPS_API_KEY": os.environ["GOOGLE_MAPS_API_KEY"]
            },
            tool_filter=["searchPlaces", "getDirections"],
        )
    )
)


# --- DIAGNOSTIC: TEST AMADEUS FLIGHT SEARCH API ---
# Run this cell to verify your API credentials and connectivity 
# independent of the Agent logic.
print("âœˆï¸� STARTING AMADEUS FLIGHT SEARCH TEST...")

# 1. Load Secrets (Robust handling for Kaggle vs Local)
try:
    # Try loading from Kaggle Secrets first
    AMADEUS_CLIENT_ID = UserSecretsClient().get_secret("AMADEUS_CLIENT_ID")
    AMADEUS_CLIENT_SECRET = UserSecretsClient().get_secret("AMADEUS_CLIENT_SECRET")
    print("âœ… Secrets loaded from Kaggle.")
except Exception:
    # Fallback to environment variables
    print("âš ï¸� Kaggle Secrets unavailable. Checking environment variables...")
    AMADEUS_CLIENT_ID = os.getenv("AMADEUS_CLIENT_ID")
    AMADEUS_CLIENT_SECRET = os.getenv("AMADEUS_CLIENT_SECRET")

# 2. Validate Keys
if not AMADEUS_CLIENT_ID or not AMADEUS_CLIENT_SECRET:
    print("â�Œ ERROR: AMADEUS_CLIENT_ID or AMADEUS_CLIENT_SECRET is missing.")
    print("   Please check your 'Add-ons -> Secrets' or environment variables.")
    # Stop execution here if keys are missing to avoid crashing later
else:
    print(f"âœ… Keys found. Client ID prefix: {AMADEUS_CLIENT_ID[:4]}...")

    # 3. Initialize Client
    try:
        amadeus = Client(
            client_id=AMADEUS_CLIENT_ID,
            client_secret=AMADEUS_CLIENT_SECRET
        )
        print("âœ… Amadeus Client initialized.")

        # 4. Run Search Test
        # Calculate a dynamic future date (e.g., 2 months from today)
        # to ensure we always search for valid flights.
        future_date = (date.today() + timedelta(days=30)).isoformat()
        
        print(f"\nğŸ”� Searching for flights: LON -> NYC on {future_date}...")
        
        response = amadeus.shopping.flight_offers_search.get(
            originLocationCode='LON',
            destinationLocationCode='NYC',
            departureDate=future_date,
            adults=1,
            max=3
        )

        if response.data:
            print(f"ğŸ�‰ SUCCESS! Found {len(response.data)} flight offers.")
            
            # Display first result details
            first_offer = response.data[0]
            price = first_offer['price']['total']
            currency = first_offer['price']['currency']
            airline = first_offer['validatingAirlineCodes'][0]
            
            print(f"   ğŸ’° Sample Price: {price} {currency}")
            print(f"   âœˆï¸� Airline: {airline}")
            print("   (Raw data validated)")
        else:
            print("âš ï¸� Request succeeded but returned no data (Empty List).")

    except ResponseError as error:
        print(f"\nâ�Œ API ERROR: {error}")
        if error.response:
            print(f"   Status Code: {error.response.status_code}")
            print(f"   Details: {error.response.body}")
            
    except Exception as e:
        print(f"\nâ�Œ UNEXPECTED ERROR: {e}")


# --- DIAGNOSTIC: TEST AMADEUS HOTEL SEARCH API ---
# Run this cell to verify your API credentials and connectivity 
# independent of the Agent logic.
print("ğŸ�¨ STARTING AMADEUS HOTEL SEARCH TEST (V3)...")

# 1. Load Secrets
try:
    AMADEUS_CLIENT_ID = UserSecretsClient().get_secret("AMADEUS_CLIENT_ID")
    AMADEUS_CLIENT_SECRET = UserSecretsClient().get_secret("AMADEUS_CLIENT_SECRET")
    print("âœ… Secrets loaded from Kaggle.")
except Exception:
    AMADEUS_CLIENT_ID = os.getenv("AMADEUS_CLIENT_ID")
    AMADEUS_CLIENT_SECRET = os.getenv("AMADEUS_CLIENT_SECRET")

if not AMADEUS_CLIENT_ID or not AMADEUS_CLIENT_SECRET:
    print("â�Œ ERROR: Missing API Keys.")
else:
    # 2. Initialize Client
    try:
        amadeus = Client(client_id=AMADEUS_CLIENT_ID, client_secret=AMADEUS_CLIENT_SECRET)
        print("âœ… Client initialized.")

        # 3. Run Test
        CITY_CODE = 'LON'
        # Dynamic dates: 1 month from now
        today = date.today()
        check_in = (today + timedelta(days=30)).isoformat()
        check_out = (today + timedelta(days=32)).isoformat()
        
        print(f"\nğŸ”� Step 1: Finding hotels in {CITY_CODE}...")
        hotels_response = amadeus.reference_data.locations.hotels.by_city.get(cityCode=CITY_CODE)

        if hotels_response.data:
            # Take first 3 hotels
            sample_hotels = hotels_response.data[:3]
            hotel_ids = [h['hotelId'] for h in sample_hotels]
            ids_string = ",".join(hotel_ids)
            print(f"âœ… Found {len(hotels_response.data)} hotels. Testing IDs: {ids_string}")
            
            print(f"ğŸ”� Step 2: Checking offers for {check_in} to {check_out}...")
            offers_response = amadeus.shopping.hotel_offers_search.get(
                hotelIds=ids_string,
                adults=1,
                checkInDate=check_in,
                checkOutDate=check_out
            )
            
            if offers_response.data:
                print(f"ğŸ�‰ SUCCESS! Found {len(offers_response.data)} offers.")
                first = offers_response.data[0]
                hotel_name = first.get('hotel', {}).get('name', 'Unknown')
                price = first['offers'][0]['price']['total']
                currency = first['offers'][0]['price']['currency']
                print(f"   ğŸ�¨ Hotel: {hotel_name}")
                print(f"   ğŸ’° Price: {price} {currency}")
            else:
                print("âš ï¸� Step 1 worked, but Step 2 returned no offers for these specific dates/hotels.")
        else:
            print(f"â�Œ No hotels found in {CITY_CODE}.")

    except ResponseError as error:
        print(f"\nâ�Œ API ERROR: {error}")
        if error.response:
            print(f"   Details: {error.response.body}")
    except Exception as e:
        print(f"\nâ�Œ UNEXPECTED ERROR: {e}")


# --- CONFIGURATION ---
REMOTE_PORT = 8000
LOCAL_PORT = 8500

print("ğŸ§¹ Cleaning up old processes...")
subprocess.run(["pkill", "-f", "ssh"]) # Kill old tunnels to free the port
time.sleep(1)

# 1. Setup Key File
try:
    print("ğŸ”‘ Decoding key...")
    b64_key = UserSecretsClient().get_secret("AWS_PEM_KEY")
    pem_content = base64.b64decode(b64_key).decode('utf-8')
    
    key_path = "/kaggle/working/aws_key.pem"
    with open(key_path, "w") as f:
        f.write(pem_content)
    os.chmod(key_path, 0o400)
except Exception as e:
    print(f"â�Œ Key Error: {e}")
    raise e

# 2. Define Config
try:
    AWS_IP = UserSecretsClient().get_secret("AWS_HOST_IP")
except Exception:
    # Fallback or Error
    AWS_IP = "54.174.128.29" # Check if this is empty!
    
if not AWS_IP: raise ValueError("AWS IP is missing")

# 3. Start SSH Tunnel with Error Capture
cmd = [
    "ssh", 
    "-o", "StrictHostKeyChecking=no",
    "-o", "ExitOnForwardFailure=yes",
    "-i", key_path,
    "-N", # No remote command
    "-L", f"{LOCAL_PORT}:127.0.0.1:{REMOTE_PORT}",
    f"ubuntu@{AWS_IP}"
]

print(f"ğŸš‡ Connecting to {AWS_IP}...")

# We use a pipe for stderr to capture the crash reason
tunnel_process = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)

time.sleep(3) # Wait for connection

# Check status
if tunnel_process.poll() is None:
    print(f"âœ… Secure Tunnel Active! Map: localhost:{LOCAL_PORT} -> AWS:Localhost:{REMOTE_PORT}")
else:
    print("\nâ�Œ TUNNEL DIED IMMEDIATELY. ERROR LOG:")
    print("=" * 40)
    # Read the error message from the dead process
    _, stderr = tunnel_process.communicate()
    print(stderr.strip())
    print("=" * 40)


async def test_remote_server_direct():
    url = "http://localhost:8500/sse"
    print(f"ğŸ”Œ TARGET: {url}")
    print("---------------------------------------------------")

    try:
        # --- STAGE 1: Network Connection ---
        print("STAGE 1: ğŸ“¡ Establishing SSE Connection...")
        async with sse_client(url) as streams:
            print("   âœ… STAGE 1 COMPLETE: Connected to SSE stream.")
            read, write = streams

            # --- STAGE 2: Session Creation ---
            print("STAGE 2: ğŸ¤� Creating Client Session...")
            async with ClientSession(read, write) as session:
                
                # --- STAGE 3: Protocol Handshake ---
                print("STAGE 3: ğŸ—£ï¸� Performing Handshake (Initialize)...")
                try:
                    await session.initialize()
                    print("   âœ… STAGE 3 COMPLETE: Handshake successful.")
                except Exception as e:
                    print(f"   â�Œ STAGE 3 FAILED: Handshake error. {e}")
                    return

                # --- STAGE 4: Tool Discovery ---
                print("STAGE 4: ğŸ“‹ Listing Available Tools...")
                try:
                    tools_list = await session.list_tools()
                    tool_names = [t.name for t in tools_list.tools]
                    print(f"   ğŸ”� Found tools: {tool_names}")
                    
                    if "get_flight_offers" in tool_names:
                        print("   âœ… STAGE 4 COMPLETE: 'get_flight_offers' found.")
                    else:
                        print("   â�Œ STAGE 4 FAILED: 'get_flight_offers' NOT found in list.")
                        return
                except Exception as e:
                    print(f"   â�Œ STAGE 4 FAILED: Could not list tools. {e}")
                    return

                # --- STAGE 5: Tool Execution ---
                print("STAGE 5: ğŸš€ Invoking 'get_flight_offers'...")

                # --- DYNAMIC DATE CALCULATION ---
                today = datetime.now()
                future_date = today + timedelta(weeks=3)
                departure_date_str = future_date.strftime('%Y-%m-%d')
                
                print(f"   ğŸ“… Dynamic Date Calculated: {departure_date_str} (Today + 3 Weeks)")

                search_args = {
                    "originLocationCode": "SYD",
                    "destinationLocationCode": "BKK",
                    "departureDate": departure_date_str,
                    "adults": 1
                }
                print(f"   ğŸ“¤ Sending Params: {search_args}")

                try:
                    # We explicitly use call_tool
                    result = await session.call_tool(
                        name="get_flight_offers",
                        arguments=search_args
                    )
                    print("   âœ… STAGE 5 COMPLETE: Function executed without crashing.")
                except TimeoutError:
                    print("   â�Œ STAGE 5 FAILED: TIMEOUT. The Amadeus API took too long to reply.")
                    return
                except Exception as e:
                    print(f"   â�Œ STAGE 5 FAILED: Execution error. {e}")
                    return

                # --- STAGE 6: Result Parsing ---
                print("STAGE 6: ğŸ“¦ Analyzing Response...")
                if result.content and len(result.content) > 0:
                    text_content = result.content[0].text
                    if "error" in text_content.lower() and "info" not in text_content.lower():
                        print("   âš ï¸�  WARNING: The tool ran, but returned an API error:")
                        print(f"   {text_content}")
                    else:
                        print(f"   âœ… SUCCESS! Received {len(text_content)} characters of data.")
                        print(f"   ğŸ”� Preview: {text_content[:200]}...")
                else:
                    print("   â�Œ STAGE 6 FAILED: Response was empty.")

    except Exception as e:
        print(f"\nâ�Œ FATAL CONNECTION ERROR: {e}")
        print("ğŸ‘‰ TIP: Check if your SSH tunnel is active: 'ssh -L 8500:localhost:8500 ...'")

# Run in Jupyter
await test_remote_server_direct()


# --- ğŸ› ï¸� HELPER FUNCTIONS ---
async def run_session(
    runner_instance,
    user_queries: list[str] | str = None,
    session_id: str = "default",
    user_id: str = "user_01"
):
    """
    Helper to run a session. 
    Updated to handle the 'TripStateService' context switching automatically.
    """
    print(f"\n ### ğŸŸ¢ Starting Session: {session_id}")

    # 1. Dynamic Setup
    app_name = getattr(runner_instance, "app_name", "default") or "default"
    if app_name == "InMemoryRunner": app_name = "InMemoryRunner"
    
    # 2. Session Management (ADK Memory)
    try:
        session = await runner_instance.session_service.get_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )
        print(f"   (Resuming ADK session)")
    except Exception:
        print(f"   (Creating new ADK session)")
        session = await runner_instance.session_service.create_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )

    # 3. Process Queries
    if user_queries:
        if isinstance(user_queries, str):
            user_queries = [user_queries]

        for query_text in user_queries:
            print(f"\nğŸ‘¤ User > {query_text}")
            
            # --- CRITICAL: SET THE CONTEXT VARIABLE ---
            # This tells TripStateService which session we are in.
            token = current_session_id.set(session_id)
            
            try:
                # Prepare Message
                query = types.Content(role="user", parts=[types.Part(text=query_text)])

                # Run Agent
                async for event in runner_instance.run_async(
                    user_id=user_id, session_id=session.id, new_message=query
                ):
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            # Detect Tool Calls
                            if part.function_call:
                                print(f"   ğŸ› ï¸� Tool Call: {part.function_call.name}")
                                # Optional: Print args for debugging
                                # print(f"      Args: {part.function_call.args}")
                            
                            # Detect Agent Speech
                            if part.text:
                                print(f"âœˆï¸� Agent > {part.text}")

                    # Detect Tool Results
                    if getattr(event, 'tool_output', None):
                         output_str = str(event.tool_output.output)[:100]
                         print(f"   ğŸ”Œ Tool Output: {output_str}...")
            
            except Exception as e:
                print(f"\nâ�Œ RUNTIME ERROR: {e}")
            
            finally:
                # --- CRITICAL: RESET CONTEXT ---
                current_session_id.reset(token)

    else:
        print("No queries provided!")

    # 4. Final State Check (Reading from Service)
    print("\n--- ğŸ“Š Final Session State (From TripStateService) ---")
    
    # We must set context again briefly to read the state
    token = current_session_id.set(session_id)
    try:
        state = TripStateService.get_state()
        if state.get("destination"):
            print(f"ğŸ“� Destination: {state.get('destination')}")
            print(f"ğŸ’° Current Spend: ${state.get('current_spend', 0)}")
            print(f"ğŸ“� Itinerary Items: {len(state.get('itinerary', []))}")
        else:
            print("âš ï¸� No trip data saved yet.")
    finally:
        current_session_id.reset(token)

print("âœ… Helper Functions Defined")


print("ğŸ”„ CONFIGURING HYBRID TOOLSET...")

# ğŸŸ¢ TIMEOUT FIX: Force the library to wait 60 seconds
mcp.shared.session.DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0
print("   â””â”€ Applied Global Timeout Patch: 60s")

# --- 1. AMADEUS TOOL (REMOTE via AWS) ---
# We keep this one live because it's running on your stable AWS tunnel
print("ğŸ”Œ Connecting Amadeus Tool to AWS via Tunnel...")
mcp_amadeus_custom_server = McpToolset(
    connection_params=SseConnectionParams(
        url="http://localhost:8500/sse" 
    ),
    tool_filter=["get_flight_offers", "get_hotel_offers"]
)

# --- 2. MAPS TOOL (MOCK MODE) ---
# âš ï¸� NOTE: Switched to Mock Mode to bypass Google Cloud "Places API" billing requirements.
print("ğŸ—ºï¸�  Configuring Local Google Maps Tool (MOCK MODE)...")

def searchPlaces(query: str, radius: int = 5000):
    """
    Search for places, attractions, or restaurants using a mock database.
    Args:
        query: The search text (e.g., 'Pizza in London')
        radius: Search radius in meters (ignored in mock)
    """
    print(f"   ğŸ�­ [MOCK MAPS] Searching for: '{query}'")
    q = query.lower()
    
    # ğŸ�• Mock Data: Pizza in London
    if "pizza" in q and "london" in q:
        return json.dumps([
            {"name": "Pizza Pilgrims", "rating": 4.7, "address": "11 Dean St, Soho, London", "status": "Open"},
            {"name": "Homeslice Neal's Yard", "rating": 4.6, "address": "13 Neal's Yard, London", "status": "Busy"},
            {"name": "Franco Manca", "rating": 4.5, "address": "Broadgate Circle, London", "status": "Open"}
        ])
    
    # ğŸ—½ Mock Data: NYC Sights
    elif "new york" in q or "nyc" in q:
        return json.dumps([
            {"name": "Central Park", "rating": 4.8, "type": "Park", "address": "New York, NY"},
            {"name": "Joe's Pizza", "rating": 4.6, "type": "Restaurant", "address": "Carmine St, NY"},
            {"name": "Empire State Building", "rating": 4.7, "type": "Attraction", "address": "20 W 34th St, NY"}
        ])

    # ğŸ�¨ Mock Data: Hotels (Generic)
    elif "hotel" in q:
        return json.dumps([
            {"name": "Grand Plaza Hotel", "rating": 4.3, "price_level": "$$$"},
            {"name": "Budget Stay Inn", "rating": 3.9, "price_level": "$"}
        ])
    
    # ğŸ¤· Default Fallback
    else:
        return json.dumps([
            {"name": f"Mock Place for '{query}'", "rating": 4.0, "address": "123 Mock Lane"}
        ])

# ğŸ”— BINDING: We assign the function to the variable name the Agent expects.
# The Agent will see this as a standard Python tool.
mcp_maps_server = searchPlaces

# Original Live Code (Commented Out for Safety)
# mcp_maps_server = McpToolset(
#     connection_params=StdioConnectionParams(
#         server_params=StdioServerParameters(
#             command="npx",
#             args=["-y", "@modelcontextprotocol/server-google-maps"],
#             env={"GOOGLE_MAPS_API_KEY": os.getenv("GOOGLE_API_KEY")}, # Requires 'Places API New' + Billing
#             tool_filter=["searchPlaces", "getDirections"],
#         )
#     )
# )

print("âœ… Hybrid Toolset Configured.")


# --- 1. SERVICE ARCHITECTURE ---

# A thread-safe context variable to track "Who is the current user?"
# This replaces the broken 'ctx' parameter mechanism entirely.
current_session_id = contextvars.ContextVar("session_id", default="default")

class TripStateService:
    """
    A robust service to manage state for multiple users simultaneously.
    It acts as the 'Database' for the Agent.
    """
    _store: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def get_state(cls) -> Dict[str, Any]:
        """Retrieves the state for the ACTIVE session."""
        session_id = current_session_id.get()
        if session_id not in cls._store:
            # Initialize empty state for new session
            cls._store[session_id] = {
                "current_spend": 0, 
                "itinerary": [], 
                "budget_limit": 0
            }
        return cls._store[session_id]

# --- 2. THE TOOLS (CLEAN & ROBUST) ---
# Notice: No 'ctx' parameter! The tools ask the Service for state.
# This guarantees the Agent can parse the function signature without crashing.

def validate_profile_completeness() -> str:
    """Checks if the session has all required fields."""
    state = TripStateService.get_state()
    missing = []
    
    if not state.get("origin"): missing.append("Origin")
    if not state.get("destination"): missing.append("Destination")
    if not state.get("start_date"): missing.append("Start Date")
    if not state.get("budget_limit"): missing.append("Budget")
    
    if missing:
        return f"â�Œ Missing: {', '.join(missing)}. Please ask the user."
    return "âœ… PROFILE COMPLETE."

def update_user_profile(
    origin: str,
    destination: str,
    start_date: str,
    budget: int,
    adults: int = 1,
    end_date: Optional[str] = None
) -> str:
    """Saves the trip details to the session state."""
    state = TripStateService.get_state() # <--- Access State Securely
    
    # Update State
    state["origin"] = origin
    state["destination"] = destination
    state["start_date"] = start_date
    state["end_date"] = end_date
    state["adults"] = adults
    state["budget_limit"] = budget
    
    # Reset financial tracking for a fresh trip
    state["current_spend"] = 0
    state["itinerary"] = []
    
    return f"âœ… Profile Saved: {origin} -> {destination} | Budget: ${budget}"

def add_to_itinerary(
    item_description: str,
    cost: int,
    category: str = "General" 
) -> str:
    """Adds a confirmed item to the itinerary and updates the budget."""
    state = TripStateService.get_state()
    
    state["itinerary"].append({
        "item": item_description,
        "cost": cost,
        "category": category
    })
    
    state["current_spend"] = state.get("current_spend", 0) + cost
    
    return check_budget_status()

def check_budget_status() -> str:
    """Read-only check of financial status."""
    state = TripStateService.get_state()
    
    limit = state.get("budget_limit", 0)
    spend = state.get("current_spend", 0)
    remaining = limit - spend
    
    status = "ğŸŸ¢" if remaining >= 0 else "ğŸ”´"
    return f"{status} Budget: ${limit} | Spent: ${spend} | Remaining: ${remaining}"

def get_trip_context() -> str:
    """Helper for the Agent to read the plan."""
    state = TripStateService.get_state()
    count = len(state.get("itinerary", []))
    return f"Current Plan: {count} items. " + check_budget_status()

print("âœ… Architecture Upgraded: Service Pattern Implemented.")


# --- 1. DEFINE SPECIALIST AGENTS ---

# AGENT A: DATA COLLECTOR
profile_guardian = LlmAgent(
    name="ProfileGuardian",
    model=Gemini(model="gemini-2.5-flash-lite", retry_config=retry_config),
    instruction="""You are the Profile Guardian.
    GOAL: Ensure the user provides Origin, Destination, Start Date, and Budget.
    
    1. If info is missing, ask for it.
    2. If info is present, call `update_user_profile`.
    """,
    tools=[update_user_profile]
)

# AGENT B: THE SHOPPER (Read-Only)
search_planner = LlmAgent(
    name="SearchPlanner",
    model=Gemini(model="gemini-2.5-flash-lite", retry_config=retry_config),
    instruction="""You are the Search Planner.
    GOAL: Find flight, hotel, and activity options.
    
    1. Always check `check_budget_status` first.
    2. Use `get_flight_offers` or `searchPlaces` to find options.
    3. **CRITICAL:** When you receive a JSON output from a tool, you **MUST** parse it internally to extract key facts (e.g., name, price, address) before generating your final response.
    4. Present options to the user clearly.
    
    CRITICAL: You CANNOT book things. You only find them.
    """,
    tools=[
        check_budget_status,
        get_trip_context,
        mcp_amadeus_custom_server,
        mcp_maps_server
    ]
)

# AGENT C: THE BUYER (Write-Only)
booking_agent = LlmAgent(
    name="BookingAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_config=retry_config),
    instruction="""You are the Booking Agent.
    GOAL: Commit items to the itinerary.
    
    1. **VALIDATION:** Before calling `add_to_itinerary`, you MUST verify you have both the **Item Name** (string) and the **Cost** (integer) from the user's request or the search results.
    2. Use `add_to_itinerary` to save the item.
    3. Report the remaining budget.
    4. If the cost is missing or invalid, do NOT call the tool and ask the user for clarification.
    """,
    tools=[
        add_to_itinerary,
        check_budget_status
    ]
)

# --- 2. DEFINE ROOT MANAGER ---

manager_agent = LlmAgent(
    name="TravelManager",
    model=Gemini(model="gemini-2.5-flash-lite", retry_config=retry_config),
    instruction="""You are the TravelWise Manager. You orchestrate the team.
    
    # ROUTING LOGIC
    1. **PROFILE:** If the user is giving dates/locations/budget -> Delegate to `ProfileGuardian`.
    2. **SEARCH:** If the user asks to find/search/look for something -> Delegate to `SearchPlanner`.
    3. **BOOK:** If the user says "Book this", "Add this", or confirms a choice -> Delegate to `BookingAgent`.
    
    # CRITICAL: REPORTING
    - After a sub-agent finishes, you MUST summarize their result to the user.
    - Example: "The BookingAgent has successfully confirmed your flight. Budget remaining: $4350."
    - NEVER leave the user with an empty response.
    """,
    # The Manager controls 3 distinct specialists
    tools=[
        AgentTool(profile_guardian), 
        AgentTool(search_planner),
        AgentTool(booking_agent)
    ]
)

# --- 3. RUNNER SETUP ---
runner = InMemoryRunner(agent=manager_agent, app_name="TravelWise")

print("âœ… Professional Architecture Initialized: Manager -> [Guardian, Searcher, Booker]")


# The Simulation Setup
# --- OPTIMIZED USER SIMULATOR ---
# Designed to provide all data in one shot to reduce turn count.
user_sim_agent = LlmAgent(
    name="UserSimulator",
    model=Gemini(model="gemini-2.5-flash-lite"),
    instruction="""You are an efficient traveler named Alex.
    
    # YOUR GOAL
    Plan a trip: London (LON) -> New York (JFK), 2025-12-18 to 2025-01-11, $5000 budget, 1 Adult.
    
    # INTERACTION STRATEGY (SAVE API CALLS)
    1. **Turn 1 (The Setup):** Provide ALL details (Origin, Dest, Dates, Budget, Adults) in the very first message. 
       - Command: "Save this profile and immediately search for flight options."
    
    2. **Turn 2 (The Booking):** If the agent shows flights, pick the first one. 
       - **CRITICAL:** You must explicitly state the airline and price to satisfy the booking tool. 
       - Example: "Book the British Airways flight for $650."
    
    3. **Turn 3 (The Food):** "Find Joe's Pizza in New York and book a $50 dinner."
    
    4. **Turn 4:** Say "TERMINATE".
    
    Be concise. Do not use filler words.
    """
)

user_runner = InMemoryRunner(agent=user_sim_agent, app_name="UserSim")
print("âœ… Efficient User Simulator Initialized.")


# The Simulation Loop

# --- CONFIG ---
SYSTEM_SESSION = f"sys_sim_{int(time.time())}"
USER_SESSION = f"usr_sim_{int(time.time())}"

# --- HELPERS ---
def log_turn_header(n):
    display(HTML(f"<div style='background-color:#e6f3ff; padding:10px; margin-top:20px; border-radius:5px;'><b>ğŸ”„ TURN {n}</b></div>"))

def log_speaker(name, text):
    icon = "ğŸ¤–" if name == "TravelWise" else "ğŸ‘¤"
    color = "#0d6efd" if name == "TravelWise" else "#198754"
    display(Markdown(f"<span style='color:{color}'>**{icon} {name}:**</span> {text}"))

async def check_tunnel():
    url = "http://localhost:8500/sse"
    try:
        async with sse_client(url) as streams:
            read, write = streams
            async with ClientSession(read, write) as session:
                await session.initialize()
                return True
    except:
        return False

# --- SIMULATION LOOP ---
async def run_final_simulation():
    if not await check_tunnel():
        display(Markdown("## âš ï¸� CRITICAL: Tunnel Down."))
        return

    print("ğŸ�¬ STARTING CLEAN SIMULATION...")
    
    try:
        sys_app = getattr(runner, "app_name", "TravelWise")
        user_app = getattr(user_runner, "app_name", "UserSim")
        await runner.session_service.create_session(app_name=sys_app, user_id="sim_user", session_id=SYSTEM_SESSION)
        await user_runner.session_service.create_session(app_name=user_app, user_id="sim_tester", session_id=USER_SESSION)
    except: pass
    
    last_system_response = "Hello! I am ready to plan your trip."
    log_speaker("TravelWise", last_system_response)
    
    for turn in range(1, 4):
        log_turn_header(turn)
        
        # --- A. USER SIMULATOR ---
        user_text = ""
        user_msg = types.Content(role="user", parts=[types.Part(text=f"System says: '{last_system_response}'. Your turn.")])
        
        async for event in user_runner.run_async(user_id="sim_tester", session_id=USER_SESSION, new_message=user_msg):
            if event.content and event.content.parts:
                user_text += event.content.parts[0].text # UserSim is simple text, no tools
        
        log_speaker("User", user_text)
        if "TERMINATE" in user_text: break

        # --- SAFE PACING ---
        for i in range(25, 0, -1):
            sys.stdout.write(f"\râ�³ Recharging... {i}s ")
            sys.stdout.flush()
            time.sleep(1)
        print("\rğŸš€ Sending to Agent...                 ")

        # --- B. TRAVELWISE SYSTEM ---
        token = current_session_id.set(SYSTEM_SESSION)
        system_text = ""
        
        try:
            async for event in runner.run_async(
                user_id="sim_user", 
                session_id=SYSTEM_SESSION, 
                new_message=types.Content(role="user", parts=[types.Part(text=user_text)])
            ):
                try:
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            # ğŸŸ¢ FIX: Use ELIF here
                            if part.function_call:
                                display(HTML(f"<div style='color:#d63384; font-family:monospace; font-size:0.8em; margin-left:20px;'>âš™ï¸� Calling: {part.function_call.name}</div>"))
                            elif part.text:
                                system_text += part.text
                    
                    if getattr(event, 'tool_output', None):
                         display(HTML("<div style='color:green; font-size:0.8em; margin-left:40px;'>âœ“ Result Received</div>"))
                except TypeError:
                    continue 

            log_speaker("TravelWise", system_text)
            last_system_response = system_text
            
        except Exception as e:
            display(Markdown(f"**â�Œ System Error:** {e}"))
        finally:
            current_session_id.reset(token)

# Run
await run_final_simulation()


# --- ğŸ•µï¸�â€�â™‚ï¸� POST-MORTEM OBSERVABILITY (ROBUST) ---
# This tool inspects the internal memory of the agent AFTER the simulation.

async def print_simulation_trace():
    # 1. Configuration (Must match the Simulation cell)
    APP_NAME = "TravelWise"
    USER_ID = "sim_user" 
    TARGET_SESSION = SYSTEM_SESSION 

    print(f"ğŸ”� INSPECTING SESSION TRACE: {TARGET_SESSION}")
    print("=" * 60)

    try:
        # 2. Fetch History from Memory Service
        session = await runner.session_service.get_session(
            app_name=APP_NAME, 
            user_id=USER_ID, 
            session_id=TARGET_SESSION
        )
    except Exception as e:
        print(f"â�Œ Could not retrieve session: {e}")
        return

    if not session or not session.events:
        print("âš ï¸� Session found, but it has no events.")
        return

    # 3. Render the Trace
    print(f"ğŸ“Š TOTAL EVENTS: {len(session.events)}\n")
    
    for i, event in enumerate(session.events):
        step_label = f"STEP {i+1}"
        
        # Safely get the role (user, model, tool/function)
        role = getattr(event, 'role', 'unknown')
        
        # User Input
        if role == "user":
            content = "[No Text]"
            if event.content and event.content.parts:
                content = event.content.parts[0].text[:60] + "..."
            print(f"[{step_label}] ğŸ‘¤ USER: {content}")
            
        # Model Output (Thoughts & Tools)
        elif role == "model":
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.function_call:
                        print(f"[{step_label}] âš™ï¸� TOOL CALL: {part.function_call.name}")
                    if part.text:
                        # Clean up newlines for display
                        clean_text = part.text[:60].replace('\n', ' ')
                        print(f"[{step_label}] ğŸ¤– AGENT: {clean_text}...")
        
        # Tool Output (API Results)
        # Note: Roles can be 'tool' or 'function' depending on API version
        elif role in ["tool", "function"]:
            # Try to grab output from various common locations
            output_text = "Data Received"
            if event.content and event.content.parts:
                # Sometimes it's in a text part
                if event.content.parts[0].text:
                    output_text = event.content.parts[0].text[:60]
            elif getattr(event, 'tool_output', None):
                # Sometimes it's a dedicated attribute
                output_text = str(event.tool_output.output)[:60]
                
            print(f"[{step_label}] ğŸ”Œ TOOL RESULT: {output_text}...")
        
        print("-" * 60)

# Run the trace inspection
await print_simulation_trace()

