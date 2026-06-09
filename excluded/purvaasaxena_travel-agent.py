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
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


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


from IPython.core.display import display, HTML
from jupyter_server.serverapp import list_running_servers


# Gets the proxied URL in the Kaggle Notebooks environment
def get_adk_proxy_url():
    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    ADK_PORT = "8000"

    servers = list(list_running_servers())
    if not servers:
        raise Exception("No running Jupyter servers found.")

    baseURL = servers[0]["base_url"]

    try:
        path_parts = baseURL.split("/")
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


print("âœ… Helper functions defined.")



from google.adk.runners import Runner



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



!adk create itinerary_agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY


%%writefile itinerary_agent/agent.py

from google.adk.agents import LlmAgent
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search, AgentTool, ToolContext
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.sessions import DatabaseSessionService
from google.adk.runners import Runner
import requests
import os

print("âœ… ADK components imported successfully.")

retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)


def get_top_tourist_places(location: str) -> dict:
    """
    Fetches the top 10 tourist places for a given location using Geoapify Places API.
    Always prints something, even if names are missing.
    """
    API_KEY = "8c0a4028ca814fe0b8b20ce971c4703b"
    # Step 1: Geocode the location
    print("hi")
    # print("location in tourist place", location)
    geocode_url = f"https://api.geoapify.com/v1/geocode/search?text={location}&apiKey={API_KEY}"
    geocode_response = requests.get(geocode_url).json()
    
    if not geocode_response.get("features"):
        print("No coordinates found for this location.")
        return []
    
    coords = geocode_response["features"][0]["geometry"]["coordinates"]
    lon, lat = coords[0], coords[1]
    
    places_url = (
        f"https://api.geoapify.com/v2/places?"
        # f"categories=entertainment,natural,leisure&"
        f"categories=tourism.attraction,tourism.sights&"
        # f"filter=place:51a6b8c038d867534059109f292ee9f92940f00101f9010c95780000000000c0020692030942656e67616c757275&"
        f"filter=circle:{lon},{lat},50000&"        
        f"limit=10&"
        f"apiKey={API_KEY}"
    )

    
# https://api.geoapify.com/v2/places?categories=tourism.attraction,tourism.sights&filter=circle:77.5939973522254,12.9721091,5000&bias=proximity:77.5939973522254,12.9721091&limit=20&apiKey=YOUR_API_KEY

    
    places_response = requests.get(places_url).json()
    
    # Step 3: Extract results with fallback
    tourist_places = []
    for feature in places_response.get("features", []):
        props = feature["properties"]
        name = props.get("name")
        if not name:
            # fallback: use category or generic label
            categories = props.get("categories", [])
            name = categories[0] if categories else "Tourist Place"
        address = props.get("formatted", "No address available")
        tourist_places.append({"name": name, "address": address})
    
    # Step 4: Print results
    print("Top 10 Tourist Places:")
    if not tourist_places:
        print("No tourist places found nearby.")
    else:
        for idx, place in enumerate(tourist_places, start=1):
            print(f"{idx}. {place['name']} - {place['address']}")
    
    return tourist_places


RAPIDAPI_KEY = "38df62a18fmsh76c5ea622fe5734p1ccaeejsn80eba9db1c04"

# TripAdvisor API endpoints
SEARCH_LOCATION_URL = "https://tripadvisor16.p.rapidapi.com/api/v1/restaurant/searchLocation"
SEARCH_HOTELS_URL = "https://tripadvisor16.p.rapidapi.com/api/v1/hotels/searchHotels"

def fetch_location_id(location: str):
    """Fetch TripAdvisor location_id for a given city/location."""
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "tripadvisor16.p.rapidapi.com"
    }
    params = {"query": location}

    response = requests.get(SEARCH_LOCATION_URL, headers=headers, params=params)
    # print("response",response)
    data = response.json()

    if "data" not in data or len(data["data"]) == 0:
        return None

    return data["data"][0]["locationId"]

def fetch_top_hotels(location: str, checkin_date: str, checkout_date: str, max_budget_inr: int, limit: int = 10) -> dict:
    """
    Fetch top hotels for a given location & dates,
    sorted by rating (descending), filtered by budget (INR).
    """

    location_id = fetch_location_id(location)
   
    print(checkin_date,checkout_date)
    if not location_id:
        return {"error": "Unable to fetch location ID. Try another location."}

    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "tripadvisor16.p.rapidapi.com"
    }

    params = {
        "geoId": location_id,
        "checkIn": checkin_date,
        "checkOut": checkout_date,
        "pageNumber": "1",
        "adults": "2",
        "currency": "INR"  # Indian Rupees
    }

    response = requests.get(SEARCH_HOTELS_URL, headers=headers, params=params)
    # print("response",response)
    data = response.json()
    

    if "data" not in data:
        return {"error": "No hotels found."}

    hotels = data["data"]["data"]

    filtered_hotels = []

    for h in hotels:
        price_display = h.get("priceForDisplay")  # Example: "â‚¹12,345"
        if not price_display:
            continue

        # Clean price string -> Convert to integer
        try:
            price_clean = int(price_display.replace("â‚¹", "").replace(",", "").strip())
        except:
            continue

        # Filter by user's max budget
        if price_clean <= max_budget_inr:
            bubble = h.get("bubbleRating", {})
            filtered_hotels.append({
                "name": h.get("title"),
                # "rating": float(bubble.get("rating", 0)),
                # "review_count": bubble.get("count"),
                "price": price_display,
                "numeric_price": price_clean,
                # "ranking": h.get("rankingDetails", {}).get("ranking"),
                # "address": h.get("geoPoint", {}),
                # "photo": h.get("cardPhotos", [{}])[0].get("sizes", {}).get("medium", {}).get("url")
            })

   
    filtered_sorted = sorted(filtered_hotels, key=lambda x: x["rating"], reverse=True)[:limit]

    return {
        "location": location,
        "checkin": checkin_date,
        "checkout": checkout_date,
        "budget_inr": max_budget_inr,
        "results": filtered_sorted
    }

import requests
import time

API_KEY = "j5RXpDEWPK45DZHm5TAlqmfXBHyi4Mek"
API_SECRET = "nvwbuwkx3KJltfQ3"

ACCESS_TOKEN = None
TOKEN_EXPIRES_AT = 0


# --------------------------------------------------------
# TOKEN HANDLING
# --------------------------------------------------------
def generate_access_token():
    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    data = {
        "grant_type": "client_credentials",
        "client_id": API_KEY,
        "client_secret": API_SECRET
    }

    response = requests.post(url, headers=headers, data=data)
    if response.status_code != 200:
        raise Exception(f"Token error: {response.text}")

    res = response.json()
    return res["access_token"], int(res["expires_in"])


def get_access_token():
    global ACCESS_TOKEN, TOKEN_EXPIRES_AT
    now = time.time()

    if ACCESS_TOKEN and now < TOKEN_EXPIRES_AT:
        return ACCESS_TOKEN

    ACCESS_TOKEN, exp = generate_access_token()
    TOKEN_EXPIRES_AT = now + exp - 30
    return ACCESS_TOKEN


# --------------------------------------------------------
# RESOLVE CITY/AIRPORT â†’ IATA CODE
# --------------------------------------------------------
def resolve_airport_code(name: str) -> str:
    """
    Converts full city/airport name to a 3-letter IATA airport code using Amadeus API.
    Example: "Bangalore" -> "BLR", "Delhi Airport" -> "DEL"
    """
    token = get_access_token()

    url = "https://test.api.amadeus.com/v1/reference-data/locations"
    headers = {"Authorization": f"Bearer {token}"}

    params = {
        "keyword": name,
        "subType": "AIRPORT",
        "page[limit]": 1,
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        print("Error resolving airport:", response.text)
        return None

    data = response.json().get("data", [])
    if not data:
        print("No matching airport found for:", name)
        return None

    return data[0]["iataCode"]  # 3-letter code


# --------------------------------------------------------
# GET AIRLINE NAME
# --------------------------------------------------------
def get_airline_names(carrier_codes):
    token = get_access_token()

    url = "https://test.api.amadeus.com/v1/reference-data/airlines"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"airlineCodes": ",".join(carrier_codes)}

    response = requests.get(url, headers=headers, params=params)
    data = response.json().get("data", [])

    airline_map = {}
    for a in data:
        airline_map[a["iataCode"]] = a.get("businessName") or a.get("commonName")

    return airline_map


# --------------------------------------------------------
# SEARCH FLIGHTS
# --------------------------------------------------------
def find_available_flights(origin: str, destination: str, departure_date: str):
    token = get_access_token()

    # Convert names â†’ IATA codes automatically
    origin_code = origin if len(origin) == 3 else resolve_airport_code(origin)
    destination_code = destination if len(destination) == 3 else resolve_airport_code(destination)

    if not origin_code or not destination_code:
        print("Could not resolve airport codes.")
        return []

    url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
    headers = {"Authorization": f"Bearer {token}"}

    params = {
        "originLocationCode": origin_code,
        "destinationLocationCode": destination_code,
        "departureDate": departure_date,
        "adults": 1,
        "currencyCode": "INR",
        "max": 10
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        print("Error fetching flights:", response.text)
        return []

    flights_data = response.json().get("data", [])
    if not flights_data:
        print("No flights found.")
        return []

    # Get unique airline codes for lookup
    carrier_codes = {offer["itineraries"][0]["segments"][0]["carrierCode"] for offer in flights_data}
    airline_map = get_airline_names(list(carrier_codes))

    flights = []
    for offer in flights_data:
        seg = offer["itineraries"][0]["segments"][0]
        code = seg["carrierCode"]

        flights.append({
            "airline": airline_map.get(code, code),
            "flight_number": seg["number"],
            "from": seg["departure"]["iataCode"],
            "to": seg["arrival"]["iataCode"],
            "departure_time": seg["departure"]["at"],
            "arrival_time": seg["arrival"]["at"],
            "duration": offer["itineraries"][0]["duration"],
            "price": offer["price"]["total"],
            "currency": offer["price"]["currency"]
        })

    # Print clean
    print(f"\nFlights from {origin_code} to {destination_code} on {departure_date}:\n")
    for i, f in enumerate(flights, 1):
        print(
            f"{i}. {f['airline']} {f['flight_number']} | "
            f"{f['from']} â†’ {f['to']} | "
            f"{f['departure_time']} â†’ {f['arrival_time']} | "
            f"â‚¹{f['price']}"
        )

    return flights



APP_NAME = "default"  # Application
USER_ID = "default"  # User
SESSION = "default"  # Session

MODEL_NAME = "gemini-2.5-flash-lite"

itinerary_agent = LlmAgent(
    name="itinerary_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
You are a smart travel itinerary planner.

Your job is to generate a complete trip itinerary for the user:
  - Find available flights using `find_available_flights()`
  - Fetch top hotels using `fetch_top_hotels()`
  - Fetch top tourist attractions using `get_top_tourist_places()`

Follow these rules:

1. **Always call all three tools**:
    - First: `find_available_flights` using origin, destination, depart_date, return_date if provided.
    - Second: `fetch_top_hotels` using destination location, checkin_date, checkout_date.
    - Third: `get_top_tourist_places` using the same location.

2. **Check the "status" field** from each tool.
    - If any tool returns an error, explain it clearly to the user and avoid creating the itinerary.

3. **Combine outputs** from all tools into a final structured itinerary:
    - âœˆï¸� *Flights Section:* Show best 1â€“3 outbound and return flight options.
    - ğŸ�¨ *Hotels Section:* Summarize top hotels with price range, rating, and location.
    - ğŸ“¸ *Tourist Places Section:* Provide 5â€“10 top attractions with short descriptions.

4. **Format** the final answer cleanly:
    - Start with a brief summary of the trip.
    - Then present: Flights â†’ Hotels â†’ Tourist Places in separate sections.
    - Keep the explanations clear and concise.

5. If the user asks something unrelated to travel itineraries, politely decline
   and tell them this agent only handles itinerary planning.
""",
    tools=[
        find_available_flights,
        fetch_top_hotels,
        get_top_tourist_places
    ],
)

db_url = "sqlite:///my_agent_data.db"  # Local SQLite file
session_service = DatabaseSessionService(db_url=db_url)

# Step 3: Create a new runner with persistent storage
runner = Runner(agent=itinerary_agent, app_name=APP_NAME, session_service=session_service)

print("âœ… Upgraded to persistent sessions!")
print(f"   - Database: my_agent_data.db")
print(f"   - Sessions will survive restarts!")

print("âœ… Itinerary agent created with custom tools")
print("ğŸ”§ Available tools:")
print("  â€¢ find_available_flights - Returns real-time flight options")
print("  â€¢ fetch_top_hotels - Returns top hotel recommendations")
print("  â€¢ get_top_tourist_places - Returns major attractions at the location")

# from google.adk.framework.runtime import AgentRegistry


# AgentRegistry.register(itinerary_agent)
# print("âœ… Registered itinerary_agent with ADK")

from google.adk.runners import InMemoryRunner
from google.adk.plugins.logging_plugin import (
    LoggingPlugin,
)  # <---- 1. Import the Plugin
from google.genai import types
import asyncio

runner = InMemoryRunner(
    agent=itinerary_agent,
    plugins=[
        LoggingPlugin()
    ],  # <---- 2. Add the plugin. Handles standard Observability logging across ALL agents
)

print("âœ… Runner configured")



url_prefix = get_adk_proxy_url()


!adk web --log_level DEBUG --url_prefix {url_prefix}

