pip install -q google-adk[a2a]


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


from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent,AGENT_CARD_WELL_KNOWN_PATH
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner , InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import subprocess
import time
import uuid
import json
import requests
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.tools import AgentTool, FunctionTool, google_search

# Hide additional warnings in the notebook
import warnings

warnings.filterwarnings("ignore")

print("âœ… ADK components Have been imported successfully.")


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


# Define a flights catalog lookup tool
# In a real system, this would query the airline's flight database
def get_flight_info(origin: str, destination: str, date: str) -> str:
    """Retrieve flight-related information based on origin, destination, and travel date..

    This function simulates a flight information lookup. In a real system, it would 
    query an airline's database or API to provide details such as availability, pricing, 
    and aircraft specifications. Currently, it uses a mock catalog for demonstration purposes .

    Args:
        origin (str): The departure city (e.g., "Paris", "Gafsa", "Tunis").
        destination (str): The arrival city.
        date (str): The date of travel in string format (e.g., "2025-12-20").

    Returns:
        str: A string containing flight information if available, or a message 
        listing available flights if the requested route or date is not found.
    """
    # Mock flight catalog - in production, this would query a real airline database
    flight_catalog = {
    "paris-rome-2025-12-20": "Flight: Air France AF123, Paris â†’ Rome, $299, Available Seats: 8, Aircraft: Airbus A320, Departure: 09:00, Arrival: 11:00",
    "gafsa-tunis-2025-12-20": "Flight: Tunisair TU456, Gafsa â†’ Tunis, $99, Available Seats: 31, Aircraft: ATR 72, Departure: 14:30, Arrival: 15:45",
    "tunis-paris-2025-12-20": "Flight: Air France AF789, Tunis â†’ Paris, $349, Available Seats: 45, Aircraft: Boeing 737, Departure: 07:15, Arrival: 10:05",
    "rome-paris-2025-12-21": "Flight: Alitalia AZ101, Rome â†’ Paris, $319, Available Seats: 22, Aircraft: Airbus A321, Departure: 12:00, Arrival: 14:00",
    "tunis-rome-2025-12-20": "Flight: Tunisair TU202, Tunis â†’ Rome, $399, Available Seats: 67, Aircraft: Boeing 737, Departure: 16:00, Arrival: 18:10",
    "paris-london-2025-12-22": "Flight: British Airways BA303, Paris â†’ London, $279, Available Seats: 28, Aircraft: Airbus A320, Departure: 08:30, Arrival: 09:40",
    "rome-tunis-2025-12-23": "Flight: Alitalia AZ505, Rome â†’ Tunis, $349, Sold Out, Expected Seats: Next week",
}

    # Construct the key based on origin, destination, and date
    flight_key = f"{origin.lower()}-{destination.lower()}-{date.strip()}"

    if flight_key in flight_catalog:
        return f"Flight Info: {flight_catalog[flight_key]}"
    else:
        available = ", ".join([f"{f.split('-')[0].title()} â†’ {f.split('-')[1].title()} on {f.split('-')[2]}" 
                           for f in flight_catalog.keys()])
        return f"Sorry, no flight information is available for {origin} â†’ {destination} on {date}. Available flights: {available}"



# Create the Flight Information Agent that specializes in providing flight details from the airline catalog
flight_info_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="flight_info_agent",
    description="External airline flight agent that provides flight schedules, availability, and ticket information.",
    instruction="""
    You are a flight information specialist for an airline.
    When asked about flights, use the get_flight_info tool to fetch data from the flight catalog.
    Provide clear and accurate flight details, including departure and arrival times, prices, available seats, and aircraft type.
    If asked about multiple routes or dates, look up each one.
    Be professional and helpful.
    In your response do not say that I do not have access to information about recommended activities  or weather forecasts , just respond with the flight details
    """,
    tools=[get_flight_info],
    output_key="flight_infos"
)

print("Flight Information Agent created successfully! ğŸ�‰")


# Convert the Flight Information Agent to an A2A-compatible application
# This creates a FastAPI/Starlette app that:
#   1. Serves the agent at the A2A protocol endpoints
#   2. Provides an auto-generated agent card
#   3. Handles A2A communication protocol
flight_info_a2a_app = to_a2a(
    flight_info_agent, port=8001  # Port where this agent will be served
)

print("Flight Information Agent is now A2A-compatible!ğŸ�‰")
print("Agent will be served at: http://localhost:8001")
print("Agent card will be at: http://localhost:8001/.well-known/agent-card.json")
print("Ready to start the server...")



# Save the flight information agent to a file that uvicorn can import
flight_info_agent_code = '''
import os
import time
import requests
import subprocess
from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.genai import types

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

def get_flight_info(origin: str, destination: str, date: str) -> str:
    """Retrieve flight details for a given origin, destination, and travel date."""
    flight_catalog = {
        "paris-rome-2025-12-20": "Flight: Air France AF123, Paris â†’ Rome, $299, Available Seats: 8, Aircraft: Airbus A320, Departure: 09:00, Arrival: 11:00",
        "gafsa-tunis-2025-12-20": "Flight: Tunisair TU456, Gafsa â†’ Tunis, $99, Available Seats: 31, Aircraft: ATR 72, Departure: 14:30, Arrival: 15:45",
        "tunis-paris-2025-12-20": "Flight: Air France AF789, Tunis â†’ Paris, $349, Available Seats: 45, Aircraft: Boeing 737, Departure: 07:15, Arrival: 10:05",
        "rome-paris-2025-12-21": "Flight: Alitalia AZ101, Rome â†’ Paris, $319, Available Seats: 22, Aircraft: Airbus A321, Departure: 12:00, Arrival: 14:00",
        "tunis-rome-2025-12-20": "Flight: Tunisair TU202, Tunis â†’ Rome, $399, Available Seats: 67, Aircraft: Boeing 737, Departure: 16:00, Arrival: 18:10",
        "paris-london-2025-12-22": "Flight: British Airways BA303, Paris â†’ London, $279, Available Seats: 28, Aircraft: Airbus A320, Departure: 08:30, Arrival: 09:40",
        "rome-tunis-2025-12-23": "Flight: Alitalia AZ505, Rome â†’ Tunis, $349, Sold Out, Expected Seats: Next week",
    }

    flight_key = f"{origin.lower()}-{destination.lower()}-{date.strip()}"
    
    if flight_key in flight_catalog:
        return f"Flight Info: {flight_catalog[flight_key]}"
    else:
        available = ", ".join([
            f"{f.split('-')[0].title()} â†’ {f.split('-')[1].title()} on {f.split('-')[2]}"
            for f in flight_catalog.keys()
        ])
        return f"Sorry, no flight information is available for {origin} â†’ {destination} on {date}. Available flights: {available}"

flight_info_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="flight_info_agent",
    description="External airline flight agent that provides flight schedules, availability, and ticket information.",
    instruction="""
    You are a flight information specialist for an airline.
    When asked about flights, use the get_flight_info tool to fetch data from the flight catalog.
    Provide clear, accurate flight details including departure, arrival, price, available seats, and aircraft type.
    If asked about multiple routes or dates, look up each one.
    Be professional and helpful.
    In your response do not say that I do not have access to information about recommended activities  or weather forecasts , just respond with the flight details
    """,
    tools=[get_flight_info],
    output_key="flight_infos"
)

# Create the A2A app
app = to_a2a(flight_info_agent, port=8001)
'''

# Write the flight info agent to a temporary file
with open("/tmp/flight_info_server.py", "w") as f:
    f.write(flight_info_agent_code)

print("ğŸ“� Flight Information agent code saved to /tmp/flight_info_server.py")

# Start uvicorn server in background
server_process = subprocess.Popen(
    [
        "uvicorn",
        "flight_info_server:app",  # Module:app format
        "--host",
        "localhost",
        "--port",
        "8001",
    ],
    cwd="/tmp",  # Run from /tmp where the file is
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env={**os.environ},  # Pass environment variables
)

print("ğŸš€ Starting Flight Information Agent server...")
print("   Waiting for server to be ready...")

# Wait for server to start (poll until it responds)
max_attempts = 30
for attempt in range(max_attempts):
    try:
        response = requests.get(
            "http://localhost:8001/.well-known/agent-card.json", timeout=1
        )
        if response.status_code == 200:
            print(f"\nâœ… Flight Information Agent server is running!")
            print(f"   Server URL: http://localhost:8001")
            print(f"   Agent card: http://localhost:8001/.well-known/agent-card.json")
            break
    except requests.exceptions.RequestException:
        time.sleep(5)
        print(".", end="", flush=True)
else:
    print("\nâš ï¸�  Server may not be ready yet. Check manually if needed.")

# Store the process so we can stop it later
globals()["flight_info_server_process"] = server_process


import requests
import json

# Fetch the agent card from the running Flight Information Agent server
try:
    response = requests.get(
        "http://localhost:8001/.well-known/agent-card.json", timeout=5
    )

    if response.status_code == 200:
        agent_card = response.json()
        print("ğŸ“‹ Flight Information Agent Card:")
        print(json.dumps(agent_card, indent=2))

        print("\nâœ¨ Key Information:")
        print(f"   Name: {agent_card.get('name')}")
        print(f"   Description: {agent_card.get('description')}")
        print(f"   URL: {agent_card.get('url')}")
        print(f"   Skills: {len(agent_card.get('skills', []))} capabilities exposed")
    else:
        print(f"â�Œ Failed to fetch agent card: {response.status_code}")

except requests.exceptions.RequestException as e:
    print(f"â�Œ Error fetching agent card: {e}")
    print("   Make sure the Flight Information Agent server is running on port 8001")



# Function to fetch activities in a destination city
def get_activities(destination: str) -> str:
    """Retrieve recommended activities for a given destination city.

    This function simulates a travel activities lookup. In a real system, it could
    query a tourism API or database to provide popular activities in the city.

    Args:
        destination (str): The city to explore (e.g., "Paris", "Gafsa", "Tunis").

    Returns:
        str: A string listing popular activities in the destination, or a message
        listing available destinations if the requested city is not found.
    """
    # Mock activities catalog
    activities_catalog = {
        "paris": "Eiffel Tower Visit, Louvre Museum Tour, Seine River Cruise",
        "london": "London Eye, British Museum, Thames River Cruise",
        "rome": "Colosseum Tour, Vatican Museums, Trevi Fountain Visit",
        "tunis": "Medina of Tunis, Bardo Museum, Carthage Ruins Tour",
    }

    city_key = destination.lower().strip()
    
    if city_key in activities_catalog:
        return f"Activities in {destination.title()}: {activities_catalog[city_key]}"
    else:
        available = ", ".join([city.title() for city in activities_catalog.keys()])
        return f"Sorry, I don't have activities listed for {destination.title()}. Available destinations: {available}"


# Create the Activity Agent that specializes in providing activities at destinations
activity_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="activity_agent",
    description="External travel agent that provides popular activities and sightseeing options in destination cities.",
    instruction="""
    You are a travel activities specialist.
    When asked about activities, use the get_activities tool to fetch data from the activities catalog.
    Provide clear and helpful recommendations for sightseeing, tours, and local experiences.
    If asked about multiple destinations, provide recommendations for each.
    Be professional and friendly.
    In your response do not say that I do not have access to information about flight details  or weather forecasts, just respond with the activities recommendations.
    """,
    tools=[get_activities],
    output_key="activities_details"
)

print("Activity Agent created successfully! ğŸ�‰")


# Convert the Activity Agent to an A2A-compatible application
# This creates a FastAPI/Starlette app that:
#   1. Serves the agent at the A2A protocol endpoints
#   2. Provides an auto-generated agent card
#   3. Handles A2A communication protocol
activity_a2a_app = to_a2a(
    activity_agent, port=8002  # Port where this agent will be served
)

print("Activity Agent is now A2A-compatible! ğŸ�‰")
print("Agent will be served at: http://localhost:8002")
print("Agent card will be at: http://localhost:8002/.well-known/agent-card.json")
print("Ready to start the server...")



# Save the Activity Agent to a file that uvicorn can import
activity_agent_code = '''
import os
import time
import requests
import subprocess
from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.genai import types

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

def get_activities(destination: str) -> str:
    """Retrieve recommended activities for a given destination city."""
    activities_catalog = {
        "paris": "Eiffel Tower Visit, Louvre Museum Tour, Seine River Cruise",
        "london": "London Eye, British Museum, Thames River Cruise",
        "rome": "Colosseum Tour, Vatican Museums, Trevi Fountain Visit",
        "tunis": "Medina of Tunis, Bardo Museum, Carthage Ruins Tour",
    }

    city_key = destination.lower().strip()
    
    if city_key in activities_catalog:
        return f"Activities in {destination.title()}: {activities_catalog[city_key]}"
    else:
        available = ", ".join([city.title() for city in activities_catalog.keys()])
        return f"Sorry, I don't have activities listed for {destination.title()}. Available destinations: {available}"

activity_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="activity_agent",
    description="External travel agent that provides popular activities and sightseeing options in destination cities.",
    instruction="""
    You are a travel activities specialist.
    When asked about activities, use the get_activities tool to fetch data from the activities catalog.
    Provide clear and helpful recommendations for sightseeing, tours, and local experiences.
    If asked about multiple destinations, provide recommendations for each.
    Be professional and friendly.
    In your response do not say that I do not have access to information about flight details  or weather forecasts, just respond with the activities recommendations.

    """,
    tools=[get_activities],
    output_key="activities_details"
)

# Create the A2A app
app = to_a2a(activity_agent, port=8002)
'''

# Write the activity agent to a temporary file
with open("/tmp/activity_server.py", "w") as f:
    f.write(activity_agent_code)

print("ğŸ“� Activity Agent code saved to /tmp/activity_server.py")

# Start uvicorn server in background
activity_server_process = subprocess.Popen(
    [
        "uvicorn",
        "activity_server:app",  # Module:app format
        "--host",
        "localhost",
        "--port",
        "8002",
    ],
    cwd="/tmp",  # Run from /tmp where the file is
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env={**os.environ},  # Pass environment variables
)

print("ğŸš€ Starting Activity Agent server...")
print("   Waiting for server to be ready...")

# Wait for server to start (poll until it responds)
max_attempts = 30
for attempt in range(max_attempts):
    try:
        response = requests.get(
            "http://localhost:8002/.well-known/agent-card.json", timeout=1
        )
        if response.status_code == 200:
            print(f"\nâœ… Activity Agent server is running!")
            print(f"   Server URL: http://localhost:8002")
            print(f"   Agent card: http://localhost:8002/.well-known/agent-card.json")
            break
    except requests.exceptions.RequestException:
        time.sleep(5)
        print(".", end="", flush=True)
else:
    print("\nâš ï¸�  Server may not be ready yet. Check manually if needed.")

# Store the process so we can stop it later
globals()["activity_server_process"] = activity_server_process



import requests
import json

# Fetch the agent card from the running Activity Agent server
try:
    response = requests.get(
        "http://localhost:8002/.well-known/agent-card.json", timeout=5
    )

    if response.status_code == 200:
        agent_card = response.json()
        print("ğŸ“‹ Activity Agent Card:")
        print(json.dumps(agent_card, indent=2))

        print("\nâœ¨ Key Information:")
        print(f"   Name: {agent_card.get('name')}")
        print(f"   Description: {agent_card.get('description')}")
        print(f"   URL: {agent_card.get('url')}")
        print(f"   Skills: {len(agent_card.get('skills', []))} capabilities exposed")
    else:
        print(f"â�Œ Failed to fetch agent card: {response.status_code}")

except requests.exceptions.RequestException as e:
    print(f"â�Œ Error fetching agent card: {e}")
    print("   Make sure the Activity Agent server is running on port 8002")


# ---- Weather Agent ----
def get_weather(city: str, date: str) -> str:
    """
    Retrieve weather information for a specific city and date.

    This function simulates a weather lookup. In a real system, it could query
    a weather API to provide detailed forecasts including temperature, precipitation, 
    and conditions.

    Args:
        city (str): The city to get the weather for (e.g., "Paris", "Rome").
        date (str): The date for the weather forecast (e.g., "2025-12-20").

    Returns:
        str: A string describing the weather for the given city and date, 
        or a message listing available forecasts if the requested city/date is not found.
    """
    # Mock weather catalog
    weather_catalog = {
        "rome-2025-12-20": "Sunny, 18Â°C",
        "london-2025-03-10": "Cloudy, 12Â°C",
        "paris-2025-12-20": "Rainy, 10Â°C",
    }

    key = f"{city.lower().strip()}-{date.strip()}"
    if key in weather_catalog:
        return f"Weather in {city.title()} on {date}: {weather_catalog[key]}"
    else:
        available = ", ".join([f"{k.split('-')[0].title()} on {k.split('-')[1]}" for k in weather_catalog.keys()])
        return f"Sorry, no weather info for {city.title()} on {date}. Available forecasts: {available}"


# Create the Weather Agent
weather_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="weather_agent",
    description="External weather agent that provides forecasts for cities and dates.",
    instruction="""
    You are a weather forecast specialist.
    When asked about weather, use the get_weather tool to fetch data from the weather catalog.
    Provide clear and accurate weather details including temperature, conditions, and precipitation.
    If asked about multiple cities or dates, provide forecasts for each.
    Be professional and helpful.
    In your response do not say that I do not have access to information about flight details  or activity recommendations, just respond with the weather forecasts.
    """,
    tools=[get_weather],
    output_key="weather_details"
)

print("Weather Agent created successfully! ğŸŒ¤ï¸�")


# Convert the Weather Agent to an A2A-compatible application
# This creates a FastAPI/Starlette app that:
#   1. Serves the agent at the A2A protocol endpoints
#   2. Provides an auto-generated agent card
#   3. Handles A2A communication protocol
weather_a2a_app = to_a2a(
    weather_agent, port=8003  # Port where this agent will be served
)

print("Weather Agent is now A2A-compatible! ğŸ�‰")
print("Agent will be served at: http://localhost:8003")
print("Agent card will be at: http://localhost:8003/.well-known/agent-card.json")
print("Ready to start the server...")


# Save the Weather Agent to a file that uvicorn can import
weather_agent_code = '''
import os
import time
import requests
import subprocess
from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.genai import types

# Retry configuration
retry_config = types.HttpRetryOptions(
    attempts=5,          # Maximum retry attempts
    exp_base=7,          # Delay multiplier
    initial_delay=1,     # Initial delay in seconds
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

def get_weather(city: str, date: str) -> str:
    """
    Retrieve weather forecast for a given city and date.

    This function simulates a weather lookup. In a real system, it could query
    a weather API to provide temperature, conditions, and precipitation.

    Args:
        city (str): Name of the city (e.g., "Paris", "Rome").
        date (str): Date of interest in string format (e.g., "2025-12-20").

    Returns:
        str: A string describing the weather, or a message listing available cities if not found.
    """
    # Mock weather catalog
    weather_catalog = {
        "paris-2025-12-20": "Rainy, 10Â°C",
        "rome-2025-12-20": "Sunny, 18Â°C",
        "london-2025-03-10": "Cloudy, 12Â°C",
        "tunis-2025-12-20": "Sunny, 22Â°C",
    }

    key = f"{city.lower().strip()}-{date.strip()}"
    
    if key in weather_catalog:
        return f"Weather in {city.title()} on {date}: {weather_catalog[key]}"
    else:
        available = ", ".join([f"{k.split('-')[0].title()} on {k.split('-')[1]}" for k in weather_catalog.keys()])
        return f"Sorry, no weather information is available for {city.title()} on {date}. Available: {available}"

# Create the Weather Agent
weather_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="weather_agent",
    description="External weather agent that provides forecasts for cities on specific dates.",
    instruction="""
    You are a weather forecast specialist.
    When asked about weather, use the get_weather tool to fetch data from the weather catalog.
    Provide clear and accurate information about temperature, conditions, and precipitation.
    If asked about multiple cities or dates, provide forecasts for each.
    Be professional and friendly.
    In your response do not say that I do not have access to information about flight details  or activity recommendations; just respond with the weather forecasts.
    """,
    tools=[get_weather],
    output_key="weather_details"
)

# Create the A2A app for the Weather Agent
app = to_a2a(weather_agent, port=8003)
'''

# Write the Weather Agent to a temporary file
with open("/tmp/weather_server.py", "w") as f:
    f.write(weather_agent_code)

print("ğŸ“� Weather Agent code saved to /tmp/weather_server.py")

# Start uvicorn server in background
weather_server_process = subprocess.Popen(
    [
        "uvicorn",
        "weather_server:app",  # Module:app format
        "--host",
        "localhost",
        "--port",
        "8003",
    ],
    cwd="/tmp",  # Run from /tmp where the file is
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env={**os.environ},  # Pass environment variables
)

print("ğŸš€ Starting Weather Agent server...")
print("   Waiting for server to be ready...")

# Wait for server to start (poll until it responds)
max_attempts = 30
for attempt in range(max_attempts):
    try:
        response = requests.get(
            "http://localhost:8003/.well-known/agent-card.json", timeout=1
        )
        if response.status_code == 200:
            print(f"\nâœ… Weather Agent server is running!")
            print(f"   Server URL: http://localhost:8003")
            print(f"   Agent card: http://localhost:8003/.well-known/agent-card.json")
            break
    except requests.exceptions.RequestException:
        time.sleep(5)
        print(".", end="", flush=True)
else:
    print("\nâš ï¸�  Server may not be ready yet. Check manually if needed.")

# Store the process so we can stop it later
globals()["weather_server_process"] = weather_server_process


import requests
import json

AGENT_URL = "http://localhost:8003/.well-known/agent-card.json"

try:
    response = requests.get(AGENT_URL, timeout=5)

    if response.status_code == 200:
        agent_card = response.json()
        print("ğŸ“‹ Weather Agent Card:")
        print(json.dumps(agent_card, indent=2))

        print("\nâœ¨ Key Information:")
        print(f"   Name       : {agent_card.get('name')}")
        print(f"   Description: {agent_card.get('description')}")
        print(f"   URL        : {agent_card.get('url')}")
        print(f"   Skills     : {len(agent_card.get('skills', []))} capabilities exposed")
    else:
        print(f"â�Œ Failed to fetch agent card. HTTP Status Code: {response.status_code}")

except requests.exceptions.RequestException as e:
    print(f"â�Œ Error fetching agent card: {e}")
    print("   Make sure the Weather Agent server is running on port 8003")



# Create a RemoteA2aAgent that connects to our Flight Catalog Agent
# This acts as a client-side proxy - the Customer Support Agent can use it like a local agent
remote_flight_catalog_agent = RemoteA2aAgent(
    name="flight_catalog_agent",
    description="Remote flight catalog agent from external vendor that provides product information.",
    # Point to the agent card URL - this is where the A2A protocol metadata lives
    agent_card=f"http://localhost:8001{AGENT_CARD_WELL_KNOWN_PATH}",
)

print("âœ… Remote Flight Catalog Agent proxy created!")
print(f"   Connected to: http://localhost:8001")
print(f"   Agent card: http://localhost:8001{AGENT_CARD_WELL_KNOWN_PATH}")
print("   The Customer Support Agent can now use this like a local sub-agent!")


# Create a RemoteA2aAgent that connects to our activities Catalog Agent
# This acts as a client-side proxy - the Customer Support Agent can use it like a local agent
remote_activity_catalog_agent = RemoteA2aAgent(
    name="activity_catalog_agent",
    description="Remote activity catalog agent from external vendor that provides product information.",
    # Point to the agent card URL - this is where the A2A protocol metadata lives
    agent_card=f"http://localhost:8002{AGENT_CARD_WELL_KNOWN_PATH}",
)

print("âœ… Remote Activity Catalog Agent proxy created!")
print(f"   Connected to: http://localhost:8002")
print(f"   Agent card: http://localhost:8002{AGENT_CARD_WELL_KNOWN_PATH}")
print("   The Customer Support Agent can now use this like a local sub-agent!")


# Create a RemoteA2aAgent that connects to our weather Catalog Agent
# This acts as a client-side proxy - the Customer Support Agent can use it like a local agent
remote_weather_catalog_agent = RemoteA2aAgent(
    name="weather_catalog_agent",
    description="Remote weather catalog agent from external vendor that provides product information.",
    # Point to the agent card URL - this is where the A2A protocol metadata lives
    agent_card=f"http://localhost:8003{AGENT_CARD_WELL_KNOWN_PATH}",
)

print("âœ… Remote Weather Catalog Agent proxy created!")
print(f"   Connected to: http://localhost:8003")
print(f"   Agent card: http://localhost:8003{AGENT_CARD_WELL_KNOWN_PATH}")
print("   The Customer Support Agent can now use this like a local sub-agent!")


# The AggregatorAgent runs *after* the parallel step to synthesize the results.
aggregator_agent = LlmAgent(
    name="AggregatorAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # It uses placeholders to inject the outputs from the parallel agents, which are now in the session state.
    instruction="""Combine these three details (flight details, activities, and weather) into a single executive response.

   Your summary should highlight the flight details, activity details, and weather forecast in the destination city for that date,  the most important key takeaways from all three reports. The final summary should be around 500 words.
    """,
    output_key="executive_response",  # This will be the final output of the entire system.
)

print("âœ… aggregator_agent created.")


# The ParallelAgent runs all its sub-agents simultaneously.
parallel_research_team = ParallelAgent(
    name="ParallelagentsTeam",
    sub_agents=[remote_flight_catalog_agent, remote_activity_catalog_agent, remote_weather_catalog_agent],
)



# This SequentialAgent defines the high-level workflow: run the parallel team first, then run the aggregator.
root_agent = SequentialAgent(
    name="TravelSystem",
    sub_agents=[parallel_research_team, aggregator_agent],
)

print("âœ… Parallel and Sequential Agents created.")


traveler_support_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="traveler_support_agent",
    description="A travel assistant that provides flights, activities, and weather by querying root agent.",
    instruction="""
    You are a professional and friendly travel assistant.
    Always call `root_agent` using A2A protocol to handle the request.
    Then return only the aggregated travel summary contained in `executive_response`.
    Do not ask for route confirmation. Do not add extra comments.
    Show only the summary from  the aggregator agent located in executive_response output_key 
    """,
    sub_agents=[root_agent],
    output_key="final_trip_summary"
)


async def test_a2a_communication(user_query: str):
    """
    Test the A2A communication between Travel Support Agent and flight, activities, and weather Catalog Agents.

    This function:
    1. Creates a new session for this conversation
    2. Sends the query to the Traveler Support Agent
    3. Support Agent communicates with Product Catalog Agent via A2A
    4. Displays the response

    Args:
        user_query: The question to ask the Traveler Support Agent
    """
    # Setup session management (required by ADK)
    session_service = InMemorySessionService()

    # Session identifiers
    app_name = "travel_app"
    user_id = "traveler_user"
    # Use unique session ID for each test to avoid conflicts
    session_id = f"demo_session_{uuid.uuid4().hex[:8]}"

    # CRITICAL: Create session BEFORE running agent (synchronous, not async!)
    # This pattern matches the deployment notebook exactly
    session = await session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )

    # Create runner for the Traveler Support Agent
    # The runner manages the agent execution and session state
    runner = Runner(
        agent=traveler_support_agent, app_name=app_name, session_service=session_service
    )

    # Create the user message
    # This follows the same pattern as the deployment notebook
    test_content = types.Content(parts=[types.Part(text=user_query)])

    # Display query
    print(f"\nğŸ‘¤ Traveler: {user_query}")
    print(f"\nğŸ�§ Travel Support Agent response:")
    print("-" * 60)

    # Run the agent asynchronously (handles streaming responses and A2A communication)
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=test_content
    ):
        # Print final response only (skip intermediate events)
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if hasattr(part, "text"):
                    print(part.text)

    print("-" * 60)


# Run the test
print("ğŸ§ª Testing A2A Communication...\n")



await test_a2a_communication(""" 
Plan a trip from tunis to paris on 2025-12-20. Include: 
Flight details (departure, arrival, aircraft, price, availability)
Recommended activities in paris
Weather forecast for paris on that date
""")


await test_a2a_communication("""
i will travel  from Paris to Rome on 2025-12-20. give me  flight details also the activities and weather in destination on that date 
""")




