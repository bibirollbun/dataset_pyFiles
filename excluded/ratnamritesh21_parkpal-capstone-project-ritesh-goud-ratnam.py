# Install core dependencies
!pip install -q google-adk requests opentelemetry-instrumentation-google-genai

# --- Core Python imports ---
import os, sys, json, time, logging, requests
from kaggle_secrets import UserSecretsClient

# --- Google ADK imports ---
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.genai import types
from google.adk.tools import load_memory, preload_memory

print("âœ… Libraries imported successfully")


# Configure Gemini + NPS API keys
user_secrets = UserSecretsClient()

# Load keys from Kaggle Secrets
GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
NPS_API_KEY     = user_secrets.get_secret("NPS_API_KEY")

# Inject Gemini key into environment
os.environ["GOOGLE_API_KEY"]            = GOOGLE_API_KEY
os.environ["GOOGLE_GENAI_API_KEY"]      = GOOGLE_API_KEY
os.environ["GENAI_API_KEY"]             = GOOGLE_API_KEY
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "0"
os.environ["USE_VERTEX_AI"]             = "0"

# Disable Kaggleâ€™s pre-loaded Vertex AI libraries to avoid conflicts
sys.modules["google.cloud.aiplatform"]        = None
sys.modules["google.cloud.aiplatform_v1beta1"] = None

print("âœ… Environment configured â€” Gemini and NPS keys loaded securely.")


from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
user_credential = user_secrets.get_gcloud_credential()
user_secrets.set_tensorflow_credential(user_credential)

print("âœ… Cloud credentials configured")

# --- Load Project ID from Kaggle Secrets ---
PROJECT_ID = user_secrets.get_secret("GCP_PROJECT_ID")
if not PROJECT_ID:
    raise ValueError("âš ï¸� Please add 'GCP_PROJECT_ID' to your Kaggle Secrets before running.")

os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
print(f"âœ… GCP Project configured securely")



# Configure logging
logging.basicConfig(
    filename="parkpal.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
print("âœ… Logging configured successfully")

# Constants
APP_NAME = "ParkPal"
USER_ID = "traveler_001"

# Gemini retry logic
retry_config = types.HttpRetryOptions(
    attempts=3,
    exp_base=2,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)
print("âœ… Retry configuration set successfully")


#  Define tools for ParkPal agent: park info, campgrounds, activities, trails, and weather

import requests
from difflib import SequenceMatcher
from kaggle_secrets import UserSecretsClient
from datetime import datetime
import pytz


#  Helper: Safe Markdown hyperlink formatter
def format_url(url):
    """Return clean markdown hyperlink for consistent display."""
    if url and url.startswith("http"):
        # Extract last part (park code or path name) for readability
        park_code = url.strip("/").split("/")[-2].upper() if "/" in url else "NPS"
        return f"[{park_code} Official Page]({url})"
    return "No link available"

#  Convert UTC timestamps (from Open-Meteo) to park's local timezone
def format_utc_to_local(utc_str: str, timezone="US/Mountain"):
    """Convert UTC timestamp to the park's local timezone."""
    utc = datetime.strptime(utc_str, "%Y-%m-%dT%H:%M")
    local_tz = pytz.timezone(timezone)
    return utc.replace(tzinfo=pytz.utc).astimezone(local_tz).strftime("%Y-%m-%d %I:%M %p %Z")


#  API setup
user_secrets = UserSecretsClient()
NPS_API_KEY = user_secrets.get_secret("NPS_API_KEY")
BASE_URL = "https://developer.nps.gov/api/v1"


def similar(a, b):
    """Helper to measure text similarity between names."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


#  Step 1: Resolve park name â†’ metadata
def resolve_park(park_name: str):
    """Find the most relevant park entry and return key details."""
    url = f"{BASE_URL}/parks"
    params = {"limit": 500, "api_key": NPS_API_KEY}
    res = requests.get(url, params=params)
    parks = res.json().get("data", [])

    best_match, best_score = None, 0.0
    for park in parks:
        score = similar(park_name, park["fullName"])
        if score > best_score:
            best_match, best_score = park, score

    if not best_match:
        return None

    return {
        "code": best_match["parkCode"],
        "name": best_match["fullName"],
        "state": best_match.get("states", ""),
        "desc": best_match.get("description", ""),
        "coords": best_match.get("latLong", ""),
        "url": best_match.get("url", ""),
    }


# Park Info Tool
def get_park_info(park_name: str) -> str:
    """Fetch general information about a U.S. National or State Park."""
    park = resolve_park(park_name)
    if not park:
        return f"â�Œ No results for '{park_name}'."

    coords = park["coords"].replace("lat:", "").replace("long:", "").replace(" ", "")
    maps_link = f"https://www.google.com/maps?q={coords}"
    return (
        f"ğŸ��ï¸� {park['name']} â€” {park['state']}\n"
        f"{park['desc']}\n"
        f"ğŸ“� {park['coords']}\n"
        f"ğŸ—ºï¸� Map: {maps_link}\n"
        f"ğŸŒ� Website: {park['url']}"
    )


#  Campgrounds Tool
def get_campgrounds(park_name: str) -> str:
    """List available campgrounds for a given park."""
    park = resolve_park(park_name)
    if not park:
        return f"â�Œ Couldn't find any park named '{park_name}'."

    url = f"{BASE_URL}/campgrounds"
    params = {"parkCode": park["code"], "api_key": NPS_API_KEY}
    res = requests.get(url, params=params)
    data = res.json().get("data", [])

    if not data:
        return f"No campgrounds found for {park['name']}."

    camps = []
    for cg in data[:5]:
        camps.append(
            f"ğŸ�•ï¸� {cg['name']} â€” {cg.get('description', 'No description available.')}\n"
            f"ğŸ”— {cg.get('url', 'No link provided')}"
        )
    return "\n\n".join(camps)


#  Activities Tool
def get_activities(park_name: str) -> str:
    """List top activities available at the park."""
    park = resolve_park(park_name)
    if not park:
        return f"â�Œ Couldn't find any park named '{park_name}'."

    url = f"{BASE_URL}/parks"
    params = {"parkCode": park["code"], "api_key": NPS_API_KEY}
    res = requests.get(url, params=params)
    data = res.json().get("data", [])
    if not data:
        return f"No activity info found for '{park_name}'."

    acts = data[0].get("activities", [])
    if not acts:
        return f"No activities listed for '{park_name}'."

    return "Top activities:\n" + "\n".join([f"ğŸ�¯ {a['name']}" for a in acts[:10]])


#  Trails Tool
def get_trails(park_name: str) -> str:
    """Fetch hiking and things-to-do info from NPS API."""
    park = resolve_park(park_name)
    if not park:
        return f"â�Œ Couldn't find any park named '{park_name}'."

    url = f"{BASE_URL}/thingstodo"
    params = {"parkCode": park["code"], "api_key": NPS_API_KEY, "limit": 10}
    res = requests.get(url, params=params)
    data = res.json().get("data", [])

    if not data:
        return f"No hiking trails or activities listed for {park['name']}."

    trails = []
    for item in data[:5]:
        title = item.get("title", "Unnamed Trail")
        desc = item.get("shortDescription", "")
        url = item.get("url", "No link")
        trails.append(f"ğŸ¥¾ {title}\n   {desc}\n   ğŸ”— {url}")

    return "Top hiking trails & activities:\n" + "\n\n".join(trails)


#  Weather Tool
def get_weather(park_name: str) -> str:
    """Fetch current weather for a given park name."""
    park = resolve_park(park_name)
    if not park or not park["coords"]:
        return f"â�Œ Can't find coordinates for '{park_name}'."

    try:
        lat = float(park["coords"].split("lat:")[1].split(",")[0].strip())
        lon = float(park["coords"].split("long:")[1].strip())
    except Exception:
        return f"â�Œ Invalid coordinates for '{park_name}'."

    weather_url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": lat, "longitude": lon, "current_weather": True}
    res = requests.get(weather_url, params=params)
    weather = res.json().get("current_weather", {})

    if not weather:
        return f"No weather data for {park['name']}."

    local_time = format_utc_to_local(weather["time"], "US/Pacific")

    return (
        f"ğŸŒ¤ï¸� Weather for {park['name']} ({park['state']})\n"
        f"ğŸŒ¡ï¸� Temp: {weather['temperature']}Â°C\n"
        f"ğŸ’¨ Wind: {weather['windspeed']} km/h\n"
        f"ğŸ•’ Local Time: {local_time}"
    )


print("âœ… All ParkPal tools defined successfully â€” ready for agent integration!")


print("\nğŸ��ï¸� Park Info:")
print(get_park_info("Grand Canyon National Park"))

print("\nğŸ�•ï¸� Campgrounds:")
print(get_campgrounds("Zion National Park"))

print("\nğŸ�¯ Activities:")
print(get_activities("Yosemite National Park"))

print("\nğŸŒ¦ï¸� Weather:")
print(get_weather("Joshua Tree NP"))




# ğŸ��ï¸� Park Information Agent
park_info_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="ParkInfoAgent",
    description="Provides detailed information about national or state parks.",
    instruction="""
    When users ask for general park information, use the get_park_info tool
    to fetch official descriptions, locations, and URLs.
    Always provide a helpful summary with links and maps.
    """,
    tools=[get_park_info],
)

# â›º Camping Agent
camping_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="CampingAgent",
    description="Fetches and explains camping options available in a specific park.",
    instruction="""
    When users ask about campgrounds, camping availability, or where to stay,
    use the get_campgrounds tool and summarize the best 3â€“5 options clearly.
    """,
    tools=[get_campgrounds],
)

# ğŸ�¯ Activities Agent
activities_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="ActivitiesAgent",
    description="Recommends activities available in parks such as hiking, biking, or boating.",
    instruction="""
    When users ask about things to do, family-friendly activities, or adventure options,
    use the get_activities tool and present the most relevant list.
    """,
    tools=[get_activities],
)

# ğŸ¥¾ Trails Agent
trails_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="TrailsAgent",
    description="Provides hiking trail and things-to-do information from NPS.",
    instruction="""
    When users ask for hiking, walking, or sightseeing suggestions,
    use the get_trails tool to fetch trail details and include links when possible.
    """,
    tools=[get_trails],
)

# ğŸŒ¦ï¸� Weather Agent
weather_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="WeatherAgent",
    description="Gives real-time weather information for national parks.",
    instruction="""
    When users mention current weather, temperatures, or planning for rain/snow,
    use the get_weather tool to provide real-time conditions with temperature and wind speed.
    """,
    tools=[get_weather],
)

print("âœ… All specialized sub-agents (Info, Camping, Activities, Trails, Weather) created successfully!")


# 4.2 â€” TripPlanner Sequential Agent
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.google_llm import Gemini

# Step 1: Info Collector Agent
info_agent = LlmAgent(
    name="InfoAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    Gather relevant park details to help plan a trip.
    Use available tools to retrieve park info, weather, trails, and campgrounds.
    Return concise structured notes.
    """,
    tools=[get_park_info, get_campgrounds, get_activities, get_trails, get_weather],
    output_key="trip_data"
)

# Step 2: Summary Agent
summary_agent = LlmAgent(
    name="SummaryAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    Using this collected trip data: {trip_data}
    Summarize it into a friendly travel itinerary with recommendations
    for activities, camping, and preparation tips.
    """,
    output_key="trip_summary"
)

# Combine sequentially
trip_planner_agent = SequentialAgent(
    name="TripPlannerAgent",
    sub_agents=[info_agent, summary_agent]
)

print("âœ… TripPlanner SequentialAgent created successfully!")


from google.adk.tools.agent_tool import AgentTool

# ğŸ§© Convert all agents (including TripPlanner) into callable tools
parkpal_tools = [
    AgentTool(agent=park_info_agent),
    AgentTool(agent=camping_agent),
    AgentTool(agent=activities_agent),
    AgentTool(agent=trails_agent),
    AgentTool(agent=weather_agent),
    AgentTool(agent=trip_planner_agent),  # ğŸ†• Sequential pipeline integrated!
]

# ğŸ�¯ Main Orchestrator Agent
parkpal_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="ParkPal",
    description="An intelligent National Parks concierge that routes queries to expert or sequential agents.",
    instruction="""
    You are ParkPal â€” a friendly and knowledgeable National Parks AI Concierge.

    You have several expert sub-agents and a TripPlanner sequential agent available.

    Your logic:
    - For simple factual questions (like park info, weather, activities, or trails), use the correct sub-agent tool.
    - For multi-day trip planning or itinerary questions (e.g., â€œplan my tripâ€�, â€œmake a 3-day itineraryâ€�), 
      delegate to TripPlannerAgent.
    - If users combine multiple topics (e.g., â€œweather and trailsâ€�), call multiple sub-agents and merge their outputs.
    - Always respond clearly and conversationally, providing relevant links and map info when available.
    """,
    tools=parkpal_tools,
)

print("âœ… ParkPal Orchestrator Agent updated with TripPlanner integration!")


# 4.4 â€” Test ParkPal Integration
from google.adk.runners import InMemoryRunner
import asyncio

runner = InMemoryRunner(agent=parkpal_agent)
print("ğŸš€ Running ParkPal Agent (in-memory mode)...")

async def test_parkpal(query):
    print(f"\nğŸ§­ User: {query}")
    response = await runner.run_debug(query)

    outputs = []
    if isinstance(response, list):
        for r in response:
            if hasattr(r, "output_text") and r.output_text:
                outputs.append(r.output_text)
    elif hasattr(response, "output_text"):
        outputs.append(response.output_text)


# Example queries
await test_parkpal("Plan a 2-day trip to Yosemite with family activities")
await test_parkpal("What are some good hiking trails in Zion National Park?")

# await test_parkpal("Howâ€™s the weather and camping at Yellowstone?")



# 5.1 â€” Initialize Session and Memory Services
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService

session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

print("âœ… Session and Memory services initialized.")


async def auto_save_to_memory(callback_context):
    """Automatically save session to memory after each agent turn."""
    await callback_context._invocation_context.memory_service.add_session_to_memory(
        callback_context._invocation_context.session
    )

print("âœ… Auto memory save callback created.")


parkpal_memory_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="ParkPalMemory",
    description="Memory-enabled version of ParkPal that remembers user preferences and personalizes suggestions.",
    instruction="""
    You are ParkPal â€” a friendly and intelligent national parks assistant with long-term memory.

    ğŸŒ¿ Memory Behavior:
    - Before each conversation, load previous user memories using `preload_memory`.
    - If you see user preferences in memory (e.g., enjoys camping, hiking, wildlife, beaches), use them to personalize suggestions.
    - Mention when you recall something, e.g. "Since you enjoy hiking and camping, you might love Yosemite."

    ğŸ§  Decision Logic:
    - Delegate factual queries to ParkPal (via tool calls).
    - If user asks for park recommendations or planning help:
        â€¢ Match their preferences (from memory) with suitable parks.
        â€¢ Call sub-agents like TrailsAgent and CampingAgent as needed.
        â€¢ Combine the outputs naturally.
    - After answering, automatically save any new information about user interests.

    ğŸ�¯ Output style:
    - Be conversational, clear, and informative.
    - Always include park names, a short description, and a link.
    """,
    tools=[
        preload_memory,
        AgentTool(agent=parkpal_agent)
    ],
    after_agent_callback=auto_save_to_memory,
)

print("âœ… ParkPalMemory agent updated with explicit memory reasoning.")


from google.adk.runners import Runner

memory_runner = Runner(
    agent=parkpal_memory_agent,
    app_name="ParkPalApp",
    session_service=session_service,
    memory_service=memory_service,
)

print("âœ… Runner with session and memory linked successfully.")


# Session 1: Tell ParkPal a preference
await memory_runner.run_debug("Visiting Death Valley National Park is on my bucket list")
print("\nğŸ§­ Session 1 complete â€” preference stored.")

# Session 2: Ask a question in a new session
await memory_runner.run_debug("Can you suggest a park  and plan a 2 day trip?")
print("\nğŸ§­ Session 2 â€” should recall hiking & camping preferences.")




# Session 2: Ask a question in a new session
await memory_runner.run_debug("Can you suggest a park from my bucket list and give the best things to do there?")
print("\nğŸ§­ Session 2 â€” should recall hiking & camping preferences.")


# âš™ï¸� 6.2 â€” Implement Logging Plugin for ParkPal

from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.runners import InMemoryRunner
import asyncio

# Create a runner with LoggingPlugin
runner_with_logs = InMemoryRunner(
    agent=parkpal_memory_agent,   # Using the memory-enabled ParkPal agent
    plugins=[
        LoggingPlugin()           # Standard observability plugin
    ]
)

print("âœ… Runner configured with LoggingPlugin for observability.")


# ğŸ§ª 6.3 â€” Test Observability in Action

queries = [
    "Whatâ€™s the weather and hiking trails like in Mount Rainier?",
]

for q in queries:
    print(f"\nğŸ§­ User: {q}\n" + "-"*70)
    response = await runner_with_logs.run_debug(q)
    print("\nğŸ¤– ParkPal says:\n", getattr(response, "output_text", "(no output text)"))
    print("\nğŸª¶ Logs above show agent reasoning and tool usage.\n")


# ğŸ§  7.3 â€” Demonstrate Context Recall and Compaction
from google.adk.memory import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

# Memory & session setup (simulate user history)
memory_service = InMemoryMemoryService()
session_service = InMemorySessionService()

runner_context = Runner(
    agent=parkpal_memory_agent,
    app_name="ParkPalContextDemo",
    session_service=session_service,
    memory_service=memory_service,
)

print("âœ… Context runner configured with memory and session services.")


# Store preference memory
await runner_context.run_debug("I prefer national parks with lakes and hiking trails.")
await runner_context.run_debug("I want to visit one in the west coast.Tell me as if you are a Ranger")

# New session recall test
print("\nğŸ§­ New Session:")
response = await runner_context.run_debug("Give me the related links.")


# âœ… 8.2 â€” Configure Google Cloud Credentials & Deployment Files

from kaggle_secrets import UserSecretsClient
import os, random

# --- Load GCP Credentials securely from Kaggle ---
user_secrets = UserSecretsClient()
user_credential = user_secrets.get_gcloud_credential()
user_secrets.set_tensorflow_credential(user_credential)


# --- Load Project ID from Kaggle Secrets ---
PROJECT_ID = user_secrets.get_secret("GCP_PROJECT_ID")
if not PROJECT_ID:
    raise ValueError("âš ï¸� Please add 'GCP_PROJECT_ID' to your Kaggle Secrets before running.")

os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
print(f"âœ… GCP Project configured securely: {PROJECT_ID}")

# --- Create deployable agent folder ---
os.makedirs("parkpal_agent", exist_ok=True)
print("ğŸ“� Created deployable folder: parkpal_agent")

# --- Create requirements file ---
with open("parkpal_agent/requirements.txt", "w") as f:
    f.write(
        "google-adk>=1.18.0\n"
        "google-cloud-aiplatform>=1.126.1\n"
        "requests\n"
        "pytz\n"
        "opentelemetry-instrumentation-google-genai\n"
    )
print("âœ… requirements.txt created")

# --- Create environment config file ---
# (We do NOT add API keys here â€” theyâ€™re securely stored in Secret Manager)
with open("parkpal_agent/.env", "w") as f:
    f.write(
        "GOOGLE_CLOUD_LOCATION=us-central1\n"
        "GOOGLE_GENAI_USE_VERTEXAI=1\n"
    )
print("âœ… .env file created (Vertex AI enabled)")

# --- Create agent engine resource configuration ---
with open("parkpal_agent/.agent_engine_config.json", "w") as f:
    f.write(
        '{\n'
        '  "min_instances": 0,\n'
        '  "max_instances": 1,\n'
        '  "resource_limits": {"cpu": "2", "memory": "2Gi"}\n'
        '}'
    )
print("âœ… agent_engine_config.json created (2 CPU / 2Gi RAM)")


code = """  
import os, logging, requests, pytz
from datetime import datetime
from difflib import SequenceMatcher
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.genai import types
import vertexai

# âœ… Initialize Vertex AI for correct region/project binding
vertexai.init(
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
)
print("âœ… Vertex AI initialized for project and location")

# ğŸ§© Logging setup
logging.basicConfig(
    filename="parkpal.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
print("âœ… Logging configured successfully")

# ğŸ”� Secure secret loading â€” works both in Kaggle and Vertex AI
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    NPS_API_KEY = user_secrets.get_secret("NPS_API_KEY")
    print("âœ… Loaded API keys from Kaggle Secrets")
except ModuleNotFoundError:
    NPS_API_KEY = os.getenv("NPS_API_KEY")
    if not  not NPS_API_KEY:
        raise EnvironmentError(
            "â�Œ Missing  NPS_API_KEY. Set it as environment variable in Vertex AI."
        )
    print("âœ… Loaded NPS_API_KEY from environment variables")



print("âœ… Using hardcoded NPS_API_KEY for testing deployment")


# ğŸŒ� Constants
APP_NAME = "ParkPal"
USER_ID = "traveler_001"

# â™»ï¸� Retry configuration for Gemini
retry_config = types.HttpRetryOptions(
    attempts=1,
    exp_base=2,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)
print("âœ… Retry configuration set")

# ğŸ› ï¸� Define tools for ParkPal agent: park info, campgrounds, activities, trails, and weather

# Helper: Safe Markdown hyperlink formatter
def format_url(url):
    if url and url.startswith("http"):
        # Extract last part (park code or path name) for readability
        park_code = url.strip("/").split("/")[-2].upper() if "/" in url else "NPS"
        return f"[{park_code} Official Page]({url})"
    return "No link available"

BASE_URL = "https://developer.nps.gov/api/v1"

def similar(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def format_utc_to_local(utc_str: str, timezone="US/Mountain"):
    utc = datetime.strptime(utc_str, "%Y-%m-%dT%H:%M")
    local_tz = pytz.timezone(timezone)
    return utc.replace(tzinfo=pytz.utc).astimezone(local_tz).strftime("%Y-%m-%d %I:%M %p %Z")

# ğŸ§© Step 1: Resolve park name â†’ metadata
def resolve_park(park_name: str):
    url = f"{BASE_URL}/parks"
    params = {"limit": 500, "api_key": NPS_API_KEY}
    res = requests.get(url, params=params)
    parks = res.json().get("data", [])

    best_match, best_score = None, 0.0
    for park in parks:
        score = similar(park_name, park["fullName"])
        if score > best_score:
            best_match, best_score = park, score

    if not best_match:
        return None

    return {
        "code": best_match["parkCode"],
        "name": best_match["fullName"],
        "state": best_match.get("states", ""),
        "desc": best_match.get("description", ""),
        "coords": best_match.get("latLong", ""),
        "url": best_match.get("url", ""),
    }

# ğŸ��ï¸� Park Info Tool
def get_park_info(park_name: str) -> str:
    park = resolve_park(park_name)
    if not park:
        return f"â�Œ No results for '{park_name}'."

    coords = park["coords"].replace("lat:", "").replace("long:", "").replace(" ", "")
    maps_link = f"https://www.google.com/maps?q={coords}"
    return (
        f"{park['name']} â€” {park['state']}\\n"
        f"{park['desc']}\\n"
        f"{park['coords']}\\n"
        f"Map: {maps_link}\\n"
        f"Website: {park['url']}"
    )

# â›º Campgrounds Tool
def get_campgrounds(park_name: str) -> str:
    park = resolve_park(park_name)
    if not park:
        return f"â�Œ Couldn't find any park named '{park_name}'."

    url = f"{BASE_URL}/campgrounds"
    params = {"parkCode": park["code"], "api_key": NPS_API_KEY}
    res = requests.get(url, params=params)
    data = res.json().get("data", [])

    if not data:
        return f"No campgrounds found for {park['name']}."

    camps = []
    for cg in data[:5]:
        camps.append(
            f"{cg['name']} â€” {cg.get('description', 'No description available.')}\\n"
            f"{cg.get('url', 'No link provided')}"
        )
    return "\\n\\n".join(camps)

# ğŸ�¯ Activities Tool
def get_activities(park_name: str) -> str:
    park = resolve_park(park_name)
    if not park:
        return f"â�Œ Couldn't find any park named '{park_name}'."

    url = f"{BASE_URL}/parks"
    params = {"parkCode": park["code"], "api_key": NPS_API_KEY}
    res = requests.get(url, params=params)
    data = res.json().get("data", [])
    if not data:
        return f"No activity info found for '{park_name}'."

    acts = data[0].get("activities", [])
    if not acts:
        return f"No activities listed for '{park_name}'."

    return "Top activities:\\n" + "\\n".join([f"{a['name']}" for a in acts[:10]])

# ğŸ¥¾ Trails Tool
def get_trails(park_name: str) -> str:
    park = resolve_park(park_name)
    if not park:
        return f"â�Œ Couldn't find any park named '{park_name}'."

    url = f"{BASE_URL}/thingstodo"
    params = {"parkCode": park["code"], "api_key": NPS_API_KEY, "limit": 10}
    res = requests.get(url, params=params)
    data = res.json().get("data", [])

    if not data:
        return f"No hiking trails or activities listed for {park['name']}."

    trails = []
    for item in data[:5]:
        title = item.get("title", "Unnamed Trail")
        desc = item.get("shortDescription", "")
        url = item.get("url", "No link")
        trails.append(f"{title}\\n{desc}\\n{url}")

    return "Top hiking trails & activities:\\n" + "\\n\\n".join(trails)

# ğŸŒ¦ï¸� Weather Tool
def get_weather(park_name: str) -> str:
    park = resolve_park(park_name)
    if not park or not park["coords"]:
        return f"â�Œ Can't find coordinates for '{park_name}'."

    try:
        lat = float(park["coords"].split("lat:")[1].split(",")[0].strip())
        lon = float(park["coords"].split("long:")[1].strip())
    except Exception:
        return f"â�Œ Invalid coordinates for '{park_name}'."

    weather_url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": lat, "longitude": lon, "current_weather": True}
    res = requests.get(weather_url, params=params)
    weather = res.json().get("current_weather", {})

    if not weather:
        return f"No weather data for {park['name']}."

    local_time = format_utc_to_local(weather["time"], "US/Pacific")

    return (
        f"Weather for {park['name']} ({park['state']})\\n"
        f"Temp: {weather['temperature']}Â°C\\n"
        f"Wind: {weather['windspeed']} km/h\\n"
        f"Local Time: {local_time}"
    )

print("âœ… All ParkPal tools defined successfully â€” ready for agent integration!")

# ğŸ��ï¸� Park Information Agent
park_info_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="ParkInfoAgent",
    description="Provides detailed information about national or state parks.",
    instruction="When users ask for general park information, use the get_park_info tool "
                "to fetch official descriptions, locations, and URLs. "
                "Always provide a helpful summary with links and maps.",
    tools=[get_park_info],
)

# â›º Camping Agent
camping_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="CampingAgent",
    description="Fetches and explains camping options available in a specific park.",
    instruction="When users ask about campgrounds, camping availability, or where to stay, "
                "use the get_campgrounds tool and summarize the best 3 to 5 options clearly.",
    tools=[get_campgrounds],
)

# ğŸ�¯ Activities Agent
activities_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="ActivitiesAgent",
    description="Recommends activities available in parks such as hiking, biking, or boating.",
    instruction="When users ask about things to do, family-friendly activities, or adventure options, "
                "use the get_activities tool and present the most relevant list.",
    tools=[get_activities],
)

# ğŸ¥¾ Trails Agent
trails_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="TrailsAgent",
    description="Provides hiking trail and things-to-do information from NPS.",
    instruction="When users ask for hiking, walking, or sightseeing suggestions, "
                "use the get_trails tool to fetch trail details and include links when possible.",
    tools=[get_trails],
)

# ğŸŒ¦ï¸� Weather Agent
weather_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="WeatherAgent",
    description="Gives real-time weather information for national parks.",
    instruction="When users mention current weather, temperatures, or planning for rain/snow, "
                "use the get_weather tool to provide real-time conditions with temperature and wind speed.",
    tools=[get_weather],
)

print("âœ… All specialized sub-agents (Info, Camping, Activities, Trails, Weather) created successfully!")

# 4.2 â€” TripPlanner Sequential Agent
info_agent = LlmAgent(
    name="InfoAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="Gather relevant park details to help plan a trip. "
                "Use available tools to retrieve park info, weather, trails, and campgrounds. "
                "Return concise structured notes.",
    tools=[get_park_info, get_campgrounds, get_activities, get_trails, get_weather],
    output_key="trip_data"
)

summary_agent = LlmAgent(
    name="SummaryAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="Using this collected trip data: {trip_data} "
                "Summarize it into a friendly travel itinerary with recommendations "
                "for activities, camping, and preparation tips.",
    output_key="trip_summary"
)

trip_planner_agent = SequentialAgent(
    name="TripPlannerAgent",
    sub_agents=[info_agent, summary_agent]
)

print("âœ… TripPlanner SequentialAgent created successfully!")

# ğŸ§© Convert all agents (including TripPlanner) into callable tools
parkpal_tools = [
    AgentTool(agent=park_info_agent),
    AgentTool(agent=camping_agent),
    AgentTool(agent=activities_agent),
    AgentTool(agent=trails_agent),
    AgentTool(agent=weather_agent),
    AgentTool(agent=trip_planner_agent),
]

# ğŸ�¯ Main Orchestrator Agent
parkpal_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="ParkPal",
    description="An intelligent National Parks concierge that routes queries to expert or sequential agents.",
    instruction="You are ParkPal, a friendly and knowledgeable National Parks AI Concierge. "
                "You have several expert sub-agents and a TripPlanner sequential agent available. "
                "Your logic: For simple factual questions (like park info, weather, activities, or trails), "
                "use the correct sub-agent tool. For multi-day trip planning or itinerary questions "
                "(e.g., 'plan my trip', 'make a 3-day itinerary'), delegate to TripPlannerAgent. "
                "If users combine multiple topics (e.g., 'weather and trails'), call multiple sub-agents "
                "and merge their outputs. Always respond clearly and conversationally, providing relevant "
                "links and map info when available.",
    tools=parkpal_tools,
)

print("âœ… ParkPal Orchestrator Agent updated with TripPlanner integration")
if __name__ == "__main__":
    print("âœ… Running standalone check â€” ParkPal agent loaded successfully!")

# âœ… Required entrypoint for Vertex AI Agent Engine
try:
    root_agent = parkpal_agent
    print("âœ… root_agent defined successfully â€” ready for Vertex AI deployment.")
except Exception as e:
    print(f"â�Œ Failed to define root_agent: {e}")
    raise

"""

with open("parkpal_agent/agent.py", "w") as f:
    f.write(code)
print("âœ… parkpal_agent/agent.py written successfully")


from vertexai import agent_engines

# âœ… 8.4 â€” Configure VPC connector and deploy to Vertex AI

# --- Configure VPC connector and service account for secure outbound access ---
os.environ["GOOGLE_CLOUD_RUN_VPC_CONNECTOR"] = (
    "projects/" + PROJECT_ID + "/locations/us-central1/connectors/vertex-egress-connector"
)
os.environ["GOOGLE_CLOUD_RUN_VPC_EGRESS"] = "all-traffic"
os.environ["GOOGLE_SERVICE_ACCOUNT"] = (
    "vertex-vpc-accessor@" + PROJECT_ID + ".iam.gserviceaccount.com"
)

print("âœ… VPC connector and service account configured for external API access")

# --- Deploy your agent to Vertex AI Agent Engine ---
!adk deploy agent_engine \
  --project=$PROJECT_ID \
  --region=us-central1 \
  parkpal_agent \
  --display_name="ParkPal AI Agent" \
  --agent_engine_config_file=parkpal_agent/.agent_engine_config.json

