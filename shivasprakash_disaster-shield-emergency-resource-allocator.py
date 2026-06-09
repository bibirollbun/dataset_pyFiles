# pip install google-adk


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")


import json
import requests
import subprocess
import time
import uuid

from google.adk.agents import Agent, LlmAgent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import AgentTool, FunctionTool, google_search, ToolContext, load_memory, preload_memory
from google.adk.agents.remote_a2a_agent import (
    RemoteA2aAgent,
    AGENT_CARD_WELL_KNOWN_PATH,
)
from google.adk.a2a.utils.agent_to_a2a import to_a2a

from google.genai import types

# Hide additional warnings in the notebook
import warnings
warnings.filterwarnings("ignore")

print("âœ… ADK components imported successfully.")


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


def mapping_tool(location: str) -> dict:
    """Looks up and returns geospatial data for a given location.

    Args:
        location: The location to get data for (e.g., "123 Main St, Cityville").

    Returns:
        A dictionary with status and data.
        Success: {"status": "success", "zone": "Zone-A", "population_density": "High", "critical_infrastructure": ["Hospital"]}
        Error: {"status": "error", "error_message": "Location not found"}
    """
    print(f"TOOL: MappingTool called for location: {location}")
    
    # Simulate a database or API call for location data
    location_database = {
        "123 Main St": {
            "zone": "Zone-A",
            "population_density": "High",
            "critical_infrastructure": ["Hospital", "Power Station"]
        },
        "456 Oak Ave": {
            "zone": "Zone-B",
            "population_density": "Medium",
            "critical_infrastructure": ["School"]
        }
    }
    
    # Find a partial match for the location
    found_location = None
    for key in location_database:
        if key in location:
            found_location = key
            break

    if found_location:
        data = location_database[found_location]
        data["status"] = "success"
        return data
    else:
        return {
            "status": "error",
            "error_message": f"Location '{location}' not found in mapping database."
        }

def resource_db_tool(zone: str) -> dict:
    """Queries the emergency resource database for a given zone.

    Args:
        zone: The zone to check for available resources (e.g., "Zone-A").

    Returns:
        A dictionary with status and resource data.
        Success: {"status": "success", "personnel": 5, "ambulances": 2, "food_kits": 100}
        Error: {"status": "error", "error_message": "Zone not found"}
    """
    print(f"TOOL: ResourceDBTool called for zone: {zone}")
    
    inventory = {
        "Zone-A": {"personnel": 5, "ambulances": 2, "food_kits": 100},
        "Zone-B": {"personnel": 8, "ambulances": 4, "food_kits": 250}
    }
    
    resources = inventory.get(zone)
    
    if resources is not None:
        resources["status"] = "success"
        return resources
    else:
        return {
            "status": "error",
            "error_message": f"Zone '{zone}' not found in resource database."
        }


# This agent is powered by an LLM and a prompt.
# It acts as our "Context Compaction" step.
data_ingest_agent = Agent(
    name="DataIngestAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    You are a data intake officer for an emergency response team.
    Your job is to parse unstructured reports from the given input and extract key information.
    Convert the following report into a structured JSON object with the keys:
    "location", "num_people_affected", "injury_types", "urgency" (scale of 1-10).
    """
)


# Define the sub-agents for the parallel team
mapping_agent = LlmAgent(
    name="MappingAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="From the structured report, extract the 'location' and use the mapping_tool to get geospatial data.",
    tools=[mapping_tool],
    output_key="mapping_summary",  # This will be the final output of the entire system.
)

resource_agent = LlmAgent(
    name="ResourceAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="From the structured report, determine the 'zone' and use the resource_db_tool to find available resources.",
    tools=[resource_db_tool],
    output_key="resource_summary",  # This will be the final output of the entire system.
)

# Create the ParallelAgent
impact_assessment_team = ParallelAgent(
    name="ImpactAssessmentTeam",
    sub_agents=[mapping_agent, resource_agent]
)

# This agent takes the output from the parallel team and synthesizes it.
synthesis_agent = LlmAgent(
    name="SynthesisAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    You are an emergency impact analyst. Your goal is to calculate a 'Need Score'.

    Analyze the combined output from the Mapping agent {mapping_summary} and Resource agent {resource_summary}.
    Consider the number of people affected, urgency, population density, and available resources.
    Calculate a 'Need Score' from 1 (low need) to 100 (high need).
    Return a JSON object with "zone", "need_score", and a "justification" for the score.
    """,
    output_key="impact_assessment",  # This will be the final output of the entire system.
)

# Chain the parallel team and the synthesis agent together
impact_assessment_agent = SequentialAgent(
    name="ImpactAssessmentWorkflow",
    sub_agents=[impact_assessment_team, synthesis_agent]
)


allocation_agent = LlmAgent(
    name="AllocationAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are the Resource Allocation Commander. Your job is to create an optimal and equitable
    distribution plan based on the Need Score and historical data.

    Use your long-term memory of past events to improve your decision.
    Prioritize actions that have proven effective.

    Propose a clear, actionable allocation plan in JSON format with keys:
    "zone", "personnel_to_dispatch", "ambulances_to_dispatch", "food_kits_to_dispatch", "priority_actions".

    Impact Assessment: {impact_assessment}
    """,
    tools=[
        load_memory
    ],  # Agent now has access to Memory and can search it whenever it decides to!
)


import os
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

def dispatch_handler(message_body: str):
    """This function is called when the DispatchAgent receives a message."""
    print("\n--- DISPATCH AGENT (A2A) RECEIVED ---")
    print("Dispatching resources based on the following plan:")
    parsed_plan = json.loads(message_body)
    print(json.dumps(parsed_plan, indent=2))
    print("--- DISPATCH COMPLETE ---")
    # In a real system, this would trigger actions in a dispatch system (e.g., send alerts to field teams).
    return "SUCCESS: Plan received and dispatched."

dispatch_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="dispatch_agent",
    description="External Vendors agent responsible for dispatching field resources.",
    instruction="""
    You are the **Dispatch Agent**. Your sole responsibility is to securely translate the final structured 
    Resource Allocation Plan into an action command and deliver it to the external logistics network.
    You **must** call the **dispatch_handler** tool exactly once per input to complete the mission. 
    Do not alter the plan.
    """,
    tools=[dispatch_handler]
)

app = to_a2a(dispatch_agent, port=8001)

# Convert the dispatch agent to an A2A-compatible application
dispatch_agent_a2a_app = to_a2a(
    dispatch_agent, port=8001  # Port where this agent will be served
)

print("âœ… Dispatch Agent is now A2A-compatible!")
print("   Agent will be served at: http://localhost:8001")
print("   Agent card will be at: http://localhost:8001/.well-known/agent-card.json")
print("   Ready to start the server...")


# The DispatchAgent is represented by an EndPoint in the A2A protocol.
# It defines a handler for what to do when it receives a message.

dispatch_agent_code = '''
import os
import json
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

def dispatch_handler(message_body: str):
    """This function is called when the DispatchAgent receives a message.
    It includes exception handling for invalid JSON input.
    """
    try:
        # Attempt to parse the incoming message body
        parsed_plan = json.loads(message_body)
        
        # If parsing succeeds, proceed with dispatch logging
        print("\n--- DISPATCH AGENT (A2A) RECEIVED ---")
        print("Dispatching resources based on the following plan:")
        print(json.dumps(parsed_plan, indent=2))
        print("--- DISPATCH COMPLETE ---")
        
        # In a real system, this would trigger actions in a dispatch system (e.g., send alerts to field teams).
        return "SUCCESS: Plan received and dispatched."
        
    except json.JSONDecodeError as e:
        # Handle the case where message_body is not valid JSON
        error_message = f"ERROR: Failed to decode JSON message body. Check the upstream agent's output format. Error: {e}"
        print(f"\n--- DISPATCH ERROR ---")
        print(error_message)
        print(f"Received raw message: {message_body[:100]}...") # Print a snippet of the bad data
        return error_message
        
    except Exception as e:
        # Catch any other unexpected errors
        unexpected_error_message = f"FATAL ERROR in dispatch_handler: {e}"
        print(f"\n--- DISPATCH UNEXPECTED ERROR ---")
        print(unexpected_error_message)
        return unexpected_error_message

dispatch_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="dispatch_agent",
    description="External Vendors agent responsible for dispatching field resources.",
    instruction="""
    You are the **Dispatch Agent**. Your sole responsibility is to securely translate the final structured 
    Resource Allocation Plan into an action command and deliver it to the external logistics network.
    You **must** call the **dispatch_handler** tool exactly once per input to complete the mission. 
    Do not alter the plan.
    """,
    tools=[dispatch_handler]
)

# Create the A2A app
app2 = to_a2a(dispatch_agent, port=8001)
'''

# Write the product catalog agent to a temporary file
with open("/tmp/dispatch_agent_server2.py", "w") as f:
    f.write(dispatch_agent_code)

print("ğŸ“� Dispatch Agent code saved to /tmp/dispatch_agent_server2.py")

# Start uvicorn server in background
# Note: We redirect output to avoid cluttering the notebook
server_process = subprocess.Popen(
    [
        "uvicorn",
        "dispatch_agent_server2:app2",  # Module:app format
        "--host",
        "localhost",
        "--port",
        "8001",
    ],
    cwd="/tmp",  # Run from /tmp where the file is
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env={**os.environ},  # Pass environment variables (including GOOGLE_API_KEY)
)

print("ğŸš€ Starting Dispatch Agent server...")
print("   Waiting for server to be ready...")

# Wait for server to start (poll until it responds)
max_attempts = 10
for attempt in range(max_attempts):
    try:
        response = requests.get(
            "http://localhost:8001/.well-known/agent-card.json", timeout=5
        )
        if response.status_code == 200:
            print(f"\nâœ… Dispatch Agent server is running!")
            print(f"   Server URL: http://localhost:8001")
            print(f"   Agent card: http://localhost:8001/.well-known/agent-card.json")
            break
    except requests.exceptions.RequestException:
        time.sleep(5)
        print(".", end="", flush=True)
else:
    print("\nâš ï¸�  Server may not be ready yet. Check manually if needed.")

# Store the process so we can stop it later
globals()["dispatch_agent_server_process"] = server_process


# Fetch the agent card from the running server
try:
    response = requests.get(
        "http://localhost:8001/.well-known/agent-card.json", timeout=5
    )

    if response.status_code == 200:
        agent_card = response.json()
        print("ğŸ“‹ Dispatch Agent Card:")
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
    print("   Make sure the Dispatch Agent server is running (previous cell)")


# The main workflow is a sequence of the agents we defined.
disaster_shield_workflow = SequentialAgent(
    name="DisasterShieldWorkflow",
    sub_agents=[
        data_ingest_agent,
        impact_assessment_agent,
        allocation_agent
    ]
)

# Set up the A2A communication channel
# This allows agents to find and communicate with each other securely.
remote_dispatch_agent = RemoteA2aAgent(
    name="dispatch_agent",
    description="Remote dispatch agent from external vendor for dispatching field resources.",
    # Point to the agent card URL - this is where the A2A protocol metadata lives
    agent_card=f"http://localhost:8001{AGENT_CARD_WELL_KNOWN_PATH}",
)

print("âœ… Remote Dispatch Agent proxy created!")
print(f"   Connected to: http://localhost:8001")
print(f"   Agent card: http://localhost:8001{AGENT_CARD_WELL_KNOWN_PATH}")
print("   The Allocation Agent can now use this like a local sub-agent!")

# The final step for the AllocationAgent is to send its plan to the DispatchAgent.
# We add this as a final instruction.
allocation_agent.instruction += """
Finally, send this allocation plan to the 'DispatchAgent' for immediate action.
"""
# We also equip the allocation_agent with the A2A transport as a tool.
allocation_agent.sub_agents = [remote_dispatch_agent]


# A sample unstructured report from an emergency channel.
raw_report = "There's a building collapse at 123 Main St after the earthquake. I see at least 15 people trapped, some look badly hurt. It's really bad, we need help fast!"

print("--- STARTING DISASTER SHIELD WORKFLOW ---")
print(f"Initial Report: '{raw_report}'\n")

# Execute the workflow.
# The ADK handles passing the output of one agent to the next.
runner = InMemoryRunner(agent=disaster_shield_workflow)
final_response = await runner.run_debug(raw_report)

print("\n--- FINAL RESPONSE FROM WORKFLOW ---")
print(final_response)

# The A2A communication happens automatically as part of the last agent's execution.
# The output from the dispatch_handler will be printed to the console.




